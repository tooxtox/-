import base64
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# 使用 OpenCV DNN 进行人脸检测（准确率更高）
USE_DNN_FACE_DETECTOR = True
DNN_PROTO_PATH = None
DNN_MODEL_PATH = None
face_net = None


def init_dnn_face_detector():
    global face_net, DNN_PROTO_PATH, DNN_MODEL_PATH

    if not USE_DNN_FACE_DETECTOR:
        return False

    try:
        plugin_dir = Path(__file__).parent
        DNN_PROTO_PATH = plugin_dir / "deploy.prototxt.txt"
        DNN_MODEL_PATH = plugin_dir / "res10_300x300_ssd_iter_140000.caffemodel"

        if DNN_PROTO_PATH.exists() and DNN_MODEL_PATH.exists():
            face_net = cv2.dnn.readNetFromCaffe(
                str(DNN_PROTO_PATH), str(DNN_MODEL_PATH)
            )
            print("[人脸识别] DNN 人脸检测器加载成功")
            return True
        else:
            print("[人脸识别] DNN 模型文件不存在，将使用 Haar Cascade")
            print("[人脸识别] 提示：可以下载以下文件提高检测准确率：")
            print(
                "  - deploy.prototxt.txt: https://github.com/opencv/opencv/blob/4.5.2/samples/dnn/face_detector/deploy.prototxt"
            )
            print(
                "  - res10_300x300_ssd_iter_140000.caffemodel: https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
            )
            return False
    except Exception as e:
        print(f"[人脸识别] DNN 人脸检测器初始化失败: {e}")
        return False


def detect_faces_dnn(frame, conf_threshold=0.5):
    if face_net is None:
        return []

    try:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
        )
        face_net.setInput(blob)
        detections = face_net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                faces.append((x1, y1, x2 - x1, y2 - y1))

        return faces
    except Exception as e:
        print(f"[人脸识别] DNN 检测错误: {e}")
        return []


# 初始化 DNN 人脸检测器
init_dnn_face_detector()


def get_face_encoding_legacy(face_roi):
    if face_roi is None or face_roi.size == 0:
        return None
    try:
        face_resized = cv2.resize(face_roi, (100, 100))
        return face_resized.flatten().astype(np.float64)
    except Exception:
        return None


class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        self.model_path = model_path
        self.model = None
        self.device = "cpu"
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                import torch
                from ultralytics import YOLO

                # 强制使用CPU，避免CUDA兼容性问题
                self.device = "cpu"
                print("[YOLO] 使用 CPU (避免CUDA兼容性问题)")

                # 全局修复 torch.load 函数
                original_load = torch.load

                def patched_load(*args, **kwargs):
                    kwargs["weights_only"] = False
                    return original_load(*args, **kwargs)

                torch.load = patched_load
                try:
                    self.model = YOLO(self.model_path)
                    self.model.to(self.device)
                    print(f"YOLO模型加载成功: {self.model_path} (设备: {self.device})")
                finally:
                    torch.load = original_load
            except Exception as e:
                print(f"YOLO模型加载失败: {e}")
                self.model = None
                self.device = "cpu"
        else:
            print(f"模型文件不存在: {self.model_path}")
            self.model = None
            self.device = "cpu"

    def detect_persons(self, frame):
        if self.model is None or frame is None or frame.size == 0:
            return []

        try:
            import torch

            # 在推理时也应用修复
            original_load = torch.load

            def patched_load(*args, **kwargs):
                kwargs["weights_only"] = False
                return original_load(*args, **kwargs)

            torch.load = patched_load
            try:
                results = self.model(frame, verbose=False, conf=0.5, device=self.device)
                persons = []

                if results and len(results) > 0:
                    result = results[0]
                    if result.boxes is not None:
                        boxes = result.boxes
                        for i in range(len(boxes)):
                            cls = int(boxes.cls[i].item())
                            if cls == 0:
                                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                                conf = float(boxes.conf[i].item())
                                persons.append(
                                    {
                                        "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                        "conf": conf,
                                    }
                                )
                return persons
            finally:
                torch.load = original_load
        except Exception as e:
            print(f"YOLO检测错误: {e}")
            return []

    def detect_faces(self, frame):
        if frame is None or frame.size == 0:
            return []

        try:
            if face_net is not None:
                faces = detect_faces_dnn(frame)
                if len(faces) > 0:
                    return faces
                else:
                    print("[人脸识别] DNN 未检测到人脸，回退到 Haar Cascade")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            return faces
        except Exception as e:
            print(f"人脸检测错误: {e}")
            return []

    def detect_faces_with_encodings(self, frame):
        if frame is None or frame.size == 0:
            return []

        try:
            faces = self.detect_faces(frame)
            results = []
            frame_h, frame_w = frame.shape[:2]
            for face in faces:
                x, y, w, h = face
                x = max(0, x)
                y = max(0, y)
                w = min(w, frame_w - x)
                h = min(h, frame_h - y)
                if w > 0 and h > 0:
                    face_roi = frame[y : y + h, x : x + w]
                    if len(frame.shape) == 3:
                        gray_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                    else:
                        gray_roi = face_roi
                    encoding = get_face_encoding_legacy(gray_roi)
                    results.append((face, encoding))
            return results
        except Exception as e:
            print(f"人脸检测(带编码)错误: {e}")
            faces = self.detect_faces(frame)
            return [(face, None) for face in faces]


class FaceDatabase:
    def __init__(self, known_faces_dir="known_faces"):
        self.known_faces_dir = Path(known_faces_dir)
        self.known_faces_dir.mkdir(exist_ok=True)

        self.known_encodings = []
        self.known_names = []
        self.recognition_threshold = 0.6
        self.load_known_faces()

    def load_known_faces(self):
        if not self.known_faces_dir.exists():
            return

        for img_path in self.known_faces_dir.glob("*.jpg"):
            try:
                name = img_path.stem
                img = cv2.imread(str(img_path))
                if img is None:
                    continue

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if face_net is not None:
                    faces = detect_faces_dnn(img)
                    if len(faces) > 0:
                        x, y, w, h = faces[0]
                        face_roi = gray[y : y + h, x : x + w]
                    else:
                        face_cascade = cv2.CascadeClassifier(
                            cv2.data.haarcascades
                            + "haarcascade_frontalface_default.xml"
                        )
                        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
                        if len(faces) > 0:
                            x, y, w, h = faces[0]
                            face_roi = gray[y : y + h, x : x + w]
                        else:
                            h, w = gray.shape
                            face_roi = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
                else:
                    face_cascade = cv2.CascadeClassifier(
                        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                    )
                    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
                    if len(faces) > 0:
                        x, y, w, h = faces[0]
                        face_roi = gray[y : y + h, x : x + w]
                    else:
                        h, w = gray.shape
                        face_roi = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]

                encoding = get_face_encoding_legacy(face_roi)
                if encoding is not None:
                    self.known_encodings.append(encoding)
                    self.known_names.append(name)
                    print(f"[人脸库] 加载已知人脸: {name}")
            except Exception as e:
                print(f"加载人脸错误 {img_path}: {e}")

        print(f"已加载 {len(self.known_names)} 个人脸数据")

    def get_face_encoding(self, frame, face_location=None):
        if frame is None or frame.size == 0:
            return None

        try:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame

            if face_location is not None:
                x, y, w, h = face_location
                face_roi = gray[y : y + h, x : x + w]
            else:
                if face_net is not None:
                    faces = detect_faces_dnn(frame)
                    if len(faces) > 0:
                        x, y, w, h = faces[0]
                        face_roi = gray[y : y + h, x : x + w]
                    else:
                        face_cascade = cv2.CascadeClassifier(
                            cv2.data.haarcascades
                            + "haarcascade_frontalface_default.xml"
                        )
                        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
                        if len(faces) > 0:
                            x, y, w, h = faces[0]
                            face_roi = gray[y : y + h, x : x + w]
                        else:
                            h, w = gray.shape
                            face_roi = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
                else:
                    face_cascade = cv2.CascadeClassifier(
                        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                    )
                    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
                    if len(faces) > 0:
                        x, y, w, h = faces[0]
                        face_roi = gray[y : y + h, x : x + w]
                    else:
                        h, w = gray.shape
                        face_roi = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]

            return get_face_encoding_legacy(face_roi)
        except Exception as e:
            print(f"获取人脸编码错误: {e}")
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            return get_face_encoding_legacy(gray)

    def add_known_face(self, name, image_data):
        try:
            if "," in image_data:
                img_bytes = base64.b64decode(image_data.split(",")[1])
            else:
                img_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

            if img is None:
                return False, "无法解码图片"

            if img is None or img.size == 0:
                return False, "图片为空"

            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            elif len(img.shape) == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

            img = img.astype(np.uint8)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if face_net is not None:
                faces = detect_faces_dnn(img)
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    face_roi = gray[y : y + h, x : x + w]
                else:
                    face_cascade = cv2.CascadeClassifier(
                        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                    )
                    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
                    if len(faces) > 0:
                        x, y, w, h = faces[0]
                        face_roi = gray[y : y + h, x : x + w]
                    else:
                        h, w = gray.shape
                        face_roi = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
            else:
                face_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                )
                faces = face_cascade.detectMultiScale(gray, 1.1, 5)
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    face_roi = gray[y : y + h, x : x + w]
                else:
                    h, w = gray.shape
                    face_roi = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]

            save_path = self.known_faces_dir / f"{name}.jpg"
            cv2.imwrite(str(save_path), img)

            encoding = get_face_encoding_legacy(face_roi)
            if encoding is not None:
                self.known_encodings.append(encoding)
                self.known_names.append(name)
                return True, "人脸添加成功"
            return False, "无法提取人脸特征"
        except Exception as e:
            return False, f"添加失败: {str(e)}"

    def add_face_from_camera(self, name, frame, bbox):
        try:
            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            face_roi = frame[y1:y2, x1:x2]

            if face_roi is None or face_roi.size == 0:
                return False, "区域无效"

            processed_face = face_roi.copy()

            if processed_face.dtype != np.uint8:
                processed_face = processed_face.astype(np.uint8)

            if len(processed_face.shape) == 2:
                processed_face = cv2.cvtColor(processed_face, cv2.COLOR_GRAY2BGR)
            elif len(processed_face.shape) == 3 and processed_face.shape[2] == 4:
                processed_face = cv2.cvtColor(processed_face, cv2.COLOR_BGRA2BGR)

            save_path = self.known_faces_dir / f"{name}.jpg"
            cv2.imwrite(str(save_path), processed_face)

            if len(processed_face.shape) == 3:
                gray = cv2.cvtColor(processed_face, cv2.COLOR_BGR2GRAY)
            else:
                gray = processed_face

            encoding = get_face_encoding_legacy(gray)
            if encoding is not None:
                self.known_encodings.append(encoding)
                self.known_names.append(name)
                return True, "人脸添加成功"

            return False, "无法提取人脸特征"
        except Exception as e:
            return False, f"添加失败: {e}"

    def recognize_face(self, face_roi, face_encoding=None):
        if len(self.known_encodings) == 0:
            return "unknown", 0.0

        try:
            if face_encoding is not None:
                query_encoding = face_encoding
            else:
                if face_roi is None or face_roi.size == 0:
                    return "unknown", 0.0
                query_encoding = get_face_encoding_legacy(face_roi)

            if query_encoding is None:
                return "unknown", 0.0

            min_dist = float("inf")
            recognized_name = "unknown"

            for i, encoding in enumerate(self.known_encodings):
                dist = np.linalg.norm(query_encoding - encoding)
                if dist < min_dist:
                    min_dist = dist
                    recognized_name = self.known_names[i]

            threshold = 8000
            if min_dist <= threshold:
                confidence = max(0, 1.0 - min_dist / 10000)
                return recognized_name, confidence
            return "unknown", max(0, 1.0 - min_dist / 10000)
        except Exception as e:
            print(f"人脸识别错误: {e}")
            return "unknown", 0.0

    def recognize_face_in_frame(self, frame, face_location):
        x, y, w, h = face_location
        face_roi = frame[y : y + h, x : x + w]
        if len(frame.shape) == 3:
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        return self.recognize_face(face_roi)


class ZoneManager:
    def __init__(self):
        self.zones = []
        self.last_zone_alert_time = {}
        self.zone_cooldown_seconds = 10

    def add_zone(self, name, points, zone_type="polygon"):
        self.zones.append({"name": name, "points": points, "type": zone_type})

    def remove_zone(self, index):
        if 0 <= index < len(self.zones):
            self.zones.pop(index)

    def get_zones(self):
        return self.zones

    def point_in_zone(self, point, zone):
        x, y = point
        pts = np.array([[int(p[0]), int(p[1])] for p in zone["points"]], np.int32)
        return cv2.pointPolygonTest(pts, (int(x), int(y)), False) >= 0

    def check_zones(self, person_locations):
        alerts = []
        now = datetime.now()
        for name, location in person_locations.items():
            for idx, zone in enumerate(self.zones):
                if not self.point_in_zone(location, zone):
                    continue
                zone_key = f"{idx}:{zone['name']}:{name}"
                if zone_key in self.last_zone_alert_time:
                    time_since_last = (
                        now - self.last_zone_alert_time[zone_key]
                    ).total_seconds()
                    if time_since_last < self.zone_cooldown_seconds:
                        continue
                alerts.append(
                    {
                        "type": "zone_alert",
                        "person": name,
                        "zone": zone["name"],
                        "timestamp": now.isoformat(),
                        "zone_index": idx,
                    }
                )
                self.last_zone_alert_time[zone_key] = now
        return alerts


class AlertManager:
    def __init__(self):
        self.alerts = []
        self.max_alerts = 100
        self.last_alert_time = {}
        self.last_sensitive_zone_alert_time = None
        self.last_llm_image_time = None
        self.cooldown_seconds = 10
        self.sensitive_zone_cooldown_seconds = 10
        self.llm_image_cooldown_seconds = 30

    def can_call_llm_image(self):
        """检查是否可以调用 LLM 图片描述"""
        now = datetime.now()
        if self.last_llm_image_time:
            time_since_last = (now - self.last_llm_image_time).total_seconds()
            if time_since_last < self.llm_image_cooldown_seconds:
                print("[警报] LLM 图片描述冷却中，跳过调用")
                return False
        self.last_llm_image_time = now
        return True

    def add_alert(self, alert_type, message, person="unknown", is_sensitive_zone=False):
        now = datetime.now()
        alert_key = f"{alert_type}:{message}:{person}"

        # 敏感区域共用冷却时间
        if is_sensitive_zone:
            if self.last_sensitive_zone_alert_time:
                time_since_last = (
                    now - self.last_sensitive_zone_alert_time
                ).total_seconds()
                if time_since_last < self.sensitive_zone_cooldown_seconds:
                    print(f"[警报] 敏感区域冷却中，跳过所有警报: {message}")
                    return None
            # 更新敏感区域最后警报时间
            self.last_sensitive_zone_alert_time = now
        else:
            # 普通警报使用单独冷却时间
            if alert_key in self.last_alert_time:
                time_since_last = (
                    now - self.last_alert_time[alert_key]
                ).total_seconds()
                if time_since_last < self.cooldown_seconds:
                    print(f"[警报] 冷却中，跳过重复警报: {message}")
                    return None

        alert = {
            "type": alert_type,
            "message": message,
            "person": person,
            "is_sensitive_zone": is_sensitive_zone,
            "timestamp": now.isoformat(),
        }
        self.alerts.insert(0, alert)
        self.last_alert_time[alert_key] = now
        if len(self.alerts) > self.max_alerts:
            self.alerts.pop()
        return alert

    def get_alerts(self, limit=50):
        return self.alerts[:limit]

    def clear_alerts(self):
        self.alerts = []
