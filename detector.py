import base64
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


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
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4, minSize=(30, 30))
            return faces
        except Exception as e:
            print(f"人脸检测错误: {e}")
            return []


class FaceDatabase:
    def __init__(self, known_faces_dir="known_faces"):
        self.known_faces_dir = Path(known_faces_dir)
        self.known_faces_dir.mkdir(exist_ok=True)

        self.known_encodings = []
        self.known_names = []
        self.load_known_faces()

    def load_known_faces(self):
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        if not self.known_faces_dir.exists():
            return

        for img_path in self.known_faces_dir.glob("*.jpg"):
            try:
                name = img_path.stem
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    faces = face_cascade.detectMultiScale(img, 1.1, 4)
                    if len(faces) > 0:
                        x, y, w, h = faces[0]
                        face_roi = img[y : y + h, x : x + w]
                        encoding = self.get_face_encoding(face_roi)
                        if encoding is not None:
                            self.known_encodings.append(encoding)
                            self.known_names.append(name)
            except Exception as e:
                print(f"加载人脸错误 {img_path}: {e}")

        print(f"已加载 {len(self.known_names)} 个人脸数据")

    def get_face_encoding(self, face_roi):
        if face_roi is None or face_roi.size == 0:
            return None
        try:
            face_resized = cv2.resize(face_roi, (100, 100))
            return face_resized.flatten()
        except Exception:
            return None

    def add_known_face(self, name, image_data):
        try:
            img_bytes = base64.b64decode(image_data.split(",")[1])
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                return False, "无法解码图片"

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_roi = gray[y : y + h, x : x + w]
            else:
                h, w = gray.shape
                face_roi = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]

            save_path = self.known_faces_dir / f"{name}.jpg"
            cv2.imwrite(str(save_path), face_roi)

            encoding = self.get_face_encoding(face_roi)
            if encoding is not None:
                self.known_encodings.append(encoding)
                self.known_names.append(name)
                return True, "人脸添加成功"
            return False, "无法提取人脸特征"
        except Exception as e:
            return False, f"添加失败: {e}"

    def add_face_from_camera(self, name, frame, bbox):
        try:
            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            face_roi = frame[y1:y2, x1:x2]

            if face_roi is None or face_roi.size == 0:
                return False, "区域无效"

            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)

            face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)

            if len(faces) > 0:
                fx, fy, fw, fh = faces[0]
                face_roi = gray[fy : fy + fh, fx : fx + fw]

            save_path = self.known_faces_dir / f"{name}.jpg"
            cv2.imwrite(str(save_path), face_roi)

            encoding = self.get_face_encoding(face_roi)
            if encoding is not None:
                self.known_encodings.append(encoding)
                self.known_names.append(name)
                return True, "人脸添加成功"

            return False, "无法提取人脸特征"
        except Exception as e:
            return False, f"添加失败: {e}"

    def recognize_face(self, face_roi):
        if face_roi is None or face_roi.size == 0 or len(self.known_encodings) == 0:
            return "unknown"

        try:
            query_encoding = self.get_face_encoding(face_roi)
            if query_encoding is None:
                return "unknown"

            min_dist = float("inf")
            recognized_name = "unknown"

            for i, encoding in enumerate(self.known_encodings):
                dist = np.linalg.norm(query_encoding - encoding)
                if dist < min_dist:
                    min_dist = dist
                    recognized_name = self.known_names[i]

            threshold = 8000
            if min_dist > threshold:
                return "unknown"

            return recognized_name
        except Exception as e:
            print(f"人脸识别错误: {e}")
            return "unknown"


class ZoneManager:
    def __init__(self):
        self.zones = []
        self.last_zone_alert_time = {}
        self.zone_cooldown_seconds = 60

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
        self.cooldown_seconds = 60
        self.sensitive_zone_cooldown_seconds = 30
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

        # 检查最近5条警报是否重复
        for existing in self.alerts[:5]:
            if existing["message"] == message:
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
