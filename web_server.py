import asyncio
import base64
import threading
import time

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request

surveillance_client = None
ASTRBOT_AVAILABLE = False

try:
    import os
    import sys

    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from detector import AlertManager, FaceDatabase, ObjectDetector, ZoneManager

    try:
        from .client import surveillance_client as _surveillance_client

        surveillance_client = _surveillance_client
    except ImportError:
        from client import surveillance_client as _surveillance_client

        surveillance_client = _surveillance_client

    try:
        from .surveillance_platform_adapter import ASTRBOT_AVAILABLE as _available

        ASTRBOT_AVAILABLE = _available
    except ImportError:
        try:
            from surveillance_platform_adapter import ASTRBOT_AVAILABLE as _available

            ASTRBOT_AVAILABLE = _available
        except ImportError:
            ASTRBOT_AVAILABLE = False
except ImportError as e:
    if "ultralytics" in str(e):
        raise ImportError(
            "缺少依赖 'ultralytics'，请运行以下命令安装依赖:\n"
            "pip install ultralytics==8.0.196\n"
            "或安装所有依赖: pip install -r requirements.txt"
        ) from e
    raise

print(f"[web_server] ASTRBOT_AVAILABLE = {ASTRBOT_AVAILABLE}")
ASTRBOT_ENABLED = ASTRBOT_AVAILABLE
print(f"[web_server] ASTRBOT_ENABLED = {ASTRBOT_ENABLED}")
llm_provider = None
target_qqs = []
platform_manager = None
_context = None
zone_preview_points = []
zone_alert_sent = False
last_zone_alert_time = None
stranger_alert_sent = False
last_stranger_alert_time = None


def init_llm(context):
    global \
        llm_provider, \
        platform_manager, \
        _context, \
        surveillance_client, \
        ASTRBOT_ENABLED
    ASTRBOT_ENABLED = True
    print("[Init] ASTRBOT_ENABLED 设置为 True")
    _context = context
    platform_manager = context.platform_manager

    if surveillance_client:
        surveillance_client.initialize(context)
        print("[Init] surveillance_client 初始化完成")
    else:
        print("[Init] surveillance_client 为 None，尝试重新导入")
        try:
            from .client import surveillance_client as _sc

            surveillance_client = _sc
            surveillance_client.initialize(context)
            print("[Init] surveillance_client 重新导入并初始化完成")
        except Exception as e:
            print(f"[Init] surveillance_client 重新导入失败: {e}")

    try:
        from astrbot.core.star.star_tools import StarTools

        StarTools.initialize(context)
        print("[Init] StarTools 初始化完成")
    except Exception as e:
        print(f"[Init] StarTools 初始化失败: {e}")

    def get_llm_provider():
        if not _context:
            print("[LLM] context 为空")
            return None

        async def get_provider():
            chat_provider_id = await _context.get_current_chat_provider_id("")
            if chat_provider_id:
                provider = await _context.provider_manager.get_provider_by_id(
                    chat_provider_id
                )
                return provider
            return None

        try:
            import asyncio

            loop = asyncio.get_event_loop()
            if loop.is_running():
                future = asyncio.create_task(get_provider())
                return future
            else:
                return loop.run_until_complete(get_provider())
        except Exception as e:
            print(f"[LLM] 获取 provider 失败: {e}")
            return None

    llm_provider = get_llm_provider
    print("[Init] llm_provider 设置完成")


async def send_alert_via_llm(
    alert_type: str, message: str, person: str, image_path: str = None
):
    global llm_provider, platform_manager, target_qqs, _context

    print(f"[调试] send_alert_via_llm 被调用: {alert_type}, {message}, {person}")
    print(f"[调试] ASTRBOT_ENABLED={ASTRBOT_ENABLED}, _context={_context}, platform_manager={platform_manager}")

    if not ASTRBOT_ENABLED:
        print("[警报] AstrBot 模式未启用")
        return

    if not _context:
        print("[警报] context 未初始化")
        return

    if not platform_manager:
        print("[警报] platform_manager 未初始化")
        return

    try:
        from astrbot.api.event import MessageChain
        from astrbot.api.message_components import Image, Plain
        from astrbot.core.star.star_tools import StarTools

        chat_provider_id = await _context.get_current_chat_provider_id("")
        print(f"[警报] 当前 provider ID: {chat_provider_id}")

        alert_prompt = f"""你是一个智能监控系统的警报通知助手。请用简洁友好的语气通知用户以下警报信息：

警报类型：{alert_type}
详细信息：{message}
人员：{person}

请生成一条100字左右的通知消息。"""

        print("[警报] 正在调用 LLM 生成通知消息...")

        response = await _context.llm_generate(
            chat_provider_id=chat_provider_id, prompt=alert_prompt
        )

        llm_message = ""
        if response.result_chain and response.result_chain.chain:
            for comp in response.result_chain.chain:
                if hasattr(comp, "text"):
                    llm_message = comp.text
                    break

        if not llm_message:
            llm_message = message

        print(f"[警报] LLM 返回: {llm_message}")

        try:
            from astrbot.api.message_components import Image, Plain
            from astrbot.core.message.message_event_result import MessageChain
            from astrbot.core.star.star_tools import StarTools

            for target_qq in target_qqs:
                msg_chain = MessageChain([Plain(text=llm_message)])

                print(f"[警报] 正在通过 StarTools 发送消息到 {target_qq}")
                await StarTools.send_message_by_id(
                    type="PrivateMessage",
                    id=target_qq,
                    message_chain=msg_chain
                )
                print(f"[警报] 发送成功到 {target_qq}!")
        except Exception as e:
            print(f"[警报] 发送失败: {e}")
            import traceback

            traceback.print_exc()
    except Exception as e:
        print(f"[警报] 错误: {e}")
        if ASTRBOT_AVAILABLE:
            try:
                from astrbot import logger

                logger.error(f"发送 LLM 警报失败: {e}")
            except Exception:
                pass


async def send_alert_via_llm_with_image(
    alert_type: str, message: str, person: str, image_path: str = None
):
    global llm_provider, platform_manager, target_qqs, _context

    print(f"[调试] send_alert_via_llm_with_image 被调用: {alert_type}, {message}, {person}")
    print(f"[调试] ASTRBOT_ENABLED={ASTRBOT_ENABLED}, _context={_context}, platform_manager={platform_manager}")

    if not ASTRBOT_ENABLED:
        print("[警报] AstrBot 模式未启用")
        return

    if not _context:
        print("[警报] context 未初始化")
        return

    if not platform_manager:
        print("[警报] platform_manager 未初始化")
        return

    try:
        from astrbot.api.event import MessageChain
        from astrbot.api.message_components import Image, Plain

        chat_provider_id = await _context.get_current_chat_provider_id("")
        print(f"[警报] 当前 provider ID: {chat_provider_id}")

        image_desc = ""

        # 优先使用传入的图片路径，否则获取最新截图
        if not image_path:
            image_path = get_latest_snapshot()
            print(f"[警报] 使用最新截图: {image_path}")

        if image_path:
            import os

            abs_path = os.path.abspath(image_path)

            provider_supports_vision = True
            vision_models = [
                "glm-4v",
                "glm-4-plus",
                "gpt-4v",
                "gpt-4o",
                "gpt-4",
                "doubao-vision",
                "qwen-vl",
                "doubao",
            ]
            provider_lower = chat_provider_id.lower()
            if not any(vm.lower() in provider_lower for vm in vision_models):
                print("[警报] 当前 provider 不支持图片输入，跳过图片描述")
                provider_supports_vision = False

            if provider_supports_vision:
                # 检查 LLM 图片描述冷却时间
                if not alert_manager.can_call_llm_image():
                    print("[警报] LLM 图片描述冷却中，使用默认描述")
                    image_desc = "监控画面中检测到人员"
                else:
                    prompt = """请详细描述这张监控摄像头图片中的内容，包括环境、人物特征、动作等，100字左右。"""
                    print(f"[警报] 正在调用 LLM 描述图片... 路径: {abs_path}")
                    response = await _context.llm_generate(
                        chat_provider_id=chat_provider_id,
                        prompt=prompt,
                        image_urls=[abs_path],
                    )

                    if response.result_chain and response.result_chain.chain:
                        for comp in response.result_chain.chain:
                            if hasattr(comp, "text"):
                                image_desc = comp.text
                                break

                    print(f"[警报] 图片描述: {image_desc}")

        alert_prompt = f"""你是一个智能监控系统的警报通知助手。请用简洁友好的语气通知用户以下警报信息：

警报类型：{alert_type}
详细信息：{message}
人员：{person}
{f"图片描述：{image_desc}" if image_desc else ""}

请生成一条100字左右的通知消息，包含环境、人物和动作等详细信息。"""

        print("[警报] 正在调用 LLM 生成通知消息...")

        response = await _context.llm_generate(
            chat_provider_id=chat_provider_id, prompt=alert_prompt
        )

        llm_message = ""
        if response.result_chain and response.result_chain.chain:
            for comp in response.result_chain.chain:
                if hasattr(comp, "text"):
                    llm_message = comp.text
                    break

        if not llm_message:
            llm_message = "警报通知"

        print(f"[警报] LLM 返回: {llm_message}")

        try:
            from astrbot.api.message_components import Image, Plain
            from astrbot.core.message.message_event_result import MessageChain
            from astrbot.core.star.star_tools import StarTools

            for target_qq in target_qqs:
                if image_path:
                    import os

                    abs_path = os.path.abspath(image_path)
                    msg_chain = MessageChain(
                        [Plain(text=llm_message), Image(file=abs_path)]
                    )
                else:
                    msg_chain = MessageChain([Plain(text=llm_message)])

                print(f"[警报] 正在通过 StarTools 发送消息到 {target_qq}")
                await StarTools.send_message_by_id(
                    type="PrivateMessage",
                    id=target_qq,
                    message_chain=msg_chain
                )
                print(f"[警报] 发送成功到 {target_qq}!")
        except Exception as e:
            print(f"[警报] 发送失败: {e}")
            import traceback

            traceback.print_exc()
    except Exception as e:
        print(f"[警报] 错误: {e}")
        if ASTRBOT_AVAILABLE:
            try:
                from astrbot import logger

                logger.error(f"发送 LLM 警报失败: {e}")
            except Exception:
                pass


app = Flask(__name__)

detector = None
face_db = None
zone_manager = None
alert_manager = None
camera = None
is_camera_active = False
current_frame = None
frame_lock = threading.Lock()
snapshot_frame = None


def save_snapshot_image(frame):
    import os
    from datetime import datetime

    snapshots_dir = "snapshots"
    if not os.path.exists(snapshots_dir):
        os.makedirs(snapshots_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{snapshots_dir}/alert_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    return filename


def get_latest_snapshot():
    import os

    snapshots_dir = "snapshots"
    if not os.path.exists(snapshots_dir):
        return None

    files = [f for f in os.listdir(snapshots_dir) if f.endswith(".jpg")]
    if not files:
        return None

    files.sort(
        key=lambda x: os.path.getmtime(os.path.join(snapshots_dir, x)), reverse=True
    )
    latest_file = os.path.join(snapshots_dir, files[0])
    return latest_file


def init_detector(model_path="yolov8n.pt"):
    global detector, face_db, zone_manager, alert_manager
    detector = ObjectDetector(model_path)
    face_db = FaceDatabase("known_faces")
    zone_manager = ZoneManager()
    alert_manager = AlertManager()


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>智能监控系统</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Arial, sans-serif; background: #1a1a2e; color: #eee; min-height: 100vh; }
        .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
        h1 { text-align: center; color: #00d4ff; margin-bottom: 20px; font-size: 2rem; }
        .main-content { display: grid; grid-template-columns: 1fr 350px; gap: 20px; }
        .video-section { background: #16213e; border-radius: 12px; padding: 15px; }
        .video-container { position: relative; background: #0f0f23; border-radius: 8px; overflow: hidden; }
        #videoFrame { width: 100%; display: block; cursor: crosshair; }
        .controls { display: flex; gap: 10px; margin-top: 15px; flex-wrap: wrap; }
        button { padding: 10px 20px; border: none; border-radius: 6px; cursor: pointer; font-size: 14px; transition: all 0.3s; }
        .btn-primary { background: #00d4ff; color: #1a1a2e; }
        .btn-danger { background: #ff4757; color: white; }
        .btn-success { background: #2ed573; color: #1a1a2e; }
        .btn-warning { background: #ffa502; color: #1a1a2e; }
        .sidebar { display: flex; flex-direction: column; gap: 15px; }
        .panel { background: #16213e; border-radius: 12px; padding: 15px; }
        .panel h3 { color: #00d4ff; margin-bottom: 12px; font-size: 1rem; border-bottom: 1px solid #2a3f5f; padding-bottom: 8px; }
        .zone-list { max-height: 200px; overflow-y: auto; }
        .zone-item { background: #1a2a4a; padding: 10px; margin-bottom: 8px; border-radius: 6px; display: flex; justify-content: space-between; align-items: center; }
        .zone-item span { font-size: 14px; }
        .zone-delete { background: #ff4757; color: white; border: none; padding: 4px 10px; border-radius: 4px; cursor: pointer; font-size: 12px; }
        .alert-list { max-height: 300px; overflow-y: auto; }
        .alert-item { background: #1a2a4a; padding: 10px; margin-bottom: 8px; border-radius: 6px; border-left: 3px solid #00d4ff; }
        .alert-item.stranger { border-left-color: #ff4757; }
        .alert-item.zone { border-left-color: #ffa502; }
        .alert-time { font-size: 11px; color: #888; }
        .alert-msg { font-size: 13px; margin-top: 4px; }
        .upload-section { display: flex; flex-direction: column; gap: 10px; }
        .upload-section input[type="text"] { padding: 10px; border: 1px solid #2a3f5f; border-radius: 6px; background: #1a2a4a; color: #eee; font-size: 14px; }
        .upload-row { display: flex; gap: 10px; }
        input[type="file"] { display: none; }
        .file-label { background: #2a3f5f; color: #eee; padding: 10px 15px; border-radius: 6px; cursor: pointer; font-size: 14px; text-align: center; flex: 1; }
        .stats { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; }
        .stat-box { background: #1a2a4a; padding: 15px; border-radius: 8px; text-align: center; }
        .stat-num { font-size: 24px; font-weight: bold; color: #00d4ff; }
        .stat-label { font-size: 12px; color: #888; margin-top: 5px; }
        .instructions { font-size: 12px; color: #888; margin-top: 10px; padding: 10px; background: #1a2a4a; border-radius: 6px; }
        .modal { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.8); z-index: 1000; }
        .modal-content { position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); background: #16213e; padding: 20px; border-radius: 12px; text-align: center; }
        .modal-content img { max-width: 300px; border-radius: 8px; margin-bottom: 15px; }
        .modal-content input { padding: 10px; border: 1px solid #2a3f5f; border-radius: 6px; background: #1a2a4a; color: #eee; font-size: 14px; margin-bottom: 10px; width: 100%; }
        @media (max-width: 900px) { .main-content { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="container">
        <h1>智能监控系统</h1>
        <div class="main-content">
            <div class="video-section">
                <div class="video-container">
                    <img id="videoFrame" src="/video_feed">
                </div>
                <div class="controls">
                    <button class="btn-primary" onclick="toggleCamera()">启动/关闭摄像头</button>
                    <button class="btn-warning" onclick="captureCurrentFrame()">拍照添加人脸</button>
                    <button class="btn-success" onclick="startDrawingZone()">绘制敏感区域</button>
                    <button class="btn-warning" onclick="finishZoneDrawing()">完成绘制</button>
                    <button class="btn-danger" onclick="clearZones()">清除所有区域</button>
                </div>
                <div class="instructions">
                    点击"绘制敏感区域"后，在视频画面中点击多个点来定义区域<br>
                    点击"完成绘制"将首个点与最后一个点相连形成封闭区域<br>
                    点击"拍照添加人脸"可截取当前画面添加为人脸数据
                </div>
            </div>
            <div class="sidebar">
                <div class="panel">
                    <h3>人脸数据管理</h3>
                    <div class="upload-section">
                        <input type="text" id="personName" placeholder="输入姓名">
                        <div class="upload-row">
                            <label class="file-label" for="faceFile">上传图片</label>
                            <input type="file" id="faceFile" accept="image/*" onchange="uploadFace()">
                        </div>
                    </div>
                </div>
                <div class="panel">
                    <h3>敏感区域</h3>
                    <div class="zone-list" id="zoneList"></div>
                </div>
                <div class="panel" id="zoneDrawingPanel" style="display:none;background:#1a3a1a;">
                    <h3 style="color:#2ed573;"> 绘制中</h3>
                    <div id="zonePointsCount" style="font-size:14px;margin-bottom:10px;">已选择 0 个点</div>
                    <button class="btn-success" onclick="finishZoneDrawing()">完成绘制</button>
                    <button class="btn-danger" onclick="cancelZoneDrawing()">取消</button>
                </div>
                <div class="panel">
                    <h3>统计</h3>
                    <div class="stats">
                        <div class="stat-box"><div class="stat-num" id="knownCount">0</div><div class="stat-label">已知人员</div></div>
                        <div class="stat-box"><div class="stat-num" id="personCount">0</div><div class="stat-label">当前人数</div></div>
                        <div class="stat-box"><div class="stat-num" id="alertCount">0</div><div class="stat-label">警报数量</div></div>
                    </div>
                </div>
                <div class="panel">
                    <h3>警报信息</h3>
                    <button class="btn-danger" style="width:100%;margin-bottom:10px;" onclick="clearAlerts()">清除警报</button>
                    <div class="alert-list" id="alertList"></div>
                </div>
            </div>
        </div>
    </div>
    <div class="modal" id="captureModal">
        <div class="modal-content">
            <h3 style="color:#00d4ff;margin-bottom:10px;">添加人脸数据</h3>
            <img id="capturePreview">
            <input type="text" id="captureName" placeholder="输入姓名">
            <div style="display:flex;gap:10px;justify-content:center;">
                <button class="btn-success" onclick="confirmCapture()">确认添加</button>
                <button class="btn-danger" onclick="closeModal()">取消</button>
            </div>
        </div>
    </div>
    <script>
        let currentZoneName='',zonePoints=[],currentCapture=null;
        let isDrawingZone=false;
        function toggleCamera(){fetch('/toggle_camera',{method:'POST'}).then(r=>r.json()).then(d=>{alert(d.message||'操作完成');});}
        function captureCurrentFrame(){fetch('/capture_face',{method:'POST'}).then(r=>r.json()).then(d=>{if(d.success)window.captureSnapshot(d.image);else alert(d.message||'无法获取视频画面');});}

        function startDrawingZone(){
            const name=prompt('请输入区域名称:');
            if(name){
                currentZoneName=name;
                zonePoints=[];
                isDrawingZone=true;
                document.getElementById('videoFrame').style.cursor='crosshair';
                document.getElementById('zonePointsCount').textContent='已选择 0 个点';
                document.getElementById('zoneDrawingPanel').style.display='block';
            }
        }

        function finishZoneDrawing(){
            if(!isDrawingZone || zonePoints.length<3){
                alert('请至少选择3个点');
                return;
            }
            fetch('/save_zone',{
                method:'POST',
                headers:{'Content-Type':'application/json'},
                body:JSON.stringify({name:currentZoneName,points:zonePoints})
            }).then(r=>r.json()).then(d=>{
                if(d.status==='saved'){
                    currentZoneName='';
                    zonePoints=[];
                    isDrawingZone=false;
                    document.getElementById('videoFrame').style.cursor='default';
                    document.getElementById('zoneDrawingPanel').style.display='none';
                    fetch('/update_zone_preview',{
                        method:'POST',
                        headers:{'Content-Type':'application/json'},
                        body:JSON.stringify({points:[]})
                    });
                    updateZones();
                }
            });
        }

        function cancelZoneDrawing(){
            currentZoneName='';
            zonePoints=[];
            isDrawingZone=false;
            document.getElementById('videoFrame').style.cursor='default';
            document.getElementById('zoneDrawingPanel').style.display='none';
        }

        document.getElementById('videoFrame').addEventListener('click',function(e){
            if(!isDrawingZone || !currentZoneName) return;
            const rect=this.getBoundingClientRect();
            const scaleX=this.naturalWidth/rect.width;
            const scaleY=this.naturalHeight/rect.height;
            const x=Math.round((e.clientX-rect.left)*scaleX);
            const y=Math.round((e.clientY-rect.top)*scaleY);
            zonePoints.push([x,y]);
            document.getElementById('zonePointsCount').textContent='已选择 '+zonePoints.length+' 个点';
            updateZonePreview();
        });

        function updateZonePreview(){
            fetch('/update_zone_preview',{
                method:'POST',
                headers:{'Content-Type':'application/json'},
                body:JSON.stringify({points:zonePoints})
            });
        }
        function clearZones(){fetch('/clear_zones',{method:'POST'});}
        function deleteZone(index){fetch('/delete_zone',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({index:index})});}
        function uploadFace(){const file=document.getElementById('faceFile').files[0];const name=document.getElementById('personName').value;if(!file||!name){alert('请输入姓名并选择图片');return;}const reader=new FileReader();reader.onload=function(e){fetch('/add_face',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:name,image:e.target.result})}).then(r=>r.json()).then(d=>{alert(d.message);if(d.success){document.getElementById('personName').value='';document.getElementById('faceFile').value='';updateStats();}});};reader.readAsDataURL(file);}
        function clearAlerts(){fetch('/clear_alerts',{method:'POST'});}
        function closeModal(){document.getElementById('captureModal').style.display='none';currentCapture=null;}
        function confirmCapture(){const name=document.getElementById('captureName').value;if(!name||!currentCapture){alert('请输入姓名');return;}fetch('/add_face_from_camera',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:name,image:currentCapture})}).then(r=>r.json()).then(d=>{alert(d.message);if(d.success){closeModal();document.getElementById('captureName').value='';updateStats();}});}
        function updateZones(){fetch('/get_zones').then(r=>r.json()).then(d=>{document.getElementById('zoneList').innerHTML=d.zones.map((z,i)=>'<div class="zone-item"><span>'+z.name+'</span><button class="zone-delete" onclick="deleteZone('+i+')">删除</button></div>').join('');});}
        function updateAlerts(){fetch('/get_alerts').then(r=>r.json()).then(d=>{document.getElementById('alertList').innerHTML=d.alerts.map(a=>'<div class="alert-item '+a.type+'"><div class="alert-time">'+a.timestamp+'</div><div class="alert-msg">'+a.message+'</div></div>').join('');document.getElementById('alertCount').textContent=d.alerts.length;});}
        function updateStats(){fetch('/get_stats').then(r=>r.json()).then(d=>{document.getElementById('knownCount').textContent=d.known_count;document.getElementById('personCount').textContent=d.person_count;});}
        setInterval(updateZones,2000);setInterval(updateAlerts,1000);setInterval(updateStats,1000);
        window.captureSnapshot=function(imageData){currentCapture=imageData;document.getElementById('capturePreview').src=imageData;document.getElementById('captureModal').style.display='block';}
    </script>
</body>
</html>
"""


def process_frame():
    global current_frame, snapshot_frame, is_camera_active, camera
    known_persons_in_frame = set()
    while True:
        try:
            if is_camera_active and camera is not None:
                try:
                    ret, frame = camera.read()
                    if ret and frame is not None:
                        persons = detector.detect_persons(frame)
                        faces_with_encodings = detector.detect_faces_with_encodings(
                            frame
                        )
                        person_locations = {}
                        known_persons_in_frame.clear()

                        for face_rect, face_encoding in faces_with_encodings:
                            x, y, w, h = face_rect
                            name, confidence = face_db.recognize_face(
                                None, face_encoding
                            )

                            if name != "unknown":
                                color = (0, 255, 0)
                                known_persons_in_frame.add(name)
                                cv2.putText(
                                    frame,
                                    f"{name} ({confidence:.2f})",
                                    (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    color,
                                    2,
                                )
                            else:
                                color = (0, 165, 255)
                                cv2.putText(
                                    frame,
                                    f"Unknown ({confidence:.2f})",
                                    (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.8,
                                    color,
                                    2,
                                )
                            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                            center_x, center_y = x + w // 2, y + h // 2
                            person_locations[name] = (center_x, center_y)

                        for person in persons:
                            x1, y1, x2, y2 = person["bbox"]
                            person_center_x = (x1 + x2) // 2
                            person_center_y = (y1 + y2) // 2

                            person_has_known_face = False
                            matched_name = None

                            for face_rect, face_encoding in faces_with_encodings:
                                fx, fy, fw, fh = face_rect
                                face_center_x = fx + fw // 2
                                face_center_y = fy + fh // 2

                                if (
                                    x1 <= face_center_x <= x2
                                    and y1 <= face_center_y <= y2
                                ):
                                    name, confidence = face_db.recognize_face(
                                        None, face_encoding
                                    )
                                    if name != "unknown" and confidence > 0.5:
                                        person_has_known_face = True
                                        matched_name = name
                                        break

                            if person_has_known_face:
                                box_color = (0, 255, 0)
                                label = f"Known: {matched_name}"
                                person_locations[matched_name] = (
                                    person_center_x,
                                    person_center_y,
                                )
                            else:
                                box_color = (0, 255, 255)
                                label = f"Person {person['conf']:.2f}"
                                person_locations[
                                    f"person_{person_center_x}_{person_center_y}"
                                ] = (
                                    person_center_x,
                                    person_center_y,
                                )

                            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                            cv2.putText(
                                frame,
                                label,
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                box_color,
                                2,
                            )

                        for zone in zone_manager.get_zones():
                            pts = np.array(zone["points"], np.int32)
                            pts = pts.reshape((-1, 1, 2))
                            cv2.polylines(frame, [pts], True, (255, 0, 255), 2)
                            cv2.putText(
                                frame,
                                zone["name"],
                                (zone["points"][0][0], zone["points"][0][1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (255, 0, 255),
                                2,
                            )

                        global zone_preview_points
                        if zone_preview_points:
                            try:
                                frame = draw_zone_preview(frame, zone_preview_points)
                            except Exception as preview_error:
                                print(f"绘制预览错误: {preview_error}")

                        alerts = zone_manager.check_zones(person_locations)
                        print(f"[调试] 检测到 {len(alerts)} 个区域警报")
                        
                        global zone_alert_sent, last_zone_alert_time
                        
                        if len(alerts) > 0:
                            has_known_person_in_zone = False
                            for alert in alerts:
                                alert_person = alert["person"]
                                if alert_person in known_persons_in_frame:
                                    has_known_person_in_zone = True
                                    break
                            
                            if has_known_person_in_zone:
                                if zone_alert_sent:
                                    print(f"[警报解除] 检测到已知人员在区域内，发送解除警报消息")
                                    alert_obj = alert_manager.add_alert(
                                        "zone_clear", "警报解除：已知人员在区域内", "known"
                                    )
                                    if alert_obj and ASTRBOT_ENABLED and surveillance_client:
                                        surveillance_client.push_alert(
                                            "zone_clear", "警报解除：已知人员在区域内", "known"
                                        )
                                        try:
                                            import asyncio
                                            from threading import Thread

                                            Thread(
                                                target=lambda: (asyncio.new_event_loop(), asyncio.set_event_loop(asyncio.new_event_loop()), asyncio.run(
                                                    send_alert_via_llm_with_image(
                                                        "警报解除",
                                                        "警报解除：已知人员在区域内",
                                                        "known",
                                                        get_latest_snapshot()
                                                    )
                                                ))
                                            ).start()
                                        except Exception as e:
                                            print(f"异步任务创建失败: {e}")
                                    zone_alert_sent = False
                            else:
                                if not zone_alert_sent or (last_zone_alert_time and (time.time() - last_zone_alert_time) > 60):
                                    print(f"[调试] 发送区域入侵警报")
                                    for alert in alerts:
                                        alert_person = alert["person"]
                                        alert_obj = alert_manager.add_alert(
                                            "zone_alert",
                                            f"{alert['person']}进入{alert['zone']}",
                                            alert["person"],
                                        )
                                        if alert_obj and ASTRBOT_ENABLED and surveillance_client:
                                            surveillance_client.push_alert(
                                                "zone_alert",
                                                f"{alert['person']}进入{alert['zone']}",
                                                alert["person"],
                                            )
                                            try:
                                                import asyncio
                                                from threading import Thread

                                                Thread(
                                                    target=lambda: (asyncio.new_event_loop(), asyncio.set_event_loop(asyncio.new_event_loop()), asyncio.run(
                                                        send_alert_via_llm(
                                                            "区域入侵",
                                                            f"{alert['person']}进入{alert['zone']}",
                                                            alert["person"],
                                                        )
                                                    ))
                                                ).start()
                                            except Exception as e:
                                                print(f"异步任务创建失败: {e}")
                                    zone_alert_sent = True
                                    last_zone_alert_time = time.time()

                        global stranger_alert_sent, last_stranger_alert_time
                        
                        if len(persons) > 0 and len(known_persons_in_frame) == 0:
                            print(f"[调试] 检测到陌生人: {len(persons)} 个人，{len(known_persons_in_frame)} 个已知人员，{len(faces_with_encodings)} 个人脸")
                            
                            if not stranger_alert_sent or (last_stranger_alert_time and (time.time() - last_stranger_alert_time) > 60):
                                alert = alert_manager.add_alert(
                                    "stranger", "检测到陌生人!", "unknown"
                                )
                                print(f"[调试] 警报对象: {alert}")
                                if (
                                    alert
                                    and ASTRBOT_ENABLED
                                    and surveillance_client
                                ):
                                    print(f"[调试] 发送陌生人警报")
                                    surveillance_client.push_alert(
                                        "stranger", "检测到陌生人!", "unknown"
                                    )
                                    try:
                                        import asyncio
                                        from threading import Thread

                                        Thread(
                                            target=lambda: (asyncio.new_event_loop(), asyncio.set_event_loop(asyncio.new_event_loop()), asyncio.run(
                                                send_alert_via_llm_with_image(
                                                    "陌生人",
                                                    "检测到陌生人!",
                                                    "unknown",
                                                    get_latest_snapshot()
                                                )
                                            ))
                                        ).start()
                                    except Exception as e:
                                        print(f"异步任务创建失败: {e}")
                                    stranger_alert_sent = True
                                    last_stranger_alert_time = time.time()
                                else:
                                    print(f"[调试] 警报未发送: alert={alert}, ASTRBOT_ENABLED={ASTRBOT_ENABLED}, surveillance_client={surveillance_client}")
                        else:
                            if stranger_alert_sent and len(known_persons_in_frame) > 0:
                                print(f"[警报解除] 检测到已知人员，发送解除陌生人警报消息")
                                alert_obj = alert_manager.add_alert(
                                    "stranger_clear", "警报解除：检测到已知人员", "known"
                                )
                                if alert_obj and ASTRBOT_ENABLED and surveillance_client:
                                    surveillance_client.push_alert(
                                        "stranger_clear", "警报解除：检测到已知人员", "known"
                                    )
                                    try:
                                        import asyncio
                                        from threading import Thread

                                        Thread(
                                            target=lambda: (asyncio.new_event_loop(), asyncio.set_event_loop(asyncio.new_event_loop()), asyncio.run(
                                                send_alert_via_llm_with_image(
                                                    "警报解除",
                                                    "警报解除：检测到已知人员",
                                                    "known",
                                                    get_latest_snapshot()
                                                )
                                            ))
                                        ).start()
                                    except Exception as e:
                                        print(f"异步任务创建失败: {e}")
                                stranger_alert_sent = False

                        snapshot_frame = frame.copy()
                        with frame_lock:
                            current_frame = frame.copy()

                        global last_snapshot_time
                        if (
                            "last_snapshot_time" not in globals()
                            or (time.time() - last_snapshot_time) > 1
                        ):
                            last_snapshot_time = time.time()
                            save_snapshot_image(frame)
                    else:
                        print("摄像头读取失败，尝试重新初始化...")
                        time.sleep(0.1)
                except Exception as camera_error:
                    print(f"摄像头操作错误: {camera_error}")
                    try:
                        if camera:
                            camera.release()
                            camera = None
                        time.sleep(0.5)
                    except Exception:
                        pass
            else:
                time.sleep(0.03)
        except Exception as e:
            print(f"处理帧错误: {e}")
            time.sleep(0.03)


process_thread = None


def start_process_thread():
    global process_thread, detector, face_db, zone_manager, alert_manager
    if detector is None:
        init_detector()
    if process_thread is None or not process_thread.is_alive():
        process_thread = threading.Thread(target=process_frame, daemon=True)
        process_thread.start()


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/video_feed")
def video_feed():
    def generate():
        while True:
            try:
                frame_to_send = None
                with frame_lock:
                    if current_frame is not None:
                        frame_to_send = current_frame.copy()
                    elif is_camera_active and camera is not None:
                        ret, frame = camera.read()
                        if ret:
                            frame_to_send = frame
                if frame_to_send is not None:
                    try:
                        _, buffer = cv2.imencode(
                            ".jpg", frame_to_send, [cv2.IMWRITE_JPEG_QUALITY, 70]
                        )
                        frame_bytes = buffer.tobytes()
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                        )
                    except Exception as e:
                        print(f"编码错误: {e}")
                time.sleep(0.05)
            except Exception as e:
                print(f"视频流错误: {e}")
                time.sleep(0.1)

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/toggle_camera", methods=["POST"])
def toggle_camera():
    global is_camera_active, camera
    if is_camera_active:
        print("关闭摄像头...")
        is_camera_active = False
        if camera:
            camera.release()
            camera = None
        print("摄像头已关闭")
        return jsonify({"active": False, "message": "摄像头已关闭"})
    else:
        print("开始尝试打开摄像头...")
        # 尝试更多的摄像头索引
        for i in range(10):
            try:
                print(f"尝试摄像头索引 {i}...")
                # 尝试不同的参数打开摄像头
                camera = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                print(f"摄像头 {i} 初始化完成")

                if camera.isOpened():
                    print(f"摄像头 {i} 打开成功")
                    # 设置摄像头参数
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    camera.set(cv2.CAP_PROP_FPS, 30)

                    # 尝试读取多帧以确保稳定性
                    success_count = 0
                    for _ in range(5):
                        ret, test_frame = camera.read()
                        if ret and test_frame is not None:
                            success_count += 1
                            print(
                                f"摄像头 {i} 第 {_ + 1} 帧读取成功，分辨率: {test_frame.shape[1]}x{test_frame.shape[0]}"
                            )
                        else:
                            print(f"摄像头 {i} 第 {_ + 1} 帧读取失败")
                        time.sleep(0.1)
                    if success_count >= 1:
                        print(f"摄像头 {i} 测试通过，开始使用")
                        is_camera_active = True
                        start_process_thread()
                        return jsonify(
                            {"active": True, "message": f"摄像头 {i} 已开启"}
                        )
                    else:
                        print(f"摄像头 {i} 无法稳定读取帧")
                        camera.release()
                        camera = None
                else:
                    print(f"摄像头 {i} 无法打开")
                    if camera:
                        camera.release()
                        camera = None
            except Exception as e:
                print(f"尝试摄像头 {i} 失败: {e}")
                if camera:
                    try:
                        camera.release()
                    except Exception:
                        pass
                    camera = None
        print("所有摄像头都无法打开")
        return jsonify({"active": False, "message": "无法打开摄像头，请检查连接"})


@app.route("/save_zone", methods=["POST"])
def save_zone():
    global zone_preview_points
    data = request.json
    name = data.get("name", "区域")
    points = data.get("points", [])
    if len(points) >= 3:
        zone_manager.add_zone(name, points)
        zone_preview_points = []
        return jsonify({"status": "saved"})
    return jsonify({"status": "error", "message": "需要至少3个点"})


@app.route("/update_zone_preview", methods=["POST"])
def update_zone_preview():
    global zone_preview_points
    data = request.json
    zone_preview_points = data.get("points", [])
    return jsonify({"status": "ok"})


@app.route("/zone_preview")
def zone_preview():
    global zone_preview_points
    return jsonify({"preview": zone_preview_points})


def draw_zone_preview(frame, points):
    import cv2
    import numpy as np

    preview_frame = frame.copy()

    if len(points) >= 1:
        for i, pt in enumerate(points):
            cv2.circle(preview_frame, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
            cv2.putText(
                preview_frame,
                str(i + 1),
                (int(pt[0]) + 8, int(pt[1]) + 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

    if len(points) >= 2:
        for i in range(len(points) - 1):
            cv2.line(
                preview_frame,
                (int(points[i][0]), int(points[i][1])),
                (int(points[i + 1][0]), int(points[i + 1][1])),
                (0, 255, 0),
                2,
            )

    if len(points) >= 3:
        cv2.line(
            preview_frame,
            (int(points[-1][0]), int(points[-1][1])),
            (int(points[0][0]), int(points[0][1])),
            (0, 255, 0),
            2,
        )

        pts = np.array([[int(p[0]), int(p[1])] for p in points], np.int32)
        overlay = preview_frame.copy()
        cv2.fillPoly(overlay, [pts], (0, 100, 0))
        cv2.addWeighted(overlay, 0.3, preview_frame, 0.7, 0, preview_frame)
        cv2.polylines(preview_frame, [pts], True, (0, 255, 0), 2)

    return preview_frame


def set_zone_preview_points(points):
    global zone_preview_points
    zone_preview_points = points


def get_zone_preview_points():
    global zone_preview_points
    return zone_preview_points


@app.route("/clear_zones", methods=["POST"])
def clear_zones():
    global zone_preview_points
    zone_manager.zones = []
    zone_preview_points = []
    return jsonify({"status": "cleared"})


@app.route("/delete_zone", methods=["POST"])
def delete_zone():
    data = request.json
    zone_manager.remove_zone(data.get("index", 0))
    return jsonify({"status": "deleted"})


@app.route("/add_face", methods=["POST"])
def add_face():
    data = request.json
    success, message = face_db.add_known_face(data["name"], data["image"])
    return jsonify({"success": success, "message": message})


@app.route("/add_face_from_camera", methods=["POST"])
def add_face_from_camera():
    data = request.json
    name = data.get("name", "")
    image_data = data.get("image", "")
    if not name:
        return jsonify({"success": False, "message": "请输入姓名"})
    img_bytes = base64.b64decode(image_data.split(",")[1])
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    bbox = [50, 50, frame.shape[1] - 50, frame.shape[0] - 50]
    success, message = face_db.add_face_from_camera(name, frame, bbox)
    return jsonify({"success": success, "message": message})


@app.route("/capture_face", methods=["POST"])
def capture_face():
    global snapshot_frame
    if snapshot_frame is None:
        return jsonify({"success": False, "message": "无视频画面"})
    _, buffer = cv2.imencode(".jpg", snapshot_frame)
    image_data = base64.b64encode(buffer).decode("utf-8")
    return jsonify({"success": True, "image": f"data:image/jpeg;base64,{image_data}"})


@app.route("/get_zones", methods=["GET"])
def get_zones():
    return jsonify({"zones": zone_manager.get_zones()})


@app.route("/get_alerts", methods=["GET"])
def get_alerts():
    return jsonify({"alerts": alert_manager.get_alerts()})


@app.route("/clear_alerts", methods=["POST"])
def clear_alerts():
    alert_manager.clear_alerts()
    return jsonify({"status": "cleared"})


@app.route("/get_stats", methods=["GET"])
def get_stats():
    person_count = 0
    if is_camera_active and camera is not None:
        ret, frame = camera.read()
        if ret:
            persons = detector.detect_persons(frame)
            person_count = len(persons)
    return jsonify(
        {"known_count": len(face_db.known_names), "person_count": person_count}
    )


@app.route("/test_send_alert", methods=["GET", "POST"])
def test_send_alert():
    if not ASTRBOT_ENABLED or not _context or not platform_manager:
        return jsonify({"success": False, "message": "LLM 未初始化或不可用"})

    async def do_send():
        await send_alert_via_llm("测试警报", "这是一条测试消息", "测试人员")

    try:
        try:
            asyncio.get_running_loop()
            asyncio.create_task(do_send())
        except RuntimeError:
            asyncio.run(do_send())
        return jsonify({"success": True, "message": "测试警报已发送"})
    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)})


@app.route("/test_send_direct", methods=["GET", "POST"])
def test_send_direct():
    print("[test_send_direct] 开始测试直接发送")

    async def do_send():
        from .client import surveillance_client

        print(
            f"[test_send_direct] surveillance_client._context: {surveillance_client._context}"
        )
        print(
            f"[test_send_direct] surveillance_client._platform_manager: {surveillance_client._platform_manager}"
        )

        try:
            from astrbot.core.star.star_tools import StarTools

            print(f"[test_send_direct] StarTools._context: {StarTools._context}")
        except Exception as e:
            print(f"[test_send_direct] 无法获取 StarTools: {e}")

        await surveillance_client.send_text("177352601", "这是一条直接测试消息")

    def run_async():
        try:
            asyncio.run(do_send())
        except Exception as e:
            print(f"[test_send_direct] run_async 错误: {e}")
            import traceback

            traceback.print_exc()

    try:
        import threading

        thread = threading.Thread(target=run_async, daemon=True)
        thread.start()
        return jsonify({"success": True, "message": "直接测试消息已发送"})
    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)})


@app.route("/test_llm_story", methods=["GET", "POST"])
def test_llm_story():
    print("[test_llm_story] 开始测试 LLM 故事生成并发送")

    context = surveillance_client._context if surveillance_client else None
    print(f"[test_llm_story] surveillance_client._context: {context}")
    print(f"[test_llm_story] _context: {_context}")

    if not ASTRBOT_ENABLED:
        return jsonify({"success": False, "message": "AstrBot 模式未启用"})

    if not context:
        context = _context

    if not context:
        return jsonify(
            {"success": False, "message": "Context 未初始化，请检查插件是否正确加载"}
        )

    async def do_send():
        try:
            from astrbot.api.message_components import Plain
            from astrbot.core.message.message_event_result import MessageChain
            from astrbot.core.star.star_tools import StarTools

            print("[test_llm_story] 正在获取 provider...")
            provider = context.get_using_provider()
            print(f"[test_llm_story] provider: {provider}")

            if not provider:
                print("[test_llm_story] 未找到 provider，尝试直接发送")
                await surveillance_client.send_text(
                    "177352601", "未配置 LLM Provider，这是测试消息"
                )
                return

            provider_id = provider.meta().id
            print(f"[test_llm_story] provider ID: {provider_id}")

            prompt = "请讲一个简短的有趣故事，不超过100字。"
            print(f"[test_llm_story] 正在调用 LLM: {prompt}")

            response = await context.llm_generate(
                chat_provider_id=provider_id, prompt=prompt
            )

            story = ""
            if response.result_chain and response.result_chain.chain:
                for comp in response.result_chain.chain:
                    if hasattr(comp, "text"):
                        story = comp.text
                        break

            print(f"[test_llm_story] LLM 返回: {story}")

            if not story:
                story = "LLM 未返回内容"

            target_qq = "177352601"
            print(f"[test_llm_story] 发送故事到 {target_qq}")

            msg_chain = MessageChain([Plain(text=f"📖 故事时间：\n\n{story}")])

            if StarTools and StarTools._context is not None:
                print("[test_llm_story] 使用 StarTools 发送")
                await StarTools.send_message_by_id(
                    type="PrivateMessage",
                    id=target_qq,
                    message_chain=msg_chain,
                    platform="aiocqhttp",
                )
                print("[test_llm_story] StarTools 发送成功")
            else:
                print(
                    "[test_llm_story] StarTools 未初始化，使用 surveillance_client 发送"
                )
                await surveillance_client.send_text(target_qq, story)

            print("[test_llm_story] 完成!")

        except Exception as e:
            print(f"[test_llm_story] 错误: {e}")
            import traceback

            traceback.print_exc()

    def run_async():
        try:
            asyncio.run(do_send())
        except Exception as e:
            print(f"[test_llm_story] run_async 错误: {e}")
            import traceback

            traceback.print_exc()

    try:
        import threading

        thread = threading.Thread(target=run_async, daemon=True)
        thread.start()
        return jsonify({"success": True, "message": "LLM 故事测试已发送"})
    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)})


@app.route("/test_simulate_alert", methods=["GET", "POST"])
def test_simulate_alert():
    print("[test_simulate_alert] 模拟监控警报")

    async def do_send():
        try:
            from astrbot.api.message_components import Plain
            from astrbot.core.message.message_event_result import MessageChain
            from astrbot.core.star.star_tools import StarTools

            context = surveillance_client._context if surveillance_client else _context
            print(f"[test_simulate_alert] context: {context}")

            if not context:
                print("[test_simulate_alert] context 为空")
                return

            provider = context.get_using_provider()
            print(f"[test_simulate_alert] provider: {provider}")

            alert_type = "人员检测"
            is_sensitive_zone = False

            environment = "室内环境，光线充足"
            person_details = "男性，约30-40岁，体型偏胖，穿着深色衣服"
            action = "在监控区域内缓慢行走，停留时间较长"
            message = f"检测到{person_details}出现在监控区域，{action}"

            if provider:
                provider_id = provider.meta().id
                print(f"[test_simulate_alert] provider ID: {provider_id}")

                if is_sensitive_zone:
                    print(
                        "[test_simulate_alert] 敏感区域警报，冷却中，跳过 LLM 和 QQ 推送"
                    )
                    return

                prompt = f"""你是一个智能监控系统的警报通知助手。请根据以下监控画面信息生成一条详细的警报通知：

环境：{environment}
警报类型：{alert_type}
人物特征：{person_details}
行为动作：{action}

请生成一条不超过100字的详细通知消息，包含环境、人物和动作信息。"""

                print("[test_simulate_alert] 正在调用 LLM...")
                response = await context.llm_generate(
                    chat_provider_id=provider_id, prompt=prompt
                )

                llm_message = ""
                if response.result_chain and response.result_chain.chain:
                    for comp in response.result_chain.chain:
                        if hasattr(comp, "text"):
                            llm_message = comp.text
                            break

                print(f"[test_simulate_alert] LLM 返回: {llm_message}")
            else:
                llm_message = f"⚠️ {alert_type}: {message}"

            if not llm_message:
                llm_message = f"⚠️ {alert_type}: {message}"

            if is_sensitive_zone:
                print("[test_simulate_alert] 敏感区域警报，冷却中，跳过 QQ 推送")
                return

            target_qq = "177352601"
            print(f"[test_simulate_alert] 发送警报到 {target_qq}")

            msg_chain = MessageChain([Plain(text=f"🚨 监控警报\n\n{llm_message}")])

            if StarTools and StarTools._context is not None:
                print("[test_simulate_alert] 使用 StarTools 发送")
                await StarTools.send_message_by_id(
                    type="PrivateMessage",
                    id=target_qq,
                    message_chain=msg_chain,
                    platform="aiocqhttp",
                )
                print("[test_simulate_alert] StarTools 发送成功")
            else:
                print(
                    "[test_simulate_alert] StarTools 未初始化，使用 surveillance_client 发送"
                )
                await surveillance_client.send_text(target_qq, llm_message)

            print("[test_simulate_alert] 完成!")

        except Exception as e:
            print(f"[test_simulate_alert] 错误: {e}")
            import traceback

            traceback.print_exc()

    def run_async():
        try:
            asyncio.run(do_send())
        except Exception as e:
            print(f"[test_simulate_alert] run_async 错误: {e}")
            import traceback

            traceback.print_exc()

    try:
        import threading

        thread = threading.Thread(target=run_async, daemon=True)
        thread.start()
        return jsonify({"success": True, "message": "模拟警报已发送"})
    except Exception as e:
        import traceback

        traceback.print_exc()
        return jsonify({"success": False, "message": str(e)})


def run_web_server():
    init_detector()
    start_process_thread()
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    run_web_server()
