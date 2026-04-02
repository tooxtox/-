try:
    from astrbot.api.star import Context, Star, register

    ASTRBOT_MODE = True
except ImportError:
    ASTRBOT_MODE = False

print(f"[CRAIC] ASTRBOT_MODE = {ASTRBOT_MODE}")

if ASTRBOT_MODE:

    @register(
        name="智能监控插件",
        version="1.0.0",
        desc="AstrBot智能监控系统插件",
        author="CRAIC",
    )
    class SurveillancePlugin(Star):
        def __init__(self, context: Context):
            print("[CRAIC] SurveillancePlugin 初始化开始")
            from .client import surveillance_client

            print(f"[CRAIC] surveillance_client id: {id(surveillance_client)}")
            print(f"[CRAIC] context: {context}")
            self.context = context
            surveillance_client.initialize(context)
            print("[CRAIC] surveillance_client.initialize 完成")
            import threading

            from . import web_server

            web_server.init_llm(context)
            print("[CRAIC] web_server.init_llm 完成")
            thread = threading.Thread(target=web_server.run_web_server, daemon=True)
            thread.start()
            print("[CRAIC] SurveillancePlugin 初始化完成")
else:
    print("[CRAIC] 非 AstrBot 模式，启动独立 Web 服务器")
    from web_server import run_web_server

    run_web_server()
