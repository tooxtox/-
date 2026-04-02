import asyncio
from datetime import datetime

try:
    from astrbot import logger
    from astrbot.core.star.star_tools import StarTools
except ImportError:
    logger = None
    StarTools = None


class SurveillanceClient:
    def __init__(self):
        self.on_alert_received = None
        self.alert_queue = asyncio.Queue()
        self.running = False
        self.target_qqs = ["YOUR_QQ_NUMBER", "YOUR_QQ_NUMBER"]
        self._context = None
        self._platform_manager = None

    def initialize(self, context):
        self._context = context
        if context:
            self._platform_manager = context.platform_manager
        if StarTools and context:
            StarTools.initialize(context)
        print("[SurveillanceClient] 初始化完�?)
        print(f"[SurveillanceClient] context: {context is not None}")
        print(
            f"[SurveillanceClient] platform_manager: {self._platform_manager is not None}"
        )

    def _get_platform_manager(self):
        if self._platform_manager:
            return self._platform_manager
        try:
            from astrbot.core.platform.manager import PlatformManager
            from astrbot.core.star.context import Context

            for inst in Context.__dict__.values():
                if isinstance(inst, PlatformManager):
                    self._platform_manager = inst
                    return inst
        except Exception:
            pass
        return None

    async def start_polling(self):
        self.running = True
        while self.running:
            try:
                alert = await asyncio.wait_for(self.alert_queue.get(), timeout=1.0)
                if self.on_alert_received and alert:
                    await self.on_alert_received(alert)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                if logger:
                    logger.error(f"警报轮询错误: {e}")

    async def send_text(self, to: str, message: str):
        print(f"[SurveillanceClient] 准备发送消息到 {to}: {message}")
        try:
            if StarTools and StarTools._context is not None:
                from astrbot.api.message_components import Plain
                from astrbot.core.message.message_event_result import MessageChain

                msg_chain = MessageChain([Plain(text=message)])
                print("[SurveillanceClient] StarTools 已初始化，使�?StarTools 发�?)
                await StarTools.send_message_by_id(
                    type="PrivateMessage",
                    id=to,
                    message_chain=msg_chain,
                    platform="aiocqhttp",
                )
                print("[SurveillanceClient] StarTools 发送消息成�?)
            else:
                print(
                    "[SurveillanceClient] StarTools 未初始化或_context为None，尝试直接发�?
                )
                if StarTools:
                    print(
                        f"[SurveillanceClient] StarTools._context = {StarTools._context}"
                    )
                await self._send_via_platform_manager(to, message, None)
        except Exception as e:
            print(f"[SurveillanceClient] 发送消息失�? {e}")
            import traceback

            traceback.print_exc()

    async def send_image(self, to: str, image_path: str):
        print(f"[SurveillanceClient] 准备发送图片到 {to}: {image_path}")
        try:
            if StarTools and StarTools._context is not None:
                from astrbot.api.message_components import Image
                from astrbot.core.message.message_event_result import MessageChain

                msg_chain = MessageChain([Image(file=image_path)])
                print(
                    "[SurveillanceClient] StarTools 已初始化，使�?StarTools 发送图�?
                )
                await StarTools.send_message_by_id(
                    type="PrivateMessage",
                    id=to,
                    message_chain=msg_chain,
                    platform="aiocqhttp",
                )
                print("[SurveillanceClient] StarTools 发送图片成�?)
            else:
                print(
                    "[SurveillanceClient] StarTools 未初始化或_context为None，尝试直接发�?
                )
                if StarTools:
                    print(
                        f"[SurveillanceClient] StarTools._context = {StarTools._context}"
                    )
                await self._send_via_platform_manager(to, None, image_path)
        except Exception as e:
            print(f"[SurveillanceClient] 发送图片失�? {e}")
            import traceback

            traceback.print_exc()

    async def _send_via_platform_manager(
        self, to: str, text: str = None, image_path: str = None
    ):
        print(f"[SurveillanceClient] _send_via_platform_manager 被调�?to={to}")
        try:
            from astrbot.api.message_components import Image, Plain
            from astrbot.core.message.message_event_result import MessageChain
            from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_message_event import (
                AiocqhttpMessageEvent,
            )
            from astrbot.core.platform.sources.aiocqhttp.aiocqhttp_platform_adapter import (
                AiocqhttpAdapter,
            )

            if self._context:
                platforms = self._context.platform_manager.get_insts()
                print(
                    f"[SurveillanceClient] 找到 {len(platforms)} 个平�? {[type(p).__name__ for p in platforms]}"
                )
            else:
                print("[SurveillanceClient] context 为空，无法获取平台列�?)
                return

            adapter = None
            for p in platforms:
                print(
                    f"[SurveillanceClient] 检查平�? {type(p).__name__}, isinstance={isinstance(p, AiocqhttpAdapter)}"
                )
                if isinstance(p, AiocqhttpAdapter):
                    adapter = p
                    print("[SurveillanceClient] 找到 AiocqhttpAdapter")
                    break

            if not adapter:
                print("[SurveillanceClient] 未找�?aiocqhttp 适配�?)
                return

            if text and image_path:
                msg_chain = MessageChain([Plain(text=text), Image(file=image_path)])
            elif text:
                msg_chain = MessageChain([Plain(text=text)])
            elif image_path:
                msg_chain = MessageChain([Image(file=image_path)])
            else:
                return

            print(
                f"[SurveillanceClient] 准备发�? is_group=False, session_id={to}, adapter.bot={adapter.bot is not None}"
            )
            print("[SurveillanceClient] 调用 AiocqhttpMessageEvent.send_message")
            await AiocqhttpMessageEvent.send_message(
                bot=adapter.bot,
                message_chain=msg_chain,
                is_group=False,
                session_id=to,
            )
            print("[SurveillanceClient] AiocqhttpMessageEvent.send_message 调用完成")
        except Exception as e:
            print(f"[SurveillanceClient] 通过 platform_manager 发送失�? {e}")
            import traceback

            traceback.print_exc()

    def push_alert(
        self,
        alert_type: str,
        message: str,
        person: str = "unknown",
        image_path: str = None,
    ):
        alert = {
            "alert_type": alert_type,
            "message": message,
            "person": person,
            "timestamp": datetime.now().isoformat(),
            "image_path": image_path,
            "session_id": "surveillance_main",
        }
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(self.alert_queue.put(alert))
            else:
                loop.run_until_complete(self.alert_queue.put(alert))
        except Exception:
            pass


surveillance_client = SurveillanceClient()
