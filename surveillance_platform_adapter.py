import asyncio
from datetime import datetime

try:
    from astrbot import logger
    from astrbot.api.event import MessageChain
    from astrbot.api.message_components import Image, Plain
    from astrbot.api.platform import (
        AstrBotMessage,
        MessageMember,
        MessageType,
        Platform,
        PlatformMetadata,
        register_platform_adapter,
    )
    from astrbot.core.platform.astr_message_event import MessageSesion

    ASTRBOT_AVAILABLE = True
except ImportError:
    ASTRBOT_AVAILABLE = False
    logger = None

from .client import surveillance_client

if ASTRBOT_AVAILABLE:

    @register_platform_adapter(
        "craic_surveillance",
        "CRAIC智能监控适配器",
        default_config_tmpl={
            "enabled": True,
            "notify_on_stranger": True,
            "notify_on_zone": True,
        },
    )
    class SurveillancePlatformAdapter(Platform):
        def __init__(
            self,
            platform_config: dict,
            platform_settings: dict,
            event_queue: asyncio.Queue,
        ) -> None:
            super().__init__(platform_config, event_queue)
            self.config = platform_config
            self.settings = platform_settings
            self.client = surveillance_client

        async def send_by_session(
            self, session: MessageSesion, message_chain: MessageChain
        ):
            sender_id = session.session_id
            for i in message_chain.chain:
                if isinstance(i, Plain):
                    await self.client.send_text(to=sender_id, message=i.text)
                elif isinstance(i, Image):
                    img_url = i.file
                    img_path = ""
                    if img_url.startswith("file:///"):
                        img_path = img_url[8:]
                    elif i.file and i.file.startswith("http"):
                        img_path = await self._download_image(img_url)
                    else:
                        img_path = img_url
                    await self.client.send_image(to=sender_id, image_path=img_path)
            await super().send_by_session(session, message_chain)

        async def _download_image(self, url: str) -> str:
            try:
                from astrbot.core.utils.io import download_image_by_url

                return await download_image_by_url(url)
            except Exception as e:
                logger.error(f"下载图片失败: {e}")
                return url

        def meta(self) -> PlatformMetadata:
            return PlatformMetadata(
                name="craic_surveillance",
                description="CRAIC智能监控适配器",
                id="craic_surveillance",
            )

        async def run(self):
            async def on_received(data):
                logger.info(f"[智能监控] 收到警报: {data}")
                abm = await self.convert_message(data=data)
                await self.handle_msg(abm)

            self.client.on_alert_received = on_received
            await self.client.start_polling()

        async def convert_message(self, data: dict) -> AstrBotMessage:
            abm = AstrBotMessage()
            abm.type = MessageType.FRIEND_MESSAGE
            abm.message_str = data["message"]
            abm.sender = MessageMember(
                user_id="surveillance_system", nickname="智能监控系统"
            )
            abm.message = [Plain(text=data["message"])]
            abm.raw_message = data
            abm.self_id = "surveillance_bot"
            abm.session_id = data["session_id"]
            abm.message_id = f"alert_{datetime.now().timestamp()}"

            return abm

        async def handle_msg(self, message: AstrBotMessage):
            from .surveillance_platform_event import SurveillancePlatformEvent

            message_event = SurveillancePlatformEvent(
                message_str=message.message_str,
                message_obj=message,
                platform_meta=self.meta(),
                session_id=message.session_id,
                client=self.client,
                raw_data=message.raw_message,
            )
            self.commit_event(message_event)
