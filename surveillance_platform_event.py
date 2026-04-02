try:
    from astrbot.api.event import AstrMessageEvent, MessageChain
    from astrbot.api.message_components import Image, Plain
    from astrbot.core.utils.io import download_image_by_url

    ASTRBOT_AVAILABLE = True
except ImportError:
    ASTRBOT_AVAILABLE = False
    AstrMessageEvent = object
    MessageChain = object


class SurveillancePlatformEvent(AstrMessageEvent if ASTRBOT_AVAILABLE else object):
    def __init__(
        self,
        message_str: str,
        message_obj,
        platform_meta,
        session_id: str,
        client,
        raw_data: dict,
    ):
        if ASTRBOT_AVAILABLE:
            super().__init__(message_str, message_obj, platform_meta, session_id)
        self.client = client
        self.raw_data = raw_data
        self.alert_type = raw_data.get("alert_type", "unknown")
        self.person = raw_data.get("person", "unknown")
        self.image_path = raw_data.get("image_path")
        self.target_qqs = (
            client.target_qqs
            if hasattr(client, "target_qqs")
            else ["YOUR_QQ_NUMBER", "YOUR_QQ_NUMBER"]
        )

    async def send(self, message):
        if not ASTRBOT_AVAILABLE:
            return

        for target_qq in self.target_qqs:
            for i in message.chain:
                if isinstance(i, Plain):
                    await self.client.send_text(to=target_qq, message=i.text)
                elif isinstance(i, Image):
                    img_url = i.file
                    img_path = ""
                    if img_url.startswith("file:///"):
                        img_path = img_url[8:]
                    elif i.file and i.file.startswith("http"):
                        img_path = await download_image_by_url(img_url)
                    else:
                        img_path = img_url
                    await self.client.send_image(to=target_qq, image_path=img_path)

        if ASTRBOT_AVAILABLE:
            await super().send(message)

    def get_alert_info(self) -> dict:
        return {
            "type": self.alert_type,
            "person": self.person,
            "message": self.message_str,
            "image": self.image_path,
            "timestamp": self.raw_data.get("timestamp", ""),
        }
