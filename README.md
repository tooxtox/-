# 智能监控插件

## 📖 项目简介

CRAIC (Smart Surveillance System) 是一个基于AstrBot的智能监控系统插件，集成了计算机视觉和人工智能技术，提供实时的人脸识别、人体检测、敏感区域监控功能，并能通过QQ发送警报通知。

## ✨ 功能特性

### 核心功能
- **🤖 人脸识别**：支持已知人员识别和陌生人检测，使用OpenCV DNN和Haar Cascade双重检测
- **👥 人体检测**：基于YOLOv8模型进行实时人体检测
- **🚨 敏感区域监控**：支持自定义绘制敏感区域，实时监控人员进入情况
- **📱 实时警报**：通过QQ发送警报通知，支持陌生人警报和区域入侵警报
- **🧠 LLM集成**：使用大模型生成详细的警报描述和图片分析
- **🌐 Web界面**：提供直观的Web界面进行监控和管理

### 特色功能
- **⚡ 实时处理**：毫秒级的视频帧处理和检测
- **🔄 警报解除**：支持智能解除警报，当检测到已知人员时自动发送解除通知
- **📸 图片描述**：使用LLM分析监控画面，生成详细的场景描述
- **🎨 可视化界面**：美观的深色主题界面，支持响应式设计
- **💾 截图保存**：自动保存警报截图，支持历史记录查看

## 🛠️ 技术栈

- **后端框架**：Flask
- **计算机视觉**：OpenCV, Ultralytics YOLO
- **AI框架**：PyTorch
- **聊天机器人**：AstrBot
- **前端技术**：HTML5, CSS3, JavaScript (ES6+)

## 📦 安装依赖

### 系统要求
- Python 3.8+
- Windows/Linux/macOS
- 摄像头设备
- AstrBot运行环境

### 安装步骤

```bash
# 从GitHub克隆插件
git clone https://github.com/tooxtox/-.git
cd -

# 或者下载ZIP包并解压
# 访问 https://github.com/tooxtox/- 下载最新版本

# 安装Python依赖
pip install -r requirements.txt
```

### 依赖说明

主要依赖包：
- `opencv-python` - 计算机视觉库
- `ultralytics` - YOLO模型
- `torch` - PyTorch深度学习框架
- `flask` - Web框架
- `numpy` - 数值计算
- `pillow` - 图像处理

## ⚙️ 配置说明

### 1. 修改QQ号码

在 `web_server.py` 中修改接收警报的QQ号码：

```python
target_qqs = ["YOUR_QQ_NUMBER"]  # 替换为你的QQ号
```

### 2. 下载YOLO模型

下载YOLOv8模型文件并放置在插件目录下：

- `yolov8n.pt` - 人体检测模型（推荐）
- `yolov8s.pt` - 更准确的模型（可选）

下载地址：https://github.com/ultralytics/assets/releases

### 3. (可选) 下载DNN人脸检测模型

为提高人脸识别准确率，可以下载以下文件：

- `deploy.prototxt.txt` - 模型配置文件
- `res10_300x300_ssd_iter_140000.caffemodel` - 模型权重文件

下载地址：
- https://github.com/opencv/opencv/blob/4.5.2/samples/dnn/face_detector/deploy.prototxt
- https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel

### 4. 创建必要的文件夹

```bash
mkdir known_faces  # 存放已知人脸图片
mkdir snapshots    # 存放警报截图
```

### 5. 配置AstrBot

确保AstrBot已正确配置：
- 设置LLM Provider（支持图片的模型更佳，如glm-4v、gpt-4o等）
- 配置QQ平台适配器（aiocqhttp）

## 🚀 使用方法

### 1. 启动插件

在AstrBot中加载插件，插件会自动启动Web服务器。

### 2. 访问Web界面

打开浏览器访问：`http://localhost:5000`

### 3. 添加人脸数据

**方法一：上传图片**
1. 在Web界面的"人脸数据管理"区域输入姓名
2. 点击"上传图片"选择人脸图片
3. 系统会自动处理并保存

**方法二：拍照添加**
1. 启动摄像头
2. 点击"拍照添加人脸"按钮
3. 在弹出的窗口中输入姓名
4. 点击"确认添加"保存

### 4. 绘制敏感区域

1. 点击"绘制敏感区域"按钮
2. 在视频画面中点击多个点来定义区域（至少3个点）
3. 点击"完成绘制"保存区域
4. 区域会自动显示在界面上

### 5. 监控和警报

- 启动摄像头后，系统会自动进行检测
- 检测到陌生人或区域入侵时会发送QQ警报
- 警报包含文字描述和监控截图
- 当检测到已知人员时，会自动发送解除警报通知

## 📁 项目结构

```
CRAIC/
├── main.py                          # 插件主入口
├── detector.py                      # 检测器实现
│   ├── ObjectDetector               # YOLO人体检测
│   ├── FaceDatabase                 # 人脸数据库管理
│   ├── ZoneManager                  # 敏感区域管理
│   └── AlertManager                 # 警报管理
├── web_server.py                    # Web服务器
│   ├── Flask应用                    # Web界面服务
│   ├── 视频流处理                   # 实时视频流
│   └── 警报发送逻辑                 # LLM集成和QQ通知
├── index.html                       # 前端界面（独立）
├── client.py                        # 监控客户端
├── surveillance_platform_adapter.py # AstrBot平台适配器
├── surveillance_platform_event.py   # 事件处理
├── astrbot_plugin.py                # 插件导出
├── requirements.txt                 # Python依赖
├── metadata.json                    # 插件元数据
├── known_faces/                     # 已知人脸目录
└── snapshots/                       # 警报截图目录
```

## 🔧 API接口

### Web接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 主页面 |
| `/video_feed` | GET | 视频流 |
| `/toggle_camera` | POST | 开关摄像头 |
| `/capture_face` | POST | 拍照添加人脸 |
| `/add_face` | POST | 上传人脸图片 |
| `/add_face_from_camera` | POST | 从摄像头添加人脸 |
| `/save_zone` | POST | 保存敏感区域 |
| `/delete_zone` | POST | 删除敏感区域 |
| `/clear_zones` | POST | 清除所有区域 |
| `/get_zones` | GET | 获取区域列表 |
| `/get_alerts` | GET | 获取警报列表 |
| `/clear_alerts` | POST | 清除警报 |
| `/get_stats` | GET | 获取统计数据 |
| `/update_zone_preview` | POST | 更新区域预览 |

## 💡 使用技巧

### 1. 提高检测准确率
- 使用清晰的人脸图片添加到数据库
- 确保光线充足
- 定期更新人脸数据

### 2. 优化警报体验
- 合理设置敏感区域，避免误报
- 使用支持视觉的LLM模型获得更好的图片描述
- 查看调试日志了解系统运行状态

### 3. 安全建议
- 不要在公开仓库中提交QQ号等隐私信息
- 定期备份已知人脸数据
- 合理设置警报冷却时间，避免频繁打扰

## ❓ 常见问题

### Q: 摄像头无法打开？
A: 检查摄像头是否被其他程序占用，尝试更换摄像头索引。

### Q: 检测准确率低？
A: 确保使用清晰的人脸图片，光线充足，可以尝试使用DNN人脸检测模型。

### Q: 警报没有发送？
A: 检查QQ号配置是否正确，确认AstrBot的LLM Provider已配置。

### Q: 如何添加多个人脸？
A: 在Web界面中重复添加操作，每个人使用不同的姓名。

### Q: 警报太频繁？
A: 调整detector.py中的冷却时间参数（cooldown_seconds）。

## 📝 更新日志

### v1.2.0
- ✨ 分离前端代码为独立HTML文件
- 🐛 修复解除警报逻辑
- ⚡ 优化警报发送机制
- 📝 完善README文档

### v1.1.0
- ✨ 添加解除警报功能
- 🐛 修复QQ消息发送bug
- ⚡ 优化检测性能
- 🎨 改进Web界面

### v1.0.0
- 🎉 初始版本发布
- ✨ 基础人脸识别功能
- ✨ 人体检测功能
- ✨ 敏感区域监控
- ✨ Web界面

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License

## 🙏 致谢

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - 人体检测模型
- [OpenCV](https://opencv.org/) - 计算机视觉库
- [AstrBot](https://github.com/Soulter/AstrBot) - 聊天机器人框架

---

**注意：本项目仅供学习和研究使用，请勿用于非法用途。**
