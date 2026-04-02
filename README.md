# 智能监控插件 (CRAIC)

AstrBot智能监控系统插件，支持人脸识别、人体检测、敏感区域监控。

## 功能特性

- **人脸识别**：识别已知人员和陌生人
- **人体检测**：使用YOLO模型检测人体
- **敏感区域监控**：定义敏感区域，监控人员进入
- **实时警报**：通过QQ发送警报通知
- **LLM集成**：使用大模型生成详细警报描述
- **Web界面**：提供Web界面进行监控和管理

## 安装依赖

```bash
pip install -r requirements.txt
```

## 配置说明

### 1. 修改QQ号码

在以下文件中修改 `YOUR_QQ_NUMBER` 为你的QQ号码：

- `web_server.py` (第53行)
- `client.py` (第17行)
- `surveillance_platform_event.py` (第33行)

### 2. 下载YOLO模型

下载YOLO模型文件并放置在插件目录下：

- `yolov8n.pt` - 人体检测模型

### 3. 创建必要的文件夹

```bash
mkdir known_faces
mkdir snapshots
```

## 使用方法

### 1. 启动插件

在AstrBot中加载插件，插件会自动启动Web服务器。

### 2. 访问Web界面

打开浏览器访问：`http://localhost:5000`

### 3. 添加人脸数据

- 在Web界面中输入姓名并上传图片
- 或使用"拍照添加人脸"功能截取当前画面

### 4. 绘制敏感区域

1. 点击"绘制敏感区域"
2. 在视频画面中点击多个点定义区域
3. 点击"完成绘制"保存区域

## 文件说明

- `main.py` - 插件主入口
- `detector.py` - 检测器实现（人脸识别、人体检测、区域管理）
- `web_server.py` - Web服务器
- `client.py` - 监控客户端
- `surveillance_platform_adapter.py` - AstrBot平台适配器
- `surveillance_platform_event.py` - 事件处理
- `astrbot_plugin.py` - 插件导出

## 注意事项

1. 确保已安装所有依赖
2. 确保YOLO模型文件存在
3. 确保已配置正确的QQ号码
4. 确保AstrBot已正确配置LLM Provider

## 许可证

MIT License
