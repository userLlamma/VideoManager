# 视频素材管理系统

一个基于FastAPI的视频素材管理系统，用于提取、分类和管理视频的关键帧。该系统可以从视频中提取场景变化帧，使用AI进行自动标签分类，并提供API接口进行素材管理。

## 功能特点

- **视频帧提取**：自动从视频中提取关键帧和场景变化
- **AI图像分类**：支持使用OpenAI Vision API或本地Qwen-VL模型对图像进行标签分类
- **灵活的标签系统**：支持标签分类、使用次数统计和标签合并
- **项目管理**：可以将素材组织到不同项目中
- **RESTful API**：完整的API接口，支持素材的CRUD操作
- **批量处理**：支持批量上传和处理视频文件

## 技术栈

- **后端框架**：FastAPI
- **数据库**：SQLAlchemy ORM (支持SQLite等)
- **图像处理**：OpenCV
- **AI图像分类**：
  - 云端选项：阿里云视觉智能API、HuggingFace、Azure Computer Vision
  - 本地选项：Qwen-VL (基于llama.cpp)

## 安装指南

### 前提条件

- Python 3.8+
- pip
- 如果使用本地分类器：llama.cpp (需支持Qwen2-VL)

### 基础安装
1. 安装依赖

```bash
conda env create -f environment.yml
```

2. 配置环境

可以通过以下两种方式配置系统：

#### 方法1：创建`.env`文件或配置环境变量

```
# 数据库配置
DATABASE_URL=sqlite:///./video_materials.db

# 文件路径配置
VIDEO_FOLDER=./app/static/uploads
OUTPUT_FOLDER=./app/static/extracted_frames

# API配置 (如果使用云端API)
API_TYPE=openai  # 可选: openai, aliyun, huggingface, azure
API_KEY=your_api_key_here

# 本地分类器配置
USE_LOCAL_CLASSIFIER=False  # 设为True启用本地分类器
LOCAL_MODEL_PATH=/path/to/Qwen2-VL-2B-Instruct-Q4_K_M.gguf
LOCAL_MODEL_SIZE=2b  # 2b或7b
QWEN_VL_MMPROJ_PATH=/path/to/qwen-qwen2-vl-2b-instruct-vision.gguf
QWEN_VL_CLI_PATH=/path/to/llama-qwen2vl-cli
GPU_LAYERS=0  # 使用GPU的层数，0表示纯CPU
```

#### 方法2：使用config.json文件

创建一个`config.json`文件（可以从`config.example.json`复制并修改）：

```json
{
  "api_type": "openai",
  "api_key": "your_api_key_here",
  "video_folder": "./app/static/uploads",
  "output_folder": "./app/static/extracted_frames",
  "extraction": {
    "min_scene_change_threshold": 30,
    "frame_sample_interval": 24,
    "quality": 90,
    "max_frames_per_video": 200
  },
  "local_classifier": {
    "use_local_classifier": true,
    "model_path": "/path/to/Qwen2-VL-2B-Instruct-Q4_K_M.gguf",
    "model_size": "2b",
    "mmproj_path": "/path/to/qwen-qwen2-vl-2b-instruct-vision.gguf",
    "cli_path": "/path/to/llama-qwen2vl-cli",
    "gpu_layers": 0
  },
  "custom_tags": [
    "人物特写", "风景", "动作", "对话", "过渡", "特效",
    "室内", "室外", "白天", "夜晚", "城市", "自然"
  ]
}
```

系统会优先使用`config.json`中的设置，其次使用环境变量或`.env`文件中的设置。

### 本地分类器安装 (可选)

如果要使用本地Qwen-VL分类器，可以使用提供的安装脚本：

```bash
python app/tools/install_qwen_vl.py --model-size 2b
```

该脚本将：
1. 检查和安装llama.cpp (如有需要)
2. 下载Qwen2-VL GGUF模型文件
3. 自动配置系统设置

### 测试本地分类器 (可选)

可以使用以下命令测试本地分类器是否正常工作：

```bash
python app/tools/test_qwen_vl_classifier.py /path/to/test_image.jpg
```

## 启动应用

```bash
uvicorn app.main:app --reload
```

应用将在 `http://localhost:8000` 启动，API文档可在 `http://localhost:8000/docs` 访问。

## API接口

系统提供以下主要API接口：

### 素材管理
- `GET /api/v1/materials` - 获取素材列表
- `GET /api/v1/materials/{id}` - 获取素材详情
- `GET /api/v1/materials/{id}/image` - 获取素材图像
- `PUT /api/v1/materials/{id}` - 更新素材信息
- `DELETE /api/v1/materials/{id}` - 删除素材
- `POST /api/v1/materials/{id}/tags` - 为素材添加标签
- `DELETE /api/v1/materials/{id}/tags/{tag_id}` - 移除素材标签

### 标签管理
- `GET /api/v1/tags` - 获取标签列表
- `GET /api/v1/tags/categories` - 获取标签类别
- `GET /api/v1/tags/{id}` - 获取标签详情
- `POST /api/v1/tags` - 创建新标签
- `PUT /api/v1/tags/{id}` - 更新标签
- `DELETE /api/v1/tags/{id}` - 删除标签
- `POST /api/v1/tags/merge` - 合并多个标签
- `GET /api/v1/tags/{id}/materials` - 获取使用此标签的素材

### 项目管理
- `GET /api/v1/projects` - 获取项目列表
- `GET /api/v1/projects/{id}` - 获取项目详情
- `POST /api/v1/projects` - 创建新项目
- `PUT /api/v1/projects/{id}` - 更新项目
- `DELETE /api/v1/projects/{id}` - 删除项目
- `GET /api/v1/projects/{id}/materials` - 获取项目中的素材
- `POST /api/v1/projects/{id}/materials` - 将素材添加到项目
- `DELETE /api/v1/projects/{id}/materials/{material_id}` - 从项目中移除素材

### 视频处理
- `POST /api/v1/processing/upload` - 上传并处理视频
- `POST /api/v1/processing/process` - 处理已存在的视频
- `POST /api/v1/processing/batch` - 批量处理文件夹中的视频

## 配置项

系统配置可通过环境变量或`config.json`文件设置，主要配置项包括：

### 视频处理配置

- `MIN_SCENE_CHANGE_THRESHOLD`: 场景变化检测灵敏度 (默认: 30)
- `FRAME_SAMPLE_INTERVAL`: 帧采样间隔 (默认: 24)
- `QUALITY`: 输出图像质量 (默认: 90)
- `MAX_FRAMES_PER_VIDEO`: 每个视频最大提取帧数 (默认: 200)

### 分类器配置

- `USE_LOCAL_CLASSIFIER`: 是否使用本地分类器
- `API_TYPE`: 云API类型 (openai, aliyun, huggingface, azure)
- `API_KEY`: API密钥
- `MAX_WORKERS`: 并行处理线程数 (默认: 4)
- `CUSTOM_TAGS`: 自定义标签列表，用于引导分类器生成特定领域的标签

### 本地分类器专用配置

- `LOCAL_MODEL_PATH`: Qwen-VL模型文件路径
- `LOCAL_MODEL_SIZE`: 模型大小 (2b 或 7b)
- `QWEN_VL_MMPROJ_PATH`: 视觉投影文件路径
- `QWEN_VL_CLI_PATH`: llama-qwen2vl-cli可执行文件路径
- `GPU_LAYERS`: 使用GPU的层数 (0表示纯CPU)

> **注意**: 当前Qwen-VL分类器主要支持英文输出标签，系统会尝试将输出转换为中文，但可能需要额外处理以获得完全的中文支持。

## 自定义标签示例

可以在配置中定义常用标签，用于引导AI分类器。**注意：当使用Qwen2-VL本地分类器时，请使用英文标签，以避免命令行执行错误**：

```json
"CUSTOM_TAGS": [
    "portrait", "landscape", "action", "dialogue", "transition", "effect",
    "indoor", "outdoor", "daytime", "night", "urban", "nature",
    "fast", "slow motion", "emotional", "narrative", "black and white", "color"
]
```

如果使用OpenAI等云端API，可以使用中文标签：

```json
"CUSTOM_TAGS": [
    "人物特写", "风景", "动作", "对话", "过渡", "特效",
    "室内", "室外", "白天", "夜晚", "城市", "自然",
    "快速", "慢动作", "情感", "叙事", "黑白", "彩色"
]
```

## 性能优化

### 本地分类器优化

1. 使用更小的模型 (2b) 可提高处理速度
2. 适当设置GPU_LAYERS参数能显著提升性能
3. 合理设置MAX_WORKERS参数，建议为CPU核心数的一半
4. 第一次使用本地分类器时会较慢，因为需要加载模型，之后会快得多

### 视频处理优化

1. 调整FRAME_SAMPLE_INTERVAL可减少处理时间
2. 增大MIN_SCENE_CHANGE_THRESHOLD可减少提取的帧数
3. 设置MAX_FRAMES_PER_VIDEO限制单个视频的处理量

## 已知问题与待办事项

### 已知问题
1. Qwen-VL分类器目前主要支持英文输出标签。虽然系统会尝试解析并转换这些标签，但可能需要额外的处理步骤来获得更好的中文支持。
2. 使用本地分类器时，首次处理可能较慢，这是因为需要加载大型模型文件。
3. 在Windows系统上，使用本地分类器可能需要额外配置路径格式。
4. 使用Qwen2-VL时自定义标签必须使用英文，否则可能导致命令行执行错误。

### 待办事项
1. **云端API集成完善**：目前多种云端API（阿里云, HuggingFace, Azure）的集成尚在调试中。
2. **混合分类处理流程**：计划实现本地与云端分级处理机制，敏感素材优先使用本地分类器处理以保护数据隐私，非敏感素材可选择使用云端API获得更高精度的分类结果。
3. **中文支持改进**：优化Qwen2-VL模型的中文输出支持，或添加更好的翻译处理机制。

## 许可证

MIT

