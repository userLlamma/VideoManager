# 视频素材管理系统

一个基于FastAPI的视频素材管理系统，用于从视频中提取关键帧、自动分类并提供检索功能。适用于影视制作团队对素材进行分类和管理。

## 功能特点

- **视频关键帧提取**：自动检测场景变化，提取有意义的帧
- **智能分类标签**：使用云端AI对图像进行分类和标签生成
- **本地AI支持**：支持使用Qwen2.5-VL模型在本地进行图像分类，无需联网
- **项目管理**：将素材组织到不同项目中
- **标签管理**：创建、合并、删除标签
- **高效搜索**：通过标签、描述或视频源搜索素材
- **支持多种AI**：OpenAI Vision、Azure Vision、Aliyun Vision、HuggingFace、以及本地Qwen2.5-VL模型

## 系统要求

- Python 3.8+
- OpenCV
- FastAPI
- SQLAlchemy
- 足够的存储空间用于视频和图像文件
- (可选)AI API密钥

## 快速开始

### 使用Docker

```bash
# 克隆仓库
git clone https://github.com/yourusername/video-material-system.git
cd video-material-system

# 设置API密钥
echo "OPENAI_API_KEY=your_api_key_here" > .env

# 启动服务
docker-compose up -d
```

### 手动安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/video-material-system.git
cd video-material-system

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # 在Windows上使用 venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 创建必要的目录
mkdir -p app/static/uploads app/static/extracted_frames

# 设置环境变量
export API_TYPE=openai
export API_KEY=your_api_key_here

# 启动服务
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 前端运行

```bash
# 进入前端目录
cd frontend

# 安装依赖
npm install

# 启动前端服务
npm start
```

或者使用自动安装脚本:

```bash
sudo ./setup.sh
```

## API文档

启动服务后，API文档可在以下URL访问：

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 配置

系统可以通过环境变量或`.env`文件进行配置，主要配置项包括：

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| API_TYPE | AI API类型(openai/azure/aliyun/huggingface) | openai |
| API_KEY | API密钥 | - |
| MAX_WORKERS | 并行处理的最大线程数 | 4 |
| MIN_SCENE_CHANGE_THRESHOLD | 场景变化检测阈值(越低越敏感) | 30 |
| FRAME_SAMPLE_INTERVAL | 帧采样间隔 | 24 |
| QUALITY | JPEG保存质量(1-100) | 90 |
| MAX_FRAMES_PER_VIDEO | 每个视频的最大提取帧数 | 200 |
| USE_LOCAL_CLASSIFIER | 是否使用本地图像分类器 | False |
| LOCAL_MODEL_PATH | 本地模型路径 | None |
| LOCAL_MODEL_SIZE | 本地模型大小(1.5b/7b/72b) | 7b |
| LOCAL_TAGS_OUTPUT_FILE | 本地标签输出文件 | local_image_tags.json |
| GPU_LAYERS | GPU加速层数(0表示CPU模式) | 0 |

以下是应该添加到 README.md 中关于 GPU 本地支持的内容：


## 本地图像分类器

系统支持使用本地Qwen2.5-VL多模态模型进行图像分类，无需依赖云端API。

### 安装本地模型

```bash
# 安装依赖 (CPU版本)
pip install llama-cpp-python pillow numpy tqdm requests

# 下载并设置模型
python app/tools/setup_local_model.py --size 7b --quant Q4_K_M
```

#### GPU加速支持

如果您有兼容的NVIDIA GPU，可以安装GPU加速版本以显著提高处理速度：

```bash
# 首先卸载CPU版本(如果已安装)
pip uninstall -y llama-cpp-python

# 安装CUDA 11.8兼容版本
pip install llama-cpp-python==0.2.25+cu118 -f https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels/releases/

# 或安装CUDA 12.1兼容版本
pip install llama-cpp-python==0.2.25+cu121 -f https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels/releases/
```

确保您已安装匹配的CUDA驱动。安装GPU版本后，您需要在配置中设置`GPU_LAYERS`大于0才能启用GPU加速。

### 配置本地分类器

编辑`.env`文件或设置环境变量:

```
USE_LOCAL_CLASSIFIER=True
LOCAL_MODEL_PATH=/path/to/models/qwen2_5-vl-7b.Q4_K_M.gguf
LOCAL_MODEL_SIZE=7b
GPU_LAYERS=0  # 使用CPU模式
# GPU_LAYERS=35  # 对于7b模型启用GPU加速(使用约35层)
# GPU_LAYERS=80  # 对于72b模型启用GPU加速(使用更多层)
```

也可以在`config.json`中配置:

```json
{
  "local_classifier": {
    "use_local_classifier": true,
    "model_path": "/path/to/models/qwen2_5-vl-7b.Q4_K_M.gguf",
    "model_size": "7b",
    "gpu_layers": 35  // 为GPU加速设置合适的层数
  }
}
```

### 本地模型选择指南

- **Qwen2.5-VL-1.5B**: 适合低配置设备(2-3GB内存)，速度较快但精度有限
- **Qwen2.5-VL-7B**: 推荐配置(8-10GB内存，CPU模式)，平衡速度和精度
  - GPU模式: 需要约4GB GPU内存，处理速度提升5-10倍
- **Qwen2.5-VL-72B**: 高精度但需要强大硬件
  - CPU模式: 需要40GB+系统内存
  - GPU模式: 需要至少12GB GPU内存，推荐24GB+

默认使用Q4_K_M量化版本，对于大多数用例提供了良好的平衡。

### GPU内存需求与性能指南

|   模型大小   | GPU内存(Q4_K_M) | 推理速度 | 精度 |
|------------|---------------|---------|-----|
| 1.5B       | ~2GB          | 快       | 一般 |
| 7B         | ~4GB          | 中       | 良好 |
| 72B        | ~12-20GB      | 慢       | 优秀 |

**注意**: GPU加速可以将处理时间从几秒/张图像减少到不到1秒/张图像，特别适合批量处理大量视频帧。


## 使用示例

### 1. 上传和处理视频

```bash
curl -X POST "http://localhost:8000/api/v1/processing/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_video.mp4" \
  -F "extract_only=false"
```

### 2. 搜索素材

```bash
curl -X GET "http://localhost:8000/api/v1/materials?search=人物&tag_ids=1,5,8&skip=0&limit=20" \
  -H "accept: application/json"
```

### 3. 创建项目

```bash
curl -X POST "http://localhost:8000/api/v1/projects" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d '{"name":"新项目","description":"项目描述..."}'
```

## 结构说明

```
video-material-system/
├── app/                      # 主应用代码
│   ├── api/                  # API端点
│   ├── core/                 # 核心功能模块
│   ├── db/                   # 数据库模型和操作
│   ├── schemas/              # Pydantic模式
│   ├── static/               # 静态文件(上传和提取的帧)
│   ├── config.py             # 配置处理
│   └── main.py               # 主应用入口
├── Dockerfile                # Docker构建文件
├── docker-compose.yml        # Docker Compose配置
├── frontend/                 # 前端代码
│   ├── front.js              # 前端主文件
│   └── package.json          # 前端依赖
├── requirements.txt          # Python依赖
├── setup.sh                  # 安装脚本
└── README.md                 # 项目说明
```

## 接口说明

系统提供以下主要API:

### 素材管理
- GET /api/v1/materials - 获取素材列表
- GET /api/v1/materials/{id} - 获取素材详情
- GET /api/v1/materials/{id}/image - 获取素材图像
- PUT /api/v1/materials/{id} - 更新素材信息
- DELETE /api/v1/materials/{id} - 删除素材

### 标签管理
- GET /api/v1/tags - 获取标签列表
- GET /api/v1/tags/categories - 获取标签类别
- POST /api/v1/tags - 创建新标签
- PUT /api/v1/tags/{id} - 更新标签
- DELETE /api/v1/tags/{id} - 删除标签
- POST /api/v1/tags/merge - 合并标签

### 项目管理
- GET /api/v1/projects - 获取项目列表
- GET /api/v1/projects/{id} - 获取项目详情
- POST /api/v1/projects - 创建新项目
- PUT /api/v1/projects/{id} - 更新项目
- DELETE /api/v1/projects/{id} - 删除项目
- GET /api/v1/projects/{id}/materials - 获取项目素材
- POST /api/v1/projects/{id}/materials - 添加素材到项目

### 处理
- POST /api/v1/processing/upload - 上传并处理视频
- POST /api/v1/processing/process - 处理已存在的视频
- POST /api/v1/processing/batch - 批量处理视频

## 许可证

MIT