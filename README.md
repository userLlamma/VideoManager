# 视频素材管理系统

一个基于FastAPI的视频素材管理系统，用于从视频中提取关键帧、自动分类并提供检索功能。适用于影视制作团队对素材进行分类和管理。

## 功能特点

- **视频关键帧提取**：自动检测场景变化，提取有意义的帧
- **智能分类标签**：使用云端AI对图像进行分类和标签生成
- **项目管理**：将素材组织到不同项目中
- **标签管理**：创建、合并、删除标签
- **高效搜索**：通过标签、描述或视频源搜索素材
- **支持多种AI**：OpenAI Vision、Azure Vision、Aliyun Vision、HuggingFace

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