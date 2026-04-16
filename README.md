# RAGWise（智询库）——企业级智能文档知识库系统

## 项目简介

RAGWise 是一个企业级智能文档知识库系统，旨在解决海量文档查询效率低、人工检索慢、传统搜索准确率差的痛点。通过全链路 RAG 技术，实现对企业内部知识的精准、可靠提取，让用户一句自然语言就能快速得到带来源引用的准确答案。

## 核心功能

- **多格式文档支持**：PDF、Word、Markdown 等格式批量上传与智能解析
- **智能分块**：Recursive + SemanticChunker 双策略
- **Hybrid 检索**：向量 + BM25 多路召回
- **Rerank 重排**：BGE-Reranker 提升检索准确率
- **幻觉抑制**：Self-Check / Fact-Verify 机制
- **来源引用**：回答附带页码/段落来源
- **FastAPI 后端**：异步接口、流式输出、API Key 鉴权
- **Gradio 前端**：文件批量上传、聊天界面、实时进度
- **Docker 部署**：完整的容器化方案
- **RAGAS 评估**：量化评估 RAG 性能

## 技术栈

- **LLM**：通义千问（Qwen-Max / Qwen-Turbo）或 OpenAI
- **Embedding**：text-embedding-v3（Qwen）或 OpenAI text-embedding-3-large
- **向量库**：Chroma（本地持久化）
- **框架**：LangChain/LCEL、FastAPI、Gradio
- **评估工具**：RAGAS + LangSmith

## 系统架构

```
本项目采用模块化分层架构，整体分为前端交互、后端服务、核心 RAG 逻辑、数据存储与评估模块五部分。

用户通过Gradio 前端进行文件上传、对话交互并查看结果；前端请求交由FastAPI 后端处理，负责接口路由、权限验证与 RAG 服务调度；核心 RAG 模块完成文档加载、文本分块、向量化、检索重排与 LLM 推理生成，并依托Chroma 向量库实现向量存储和相似度检索；同时系统内置评估模块，可通过 RAGAS 对回答效果与检索性能做自动化评估与分析，各模块协同完成完整的检索增强问答流程。
```

## 项目结构

```
RAGWise/
├── backend/                  # FastAPI
│   ├── app/
│   │   ├── main.py
│   │   ├── services/         # rag_service.py
│   │   └── schemas/
│   └── Dockerfile
├── frontend/                 # Gradio
│   ├── app.py
│   └── Dockerfile
├── rag/                      # 核心逻辑
│   ├── __init__.py
│   ├── chain.py              # LCEL Chain
│   ├── embeddings.py
│   ├── loader.py
│   ├── reranker.py
│   ├── retriever.py
│   ├── splitter.py
│   └── vectorstore.py
├── chroma_db/                # 向量数据库
├── data/                     # 测试文档
├── evaluations/              # RAGAS 测试集 + 报告
├── tests/
├── .env                      # 环境变量配置
├── docker-compose.yml
├── pyproject.toml
├── railway.json              # Railway部署配置
├── README.md
└── architecture.drawio       # 架构图
```

## 快速开始

### 1. 安装依赖

```bash
# 使用Poetry
poetry install

# 或使用pip
pip install -r requirements.txt
```

### 2. 配置环境变量

创建 `.env` 文件，配置以下环境变量：

```
# LLM配置
OPENAI_API_KEY=your_api_key
# 或使用通义千问
QWEN_API_KEY=your_api_key

# 向量库配置
CHROMA_PERSIST_DIRECTORY=./chroma_db

# API Key鉴权
API_KEY=your_secret_key
```

### 3. 启动服务

```bash
# 启动后端服务
cd backend
uvicorn app.main:app --reload

# 启动前端服务
cd frontend
python app.py
```

### 4. 访问系统

- 前端：<http://localhost:7860>
- 后端 API 文档：<http://localhost:8000/docs>

## 评估与优化

1. 准备 30 条 QA 测试集
2. 运行 RAGAS 评估：
   ```bash
   python evaluations/run_ragas.py
   ```
3. 查看 LangSmith 追踪：<https://smith.langchain.com>

## 部署

### 1. Docker Compose 部署（本地/服务器）

```bash
# 构建并启动服务
docker-compose up --build

# 后台运行
docker-compose up -d

# 停止服务
docker-compose down
```

访问地址：

- 前端：<http://localhost:7860>
- 后端 API 文档：<http://localhost:8000/docs>

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License

Copyright (c) 2026 **yyyJ8**&#x20;

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:&#x20;

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.&#x20;

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
