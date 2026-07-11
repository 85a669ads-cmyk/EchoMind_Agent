# 🧠 EchoMind 2.0 — 仿生认知记忆智能体

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](Dockerfile)

> **EchoMind 2.0** 不是一个简单的聊天机器人，而是一个**具备仿生认知架构的持久记忆智能体**。
> 它超越了基础的 RAG（检索增强生成），引入了**记忆分层、动态巩固、冲突解决与可视化反思**机制，旨在解决长周期交互中的"记忆幻觉"、"上下文遗忘"和"信息过载"问题。

---

## 🏗️ 仿生记忆架构 (Bionic Memory Architecture)

```text
[用户交互层 (Chainlit / REST API)]
        │
        ▼
[API 网关层 (FastAPI + WebSocket)]
        │
        ├──> [Agent 核心引擎 (ReAct / Plan-and-Solve)]
        │         │
        │         ├──> [短期记忆/工作记忆 (In-Memory + Redis)] -> 当前多轮对话
        │         │
        │         ├──> [长期记忆系统 (Memory Core)]
        │         │         ├── 情景记忆: 向量数据库 (DashVector) + 时间衰减
        │         │         ├── 语义记忆: 知识图谱/关系型数据库
        │         │         └── 程序性记忆: 工具调用记录 (Tool Logs)
        │         │
        │         ──> [记忆管理后台 (Memory Manager)]
        │                   ├── 记忆巩固器 (Consolidator)
        │                   ├── 冲突解决器 (Conflict Resolver)
        │                   └── 遗忘引擎 (Forgetting Engine) — 艾宾浩斯曲线
        │
        └──> [外部工具层 (Tools)] -> 搜索、时间、日历
```

### 核心技术栈

| 层级 | 技术 | 用途 |
|------|------|------|
| 后端框架 | **FastAPI** | 异步、高性能、自动 API 文档 |
| 前端交互 | **Chainlit** | 专为 LLM 设计，原生支持思考过程展示 |
| 大模型 | **Qwen-Max** + **Qwen-Plus** | 复杂推理 + 记忆提取/总结 |
| 向量存储 | **DashVector** | 语义检索（可回退本地模式） |
| 缓存 | **Redis** / In-Memory | 短期记忆 + 会话管理 |
| 工程化 | **Docker + Poetry** | 环境一致性与依赖管理 |

---

## 🚀 快速开始

### 前置要求

- Python 3.10+
- [阿里云 DashScope API Key](https://dashscope.console.aliyun.com/)

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd echomind
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 DASHSCOPE_API_KEY
```

### 3. 安装依赖

```bash
# 使用 Poetry
poetry install

# 或使用 pip
pip install fastapi uvicorn dashscope pydantic python-dotenv chainlit
```

### 4. 启动服务

```bash
# FastAPI 后端
python app.py

# 或使用 uvicorn
uvicorn app:create_app --host 0.0.0.0 --port 8000 --factory

# Chainlit 前端（可选）
chainlit run frontend/chainlit_app.py
```

### 5. 访问

- **API 文档**: http://localhost:8000/docs
- **健康检查**: http://localhost:8000/api/v1/health
- **Chainlit UI**: http://localhost:8000 (如使用 Chainlit)

### 6. Docker 部署

```bash
docker-compose up -d
```

---

## 📡 API 接口

| 方法 | 路径 | 描述 |
|------|------|------|
| `GET` | `/` | 服务信息 |
| `GET` | `/api/v1/health` | 健康检查（含阿里云服务状态） |
| `POST` | `/api/v1/chat` | 非流式对话 |
| `POST` | `/api/v1/chat/stream` | 流式对话 (SSE) |
| `GET` | `/api/v1/memory/stats` | 记忆统计仪表盘 |
| `POST` | `/api/v1/memory/consolidate` | 手动触发记忆巩固 |
| `POST` | `/api/v1/memory/purge` | 手动触发遗忘清理 |

### 对话示例

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user_001",
    "message": "我喜欢喝冰美式咖啡",
    "stream": false
  }'
```

返回包含 ReAct 思考链和记忆检索结果：

```json
{
  "user_id": "user_001",
  "reply": "已记住！你喜欢喝冰美式咖啡 ☕",
  "thought_chain": [
    {
      "thought": "用户表达了一个偏好...",
      "step_index": 1
    }
  ],
  "retrieved_memories": [
    {"content": "用户偏好：喜欢喝冰美式", "score": 0.92}
  ]
}
```

---

## 🧠 核心创新点

### 1. 记忆分层系统

- **短期工作记忆** (Working Memory): 存储最近 N 轮对话，TTL 30分钟
- **长期记忆** (Long-Term Memory): DashVector 向量语义检索，持久化存储

### 2. 记忆巩固机制 (Consolidation)

当短期记忆达到阈值（默认10条），自动调用 Qwen-Plus 将零散对话总结为结构化长期记忆：

> "用户偏好：喜欢喝冰美式咖啡，对花生过敏，工作是软件工程师"

### 3. 冲突解决 (Conflict Resolution)

当新旧记忆相似度 > 0.85 时，自动调用 LLM 判断冲突并解决：

- `旧记忆: "我喜欢苹果"` → `新记忆: "其实我讨厌苹果"` → **解决: 替换旧记忆**

### 4. 艾宾浩斯遗忘曲线

```
Score = Importance × (1 / (1 + k × time_diff)) × log(1 + access_count)
```

- 高重要性 + 高频访问 → 不易遗忘
- 低价值旧记忆 → 自动清理

### 5. ReAct 范式思考

Agent 在回答前输出完整思考链：

```
Thought: 用户询问餐厅推荐，我需要先检索他的饮食偏好
Action: retrieve_memory("餐厅", "偏好")
Observation: 用户对花生过敏，喜欢日料
Final Answer: 推荐XX日料店，已避开花生类菜品
```

---

## 📁 项目结构

```
echomind/
├── app/
│   ├── api/              # FastAPI 路由
│   │   └── routes.py     # 对话、健康检查、记忆仪表盘接口
│   ├── core/             # Agent 核心逻辑
│   │   ├── config.py     # 配置管理
│   │   ├── llm_client.py # DashScope LLM 客户端
│   │   └── agent_engine.py # ReAct 智能体引擎
│   ├── memory/           # 记忆管理器
│   │   ├── working_memory.py    # 短期工作记忆
│   │   ├── long_term_memory.py  # 长期记忆（DashVector）
│   │   ├── consolidator.py      # 记忆巩固器
│   │   ├── conflict_resolver.py # 冲突解决器
│   │   └── forgetting_engine.py # 遗忘引擎
│   ├── models/           # Pydantic 数据模型
│   │   ├── memory_models.py
│   │   ├── chat_models.py
│   │   └── user_models.py
│   └── tools/            # 外部工具
│       ├── base_tools.py
│       ├── search_tool.py
│       └── time_tool.py
├── frontend/
│   └── chainlit_app.py   # Chainlit 前端界面
├── tests/                # 单元测试
├── app.py                # 应用入口
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

---

## 🧪 测试

```bash
# 运行所有测试
pytest tests/ -v

# 测试覆盖率
pytest tests/ --cov=app --cov-report=html
```

---

## 📄 许可证

本项目采用 [MIT License](LICENSE)。

---

## 🙏 致谢

- [阿里云 DashScope](https://dashscope.aliyun.com/) — 通义千问大模型服务
- [阿里云 DashVector](https://help.aliyun.com/product/2510317.html) — 向量检索服务
- [Chainlit](https://chainlit.io/) — LLM 前端框架
- [FastAPI](https://fastapi.tiangolo.com/) — 现代 Python Web 框架