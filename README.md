# llm2reranker

用 LLM 实现的 Reranker 服务，接口兼容 Cohere Rerank API 格式，底层调用任意 OpenAI 兼容的 LLM。

## 原理

将 query 和 documents 发给 LLM，要求其以 JSON Schema 约束的格式输出每个文档的相关性分数（0~1），再按分数降序返回。

## 快速开始

### 1. 配置环境变量

复制 `.env.example` 为 `.env` 并填写：

```env
OPENAI_API_KEY="sk-xxx"
OPENAI_BASE_URL="https://api.openai.com/v1"
INTERNAL_LLM_MODEL="gpt-4.1-mini"
```

| 变量 | 说明 |
|------|------|
| `OPENAI_API_KEY` | LLM 服务的 API Key |
| `OPENAI_BASE_URL` | 兼容 OpenAI 格式的 API 地址（可替换为其他服务） |
| `INTERNAL_LLM_MODEL` | 实际用于排序的模型名 |

### 2. 安装依赖并启动

```bash
uv sync
uvicorn app:app --host 0.0.0.0 --port 8000
```

## API

### POST /v1/rerank

请求体：

```json
{
  "model": "bge-reranker-v2-m3",
  "query": "什么是机器学习？",
  "documents": ["机器学习是...", "深度学习是..."],
  "top_n": 3,
  "return_documents": true
}
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | string | 必填 | 仅回显用，不影响实际使用的 LLM |
| `query` | string | 必填 | 查询文本 |
| `documents` | array | 必填 | 字符串数组或 `[{"text": "..."}]` 格式 |
| `top_n` | int | 10 | 返回最相关的前 N 个 |
| `return_documents` | bool | true | 是否在结果中返回原文 |

响应体（兼容 Cohere Rerank 格式）：

```json
{
  "id": "abc123",
  "results": [
    {
      "index": 0,
      "relevance_score": 0.95,
      "document": {"text": "机器学习是..."}
    }
  ],
  "meta": {
    "billed_units": {"input_tokens": 100, "output_tokens": 50, "search_units": 0, "classifications": 0},
    "tokens": {"input_tokens": 100, "output_tokens": 50}
  }
}
```

### GET /healthz

健康检查，返回 `{"ok": true}`。

## 容错

LLM 调用失败时自动降级：按文档原始顺序返回，分数从 0.2 开始递减。
