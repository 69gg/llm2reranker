import os
import json
import time
import uuid
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

from openai import AsyncOpenAI

from dotenv import load_dotenv

# -----------------------------
# 配置
# -----------------------------
load_dotenv()  # 从 .env 文件加载环境变量

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
INTERNAL_LLM_MODEL = os.getenv("INTERNAL_LLM_MODEL", "gpt-4.1-mini")

client = AsyncOpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL,
)


# -----------------------------
# 入参/出参（兼容 rerank 常见格式）
# -----------------------------
class RerankRequest(BaseModel):
    model_config = ConfigDict(extra="allow")  # 允许透传额外字段（兼容性更强）

    model: str = Field(..., description="外部请求的reranker模型名（仅记录/兼容字段）")
    query: str
    documents: List[Union[str, Dict[str, Any]]]  # 兼容: ["text", ...] 或 [{"text": "..."}]
    top_n: int = 10
    return_documents: bool = True


class RerankResultDocument(BaseModel):
    text: str


class RerankResult(BaseModel):
    # 关键：return_documents=false 时必须完全不出现 document 字段
    document: Optional[RerankResultDocument] = None
    index: int
    relevance_score: float

    model_config = ConfigDict(
        extra="forbid",
        ser_json_exclude_none=True,  # ✅ None 字段不序列化 => document 字段彻底消失
    ) # type: ignore


class RerankMetaBilledUnits(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    search_units: int = 0
    classifications: int = 0


class RerankMetaTokens(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class RerankMeta(BaseModel):
    billed_units: RerankMetaBilledUnits
    tokens: RerankMetaTokens


class RerankResponse(BaseModel):
    id: str
    results: List[RerankResult]
    meta: RerankMeta


# -----------------------------
# LLM 输出 JSON Schema（只允许 index + relevance_score）
# -----------------------------
RERANK_JSON_SCHEMA: Dict[str, Any] = {
    "name": "rerank_output",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "results": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "index": {"type": "integer", "minimum": 0},
                        "relevance_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    },
                    "required": ["index", "relevance_score"],
                },
            }
        },
        "required": ["results"],
    },
}


# -----------------------------
# 工具函数：抽取 documents 文本
# -----------------------------
def normalize_documents(docs: List[Union[str, Dict[str, Any]]]) -> List[str]:
    out: List[str] = []
    for d in docs:
        if isinstance(d, str):
            out.append(d)
        elif isinstance(d, dict):
            if "text" in d and isinstance(d["text"], str):
                out.append(d["text"])
            elif "content" in d and isinstance(d["content"], str):
                out.append(d["content"])
            else:
                out.append(json.dumps(d, ensure_ascii=False))
        else:
            out.append(str(d))
    return out


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def enforce_unique_and_bounds(
    results: List[Dict[str, Any]],
    doc_count: int,
    top_n: int,
) -> List[Dict[str, Any]]:
    """确保 index 合法、去重、并截断到 top_n。"""
    seen = set()
    cleaned: List[Dict[str, Any]] = []

    for r in results:
        idx = int(r["index"])
        if idx < 0 or idx >= doc_count:
            continue
        if idx in seen:
            continue
        seen.add(idx)

        score = float(r.get("relevance_score", 0.0))
        cleaned.append({"index": idx, "relevance_score": clamp01(score)})

    cleaned.sort(key=lambda x: x["relevance_score"], reverse=True)

    # 不足 top_n 自动补齐（防止上游依赖 top_n 个结果）
    if len(cleaned) < min(top_n, doc_count):
        missing = [i for i in range(doc_count) if i not in seen]
        base = 0.05
        step = 0.01
        for j, idx in enumerate(missing):
            cleaned.append({"index": idx, "relevance_score": clamp01(base - j * step)})
            if len(cleaned) >= min(top_n, doc_count):
                break

    cleaned.sort(key=lambda x: x["relevance_score"], reverse=True)
    return cleaned[: min(top_n, doc_count)]


# -----------------------------
# 核心：调用 LLM 做 rerank（全异步）
# -----------------------------
async def llm_rerank(query: str, docs: List[str], top_n: int) -> Dict[str, Any]:
    numbered_docs = "\n".join([f"[{i}] {t}" for i, t in enumerate(docs)])

    system = (
        "你是一个高精度的检索重排序器（reranker）。"
        "你必须只输出符合给定 JSON Schema 的JSON，不要输出多余文字。"
        "只返回每个结果的 index 和 relevance_score（0~1），不要返回文档文本。"
        "relevance_score 越大越相关。index 不要重复。"
        "results 的数量尽量等于 top_n。"
    )

    user = (
        f"Query:\n{query}\n\n"
        f"Documents:\n{numbered_docs}\n\n"
        f"Task:\n从 Documents 中选出最相关的 top_n={top_n} 个，"
        f"给出它们的 index 和 relevance_score（0~1）。"
    )

    resp = await client.chat.completions.create(
        model=INTERNAL_LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": RERANK_JSON_SCHEMA,
        }, # type: ignore
    )

    content = resp.choices[0].message.content or "{}"
    data = json.loads(content)

    usage = getattr(resp, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

    ranked_raw = data.get("results", [])
    ranked = enforce_unique_and_bounds(ranked_raw, doc_count=len(docs), top_n=top_n)

    return {
        "ranked": ranked,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        },
    }


# -----------------------------
# FastAPI：对外 rerank endpoint
# -----------------------------
app = FastAPI(title="llm2reranker", version="0.2.0")


@app.post(
    "/v1/rerank",
    response_model=RerankResponse,
    response_model_exclude_none=True,  # ✅ 确保 document=None 时字段不出现（完全兼容 return_documents=false 格式）
)
async def rerank(req: RerankRequest):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")

    docs_text = normalize_documents(req.documents)
    if not docs_text:
        raise HTTPException(status_code=400, detail="documents is empty")

    top_n = req.top_n if req.top_n and req.top_n > 0 else 10

    _t0 = time.time()
    rr = await llm_rerank(req.query, docs_text, top_n=top_n)
    _elapsed = time.time() - _t0  # 不用就留着，方便你以后打日志

    ranked: List[Dict[str, Any]] = rr["ranked"]
    prompt_tokens = int(rr["usage"].get("prompt_tokens", 0))
    completion_tokens = int(rr["usage"].get("completion_tokens", 0))

    results: List[RerankResult] = []
    for item in ranked:
        idx = int(item["index"])
        score = float(item["relevance_score"])

        if req.return_documents:
            results.append(
                RerankResult(
                    document=RerankResultDocument(text=docs_text[idx]),
                    index=idx,
                    relevance_score=score,
                )
            )
        else:
            # ✅ document 字段完全不出现（靠 response_model_exclude_none + ser_json_exclude_none）
            results.append(
                RerankResult(
                    document=None,
                    index=idx,
                    relevance_score=score,
                )
            )

    meta = RerankMeta(
        billed_units=RerankMetaBilledUnits(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            search_units=0,
            classifications=0,
        ),
        tokens=RerankMetaTokens(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
        ),
    )

    return RerankResponse(
        id=uuid.uuid4().hex,
        results=results,
        meta=meta,
    )


@app.get("/healthz")
async def healthz():
    return {"ok": True}