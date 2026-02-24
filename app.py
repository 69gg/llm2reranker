import os
import json
import time
import uuid
import re
import logging
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

from openai import AsyncOpenAI

from dotenv import load_dotenv

# -----------------------------
# 配置
# -----------------------------

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(), format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
INTERNAL_LLM_MODEL = os.getenv("INTERNAL_LLM_MODEL", "gpt-4.1-mini")
THINKING = os.getenv("THINKING", "false").lower() == "true"

client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


# -----------------------------
# 入参/出参
# -----------------------------
class RerankRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    query: str
    documents: List[Union[str, Dict[str, Any]]]
    top_n: int = 10
    return_documents: bool = True


class RerankResultDocument(BaseModel):
    text: str


class RerankResult(BaseModel):
    document: Optional[RerankResultDocument] = None
    index: int
    relevance_score: float

    model_config = ConfigDict(
        extra="forbid",
        ser_json_exclude_none=True,  # ✅ None 不序列化 => return_documents=false 时不出现 document
    )


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
# utils
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
    return cleaned[: min(top_n, doc_count)]


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)


def parse_results_from_message_content(content: str) -> List[Dict[str, Any]]:
    """
    适配这种：
      ```json
      {"results":[...]}
      ```
    或者直接就是 {"results":[...]}。
    """
    if content is None:
        raise RuntimeError("message.content is None")

    s = content.strip()
    m = _JSON_FENCE_RE.search(s)
    if m:
        s = m.group(1).strip()

    data = json.loads(s)
    if not isinstance(data, dict) or "results" not in data:
        raise RuntimeError(f"content JSON missing 'results': {data!r}")

    results = data["results"]
    if not isinstance(results, list):
        raise RuntimeError("content JSON 'results' is not a list")

    return results


# -----------------------------
# Tool schema（模型支持就走 tool_calls；不支持也没关系）
# -----------------------------
RERANK_TOOL = {
    "type": "function",
    "function": {
        "name": "rerank",
        "description": "Return reranking results as indices and relevance scores (0~1). Do NOT return document text.",
        "parameters": {
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
    },
}


# -----------------------------
# LLM rerank
# -----------------------------
async def llm_rerank(query: str, docs: List[str], top_n: int) -> Dict[str, Any]:
    numbered_docs = "\n".join([f"[{i}] {t}" for i, t in enumerate(docs)])

    system = (
        "你是一个高精度的检索重排序器（reranker）。"
        "优先通过工具调用 rerank() 返回结果。"
        "如果无法进行工具调用，则直接输出一个 JSON（不要输出多余文字），格式必须是："
        '{"results":[{"index":0,"relevance_score":0.9}, ...]}。'
        "不要返回文档文本。"
        "relevance_score 取值 0~1，越大越相关。index 不要重复。"
        "results 数量尽量等于 top_n（不足则全部）。"
    )

    user = (
        f"Query:\n{query}\n\n"
        f"Documents:\n{numbered_docs}\n\n"
        f"Task:\n从 Documents 中选出最相关的 top_n={top_n} 个。"
        f"只返回它们的 index 和 relevance_score。"
    )

    resp = await client.chat.completions.create(
        model=INTERNAL_LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
        tools=[RERANK_TOOL],
        tool_choice={"type": "function", "function": {"name": "rerank"}},
        extra_body={"thinking": THINKING},
    )

    usage = getattr(resp, "usage", None)
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

    msg = resp.choices[0].message

    # 1) 优先走 tool_calls
    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        tc0 = tool_calls[0]
        fn = getattr(tc0, "function", None)
        if not fn or getattr(fn, "name", "") != "rerank":
            raise RuntimeError(f"Unexpected tool call: {tc0}")
        args_str = getattr(fn, "arguments", None)
        if not args_str:
            raise RuntimeError("Tool call arguments is empty.")
        logger.debug("tool_call args len=%d preview=%.200s", len(args_str), args_str)
        args = json.loads(args_str)
        ranked_raw = args["results"]
    else:
        # 2) 适配：模型把 JSON 放进 message.content（甚至包在 ```json``` 里）
        content_raw = getattr(msg, "content", "") or ""
        logger.debug("content fallback len=%d preview=%.200s", len(content_raw), content_raw)
        ranked_raw = parse_results_from_message_content(content_raw)

    ranked = enforce_unique_and_bounds(ranked_raw, doc_count=len(docs), top_n=top_n)

    return {
        "ranked": ranked,
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
    }


# -----------------------------
# FastAPI
# -----------------------------
app = FastAPI(title="llm2reranker", version="0.4.0")


@app.post("/v1/rerank", response_model=RerankResponse, response_model_exclude_none=True)
async def rerank(req: RerankRequest):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")

    docs_text = normalize_documents(req.documents)
    if not docs_text:
        raise HTTPException(status_code=400, detail="documents is empty")

    top_n = req.top_n if req.top_n and req.top_n > 0 else 10

    logger.info("rerank query=%r docs=%d top_n=%d model=%s", req.query[:80], len(docs_text), top_n, req.model)
    _t0 = time.time()
    rr = await llm_rerank(req.query, docs_text, top_n=top_n)
    _elapsed = time.time() - _t0

    ranked: List[Dict[str, Any]] = rr["ranked"]
    prompt_tokens = int(rr["usage"].get("prompt_tokens", 0))
    completion_tokens = int(rr["usage"].get("completion_tokens", 0))
    logger.info("rerank done elapsed=%.2fs prompt=%d completion=%d results=%d", _elapsed, prompt_tokens, completion_tokens, len(ranked))

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
            results.append(RerankResult(document=None, index=idx, relevance_score=score))

    meta = RerankMeta(
        billed_units=RerankMetaBilledUnits(
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            search_units=0,
            classifications=0,
        ),
        tokens=RerankMetaTokens(input_tokens=prompt_tokens, output_tokens=completion_tokens),
    )

    return RerankResponse(id=uuid.uuid4().hex, results=results, meta=meta)


@app.get("/healthz")
async def healthz():
    return {"ok": True}