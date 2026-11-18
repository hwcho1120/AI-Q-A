"""5단계: 대회 제출용 API 서버.

실행:  uvicorn app:app --host 0.0.0.0 --port 8000
호출:  POST /ask  {"question": "장애인 주차요금 감면되나요?"}

채점은 운영진이 마감 직후 1회 실행합니다. 문항당 90초를 넘기거나 오류가 나면 0점입니다.
그래서 어떤 경우에도 정해진 JSON을 돌려주도록 감쌌습니다.
"""
import asyncio
import logging
import re

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

import config
import rag

app = FastAPI(title="서울시 조례 Q&A")
log = logging.getLogger("uvicorn.error")


class Ask(BaseModel):
    question: str


class Turn(BaseModel):
    q: str
    a: str = ""
    c: list[str] = []


class Chat(BaseModel):
    question: str
    history: list[Turn] = []


@app.get("/health")
def health():
    return {"ok": True, "chunk_unit": config.CHUNK_UNIT, "verify": config.USE_VERIFY}


@app.post("/ask")
async def ask(req: Ask):
    try:
        return await asyncio.wait_for(asyncio.to_thread(rag.answer, req.question),
                                      timeout=config.ANSWER_TIMEOUT_SEC)
    except Exception as e:  # noqa: BLE001  시간 초과·API 오류 모두 '확인 불가'로 응답
        log.warning("ask failed: %r", e)
        return dict(rag.UNKNOWN)


def _chat(req: Chat):
    """이어진 질문: 앞 대화를 참고해 질문을 독립된 문장으로 다시 쓴 뒤, 같은 엔진(rag.answer)으로 답한다."""
    standalone = rag.condense(req.question, [t.model_dump() for t in req.history[-3:]])
    out = rag.answer(standalone)
    return {**out, "standalone": standalone}


@app.post("/chat")
async def chat(req: Chat):
    """화면용 엔드포인트. 채점용 /ask 와 별개이며, 응답은 /ask 의 4개 필드에 standalone(다시 쓴 질문)이 더해진다."""
    try:
        return await asyncio.wait_for(asyncio.to_thread(_chat, req), timeout=config.ANSWER_TIMEOUT_SEC)
    except Exception as e:  # noqa: BLE001  시간 초과·API 오류 모두 '확인 불가'로 응답
        log.warning("chat failed: %r", e)
        return {**rag.UNKNOWN, "standalone": req.question}


# ------------------------------------------------------------ 시민용 화면 (채점 API와 별개)
STATIC = config.ROOT / "static"
_AMEND = re.compile(r"<(?:개정|신설|전문개정|본조신설|삭제)[^>]*>|\[(?:본조신설|전문개정|제목개정|종전)[^\]]*\]")
_evidence = {}


def _evidence_index():
    """인용 라벨 → 원문. 별표처럼 한 라벨에 조각이 여럿이면 이어 붙인다."""
    if not _evidence:
        for c in rag._load()[0]:
            e = _evidence.setdefault(c["label"], {"label": c["label"], "heading": "", "parts": [],
                                                  "source_url": c["source_url"], "effective_date": c["effective_date"]})
            lines = c["text"].split("\n")
            if lines[0].startswith("[") and lines[0].endswith("]"):     # 항 조각의 머리말 '[법 제N조(제목)]'
                e["heading"] = e["heading"] or lines[0][1:-1]
                lines = lines[1:]
            e["parts"].append("\n".join(lines))
    return _evidence


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/evidence")
def evidence(label: str):
    """화면에서 근거 조항 원문을 펼칠 때 쓴다. (/ask 응답 형식과 무관)"""
    e = _evidence_index().get(label)
    if not e:
        return JSONResponse({"error": "not found"}, status_code=404)
    text = _AMEND.sub("", "\n\n".join(e["parts"]))
    text = re.sub(r"[ \t]+\n", "\n", re.sub(r"\n{3,}", "\n\n", text)).strip()
    return {"label": e["label"], "heading": e["heading"], "text": text,
            "source_url": e["source_url"], "effective_date": e["effective_date"]}
