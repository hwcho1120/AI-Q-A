"""Fact Law 시민용 화면과 이어진 질문 API. 채점용 POST /ask 와는 분리되어 있습니다.

GET  /factlaw     시민용 질문 화면 (static/factlaw.html)
POST /api/chat    이어진 질문: 앞 대화를 참고해 질문을 독립된 문장으로 다시 쓴 뒤 rag.answer로 답한다

화면은 첫 질문에 POST /ask, 이어진 질문에 POST /api/chat, 근거 원문에 GET /api/source 를 부른다.
"""
import asyncio
import logging
import time
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse
from pydantic import BaseModel

import config
import rag

ROOT = Path(__file__).parent
router = APIRouter()
log = logging.getLogger("uvicorn.error")

CONDENSE_PROMPT = """서울시 조례 질의응답 대화다. [마지막 질문]이 앞 대화를 가리키는 말('그러면', '그럼', '그건', '거기', '장애인이면' 처럼 대상이 빠진 표현)을 담고 있으면,
이전 대화의 대상·장소·조건을 채워서 혼자 읽어도 뜻이 통하는 한 문장 질문으로 바꿔라.
- 마지막 질문이 이미 혼자서 뜻이 통하거나 화제가 바뀌었으면 원문 그대로 둔다.
- 답을 추측하거나 앞 대화에 없는 정보를 지어내지 않는다.
- 질문 한 문장만 출력한다. 설명이나 따옴표는 붙이지 않는다."""


def condense(question, history):
    """이어진 질문을 앞 대화를 참고해 독립된 질문으로 다시 쓴다.
    일시적 오류에 대비해 한 번 더 시도하고, 그래도 실패하면 원문 그대로 쓴다(실패 원인은 로그에 남긴다)."""
    if not history:
        return question
    import upstage
    parts = []
    for t in history[-3:]:
        cites = ", ".join(c.replace("서울특별시 ", "") for c in (t.get("c") or [])[:3])
        parts.append(f"질문: {t.get('q', '')}\n답변: {(t.get('a') or '')[:200]}" + (f"\n근거: {cites}" if cites else ""))
    prompt = "[이전 대화]\n" + "\n\n".join(parts) + f"\n\n[마지막 질문]\n{question}"
    for attempt in range(2):
        try:
            out = upstage.chat([{"role": "system", "content": CONDENSE_PROMPT},
                                {"role": "user", "content": prompt}]).strip().splitlines()
            rewritten = (out[0].strip()[:200] if out else "")
            if rewritten:
                return rewritten
            log.warning("condense: 빈 응답 (시도 %d)", attempt + 1)
        except Exception as e:  # noqa: BLE001
            log.warning("condense 실패 (시도 %d): %r", attempt + 1, e)
        time.sleep(0.8)
    return question


class Turn(BaseModel):
    q: str
    a: str = ""
    c: list[str] = []


class Chat(BaseModel):
    question: str
    history: list[Turn] = []


def _chat(req: Chat):
    t0, trace = time.time(), {}
    standalone = condense(req.question, [t.model_dump() for t in req.history[-3:]])
    try:
        result = rag.answer(standalone, trace)
    except Exception as e:  # noqa: BLE001
        _log("chat", standalone, dict(rag.UNKNOWN), trace, time.time() - t0, repr(e))
        raise
    _log("api", standalone, result, trace, time.time() - t0)
    # reason: 화면 안내용 처리 단계(예: other_local_gov = 자치구·타 시도 질문 규칙 차단). 채점용 /ask 응답에는 없습니다.
    return {**result, "standalone": standalone, "reason": trace.get("stop", "")}


def _log(*args):
    """담당자 검토 기록. 실패해도 답변에 영향을 주지 않는다."""
    try:
        import review
        review.log_qa(*args)
    except Exception:  # noqa: BLE001
        pass


@router.get("/factlaw")
def page():
    return FileResponse(ROOT / "static" / "factlaw.html")


@router.post("/api/chat")
async def chat(req: Chat):
    """응답은 /ask 의 4개 필드에 standalone(다시 쓴 질문)이 더해진다. 어떤 예외도 '확인 불가'로 응답한다."""
    try:
        return await asyncio.wait_for(asyncio.to_thread(_chat, req), timeout=config.ANSWER_TIMEOUT_SEC)
    except Exception as e:  # noqa: BLE001
        log.warning("chat failed: %r", e)
        return {**rag.UNKNOWN, "standalone": req.question, "reason": "error"}
