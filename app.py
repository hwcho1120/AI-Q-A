"""5단계: 대회 제출용 API 서버.

실행:  uvicorn app:app --host 0.0.0.0 --port 8000
호출:  POST /ask  {"question": "장애인 주차요금 감면되나요?"}

채점은 운영진이 마감 직후 1회 실행합니다. 문항당 90초를 넘기거나 오류가 나면 0점입니다.
그래서 어떤 경우에도 정해진 JSON을 돌려주도록 감쌌습니다.
"""
import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import config
import rag

log = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        rag._load()
        log.info("RAG index warmed up (unit=%s)", config.CHUNK_UNIT)
    except Exception as e:  # noqa: BLE001
        log.warning("warmup failed: %r", e)
    yield


app = FastAPI(title="서울시 조례 Q&A", lifespan=lifespan)

try:  # 시연 화면(/, /api/*). 실패해도 채점용 /ask 는 그대로 동작해야 합니다.
    import ui_api
    app.include_router(ui_api.router)
except Exception as _e:  # noqa: BLE001
    log.warning("UI disabled: %r", _e)


@app.exception_handler(RequestValidationError)
async def bad_request(request: Request, exc: RequestValidationError):
    """요청 본문이 깨지거나 필드가 달라도 대회 형식 JSON(확인 불가)으로 응답합니다."""
    log.warning("bad request: %s", str(exc)[:200])
    return JSONResponse(status_code=200, content=dict(rag.UNKNOWN))


class Ask(BaseModel):
    question: str


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
