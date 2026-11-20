"""시연용 화면(UI)과 보조 API. 채점용 POST /ask 와는 분리되어 있습니다.

GET  /api/source?label= 인용 조항의 원문 (조 전체 + 해당 항 표시 / 별표는 표)
GET  /admin             담당자 승인 화면 (/api/admin/*, ADMIN_TOKEN 설정 시 토큰 필요)
"""
import json
import re
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse
from pydantic import BaseModel

import os

from fastapi import Header, HTTPException, Query

import review

ROOT = Path(__file__).parent
router = APIRouter()
# 관리자 화면 보호: 환경변수 ADMIN_TOKEN을 정하면 /admin API에 그 토큰이 필요합니다 (배포 시 반드시 설정).
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")


def _admin(token):
    if ADMIN_TOKEN and token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="관리자 토큰이 필요합니다")


class Decision(BaseModel):
    status: str                 # approved / rejected / corrected / pending / unknown
    note: str = ""
    answer: str | None = None   # corrected일 때 수정 답변
    citation: list[str] | None = None


@router.get("/admin")
def admin_page():
    return FileResponse(ROOT / "static" / "admin.html")


@router.get("/api/admin/stats")
def admin_stats(x_admin_token: str = Header(""), token: str = Query("")):
    _admin(x_admin_token or token)
    return review.stats()


@router.get("/api/admin/items")
def admin_items(status: str = "pending",x_admin_token: str = Header(""), token: str = Query("")):
    _admin(x_admin_token or token)
    return review.listing(status)


@router.post("/api/admin/items/{qid}")
def admin_decide(qid: int, d: Decision, x_admin_token: str = Header(""), token: str = Query("")):
    _admin(x_admin_token or token)
    corrected = None
    if d.status == "corrected":
        corrected = {"answer": d.answer or "", "citation": [c for c in (d.citation or []) if c.strip()]}
    try:
        return review.decide(qid, d.status, d.note, corrected)
    except ValueError:
        raise HTTPException(status_code=400, detail="status는 approved/rejected/corrected/pending")


@router.get("/api/admin/export")
def admin_export(x_admin_token: str = Header(""), token: str = Query("")):
    _admin(x_admin_token or token)
    return review.export_eval()

_ord = None


def _ordinances():
    global _ord
    if _ord is None:
        _ord = json.loads((ROOT / "data" / "ordinances.json").read_text(encoding="utf-8"))
    return _ord


_ART = re.compile(r"(제\d+조(?:의\d+)?)(?:\s*제(\d+)항)?")


@router.get("/api/source")
def source(label: str):
    laws = sorted(_ordinances(), key=lambda o: -len(o["name"]))
    law = next((o for o in laws if label.startswith(o["name"])), None)
    if not law:
        return {"found": False, "label": label}
    rest = label[len(law["name"]):].strip()
    base = {"found": True, "label": label, "law": law["name"],
            "effective_date": law["effective_date"], "source_url": law["source_url"]}

    if rest.startswith("별표") or rest.startswith("별도"):
        key = rest.replace(" ", "")
        md = ROOT / "data" / "annex" / f"{law['mst']}_{key}.md"
        ann = next((a for a in law.get("annexes", []) if a["label"].replace(" ", "") == key), None)
        if md.exists():
            return {**base, "kind": "annex", "title": ann["title"] if ann else rest,
                    "markdown": md.read_text(encoding="utf-8")}
        if ann:
            return {**base, "kind": "annex_raw", "title": ann["title"], "text": ann["text"]}
        return {**base, "found": False}

    m = _ART.match(rest)
    if not m:
        return {**base, "found": False}
    art = next((a for a in law["articles"] if a["article"] == m.group(1)), None)
    if not art:
        return {**base, "found": False}
    paras = art.get("paragraphs") or []
    if not paras:
        paras = [{"no": None, "text": re.sub(r"^제\d+조(?:의\d+)?\([^)]*\)\s*", "", art["text"])}]
    return {**base, "kind": "article", "article": art["article"], "title": art.get("title", ""),
            "paragraphs": [{"no": p.get("no"), "text": p["text"]} for p in paras],
            "highlight": int(m.group(2)) if m.group(2) else None}
