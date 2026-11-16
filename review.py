"""관리자 승인(Human-in-the-loop) 저장소.

업스테이지 Studio 워크플로의 Review → Accept / Deny → Approve 단계를 구현합니다.
  - Review  : 3단계 검증 결과로 자동 분류 (assess)
  - Accept  : 문제없이 통과한 답변은 바로 공개 (status 'auto')
  - Deny    : 답은 했지만 검증에서 문제가 있었던 답변은 담당자 승인 전까지 공개하지 않음 (status 'pending')
  - Approve : 담당자가 원문과 대조해 승인·수정하면 공개, 반려하면 비공개
'확인 불가'는 잘못된 정보가 나갈 위험이 없으므로 즉시 응답하고, '놓친 근거가 없는지' 점검 목록(status 'unknown')에만 올립니다.
승인·수정 결과는 평가 문항 형식으로 내보내 다음 개선에 씁니다 (피드백 루프).

채점용 POST /ask 는 90초 안에 자동 응답해야 하므로 승인을 기다리지 않고, 기록만 남겨 사후 검토합니다.
기록은 부가 기능이므로 어떤 오류가 나도 답변을 막지 않습니다. 저장: logs/review.db (SQLite)
"""
import json
import secrets
import sqlite3
import threading
import time
from pathlib import Path

import config

DB = Path(config.ROOT) / "logs" / "review.db"
_lock = threading.Lock()
_ready = False

STATUS = ("pending", "auto", "unknown", "approved", "rejected", "corrected")

STAGE_REASON = {
    "min_score": "검색 단계: 관련 조문 점수 미달",
    "no_basis": "근거 판단: 질문 핵심이 조례에 없음",
    "found_false": "근거 판단: 근거로 답할 수 없음",
    "no_evidence": "근거 판단: 인용할 조항 없음",
    "repair_found_false": "검증: 고쳐 쓴 뒤에도 근거 부족",
    "not_grounded": "검증: 답변이 원문과 불일치",
    "error": "오류·시간 초과",
}


def assess(result, trace, elapsed=None, error=None):
    """Review 단계: (분류, 이유 목록). 분류는 'auto' / 'pending' / 'unknown'."""
    trace = trace or {}
    stage = trace.get("stop") or ("error" if error else "")
    if error or stage == "error":
        return "unknown", ["오류·시간 초과"]
    if not result.get("citation"):
        return "unknown", ["확인 불가 — " + STAGE_REASON.get(stage, stage or "근거 없음")]
    risk = []
    if trace.get("repair"):
        risk.append("검증 지적으로 답변을 고쳐 씀")
    if trace.get("dropped_sentences"):
        risk.append(f"근거 없는 문장 {trace['dropped_sentences']}개 삭제")
    hits = trace.get("hits") or []
    cos = hits[0][2] if hits and len(hits[0]) > 2 else None
    if cos is not None and cos < config.MIN_SCORE + 0.15:
        risk.append(f"검색 유사도 낮음({cos:.2f})")
    info = [f"응답 지연({elapsed:.0f}초)"] if elapsed and elapsed > 30 else []
    return ("pending" if risk else "auto"), risk + info


def _conn():
    DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(DB, timeout=10)
    c.row_factory = sqlite3.Row
    return c


def _init():
    global _ready
    if _ready:
        return
    with _conn() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS qa (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL, source TEXT, question TEXT, result TEXT, stage TEXT,
            elapsed REAL, flags TEXT, top_hits TEXT, status TEXT,
            reviewer_note TEXT, corrected TEXT, reviewed_ts REAL,
            ticket TEXT, held INTEGER DEFAULT 0)""")
    _ready = True


def log_qa(source, question, result, trace=None, elapsed=None, error=None, force_review=False):
    """질의응답 1건 기록 → {'id', 'ticket', 'status', 'flags', 'held'}.
    held=True면 시민에게 답을 보여주지 않고 승인을 기다립니다 (웹 화면만 해당).
    실패해도 예외를 밖으로 내보내지 않고 None을 돌려줍니다."""
    try:
        trace = trace or {}
        status, flags = assess(result, trace, elapsed, error)
        if force_review and status == "auto":
            status, flags = "pending", ["시연: 모든 답변 담당자 승인"] + flags
        held = source == "web" and status == "pending"
        ticket = secrets.token_urlsafe(12)
        hits = [h[0] for h in (trace.get("hits") or [])[:5]]
        with _lock:
            _init()
            with _conn() as c:
                cur = c.execute(
                    "INSERT INTO qa (ts, source, question, result, stage, elapsed, flags, top_hits, status, ticket, held)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (time.time(), source, question, json.dumps(result, ensure_ascii=False),
                     trace.get("stop") or ("error" if error else ""), elapsed,
                     json.dumps(flags, ensure_ascii=False), json.dumps(hits, ensure_ascii=False),
                     status, ticket, int(held)))
                return {"id": cur.lastrowid, "ticket": ticket, "status": status, "flags": flags, "held": held}
    except Exception:  # noqa: BLE001
        return None


def _row(r):
    d = dict(r)
    for k in ("result", "flags", "top_hits", "corrected"):
        d[k] = json.loads(d[k]) if d.get(k) else None
    d.pop("ticket", None)            # 관리자 목록에 시민 조회용 키는 노출하지 않음
    return d


def listing(status=None, limit=200):
    with _lock:
        _init()
        with _conn() as c:
            if status in STATUS:
                rows = c.execute("SELECT * FROM qa WHERE status=? ORDER BY id DESC LIMIT ?", (status, limit))
            else:
                rows = c.execute("SELECT * FROM qa ORDER BY id DESC LIMIT ?", (limit,))
            return [_row(r) for r in rows]


def stats():
    with _lock:
        _init()
        with _conn() as c:
            counts = {r["status"]: r["n"] for r in c.execute("SELECT status, COUNT(*) n FROM qa GROUP BY status")}
            held = c.execute("SELECT COUNT(*) FROM qa WHERE held=1 AND status='pending'").fetchone()[0]
            avg = c.execute("SELECT AVG(elapsed) FROM qa WHERE elapsed IS NOT NULL").fetchone()[0]
    total = sum(counts.values())
    answered = total - counts.get("unknown", 0)
    return {"total": total, "counts": counts, "held": held, "avg_elapsed": round(avg or 0, 1),
            "auto_rate": round(counts.get("auto", 0) / answered, 3) if answered else None}


def decide(qid, status, note="", corrected=None):
    if status not in ("approved", "rejected", "corrected", "pending", "unknown"):
        raise ValueError("status")
    with _lock:
        _init()
        with _conn() as c:
            c.execute("UPDATE qa SET status=?, reviewer_note=?, corrected=?, reviewed_ts=? WHERE id=?",
                      (status, note, json.dumps(corrected, ensure_ascii=False) if corrected else None,
                       time.time(), qid))
            r = c.execute("SELECT * FROM qa WHERE id=?", (qid,)).fetchone()
    return _row(r) if r else None


def public(ticket):
    """시민 화면이 승인 대기 중인 답을 조회. 승인·수정된 뒤에만 답을 돌려줍니다."""
    with _lock:
        _init()
        with _conn() as c:
            r = c.execute("SELECT * FROM qa WHERE ticket=?", (ticket,)).fetchone()
    if not r:
        return {"state": "not_found"}
    d = _row(r)
    if d["status"] == "pending":
        return {"state": "pending"}
    if d["status"] == "rejected":
        return {"state": "rejected", "note": d["reviewer_note"] or ""}
    final = d["result"]
    if d["status"] == "corrected" and d["corrected"]:
        final = {**d["result"], **d["corrected"]}
        if not final.get("citation"):
            final.update(source_url="", effective_date="")
    return {"state": d["status"], "result": final, "note": d["reviewer_note"] or ""}


def export_eval():
    """승인·수정된 건 → 평가 문항 형식(diagnose.py --questions). 반려 건은 실패 분석용으로 따로."""
    eval_set, rejected = [], []
    for r in listing(limit=10000):
        if r["status"] == "rejected":
            rejected.append({"id": f"R{r['id']}", "question": r["question"],
                             "note": r["reviewer_note"] or "", "system_answer": r["result"]})
            continue
        if r["status"] not in ("approved", "corrected"):
            continue
        final = r["corrected"] if r["status"] == "corrected" and r["corrected"] else r["result"]
        cites = final.get("citation") or []
        eval_set.append({"id": f"R{r['id']}", "type": "관리자 검토", "question": r["question"],
                         "gold": [[c] for c in cites], "answer_key": final.get("answer", ""),
                         "note": r["reviewer_note"] or ""})
    return {"eval_questions": eval_set, "rejected_for_analysis": rejected}
