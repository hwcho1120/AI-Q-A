"""부하 테스트: 동시에 여러 문항을 보내도 429·오류 없이 90초 안에 답하는지 확인합니다.

  python -X utf8 loadtest.py --url http://localhost:8000             동시 2·3·4문항, 각 단계 8문항
  python -X utf8 loadtest.py --url http://localhost:8000 --levels 2,4 --per-level 12
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
FIELDS = {"answer": str, "citation": list, "source_url": str, "effective_date": str}


def load_questions():
    qs = []
    for f in ("sample_questions.json", "holdout_questions.json"):
        p = ROOT / f
        if p.exists():
            qs += json.loads(p.read_text(encoding="utf-8"))
    return qs


def one(url, q):
    t0 = time.time()
    try:
        r = requests.post(f"{url}/ask", json={"question": q["question"]}, timeout=95)
        sec = time.time() - t0
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        fmt_ok = set(body) == set(FIELDS) and all(isinstance(body[k], t) for k, t in FIELDS.items())
        unknown = body.get("answer", "").startswith("확인 불가")
        return {"id": q["id"], "status": r.status_code, "sec": sec, "fmt_ok": fmt_ok,
                "unexpected_unknown": bool(q["gold"]) and unknown, "error": None}
    except Exception as e:  # noqa: BLE001
        return {"id": q["id"], "status": None, "sec": time.time() - t0, "fmt_ok": False,
                "unexpected_unknown": False, "error": repr(e)[:120]}


def health(url):
    try:
        return requests.get(f"{url}/health", timeout=10).json()
    except Exception:  # noqa: BLE001
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--levels", default="2,3,4", help="동시 요청 수 목록")
    ap.add_argument("--per-level", type=int, default=8, help="단계마다 보낼 문항 수")
    a = ap.parse_args()
    url = a.url.rstrip("/").removesuffix("/ask")
    qs = load_questions()
    h0 = health(url)
    if not h0:
        raise SystemExit(f"{url}/health 응답 없음: 서버가 떠 있는지 확인하세요.")
    print(f"서버: warm={h0.get('warm')} warmup={h0.get('warmup_sec')}s has_key={h0.get('has_key')}\n")
    all_rows, offset = [], 0
    for n in [int(x) for x in a.levels.split(",")]:
        batch = [qs[(offset + i) % len(qs)] for i in range(a.per_level)]
        offset += a.per_level
        s0 = health(url).get("upstage", {})
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=n) as ex:
            rows = list(ex.map(lambda q: one(url, q), batch))
        wall = time.time() - t0
        s1 = health(url).get("upstage", {})
        d = {k: s1.get(k, 0) - s0.get(k, 0) for k in ("calls", "retry_429", "retry_other", "failed", "ask_failed")}
        bad_http = [r for r in rows if r["status"] != 200]
        print(f"[동시 {n}] {len(rows)}문항 {wall:.1f}s | 최대 {max(r['sec'] for r in rows):.1f}s"
              f" 평균 {sum(r['sec'] for r in rows) / len(rows):.1f}s | HTTP오류 {len(bad_http)}"
              f" 형식이상 {sum(not r['fmt_ok'] for r in rows)} 90초초과 {sum(r['sec'] > 90 for r in rows)}"
              f" | 429재시도 {d['retry_429']} 기타재시도 {d['retry_other']} API실패 {d['failed']}"
              f" 덮인오류 {d['ask_failed']} | 일반문항 확인불가 {sum(r['unexpected_unknown'] for r in rows)}")
        for r in rows:
            if r["error"] or r["status"] != 200:
                print("    ", r["id"], r["status"], r["error"])
        all_rows += rows
    h1 = health(url).get("upstage", {})
    tot = {k: h1.get(k, 0) - h0.get("upstage", {}).get(k, 0) for k in ("retry_429", "failed", "ask_failed")}
    ok = (all(r["status"] == 200 and r["fmt_ok"] and r["sec"] <= 90 for r in all_rows)
          and tot["failed"] == 0 and tot["ask_failed"] == 0)
    print(f"\n전체 {len(all_rows)}문항 | 최대 응답 {max(r['sec'] for r in all_rows):.1f}s"
          f" | 429 재시도 {tot['retry_429']}회 · API 실패 {tot['failed']} · 덮인 오류 {tot['ask_failed']}"
          f" → {'통과' if ok else '실패'}")


if __name__ == "__main__":
    main()
