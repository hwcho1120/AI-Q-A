"""셀프 채점기: 공개 예시 문항으로 대회 채점 방식을 흉내 냅니다.

  근거 정확도  항까지 일치 1.0 · 조까지만 일치 0.5 · 별표 인용도 정답 인정
  함정 문항    answer가 "확인 불가"이고 citation이 []이면 정답
  형식         source_url · effective_date가 비어 있지 않은지, 90초 안에 응답했는지

실행:
  python -X utf8 check.py --retrieval      검색만 점검 (API 키 없이도 동작)
  python -X utf8 check.py                  로컬 함수로 전체 채점
  python -X utf8 check.py --url http://localhost:8000/ask   서버로 채점 (제출 전 최종 점검)

※ 실제 채점 규칙의 세부(인용을 여러 개 냈을 때의 처리 등)는 운영진 공지가 기준입니다.
"""
import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import config

CITE = re.compile(r"(서울특별시.*?(?:조례|규칙))\s*(제\d+조(?:의\d+)?|별표\s*\d*|별도\s*\d+)\s*(제\d+항)?")


def parse(c):
    m = CITE.search(c.replace(" ", " "))
    if not m:
        return None
    law = re.sub(r"\s+", "", m.group(1))
    art = re.sub(r"\s+", "", m.group(2))
    para = m.group(3)
    return law, art, para


def match(pred, gold):
    p, g = parse(pred), parse(gold)
    if not p or not g or p[0] != g[0] or p[1] != g[1]:
        return 0.0
    if g[2] is None or p[2] == g[2]:
        return 1.0
    return 0.5


def score_citation(preds, gold):
    """gold: [[대안1, 대안2], [다음 필수 근거], ...]"""
    if not gold:
        return None
    total = 0.0
    for alts in gold:
        total += max((match(p, a) for p in preds for a in alts), default=0.0)
    return total / len(gold)


def run_retrieval(qs, workers=1):
    import rag
    print(f"[검색 점검] 단위={config.CHUNK_UNIT}  top_k={config.TOP_K}  문항 {len(qs)}개\n")

    def one(q):
        try:
            return [h["label"] for h in rag.retrieve(q["question"])]
        except Exception as e:  # noqa: BLE001
            return []

    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(workers) as ex:
            found = list(ex.map(one, qs))
    else:
        found = [one(q) for q in qs]
    scores, by_type = [], {}
    for q, labels in zip(qs, found):
        if not q["gold"]:
            continue
        sc = score_citation(labels, q["gold"])
        scores.append(sc)
        by_type.setdefault(q["type"], []).append(sc)
        if sc < 1 and len(qs) <= 60:
            print(f"{q['id']:5} {'△' if sc > 0 else '✗'} {sc:.2f}  {q['question'][:34]}\n      찾은 것: {labels[:3]}")
    print("유형별 (정답 근거가 top-%d 안에 들어온 비율):" % config.TOP_K)
    for t, v in sorted(by_type.items()):
        print(f"  {t:28} n={len(v):4}  {sum(v) / len(v):.0%}")
    print(f"\n전체(함정 제외, n={len(scores)}): {sum(scores) / max(len(scores), 1):.0%}")


def run_full(qs, url=None, workers=1, tag="", merge_into=None):
    if url:
        import requests
        call = lambda q: requests.post(url, json={"question": q}, timeout=90).json()  # noqa: E731
    else:
        import rag
        call = rag.answer

    def one(q):
        t0 = time.time()
        try:
            r = call(q["question"])
            err = None
        except Exception as e:  # noqa: BLE001
            r, err = {}, repr(e)
        sec = time.time() - t0
        unknown = r.get("answer", "").startswith("확인 불가") and r.get("citation") == []
        if q["gold"]:
            sc = 0.0 if err or sec > 90 else score_citation(r.get("citation", []), q["gold"])
        else:
            sc = 1.0 if unknown and not err and sec <= 90 else 0.0
        fmt_ok = unknown or (r.get("source_url", "").startswith("http") and
                             re.fullmatch(r"\d{4}-\d{2}-\d{2}", r.get("effective_date", "")))
        return {"id": q["id"], "type": q["type"], "question": q["question"], "score": sc,
                "sec": round(sec, 1), "format_ok": bool(fmt_ok), "error": err, "response": r,
                "gold": q["gold"], "answer_key": q["answer_key"]}

    if workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(workers) as ex:
            rows = list(ex.map(one, qs))
    else:
        rows = [one(q) for q in qs]
    if merge_into is not None:      # 오류였던 행만 새 결과로 교체
        new = {r["id"]: r for r in rows}
        print(f"재실행 {len(rows)}건 중 아직 오류: {sum(1 for r in rows if r['error'])}건")
        rows = [new.get(r["id"], r) for r in merge_into]
    for row in rows:
        print(f"{row['id']:5} {row['score']:.2f}  {row['sec']:5.1f}s  {'형식OK' if row['format_ok'] else '형식✗'}  "
              f"{row['response'].get('citation')}")
    cite_scores = [r["score"] for r in rows if r["gold"]]
    trap_ok = [r["score"] for r in rows if not r["gold"]]
    print("\n유형별:")
    for t in sorted({r["type"] for r in rows}):
        sc = [r["score"] for r in rows if r["type"] == t]
        print(f"  {t:28} n={len(sc):4}  {sum(sc) / len(sc):.2f}")
    print("\n근거 정확도(일반 문항 평균):", f"{sum(cite_scores) / max(len(cite_scores), 1):.2f}", f"(n={len(cite_scores)})")
    print("함정 문항 정답률:", f"{sum(trap_ok) / max(len(trap_ok), 1):.0%}", f"(n={len(trap_ok)})")
    print("오류:", sum(1 for r in rows if r["error"]), " 90초 초과:", sum(1 for r in rows if r["sec"] > 90),
          " 형식✗:", sum(1 for r in rows if not r["format_ok"]))
    out = config.ROOT / "results"
    out.mkdir(exist_ok=True)
    path = out / f"check_{datetime.now():%m%d_%H%M}_{config.CHUNK_UNIT}_v{int(config.USE_VERIFY)}{tag}.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("저장:", path)


def stratified(qs, n, seed=0):
    """유형별 비율을 유지하며 n개를 뽑습니다 (같은 seed면 같은 문항)."""
    import random
    rnd = random.Random(seed)
    by = {}
    for q in qs:
        by.setdefault(q["type"], []).append(q)
    out = []
    for t, items in by.items():
        k = max(1, round(n * len(items) / len(qs)))
        out += rnd.sample(items, min(k, len(items)))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", action="store_true")
    ap.add_argument("--url")
    ap.add_argument("--questions", default=str(config.ROOT / "sample_questions.json"))
    ap.add_argument("--sample", type=int, help="유형별 비율을 유지해 N개만 채점")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1, help="동시 호출 수 (429가 나면 낮추세요)")
    ap.add_argument("--rerun-errors", metavar="RESULTS", help="이전 결과 파일에서 error가 있던 문항만 다시 돌려 합친다")
    a = ap.parse_args()
    qs = json.loads(open(a.questions, encoding="utf-8").read())
    if a.sample:
        qs = stratified(qs, a.sample, a.seed)
    tag = f"_{Path(a.questions).stem}" + (f"_s{a.sample}" if a.sample else "")
    if a.rerun_errors:
        prev = json.loads(open(a.rerun_errors, encoding="utf-8").read())
        bad = {r["id"] for r in prev if r["error"] or r["sec"] > 90}
        print(f"이전 결과 {len(prev)}건 중 오류·90초 초과 {len(bad)}건만 다시 실행")
        qs = [q for q in qs if q["id"] in bad]
        run_full(qs, a.url, a.workers, tag + "_rerun", merge_into=prev)
    elif a.retrieval:
        run_retrieval(qs, a.workers)
    else:
        run_full(qs, a.url, a.workers, tag)
