"""대형 테스트셋 채점기: 교차검증 비교표 양식(Recall·Precision·F1·Exact Match·정답률·함정·과잉 확인불가·시간).

  일반 근거 정확도(Recall)  check.score_citation (항 일치 1.0 · 조까지만 0.5), 일반 문항 평균
  Strict Precision          낸 인용 중 정답과 '항까지' 일치하는 비율 (인용 0개면 0)
  F1                        문항별 Precision·Recall의 조화평균을 평균
  Exact Match               Precision=1 이고 Recall=1 인 문항 비율
  답변 정답률               diagnose.judge (CORRECT / CORRECT+PARTIAL), 일반 문항 기준
  함정 방어율               함정 문항에서 '확인 불가' + citation [] 비율
  과잉 확인불가             일반 문항인데 '확인 불가'로 답한 수
  자치구 유출               6개 조례 밖의 조례·규칙을 인용한 응답 수

실행:  python -X utf8 grade.py test_set_1000.json [test_set_2_1000.json ...] --workers 8
결과:  results/grade_<세트>_<시각>.json, 마지막에 비교표(Markdown) 출력
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import config
import rag
from check import match, parse, score_citation
from diagnose import judge

LAWS = {parse(f"{c['law']} 제1조")[0] for c in json.loads(
    (config.DATA / f"chunks_{config.CHUNK_UNIT}.json").read_text(encoding="utf-8"))}


def run_one(q):
    t0 = time.time()
    try:
        r, err = rag.answer(q["question"]), None
    except Exception as e:  # noqa: BLE001
        r, err = dict(rag.UNKNOWN), repr(e)
    sec = time.time() - t0
    cites = r.get("citation", [])
    unknown = r.get("answer", "").startswith("확인 불가") and cites == []
    row = {"id": q["id"], "type": q.get("type"), "question": q["question"], "gold": q["gold"],
           "answer_key": q.get("answer_key", ""), "response": r, "sec": round(sec, 2), "error": err,
           "unknown": unknown,
           "leak": any((parse(c) or ("?",))[0] not in LAWS for c in cites)}
    if q["gold"]:
        ok90 = not err and sec <= 90
        rec = score_citation(cites, q["gold"]) if ok90 else 0.0
        alts = [a for g in q["gold"] for a in g]
        strict = [any(match(c, a) == 1.0 for a in alts) for c in cites]
        prec = sum(strict) / len(cites) if cites and ok90 else 0.0
        row.update(recall=rec, precision=prec,
                   f1=2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                   exact=float(prec == 1.0 and rec == 1.0))
        c, why = judge(q, r)
        row.update(content=c, content_reason=why)
    else:
        row["trap_ok"] = float(unknown and not err and sec <= 90)
    return row


def summary(rows):
    g = [r for r in rows if r["gold"]]
    t = [r for r in rows if not r["gold"]]
    avg = lambda xs: sum(xs) / max(len(xs), 1)  # noqa: E731
    secs = [r["sec"] for r in rows]
    return {
        "n": len(rows), "n_general": len(g), "n_trap": len(t),
        "recall": avg([r["recall"] for r in g]),
        "precision": avg([r["precision"] for r in g]),
        "f1": avg([r["f1"] for r in g]),
        "exact": avg([r["exact"] for r in g]),
        "correct": sum(r["content"] == 1.0 for r in g),
        "partial": sum(r["content"] == 0.5 for r in g),
        "judge_fail": sum(r["content"] is None for r in g),
        "trap_ok": int(sum(r["trap_ok"] for r in t)),
        "false_refusal": sum(r["unknown"] for r in g),
        "avg_sec": avg(secs), "max_sec": max(secs), "p90_sec": sorted(secs)[int(len(secs) * 0.9) - 1],
        "within90": sum(s <= 90 for s in secs),
        "errors": sum(bool(r["error"]) for r in rows), "leak": sum(r["leak"] for r in rows),
    }


def table(named):
    names = list(named)
    s = [named[n] for n in names]
    pct = lambda a, b: f"{a / max(b, 1):.2%}"  # noqa: E731
    lines = ["| 채점 지표 | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    def row(label, f):
        lines.append(f"| {label} | " + " | ".join(f(x) for x in s) + " |")
    row("일반 근거 정확도 (Recall)", lambda x: f"{x['recall']:.4f} ({x['n_general']}문항)")
    row("Strict Precision / F1 / Exact Match",
        lambda x: f"{x['precision']:.4f} / {x['f1']:.4f} / {x['exact']:.1%}")
    row("답변 정답률 (CORRECT / +PARTIAL)",
        lambda x: f"{pct(x['correct'], x['n_general'])} / {pct(x['correct'] + x['partial'], x['n_general'])}"
                  f" ({x['correct']} 정답 · {x['partial']} 부분)")
    row("함정 방어율", lambda x: f"{pct(x['trap_ok'], x['n_trap'])} ({x['trap_ok']}/{x['n_trap']})")
    row("과잉 확인 불가 (false_refusal)",
        lambda x: f"{x['false_refusal']}건 ({pct(x['false_refusal'], x['n_general'])})")
    row("평균 / 90%구간 / 최대 시간", lambda x: f"{x['avg_sec']:.2f}s / {x['p90_sec']:.2f}s / {x['max_sec']:.2f}s")
    row("90초 통과 · 오류 · 자치구 유출",
        lambda x: f"{x['within90']}/{x['n']} · 오류 {x['errors']} · 유출 {x['leak']}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sets", nargs="+")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = config.ROOT / "results"
    out.mkdir(exist_ok=True)
    named = {}
    for path in a.sets:
        qs = json.loads(open(path, encoding="utf-8").read())
        t0 = time.time()
        with ThreadPoolExecutor(a.workers) as ex:
            rows = list(ex.map(run_one, qs))
        s = summary(rows)
        s["wall_min"] = round((time.time() - t0) / 60, 1)
        name = Path(path).stem
        named[name] = s
        f = out / f"grade_{name}_{datetime.now():%m%d_%H%M%S}.json"
        f.write_text(json.dumps({"summary": s, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
        print(name, json.dumps(s, ensure_ascii=False), "\n저장:", f, flush=True)
    print("\n" + table(named))


if __name__ == "__main__":
    main()
