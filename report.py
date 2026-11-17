"""check.py가 저장한 결과(results/*.json)로 '교차검증 비교표'의 지표를 계산합니다.

  python3 -X utf8 report.py results/check_...json              # 인용·함정·시간 지표 (API 호출 없음)
  python3 -X utf8 report.py results/check_...json --judge       # + 답변 정답률 (solar-pro4가 answer_key와 비교, 문항당 1회 호출)

지표 정의 (운영진 공식 정의가 아니라 이 프로젝트의 정의입니다)
  Recall      정답 근거가 있는 문항에서 check.score_citation 평균 (조+항 1.0, 조만 0.5, 별표 인정)
  Precision   내놓은 인용 중 정답 슬롯과 조+항까지 일치한 비율 (문항별 평균). 인용이 없으면 0
  F1          Precision과 Recall 평균값의 조화평균
  Exact Match 인용이 정답과 정확히 같은 문항 (Recall=1, Precision=1)
  답변 정답률  CORRECT / PARTIAL / WRONG 을 LLM이 answer_key와 비교해 판정. 함정은 '확인 불가'면 정답
  함정 방어율  gold가 없는 문항에서 '확인 불가'로 답한 비율
  과잉 확인 불가  gold가 있는데 '확인 불가'로 답한 비율
"""
import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import check
import config
import rag

JUDGE_SCHEMA = {"type": "json_schema", "json_schema": {"name": "judge", "strict": True, "schema": {
    "type": "object", "properties": {"verdict": {"type": "string", "enum": ["CORRECT", "PARTIAL", "WRONG"]}},
    "required": ["verdict"], "additionalProperties": False}}}

JUDGE_PROMPT = """서울시 조례 질의응답의 채점관이다. [정답 요지]와 [모델 답변]을 비교해 판정하라.
- CORRECT: 핵심 결론(예/아니오, 금액, 나이, 대상 등)이 정답 요지와 일치하고 중요한 조건·예외를 빠뜨리지 않음
- PARTIAL: 결론은 맞지만 중요한 조건·예외·수치가 빠졌거나 일부만 맞음
- WRONG: 결론이 다르거나, 답을 못 했거나('확인 불가' 포함), 질문과 무관함"""


def judge_one(row):
    import upstage
    ans = row["response"].get("answer", "")
    if ans.startswith("확인 불가"):
        return "WRONG"
    try:
        out = upstage.chat([{"role": "system", "content": JUDGE_PROMPT},
                            {"role": "user", "content": f"[질문]\n{row['question']}\n\n[정답 요지]\n{row['answer_key']}\n\n[모델 답변]\n{ans}"}],
                           response_format=JUDGE_SCHEMA)
        return json.loads(out)["verdict"]
    except Exception:  # noqa: BLE001
        return "ERROR"


def cited_ok(c, gold):
    return any(check.match(c, alt) == 1.0 for slot in gold for alt in slot)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    rows = json.loads(Path(a.path).read_text(encoding="utf-8"))
    gold_rows = [r for r in rows if r["gold"]]
    trap_rows = [r for r in rows if not r["gold"]]

    # ---- 인용 지표
    rec, prec, em = [], [], []
    for r in gold_rows:
        cites = r["response"].get("citation", [])
        rc = check.score_citation(cites, r["gold"]) if cites else 0.0
        pr = sum(cited_ok(c, r["gold"]) for c in cites) / len(cites) if cites else 0.0
        rec.append(rc)
        prec.append(pr)
        em.append(rc == 1.0 and pr == 1.0)
    R, P = sum(rec) / len(rec), sum(prec) / len(prec)
    F1 = 2 * P * R / (P + R) if P + R else 0.0

    unknown = lambda r: r["response"].get("answer", "").startswith("확인 불가") and not r["response"].get("citation")  # noqa: E731
    trap_ok = sum(unknown(r) for r in trap_rows)
    refuse = sum(unknown(r) for r in gold_rows)
    secs = [r["sec"] for r in rows]
    targets = tuple(config.TARGET_ORDINANCES)
    leak = sum(1 for r in rows for c in r["response"].get("citation", [])
               if rag.DISTRICT.search(c) or not c.startswith(targets))
    errors = sum(1 for r in rows if r["error"])

    out = {
        "n": len(rows), "n_gold": len(gold_rows), "n_trap": len(trap_rows),
        "recall": R, "precision": P, "f1": F1, "exact": sum(em) / len(em), "exact_n": sum(em),
        "trap_ok": trap_ok, "false_refusal": refuse,
        "avg_sec": sum(secs) / len(secs), "max_sec": max(secs), "within_90": sum(s <= 90 for s in secs),
        "errors": errors, "district_leak": leak, "format_bad": sum(1 for r in rows if not r["format_ok"]),
    }

    # ---- 답변 정답률 (LLM 판정)
    if a.judge:
        cache = Path(a.path).with_suffix(".judge.json")
        verdicts = json.loads(cache.read_text(encoding="utf-8")) if cache.exists() else {}
        todo = [r for r in gold_rows if r["id"] not in verdicts]
        with ThreadPoolExecutor(a.workers) as ex:
            for r, v in zip(todo, ex.map(judge_one, todo)):
                verdicts[r["id"]] = v
        cache.write_text(json.dumps(verdicts, ensure_ascii=False, indent=1), encoding="utf-8")
        vs = [verdicts[r["id"]] for r in gold_rows]
        out["correct"] = vs.count("CORRECT")
        out["partial"] = vs.count("PARTIAL")
        out["judge_error"] = vs.count("ERROR")

    print(json.dumps(out, ensure_ascii=False, indent=1))
    print_table(out)


def pct(x, n):
    return f"{x / n:.2%}"


def print_table(o):
    ng, nt, n = o["n_gold"], o["n_trap"], o["n"]
    print("\n| 채점 지표 | 우리 결과 |\n|---|---|")
    print(f"| 일반 근거 정확도 (Recall, {ng}문항) | {o['recall']:.4f} |")
    print(f"| Strict Precision / F1 / Exact Match | {o['precision']:.4f} / {o['f1']:.4f} / {o['exact']:.1%} |")
    if "correct" in o:
        full, part = o["correct"], o["correct"] + o["partial"]
        print(f"| 답변 정답률 (CORRECT / PARTIAL 포함) | {pct(full, ng)} / {pct(part, ng)} ({full} 완전정답 / {o['partial']} 부분정답) |")
    print(f"| 함정 방어율 ({nt}문항) | {pct(o['trap_ok'], nt)} ({o['trap_ok']}/{nt}) |")
    print(f"| 과잉 확인 불가 (false_refusal) | {o['false_refusal']}건 ({pct(o['false_refusal'], ng)}) |")
    print(f"| 평균 / 최대 시간 · 90초 통과 | {o['avg_sec']:.2f}s / {o['max_sec']:.2f}s ({o['within_90']}/{n}) |")
    print(f"| 오류 · 자치구 유출 · 형식 오류 | {o['errors']}건 · {o['district_leak']}건 · {o['format_bad']}건 |")


if __name__ == "__main__":
    main()
