"""진단기: 문항마다 N번 돌려 흔들림(분산)과 '확인 불가'가 난 단계를 집계합니다.

두 가지를 채점합니다.
  근거   인용 조항이 정답 조항과 맞는가 (check.py 방식: 항까지 1.0, 조까지 0.5)
  내용   답변의 결론·핵심 숫자가 정답 요지(answer_key)와 맞는가 (Solar 판정: 맞음 1 / 일부 0.5 / 틀림 0)
         함정 문항은 '확인 불가'면 1

실행:  python -X utf8 diagnose.py                          (공개 예시, 3회)
       python -X utf8 diagnose.py -n 2 --questions eval_questions.json
       python -X utf8 diagnose.py --only C1,M1
       python -X utf8 diagnose.py --judge results/diag_x.json  (이미 돌린 결과에 내용 채점만)
결과:  results/diag_<문항세트>_<시각>.json  (문항별 모든 실행의 trace·판정 포함)
"""
import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import config
import rag
from check import score_citation

JUDGE_SYSTEM = """너는 법령 Q&A 채점관이다. [질문], [정답 요지], [답변]을 비교한다. 아래 순서대로 채운다.
1. key_conclusion: 정답 요지가 질문에 내린 결론을 한 문장으로 (예: "면제된다", "둘 다는 안 된다", "500원")
2. answer_conclusion: 답변이 질문에 내린 결론을 한 문장으로. 답변 안에서 앞뒤가 다르면 질문에 직접 답한 문장
   (보통 첫 문장)을 기준으로 하고, contradiction을 true로 한다.
3. same_conclusion: 두 결론이 같은 뜻인가 (가능↔불가, 면제↔부과, 금액이 다르면 false)
4. verdict:
   - correct: same_conclusion이 true이고 contradiction이 false이며, 정답 요지의 핵심 숫자·조건이 틀리지 않았다.
   - partial: same_conclusion이 true지만 핵심 조건·숫자·단서 일부가 빠졌거나 부정확하다.
   - wrong: same_conclusion이 false이거나, contradiction이 true이거나, 핵심 숫자가 틀렸다.
정답 요지에 없는 부가 설명은 틀리지 않았으면 감점하지 않는다. 표현 방식(100분의 50 = 50%)은 같은 것으로 본다."""

JUDGE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "judge",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"key_conclusion": {"type": "string"},
                           "answer_conclusion": {"type": "string"},
                           "contradiction": {"type": "boolean"},
                           "same_conclusion": {"type": "boolean"},
                           "verdict": {"type": "string", "enum": ["correct", "partial", "wrong"]},
                           "reason": {"type": "string"}},
            "required": ["key_conclusion", "answer_conclusion", "contradiction", "same_conclusion",
                         "verdict", "reason"],
            "additionalProperties": False,
        },
    },
}
JUDGE_SCORE = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}


def judge(q, resp):
    """답변 내용 채점. 함정 문항은 '확인 불가'인지로만 판단합니다."""
    unknown = resp["answer"].startswith("확인 불가") and resp["citation"] == []
    if not q["gold"]:
        return float(unknown), "함정: 확인 불가" if unknown else "함정에 답함"
    if unknown:
        return 0.0, "확인 불가로 답함"
    import upstage
    try:
        raw = upstage.chat([{"role": "system", "content": JUDGE_SYSTEM},
                            {"role": "user", "content": f"[질문]\n{q['question']}\n\n[정답 요지]\n{q['answer_key']}"
                                                        f"\n\n[답변]\n{resp['answer']}"}],
                           response_format=JUDGE_SCHEMA)
        out = json.loads(raw)
        verdict = out["verdict"]
        if not out["same_conclusion"] or out["contradiction"]:   # 단계별 판정과 최종 판정이 어긋나면 엄격한 쪽
            verdict = "wrong"
        return JUDGE_SCORE[verdict], f"{out['reason']} (정답:{out['key_conclusion']} / 답:{out['answer_conclusion']})"
    except Exception as e:  # noqa: BLE001
        return None, f"채점 실패: {e!r}"


def run_once(q):
    trace = {}
    try:
        r = rag.answer(q["question"], trace)
        err = None
    except Exception as e:  # noqa: BLE001
        r, err = dict(rag.UNKNOWN), repr(e)
        trace["stop"] = "error"
    unknown = r["answer"].startswith("확인 불가") and r["citation"] == []
    sc = score_citation(r["citation"], q["gold"]) if q["gold"] else float(unknown)
    content, why = judge(q, r)
    return {"score": sc, "content": content, "content_reason": why,
            "stop": trace.get("stop"), "error": err, "response": r, "trace": trace}


def summarize(qs, per_q_runs, n_label):
    means = []
    for q, runs in zip(qs, per_q_runs):
        cite = [r["score"] for r in runs]
        cont = [r["content"] for r in runs if r.get("content") is not None]
        cm = sum(cite) / len(cite)
        tm = sum(cont) / len(cont) if cont else 0.0
        means.append((q, cm, tm))
        stops = Counter(r["stop"] for r in runs)
        flag = "  ← 흔들림" if len(set(cite)) > 1 or len(set(cont)) > 1 else ""
        print(f"{q['id']:3} 근거 {cm:.2f} 내용 {tm:.2f}  근거{cite} 내용{cont}  {dict(stops)}{flag}")
        for r in runs:
            if r.get("content") is not None and r["content"] < 1 and q["gold"]:
                print(f"      ✗ 내용: {r['content_reason'][:110]}")
                break
    gold = [(c, t) for q, c, t in means if q["gold"]]
    trap = [c for q, c, t in means if not q["gold"]]
    ng = max(len(gold), 1)
    print(f"\n[{n_label}] 근거 정확도 {sum(c for c, _ in gold) / ng:.3f}   "
          f"답변 내용 정확도 {sum(t for _, t in gold) / ng:.3f}   함정 정답률 {sum(trap) / max(len(trap), 1):.0%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("--only")
    ap.add_argument("--questions", default=str(config.ROOT / "sample_questions.json"))
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--judge", help="이미 돌린 diag 결과 파일에 내용 채점만 다시 합니다")
    a = ap.parse_args()

    if a.judge:
        report = json.loads(open(a.judge, encoding="utf-8").read())
        qs = [{k: x[k] for k in ("id", "question", "gold", "answer_key")} for x in report]
        jobs = [(q, r) for q, x in zip(qs, report) for r in x["runs"]]
        with ThreadPoolExecutor(a.workers) as ex:
            verdicts = list(ex.map(lambda j: judge(j[0], j[1]["response"]), jobs))
        for (q, r), (sc, why) in zip(jobs, verdicts):
            r["content"], r["content_reason"] = sc, why
        summarize(qs, [x["runs"] for x in report], f"재채점 {Path(a.judge).name}")
        Path(a.judge).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        return

    qs = json.loads(open(a.questions, encoding="utf-8").read())
    if a.only:
        keep = set(a.only.split(","))
        qs = [q for q in qs if q["id"] in keep]
    jobs = [q for q in qs for _ in range(a.n)]
    with ThreadPoolExecutor(a.workers) as ex:
        results = list(ex.map(run_once, jobs))
    per_q = [results[i * a.n:(i + 1) * a.n] for i in range(len(qs))]
    summarize(qs, per_q, f"{Path(a.questions).stem} {a.n}회")

    report = [{"id": q["id"], "question": q["question"], "gold": q["gold"], "answer_key": q["answer_key"],
               "runs": runs} for q, runs in zip(qs, per_q)]
    out = config.ROOT / "results"
    out.mkdir(exist_ok=True)
    path = out / f"diag_{Path(a.questions).stem}_{datetime.now():%m%d_%H%M%S}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("저장:", path)
    import upstage
    u = upstage.usage_snapshot()
    tot = {k: sum(v[k] for v in u.values()) for k in ("calls", "prompt_tokens", "completion_tokens")}
    print(f"사용량: 호출 {tot['calls']}회, 입력 {tot['prompt_tokens']:,} 토큰, 출력 {tot['completion_tokens']:,} 토큰 "
          f"(문항당 {tot['calls'] / max(len(jobs), 1):.1f}회) {u}")
    with open(out / "usage_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"time": datetime.now().isoformat(timespec="seconds"), "set": Path(a.questions).stem,
                            "n_runs": len(jobs), "usage": u}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
