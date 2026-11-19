"""표현 강건성 점검: 같은 질문을 어순·조사·말투만 바꿔 물었을 때 답이 흔들리는지 봅니다.

변형 4종 (뜻·숫자·고유명사는 그대로)
  reorder      어순 변경 (핵심 대상을 뒤로 보내거나 앞뒤 절을 바꿈)
  no_particle  조사 생략 (검색창에 치듯 단어 나열)
  casual       반말
  honorific    격식 있는 존댓말

실행:  python -X utf8 robustness.py gen --src test_set_3_500.json --n 5 --traps 10   (변형 생성 → results/robust_variants.json)
       python -X utf8 robustness.py run                                              (채점 → results/robust_<시각>.json)
"""
import argparse
import json
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import config

VARIANTS = ["reorder", "no_particle", "casual", "honorific"]
OUT = config.ROOT / "results" / "robust_variants.json"

GEN_SYSTEM = """너는 한국어 질문 변형기다. 주어진 질문의 뜻·조건·숫자·고유명사(조례명, 시설명, 금액, 나이 등)는 한 글자도 바꾸지 말고
표현만 바꾼 변형 4개를 만든다.
- reorder: 어순을 바꾼다. 핵심 대상이나 조건을 문장 뒤로 보내거나 절의 순서를 바꾼다. 말투는 원문과 비슷하게.
- no_particle: 조사·어미를 최대한 빼고 검색창에 치듯 핵심 단어만 나열한다. (예: "경차 공영주차장 할인 얼마")
- casual: 친구에게 묻듯 반말로 쓴다. (예: "~돼?", "~얼마야?")
- honorific: 격식 있는 존댓말로 쓴다. (예: "~인지 여쭙고자 합니다", "~하시는지 알려 주시겠습니까?")
새 정보나 조건을 추가하지 않는다."""

GEN_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "variants", "strict": True,
        "schema": {"type": "object",
                   "properties": {k: {"type": "string"} for k in VARIANTS},
                   "required": VARIANTS, "additionalProperties": False},
    },
}


def gen(a):
    import upstage
    qs = json.load(open(a.src, encoding="utf-8"))
    rnd = random.Random(a.seed)
    by_type = defaultdict(list)
    for q in qs:
        by_type[q["type"]].append(q)
    picked = []
    for t, items in sorted(by_type.items()):
        if items[0]["gold"]:
            picked += rnd.sample(items, min(a.n, len(items)))
    traps = [q for q in qs if not q["gold"]]
    picked += rnd.sample(traps, min(a.traps, len(traps)))

    def one(q):
        raw = upstage.chat([{"role": "system", "content": GEN_SYSTEM},
                            {"role": "user", "content": q["question"]}], response_format=GEN_SCHEMA)
        return {**q, "variants": {"original": q["question"], **json.loads(raw)}}

    with ThreadPoolExecutor(8) as ex:
        out = list(ex.map(one, picked))
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(out)}문항 × 5형태 → {OUT}")


def run(a):
    import rag
    from check import score_citation
    from diagnose import judge
    qs = json.load(open(OUT, encoding="utf-8"))
    jobs = [(q, k) for q in qs for k in ["original"] + VARIANTS]

    def one(job):
        q, k = job
        text = q["variants"][k]
        try:
            r = rag.answer(text)
        except Exception:  # noqa: BLE001
            r = dict(rag.UNKNOWN)
        unknown = r["answer"].startswith("확인 불가") and not r["citation"]
        row = {"id": q["id"], "type": q["type"], "form": k, "question": text, "response": r, "unknown": unknown}
        if q["gold"]:
            row["cite"] = score_citation(r["citation"], q["gold"])
            row["content"], row["content_reason"] = judge({**q, "question": q["question"]}, r)
        else:
            row["trap_ok"] = float(unknown)
        return row

    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(one, jobs))
    path = config.ROOT / "results" / f"robust_{datetime.now():%m%d_%H%M%S}.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    orig = {r["id"]: r for r in rows if r["form"] == "original"}
    print(f"| 질문 형태 | 근거 정확도 | 답변 정답률 | 과잉 확인 불가 | 원문과 같은 인용 | 함정 방어 |")
    print("|---|---|---|---|---|---|")
    for k in ["original"] + VARIANTS:
        g = [r for r in rows if r["form"] == k and "cite" in r]
        t = [r for r in rows if r["form"] == k and "trap_ok" in r]
        same = sum(set(r["response"]["citation"]) == set(orig[r["id"]]["response"]["citation"]) for r in g)
        cont = [r["content"] for r in g if r["content"] is not None]
        print(f"| {k} | {sum(r['cite'] for r in g) / len(g):.3f} | {sum(cont) / max(len(cont), 1):.3f} "
              f"| {sum(r['unknown'] for r in g)}/{len(g)} | {same}/{len(g)} "
              f"| {sum(r['trap_ok'] for r in t):.0f}/{len(t)} |")
    print("\n원문은 맞혔는데 변형에서 틀린 경우:")
    for r in rows:
        o = orig[r["id"]]
        if r["form"] == "original":
            continue
        if "cite" in r and (r["cite"] < o["cite"] or (r["content"] or 0) < (o["content"] or 0)):
            print(f"  {r['id']} [{r['form']}] 근거 {o['cite']:.2f}→{r['cite']:.2f} 내용 {o['content']}→{r['content']}"
                  f"{' (확인 불가)' if r['unknown'] else ''} | {r['question'][:70]}")
        if "trap_ok" in r and o["trap_ok"] and not r["trap_ok"]:
            print(f"  {r['id']} [{r['form']}] 함정이 새어 답함 | {r['question'][:70]} → {r['response']['answer'][:60]}")
    print("저장:", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen", "run"])
    ap.add_argument("--src", default="test_set_3_500.json")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--traps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    gen(a) if a.cmd == "gen" else run(a)
