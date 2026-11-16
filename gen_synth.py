"""합성 평가셋 생성: 조항 하나를 주고 '이 조항이 답인 시민 질문'을 Solar가 만들게 합니다.

정답 조항은 생성에 쓴 조항으로 자동으로 정해집니다.
  - 조례 6종에서 조례별로 고르게 표본 추출 (삭제 조문·아주 짧은 조문 제외)
  - 유형(단순 조회/조건 판단/여러 조항 결합)을 돌아가며 지정, 조문 단어를 그대로 쓰지 않게 지시
  - 함정 문항은 6종 조례 범위 밖 주제로 따로 생성
한계: 다른 조항으로도 답할 수 있는 질문이 섞일 수 있음(정답 애매) → 결과 해석 시 감안

실행:  python -X utf8 gen_synth.py -n 200 --traps 30 --seed 7
결과:  synth_questions.json
"""
import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor

import config
import upstage

GEN_SYSTEM = """너는 서울시 조례 Q&A 평가 문항 출제자다. 주어진 [조항] 하나가 정답 근거가 되는 시민의 질문을 하나 만든다.
- 질문은 일반 시민이 일상에서 할 법한 말투로 쓴다. 조문에 쓰인 단어를 그대로 옮기지 말고 쉬운 말로 바꾼다.
- 질문만 보고도 이 조항이 정답임을 알 수 있어야 한다(같은 조례의 다른 조항으로는 답할 수 없게 구체적으로).
- 지정된 유형을 따른다: 단순 조회(값·기간·대상을 묻기), 조건 판단(내 상황에 적용되는지 예/아니요로 묻기), 여러 조항 결합(원칙과 단서·예외를 함께 알아야 답할 수 있게).
- answer_key에는 조항에 근거한 정답 요지를 한 줄로 쓴다. 숫자는 조항 그대로.
- 조항이 시민 질문으로 만들기 어려운 내부 행정 절차라도, 공무원·관계자가 물을 법한 질문으로 만든다."""

TRAP_SYSTEM = """너는 서울시 조례 Q&A 평가의 '함정 문항' 출제자다. 아래 6종 서울특별시 조례 어디에도 근거가 없는 질문을 만든다.
대상 조례: 주차장 설치 및 관리 / 동물보호 / 자전거이용 활성화 / 도시공원 / 청년 기본 / 금연환경 조성 및 간접흡연 피해방지.
함정 종류를 지정된 대로 따른다:
- 범위 밖 주제: 위 조례들과 비슷해 보이지만 국가 법률(도로교통법·국민건강증진법 등)이나 다른 서울시 조례에 있는 내용
- 자치구: 'OO구 조례·규칙'에 따른 내용을 묻는 질문
- 없는 값: 위 조례 주제이지만 조례에 정해지지 않은 구체적 금액·기간·횟수를 묻는 질문
질문은 시민 말투로, 그럴듯하게."""


def _schema(name, props):
    return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": {
        "type": "object", "properties": props, "required": list(props), "additionalProperties": False}}}


GEN_SCHEMA = _schema("gen", {"question": {"type": "string"}, "answer_key": {"type": "string"}})
TRAP_SCHEMA = _schema("trap", {"question": {"type": "string"}, "why_no_basis": {"type": "string"}})
TYPES = ["단순 조회", "조건 판단", "여러 조항 결합"]
TRAP_KINDS = ["범위 밖 주제", "자치구", "없는 값"]


def pick_targets(n, seed):
    chunks = json.loads((config.DATA / "chunks_paragraph.json").read_text(encoding="utf-8"))
    main = {}
    for c in chunks:
        if c.get("sub") or len(c["context"]) < 60 or "삭제" in c["context"][-12:]:
            continue
        main.setdefault(c["label"], c)                  # 라벨당 하나 (별표 조각 중복 제거)
    by_law = {}
    for c in main.values():
        by_law.setdefault(c["law"], []).append(c)
    rnd = random.Random(seed)
    for v in by_law.values():
        rnd.shuffle(v)
    picked, i = [], 0
    while len(picked) < n and any(by_law.values()):    # 조례별로 돌아가며 하나씩 (고르게)
        law = list(by_law)[i % len(by_law)]
        if by_law[law]:
            picked.append(by_law[law].pop())
        i += 1
    return picked


# 2차 유형: 1차(단순·조건·결합)에서 다루지 않은 질문 형태
TYPES_V2 = {
    "계산형": "조항(특히 요금표·비율)의 값으로 시간·횟수·인원을 곱하거나 할인율을 적용해야 답이 나오는 질문. answer_key에 계산식 포함.",
    "비교형": "조항 안의 두 대상(급지·연령·시설·종류)의 차이나 어느 쪽이 더 큰지 묻는 질문.",
    "경계값": "조항의 나이·면적·기간·대수 기준의 바로 위나 바로 아래 값에 해당하는 사람·상황이 적용되는지 묻는 질문. 기준 밖이면 정답은 '아니요'.",
    "부정형": "'~안 해도 되나요?', '~하면 안 되나요?', '~못 받나요?'처럼 부정형으로 묻는 질문.",
    "긴 사연형": "3~4문장의 개인 사정을 길게 설명한 뒤 마지막에 질문하는 형태. 사연 속에 관련 없는 정보도 섞는다.",
    "구어·오타형": "반말·줄임말·맞춤법 실수(예: 되요, 몇일)가 섞인 짧은 메신저 말투 질문.",
    "위임형": "조항이 세부를 '시장이 따로 정한다/규칙으로 정한다'고 위임한 부분의 구체 내용을 묻는 질문. 정답은 제도의 존재와 위임 사실.",
    "공무원 업무형": "구청·시청 담당 공무원이 업무 처리 중 확인할 법한 기한·절차·권한·구성 질문.",
}
NEAR_TRAP = ("이 조항과 주제는 거의 같지만 핵심 요소 하나(대상·장소·금액·행위)를 조항에 없는 것으로 바꿔서, "
             "이 조례로는 답할 수 없게 만든 질문. 예를 들어 조항이 A 대상만 정하면 조항에 없는 B 대상을 묻는다.")

CHECK_SYSTEM = """너는 평가 문항 검수자다. [조항]과 [질문]을 보고 판정한다.
- answerable: 이 조항만으로 질문에 답할 수 있는가 (근접 함정 문항이면 답할 수 없어야 정상)
- ambiguous: 같은 조례의 다른 조항이 더 직접적인 답일 가능성이 큰가, 또는 질문이 여러 해석이 가능한가"""
CHECK_SCHEMA = _schema("check", {"answerable": {"type": "boolean"}, "ambiguous": {"type": "boolean"}})


def check_one(q, c):
    """정답이 애매한 문항을 걸러냅니다. 통과하면 True."""
    try:
        raw = upstage.chat([{"role": "system", "content": CHECK_SYSTEM},
                            {"role": "user", "content": f"[조항] {c['label']}\n{c['context'][:2500]}\n\n[질문]\n{q['question']}"}],
                           response_format=CHECK_SCHEMA)
        out = json.loads(raw)
        want = not q["type"].startswith("근거 없음")
        return out["answerable"] == want and not out["ambiguous"]
    except Exception:  # noqa: BLE001
        return False


def gen_one(args, types=None, prefix="Q", check=False):
    k, c = args
    if types:
        names = list(types) + ["근접 함정"]
        qtype = names[k % len(names)]
        how = types.get(qtype, NEAR_TRAP)
    else:
        qtype, how = TYPES[k % len(TYPES)], ""
    try:
        raw = upstage.chat([{"role": "system", "content": GEN_SYSTEM},
                            {"role": "user", "content": f"[유형] {qtype} {how}\n[조항] {c['label']}\n{c['context'][:2500]}"}],
                           response_format=GEN_SCHEMA, temperature=0.8)
        out = json.loads(raw)
        trap = qtype == "근접 함정"
        q = {"id": f"{prefix}{k:03d}", "type": "근거 없음 (함정)" if trap else qtype, "subtype": qtype,
             "question": out["question"], "gold": [] if trap else [[c["label"]]],
             "answer_key": "확인 불가 (근접 함정)" if trap else out["answer_key"], "source": "synth",
             "from": c["label"]}
        if check and not check_one(q, c):
            return None
        return q
    except Exception as e:  # noqa: BLE001
        print("생성 실패", c["label"], e)
        return None


def gen_trap(k):
    kind = TRAP_KINDS[k % len(TRAP_KINDS)]
    try:
        raw = upstage.chat([{"role": "system", "content": TRAP_SYSTEM},
                            {"role": "user", "content": f"[함정 종류] {kind}\n질문 하나를 만들어라. (다양하게, 번호 {k})"}],
                           response_format=TRAP_SCHEMA, temperature=1.0)
        out = json.loads(raw)
        return {"id": f"X{k:03d}", "type": "근거 없음 (함정)", "question": out["question"], "gold": [],
                "answer_key": f"확인 불가 ({kind}: {out['why_no_basis'][:60]})", "source": "synth"}
    except Exception as e:  # noqa: BLE001
        print("함정 생성 실패", e)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=200, help="전체 문항 수 (함정 포함)")
    ap.add_argument("--traps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(config.ROOT / "synth_questions.json"))
    ap.add_argument("--v2", action="store_true", help="2차 유형(계산·비교·경계값·부정·사연·구어·위임·업무·근접 함정)")
    ap.add_argument("--check", action="store_true", help="정답이 애매한 문항을 검수해 걸러냄")
    ap.add_argument("--prefix", default="Q")
    a = ap.parse_args()
    targets = pick_targets(a.n - a.traps, a.seed)
    types = TYPES_V2 if a.v2 else None
    with ThreadPoolExecutor(a.workers) as ex:
        qs = list(ex.map(lambda t: gen_one(t, types, a.prefix, a.check), enumerate(targets)))
        traps = list(ex.map(gen_trap, range(a.traps))) if a.traps else []
    out = [q for q in qs + traps if q]
    print(f"검수 탈락 {len(targets) - sum(1 for q in qs if q)}개 | 사용량 {upstage.usage_snapshot()}")
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    from collections import Counter
    print("생성", len(out), dict(Counter(q["type"] for q in out)))
    print("조례별", dict(Counter(q["gold"][0][0].split(" 제")[0].split(" 별")[0][6:] for q in out if q["gold"])))


if __name__ == "__main__":
    main()
