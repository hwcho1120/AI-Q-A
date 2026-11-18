"""4단계: 검색 → Solar 답변 → 검증 → 대회 응답 형식.

설계 원칙 세 가지
  1. LLM은 근거 '번호'만 고릅니다. 조항 표기(citation)는 코드가 조각 라벨로 만듭니다.
  2. source_url · effective_date도 코드가 채웁니다. LLM이 링크나 날짜를 쓰게 두지 않습니다.
  3. 의심스러우면 '확인 불가'. 근거 없는 답은 맞아도 점수가 없고, 함정 문항에서는 감점입니다.
"""
import json
import math
import time as _time
import re
from collections import Counter

import numpy as np

import config

UNKNOWN = {"answer": "확인 불가", "citation": [], "source_url": "", "effective_date": ""}
DISTRICT = re.compile(r"서울특별시\s*\S+구\s")      # 'OO구' 자치구 조례 인용 방지
NAMED_LOCAL_GOV_PAT = re.compile(
    r"(?:강남|서초|송파|마포|성동|영등포|종로|용산|광진|동대문|중랑|성북|강북|도봉|노원|은평|서대문|양천|강서|구로|금천|동작|관악|강동)구"
    r"|(?:서울(?:특별시)?\s*|(?<=[\s,]))중구(?=[에의에서을를은는도과와,\s])"
    r"|수원시|성남시|고양시|용인시|부산시|인천시|대구시|대전시|광주시|울산시|세종시"
)
EXT_HARD_TRAP_PAT = re.compile(
    r"PC방|피시방|노래방|당구장|운전면허\s*벌점|면허\s*벌점|\b징역|징역형?|범칙금|형사\s*처벌|검찰|경찰(?:청|서|관|에)|법원(?:\s|을|에|의)"
)

_cache = {}


# ------------------------------------------------------------ 검색
def _bigrams(text):
    t = re.sub(r"\s+", "", text)
    return [t[i:i + 2] for i in range(len(t) - 1)]


def _load():
    unit = config.CHUNK_UNIT
    if unit in _cache:
        return _cache[unit]
    chunks = json.loads((config.DATA / f"chunks_{unit}.json").read_text(encoding="utf-8"))
    vec_path = config.DATA / f"vectors_{unit}.npy"
    vecs = np.load(vec_path) if vec_path.exists() else None
    if vecs is not None:
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    # 키워드 검색용 (글자 2개 단위 BM25)
    docs = [Counter(_bigrams(c["label"] + " " + c["text"])) for c in chunks]
    df = Counter(g for d in docs for g in d)
    avg = sum(sum(d.values()) for d in docs) / len(docs)
    for c in chunks:                       # 예전 색인 파일과의 호환
        c.setdefault("context", c["text"])
        c.setdefault("group", c["label"])
    _cache[unit] = (chunks, vecs, docs, df, avg)
    # 같은 조에 속한 항들 (조 → [대표 조각...]): 형제 항 확장용
    groups = {}
    for c in chunks:
        if not c.get("sub"):
            groups.setdefault(c["group"], []).append(c)
    _cache[unit + ":groups"] = groups
    return _cache[unit]


def _bm25(query, docs, df, avg, k1=1.2, b=0.75):
    n = len(docs)
    q = set(_bigrams(query))
    scores = np.zeros(n)
    for i, d in enumerate(docs):
        ln = sum(d.values())
        s = 0.0
        for g in q:
            if g in d:
                idf = math.log(1 + (n - df[g] + 0.5) / (df[g] + 0.5))
                s += idf * d[g] * (k1 + 1) / (d[g] + k1 * (1 - b + b * ln / avg))
        scores[i] = s
    return scores


REWRITE_SYSTEM = """너는 서울특별시 조례 검색어 생성기다.
시민의 일상 질문을 조례 조문에 실제로 쓰일 법률 용어로 바꾼 검색어를 2~3개 만든다.
- 쉬운 말을 조문 용어로 바꾼다. 예) 돈을 내다 → 요금·사용료 부과, 할인 → 감면, 술 → 음주·금주구역, 나이 → 정의·연령
- 둘 이상의 할인·감면을 함께 받는지 묻는 질문이면 '감면 사유가 둘 이상일 때 중복 감면' 검색어도 만든다.
- 금지 행위 질문이면 그 위반에 대한 '과태료 부과' 검색어도 만든다.
- 용어의 뜻·대상·범위를 묻는 질문이면 '정의' 검색어도 만든다.
- 답을 쓰지 말고 검색어만 만든다. 각 검색어는 20자 안팎.
pseudo_article: 이 질문에 답이 되는 조례 조문이 있다면 어떻게 쓰여 있을지, 조례 문체로 한두 문장을 지어 쓴다.
  (예: "소유자등은 ~할 때에는 다음 각 호의 사항을 준수하여야 한다", "시장은 ~에 해당하는 경우 ~를 감면할 수 있다")
  검색용일 뿐 답으로 쓰지 않으므로 금액·숫자는 넣지 않는다."""

REWRITE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "queries",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"queries": {"type": "array", "items": {"type": "string"}},
                           "pseudo_article": {"type": "string"}},
            "required": ["queries", "pseudo_article"],
            "additionalProperties": False,
        },
    },
}


def rewrite_queries(question):
    """질문 재작성: 쉬운 말 → 조문 용어 검색어 + 가상 조문(HyDE). 실패하면 빈 목록(원 질문만으로 검색)."""
    if not (config.USE_REWRITE and config.UPSTAGE_API_KEY):
        return []
    import upstage
    try:
        raw = upstage.chat([{"role": "system", "content": REWRITE_SYSTEM},
                            {"role": "user", "content": question}], response_format=REWRITE_SCHEMA)
        out = json.loads(raw)
        qs = [q.strip() for q in out.get("queries", []) if q.strip()][:3]
        if config.USE_HYDE and out.get("pseudo_article", "").strip():
            qs.append(out["pseudo_article"].strip()[:300])
        return qs
    except Exception:  # noqa: BLE001  재작성은 보조 단계라 실패해도 계속 진행
        return []


def _score(queries, chunks, vecs, docs, df, avg):
    """질문 여러 개(원 질문 + 재작성)의 검색 결과를 순위 기반(RRF)으로 합칩니다.

    점수 최댓값으로 합치면 한 검색어가 높은 점수를 몰아가 다른 검색어의 1~3위가 묻힙니다.
    순위로 합치면 각 검색어의 상위 결과가 고르게 들어옵니다. 같은 라벨(호 단위 보조 조각)은 한 번만 셉니다.
    """
    use_vec = vecs is not None and config.UPSTAGE_API_KEY
    if use_vec:
        import upstage
        qv = np.array(upstage.embed(queries, config.EMBED_QUERY), dtype=np.float32)
        qv = qv / np.linalg.norm(qv, axis=1, keepdims=True)
        cos_all = vecs @ qv.T                                   # (조각 수, 질문 수)
    by_label, rep = {}, {}                                      # 라벨별 합산 점수, 대표 조각 번호
    for j, q in enumerate(queries):
        kw = _bm25(q, docs, df, avg)
        kw = kw / kw.max() if kw.max() > 0 else kw
        s = config.W_EMBED * cos_all[:, j] + (1 - config.W_EMBED) * kw if use_vec else kw
        weight = 1.0 if j == 0 else config.REWRITE_WEIGHT       # 원 질문을 약간 우대
        rank, seen = 0, set()
        for i in np.argsort(s)[::-1][:config.RRF_DEPTH]:
            lab = chunks[i]["label"]
            if lab in seen:
                continue
            seen.add(lab)
            by_label[lab] = by_label.get(lab, 0.0) + weight / (config.RRF_K + rank)
            rep.setdefault(lab, i)
            rank += 1
    fused = np.zeros(len(chunks))
    for lab, sc in by_label.items():
        fused[rep[lab]] = sc
    best_cos = cos_all[:, 0] if use_vec else None               # 확인 불가 기준선은 원 질문 기준
    return fused, best_cos


# 조례 이름의 핵심어 → 질문에서 그 조례 대상을 직접 가리키는 표현
_LAW_KEYWORDS = {"도시공원": r"공원", "자전거": r"자전거", "청년": r"청년",
                 "금연": r"금연|흡연|담배", "동물보호": r"동물|반려|맹견|유기견|길고양이"}


def retrieve(question, top_k=config.TOP_K, queries=None):
    """임베딩(의미) + 키워드(용어) 혼합 검색 + 질문 재작성. 같은 라벨은 하나로 합칩니다."""
    chunks, vecs, docs, df, avg = _load()
    if queries is None:
        queries = rewrite_queries(question)
    score, cos = _score([question] + list(queries), chunks, vecs, docs, df, avg)
    ranked, seen = [], set()
    for i in np.argsort(score)[::-1]:
        if score[i] <= 0:
            break
        c = chunks[i]
        if c["label"] in seen:
            continue
        seen.add(c["label"])
        ranked.append({**c, "score": float(score[i]),
                       "cos": float(cos[i]) if cos is not None else None})
    if config.LAW_FOCUS and ranked:
        # 조례 집중: 상위권 점수 합이 1등 조례의 LAW_FOCUS_RATIO 이상인 조례만 남깁니다.
        # (관련 없는 조례 조항이 섞여 혼동되는 것을 막되, '맹견+어린이공원'처럼 두 조례가 다 그럴듯하면 둘 다 남김)
        votes = {}
        for h in ranked[:top_k * 2]:
            votes[h["law"]] = votes.get(h["law"], 0.0) + h["score"]
        top = max(votes.values())
        keep = {law for law, v in votes.items() if v >= top * config.LAW_FOCUS_RATIO}
        # 질문이 조례의 대상을 직접 말하면(예: '공원' → 도시공원 조례) 점수와 상관없이 남깁니다.
        # ('공원에 반려동물 공간' 질문이 동물보호 조례로만 쏠려 도시공원 조례가 빠지던 문제)
        keep |= {law for law in votes if any(k in law and re.search(p, question)
                                                 for k, p in _LAW_KEYWORDS.items())}
        ranked = [h for h in ranked if h["law"] in keep]
    return ranked[:top_k]


def expand_siblings(hits, max_total=None):
    """상위 근거와 같은 조에 속한 다른 항도 근거 후보로 덧붙입니다.

    예) 제7조 제1항(감면 사유)이 걸리면 제7조 제3항(중복 감면 금지)도 함께 보이게.
    """
    max_total = max_total or config.MAX_CONTEXT
    groups = _cache.get(config.CHUNK_UNIT + ":groups", {})
    out = list(hits)
    seen = {h["label"] for h in hits}
    for h in hits[:config.SIBLING_FROM_TOP]:
        for s in groups.get(h["group"], []):
            if len(out) >= max_total:
                return out
            if s["label"] not in seen:
                seen.add(s["label"])
                out.append({**s, "score": 0.0, "cos": None, "sibling_of": h["label"]})
    return out


# 조문 안의 같은 조례 참조: "제5조제4항", "제5조 제1항", "제1항"(같은 조 안). 「법」·법·영·규칙 조문은 제외
_REF = re.compile(r"(제\d+조(?:의\d+)?)?\s?제(\d+)항|(제\d+조(?:의\d+)?)(?![의\d])")
_EXTERNAL = re.compile(r"(」|법|영|규칙|시행령|시행규칙)$")   # 바로 앞이 다른 법령이면 제외


def _refs(chunk):
    """조각 본문이 가리키는 같은 조례의 조·항 라벨 후보를 돌려줍니다."""
    body = chunk.get("context", chunk["text"]).split("\n", 1)[-1]
    group = chunk.get("group", "")
    law = chunk.get("law", "")
    out = []
    prev_end, prev_external = -100, False
    for m in _REF.finditer(body):
        before = body[max(0, m.start() - 8):m.start()].rstrip()
        gap = body[prev_end:m.start()].strip() if m.start() - prev_end < 8 else None
        # 바로 앞이 다른 법령이거나, 다른 법령 조문 목록에 '및·,·또는'으로 이어진 참조면 제외
        external = bool(_EXTERNAL.search(before)) or (prev_external and gap in ("및", ",", "ㆍ", "또는", ""))
        prev_end, prev_external = m.end(), external
        if external:
            continue
        art, para, art_only = m.groups()
        if para:
            base = f"{law} {art}" if art else group      # "제1항"만 있으면 같은 조
            out.append(f"{base} 제{para}항")
        elif art_only:
            out.append(f"{law} {art_only}")               # 조 전체 (항 없는 조이거나 조 단위 참조)
    return out


def expand_refs(hits, max_total=None, hops=2):
    """근거 조문이 '제○조제○항에 따라'처럼 가리키는 같은 조례 조항을 덧붙입니다.

    예) 과태료 조항(제10조 제1항: '제5조제4항을 위반하여') → 제5조 제4항 → 제5조 제1항(금연구역 목록)
    질문의 대상이 실제로 적용 범위에 드는지 모델이 확인할 수 있게 하기 위함입니다.
    """
    max_total = max_total or config.MAX_CONTEXT
    groups = _cache.get(config.CHUNK_UNIT + ":groups", {})
    by_label = {c["label"]: c for g in groups.values() for c in g}
    out = list(hits)
    seen = {h["label"] for h in hits}
    frontier = hits[:config.SIBLING_FROM_TOP]
    for _ in range(hops):
        nxt = []
        for h in frontier:
            for ref in _refs(h):
                cands = [by_label[ref]] if ref in by_label else groups.get(ref, [])
                for c in cands[:config.REF_MAX_PER]:          # 조 전체 참조는 앞 항 몇 개만
                    if len(out) >= max_total:
                        return out
                    if c["label"] not in seen:
                        seen.add(c["label"])
                        item = {**c, "score": 0.0, "cos": None, "ref_of": h["label"]}
                        out.append(item)
                        nxt.append(item)
        frontier = nxt
    return out


_ANNEX_OF = re.compile(r"\((제\d+조(?:의\d+)?)\s*(?:제?(\d+)항)?\s*관련\)")       # 별표 제목의 '(제9조 제1항관련)'
_ANNEX_REF = re.compile(r"별표\s*(\d*)")
_DEF_TERM = re.compile(r'"([^"]{1,20})"(?:이)?(?:란|라 함은|이라 함은)')


def _links():
    """색인에서 미리 계산하는 연결: 별표 ↔ 관련 본문 조항(양방향), 조례별 정의 조항과 정의된 용어."""
    key = config.CHUNK_UNIT + ":links"
    if key in _cache:
        return _cache[key]
    _load()
    groups = _cache[config.CHUNK_UNIT + ":groups"]
    labels = {c["label"] for g in groups.values() for c in g}
    links, defs = {}, {}
    for g in groups.values():
        for c in g:
            law, lab = c["law"], c["label"]
            head = c.get("context", c["text"]).split("\n", 1)[0]
            body = c.get("context", c["text"])
            if " 별표" in lab:                                   # 별표 → 관련 본문
                m = _ANNEX_OF.search(head.replace(" ", "")) or _ANNEX_OF.search(head)
                if m:
                    art, para = m.group(1), m.group(2)
                    tgt = f"{law} {art} 제{para}항" if para else f"{law} {art}"
                    cands = [tgt] if tgt in labels else [x["label"] for x in groups.get(f"{law} {art}", [])][:2]
                    for t in cands:
                        links.setdefault(lab, []).append(t)
                        links.setdefault(t, []).append(lab)       # 본문 → 별표 (역방향)
            else:
                for no in _ANNEX_REF.findall(body.split("\n", 1)[-1]):  # 본문이 '별표 1'을 언급
                    tgt = f"{law} 별표 {no}" if no else f"{law} 별표"
                    if tgt in labels:
                        links.setdefault(lab, []).append(tgt)
                if "(정의)" in head:                             # 정의 조항
                    terms = set(_DEF_TERM.findall(body))
                    if terms:
                        d = defs.setdefault(law, {"labels": [], "terms": set()})
                        d["labels"].append(lab)
                        d["terms"] |= terms
    _cache[key] = (links, defs)
    return links, defs


def expand_links(hits, question, max_total=None):
    """색인에서 미리 계산한 연결로 근거를 보강합니다.
    ① 별표가 걸리면 그 별표를 정한 본문 조항을, 본문이 걸리면 그 조항이 가리키는 별표를 덧붙입니다.
       (요금표만 보고 '원칙은 무료' 같은 본문 조건을 놓치는 것을 막기 위함)
    ② 질문에 조례가 정의한 용어(예: 등록대상동물, 청년)가 나오면 그 조례의 정의 조항을 덧붙입니다.
    """
    max_total = max_total or config.MAX_CONTEXT
    links, defs = _links()
    groups = _cache[config.CHUNK_UNIT + ":groups"]
    by_label = {c["label"]: c for g in groups.values() for c in g}
    out, seen = list(hits), {h["label"] for h in hits}

    def add(lab, why):
        if lab in by_label and lab not in seen and len(out) < max_total:
            seen.add(lab)
            out.append({**by_label[lab], "score": 0.0, "cos": None, "link_of": why})

    for h in hits[:config.SIBLING_FROM_TOP]:
        for lab in links.get(h["label"], []):
            add(lab, h["label"])
    q = re.sub(r"\s+", "", question)
    for law in dict.fromkeys(h["law"] for h in hits[:config.SIBLING_FROM_TOP]):
        d = defs.get(law)
        if d and any(re.sub(r"\s+", "", t) in q for t in d["terms"]):
            for lab in d["labels"]:
                add(lab, "정의")
    return out


QTYPES = ["값 조회", "해당 여부", "원칙과 예외", "절차·권한"]
ANSWER_FRAMES = """- qtype을 먼저 고르고, 유형별 순서대로 sentences를 쓴다.
  · 값 조회(금액·기간·나이·비율 등): ① 값 ② 그 값이 적용되는 조건(급지·대상·단위) ③ 단서가 있으면 단서. 계산이 필요하면 ①을 "원문 값 × … = 결과" 계산식으로 쓴다.
  · 해당 여부(내가/이 경우 적용되나): ① 결론(해당한다/해당하지 않는다) ② 판단 기준이 된 조건 ③ 단서·예외.
  · 원칙과 예외(감면 중복, 원칙 무료+예외 유료 등): ① 원칙 ② 예외·단서 ③ 질문 상황에 적용한 결과.
  · 절차·권한(누가, 어떻게, 언제까지): ① 주체와 할 일 ② 기한·방법 ③ 세부가 위임돼 있으면 위임 사실."""


# ------------------------------------------------------------ 답변
SYSTEM = """너는 서울특별시 조례 질의응답 도우미다.
규칙:
- 반드시 아래 [근거]에 적힌 내용만으로 답한다. 근거에 없는 내용은 추측하거나 덧붙이지 않는다.
- conclusion 필드는 sentences를 다 쓴 뒤에, 첫 문장이 내린 결론을 요약해 고른다. 질문이 예·아니요로 답할 질문("~할 수 있나요/되나요/받을 수 있나요/내야 하나요")이면
  질문이 묻는 그대로에 대한 답을 고른다. 예) "둘 다 받을 수 있나요?"→받을 수 없으면 "아니요", "돈을 내야 하나요?"→원칙적으로 무료면 "아니요".
  "얼마/몇 살/어떻게 되나요"처럼 값·결과를 묻는 질문이면 "해당없음".
- 답변은 sentences에 한 문장씩 나눠 쓴다. 각 문장의 evidence에는 그 문장의 내용이 실제로 적힌 근거 조항 이름(<근거 조항="..."> 값)을 고른다.
  문장에 쓴 숫자(금액·비율·나이·기간)는 그 문장의 evidence 조항 원문에 그대로 있어야 한다. 계산한 값이면 계산식(예: 400원 × 12)을 함께 쓴다.
  계산이 필요한 질문은 원문 값 → 계산식 → 결과 순서로, 계산 과정이 보이게 한 문장에 쓴다.
- 질문이 구체적인 수치(대수·인원·금액·기간·비율)를 주고 결과나 기준 충족 여부를 물으면, 근거의 기준·비율을 그 수치에 적용해
  계산한다. 결과값이 원문에 그대로 없어도 기준이 근거에 있으면 basis는 "있음"이다. 첫 문장에 최종 결과(예: 200대, 13명,
  기준 미달로 위반)를 쓰고, 계산식은 "500대 × 40/100 = 200대"처럼 숫자·×·÷·+·−·= 로 쓴다(과반수는 "12명의 과반수 = 7명"). 여러 인원·품목·시간 구간(기본 시간 + 초과 시간)을 합산할 때는 각 항목 단가·가산·할인을 표에서 하나씩 확인해 합산 결과를 검산하고, '2시간당'·'1구획당'처럼 기본 요금 단위가 묶음 기준이면 그 묶음 요금을 시간 수로 다시 곱하지 않는다.
- 근거 조항이 '제○조제○항을 준용한다'고 적고 [근거]에 그 준용 대상 조항이 함께 있으면, 준용 대상 조항의 요건·단서·예외까지 그대로 적용해 결론을 내린다.
- 답변할 때 '~제○조를 봐야 합니다'처럼 조항 번호 안내로만 끝내지 말고, 그 근거 조항이 실제로 정한 구체적 내용(대상 포함 여부, 기준·수치·별표/별도 명칭 등)을 직접 풀어서 답한다.
- 근거 조항에 있는 비슷한 제도(예: 다자녀 주차요금 감면)를 질문이 묻는 전혀 다른 제도(예: 출산 축하 포인트 적립)로 억지 연결하지 않는다. 질문의 핵심 제도 자체가 조례에 없으면 basis는 "없음", found는 false로 한다.
- 질문이 둘 이상을 물으면("~와 ~를 각각", "또한 ~는?") 물음마다 답하는 문장을 빠짐없이 쓴다(문장 수 제한보다 우선).
- 첫 문장은 "네"나 "아니요"로 시작하지 말고 결론(값·결과·가능 여부)을 바로 쓴다. (앞머리 "네,/아니요,"는 시스템이 붙인다)
- 이어서 결론을 뒷받침하는 조건·숫자(금액·비율·나이)와 '다만' 같은 단서·예외를 쓴다. 질문과 관계없는 절차·위임·이의제기 설명은 쓰지 않는다.
- 일반 시민이 이해할 수 있는 쉬운 말로 2~4문장. 숫자는 근거에 적힌 값 그대로 쓴다.
- 근거 조항이 세부 기준·방법을 '시장이 따로 정한다', '규칙으로 정한다'고 위임했으면 그 사실을 한 문장으로 알려 준다.
- 질문이 부정형("~안 되나요?", "못 ~나요?", "~안 해도 되나요?")이면 네/아니요가 헷갈리므로 conclusion은 "해당없음"으로 하고,
  첫 문장에 "~할 수 있습니다/~해야 합니다"처럼 결론을 긍정문으로 분명히 쓴다.
- 질문이 선택형("A인가요, 아니면 B인가요?")이면 conclusion은 "해당없음"으로 하고, 첫 문장에 어느 쪽인지
  (A인지, B인지, 둘 다인지, 둘 다 아닌지)를 바로 쓴다. 예) "카드 소지자와 카드에 등재된 가족 모두 면제 대상입니다."
  단, 근거가 A·B 중 어느 쪽인지 정하지 않았으면(지급 방식·서류·절차 같은 세부가 근거에 없으면) basis는 "없음"이다.
- '다만', 괄호( ) 안의 단서는 원칙에 붙는 조건이다. 단서가 적용 범위를 넓히는지('~도 포함한다', '~를 포함') 좁히는지
  ('~는 제외한다', '~에 한정한다', '그러하지 아니하다') 정확히 읽고, 결론을 원칙과 단서를 함께 적용한 결과로 쓴다.
  예) "A 소지자는 면제(다만, B 시설은 C 소지자를 포함)" → B 시설에서도 A 소지자는 면제이고, C 소지자도 추가로 면제.
- 감면·혜택이 둘 이상 겹치는 질문이면, 근거에 중복 적용을 정한 조항이 있는지 먼저 확인하고 그 조항에 따라 답한다.
- 다른 조례의 조항을 끌어와 섞지 않는다. 질문이 묻는 조례의 근거만 쓴다.
- 결론 문장의 evidence에는 결론을 직접 정한 조항을 맨 앞에 둔다. 답 전체에서 쓰는 조항은 결론에 꼭 필요한 1~3개로 줄이고, 부수 설명용 조항은 쓰지 않는다.
- 질문의 장소·대상·상황에 어떤 규정(과태료·감면·금지 등)이 적용된다고 답하려면, 그 장소·대상이 근거 조항의 적용 범위
  (지정 대상 목록, 감면 대상 목록 등)에 실제로 적혀 있어야 한다. 적혀 있지 않으면 적용된다고 추측하지 말고 found를 false로 한다.
- 근거로 질문에 답할 수 없으면(질문한 규정·금액이 근거에 없으면) found를 false로 한다.
- 자치구(OO구) 조례나 규칙은 근거로 쓰지 않는다.
- basis 필드는 답을 쓰기 전에 먼저 정한다. 질문의 핵심(묻는 제도·행위·대상자·금액·기간·절차)이 근거에 정해져 있는가?
  · "있음": 질문이 묻는 그 대상이 근거 조항에 정해져 있다. 세부 기준·방법을 '시장이 따로 정한다', '규칙으로 정한다'고
    위임했더라도 제도·대상 자체가 조항에 있으면 "있음"이다. 이때 첫 문장은 "~ 제도는 조례 제○조에 있고, 세부 기준은 ~가 따로 정합니다"처럼
    제도가 있다는 것부터 쓴다 ("정해져 있지 않습니다"로 시작하지 않는다).
    조례가 그 행위를 금지·제한한다고 정해 둔 경우도 "있음"이다 (근거 있는 '아니요').
  · "없음": 질문의 핵심이 근거 어디에도 정해져 있지 않고, 주변 주제의 조항만 있다.
    "조례에는 ~에 관한 규정이 없습니다"라는 답이 될 수밖에 없다면 "없음"이다. 이때 주변 조항으로 설명하지 않는다.
    질문이 여러 가지를 묻는데 핵심 질문의 답이 없고 부수적인 것만 있으면 "없음"이다.
  · 질문의 대상 B가 조항에 나오지 않을 때는 질문 형태로 판단한다.
    - B에 대한 값·절차·방법(얼마, 몇, 어떻게, 어디서)을 묻는데 조항에 B가 없으면 "없음".
      ("조항은 A만 정하므로 B는 해당하지 않습니다" 같은 추론으로 답하지 않는다)
    - B가 해당되는지(예/아니요)를 묻고, 조항이 기준(나이·면적·대수)이나 대상 목록을 명시해 B가 그 밖에 있음이
      분명하면 "있음"(근거 있는 '아니요')이다. 조항이 B를 명시적으로 제외·한정·금지한 경우도 "있음".
""" + (ANSWER_FRAMES if config.USE_ANSWER_FRAMES else "") + """
결과는 JSON으로 낸다."""

REPAIR_SYSTEM = SYSTEM + """

[추가 지시] 직전 답변이 검증에서 '근거에 없는 내용이 있다'고 지적됐다. 검증은 직전 답변의 문장별 evidence로 고른 근거만 보고 이뤄졌다.
- 근거를 잘못 골랐을 수 있으니, 전체 [근거]에서 답의 내용이 실제로 적힌 조항을 다시 확인해 고른다.
- 지적된 문장은 고쳐 쓰지 말고 삭제한다. 새로운 주장(예: '~는 적용되지 않는다')을 덧붙이지 않는다.
- 그래도 질문에 답할 수 없으면 found를 false로 한다."""

def _schema(hits):
    """응답 스키마. evidence는 이번에 보여 준 근거의 조항 이름 중에서만 고를 수 있습니다(enum).

    번호로 고르게 하면 하나씩 밀려 고르는 실수가 반복돼서, 조항 이름 자체를 선택지로 줍니다.
    목록 밖의 조항은 스키마가 막으므로 지어낸 조항이 나올 수 없습니다.
    """
    labels = list(dict.fromkeys(h["label"] for h in hits)) or ["(없음)"]
    schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "law_answer",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "basis": {"type": "string", "enum": ["있음", "없음"]},   # 답을 쓰기 전에 먼저 판단
                    "found": {"type": "boolean"},
                    "sentences": {"type": "array", "items": {
                        "type": "object",
                        "properties": {"text": {"type": "string"},
                                       "evidence": {"type": "array", "items": {"type": "string", "enum": labels}}},
                        "required": ["text", "evidence"],
                        "additionalProperties": False}},
                    # 결론은 답변 문장을 다 쓴 뒤에 고릅니다 (먼저 고르면 나중에 쓴 본문과 어긋나는 경우가 있었음)
                    "conclusion": {"type": "string", "enum": ["네", "아니요", "해당없음"]},
                },
                "required": ["basis", "found", "sentences", "conclusion"],
                "additionalProperties": False,
            },
        },
    }
    if config.USE_ANSWER_FRAMES:            # 유형별 답변 틀: 유형을 먼저 고르게 함 (측정 결과 점수가 내려가 기본값은 끔)
        props = schema["json_schema"]["schema"]["properties"]
        schema["json_schema"]["schema"]["properties"] = {"basis": props["basis"], "found": props["found"],
                                                         "qtype": {"type": "string", "enum": QTYPES},
                                                         "sentences": props["sentences"], "conclusion": props["conclusion"]}
        schema["json_schema"]["schema"]["required"] = ["basis", "found", "qtype", "sentences", "conclusion"]
    return schema


def _context(hits):
    """근거마다 경계를 분명히 표시합니다."""
    return "\n\n".join(f'<근거 조항="{h["label"]}">\n{h.get("context", h["text"])}\n</근거>' for h in hits)


def _ask(system, user, hits, effort=None):
    import upstage
    out = json.loads(upstage.chat([{"role": "system", "content": system},
                                   {"role": "user", "content": user}], response_format=_schema(hits),
                                  reasoning_effort=(effort or config.ANSWER_REASONING) or None))
    return _assemble(out)


def _decision(o):
    """답의 '판단'만 뽑습니다: 근거 유무, 결론 근거 조항. ('해당없음'과 '네'처럼 방향이 반대가 아닌 차이는 무시)"""
    if o.get("basis") == "없음" or not o.get("found"):
        return ("없음", None)
    return ("있음", (o.get("evidence") or [None])[0])


def _ask_consistent(system, user, hits, trace):
    """같은 질문에 답을 N개 동시에 만들어 판단을 다수결로 정합니다 (자기 일관성).
    동시에 호출하므로 응답 시간은 답 1개와 비슷합니다. 판단이 갈린 것 자체도 trace에 남깁니다."""
    n = config.CONSISTENCY_N
    if n < 2:
        return _ask(system, user, hits)
    import time as _t
    from collections import Counter
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
    # 첫 답이 나오면 나머지는 CONSISTENCY_GRACE_SEC 초만 더 기다립니다 (가장 느린 답을 기다리지 않음 → 응답 시간 상한)
    ex = ThreadPoolExecutor(n)
    futs = [ex.submit(_ask, system, user, hits) for _ in range(n)]
    done, pending = wait(futs, return_when=FIRST_COMPLETED)
    deadline = _t.time() + config.CONSISTENCY_GRACE_SEC
    while pending and _t.time() < deadline:
        more, pending = wait(pending, timeout=max(0.0, deadline - _t.time()), return_when=FIRST_COMPLETED)
        done |= more
    ex.shutdown(wait=False, cancel_futures=True)   # 늦은 답은 버림
    outs = []
    for f in done:
        try:
            outs.append(f.result())
        except Exception:  # noqa: BLE001  일부 실패해도 나머지로 진행
            pass
    if not outs:                                   # 먼저 끝난 것이 전부 실패했으면 하나를 끝까지 기다림
        for f in pending:
            try:
                outs.append(f.result())
                break
            except Exception:  # noqa: BLE001
                pass
    if not outs:
        raise RuntimeError("답변 생성 전부 실패")
    trace["consistency_n"] = len(outs)
    decisions = [_decision(o) for o in outs]
    # 네/아니요가 서로 반대인 경우도 판단 불일치로 봅니다
    pol = {o.get("conclusion") for o in outs if o.get("conclusion") in ("네", "아니요")}
    trace["consistency"] = [list(d) + [o.get("conclusion")] for d, o in zip(decisions, outs)]
    top, cnt = Counter(decisions).most_common(1)[0]
    trace["consistency_agree"] = cnt == len(outs) and len(pol) <= 1
    winners = [o for o, d in zip(outs, decisions) if d == top]
    # 결론(네/아니요/해당없음)도 다수결로 고릅니다
    major = Counter(o.get("conclusion") for o in winners).most_common(1)[0][0]
    return next(o for o in winners if o.get("conclusion") == major)


def _assemble(out):
    """문장별 답변을 합쳐 answer / evidence(문장 순서대로 중복 제거)를 만듭니다."""
    sents = [x for x in out.get("sentences", []) if x.get("text", "").strip()]
    out["sentences"] = sents
    out["answer"] = " ".join(x["text"].strip() for x in sents)
    out["evidence"] = list(dict.fromkeys(lab for x in sents for lab in x.get("evidence", [])))
    return out


# ---- 코드 검사: 판정과 답변의 어긋남, 문장별 숫자 대조 (LLM 판단이 아닌 결정론적 검사)
_NO_RULE = re.compile(r"(규정|기준|조항|내용|정함|정해|명시|나와|언급|근거|적혀|수치)[^.。]{0,15}?(없습니다|없어|않습니다|않아|않고)")
_CANNOT = re.compile(r"수\s?(?:는|도)?\s?없")
_ARTREF = re.compile(r"제\s?\d+\s?(?:조(?:의\s?\d+)?|항|호)|별표\s?\d*")
_NUM = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(만|천)?")
_CALC = re.compile(r"[×x*=+/÷]|합계|계산|곱하|나누|더하")


_DELEGATED = re.compile(r"위임|(?:따로|별도로|규칙으로|시장이|구청장이)\s*정(?:한|하|해\s*두|해지)")


# '기준에 해당하지 않습니다'처럼 규정을 적용한 결론(해당·적용·충족 여부)은 규정 부재가 아닙니다.
_APPLY_NEG = re.compile(r"해당하지|해당되지|적용되지|적용하지|충족하지|포함되지|속하지|대상이 아니")


# 첫 절에서 이미 적극적 규정('~해야/할 수 있/규정되어 있...')을 답한 뒤 부수 항목이 없음을 적는 경우는 규정 부재에서 제외
_AFFIRM_CLAUSE = re.compile(r"(?:하여야|해야|할\s*수\s*있|규정되어\s*있|적용되|부과|지급|지원|면제|감면|납부)")


def _claims_no_rule(out):
    """첫 문장이 '~규정이 없습니다'류이면 모델 판정(basis)과 상관없이 '근거 없음'으로 봅니다.
    단, '세부는 ~가 따로 정한다(위임)'는 답이나 앞 절에서 적극적 규정을 먼저 설명한 경우는 제외합니다."""
    first = (out.get("sentences") or [{}])[0].get("text", "")
    # '~기준으로 달마다 받을 수 없습니다'처럼 '할 수 없다'(불가 결론)는 규정 부재가 아니므로 제외합니다.
    hit = any(
        not _CANNOT.search(m.group(0))
        and not _APPLY_NEG.search(m.group(0))
        and not ("," in first[:m.start()] and _AFFIRM_CLAUSE.search(first[:m.start()].rsplit(",", 1)[0]))
        for m in _NO_RULE.finditer(first)
    )
    return hit and not _DELEGATED.search(out.get("answer", "") or first)


# ---- 계산 검사: 답변에 쓴 계산식(A × B = C)을 코드가 직접 계산해 결과를 대조합니다.
_EXPR_TOK = re.compile(
    r"\s*(?:(?P<frac>(?P<den>\d[\d,]*(?:\.\d+)?)\s*분의\s*(?P<numer>\d[\d,]*(?:\.\d+)?))"
    r"|(?P<num>\d[\d,]*(?:\.\d+)?)\s*(?P<mul>억|만|천)?\s*"
    r"(?P<unit>퍼센트|%|원|대|명|개월|일|년|회|㎡|제곱미터|배|킬로미터|km|시간|분|세|살|건|개|구획)?"
    r"|(?P<op>[×xX*÷/+\-−])|(?P<eq>=)|(?P<par>[()]))")
_MUL = {"억": 1e8, "만": 1e4, "천": 1e3}


def _f(s):
    return float(s.replace(",", ""))


def _expr_runs(text):
    """문장에서 '숫자·연산자·='가 이어진 구간을 [(토큰 문자열 리스트 by '=')...]로 뽑습니다."""
    runs, cur, i = [], [], 0
    while i < len(text):
        m = _EXPR_TOK.match(text, i)
        if not m or m.end() == i or not any(m.group(k) for k in ("frac", "num", "op", "eq", "par")):
            if cur:
                runs.append(cur)
                cur = []
            i += 1
            continue
        if m.group("frac"):
            cur.append(f"({_f(m.group('numer'))}/{_f(m.group('den'))})")
        elif m.group("num"):
            v = _f(m.group("num")) * _MUL.get(m.group("mul") or "", 1)
            if m.group("unit") in ("%", "퍼센트"):
                v /= 100
            cur.append(repr(v))
        elif m.group("op"):
            cur.append({"×": "*", "x": "*", "X": "*", "÷": "/", "−": "-"}.get(m.group("op"), m.group("op")))
        else:
            cur.append(m.group("eq") or m.group("par"))
        i = m.end()
    if cur:
        runs.append(cur)
    out = []
    for run in runs:
        if "=" not in run:
            continue
        parts, p = [], []
        for t in run:
            if t == "=":
                parts.append(p)
                p = []
            else:
                p.append(t)
        parts.append(p)
        out.append(parts)
    return out


def _eval(tokens):
    s = "".join(tokens)
    if not s or not re.fullmatch(r"[\d.e+\-*/() ]+", s) or not re.search(r"\d", s):
        return None
    try:
        return float(eval(s, {"__builtins__": {}}, {}))  # noqa: S307  숫자·사칙연산만 통과한 문자열
    except Exception:  # noqa: BLE001
        return None


_DATE_LIT = re.compile(r"(?:\d{2,4}\s*년\s*)?\d{1,2}\s*월\s*\d{1,2}\s*일")


def _calc_check(text):
    """(틀린 계산 [(식, 쓴 값, 올바른 값)], 맞게 계산된 값 집합)"""
    wrong, ok = [], set()
    # 날짜 표현(예: '6월 1일 + 7일 = 6월 8일')은 '월'에서 끊겨 '1+7=6'으로 오판하지 않도록 가립니다.
    cleaned = _DATE_LIT.sub(" ", text)
    for parts in _expr_runs(cleaned):
        vals = [_eval(p) for p in parts]
        for a, b, pa in zip(vals, vals[1:], parts):
            if a is None or b is None:
                continue
            if abs(a - b) <= max(abs(a), 1) * 0.005:
                ok.update((a, b))
            elif len(pa) > 1:                  # 식 → 결과가 어긋남 (단일 숫자끼리의 '='는 판단하지 않음)
                wrong.append(("".join(pa), b, a))
    return wrong, ok


_CONSTS = {2.0, 10.0, 12.0, 100.0, 1000.0, 365.0}


def _derivable(v, pool, qvals):
    """질문의 수치와 근거 수치로 사칙연산 1~2번(두 번째는 질문 수치·상수와)이면 나오는 값인가. 과반수도 허용."""
    key = lambda x: round(x, 6)  # noqa: E731
    P = {key(x) for x in pool | _CONSTS}
    S = P | {key(x // 2 + 1) for x in pool}           # 과반수

    def one(t):
        if key(t) in S:
            return True
        for a in P:
            for c in (t - a, a - t, t + a, t / a if a else None, t * a, a / t if t else None):
                if c is not None and key(c) in P:
                    return True
        return False
    if one(v):
        return True
    for q in qvals | _CONSTS:
        for c in (v - q, v + q, q - v, v / q if q else None, v * q):
            if c is not None and one(c):
                return True
    return False


def _values(text):
    vals = set()
    for n, unit in _NUM.findall(_ARTREF.sub(" ", text)):
        try:
            v = float(n.replace(",", ""))
        except ValueError:
            continue
        vals.add(v * {"만": 10000, "천": 1000}.get(unit, 1))
        vals.add(v)
    return vals


def _number_issues(out, hits, question):
    """문장마다 숫자가 그 문장이 고른 근거 원문(없으면 고른 근거 전체)에 있는지 대조합니다."""
    by_label = {h["label"]: h for h in hits}
    all_used = " ".join(by_label[l].get("context", "") for l in out.get("evidence", []) if l in by_label)
    allowed_q = _values(question)
    used_vals = _values(all_used)
    issues, calc_ok = [], set()
    sents = out.get("sentences", [])
    for i, snt in enumerate(sents):                   # ① 계산식은 코드가 직접 계산해 대조
        wrong, ok = _calc_check(snt["text"])
        calc_ok |= ok
        for expr, got, right in wrong:
            fmt = lambda x: f"{x:,.0f}" if abs(x - round(x)) < 1e-9 else f"{x:,.4g}"  # noqa: E731
            issues.append((i, snt["text"], got, f"계산이 틀렸다: {expr} = {fmt(right)} (답변은 {fmt(got)})"))
    # 질문이 구체적인 수치를 주면(계산·판단형) 그 수치(1명·2명 등 및 '각 N' 2배수 포함)와 근거 값으로 계산되는 결과도 허용합니다.
    scenario = allowed_q | {2 * v for v in allowed_q if v <= 10}
    for i, snt in enumerate(sents):                   # ② 숫자 대조
        text = snt["text"]
        if _CALC.search(text):
            continue                          # 계산식을 밝힌 문장은 ①에서 계산을 확인함
        src = " ".join(by_label[l].get("context", "") for l in snt.get("evidence", []) if l in by_label)
        src_vals = _values(src) | used_vals              # 문장 근거 + 같은 답변이 고른 근거 전체
        for v in _values(text):
            if v in allowed_q or v <= 1 or v in src_vals or v in calc_ok:
                continue
            if v == 2 and 1.0 in src_vals and re.search(r"2\s*(?:개|명|회|건|차례|가지|곳|종)\s*(?:이상|초과|부터)", text):
                continue
            if scenario and _derivable(v, used_vals | allowed_q | calc_ok, scenario):
                continue
            issues.append((i, text, v, None))
    return issues


_YESNO_HEAD = re.compile(r"^\s*(네|예|아니요|아니오)\s*[,.]\s*")


_NEG_Q = re.compile(r"(?:^|\s)(?:안|못)(?:\s+\S|해|하면|되|돼|받|가도|써도)|않(?:아|나요|을까|는)|없(?:나요|을까|어요\?|죠)")


# 선택형 질문: "A인가요, 아니면 B인가요?" → 네/아니요가 어느 쪽에 대한 답인지 모호
_CHOICE_Q = re.compile(r"아니면|혹은")


def _no_yesno(question):
    """네/아니요 앞머리를 붙이지 않을 질문: 부정형('못 받나요?')과 선택형('A인가요, 아니면 B인가요?')."""
    return bool(question) and bool(_NEG_Q.search(question) or _CHOICE_Q.search(question))


def _finalize(out, question=""):
    """앞머리 '네,/아니요,'는 모델이 따로 고른 conclusion으로 코드가 붙입니다 (본문과 모순 방지).
    부정형·선택형 질문에는 네/아니요가 뜻을 헷갈리게 하므로 붙이지 않습니다."""
    text = _YESNO_HEAD.sub("", out["answer"]).strip()
    c = out.get("conclusion")
    if _no_yesno(question):
        return text
    return f"{c}, {text}" if c in ("네", "아니요") else text


def _verify_ctx(out, hits):
    """검증용 근거: 답변 문장들이 고른 근거 전체 (출력용 citation의 개수 제한을 적용하지 않음)."""
    by_label = {h["label"]: h for h in hits}
    return [by_label[l] for l in out.get("evidence", []) if l in by_label and not DISTRICT.search(l)]


def _pick(out, hits):
    by_label = {h["label"]: h for h in hits}
    used = [by_label[lab] for lab in out.get("evidence", []) if lab in by_label]
    used = [h for h in used if not DISTRICT.search(h["label"])]
    if used:
        # 첫 근거(결론 근거)와 같은 조례만 인용합니다. 다른 조례 조항이 섞이는 것을 막고 인용 수를 제한합니다.
        law = used[0]["law"]
        used = [h for h in used if h["law"] == law]
        used = list({h["label"]: h for h in used}.values())[:config.MAX_CITATIONS]
    return used


_SAME_ART_PARA = re.compile(r"(?<![조\d])(?<!조 )제\s?(\d+)\s?항")
_PARA_LABEL = re.compile(r"^(.*제\d+조(?:의\d+)?) 제(\d+)항$")


def _with_sibling_refs(citation):
    """인용한 항의 본문이 같은 조의 다른 항을 가리키면('제1항에 따라', '제2항의') 그 항도 인용합니다.
    해당 항이 적용 범위·대상을 정하는 경우가 많아, 둘을 함께 내야 근거가 완전합니다."""
    _load()
    groups = _cache[config.CHUNK_UNIT + ":groups"]
    by_label = {c["label"]: c for g in groups.values() for c in g}
    out = list(citation)
    for lab in citation:
        m = _PARA_LABEL.match(lab)
        if not m or lab not in by_label:
            continue
        body = by_label[lab].get("context", "").split("\n", 1)[-1]
        for n in dict.fromkeys(_SAME_ART_PARA.findall(body)):
            ref = f"{m.group(1)} 제{n}항"
            if len(out) >= config.CITE_SIBLING_MAX_TOTAL:
                return out
            if ref != lab and ref in by_label and ref not in out:
                out.append(ref)
    return out


def _with_annex_parent(citation):
    """별표를 인용했으면 그 별표를 정한 본문 조항(별표 제목의 '(제33조제1항 관련)')도 뒤에 덧붙입니다.
    별표는 본문 조항이 위임한 표이므로 둘 다 근거입니다. 인용 개수 상한(MAX_CITATIONS) 안에서만 추가합니다."""
    links, _ = _links()
    out = list(citation)
    for lab in citation:
        if " 별표" not in lab:
            continue
        for parent in dict.fromkeys(links.get(lab, [])):
            if len(out) >= config.MAX_CITATIONS:
                return out
            if " 별표" not in parent and parent not in out:
                out.append(parent)
                break
    return out


def answer(question, trace=None):
    """trace에 dict를 넘기면 단계별 진단 정보를 채웁니다 (어디서 '확인 불가'가 났는지 분석용)."""
    import upstage
    if trace is None:
        trace = {}
    t_start = _time.time()
    if config.USE_VERIFY and (NAMED_LOCAL_GOV_PAT.search(question) or EXT_HARD_TRAP_PAT.search(question)):
        trace["stop"] = "min_score"
        return dict(UNKNOWN)
    queries = rewrite_queries(question)
    trace["queries"] = queries
    top = retrieve(question, queries=queries)

    # 검증 ①: 검색 단계에서 이미 관련 조문이 없어 보이면 바로 '확인 불가'
    if config.USE_VERIFY and top and top[0]["cos"] is not None and top[0]["cos"] < config.MIN_SCORE:
        trace["stop"] = "min_score"
        return dict(UNKNOWN)

    hits = expand_links(expand_refs(expand_siblings(top)), question)
    trace["hits"] = [(h["label"], round(h["score"], 3), h["cos"] and round(h["cos"], 3)) for h in hits]
    user_msg = f"[근거]\n{_context(hits)}\n\n[질문]\n{question}"
    out = _ask_consistent(SYSTEM, user_msg, hits, trace)
    trace["llm"] = out
    used = _pick(out, hits)
    if out.get("basis") == "없음" or _claims_no_rule(out):
        trace["stop"] = "no_basis"              # 질문의 핵심이 조례에 없음 → 주변 조항으로 설명하지 않고 확인 불가
        return dict(UNKNOWN)
    if not out.get("found") or not used:
        trace["stop"] = "found_false" if not out.get("found") else "no_evidence"
        return dict(UNKNOWN)

    # 검증 ②-a (코드): 문장별 숫자가 그 문장의 근거 원문에 있는가
    # 검증 ②-b (LLM): 답이 고른 근거에 기반하는가 (Groundedness)
    if config.USE_VERIFY:
        def check(o):
            """코드 숫자 대조 → 통과하면 LLM 문장별 검증. (verdict, reason, 근거 없는 문장 번호)"""
            ni = _number_issues(o, hits, question)
            if ni:
                return ("notGrounded", "; ".join(
                    msg or f"다음 문장의 숫자가 근거 원문에 없다: '{t[:80]}'의 {v:g}" for _, t, v, msg in ni),
                    sorted({i for i, _, _, _ in ni}))
            c = o.get("conclusion")
            prefix = f"{c}, " if c in ("네", "아니요") and not _no_yesno(question) else ""
            return upstage.groundedness_sentences(_context(_verify_ctx(o, hits)),
                                                  [_YESNO_HEAD.sub("", x["text"]) for x in o["sentences"]],
                                                  question, prefix)

        def drop(o, bad):
            o["sentences"] = [x for i, x in enumerate(o["sentences"]) if i not in set(bad)]
            _assemble(o)
            trace["dropped_sentences"] = trace.get("dropped_sentences", 0) + len(bad)

        verdict, reason, bad = check(out)
        trace["verdict"], trace["reason"], trace["unsupported"] = verdict, reason, bad
        if verdict == "notGrounded" and bad and 0 not in bad and len(bad) < len(out["sentences"]):
            # 결론 문장(첫 문장)은 근거가 있음 → 근거 없는 문장만 코드가 지웁니다 (전체 재작성 없이)
            drop(out, bad)
            verdict = "grounded_after_drop"
        elif verdict == "notGrounded" and config.USE_REPAIR and _time.time() - t_start < config.REPAIR_BUDGET_SEC:
            # 결론 문장 자체가 문제 → 한 번만 고쳐 쓰기
            out = _ask(REPAIR_SYSTEM, f"{user_msg}\n\n[직전 답변]\n{out['answer']}\n\n[검증 지적]\n{reason}", hits)
            trace["repair"] = out
            if out.get("basis") == "없음" or _claims_no_rule(out) or not out.get("found") or not out["sentences"]:
                trace["stop"] = "repair_found_false"
                return dict(UNKNOWN)
            verdict, reason, bad = check(out)
            trace["verdict2"], trace["reason2"], trace["unsupported2"] = verdict, reason, bad
            if verdict == "notGrounded" and bad and 0 not in bad and len(bad) < len(out["sentences"]):
                drop(out, bad)
                verdict = "grounded_after_drop"
        used = _pick(out, hits)
        if verdict == "notGrounded" or not used:   # notSure는 통과 (함정은 basis/found 단계에서 거름)
            trace["stop"] = "not_grounded"
            return dict(UNKNOWN)

    trace["sentences"] = [{"text": x["text"], "evidence": x.get("evidence", [])} for x in out["sentences"]]
    trace["stop"] = "answered"
    citation = list(dict.fromkeys(h["label"] for h in used))   # 순서 유지, 중복 제거
    if config.CITE_ANNEX_PARENT:
        citation = _with_annex_parent(citation)
    if config.CITE_SIBLING_REFS:
        citation = _with_sibling_refs(citation)
    first = used[0]
    return {"answer": _finalize(out, question), "citation": citation,
            "source_url": first["source_url"], "effective_date": first["effective_date"]}


def answer_without_rag(question):
    """시연용 '전' 장면: 검색 없이 LLM만으로 답하게 하면 어떻게 되는가."""
    import upstage
    return upstage.chat([
        {"role": "system", "content": "너는 서울시 조례 전문가다. 답과 함께 근거 조항(제○조 제○항)과 원문 링크, 시행일을 적어라."},
        {"role": "user", "content": question}], temperature=0.3)
