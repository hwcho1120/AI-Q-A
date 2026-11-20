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

# 자치구(25개 구)·타 시·도 지명 감지 (거주지 언급 + 서울시 조례 명시 질의는 정상 통과)
_OTHER_LOCAL_GOV = re.compile(
    r"(?:강남|서초|송파|마포|성동|영등포|종로|용산|광진|동대문|중랑|성북|강북|도봉|노원|은평|서대문|양천|강서|구로|금천|동작|관악|강동)구"
    r"|(?:서울(?:특별시)?\s*|(?<=[\s,]))중구(?=[에의에서을를은는도과와,\s]|$)"
    r"|(?:해운대|수영|부산진|동래|사하|연제|수성|달서|유성|연수|남동|부평|계양|미추홀|분당|수정|중원|일산동|일산서|덕양|팔달|영통|장안|권선|기흥|수지|처인)구"
    r"|(?:부산|인천|대구|대전|광주|울산|세종|수원|성남|고양|용인|부천|안양|화성|안산|남양주|평택|김포|파주|시흥|광명|하남|군포|오산|의정부|구리|과천|의왕|양주|포천|여주|이천|안성|춘천|원주|강릉|청주|충주|제천|천안|아산|서산|당진|공주|논산|전주|군산|익산|정읍|목포|여수|순천|광양|나주|포항|구미|경주|안동|김천|경산|창원|김해|진주|양산|거제|통영|사천|밀양|제주|서귀포)(?:광역|특별자치)?시"
    r"|경기도|강원(?:특별자치)?도|충청[남북]도|전라[남북]도|전북특별자치도|경상[남북]도|제주(?:특별자치)?도"
    r"|(?<![가-힣])(?:충북|충남|전북|전남|경북|경남)(?=[에의에서을를은는도과와\s]|$)"
)
_LOCAL_GOV_TARGET = re.compile(
    r"(?:강남|서초|송파|마포|성동|영등포|종로|용산|광진|동대문|중랑|성북|강북|도봉|노원|은평|서대문|양천|강서|구로|금천|동작|관악|강동|중)구\s*"
    r"(?:의\s*|에서\s*(?:운영|시행|정하|관리|지원|부과)\S*\s*)?"
    r"(?:조례|규칙|공영|노상|노외|부설|거주자\s*우선|주차장|공원|도시공원|동물|유기|반려|자전거|청년\s*기본|금연|과태료|요금|입장료|사용료|이용료|수수료)"
)
_SEOUL_EXPLICIT = re.compile(
    r"서울대공원|서울공공자전거|따릉이"
    r"|서울(?:특별시|시)?\s*(?:조례|주차장|동물보호|자전거|도시공원|청년|금연|공영|노상|노외|금연구역|반려견|가\s*관리|에서\s*운영|에서\s*시행|상\s)"
)
_GU_BARE_OR_OWNED = re.compile(
    r"구립\s*(?:공영\s*)?(?:주차장|공원|놀이터|보호센터|보호소|도서관|체육|시설)"
    r"|우리\s*(?:동네\s*)?구(?:청)?\s*(?:의|에서\s*(?:운영|관리|시행)\S*)?\s*(?:공영\s*)?(?:주차장|주차\s*요금|공원|조례|규칙)"
    r"|(?<![가-힣])(?:강남|서초|송파|마포|성동|영등포|종로|용산|광진|동대문|중랑|성북|강북|도봉|노원|은평|서대문|양천|강서|구로|금천|동작|관악|강동)\s*"
    r"(?:쪽|지역|근처|동네|일대)?\s*(?:에\s*있는\s*)?(?:공영\s*)?(?:노상|노외)?\s*(?:주차장|주차\s*요금|공원)"
)


_EXT_CRIMINAL_TRAP = re.compile(
    r"경범죄\s*처벌법|형법|형사\s*처벌|경찰(?:에|로)\s*(?:바로\s*)?(?:송치|입건|고발)|검찰\s*송치|\b징역|징역형|운전면허\s*벌점|면허\s*벌점|범칙금"
)


def _is_other_local_gov(question):
    """특정 자치구·타 지자체의 조례·시설·요금·정책이나 조례 밖 형사처벌·범칙금을 묻는 질문인지 판별합니다.
    질문자가 거주지나 배경만 언급하고 서울시 조례·시설을 명시적으로 묻는 경우는 통과시킵니다."""
    if _EXT_CRIMINAL_TRAP.search(question) and not re.search(r"서울(?:특별시|시)?\s*조례", question):
        return True
    if _GU_BARE_OR_OWNED.search(question) and not _SEOUL_EXPLICIT.search(question):
        return True
    if not _OTHER_LOCAL_GOV.search(question):
        return False
    if _LOCAL_GOV_TARGET.search(question):
        return True
    return not _SEOUL_EXPLICIT.search(question)


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
- 질문에 두 가지 이상의 쟁점·시설·조건(예: '또한 ~는?', 'A와 B를 함께 고려할 때', '전용주차구획 + 할인', '입장료 + 이용료')이 섞여 있으면, 각 쟁점·조항별로 서로 다른 검색어를 분리해 만든다.
- '따릉이(공공자전거)' 대여·이용요금 질문은 '공공자전거 이용요금 시장 위임'으로 검색어를 만들고, '자전거주차장 요금표' 검색어는 만들지 않는다.
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
_LAW_KEYWORDS = {
    "도시공원": r"공원|서울대공원|야영장|잔디광장|입장료|도시공원위원회|행정\(?2\)?부시장",
    "자전거": r"자전거|활성화계획|따릉이",
    "청년": r"청년",
    "금연": r"금연|흡연|담배",
    "동물보호": r"동물|반려|맹견|유기견|길고양이",
    "주차장": r"주차장|부설주차장|노상주차장|노외주차장|주차구획|주차요금",
}


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
    상위 SIBLING_FROM_TOP 외에, 같은 조의 항이 hits에 2개 이상 걸린 핵심 조도 나머지 항(제1항 등)을 보강합니다.
    """
    max_total = max_total or config.MAX_CONTEXT
    groups = _cache.get(config.CHUNK_UNIT + ":groups", {})
    out = list(hits)
    seen = {h["label"] for h in hits}
    grp_counts = Counter(h["group"] for h in hits)
    target_groups = list(dict.fromkeys(
        [h["group"] for h in hits[:config.SIBLING_FROM_TOP]]
        + [h["group"] for h in hits if grp_counts[h["group"]] >= 2]
    ))
    for grp in target_groups:
        for s in groups.get(grp, []):
            if len(out) >= max_total:
                return out
            if s["label"] not in seen:
                seen.add(s["label"])
                out.append({**s, "score": 0.0, "cos": None, "sibling_of": grp})
    return out


# 조문 안의 같은 조례 참조: "제5조제4항", "제5조 제1항", "제1항"(같은 조 안). 「법」·법·영·규칙 조문과 개정이력([제26조에서 이동])은 제외
_REF = re.compile(r"(제\d+조(?:의\d+)?)?\s?제(\d+)항|(제\d+조(?:의\d+)?)(?![의\d])")
_EXTERNAL = re.compile(r"(」|법|영|규칙|시행령|시행규칙)$")   # 바로 앞이 다른 법령이면 제외
_AMEND_HIST = re.compile(r"\[(?:제\d+조(?:의\d+)?에서\s*이동|종전\s*제\d+조|본조신설|전문개정|제목개정)[^\]]*\]|<개정[^>]*>|<신설[^>]*>")


def _refs(chunk):
    """조각 본문이 가리키는 같은 조례의 조·항 라벨 후보를 돌려줍니다."""
    body = _AMEND_HIST.sub(" ", chunk.get("context", chunk["text"]).split("\n", 1)[-1])
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


_REV_REF_CUE = re.compile(r"에도\s*불구하고|준용|위반하여|과태료|감면|면제|제외|예외|아니하다|제한")


def _rev_refs():
    """역방향 참조 인덱스: target 조항을 예외·단서·과태료·준용으로 가리키는 같은 조례 조항 목록."""
    key = config.CHUNK_UNIT + ":rev_refs"
    if key in _cache:
        return _cache[key]
    _load()
    groups = _cache.get(config.CHUNK_UNIT + ":groups", {})
    by_label = {c["label"]: c for g in groups.values() for c in g}
    rev = {}
    for lab, c in by_label.items():
        body = c.get("context", c["text"]).split("\n", 1)[-1]
        if not _REV_REF_CUE.search(body):
            continue
        for ref in _refs(c):
            targets = [ref] if ref in by_label else [x["label"] for x in groups.get(ref, [])[:2]]
            for tgt in targets:
                if tgt != lab:
                    rev.setdefault(tgt, [])
                    if lab not in rev[tgt]:
                        rev[tgt].append(lab)
    _cache[key] = rev
    return rev


def expand_refs(hits, max_total=None, hops=2):
    """근거 조문이 가리키는 조항(정방향)과 근거 조문을 예외·과태료·준용으로 가리키는 조항(역방향)을 함께 덧붙입니다.

    예) 과태료 조항(제10조 제1항: '제5조제4항을 위반하여') → 제5조 제4항 → 제5조 제1항(금연구역 목록)
    예) 원칙 조항(제1항) ← 예외·단서 조항('제1항에도 불구하고…')
    """
    max_total = max_total or config.MAX_CONTEXT
    groups = _cache.get(config.CHUNK_UNIT + ":groups", {})
    by_label = {c["label"]: c for g in groups.values() for c in g}
    rev = _rev_refs()
    out = list(hits)
    seen = {h["label"] for h in hits}
    frontier = hits[:config.SIBLING_FROM_TOP]
    for hop in range(hops):
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
            if hop == 0:                                      # 1-hop 역방향(예외·과태료·준용) 보강
                for src_lab in rev.get(h["label"], [])[:2]:
                    if len(out) >= max_total:
                        return out
                    if src_lab in by_label and src_lab not in seen:
                        seen.add(src_lab)
                        item = {**by_label[src_lab], "score": 0.0, "cos": None, "rev_ref_of": h["label"]}
                        out.append(item)
                        nxt.append(item)
        frontier = nxt
    return out


_ANNEX_OF = re.compile(r"\(제(\d+)조?(?:의(\d+))?\s*(?:제?(\d+)항)?\s*관련\)")       # 별표 제목의 '(제9조 제1항관련)' 및 오타 '(제21제2항 관련)' 포함
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
                    art = f"제{m.group(1)}조" + (f"의{m.group(2)}" if m.group(2) else "")
                    para = m.group(3)
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
  기준 미달로 위반)를 쓰고, 계산식은 "500대 × 40/100 = 200대"처럼 숫자·×·÷·+·−·= 로 쓴다(과반수는 "12명의 과반수 = 7명"). 여러 인원·품목·시간 구간(기본 시간 + 초과 시간)을 합산할 때는 각 항목 단가·가산·할인을 표에서 하나씩 확인해 합산 결과를 검산한다. 요금표 기본 단위가 '2시간' 등 N시간 묶음이면 '시간당 이용료'는 반드시 기본요금 ÷ N으로 구하고 그 묶음 요금을 시간 수로 다시 곱하지 않으며, '야간 가산(토·공휴일 제외)' 같은 괄호 제외 단서는 먼저 적용 여부를 가린 뒤 가산한다. 시간제 환산 요금과 1일 상한 요금이 함께 있으면 둘을 비교해 최종 납부액을 결론에 쓴다.
- 경계값과 기간 기산을 정확히 지킨다: '이상·이하·이내'는 경계값을 포함하고, '초과·미만'은 경계값을 제외한다(예: '6세 미만' 면제에 10세는 미해당). 시작일 포함 기간 기산 시 'N월 1일부터 7일 이상'은 시작일을 1일째로 세어 7일째인 N월 7일(1 + 7 - 1 = 7일)에 기간이 충족된다.
- 근거 조항 본문 안에 상위 법령 조문(예: '「국민건강증진법」 제9조의5 및 같은 법 시행령 제16조의5', '법 제36조')이 적혀 있으면, 답변 본문에서 그 상위 법령 근거도 생략하지 말고 함께 밝힌다(단, evidence에는 서울시 조례 조항만 고른다).
- 근거 조항이 '제○조제○항을 준용한다'고 적고 [근거]에 그 준용 대상 조항이 함께 있으면, 준용 대상 조항의 요건·단서·예외까지 그대로 적용해 결론을 내린다. 또한 근거 조항이나 별표가 다른 법령·시행규칙·고시·공고를 준용하거나 따르도록 정한 경우에는 그 위임·준용을 정한 이 조례의 조항(또는 별표)을 evidence로 고르고, 어느 법령·공고를 따르는지 안내한다.
- 답변할 때 '~제○조를 봐야 합니다'처럼 조항 번호 안내로만 끝내지 말고, 그 근거 조항이 실제로 정한 구체적 내용(대상 포함 여부, 기준·수치·별표/별도 명칭 등)을 직접 풀어서 답한다.
- 근거 끝에 '(질문과 관련된 호: …)'가 표시돼 있으면 그 호부터 우선 확인하되, 최종 판단은 근거 본문 전체를 보고 내린다.
- 근거 조항에 있는 비슷한·상위 제도(예: 다자녀 주차요금 감면, 일반 사진·영상 촬영 대관 및 이행보증금)를 질문이 묻는 조례에 없는 특정 대상·제도(예: 출산 축하 포인트 적립, 드론 촬영 허가 보증금)로 억지 연결하지 않는다. 질문의 핵심 대상·제도가 조례에 없으면 basis는 "없음", found는 false로 한다.
- 질문이 둘 이상을 물으면("~와 ~를 각각", "또한 ~는?") 물음마다 답하는 문장을 빠짐없이 쓰고, 각 문장의 evidence에 그 물음을 직접 정한 조항을 맨 앞에 넣는다(문장 수 제한보다 우선).
- 질문이 특정 항목들(예: A·B·C)의 기준·동일 여부를 물으면 질문한 항목들만 정확히 비교해 '동일한 기준(예: 모두 1년 이내)이 적용됩니다'처럼 직접 결론 내리고, 조문에 같이 적힌 묻지 않은 다른 항목(D)이나 다른 위원회·다른 제도·다른 시설 규정을 끌어와 '동일하지 않다'고 결론을 뒤집거나 불필요하게 덧붙이지 않는다.
- '공공자전거(따릉이)' 대여·이용요금(제12조의2·제12조의3)과 '자전거주차장' 주차요금(별표, 제9조: 1시간 200원, 1일 1,000원, 월 15,000원)은 서로 다른 제도이므로, 따릉이 이용권·정기권 요금 질문에 자전거주차장 별표·제9조를 절대 적용·인용하지 않는다. 대신 조례 제12조의2 제9항이 '이용요금 등 필요한 사항은 시장이 따로 정한다'고 위임하고 있으므로 반드시 basis="있음", found=true로 하여 제12조의2 제9항을 인용하고 시장이 따로 정함을 안내한다.
- 첫 문장은 "네"나 "아니요"로 시작하지 말고 결론(값·결과·가능 여부)을 바로 쓴다. (앞머리 "네,/아니요,"는 시스템이 붙인다)
- 이어서 결론을 뒷받침하는 조건·숫자(금액·비율·나이)와 '다만' 같은 단서·예외를 쓴다. 질문과 관계없는 절차·위임·이의제기 설명은 쓰지 않는다.
- 일반 시민이 이해할 수 있는 쉬운 말로 2~4문장. 숫자는 근거에 적힌 값 그대로 쓴다.
- 질문이 묻는 사항(세부 기준·운영시간·절차·금액 등) 자체가 '시장이 따로 정한다', '규칙으로 정한다'고 위임돼 있으면 그 사실을 한 문장으로 알려 준다. 단, 조항이 지원 항목·대상·요건을 각 호로 한정해 열거('그 밖에 ~가 정하는 사항' 없이)했고 다른 항이 '절차와 방법'만 위임한 경우에는, 조례상 지원 항목·대상은 열거된 것들로 한정된다고 명확히 답한다('절차·방법 위임'을 들어 항목 자체가 늘어날 수 있다고 추측하거나 결론을 흐리지 않는다).
- 질문이 부정형("~안 되나요?", "못 ~나요?", "~안 해도 되나요?")이면 네/아니요가 헷갈리므로 conclusion은 "해당없음"으로 하고,
  첫 문장에 "~할 수 있습니다/~해야 합니다"처럼 결론을 긍정문으로 분명히 쓴다.
- 질문이 선택형("A인가요, 아니면 B인가요?")이면 conclusion은 "해당없음"으로 하고, 첫 문장에 어느 쪽인지
  (A인지, B인지, 둘 다인지, 둘 다 아닌지)를 바로 쓴다. 예) "카드 소지자와 카드에 등재된 가족 모두 면제 대상입니다."
  단, 근거가 A·B 중 어느 쪽인지 정하지 않았으면(지급 방식·서류·절차 같은 세부가 근거에 없으면) basis는 "없음"이다.
- '다만', 괄호( ) 안의 단서는 원칙에 붙는 조건이다. 단서가 적용 범위를 넓히는지('~도 포함한다', '~를 포함') 좁히는지
  ('~는 제외한다', '~에 한정한다', '그러하지 아니하다') 정확히 읽고, 결론을 원칙과 단서를 함께 적용한 결과로 쓴다.
  특히 법령의 이중부정 단서('~하여야 한다 / ~하지 못한다 / ~면하지 못한다. 다만, X의 경우는 그러하지 아니하다')는 앞 문장 전체의 주된 서술어(결론)를 정반대로 뒤집는다(중간의 '제외하고는' 절만 뒤집는 것이 아니다). 예) "A를 증명한 경우를 제외하고는 손해배상 책임을 면하지 못한다. 다만, X의 경우는 그러하지 아니하다" → 주된 서술어('책임을 면하지 못한다')를 부정하므로 X의 경우에는 A 입증과 무관하게 '손해배상 책임을 면한다(책임을 지지 않는다)'가 결론이다. 또한 '~하여야 한다'(의무)와 '~할 수 있다'(재량)를 정확히 구분해 답한다.
  예) "A 소지자는 면제(다만, B 시설은 C 소지자를 포함)" → B 시설에서도 A 소지자는 면제이고, C 소지자도 추가로 면제.
- 감면·혜택이 둘 이상 겹치는 질문이면, ① 근거에 중복 적용을 정한 조항(예: '높은 하나만 적용')이 있으면 그 조항에 따라 최종 감면율을 첫 문장에 명확히 결론 내리고, ② 중복 적용 규정이 조례에 따로 없더라도 하나라도 '전액 감면(면제)' 사유에 해당하면 최종 결론은 반드시 '전액 감면(100% 감면)'으로 첫 문장에 단정한다('중복 규정이 없어 확정할 수 없다'고 하지 말고 전액 감면 적용을 결론 내린 뒤, '등록대행 수수료는 감면 제외' 같은 본문 단서를 함께 쓴다).
- 다른 조례의 조항을 끌어와 섞지 않는다. 질문이 묻는 조례의 근거만 쓴다.
- 결론 문장의 evidence에는 결론을 직접 정한 조항을 맨 앞에 둔다. 원칙 조항과 예외·단서·적용요건 조항을 함께 사용해 답했다면, 사용한 조항을 모두 evidence에 빠짐없이 넣는다(답변에 쓰지 않은 부수 조항은 넣지 않으며 전체 1~3개 이내로 유지).
- 질문의 장소·대상·상황에 어떤 규정(과태료·감면·금지 등)이 적용된다고 답하려면, 그 장소·대상이 근거 조항의 적용 범위
  (지정 대상 목록, 감면 대상 목록 등)에 실제로 적혀 있어야 한다. 적혀 있지 않으면 적용된다고 추측하지 말고 found를 false로 한다.
- 질문한 핵심 대상·주제가 조례 6종 어디에도 직접 나와 있지 않으면 found를 false로 한다 (단, 질문의 대상이 조문에 직접 나와 있고 세부 운영시간·금액·기준·절차만 시장·규칙·고시·다른 법령에 위임·준용한 경우는 found를 true로 하여 그 위임·준용 조항을 인용한다).
- 질문이 서울의 특정 자치구(강남구·서초구·마포구 등 25개 구)나 타 시·도(부산시·경기도·수원시 등)가 운영하는 시설·요금·과태료·지원·조례·규칙을 묻는 경우, 서울시 조례로 대신 답하지 말고 반드시 basis="없음", found=false로 한다(질문자가 거주지만 밝히면서 서울시 조례·서울대공원·서울시 관리 시설을 명시적으로 묻는 경우만 예외). 자치구(OO구) 조례나 규칙은 절대 근거로 쓰지 않는다.
- basis 필드는 답을 쓰기 전에 먼저 정한다. 질문의 핵심(묻는 대상·제도·행위)이 근거에 직접 나와 있고, 답할 규정 또는 위임·준용 조항이 있는가?
  · "있음": 질문이 묻는 대상(예: 반려견 놀이터, 자전거보관소, 공영주차장)이 근거 조문에 직접 나와 있으며,
    질문이 묻는 내용(규정·수치·기준)이 조문에 있거나, 혹은 세부 내용(운영시간·구체 금액·규격·절차 등)을
    '설치 및 운영에 필요한 사항은 시장이 따로 정한다'처럼 포괄 위임·개별 위임·준용('시장·구청장이 따로 정한다', '규칙·고시·공고로 정한다', '예산의 범위에서', '가이드라인을 마련할 수 있다', '다른 법령·공고를 준용한다')해 둔 경우에도 반드시 "있음"(found=true)이다!
    (예: "반려견 놀이터 운영시간은?" → 제29조 제3항, "따릉이 이용권 요금은?" → 제12조의2 제9항, "청년 대중교통비 한도는?" → 제15조 제3항(예산의 범위), "출입안내 표시판 규격은?" → 제7조의4 제3항(가이드라인) 모두 "있음"!)
    이때는 조례에 구체 수치가 정해져 있지 않다는 점과 누가/어디서 정하는지를 안내하고
    (예: "조례에는 운영 시간이 정해져 있지 않습니다. 제29조 제3항에 따라 시장이 따로 정하므로 서울시나 해당 시설 안내에서 확인하세요."),
    위임·준용을 직접 정한 조항(1~2개)만 evidence로 고른다 (같은 조의 불필요한 다른 항은 넣지 않는다).
    조례가 그 행위를 금지·제한한다고 정해 둔 경우도 "있음"이다 (근거 있는 '아니요').
  · "없음": 질문한 대상·행위 자체가 근거 어디에도 직접 적혀 있지 않고 주변 주제의 조항만 있는 경우(예: 드론 촬영, 서울도서관, 헬멧 과태료).
    "조례에는 ~에 관한 규정이 없습니다"라는 답이 될 수밖에 없다면 "없음"이다. 이때 비슷한 일반 조항으로 억지 연결해 설명하지 않는다.
    질문이 여러 가지를 묻는데 핵심 질문의 답이 없고 부수적인 것만 있으면 "없음"이다.
  · 질문의 대상 B가 조항에 나오지 않을 때는 질문 형태로 판단한다.
    - B가 조항에 직접 나와 있지 않은 별개 대상·수단·행위(예: 조문에 '사진·영상'만 있는데 '드론'을 묻는 경우)일 때, B에 대한 값·절차·의무(얼마, 몇, 해야 하나요)를 물으면 "없음".
      ("조항은 A만 정하므로 B는 해당하지 않습니다" 같은 추론으로 답하지 않는다)
    - 조문에 대상 X는 직접 나와 있고 질문이 X의 세부 항목(운영시간, 요금 등)만 묻는데 조문이 'X에 필요한 사항은 시장/규칙으로 정한다'고 위임한 경우는 "있음".
    - B가 해당되는지(예/아니요)를 묻고, 조항이 기준(나이·기간·개월·면적·대수·인원)이나 대상·심의 목록 또는 포괄 요건('시장이 회의에 부치는 사항' 등)을 명시해 B의 해당 여부나 요건 충족 여부를 판단할 수 있으면 반드시 "있음"(found=true)이다. 특히 기간·나이·수치 요건 미달(예: 6개월 이상 요건에 4개월·6개월 미만 입원)로 '요건에 해당하지 않아 할 수 없다'고 결론이 나는 경우에도 조문에 따라 판단한 것이므로 반드시 basis="있음", found=true로 답한다. 조항이 B를 명시적으로 제외·한정·금지한 경우도 "있음".
""" + (ANSWER_FRAMES if config.USE_ANSWER_FRAMES else "") + """
결과는 JSON으로 낸다."""

REPAIR_SYSTEM = SYSTEM + """

[추가 지시] 직전 답변이 검증에서 '근거에 없는 내용이 있다'고 지적됐다. 검증은 직전 답변의 문장별 evidence로 고른 근거만 보고 이뤄졌다.
- 근거를 잘못 골랐을 수 있으니, 전체 [근거]에서 답의 내용이 실제로 적힌 조항을 다시 확인해 고른다.
- 지적된 결론 오류(단서·이중부정·경계값·계산·위임 과해석)는 원문과 대조해 정확한 결론으로 바로잡아 쓰고, 지적된 부수 문장은 삭제한다. 새로운 추측을 덧붙이지 않는다.
- 질문의 일부 쟁점만 조례에 없을 때는 근거가 있는 핵심 질문 부분만 정확히 답하고(없는 부분만 '규정이 없다'고 짧게 구분), 유효한 조항이 있는데도 섣불리 found=false로 뒤집지 않는다.
- 그래도 질문 전체에 답할 근거가 없으면 found를 false로 한다."""

SAFE_VERDICT_SYSTEM = SYSTEM + """

[보수적 판결 지시] 직전 답변들이 검증을 통과하지 못했다. [확정 인용 조항]의 원문 문언만 그대로 옮기듯이 가장 보수적으로 1~2문장만 다시 작성하라.
- [확정 인용 조항] 밖의 추론·확대해석·계산 과장을 일절 붙이지 말라.
- 근거에서 직접 확인할 수 있는 사실(또는 시장·규칙 위임 사실)만 정확히 적고, 조항에 실제로 근거가 없으면 found를 false로 하라."""

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


_HO_LINE = re.compile(r"^\s*(\d+(?:의\d+)?)\.\s")


def _related_items(hit, question, min_items=5, keep=3):
    """호가 5개 이상인 긴 항에서 질문과 글자 2-gram IDF 겹침이 높은 호 번호(최대 3개)를 찾아 표시합니다."""
    if not question:
        return []
    text = hit.get("context", hit["text"])
    lines = text.splitlines()
    head_lines, items = [], []
    for ln in lines:
        m = _HO_LINE.match(ln)
        if m:
            items.append([m.group(1), ln])
        elif items:
            items[-1][1] += " " + ln.strip()
        else:
            head_lines.append(ln)
    if len(items) < min_items:
        return []
    chunks, _, _, df, _ = _load()
    n = len(chunks)
    head = set(_bigrams(" ".join(head_lines)))
    local = Counter(g for _, body in items for g in set(_bigrams(body)))
    common = {g for g, c in local.items() if c >= max(2, len(items) * 0.25)}
    q = set(_bigrams(question)) - head - common
    if not q:
        return []
    picked = []
    for _ in range(keep):
        best, best_sc, best_g = None, 0.0, set()
        for num, body in items:
            if num in picked:
                continue
            g = q & set(_bigrams(body))
            if not g:
                continue
            sc = sum(math.log(1 + (n - df[x] + 0.5) / (df[x] + 0.5)) for x in g)
            if sc > best_sc:
                best, best_sc, best_g = num, sc, g
        if best is None or best_sc < 6.0:
            break
        picked.append(best)
        q -= best_g
    return picked


def _context(hits, question=""):
    """근거마다 경계를 분명히 표시합니다. 호가 많은 긴 항에는 질문 관련 호 힌트를 덧붙입니다."""
    blocks = []
    for h in hits:
        body = h.get("context", h["text"])
        rel = _related_items(h, question) if question else []
        hint = f"\n(질문과 관련된 호: {', '.join(r + '호' for r in rel)})" if rel else ""
        blocks.append(f'<근거 조항="{h["label"]}">\n{body}{hint}\n</근거>')
    return "\n\n".join(blocks)


def _ask(system, user, hits, effort=None, temperature=0.0):
    import upstage
    out = json.loads(upstage.chat([{"role": "system", "content": system},
                                   {"role": "user", "content": user}], response_format=_schema(hits),
                                  temperature=temperature,
                                  reasoning_effort=(effort or config.ANSWER_REASONING) or None))
    return _assemble(out)


def _decision(o):
    """답의 '판단'만 뽑습니다: 근거 유무, 결론 근거 조항. ('해당없음'과 '네'처럼 방향이 반대가 아닌 차이는 무시)"""
    if o.get("basis") == "없음" or not o.get("found"):
        return ("없음", None)
    return ("있음", (o.get("evidence") or [None])[0])


_HARD_CALC_Q = re.compile(r"합산|연속|초과.*(?:할인|시간)|가산|보궐|결원|임기.*(?:언제|며칠)|과반수|총\s*(?:입장료|이용료|요금)|각각\s*\d|계산|공고.*(?:며칠|일)|며칠.*공고")
_HARD_COND_Q = re.compile(r"원칙.*예외|단서|불구하고|그러하지|함께\s*고려|별개로|주의의무|면제되는\s*건가|허가.*무관|의무가\s*(?:여전히|있는|별도)|열지\s*않아도|않아도\s*되|해당하지\s*않|면하지\s*못|면할\s*수|중복.*감면|감면.*중복")


def _auto_effort(question):
    """복합 계산·이중부정 단서·다중 조건 질의에만 선별적으로 AUTO_REASONING('low')을 켭니다."""
    if config.ANSWER_REASONING:
        return config.ANSWER_REASONING
    if not config.AUTO_REASONING:
        return None
    nums = _values(question)
    if (len(nums) >= 2 and len(question) >= 35) or (_HARD_CALC_Q.search(question) and len(question) >= 35):
        return config.AUTO_REASONING
    if _HARD_COND_Q.search(question) and len(question) >= 45:
        return config.AUTO_REASONING
    return None


def _ask_consistent(system, hits, question, trace, effort=None):
    """같은 질문에 근거 순서(정순·역순·셔플)와 온도를 달리해 답을 N개 동시에 만들고 판단을 다수결로 정합니다."""
    n = config.CONSISTENCY_N
    if n < 2:
        user = f"[근거]\n{_context(hits, question)}\n\n[질문]\n{question}"
        return _ask(system, user, hits, effort=effort)
    import random
    import time as _t
    from collections import Counter
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    def juror_job(k):
        if k == 0:
            j_hits, temp = hits, 0.0
        elif k == 1:
            j_hits, temp = list(reversed(hits)), 0.0
        else:
            j_hits = random.Random(42 + k).sample(hits, len(hits))
            temp = 0.35
        user = f"[근거]\n{_context(j_hits, question)}\n\n[질문]\n{question}"
        return _ask(system, user, j_hits, effort=effort, temperature=temp)

    # 첫 답이 나오면 나머지는 CONSISTENCY_GRACE_SEC 초만 더 기다립니다 (가장 느린 답을 기다리지 않음 → 응답 시간 상한)
    ex = ThreadPoolExecutor(n)
    futs = [ex.submit(juror_job, k) for k in range(n)]
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
    is_ok = [o.get("basis") != "없음" and bool(o.get("found")) and not _claims_no_rule(o) and bool(o.get("evidence"))
             for o in outs]
    need = len(outs) // 2 + 1
    pol = {o.get("conclusion") for o, ok in zip(outs, is_ok) if ok and o.get("conclusion") in ("네", "아니요")}
    trace["consistency"] = [[("있음" if ok else "없음"), o.get("evidence"), o.get("conclusion")]
                            for o, ok in zip(outs, is_ok)]
    valid_outs = [o for o, ok in zip(outs, is_ok) if ok]
    if len(valid_outs) < need:
        trace["consistency_agree"] = len(valid_outs) == 0
        return next((o for o, ok in zip(outs, is_ok) if not ok), outs[0])
    # 조항별 다수결 합의 집합을 구하고(Jaccard 최고 배심원 선택), 결론 방향도 다수결로 정합니다.
    lab_cnt = Counter(lab for o in valid_outs for lab in dict.fromkeys(o.get("evidence", [])))
    v_need = len(valid_outs) // 2 + 1
    consensus = {lab for lab, c in lab_cnt.items() if c >= v_need} or {lab_cnt.most_common(1)[0][0]}
    major = Counter(o.get("conclusion") for o in valid_outs).most_common(1)[0][0]

    def score_juror(o):
        ev = set(o.get("evidence", []))
        jacc = len(ev & consensus) / max(len(ev | consensus), 1)
        return (int(o.get("conclusion") == major), jacc, -abs(len(ev) - len(consensus)))

    best = max(valid_outs, key=score_juror)
    trace["consistency_consensus"] = sorted(consensus)
    trace["consistency_agree"] = (len(valid_outs) == len(outs) and len(pol) <= 1
                                  and all(set(o.get("evidence", [])) == consensus for o in valid_outs))
    return best


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


_DELEGATED = re.compile(
    r"위임|준용|시행규칙|시행령|고시|공고|예산의\s*범위|다른\s*법령|가이드라인"
    r"|(?:따로|별도로|규칙(?:으로|에서)|조례(?:으로|에서)?|시장이|구청장이|관리자\S*)\s*(?:따로\s*)?정(?:한|하|할|해\s*두|해지)"
)


# '기준에 해당하지 않습니다'·'규정에 맞지 않습니다'·'예외 사유가 존재하지 않습니다'는 규정을 적용한 결론이므로 규정 부재가 아닙니다.
_APPLY_NEG = re.compile(r"해당하지|해당되지|적용되지|적용하지|충족하지|포함되지|속하지|대상이 아니|맞지|부합하지|적합하지|일치하지|어긋나|존재하지")
_EXCEPT_NEG = re.compile(r"예외\s*(?:사유|조항|규정|기준)?[이가은는]?\s*[^.。]{0,20}?(?:없습니다|없어|존재하지\s*않)")


# 첫 절에서 이미 적극적 규정('~해야/정하고 있/규정되어 있...')을 답한 뒤 부수 항목이 없음을 적는 경우는 규정 부재에서 제외('정할 수 있' 위임 제외)
_AFFIRM_CLAUSE = re.compile(r"(?:하여야|해야|(?<!정)할\s*수\s*있|정하고\s*있|정해져\s*있|규정되어\s*있|적용되|부과|지급|지원|면제|감면|납부)")


def _claims_no_rule(out):
    """첫 문장이 '~규정이 없습니다'류이면 모델 판정(basis)과 상관없이 '근거 없음'으로 봅니다.
    단, '세부는 ~가 따로 정한다(위임)'는 답이나 앞 절에서 적극적 규정을 먼저 설명한 경우는 제외합니다."""
    first = (out.get("sentences") or [{}])[0].get("text", "")
    # '~기준으로 달마다 받을 수 없습니다'처럼 '할 수 없다'(불가 결론)는 규정 부재가 아니므로 제외합니다.
    hit = any(
        not _CANNOT.search(m.group(0))
        and not _APPLY_NEG.search(m.group(0))
        and not _EXCEPT_NEG.search(first[max(0, m.start() - 20):m.end() + 5])
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
            if v == 100.0 and "전액" in all_used and re.search(r"100\s*(?:%|퍼센트)", text):
                continue
            if v == 2 and 1.0 in src_vals and re.search(r"2\s*(?:개|명|회|건|차례|가지|곳|종)\s*(?:이상|초과|부터)", text):
                continue
            if scenario and _derivable(v, used_vals | allowed_q | calc_ok, scenario):
                continue
            issues.append((i, text, v, None))
    return issues


_YESNO_HEAD = re.compile(r"^\s*(네|예|아니요|아니오)\s*[,.]\s*")


_NEG_Q = re.compile(r"(?:^|\s)(?:안|못)(?:\s+\S|해|하면|되|돼|받|가도|써도)|않(?:아|나요|을까|는)|없(?:나요|을까|어요\?|죠|음\?|냐)")


# 선택형 질문: "A인가요, 아니면 B인가요?" → 네/아니요가 어느 쪽에 대한 답인지 모호
_CHOICE_Q = re.compile(r"아니면|혹은")
_WH_Q = re.compile(r"얼마|몇|며칠|언제|어디|무엇|뭔가|뭐|누가|누구|어떻게|어떤|어느|왜")
_NEG_FIRST_END = re.compile(
    r"(?:지\s?않아도\s?(?:됩니다|돼요|된다)|않아도\s?됩니다|지\s?않습니다|않습니다|없습니다|없어요"
    r"|안\s?됩니다|수\s?없습니다|필요가?\s?없|해당하지\s?않|불가합니다|받지\s?않)[^가-힣]*$"
)


def _is_wh_only(question):
    clauses = [c for c in re.split(r"[?？]", question) if c.strip()]
    asks = clauses if question.strip().endswith(("?", "？")) else clauses[:-1] or clauses
    return bool(asks) and all(_WH_Q.search(c) for c in asks)


def _no_yesno(question):
    """네/아니요 앞머리를 붙이지 않을 질문: 부정형('못 받나요?'), 선택형('A인가요, 아니면 B인가요?'), 순수 의문사('얼마/몇/언제')."""
    return bool(question) and bool(_NEG_Q.search(question) or _CHOICE_Q.search(question) or _is_wh_only(question))


def _finalize(out, question=""):
    """앞머리 '네,/아니요,'는 모델이 따로 고른 conclusion으로 코드가 붙입니다 (본문과 모순 방지).
    부정형·선택형·의문사 질문에는 네/아니요를 붙이지 않으며, 첫 문장 극성과 '네'가 어긋나면 교정합니다."""
    text = _YESNO_HEAD.sub("", out["answer"]).strip()
    c = out.get("conclusion")
    if _no_yesno(question):
        return text
    first = re.split(r"(?<=[.다요])\s+", text, maxsplit=1)[0]
    if c == "네" and _NEG_FIRST_END.search(first):
        c = "아니요"
    return f"{c}, {text}" if c in ("네", "아니요") else text


def _verify_ctx(out, hits):
    """검증용 근거: 답변 문장들이 고른 근거 + 같은 조의 형제 항(hits에 있는 것, 최대 8개).
    예) 제10조의3 제2항만 인용하고 같은 조 제3항의 사실(위원장·부위원장)을 함께 쓴 정당한 답을 오거절하지 않습니다."""
    by_label = {h["label"]: h for h in hits}
    used = [by_label[l] for l in out.get("evidence", []) if l in by_label and not DISTRICT.search(l)]
    seen = {h["label"] for h in used}
    used_groups = {h["group"] for h in used}
    for h in hits:
        if len(used) >= 8:
            break
        if h["group"] in used_groups and h["label"] not in seen and not DISTRICT.search(h["label"]):
            seen.add(h["label"])
            used.append(h)
    return used


_DEF_Q = re.compile(r"정의|뜻|의미|무엇을\s*말|누구를\s*말|개념|용어")


def _is_def_chunk(h):
    return "(정의)" in h.get("context", h["text"]).split("\n", 1)[0]


def _pick(out, hits, question=""):
    """문장별 주 근거(evidence[0])와 별표를 우선 보존하고, 정의 조항(제2조)이 비정의 질문에서 핵심 조항을 밀어내지 않게 합니다."""
    by_label = {h["label"]: h for h in hits}
    sents = out.get("sentences", [])
    primary = [ev[0] for s in sents for ev in [s.get("evidence", [])] if ev]
    annexes = [lab for s in sents for lab in s.get("evidence", []) if " 별표" in lab]
    secondary = [lab for s in sents for lab in s.get("evidence", [])[1:]]
    ordered = list(dict.fromkeys(primary + annexes + secondary + out.get("evidence", [])))
    used = [by_label[lab] for lab in ordered if lab in by_label]
    used = [h for h in used if not DISTRICT.search(h["label"])]
    if used:
        # 첫 근거(결론 근거)와 같은 조례만 인용합니다. 다른 조례 조항이 섞이는 것을 막고 인용 수를 제한합니다.
        law = used[0]["law"]
        used = [h for h in used if h["law"] == law]
        used = list({h["label"]: h for h in used}.values())
        if len(used) > config.MAX_CITATIONS and not _DEF_Q.search(question or ""):
            subst = [h for h in used if not _is_def_chunk(h)]
            defs = [h for h in used if _is_def_chunk(h)]
            if subst:
                used = subst + defs
        used = used[:config.MAX_CITATIONS]
    return used


_SAME_ART_PARA = re.compile(r"(?<![조\d])(?<!조 )제\s?(\d+)\s?항")
_PARA_LABEL = re.compile(r"^(.*제\d+조(?:의\d+)?) 제(\d+)항$")
_OP_CROSS_REF = re.compile(r"(?<![법영칙」]\s)(?<![법영칙」])(제\d+조(?:의\d+)?)(?:\s?제(\d+)항)?(?=(?:에\s*따른|에\s*따라|을\s*준용|를\s*준용|의\s*(?:규정을\s*)?준용))")


def _with_sibling_refs(citation, hits=None):
    """인용한 항의 본문이 같은 조의 다른 항('제1항에 따라') 또는 검색된 같은 조례의 직접 연결 조항('제11조에 따른', '준용')을
    가리키면 CITE_SIBLING_MAX_TOTAL(3개) 상한 안에서 보강합니다."""
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
    if hits and len(out) < config.CITE_SIBLING_MAX_TOTAL:
        hit_labels = {h["label"] for h in hits}
        for lab in citation:
            if " 별표" in lab:
                continue
            c = by_label.get(lab)
            if not c:
                continue
            body = _AMEND_HIST.sub(" ", c.get("context", "").split("\n", 1)[-1])
            law = c["law"]
            for art, para in _OP_CROSS_REF.findall(body):
                ref = f"{law} {art} 제{para}항" if para else f"{law} {art}"
                if len(out) >= config.CITE_SIBLING_MAX_TOTAL:
                    return out
                if ref != lab and ref in by_label and ref in hit_labels and ref not in out and not _is_def_chunk(by_label[ref]):
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


def _with_answer_mentions(citation, answer_text, hits):
    """답변 본문에서 명시적으로 언급한 같은 조례 조항('제5조 제1항', '별표 2' 등)이 hits에 있는데
    evidence에서 빠졌다면 MAX_CITATIONS(3개) 한도 안에서 보충합니다 (초과 인용 감점 0건 유지)."""
    if not citation or not answer_text or len(citation) >= config.MAX_CITATIONS:
        return citation
    law = hits[0]["law"] if hits else ""
    by_label = {h["label"]: h for h in hits}
    if citation[0] in by_label:
        law = by_label[citation[0]]["law"]
    out = list(citation)
    prev_end, prev_ext = -100, False
    for m in _REF.finditer(answer_text):
        before = answer_text[max(0, m.start() - 8):m.start()].rstrip()
        gap = answer_text[prev_end:m.start()].strip() if m.start() - prev_end < 8 else None
        ext = bool(_EXTERNAL.search(before)) or (prev_ext and gap in ("및", ",", "ㆍ", "또는", ""))
        prev_end, prev_ext = m.end(), ext
        if ext:
            continue
        art, para, art_only = m.groups()
        cand = None
        if art and para:
            cand = f"{law} {art} 제{para}항"
        elif art_only:
            cand = f"{law} {art_only}"
        if cand and cand in by_label and cand not in out:
            out.append(cand)
            if len(out) >= config.MAX_CITATIONS:
                return out
    for no in _ANNEX_REF.findall(answer_text):
        cand = f"{law} 별표 {no}" if no else f"{law} 별표"
        if cand in by_label and cand not in out:
            out.append(cand)
            if len(out) >= config.MAX_CITATIONS:
                return out
    return out


_TTAREUNGYI_Q = re.compile(r"따릉이|공공자전거")
_BIKE_PARK_Q = re.compile(r"주차")


def answer(question, trace=None):
    """trace에 dict를 넘기면 단계별 진단 정보를 채웁니다 (어디서 '확인 불가'가 났는지 분석용)."""
    import upstage
    if trace is None:
        trace = {}
    t_start = _time.time()
    if not question or not str(question).strip():
        trace["stop"] = "empty"
        return dict(UNKNOWN)
    if config.USE_VERIFY and _is_other_local_gov(question):
        trace["stop"] = "other_local_gov"
        return dict(UNKNOWN)
    queries = rewrite_queries(question)
    trace["queries"] = queries
    top = retrieve(question, queries=queries)

    # 검증 ①: 검색 단계에서 이미 관련 조문이 없어 보이면 바로 '확인 불가'
    if config.USE_VERIFY and top and top[0]["cos"] is not None and top[0]["cos"] < config.MIN_SCORE:
        trace["stop"] = "min_score"
        return dict(UNKNOWN)

    hits = expand_links(expand_refs(expand_siblings(top)), question)
    # 따릉이(공공자전거) 이용 질문에 자전거주차장 요금표(제9조·별표)가 섞여 혼동되는 것을 사전에 차단합니다.
    if _TTAREUNGYI_Q.search(question) and not _BIKE_PARK_Q.search(question):
        filtered = [h for h in hits if not ("자전거이용" in h["law"] and (h["label"].endswith("별표") or " 제9조" in h["label"]))]
        if filtered:
            hits = filtered
    trace["hits"] = [(h["label"], round(h["score"], 3), h["cos"] and round(h["cos"], 3)) for h in hits]
    user_msg = f"[근거]\n{_context(hits, question)}\n\n[질문]\n{question}"
    effort = _auto_effort(question)
    if effort:
        trace["auto_effort"] = effort
    out = _ask_consistent(SYSTEM, hits, question, trace, effort=effort)
    trace["llm"] = out
    used = _pick(out, hits, question)
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

        _CONCL_OR_CALC = re.compile(r"[×x*=+/÷]|합계|총\s|최종|따라서|결론|납부|계산|감면|면제|중복|우선|높은\s*하나|제외|단서|다만")

        def can_safe_drop(o, bad):
            return (
                bool(bad) and 0 not in bad and len(bad) < len(o["sentences"])
                and not any(_CONCL_OR_CALC.search(o["sentences"][i]["text"]) for i in bad)
            )

        def try_safe_verdict(sv_hits):
            if not (config.USE_SAFE_VERDICT and sv_hits and _time.time() - t_start < config.REPAIR_BUDGET_SEC):
                return None, None
            sv_ctx = f"[확정 인용 조항]\n{_context(sv_hits, question)}\n\n[질문]\n{question}"
            sv = _ask(SAFE_VERDICT_SYSTEM, sv_ctx, sv_hits)
            trace["safe_verdict"] = sv
            if sv.get("basis") == "없음" or _claims_no_rule(sv) or not sv.get("found") or not sv["sentences"]:
                return None, None
            v3, r3, _ = check(sv)
            trace["verdict3"], trace["reason3"] = v3, r3
            return (sv, v3) if v3 != "notGrounded" else (None, v3)

        verdict, reason, bad = check(out)
        trace["verdict"], trace["reason"], trace["unsupported"] = verdict, reason, bad
        if verdict == "notGrounded" and can_safe_drop(out, bad):
            # 결론·계산 문장이 아닌 부수 문장만 근거가 없음 → 코드가 안전하게 지웁니다 (전체 재작성 없이)
            drop(out, bad)
            verdict = "grounded_after_drop"
        elif verdict == "notGrounded" and config.USE_REPAIR and _time.time() - t_start < config.REPAIR_BUDGET_SEC:
            # 결론·계산 문장 자체가 문제 → 한 번만 고쳐 쓰기 (검증 실패 건만 추론 ON)
            fix_hint = "\n\n[우선 유지할 인용 후보] " + ", ".join(h["label"] for h in used) if (config.RETRY_FIX_CITATIONS and used) else ""
            rep_hits = used if (config.RETRY_FIX_CITATIONS and used and "근거를 잘못" not in reason and "다른 조항" not in reason and "시설" not in reason) else hits
            rep = _ask(REPAIR_SYSTEM, f"{user_msg}\n\n[직전 답변]\n{out['answer']}\n\n[검증 지적]\n{reason}{fix_hint}",
                       rep_hits, effort=config.REPAIR_REASONING)
            trace["repair"] = rep
            if rep.get("basis") == "없음" or _claims_no_rule(rep) or not rep.get("found") or not rep["sentences"]:
                sv, v3 = (try_safe_verdict(used) if ("대상이" not in reason and "규정이 없" not in reason) else (None, None))
                if not sv:
                    trace["stop"] = "repair_found_false"
                    return dict(UNKNOWN)
                out, verdict = sv, v3
            else:
                out = rep
                verdict, reason, bad = check(out)
                trace["verdict2"], trace["reason2"], trace["unsupported2"] = verdict, reason, bad
                if verdict == "notGrounded" and can_safe_drop(out, bad):
                    drop(out, bad)
                    verdict = "grounded_after_drop"
                elif verdict == "notGrounded":
                    sv, v3 = try_safe_verdict(_pick(out, hits, question) or used)
                    if sv:
                        out, verdict = sv, v3
        used = _pick(out, hits, question)
        if verdict == "notGrounded" or not used:   # notSure는 통과 (함정은 basis/found 단계에서 거름)
            trace["stop"] = "not_grounded"
            return dict(UNKNOWN)

    trace["sentences"] = [{"text": x["text"], "evidence": x.get("evidence", [])} for x in out["sentences"]]
    trace["stop"] = "answered"
    citation = list(dict.fromkeys(h["label"] for h in used))   # 순서 유지, 중복 제거
    if config.CITE_ANNEX_PARENT:
        citation = _with_annex_parent(citation)
    if config.CITE_SIBLING_REFS:
        citation = _with_sibling_refs(citation, hits)
    first = used[0]
    return {"answer": _finalize(out, question), "citation": citation,
            "source_url": first["source_url"], "effective_date": first["effective_date"]}


def answer_without_rag(question):
    """시연용 '전' 장면: 검색 없이 LLM만으로 답하게 하면 어떻게 되는가."""
    import upstage
    return upstage.chat([
        {"role": "system", "content": "너는 서울시 조례 전문가다. 답과 함께 근거 조항(제○조 제○항)과 원문 링크, 시행일을 적어라."},
        {"role": "user", "content": question}], temperature=0.3)
