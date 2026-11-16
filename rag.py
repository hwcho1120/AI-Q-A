"""4단계: 검색 → Solar 답변 → 검증 → 대회 응답 형식.

설계 원칙 세 가지
  1. LLM은 근거 '번호'만 고릅니다. 조항 표기(citation)는 코드가 조각 라벨로 만듭니다.
  2. source_url · effective_date도 코드가 채웁니다. LLM이 링크나 날짜를 쓰게 두지 않습니다.
  3. 의심스러우면 '확인 불가'. 근거 없는 답은 맞아도 점수가 없고, 함정 문항에서는 감점입니다.

고도화 내역:
  - 질문 분석 및 법령 용어 확장 (solar-mini4 Query Rewrite)
  - 라벨 단위 다양성 보장 + 동일 라벨 호(item) 병합
  - 형제 항 확장 (Same-Article Sibling Expansion) + 별표 연결 + 2-Hop 연쇄 검색
  - solar-pro4 (reasoning_effort="low") 기반 답변 및 예/아니오 품질 규칙 적용
  - solar-decide 기반 Groundedness Check 검증
"""
import json
import math
import re
import threading
import time
from collections import Counter, defaultdict

import numpy as np

import config

UNKNOWN = {"answer": "확인 불가", "citation": [], "source_url": "", "effective_date": ""}
DISTRICT = re.compile(r"서울특별시\s*\S+구\s")      # 'OO구' 자치구 조례 인용 방지
NAMED_LOCAL_GOV_PAT = re.compile(
    r"(?:강남|서초|송파|마포|성동|영등포|종로|용산|광진|동대문|중랑|성북|강북|도봉|노원|은평|서대문|양천|강서|구로|금천|동작|관악|강동)구"
    r"|(?:서울(?:특별시)?\s*|(?<=[\s,]))중구(?=[에의에서을를은는도과와,\s])"
    r"|수원시|성남시|고양시|용인시|부산시|인천시|대구시|대전시|광주시"
)
EXT_HARD_TRAP_PAT = re.compile(
    r"PC방|피시방|노래방|당구장|운전면허\s*벌점|면허\s*벌점|\b징역|징역형?|범칙금|형사\s*처벌|검찰|경찰(?:청|서|관|에)|법원(?:\s|을|에|의)"
)
ANNEX_REF = re.compile(r"(별표\s*\d+|별표)")
COMBINE_KW = ("둘 이상", "중복", "하나만", "감면", "제외", "다만", "과태료", "면제", "별표", "적용하지")

_cache = {}
_cache_lock = threading.Lock()


# ------------------------------------------------------------ 색인 로드 & BM25
def _bigrams(text):
    t = re.sub(r"\s+", "", text)
    return [t[i:i + 2] for i in range(len(t) - 1)]


def _load():
    unit = config.CHUNK_UNIT
    if unit in _cache:
        return _cache[unit]
    with _cache_lock:
        if unit in _cache:
            return _cache[unit]
        chunks = json.loads((config.DATA / f"chunks_{unit}.json").read_text(encoding="utf-8"))
        vec_path = config.DATA / f"vectors_{unit}.npy"
        vecs = np.load(vec_path) if vec_path.exists() else None
        if vecs is not None:
            vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

        search_texts = [c.get("search_text", c["label"] + " " + c["text"]) for c in chunks]
        docs = [Counter(_bigrams(t)) for t in search_texts]
        df = Counter(g for d in docs for g in d)
        avg = sum(sum(d.values()) for d in docs) / max(len(docs), 1)

        article_map = defaultdict(list)
        annex_map = defaultdict(list)
        label_map = defaultdict(list)
        for idx, c in enumerate(chunks):
            label_map[c["label"]].append(idx)
            art = c.get("article")
            if c.get("chunk_type") == "annex":
                annex_map[(c["law"], art)].append(idx)
            elif art:
                article_map[(c["law"], art)].append(idx)

        _cache[unit] = (chunks, vecs, docs, df, avg, article_map, annex_map, label_map)
        return _cache[unit]


def _bm25(query, docs, df, avg, k1=1.2, b=0.75):
    n = len(docs)
    q = set(_bigrams(query))
    scores = np.zeros(n)
    for i, d in enumerate(docs):
        ln = sum(d.values())
        if ln == 0:
            continue
        s = 0.0
        for g in q:
            if g in d:
                idf = math.log(1 + (n - df[g] + 0.5) / (df[g] + 0.5))
                s += idf * d[g] * (k1 + 1) / (d[g] + k1 * (1 - b + b * ln / avg))
        scores[i] = s
    return scores


# ------------------------------------------------------------ 질문 분석 & 다중 쿼리 검색
ANALYZE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "query_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "legal_queries": {"type": "array", "items": {"type": "string"}},
                "is_yes_no": {"type": "boolean"},
                "external_law": {"type": "boolean"},
                "is_composite": {"type": "boolean"},
            },
            "required": ["legal_queries", "is_yes_no", "external_law", "is_composite"],
            "additionalProperties": False,
        },
    },
}

ANALYZE_SYS = """너는 서울특별시 조례 6종(주차장, 동물보호, 자전거, 도시공원, 청년, 금연) 검색 최적화기다.
질문을 보고 다음 4가지를 JSON으로 출력하라.
1. legal_queries: 일상어 질문을 대한민국 법령·자치법규(조례) 원문에 쓰이는 정식 행정·법률 용어로 바꾼 검색 쿼리 2~3개.
   - 축약어나 구어체는 정식 법령 명칭·한자어(예: 차->자동차, 술->음주, 나이제한/나이 상한·하한->용어의 정의 제3조 청년 연령 19세 39세, 열 명 중 세 명->10분의 3, 둘 다->둘 이상 중복)로 풀어서 포함하라.
   - 질문의 직접적 대상뿐 아니라 그로 인한 처분·감면·단서·위임 규정 키워드도 함께 생성하라.
2. is_yes_no: 질문의 주된 요구가 예/아니오로 판단하는 단일 질문("~인가요?", "~되나요?", "~할 수 있나요?", "~내야 하나요?")이면 true, 금액·나이·방법·절차 등을 묻거나 복합 질문("~인가요? 그렇다면 얼마인가요?")이면 false.
3. external_law: 질문이 명백히 서울시 조례 6종 범위 밖(특정 OO구의 자체 요금·할인·지원, 타 시·도 조례, 징역·벌금·범칙금·면허벌점 등 형사처벌, 실내 PC방·노래방 등 국민건강증진법 전용 실내시설, 전기차 충전구역 일반차 정차 과태료)일 때만 true로 하라.
   - ⚠️ 반드시 false: "서울시 금연구역 지정·과태료(지하철역 출입구 10m, 버스정류소, 택시승차대, 공원 등)", "서울시 구청/자치구 청사 부설주차장 감면", 또는 "대부업법·근로기준법·건축법·동물보호법 등을 배경으로 언급하며 서울시장·서울시의 조치/규정을 묻는 질문"은 반드시 false로 하라.
4. is_composite: 질문이 둘 이상의 조건·예외·우선순위·의무·절차나 서로 다른 항목/조항을 동시에 구분하거나 결합해 답해야 하면 true. 단순히 말투가 길고 서론이 긴 1개 사실/요건 조회 질문은 반드시 false."""

COMPOSITE_HINT_PAT = re.compile(
    r"동시에|모두\s*(?:고려|적용|판단|충족)|각각|구분하여|우선(?:순위|적용)|중복\s*(?:적용|감면|할인)|둘\s*다|경우와\b|뿐만\s*아니라|함께\s*고려"
)


def _analyze_question(question, deadline=None):
    if not config.UPSTAGE_API_KEY:
        return {"legal_queries": [], "is_yes_no": False, "external_law": False, "is_composite": False}
    try:
        import upstage
        raw = upstage.chat(
            [{"role": "system", "content": ANALYZE_SYS},
             {"role": "user", "content": question}],
            response_format=ANALYZE_SCHEMA,
            model="solar-mini4",
            deadline=deadline,
        )
        return json.loads(raw)
    except Exception:  # noqa: BLE001
        return {"legal_queries": [], "is_yes_no": False, "external_law": False, "is_composite": False}


def _score_queries(queries, chunks, vecs, docs, df, avg, deadline=None):
    """원본 질문 + 확장 쿼리들의 혼합 점수 및 원본 질문 순수 코사인 점수를 계산합니다."""
    kw_list = []
    for q in queries:
        kw = _bm25(q, docs, df, avg)
        kw_list.append(kw / kw.max() if kw.max() > 0 else kw)

    if vecs is not None and config.UPSTAGE_API_KEY:
        import upstage
        embs = upstage.embed(queries, config.EMBED_QUERY, deadline=deadline)
        cos_list = []
        for e in embs:
            qv = np.array(e, dtype=np.float32)
            cos_list.append(vecs @ (qv / np.linalg.norm(qv)))
        raw_cos = cos_list[0]
        per_q = [0.68 * c + 0.32 * k for c, k in zip(cos_list, kw_list)]
    else:
        raw_cos = None
        per_q = kw_list

    if len(per_q) == 1:
        total = per_q[0]
    else:
        ext_max = np.max(np.vstack(per_q[1:]), axis=0)
        total = 0.52 * per_q[0] + 0.48 * ext_max
    return total, raw_cos


def _pick_diverse_chunks(score, chunks, top_k, max_per_label=2):
    """같은 label의 호(item)가 슬롯을 독점하지 않도록 다양성을 유지하며 상위 조각을 뽑습니다."""
    order = np.argsort(score)[::-1]
    picked = []
    label_counts = Counter()
    for idx in order:
        i = int(idx)
        lbl = chunks[i]["label"]
        if label_counts[lbl] >= max_per_label:
            continue
        picked.append(i)
        label_counts[lbl] += 1
        if len(picked) >= top_k:
            break
    return picked


def retrieve(question, top_k=config.TOP_K):
    """질문 확장 + 혼합 검색 (라벨 다양성 적용). check.py --retrieval과 호환됩니다."""
    chunks, vecs, docs, df, avg, _, _, _ = _load()
    analysis = _analyze_question(question)
    queries = [question] + [q for q in analysis.get("legal_queries", []) if q and q != question][:3]
    score, raw_cos = _score_queries(queries, chunks, vecs, docs, df, avg)
    # label 기준으로 top_k개의 서로 다른 조항/항이 포함되도록 선정
    picked = _pick_diverse_chunks(score, chunks, top_k, max_per_label=1)
    return [{**chunks[i], "score": float(score[i]),
             "cos": float(raw_cos[i]) if raw_cos is not None else None} for i in picked]


# ------------------------------------------------------------ 2단계 확장 & 병합
FOLLOWUP_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "followup_search",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "need_more": {"type": "boolean"},
                "followup_queries": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["need_more", "followup_queries"],
            "additionalProperties": False,
        },
    },
}

FOLLOWUP_SYS = """너는 법령 RAG 2단계 검색 판단기다.
[질문]과 1차로 찾은 [근거 요약]을 보고, 답을 완성하기 위해 연결된 후속 조문(예: 금주구역 지정에 따른 음주 과태료 조항, 인용된 별표, 예외/감면 조항, 절차 조항 등)을 추가로 검색해야 하는지 판단하라.
- 이미 근거가 충분하거나, 6개 서울시 조례에 없는 내용(함정 질문)이면 need_more=false, followup_queries=[] 로 하라.
- 1차 근거에 등장한 핵심 용어(예: 금주구역, 별표 등)와 질문의 결과(예: 과태료, 요금, 감면 중복 등)를 연결하는 검색 쿼리 1~2개를 생성하라."""


def _followup_queries(question, top_hits, deadline=None):
    if not config.UPSTAGE_API_KEY:
        return []
    summary = "\n".join(f"- {h['label']}: {h['text'][:240].replace(chr(10), ' ')}" for h in top_hits[:5])
    try:
        import upstage
        raw = upstage.chat(
            [{"role": "system", "content": FOLLOWUP_SYS},
             {"role": "user", "content": f"[질문]\n{question}\n\n[근거 요약]\n{summary}"}],
            response_format=FOLLOWUP_SCHEMA,
            model="solar-mini4",
            deadline=deadline,
        )
        res = json.loads(raw)
        if res.get("need_more"):
            return [q for q in res.get("followup_queries", []) if q][:2]
    except Exception:  # noqa: BLE001
        pass
    return []


def _expand_candidates(question, base_indices, base_score, chunks, vecs, docs, df, avg,
                       article_map, annex_map, is_composite=False, deadline=None):
    """1차 후보에 (1) 같은 조 형제 항, (2) 언급된 별표, (3) 2-hop 연쇄 검색 결과를 보강합니다."""
    cand_order = list(base_indices)
    cand_set = set(base_indices)
    ordered_labels = list(dict.fromkeys(chunks[i]["label"] for i in base_indices))

    def add_cand(idx):
        if idx not in cand_set:
            cand_set.add(idx)
            cand_order.append(idx)

    # 1) 본문에서 언급된 별표 연결 보강
    for i in base_indices[:5]:
        txt = chunks[i]["text"]
        law_name = chunks[i]["law"]
        for m in ANNEX_REF.findall(txt):
            ax_lbl = re.sub(r"\s+", " ", m).strip()
            if ax_lbl.startswith("별표") and len(ax_lbl) > 2 and " " not in ax_lbl:
                ax_lbl = "별표 " + ax_lbl[2:]
            ax_indices = annex_map.get((law_name, ax_lbl), [])
            if ax_indices:
                for idx in sorted(ax_indices, key=lambda idx: base_score[idx], reverse=True)[:2]:
                    add_cand(idx)

    # 2) 2-hop 연쇄 검색 (Multi-hop)
    top_hits_for_hop = [chunks[i] for i in base_indices[:5]]
    f_queries = _followup_queries(question, top_hits_for_hop, deadline=deadline)
    if f_queries:
        f_score, _ = _score_queries(f_queries, chunks, vecs, docs, df, avg, deadline=deadline)
        for i in _pick_diverse_chunks(f_score, chunks, top_k=5, max_per_label=1):
            add_cand(i)

    # 3) 같은 조(article) 내 핵심 형제 항 보강 (복합 질문이면 넓게, 단일 질문이면 핵심만)
    art_limit = 3 if is_composite else 2
    sib_limit = 3 if is_composite else 2
    top_articles = list(dict.fromkeys((chunks[i]["law"], chunks[i].get("article")) for i in base_indices[:6]))
    for law_name, art in top_articles[:art_limit]:
        if not art or str(art).startswith("별표"):
            continue
        sibs = article_map.get((law_name, art), [])
        distinct_paras = {chunks[s].get("para_no") for s in sibs}
        added = 0
        for s in sibs:
            if s in cand_set:
                continue
            if chunks[s]["label"] in ordered_labels:
                continue
            txt = chunks[s]["text"]
            if is_composite or len(distinct_paras) <= 4 or any(kw in txt for kw in COMBINE_KW):
                add_cand(s)
                added += 1
                if added >= sib_limit:
                    break

    return cand_order


def _merge_by_label(indices, chunks, label_map=None, max_labels=12):
    """동일 label을 가진 조각을 하나로 묶어 컨텍스트를 구성합니다.
    본문 조항이 호(item) 단위로 분할된 경우, 라벨이 선정되면 해당 항의 전체 호를 복원해 누락을 방지합니다."""
    grouped = defaultdict(list)
    label_order = []
    for idx in indices:
        lbl = chunks[idx]["label"]
        if lbl not in grouped:
            if len(label_order) >= max_labels:
                continue
            label_order.append(lbl)
        grouped[lbl].append(chunks[idx])

    merged_hits = []
    for lbl in label_order:
        items = grouped[lbl]
        first = items[0]
        if label_map and first.get("chunk_type") == "item":
            all_items = [chunks[i] for i in label_map.get(lbl, [])]
            if sum(len(x["text"]) for x in all_items) <= 12000:
                items = all_items
        items = sorted(items, key=lambda c: c["id"])
        if len(items) == 1:
            combined_text = items[0]["text"]
        else:
            # 서두/헤더 중복 줄이기
            parts = [items[0]["text"]]
            for nxt in items[1:]:
                t = nxt["text"]
                if t.startswith("["):
                    lines = t.splitlines()
                    # 첫 줄([조례 제O조...])과 직전 조각과 겹치는 서두 문장 제거
                    body_lines = [ln for ln in lines[1:] if ln not in parts[0]]
                    parts.append("\n".join(body_lines) if body_lines else t)
                else:
                    parts.append(t)
            combined_text = "\n".join(p for p in parts if p).strip()
        merged_hits.append({
            "label": lbl,
            "text": combined_text,
            "source_url": first["source_url"],
            "effective_date": first["effective_date"],
        })
    return merged_hits


# ------------------------------------------------------------ 답변 생성 & 검증
SYSTEM = """너는 서울특별시 조례 질의응답 도우미다.
규칙:
- 반드시 아래 [근거]에 적힌 내용만으로 답한다. 근거에 없는 내용은 절대 추측하거나 지어내지 않는다.
- 답은 일반 시민이 이해할 수 있는 쉬운 말로 1~3문장으로 간결하게 쓴다. 금액·비율·나이·시간 같은 숫자와 단서 조항("다만, ~", 예외 규정)은 빠뜨리지 말고 정확히 쓴다.
- 요금·감면·가산·비율 계산 시 '최초 N시간 이내 입출차 확인', '이상/초과/이하/미만', '중복 적용 불가' 등 전제 조건을 엄격히 대입한다. (예: 최초 2시간 이내 요건인 경우, 3시간 이용 시 요건 미충족이므로 2시간분만 임의 할인 계산 금지)
- 질문에서 묻지 않은 인접 조항·부수 절차·별도 규정은 덧붙이지 않는다. 단, 질문이 여러 상황·의무·예외를 구분해 답하라고 요구하면 원칙 조항("하여야 한다/무료로 한다")과 예외·단서·준용("다만", "제O조를 준용한다", "우선한다")·별표 내 시설 분류(예: 운동시설 vs 강의실 등)를 정확히 대조해 빠뜨리지 말고 답한다.
- 질문에 예/아니오로 답할 수 있는 조건 판단형 질문(예: ~인가요?, ~되나요?, ~내야 하나요?)인 경우, 반드시 "네," 또는 "아니요,"로 답변을 시작한다. 특히 "법적으로 없는 건지, 반드시 해야 하는 건지" 또는 "~가 안 되는 거 맞죠?" 같은 선택형·이중부정 질문은 질문 문장의 핵심 서술(없는지/맞는지)에 맞춰 '네'/'아니요' 방향이 뒤집히지 않게 신중히 판단한다.
- evidence_ids에는 답변 결론·수치·단서·별표 요금 산출에 실제로 사용한 [근거] 번호를 빠뜨리지 말고 모두 고른다(답변에 전혀 쓰지 않은 주변 조항만 제외할 것).
- [함정 방어 1 - 근거 부재] 질문이 구체적 수치(거리·비율·횟수·금액·규격·시간·인원), 구체적 목록, 형사처벌(징역·벌금·벌점), 가상 서비스/예외(새 자전거 지급·배송·65세 공영주차 무료 등), 또는 위임된 세부 기준을 묻는데 [근거]에 포괄 위임만 있거나 질문이 요구하는 규정 자체가 없다면, "명시되어 있지 않다/규정이 없다/아니요, 그런 규정은 없습니다"처럼 설명하지 말고 반드시 found=false, answer="확인 불가", evidence_ids=[]로 답하라.
- [함정 방어 2 - 상위법/자치구/실내시설] 질문이 국가 상위법(도로교통법·국민건강증진법·동물보호법·친환경자동차법 등)상의 징역·벌금·범칙금·벌점·경찰/검찰/법원 절차·전기차 충전구역 정차 과태료, 실내 PC방·노래방·당구장 등 국민건강증진법상 실내 다중이용시설 흡연 과태료, 또는 특정 자치구(강남구·마포구·성동구 등 OO구)의 자체 요금·감면을 물으면 서울시 조례로 억지 연결하거나 "조례에는 ~가 없습니다"라고 단정하지 말고 반드시 found=false, answer="확인 불가", evidence_ids=[]로 답하라.
결과는 JSON으로 낸다."""

MISSING_RULE_PAT = re.compile(
    r"^(?:[^.]{0,120})(?:이\s*|제시된\s*)?조례(?:에|에는|상)\s*[^.]{0,45}"
    r"(?:명시되어\s*있지\s*않|정해져\s*있지\s*않|규정되어\s*있지\s*않|직접적인\s*규정(?:이|은)\s*없|근거\s*조항이\s*없|근거가\s*없)"
    r"|(?:다만,\s*)?(?:거리\s*기준|감경\s*비율|분할\s*납부|형사처벌|징역|운전면허\s*벌점|벌점)[^.]{0,35}"
    r"(?:명시되어\s*있지\s*않|정해져\s*있지\s*않|규정되어\s*있지\s*않|조례에(?:는)?\s*없)"
    r"|제시된\s*근거만으로는\s*확인할\s*수\s*없|조례\s*근거만으로는\s*확인할\s*수\s*없|제공된\s*근거만으로는\s*확인할\s*수\s*없"
    r"|구체적인\s*(?:감면\s*비율|과태료|금액|수수료|기준|절차|예외\s*규정)은\s*확인할\s*수\s*없|직접\s*부과하는\s*과태료\s*규정은\s*확인되지\s*않"
    r"|조례가\s*아닌\s*규칙으로\s*정하도록\s*위임되어"
)
NO_TRAP_RULE_PAT = re.compile(
    r"(?:규정|조항|서비스|기준|제도)[^.]{0,30}(?:조례에(?:는)?\s*없습니다|규정[은이]\s*(?:조례에\s*)?(?:없습니다|없으며)|명시된\s*규정[은이]\s*없습니다|제정되어\s*있지\s*않습니다)"
    r"|^(?:아니요,\s*)?(?:그런\s*예외\s*규정|[^.]{0,55}(?:전액\s*지원|절반만\s*부과|새\s*자전거를\s*대신\s*지급|집까지\s*배송해\s*주는\s*서비스)[^.]{0,15}규정)[은이]\s*없습니다"
    r"|조례\s*별표\s*\d+에서\s*확인할\s*수\s*없"
)
EXT_DENY_PAT = re.compile(r"(?:형사\s*처벌|징역형?|벌점|도로교통법|국민건강증진법|동물보호법)[^.]{0,30}(?:없습니다|나오지\s*않|부과되지\s*않|적용되지\s*않)")

SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "law_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "found": {"type": "boolean"},
                "answer": {"type": "string"},
                "evidence_ids": {"type": "array", "items": {"type": "integer"}},
            },
            "required": ["found", "answer", "evidence_ids"],
            "additionalProperties": False,
        },
    },
}


def _context(hits):
    return "\n\n".join(f"[근거 {i}] {h['label']}\n{h['text']}" for i, h in enumerate(hits, 1))


def answer(question, trace=None):
    import upstage
    if trace is None:
        trace = {}
    deadline = time.time() + max(getattr(config, "ANSWER_TIMEOUT_SEC", 75) - 5, 30)
    try:
        chunks, vecs, docs, df, avg, article_map, annex_map, label_map = _load()
        # 자치구/타기관/상위법 함정 질문 방어 (확실한 패턴은 즉시 차단, 모호한 경우는 유사도 함께 확인)
        if config.USE_VERIFY and (NAMED_LOCAL_GOV_PAT.search(question) or EXT_HARD_TRAP_PAT.search(question)):
            trace["stop"] = "min_score"
            return dict(UNKNOWN)

        analysis = _analyze_question(question, deadline=deadline)

        is_composite = bool(
            analysis.get("is_composite")
            or COMPOSITE_HINT_PAT.search(question)
        )

        queries = [question] + [q for q in analysis.get("legal_queries", []) if q and q != question][:3]
        trace["queries"] = queries
        base_score, raw_cos = _score_queries(queries, chunks, vecs, docs, df, avg, deadline=deadline)

        # 검증 ①: 원본 질문의 최고 코사인 유사도가 기준선 미만이거나, 외부법령 판정 + 낮은 조례 유사도면 '확인 불가'
        if config.USE_VERIFY and raw_cos is not None:
            max_cos = float(raw_cos.max())
            if max_cos < config.MIN_SCORE or (analysis.get("external_law") and max_cos < 0.42):
                trace["stop"] = "min_score"
                return dict(UNKNOWN)

        # 1차 다양성 후보 선정 (상위 10개 청크)
        base_indices = _pick_diverse_chunks(base_score, chunks, top_k=10, max_per_label=2)

        # 2단계 확장: 형제 항 + 별표 연결 + 2-hop 연쇄 검색 (복합 질문은 더 넓게)
        expanded_indices = _expand_candidates(
            question, base_indices, base_score, chunks, vecs, docs, df, avg,
            article_map, annex_map, is_composite=is_composite, deadline=deadline
        )
        hits = _merge_by_label(
            expanded_indices, chunks, label_map=label_map,
            max_labels=15 if is_composite else 12
        )
        lbl_best = {}
        for idx in expanded_indices:
            lbl = chunks[idx]["label"]
            s = float(base_score[idx])
            if s > lbl_best.get(lbl, -1.0):
                lbl_best[lbl] = s
        trace["hits"] = [(h["label"], round(lbl_best.get(h["label"], 0.0), 3)) for h in hits]

        if analysis.get("is_yes_no"):
            yes_no_hint = "\n(참고: 이 질문은 예/아니오 판단형입니다. 답을 찾을 수 있다면 첫 단어를 질문에 맞춰 '네,' 또는 '아니요,'로 시작하세요.)"
        else:
            yes_no_hint = "\n(참고: 이 질문은 정보 조회/복합 질문이므로 문장을 '네,' 또는 '아니요,'로 시작하지 말고 바로 핵심 사실부터 자연스럽게 답하세요.)"
        if is_composite:
            yes_no_hint += " (복합 질문이므로 원칙·예외·준용·우선순위와 각각의 직접 근거 번호를 빠뜨리지 마세요.)"

        user_msg = f"[근거]\n{_context(hits)}\n\n[질문]\n{question}{yes_no_hint}"
        out = None
        for _ in range(2):
            if time.time() >= deadline:
                trace["stop"] = "error"
                return dict(UNKNOWN)
            raw = upstage.chat(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg}],
                response_format=SCHEMA,
                reasoning_effort="minimal",
                deadline=deadline,
            )
            try:
                out = json.loads(raw)
                break
            except json.JSONDecodeError:
                continue
        if not out:
            trace["stop"] = "error"
            return dict(UNKNOWN)
        trace["llm"] = out

        def _validate_out(cand_out):
            c_used = [hits[i - 1] for i in cand_out.get("evidence_ids", []) if 1 <= i <= len(hits)]
            c_used = [h for h in c_used if not DISTRICT.search(h["label"])]
            if not cand_out.get("found") or not c_used:
                return None, "", ("found_false" if not cand_out.get("found") else "no_evidence")
            c_ans = cand_out.get("answer", "")
            if config.USE_VERIFY:
                if EXT_DENY_PAT.search(c_ans) or NO_TRAP_RULE_PAT.search(c_ans):
                    return None, "", "found_false"
                if not c_ans.lstrip().startswith(("네,", "아니요,")) and MISSING_RULE_PAT.search(c_ans):
                    return None, "", "found_false"
            return c_used, c_ans, None

        used, ans_txt, stop_reason = _validate_out(out)
        if stop_reason:
            trace["stop"] = stop_reason
            return dict(UNKNOWN)

        # 검증 ②: 답이 고른 근거에 실제로 기반하는지 solar-decide Groundedness Check + 실패 시 1회 보정
        if config.USE_VERIFY and time.time() + 4 < deadline:
            try:
                verdict = upstage.groundedness(_context(used), ans_txt, deadline=deadline)
            except Exception:  # noqa: BLE001
                verdict = "notSure"
            trace["verdict"] = verdict
            if verdict == "notGrounded":
                repaired = False
                if time.time() + 10 < deadline:
                    repair_msg = (
                        f"{user_msg}\n\n[이전 초안]\n{ans_txt}\n선택한 근거 번호: {out.get('evidence_ids', [])}\n\n"
                        "[검증 피드백]\n이전 초안이 선택한 [근거] 번호만으로는 완전히 뒷받침되지 않아 검증을 통과하지 못했습니다. "
                        "답변에서 언급한 모든 주장·수치·조항의 [근거] 번호를 evidence_ids에 빠짐없이 포함하고, "
                        "원문에 없는 추측·옆 조항 내용은 제거해 다시 답하세요. 실제로 근거가 없다면 found=false로 답하세요."
                    )
                    try:
                        r_raw = upstage.chat(
                            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": repair_msg}],
                            response_format=SCHEMA,
                            reasoning_effort="minimal",
                            deadline=deadline,
                        )
                        r_out = json.loads(r_raw)
                        trace["repair"] = r_out
                        r_used, r_ans, r_stop = _validate_out(r_out)
                        if not r_stop and time.time() + 4 < deadline:
                            try:
                                v2 = upstage.groundedness(_context(r_used), r_ans, deadline=deadline)
                            except Exception:  # noqa: BLE001
                                v2 = "notSure"
                            trace["verdict2"] = v2
                            if v2 != "notGrounded":
                                used, ans_txt = r_used, r_ans
                                repaired = True
                    except Exception:  # noqa: BLE001
                        pass
                if not repaired:
                    trace["stop"] = "not_grounded"
                    return dict(UNKNOWN)

        trace["stop"] = "answered"
        citation = list(dict.fromkeys(h["label"] for h in used))   # 순서 유지, 중복 제거
        first = used[0]
        return {"answer": ans_txt, "citation": citation,
                "source_url": first["source_url"], "effective_date": first["effective_date"]}
    except Exception:  # noqa: BLE001
        trace["stop"] = "error"
        return dict(UNKNOWN)


def answer_without_rag(question):
    """시연용 '전' 장면: 검색 없이 LLM만으로 답하게 하면 어떻게 되는가."""
    import upstage
    return upstage.chat([
        {"role": "system", "content": "너는 서울시 조례 전문가다. 답과 함께 근거 조항(제○조 제○항)과 원문 링크, 시행일을 적어라."},
        {"role": "user", "content": question}], temperature=0.3)
