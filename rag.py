"""4단계: 검색 → Solar 답변 → 검증 → 대회 응답 형식.

설계 원칙 세 가지
  1. LLM은 근거 '번호'만 고릅니다. 조항 표기(citation)는 코드가 조각 라벨로 만듭니다.
  2. source_url · effective_date도 코드가 채웁니다. LLM이 링크나 날짜를 쓰게 두지 않습니다.
  3. 의심스러우면 '확인 불가'. 근거 없는 답은 맞아도 점수가 없고, 함정 문항에서는 감점입니다.
"""
import json
import math
import re
from collections import Counter

import numpy as np

import config

UNKNOWN = {"answer": "확인 불가", "citation": [], "source_url": "", "effective_date": ""}
DISTRICT = re.compile(r"서울특별시\s*\S+구\s")      # 'OO구' 자치구 조례 인용 방지

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


# ------------------------------------------------------------ 답변
SYSTEM = """너는 서울특별시 조례 질의응답 도우미다.
규칙:
- 반드시 아래 [근거]에 적힌 내용만으로 답한다. 근거에 없는 내용은 추측하거나 덧붙이지 않는다.
- conclusion 필드: 질문이 예·아니요로 답할 질문("~할 수 있나요/되나요/받을 수 있나요/내야 하나요")이면
  질문이 묻는 그대로에 대한 답을 고른다. 예) "둘 다 받을 수 있나요?"→받을 수 없으면 "아니요", "돈을 내야 하나요?"→원칙적으로 무료면 "아니요".
  "얼마/몇 살/어떻게 되나요"처럼 값·결과를 묻는 질문이면 "해당없음".
- answer는 "네"나 "아니요"로 시작하지 말고, 첫 문장에 결론(값·결과·가능 여부)을 바로 쓴다. (앞머리 "네,/아니요,"는 시스템이 붙인다)
- 이어서 결론을 뒷받침하는 조건·숫자(금액·비율·나이)와 '다만' 같은 단서·예외를 쓴다. 질문과 관계없는 절차·위임·이의제기 설명은 쓰지 않는다.
- 일반 시민이 이해할 수 있는 쉬운 말로 2~4문장. 숫자는 근거에 적힌 값 그대로 쓴다.
- '다만', 괄호( ) 안의 단서는 원칙에 붙는 조건이다. 단서가 적용 범위를 넓히는지('~도 포함한다', '~를 포함') 좁히는지
  ('~는 제외한다', '~에 한정한다', '그러하지 아니하다') 정확히 읽고, 결론을 원칙과 단서를 함께 적용한 결과로 쓴다.
  예) "A 소지자는 면제(다만, B 시설은 C 소지자를 포함)" → B 시설에서도 A 소지자는 면제이고, C 소지자도 추가로 면제.
- 감면·혜택이 둘 이상 겹치는 질문이면, 근거에 중복 적용을 정한 조항이 있는지 먼저 확인하고 그 조항에 따라 답한다.
- 다른 조례의 조항을 끌어와 섞지 않는다. 질문이 묻는 조례의 근거만 쓴다.
- evidence에는 답의 내용이 실제로 적힌 근거의 조항 이름(<근거 조항="..."> 값)을 고른다. 결론을 직접 정한 근거를 맨 앞에 두고, 결론에 꼭 필요한 조항만 1~3개 넣는다. 부수 설명용 조항은 넣지 않는다.
- 질문의 장소·대상·상황에 어떤 규정(과태료·감면·금지 등)이 적용된다고 답하려면, 그 장소·대상이 근거 조항의 적용 범위
  (지정 대상 목록, 감면 대상 목록 등)에 실제로 적혀 있어야 한다. 적혀 있지 않으면 적용된다고 추측하지 말고 found를 false로 한다.
- 근거로 질문에 답할 수 없으면(질문한 규정·금액이 근거에 없으면) found를 false로 한다.
- 자치구(OO구) 조례나 규칙은 근거로 쓰지 않는다.
결과는 JSON으로 낸다."""

REPAIR_SYSTEM = SYSTEM + """

[추가 지시] 직전 답변이 검증에서 '근거에 없는 내용이 있다'고 지적됐다. 검증은 직전 답변이 evidence로 고른 근거만 보고 이뤄졌다.
- 근거를 잘못 골랐을 수 있으니, 전체 [근거]에서 답의 내용이 실제로 적힌 조항을 다시 확인해 고른다.
- 지적된 문장은 고쳐 쓰지 말고 삭제한다. 새로운 주장(예: '~는 적용되지 않는다')을 덧붙이지 않는다.
- 그래도 질문에 답할 수 없으면 found를 false로 한다."""

def _schema(hits):
    """응답 스키마. evidence는 이번에 보여 준 근거의 조항 이름 중에서만 고를 수 있습니다(enum).

    번호로 고르게 하면 하나씩 밀려 고르는 실수가 반복돼서, 조항 이름 자체를 선택지로 줍니다.
    목록 밖의 조항은 스키마가 막으므로 지어낸 조항이 나올 수 없습니다.
    """
    labels = list(dict.fromkeys(h["label"] for h in hits)) or ["(없음)"]
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "law_answer",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "found": {"type": "boolean"},
                    "conclusion": {"type": "string", "enum": ["네", "아니요", "해당없음"]},
                    "answer": {"type": "string"},
                    "evidence": {"type": "array", "items": {"type": "string", "enum": labels}},
                },
                "required": ["found", "conclusion", "answer", "evidence"],
                "additionalProperties": False,
            },
        },
    }


def _context(hits):
    """근거마다 경계를 분명히 표시합니다."""
    return "\n\n".join(f'<근거 조항="{h["label"]}">\n{h.get("context", h["text"])}\n</근거>' for h in hits)


def _ask(system, user, hits):
    import upstage
    out = json.loads(upstage.chat([{"role": "system", "content": system},
                                   {"role": "user", "content": user}], response_format=_schema(hits),
                                  reasoning_effort=config.ANSWER_REASONING or None))
    return out


_YESNO_HEAD = re.compile(r"^\s*(네|예|아니요|아니오)\s*[,.]\s*")


def _finalize(out):
    """앞머리 '네,/아니요,'는 모델이 따로 고른 conclusion으로 코드가 붙입니다 (본문과 모순 방지)."""
    text = _YESNO_HEAD.sub("", out["answer"]).strip()
    c = out.get("conclusion")
    return f"{c}, {text}" if c in ("네", "아니요") else text


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


def answer(question, trace=None):
    """trace에 dict를 넘기면 단계별 진단 정보를 채웁니다 (어디서 '확인 불가'가 났는지 분석용)."""
    import upstage
    if trace is None:
        trace = {}
    queries = rewrite_queries(question)
    trace["queries"] = queries
    top = retrieve(question, queries=queries)

    # 검증 ①: 검색 단계에서 이미 관련 조문이 없어 보이면 바로 '확인 불가'
    if config.USE_VERIFY and top and top[0]["cos"] is not None and top[0]["cos"] < config.MIN_SCORE:
        trace["stop"] = "min_score"
        return dict(UNKNOWN)

    hits = expand_refs(expand_siblings(top))
    trace["hits"] = [(h["label"], round(h["score"], 3), h["cos"] and round(h["cos"], 3)) for h in hits]
    user_msg = f"[근거]\n{_context(hits)}\n\n[질문]\n{question}"
    out = _ask(SYSTEM, user_msg, hits)
    trace["llm"] = out
    used = _pick(out, hits)
    if not out.get("found") or not used:
        trace["stop"] = "found_false" if not out.get("found") else "no_evidence"
        return dict(UNKNOWN)

    # 검증 ②: 답이 고른 근거에 실제로 기반하는지 Groundedness Check
    if config.USE_VERIFY:
        verdict, reason = upstage.groundedness_detail(_context(used), _finalize(out), question)
        trace["verdict"], trace["reason"] = verdict, reason
        if verdict == "notGrounded" and config.USE_REPAIR:
            # 한 번만 고쳐 쓰기: 지적된 이유를 주고 근거에 있는 내용만으로 다시 답하게 합니다
            out = _ask(REPAIR_SYSTEM, f"{user_msg}\n\n[직전 답변]\n{out['answer']}\n\n[검증 지적]\n{reason}", hits)
            trace["repair"] = out
            used = _pick(out, hits)
            if not out.get("found") or not used:
                trace["stop"] = "repair_found_false"
                return dict(UNKNOWN)
            verdict, reason = upstage.groundedness_detail(_context(used), _finalize(out), question)
            trace["verdict2"], trace["reason2"] = verdict, reason
        if verdict == "notGrounded":            # notSure는 통과시킵니다 (함정은 found=false 단계에서 거름)
            trace["stop"] = "not_grounded"
            return dict(UNKNOWN)

    trace["stop"] = "answered"
    citation = list(dict.fromkeys(h["label"] for h in used))   # 순서 유지, 중복 제거
    first = used[0]
    return {"answer": _finalize(out), "citation": citation,
            "source_url": first["source_url"], "effective_date": first["effective_date"]}


def answer_without_rag(question):
    """시연용 '전' 장면: 검색 없이 LLM만으로 답하게 하면 어떻게 되는가."""
    import upstage
    return upstage.chat([
        {"role": "system", "content": "너는 서울시 조례 전문가다. 답과 함께 근거 조항(제○조 제○항)과 원문 링크, 시행일을 적어라."},
        {"role": "user", "content": question}], temperature=0.3)
