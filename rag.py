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
    _cache[unit] = (chunks, vecs, docs, df, avg)
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


REWRITE_PROMPT = """시민의 질문을 서울특별시 조례 조문에서 검색하기 좋은 문장으로 바꿔라.
- 조문에 실제로 쓰이는 용어로 바꾼다 (예: '몇 살' → '정의 연령 이상 이하', '돈 내나요' → '요금 사용료 수수료', '같이 받나요' → '둘 이상 감면 중복 적용').
- 질문에 나온 핵심 명사(대상·시설·제도명)는 그대로 유지한다.
- 답을 추측하지 말고, 검색어만 한 줄(60자 이내)로 쓴다."""

_rewrite_cache = {}


def rewrite(question):
    """질문을 조문 용어로 다시 씁니다. 실패하면 원문 질문을 그대로 씁니다."""
    if question in _rewrite_cache:
        return _rewrite_cache[question]
    try:
        import upstage
        out = upstage.chat([{"role": "system", "content": REWRITE_PROMPT},
                            {"role": "user", "content": question}]).strip().splitlines()[0][:120]
    except Exception:  # noqa: BLE001
        out = question
    _rewrite_cache[question] = out
    return out


def _score(query, chunks, vecs, docs, df, avg):
    kw = _bm25(query, docs, df, avg)
    kw_norm = kw / kw.max() if kw.max() > 0 else kw
    if vecs is not None and config.UPSTAGE_API_KEY:
        import upstage
        q = np.array(upstage.embed([query], config.EMBED_QUERY)[0], dtype=np.float32)
        cos = vecs @ (q / np.linalg.norm(q))
        return cos, 0.7 * cos + 0.3 * kw_norm      # TODO: 가중치를 공개 예시로 조정해 보세요
    return None, kw_norm


_ARTICLE = re.compile(r"^(.*? 제\d+조(?:의\d+)?)")


def retrieve(question, top_k=config.TOP_K):
    """임베딩(의미) + 키워드(용어) 혼합 검색. 임베딩이 없으면 키워드만 씁니다.

    USE_REWRITE: 원문 질문과 조문 용어로 다시 쓴 질문을 각각 검색해 조각별로 높은 점수를 씁니다.
                 MIN_SCORE 판단용 cos는 원문 질문 기준입니다 (재작성이 함정 문항 점수를 올리지 않게).
    USE_SIBLINGS: 상위 결과와 같은 조의 다른 항을 뒤에 덧붙입니다 (원칙 + 예외·중복 규정을 함께 보기 위해).
    """
    chunks, vecs, docs, df, avg = _load()
    cos, score = _score(question, chunks, vecs, docs, df, avg)
    if config.USE_REWRITE:
        rq = rewrite(question)
        if rq and rq != question:
            _, score2 = _score(rq, chunks, vecs, docs, df, avg)
            score = np.maximum(score, score2)
    idx = list(np.argsort(score)[::-1][:top_k])

    extra = []
    if config.USE_SIBLINGS:
        seen = set(idx)
        for i in idx[:config.SIBLING_TOP]:
            m = _ARTICLE.match(chunks[i]["label"])
            if not m or chunks[i]["label"].startswith(m.group(1) + " 별표"):
                continue
            for j, c in enumerate(chunks):
                if j not in seen and c["label"].startswith(m.group(1) + " 제") and c["label"].endswith("항"):
                    extra.append(j)
                    seen.add(j)
        extra = extra[:config.SIBLING_MAX]
    return [{**chunks[i], "score": float(score[i]),
             "cos": float(cos[i]) if cos is not None else None} for i in idx + extra]


# ------------------------------------------------------------ 답변
SYSTEM = """너는 서울특별시 조례 질의응답 도우미다.
규칙:
- 반드시 아래 [근거]에 적힌 내용만으로 답한다. 근거에 없는 내용은 추측하지 않는다.
- 답은 일반 시민이 이해할 수 있는 쉬운 말로 2~4문장으로 쓴다. 금액·비율·나이 같은 숫자는 그대로 쓴다.
- 답에 실제로 쓴 근거의 번호를 evidence_ids에 넣는다. 여러 조항을 함께 썼다면 모두 넣는다.
- 근거로 답할 수 없으면 found를 false로 한다.
- 자치구(OO구) 조례나 규칙은 근거로 쓰지 않는다.
결과는 JSON으로 낸다."""

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


# 답 문장이 스스로 '규정 없음'이라고 말하는 경우 (found=true여도 확인 불가로 돌린다)
NO_RULE = re.compile(r"(규정|내용|근거|기준|사항)[이가은는에]?\s*(전혀\s*)?"
                     r"(없(습니다|어요|다|으며|고|어서|기\s*때문)|나와\s*있지\s*않|명시되(어|지)\s*(있지\s*)?않|확인되지\s*않)"
                     r"|확인할\s*수\s*없|정해져\s*있지\s*않|규정하고\s*있지\s*않|언급되(어|지)\s*(있지\s*)?않")


def _context(hits):
    return "\n\n".join(f"[근거 {i}] {h['label']}\n{h['text']}" for i, h in enumerate(hits, 1))


_def_cache = {}


def _definition_chunk(label):
    """label이 속한 조례의 '(정의)' 조 조각을 돌려준다 (없으면 None)."""
    law = label.split(" 제")[0].split(" 별")[0]
    if law not in _def_cache:
        _def_cache[law] = next((c for c in _load()[0] if c["label"].startswith(law + " 제")
                                and re.match(r"\s*(\[[^\]]*\]\s*)?제\d+조\(정의\)", c["text"])), None)
    return _def_cache[law]


_REF = re.compile(r"제(\d+)조(?:의(\d+))?(?:\s*제(\d+)항)?\s*관련")


def _annex_refs(hit, labels):
    """별표 조각 제목의 '(제33조제1항 관련)'에서 그 별표가 달린 조·항 라벨을 만든다. 실제 있는 라벨만 돌려준다."""
    first = hit["text"].split("\n", 1)[0]
    law = hit["label"].rsplit(" 별", 1)[0]
    out = []
    for m in _REF.finditer(first):
        art = f"{law} 제{m.group(1)}조" + (f"의{m.group(2)}" if m.group(2) else "")
        for cand in ([f"{art} 제{m.group(3)}항"] if m.group(3) else []) + [art]:
            if cand in labels:
                out.append(cand)
                break
    return out


def answer(question, trace=None):
    """trace: dict를 넘기면 어느 단계에서 끝났는지(stage)와 중간값을 기록합니다 (디버깅·측정용)."""
    import upstage
    t = trace if trace is not None else {}
    hits = retrieve(question)
    if config.USE_DEFINITIONS and hits:     # 1순위 조례의 정의 조항을 근거 후보에 항상 포함
        d = _definition_chunk(hits[0]["label"])
        if d and all(h["id"] != d["id"] for h in hits):
            hits = hits + [{**d, "score": 0.0, "cos": None}]
    t["hits"] = [h["label"] for h in hits]

    # 검증 ①: 검색 단계에서 이미 관련 조문이 없어 보이면 바로 '확인 불가'
    if config.USE_VERIFY and hits and hits[0]["cos"] is not None and hits[0]["cos"] < config.MIN_SCORE:
        t["stage"] = "score_gate"
        return dict(UNKNOWN)

    raw = upstage.chat(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": f"[근거]\n{_context(hits)}\n\n[질문]\n{question}"}],
        response_format=SCHEMA)
    out = json.loads(raw)
    t["llm"] = out
    t["cos"] = [h["cos"] for h in hits]
    used = [hits[i - 1] for i in out.get("evidence_ids", []) if 1 <= i <= len(hits)]
    used = [h for h in used if not DISTRICT.search(h["label"])]
    t["used"] = [h["label"] for h in used]
    t["used_cos"] = [h["cos"] for i in out.get("evidence_ids", []) if 1 <= i <= len(hits) for h in [hits[i - 1]]]
    if not out.get("found"):
        t["stage"] = "found_false"
        return dict(UNKNOWN)
    if not used:
        t["stage"] = "no_evidence"
        return dict(UNKNOWN)
    if config.USE_NORULE_GUARD and NO_RULE.search(out["answer"]):
        t["stage"] = "no_rule_guard"
        return dict(UNKNOWN)

    # 검증 ②: 답이 고른 근거에 실제로 기반하는지 Groundedness Check
    if config.USE_VERIFY and config.USE_GROUNDED_CHECK:
        verdict = upstage.groundedness(_context(used), out["answer"])
        t["verdict"] = verdict
        if verdict == "notGrounded":            # TODO: notSure는 어떻게 다룰지 정해 보세요
            t["stage"] = "not_grounded"
            return dict(UNKNOWN)

    t["stage"] = "ok"
    if config.USE_PARA_COMPLETE:    # 인용한 항과 같은 조의 다른 항이 검색 결과에 있으면 함께 인용
        art = lambda lb: re.sub(r" 제\d+항$", "", lb)  # noqa: E731
        have = {h["id"] for h in used}
        for u in list(used):
            if not u["label"].endswith("항"):
                continue
            more = [h for h in hits if h["id"] not in have and h["label"].endswith("항") and art(h["label"]) == art(u["label"])]
            for h in more[:config.PARA_COMPLETE_MAX]:
                used.append(h)
                have.add(h["id"])
    citation = list(dict.fromkeys(h["label"] for h in used))   # 순서 유지, 중복 제거
    if config.USE_ANNEX_REFS:       # 별표를 인용하면 그 별표가 달린 조·항도 함께 인용
        labels = {c["label"] for c in _load()[0]}
        for h in used:
            if " 별" in h["label"]:
                citation += [r for r in _annex_refs(h, labels) if r not in citation]
    first = used[0]
    return {"answer": out["answer"], "citation": citation,
            "source_url": first["source_url"], "effective_date": first["effective_date"]}


def answer_without_rag(question):
    """시연용 '전' 장면: 검색 없이 LLM만으로 답하게 하면 어떻게 되는가."""
    import upstage
    return upstage.chat([
        {"role": "system", "content": "너는 서울시 조례 전문가다. 답과 함께 근거 조항(제○조 제○항)과 원문 링크, 시행일을 적어라."},
        {"role": "user", "content": question}], temperature=0.3)
