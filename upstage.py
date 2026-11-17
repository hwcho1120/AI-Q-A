"""Upstage API 호출을 한곳에 모았습니다. (Chat · Embedding · Document Parse · Groundedness Check)

Solar는 OpenAI SDK와 호환됩니다. base_url만 https://api.upstage.ai/v1 로 바꾸면 됩니다.
"""
import json
import threading
import time

import requests

import config

_client = None


def client():
    global _client
    if _client is None:
        if not config.UPSTAGE_API_KEY:
            # SystemExit이 아니라 일반 예외로: 서버(app.py)가 잡아서 '확인 불가'로 응답할 수 있게
            raise RuntimeError("UPSTAGE_API_KEY가 없습니다. .env.example을 .env로 복사하고 키를 넣어 주세요.")
        from openai import OpenAI
        # SDK 기본 타임아웃은 600초라 호출 하나가 멈추면 문항 전체가 90초를 넘깁니다(실측 608초).
        # 평소 호출은 수 초이므로 20초에서 끊고 다시 시도합니다.
        _client = OpenAI(api_key=config.UPSTAGE_API_KEY, base_url=config.UPSTAGE_BASE_URL,
                         timeout=20.0, max_retries=2)
    return _client


def _retry(fn, tries=4):
    """429(too_many_requests)는 잠시 기다렸다 다시 부릅니다. 한도는 Organization 단위로 공유됩니다."""
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if "429" in str(e) and i < tries - 1:
                time.sleep(2 ** i)
                continue
            raise


# ---- 사용량 집계: 응답의 usage(토큰 수)를 모델별로 누적합니다. 크레딧 소모 추정용.
_usage_lock = threading.Lock()
USAGE = {}


def _record(model, resp):
    u = getattr(resp, "usage", None)
    with _usage_lock:
        d = USAGE.setdefault(model, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0})
        d["calls"] += 1
        if u is not None:
            d["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
            d["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0
    return resp


def usage_snapshot():
    with _usage_lock:
        return json.loads(json.dumps(USAGE))


def _create(**kw):
    return _record(kw["model"], _retry(lambda: client().chat.completions.create(**kw)))


def chat(messages, response_format=None, temperature=0.0, reasoning_effort=None):
    kw = dict(model=config.CHAT_MODEL, messages=messages, temperature=temperature)
    if response_format:
        kw["response_format"] = response_format
    if reasoning_effort:                      # solar-pro4: 생략하면 추론 OFF, "low" 이상이면 ON
        kw["reasoning_effort"] = reasoning_effort
    if not response_format:
        resp = _create(**kw)
        return resp.choices[0].message.content
    # 구조화 출력: 응답이 비거나 JSON이 깨지면 한 번 더, 그래도 안 되면 추론 모드를 끄고 다시 시도합니다.
    attempts = [kw, kw, {k: v for k, v in kw.items() if k != "reasoning_effort"}]
    last = None
    for a in attempts:
        resp = _create(**a)
        content = resp.choices[0].message.content
        try:
            json.loads(content or "")
            return content
        except (TypeError, ValueError) as e:
            last = e
    raise ValueError(f"구조화 출력 실패(3회): {last!r}")


def embed(texts, model):
    out = []
    for i in range(0, len(texts), 50):        # 한 번에 너무 많이 보내지 않습니다
        batch = texts[i:i + 50]
        resp = _record(model, _retry(lambda: client().embeddings.create(model=model, input=batch)))
        out.extend(d.embedding for d in resp.data)
    return out


def document_parse(path):
    """Document Parse: 문서를 구조가 살아 있는 HTML/Markdown으로 바꿉니다. 표가 표로 남습니다."""
    url = f"{config.UPSTAGE_BASE_URL}/document-digitization"
    headers = {"Authorization": f"Bearer {config.UPSTAGE_API_KEY}"}
    with open(path, "rb") as f:
        resp = requests.post(url, headers=headers, timeout=300,
                             files={"document": f},
                             data={"model": "document-parse",
                                   "output_formats": json.dumps(["html", "markdown"])})
    resp.raise_for_status()
    content = resp.json().get("content", {})
    return content.get("markdown") or content.get("html") or content.get("text") or ""


_GROUND_SYSTEM = """너는 답변 검증기다. [질문]에 대한 [답변]이 [근거]에 기반하는지 판정한다.
- grounded: 답변의 핵심 사실(결론, 금액·비율·나이 같은 숫자, 적용 조건·예외)이 근거에 명시돼 있거나 근거로부터 바로 도출된다.
- notGrounded: 핵심 사실 중 근거와 모순되거나 근거에 전혀 없는 것(지어낸 숫자·조건·결론)이 있다.
- notSure: 판단하기 애매하다.
- 답변의 결론(네/아니요, 가능/불가, 면제/부과)이 근거를 적용한 결과와 반대이면 notGrounded.
  특히 '다만'·괄호 단서가 범위를 넓히는지(포함) 좁히는지(제외) 확인하고, 단서를 거꾸로 읽은 결론은 notGrounded.
- 답변이 질문의 장소·대상에 어떤 규정이 적용된다고 말하는데, 그 장소·대상이 근거 조항의 적용 범위(지정 대상 목록 등)에 적혀 있지 않으면 notGrounded.
다음은 근거에 없는 사실로 보지 않는다:
- 질문에 나온 대상(장소·사람·상황)을 답변에서 그대로 언급하는 것. 예) 질문의 '어린이공원'에 근거의 '금주구역 지정 대상 공원' 규정을 적용
- 쉬운 말로 바꿔 쓴 표현, 일반적인 안내 문구("현장에서 확인하세요")"""

_GROUND_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "groundedness",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["grounded", "notGrounded", "notSure"]},
                "reason": {"type": "string"},
            },
            "required": ["verdict", "reason"],
            "additionalProperties": False,
        },
    },
}


def groundedness_detail(context, answer, question=""):
    """(verdict, reason). verdict: grounded / notGrounded / notSure

    전용 모델 groundedness-check가 서비스 종료되어(400), Solar에 구조화 출력으로 판정을 맡깁니다.
    """
    raw = chat([{"role": "system", "content": _GROUND_SYSTEM},
                {"role": "user", "content": f"[근거]\n{context}\n\n[질문]\n{question}\n\n[답변]\n{answer}"}],
               response_format=_GROUND_SCHEMA)
    out = json.loads(raw)
    return out.get("verdict", "notSure"), out.get("reason", "")


_GROUND_SENT_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "groundedness_sent",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["grounded", "notGrounded", "notSure"]},
                "unsupported": {"type": "array", "items": {"type": "integer"}},
                "reason": {"type": "string"},
            },
            "required": ["verdict", "unsupported", "reason"],
            "additionalProperties": False,
        },
    },
}


def groundedness_sentences(context, sentences, question="", prefix=""):
    """문장 번호를 붙여 검증하고, 근거 없는 문장 번호(0부터)를 함께 돌려줍니다: (verdict, reason, [idx...])

    문장 단위로 알려 주면, 결론 문장이 멀쩡할 때 코드가 문제 문장만 지울 수 있습니다(전체 재작성 불필요).
    """
    numbered = "\n".join(f"[{i}] {(prefix if i == 1 else '') + t}" for i, t in enumerate(sentences, 1))
    sys_msg = _GROUND_SYSTEM + "\nunsupported에는 근거에 없는 사실이 들어 있는 문장의 번호([n])를 모두 넣는다. 없으면 빈 배열."
    raw = chat([{"role": "system", "content": sys_msg},
                {"role": "user", "content": f"[근거]\n{context}\n\n[질문]\n{question}\n\n[답변 문장]\n{numbered}"}],
               response_format=_GROUND_SENT_SCHEMA)
    out = json.loads(raw)
    bad = sorted({i - 1 for i in out.get("unsupported", []) if 1 <= i <= len(sentences)})
    return out.get("verdict", "notSure"), out.get("reason", ""), bad


def groundedness(context, answer):
    return groundedness_detail(context, answer)[0]
