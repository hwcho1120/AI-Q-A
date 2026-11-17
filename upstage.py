"""Upstage API 호출을 한곳에 모았습니다. (Chat · Embedding · Document Parse · Groundedness Check)

Solar는 OpenAI SDK와 호환됩니다. base_url만 https://api.upstage.ai/v1 로 바꾸면 됩니다.
"""
import json
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
        # 호출 하나가 멈추면 수백 초를 쓰므로(SDK 기본 600초 + 내부 재시도 2회) 타임아웃을 짧게 두고 재시도는 _retry에서만 한다
        _client = OpenAI(api_key=config.UPSTAGE_API_KEY, base_url=config.UPSTAGE_BASE_URL,
                         timeout=config.CALL_TIMEOUT_SEC, max_retries=0)
    return _client


def _retry(fn, tries=7):
    """일시적 오류는 잠시 기다렸다 다시 부릅니다. 한 문항 90초 제한 안에 끝나도록 총 시간 예산(RETRY_BUDGET_SEC)을 둡니다.
    - 429(요청 한도): 대기 1·2·4·8·16·16초(+무작위 0~1초). 한도는 Organization 단위로 공유됩니다.
    - 타임아웃·연결 오류·5xx: 1초 뒤 재시도.
    """
    import random
    t0 = time.time()
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            name = type(e).__name__
            limited = "429" in msg or "RateLimit" in name or "request limit" in msg
            transient = "Timeout" in name or "Connection" in name or any(f" {c}" in msg[:30] for c in ("500", "502", "503", "504"))
            wait = (min(2 ** i, 16) + random.random()) if limited else (1.0 if transient else None)
            if wait is None or i == tries - 1 or time.time() - t0 + wait > config.RETRY_BUDGET_SEC:
                raise
            time.sleep(wait)


def chat(messages, response_format=None, temperature=0.0, reasoning_effort=None):
    kw = dict(model=config.CHAT_MODEL, messages=messages, temperature=temperature)
    if response_format:
        kw["response_format"] = response_format
    if reasoning_effort:                      # solar-pro4: 생략하면 추론 OFF, "low" 이상이면 ON
        kw["reasoning_effort"] = reasoning_effort
    resp = _retry(lambda: client().chat.completions.create(**kw))
    return resp.choices[0].message.content


def embed(texts, model):
    out = []
    for i in range(0, len(texts), 50):        # 한 번에 너무 많이 보내지 않습니다
        batch = texts[i:i + 50]
        resp = _retry(lambda: client().embeddings.create(model=model, input=batch))
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


def groundedness(context, answer):
    """답변이 근거(context)에 실제로 기반하는지 판정합니다: grounded / notGrounded / notSure"""
    # groundedness-check 모델이 콘솔 목록에서 사라져 solar-pro4로 같은 판정을 내립니다.
    schema = {"type": "json_schema", "json_schema": {"name": "grounded", "strict": True, "schema": {
        "type": "object",
        "properties": {"verdict": {"type": "string", "enum": ["grounded", "notGrounded", "notSure"]}},
        "required": ["verdict"], "additionalProperties": False}}}
    prompt = ("아래 [근거]만 보고 [답변]의 핵심 결론(가능 여부·금액·나이·대상 같은 숫자와 조건)이 근거에서 확인되는지 판정하라.\n"
              "- grounded: 핵심 결론이 근거로 뒷받침됨. 쉬운 말로 풀어쓴 표현, 근거의 일부만 쓴 경우, 조항 번호 표기는 문제 삼지 않는다.\n"
              "- notGrounded: 핵심 결론이 근거와 어긋나거나, 근거에 전혀 없는 내용을 사실처럼 단정함\n"
              "- notSure: 판단하기 어려움\n\n"
              f"[근거]\n{context}\n\n[답변]\n{answer}")
    try:
        raw = chat([{"role": "user", "content": prompt}], response_format=schema)
        return json.loads(raw).get("verdict", "notSure")
    except Exception:  # noqa: BLE001  검증 장치 오류로 서비스가 죽지 않게
        return "notSure"
