"""Upstage API 호출을 한곳에 모았습니다. (Chat · Embedding · Document Parse · Groundedness Check)

Solar는 OpenAI SDK와 호환됩니다. base_url만 https://api.upstage.ai/v1 로 바꾸면 됩니다.
"""
import json
import random
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
        _client = OpenAI(
            api_key=config.UPSTAGE_API_KEY,
            base_url=config.UPSTAGE_BASE_URL,
            timeout=45.0,
            max_retries=0,
        )
    return _client


def _retry(fn, tries=3, deadline=None):
    """429/일시적 서버·네트워크 오류 시 지수 백오프 + 지터로 재시도합니다 (시간 예산 내)."""
    for i in range(tries):
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError("Request budget exceeded before retry")
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            transient = any(code in msg for code in ("429", "500", "502", "503", "504", "timeout", "Timeout", "Connection"))
            if transient and i < tries - 1:
                sleep_s = min((1.2 ** i) + random.uniform(0.1, 0.5), 4.0)
                if deadline is not None and time.time() + sleep_s >= deadline:
                    raise
                time.sleep(sleep_s)
                continue
            raise


def chat(messages, response_format=None, temperature=0.0, reasoning_effort=None, model=None, deadline=None):
    kw = dict(model=model or config.CHAT_MODEL, messages=messages, temperature=temperature)
    if response_format:
        kw["response_format"] = response_format
    if reasoning_effort:                      # solar-pro4: 생략하면 추론 OFF, "low" 이상이면 ON
        kw["reasoning_effort"] = reasoning_effort
    resp = _retry(lambda: client().chat.completions.create(**kw), deadline=deadline)
    return resp.choices[0].message.content


def embed(texts, model, deadline=None):
    out = []
    for i in range(0, len(texts), 50):        # 한 번에 너무 많이 보내지 않습니다
        batch = texts[i:i + 50]
        resp = _retry(lambda b=batch: client().embeddings.create(model=model, input=b), deadline=deadline)
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


def groundedness(context, answer, deadline=None):
    """답변이 근거(context)에 실제로 기반하는지 판정합니다: grounded / notGrounded / notSure"""
    def _call():
        resp = requests.post(
            f"{config.UPSTAGE_BASE_URL}/systemone",
            headers={"Authorization": f"Bearer {config.UPSTAGE_API_KEY}"},
            json={
                "model": "solar-decide",
                "state": f"[근거]\n{context}\n\n[답변]\n{answer}",
                "questions": {
                    "verdict": {
                        "type": "choice",
                        "instructions": "Verify whether the factual claims, conditions, and numbers in [답변] are supported by [근거]. Paraphrasing legal phrasing into natural plain Korean is grounded as long as the core meaning and facts are accurate.",
                        "criteria": {
                            "grounded": "Core facts, conditions, and numbers match [근거], even if phrased in simpler Korean.",
                            "notGrounded": "[답변] invents facts, rules, or numbers not present in [근거], or contradicts [근거].",
                            "notSure": "Partially supported or ambiguous.",
                        },
                    }
                },
            },
            timeout=15,
        )
        resp.raise_for_status()
        verdict = resp.json()["answers"]["verdict"]
        probs = verdict.get("probabilities", {})
        if verdict.get("choice") == "notGrounded" and probs.get("notGrounded", 1.0) < 0.70:
            return "notSure"
        return verdict.get("choice", "notSure")

    return _retry(_call, tries=3, deadline=deadline)
