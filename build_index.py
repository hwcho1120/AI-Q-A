"""3단계: 조문을 조각(chunk)으로 나누고 임베딩해 검색 색인을 만듭니다.

조각마다 '인용 라벨'을 붙여 두는 것이 핵심입니다.
  예) 서울특별시 주차장 설치 및 관리 조례 제7조 제3항
LLM은 이 라벨을 고르기만 하고, 조항 표기를 직접 쓰지 않습니다.

실행:  python -X utf8 build_index.py             (CHUNK_UNIT=paragraph 기본)
       set CHUNK_UNIT=article & python -X utf8 build_index.py   (조 단위, 시연 '전' 장면)
결과:  data/chunks_<단위>.json, data/vectors_<단위>.npy
"""
import json
import re

import numpy as np

import config
from parse_annex import annex_path

ITEM_HEAD = re.compile(r"^\s*\d+(?:의\d+)?\.\s")     # 호: "1. ", "2의2. "
SUB_MIN_LEN = 400                                   # 이보다 긴 조·항만 호 단위 보조 조각을 만듭니다
ANNEX_CONTEXT_MAX = 6000                            # LLM에 넘길 별표 원문 최대 길이


def split_items(body):
    """본문을 [머리말, 1호(+목), 2호(+목), ...]로 나눕니다. 호가 없으면 [본문]."""
    parts, cur = [], []
    for line in body.splitlines():
        if ITEM_HEAD.match(line) and cur:
            parts.append("\n".join(cur))
            cur = []
        cur.append(line.strip())
    if cur:
        parts.append("\n".join(cur))
    return parts


def _add(chunks, base, head, body, unit):
    """검색 조각 추가. 긴 조·항은 호 단위 보조 조각도 만들지만 라벨·context(LLM에 줄 원문)는 그대로입니다.

    호 하나하나가 짧은 질문과 잘 맞도록 하되, 인용은 채점 단위(조·항)로 유지하기 위함입니다.
    """
    full = f"[{head}]\n{body}"
    chunks.append({**base, "text": full, "context": full})
    if unit == "article" or len(body) < SUB_MIN_LEN:
        return
    parts = split_items(body)
    if len(parts) < 3:
        return
    intro = parts[0][:150]
    for it in parts[1:]:
        chunks.append({**base, "text": f"[{head}]\n{intro}\n{it}", "context": full, "sub": True})


def make_chunks(laws, unit):
    chunks = []
    for law in laws:
        meta = {"law": law["name"], "effective_date": law["effective_date"], "source_url": law["source_url"]}
        for a in law["articles"]:
            head = f"{law['name']} {a['article']}({a['title']})"
            group = f"{law['name']} {a['article']}"
            if unit == "article" or len(a["paragraphs"]) <= 1 or a["paragraphs"][0]["no"] is None:
                _add(chunks, {**meta, "label": group, "group": group}, head, a["text"], unit)
                continue
            for p in a["paragraphs"]:
                _add(chunks, {**meta, "label": f"{group} 제{p['no']}항", "group": group},
                     head, p["text"], unit)
        for ax in law["annexes"]:
            if not ax["label"].startswith("별표"):
                continue
            parsed = annex_path(law["mst"], ax["label"])
            if config.USE_PARSED_ANNEX and parsed.exists():
                body, src = parsed.read_text(encoding="utf-8"), "document-parse"
            else:
                body, src = ax["text"], "law.go.kr 원문(괘선 표)"
            # 긴 별표는 검색용으로 잘라 넣되, LLM에는 별표 전체(context)를 줍니다.
            # (조각만 주면 표 없이 비고만 전달되는 경우가 생김)
            label = f"{law['name']} {ax['label']}"
            head = f"[{law['name']} {ax['label']} {ax['title']}]"
            full = f"{head}\n{body[:ANNEX_CONTEXT_MAX]}"
            for i in range(0, max(len(body), 1), 1800):
                t = f"{head}\n{body[i:i + 1800]}"
                chunks.append({**meta, "label": label, "group": label, "text": t, "context": full,
                               "annex_source": src})
    for i, c in enumerate(chunks):
        c["id"] = i
    return chunks


def main():
    laws = json.loads((config.DATA / "ordinances.json").read_text(encoding="utf-8"))
    unit = config.CHUNK_UNIT
    chunks = make_chunks(laws, unit)
    (config.DATA / f"chunks_{unit}.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"조각 {len(chunks)}개 ({unit} 단위)")

    if not config.UPSTAGE_API_KEY:
        print("⚠ API 키가 없어 임베딩은 건너뜁니다. 키워드 검색만 동작합니다.")
        return
    import upstage
    vecs = upstage.embed([c["text"] for c in chunks], config.EMBED_PASSAGE)
    np.save(config.DATA / f"vectors_{unit}.npy", np.array(vecs, dtype=np.float32))
    print("임베딩 저장 완료")


if __name__ == "__main__":
    main()
