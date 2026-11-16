"""3단계: 조문을 조각(chunk)으로 나누고 임베딩해 검색 색인을 만듭니다.

조각마다 '인용 라벨'을 붙여 두는 것이 핵심입니다.
  예) 서울특별시 주차장 설치 및 관리 조례 제7조 제3항
LLM은 이 라벨을 고르기만 하고, 조항 표기를 직접 쓰지 않습니다.

고도화 내역:
  1. 호(1. 2. ...) 단위 서브 청킹: 긴 조/항은 호 단위로 분할하되 인용 라벨은 소속 조/항으로 유지합니다.
  2. 별표 구조적 청킹: 섹션(#)과 Markdown 표 헤더를 보존하여 행 단위로 분할하고, 연결 본문 조항을 함께 표기합니다.
  3. 검색용 텍스트 정제: <개정 ...> 등 개정 이력 노이즈를 제거해 임베딩/키워드 검색 정밀도를 높입니다.
"""
import json
import re

import numpy as np

import config
from parse_annex import annex_path

REVISION_NOISE = re.compile(r"<(?:개정|신설|전문개정|삭제|제목개정)[^>]*>|\[(?:본조신설|전문개정|제목개정)[^\]]*\]")
ITEM_HEAD = re.compile(r"^\s*(\d+(?:의\d+)?)\.\s+")


def clean_for_search(text):
    """검색(임베딩·BM25)에 방해되는 개정 이력 태그 등을 정리합니다."""
    cleaned = REVISION_NOISE.sub("", text)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def split_items(text):
    """본문을 서두(preamble)와 호(1. 2. ...) 리스트로 분리합니다."""
    lines = text.splitlines()
    preamble_lines = []
    items = []
    cur_item = None

    for line in lines:
        s = line.strip()
        if not s:
            continue
        m = ITEM_HEAD.match(s)
        if m:
            if cur_item is not None:
                items.append(cur_item)
            cur_item = {"no": m.group(1), "lines": [s]}
        elif cur_item is None:
            preamble_lines.append(s)
        else:
            cur_item["lines"].append(s)

    if cur_item is not None:
        items.append(cur_item)

    preamble = "\n".join(preamble_lines).strip()
    return preamble, ["\n".join(it["lines"]).strip() for it in items]


def _first_cell(row_line):
    cells = [c.strip() for c in row_line.strip().strip("|").split("|")]
    return cells[0] if cells else ""


def _split_markdown_table(table_lines, max_chars=1200):
    """Markdown 표의 상단 헤더(다중 헤더 포함)를 유지하면서 행 단위로 분할합니다."""
    sep_idx = None
    for idx, ln in enumerate(table_lines):
        if re.match(r"^\s*\|[\s\-:|]+\|\s*$", ln) and "---" in ln:
            sep_idx = idx
            break
    if sep_idx is None:
        return ["\n".join(table_lines)]

    header_end = sep_idx + 1
    top_first_cell = _first_cell(table_lines[0]) if table_lines else ""
    while header_end < len(table_lines) and top_first_cell and _first_cell(table_lines[header_end]) == top_first_cell:
        header_end += 1

    header = "\n".join(table_lines[:header_end])
    body_rows = table_lines[header_end:]
    if not body_rows:
        return [header]

    chunks = []
    cur_rows = []
    cur_len = len(header)
    for row in body_rows:
        if cur_rows and cur_len + len(row) + 1 > max_chars:
            chunks.append(header + "\n" + "\n".join(cur_rows))
            cur_rows = [row]
            cur_len = len(header) + len(row) + 1
        else:
            cur_rows.append(row)
            cur_len += len(row) + 1
    if cur_rows:
        chunks.append(header + "\n" + "\n".join(cur_rows))
    return chunks


def chunk_annex_markdown(body, max_chars=1450):
    """별표 Markdown을 섹션과 표 구조를 살려 분할합니다."""
    if len(body) <= max_chars:
        return [body]

    lines = body.splitlines()
    blocks = []
    cur_section = ""
    i = 0
    buf = []

    def flush_buf():
        nonlocal buf
        text = "\n".join(buf).strip()
        if text:
            blocks.append(("text", cur_section, text))
        buf = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if re.match(r"^#{1,3}\s+", stripped) or re.match(r"^\d+\.\s+(?:운동|휴양|교양|기타)", stripped) or stripped == "<비고>":
            flush_buf()
            if not stripped.startswith("# 서울특별시") and not stripped.startswith("# ■"):
                cur_section = stripped
            buf.append(line)
            i += 1
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            flush_buf()
            tbl = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                tbl.append(lines[i])
                i += 1
            for tbl_part in _split_markdown_table(tbl, max_chars=max_chars):
                blocks.append(("table", cur_section, tbl_part))
            continue

        buf.append(line)
        i += 1
    flush_buf()

    # 인접한 짧은 블록들을 max_chars 이내로 묶되, 섹션 헤더는 유지
    merged = []
    cur_txt = ""
    for kind, sec, txt in blocks:
        prefix = f"{sec}\n" if sec and sec not in txt else ""
        piece = f"{prefix}{txt}".strip()
        if cur_txt and len(cur_txt) >= 250 and len(cur_txt) + len(piece) + 2 > max_chars:
            merged.append(cur_txt)
            cur_txt = piece
        else:
            cur_txt = f"{cur_txt}\n\n{piece}".strip() if cur_txt else piece
    if cur_txt:
        merged.append(cur_txt)
    return merged


def _find_annex_refs(law):
    """별표 라벨(예: '별표 1')을 본문에서 인용하는 조항 목록을 찾습니다."""
    refs = {}
    for ax in law["annexes"]:
        lbl = ax["label"]
        pat = re.compile(rf"(?:「)?{re.escape(lbl).replace(' ', r'\s*')}(?:」)?(?=[와과의을를에로\s.,)]|$)")
        matched = []
        for a in law["articles"]:
            for p in a["paragraphs"]:
                if pat.search(p["text"]):
                    art_lbl = f"{a['article']}({a['title']})" if a["title"] else a["article"]
                    p_lbl = f" 제{p['no']}항" if p["no"] else ""
                    matched.append(f"{art_lbl}{p_lbl}")
        refs[lbl] = list(dict.fromkeys(matched))
    return refs


def make_chunks(laws, unit):
    chunks = []
    for law in laws:
        meta = {
            "law": law["name"],
            "effective_date": law["effective_date"],
            "source_url": law["source_url"],
        }
        annex_refs = _find_annex_refs(law)

        for a in law["articles"]:
            head = f"{law['name']} {a['article']}({a['title']})"
            if unit == "article":
                txt = a["text"]
                chunks.append({
                    **meta,
                    "article": a["article"],
                    "para_no": None,
                    "chunk_type": "article",
                    "label": f"{law['name']} {a['article']}",
                    "text": txt,
                    "search_text": f"{head}\n{clean_for_search(txt)}",
                })
                continue

            # paragraph 단위 (긴 조/항의 경우 호 단위 서브 청킹, 인용 라벨은 항/조 유지)
            for p in a["paragraphs"]:
                has_para_no = p["no"] is not None and len(a["paragraphs"]) > 1
                lbl = f"{law['name']} {a['article']} 제{p['no']}항" if has_para_no else f"{law['name']} {a['article']}"
                full_p_text = f"[{head}]\n{p['text']}" if has_para_no else p["text"]

                preamble, items = split_items(p["text"])
                if len(items) >= 2 and len(p["text"]) >= 280:
                    pre_block = f"[{head}]\n{preamble}" if has_para_no or preamble else f"[{head}]"
                    for idx_it, it_txt in enumerate(items, 1):
                        sub_txt = f"{pre_block}\n{it_txt}".strip()
                        chunks.append({
                            **meta,
                            "article": a["article"],
                            "para_no": p["no"] if has_para_no else None,
                            "chunk_type": "item",
                            "item_idx": idx_it,
                            "label": lbl,
                            "text": sub_txt,
                            "search_text": f"{lbl} ({a['title']})\n{clean_for_search(preamble)}\n{clean_for_search(it_txt)}",
                        })
                else:
                    chunks.append({
                        **meta,
                        "article": a["article"],
                        "para_no": p["no"] if has_para_no else None,
                        "chunk_type": "paragraph",
                        "label": lbl,
                        "text": full_p_text,
                        "search_text": f"{lbl} ({a['title']})\n{clean_for_search(full_p_text)}",
                    })

        for ax in law["annexes"]:
            if not ax["label"].startswith("별표"):
                continue
            parsed = annex_path(law["mst"], ax["label"])
            if config.USE_PARSED_ANNEX and parsed.exists():
                body, src = parsed.read_text(encoding="utf-8"), "document-parse"
            else:
                body, src = ax["text"], "law.go.kr 원문(괘선 표)"

            refs = annex_refs.get(ax["label"], [])
            ref_line = f" (관련 본문: {', '.join(refs[:5])})" if refs else ""
            ax_head = f"[{law['name']} {ax['label']} {ax['title']}{ref_line}]"

            parts = chunk_annex_markdown(body, max_chars=1300) if src == "document-parse" else [
                body[i:i + 1800] for i in range(0, max(len(body), 1), 1800)
            ]
            for part in parts:
                txt = f"{ax_head}\n{part}"
                chunks.append({
                    **meta,
                    "article": ax["label"],
                    "para_no": None,
                    "chunk_type": "annex",
                    "label": f"{law['name']} {ax['label']}",
                    "text": txt,
                    "search_text": clean_for_search(txt),
                    "annex_source": src,
                })

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
    texts_to_embed = [c.get("search_text", c["text"]) for c in chunks]
    vecs = upstage.embed(texts_to_embed, config.EMBED_PASSAGE)
    np.save(config.DATA / f"vectors_{unit}.npy", np.array(vecs, dtype=np.float32))
    print("임베딩 저장 완료")


if __name__ == "__main__":
    main()
