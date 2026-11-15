"""라이브 점검: 시스템이 낸 답변을 법제처 OPEN API 원문과 직접 대조합니다.

정답표(gold)와 비교하는 check.py와 달리, 정답표 없이 '실제 법령'에 비춰 봅니다.
  1. 인용 조·항이 법제처 원문에 실제로 있는가
  2. 답변 속 숫자(금액·비율·나이·일수)가 인용 조항 원문에 있는가
  3. effective_date가 법제처 시행일과 같고, 기준일(BASE_DATE) 이전 시행분인가
  4. source_url이 열리는가

실행:  python -X utf8 verify_live.py                       (results/의 최신 diag_*.json 전부)
       python -X utf8 verify_live.py results/diag_x.json
"""
import glob
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter

import requests

import config
from fetch_ordinances import CIRCLED, SEARCH_URL, SERVICE_URL

NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
CITE = re.compile(r"^(서울특별시 .+?(?:조례|피해방지조례))\s+(제\d+조(?:의\d+)?|별표(?:\s*\d+)?)(?:\s+제(\d+)항)?$")

_law_cache = {}


def fetch_live(name):
    """법제처 API에서 서울특별시 조례를 검색해 현행본 본문을 받아 조·항·별표 텍스트로 정리합니다."""
    if name in _law_cache:
        return _law_cache[name]
    r = requests.get(SEARCH_URL, params=dict(OC=config.LAW_OC, target="ordin", type="XML",
                                             query=name, display=100), timeout=30)
    root = ET.fromstring(r.content)
    hits = [(l.findtext("자치법규일련번호"), l.findtext("시행일자")) for l in root.findall("law")
            if l.findtext("지자체기관명") == "서울특별시" and l.findtext("자치법규명") == name]
    if not hits:
        raise RuntimeError(f"법제처 검색 실패: {name} / {root.findtext('result')}")
    mst, eff = sorted(hits, key=lambda h: h[1], reverse=True)[0]
    x = ET.fromstring(requests.get(SERVICE_URL, params=dict(OC=config.LAW_OC, target="ordin",
                                                            MST=mst, type="XML"), timeout=60).content)
    arts = {}
    for jo in x.find("조문").findall("조"):
        text = (jo.findtext("조내용") or "").strip()
        m = re.match(r"^\s*(제\d+조(?:의\d+)?)", text)
        if jo.findtext("조문여부") != "Y" or not m:
            continue
        paras, cur = {}, None
        # 제1항(①)은 '제7조(제목) ①…'처럼 조 제목과 같은 줄에 오므로 제목을 떼고 나눕니다
        body = re.sub(r"^\s*제\d+조(?:의\d+)?\s*(?:\([^)]*\))?\s*", "", text)
        for line in body.splitlines():
            s = line.strip()
            if s and s[0] in CIRCLED:
                cur = CIRCLED[s[0]]
            if cur:
                paras[cur] = paras.get(cur, "") + "\n" + s
        arts[m.group(1)] = {"text": text, "paras": paras}
    annex = {}
    for b in x.findall("별표단위"):
        no = int(b.findtext("별표번호") or 0)
        label = f"별표 {no}" if no else "별표"
        annex[label] = (b.findtext("별표내용") or "")
    info = {"mst": mst, "effective": f"{eff[:4]}-{eff[4:6]}-{eff[6:]}", "arts": arts, "annex": annex}
    _law_cache[name] = info
    return info


def _nums(s):
    return {n.replace(",", "") for n in NUM.findall(s)}


ARTREF = re.compile(r"제\s?\d+\s?(?:조(?:의\s?\d+)?|항|호)")


def check_answer(resp, url_ok_cache, question=""):
    """한 답변을 라이브 원문과 대조. 문제 목록을 돌려줍니다 (빈 목록이면 통과)."""
    issues = []
    if resp.get("answer", "").startswith("확인 불가"):
        return issues
    live_texts = []
    for c in resp.get("citation", []):
        m = CITE.match(c)
        if not m:
            issues.append(f"인용 형식 해석 불가: {c}")
            continue
        name, art, para = m.group(1), m.group(2).replace(" ", " "), m.group(3)
        try:
            law = fetch_live(name)
        except Exception as e:  # noqa: BLE001
            issues.append(str(e))
            continue
        if art.startswith("별표"):
            key = art if art in law["annex"] else art.replace("별표", "별표 ").replace("  ", " ")
            if key not in law["annex"]:
                issues.append(f"별표 없음: {c}")
            else:
                live_texts.append(law["annex"][key])
            continue
        a = law["arts"].get(art)
        if not a:
            issues.append(f"조 없음: {c}")
            continue
        if para:
            if int(para) not in a["paras"]:
                issues.append(f"항 없음: {c}")
                continue
            live_texts.append(a["paras"][int(para)])
        else:
            live_texts.append(a["text"])
        if resp.get("effective_date") and law["arts"] and resp["effective_date"] != law["effective"] \
                and c == resp["citation"][0]:
            issues.append(f"시행일 불일치: 답 {resp['effective_date']} / 법제처 {law['effective']}")
        if law["effective"].replace("-", "") > config.BASE_DATE:
            issues.append(f"기준일 이후 시행본: {law['effective']}")
    # 답변 숫자 대조 (조항 번호·연도 같은 것은 제외하고 3자리 이상 또는 %·세·일·원 앞 숫자 위주)
    if live_texts:
        live_all = "\n".join(live_texts)
        live_n = _nums(live_all) | _nums(question)        # 질문에 나온 숫자(나이 등)는 허용
        for n in _nums(ARTREF.sub("", resp["answer"])):   # '제20조제1항제2호' 같은 조항 번호는 제외
            if n in live_n or len(n) <= 1:
                continue
            # 100분의 50 ↔ 50% 같은 표기 변환 허용
            if n in re.findall(r"100분의\s*(\d+)", live_all):
                continue
            issues.append(f"원문에 없는 숫자: {n}")
    u = resp.get("source_url")
    if u:
        if u not in url_ok_cache:
            try:
                url_ok_cache[u] = requests.get(u, timeout=20).status_code == 200
            except Exception:  # noqa: BLE001
                url_ok_cache[u] = False
        if not url_ok_cache[u]:
            issues.append(f"source_url 열리지 않음: {u}")
    return issues


def main():
    files = sys.argv[1:] or [sorted(glob.glob(str(config.ROOT / "results" / f"diag_{t}_*.json")))[-1]
                             for t in ("eval_questions", "sample_questions")]
    url_cache, stat = {}, Counter()
    for f in files:
        print(f"\n### {f}")
        for q in json.load(open(f, encoding="utf-8")):
            for k, run in enumerate(q["runs"]):
                resp = run["response"]
                stat["답변"] += 1
                if resp["answer"].startswith("확인 불가"):
                    stat["확인 불가"] += 1
                    continue
                stat["근거 있는 답변"] += 1
                issues = check_answer(resp, url_cache, q["question"])
                if issues:
                    stat["문제 있음"] += 1
                    print(f"{q['id']}#{k}  {q['question'][:30]}")
                    for i in issues:
                        print("    -", i)
                    print("    답:", resp["answer"][:120])
    print("\n== 라이브 법령 정보")
    for name, law in _law_cache.items():
        print(f"  {name}  MST {law['mst']}  시행 {law['effective']}  조 {len(law['arts'])}  별표 {len(law['annex'])}")
    print("\n== 요약", dict(stat))


if __name__ == "__main__":
    main()
