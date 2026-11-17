# 서울시 조례 Q&A 스타터 킷

공공 법령 Q&A AI 경진대회용 **기준선(baseline)** 코드입니다.
그대로 돌리면 대회 형식에 맞는 API가 뜹니다. 점수를 올리는 일은 여러분 몫입니다.

```
법제처 OPEN API ─▶ 조·항 단위 조각 ─▶ 검색(임베딩+키워드) ─▶ Solar 답변 ─▶ 검증 ─▶ 대회 JSON
       │                 ▲
       └─ 별표(요금표) ─▶ Document Parse
```

## 빠른 시작 (10분)

```bash
pip install -r requirements.txt
cp .env.example .env               # UPSTAGE_API_KEY 입력 (console.upstage.ai/api-keys)

python -X utf8 fetch_ordinances.py # 1. 조례 6종 수집 → data/ordinances.json
python -X utf8 parse_annex.py      # 2. 별표를 Document Parse로 읽기 → data/annex/
python -X utf8 build_index.py      # 3. 조각 만들기 + 임베딩
python -X utf8 check.py            # 4. 공개 예시로 셀프 채점
uvicorn app:app --port 8000        # 5. 제출용 API 서버
```

Windows에서 한글이 깨지면 `python` 대신 `python -X utf8`을 쓰세요.

## 파일 = 강의 시연 장면

| 파일 | 하는 일 | 강의에서 본 장면 |
|---|---|---|
| `fetch_ordinances.py` | 법제처 API로 조례 수집, 조·항 분리 | 검색 결과에 'OO구' 조례가 섞여 나오는 것 |
| `parse_annex.py` | 별표 파일을 Document Parse로 읽기 | 괘선 표 vs DP 표 (`demo.py annex`) |
| `build_index.py` | 조각마다 인용 라벨 붙이고 임베딩 | 조 단위 vs 항 단위 (`demo.py chunk`) |
| `rag.py` | 검색 → Solar → 검증 → JSON | LLM 단독 vs RAG (`demo.py llm-only`), 함정 (`demo.py trap`) |
| `app.py` | `POST /ask` 서버, 90초·오류 방어 | 제출 형태로 감싸기 |
| `check.py` | 대회 채점 방식 흉내 | 셀프 채점표 |
| `config.py` | 모델명, 대상 조례, 시연 스위치 | 전/후 전환 |

## 설계 원칙 세 가지 (바꾸기 전에 먼저 이해하세요)

1. **LLM은 근거 번호만 고릅니다.** `citation` 문자열은 코드가 조각 라벨로 만듭니다. 표기 실수와 지어낸 조항이 원천 차단됩니다.
2. **링크·시행일은 코드가 채웁니다.** `source_url`은 시행 버전이 고정된 법제처 링크, `effective_date`는 API의 시행일자입니다.
3. **의심스러우면 '확인 불가'.** 검색 점수가 낮거나, Groundedness Check가 `notGrounded`면 답하지 않습니다.

## 점수 올리는 지점 지도

기준선은 일부러 여기까지만 했습니다. `TODO`를 검색해 보세요.

| 채점 항목 | 지금 상태 | 해 볼 만한 것 |
|---|---|---|
| 근거 정확도 (40) | 항 단위 조각, 혼합 검색 | 호(1. 2.) 단위까지 쪼개되 라벨은 항으로 유지 · 질문 재작성(쉬운 말 → 조문 용어) · 재순위화 · `TOP_K` 조정 |
| 근거 정확도: 별표 | DP 결과를 1,800자씩 자름 | 표 행 단위로 쪼개기 · 별표가 연결된 조(예: 제6조 제1항)를 함께 넣기 |
| 여러 조항 결합 | 근거 여러 개 선택 허용 | 원칙 조항 + 예외·감면 조항을 함께 찾도록 2단계 검색 · `reasoning_effort` 켜 보기 |
| 신뢰도 (20) | 점수 기준선 + Groundedness | `MIN_SCORE`를 공개 예시로 맞추기 · `notSure` 처리 방식 정하기 |
| 답변 품질 (20) | 2~4문장 쉬운 말 | 조건 판단형은 "네/아니요"로 시작 · 숫자와 단서 조항(다만, 예외) 빠뜨리지 않기 |
| 서비스 시연 (20) | API만 있음 | 일반 시민용 화면, 인용 조항 클릭 시 원문 열기 |

## 제출 전 체크리스트

- [ ] `check.py --url <내 서버>/ask`로 공개 예시를 돌려 **오류 0건, 모두 90초 이내**
- [ ] 함정 문항에서 `"answer": "확인 불가", "citation": []`가 나온다
- [ ] 인용에 'OO구' 조례·규칙이 하나도 없다
- [ ] `source_url`과 `effective_date`가 모든 답에 채워져 있다 (확인 불가 제외)
- [ ] 서버가 마감 후 채점 시점까지 **켜져 있다** (운영진이 1회 실행)
- [ ] 동시에 여러 문항이 들어와도 429 오류로 죽지 않는다

## 자주 막히는 곳

| 증상 | 원인 | 해결 |
|---|---|---|
| `invalid_api_key` | 키가 비었거나 틀림 | `.env`의 `UPSTAGE_API_KEY` 확인 |
| `400` 모델 오류 | 모델 이름 | `solar-pro4` (pro3·pro2·mini는 2026-10-30 종료) |
| `429 too_many_requests` | 호출 속도 한도 | 자동 재시도가 들어 있습니다. 팀원이 한 키를 같이 쓰면 더 잘 걸립니다 |
| 별표 파싱 실패 | 파일 형식 | 한글에서 PDF로 저장해 다시 시도 |
| 한글 깨짐 | Windows 기본 인코딩(cp949) | `python -X utf8` |
| 법제처 API 응답 없음 | OC 미등록 | open.law.go.kr에서 OPEN API 신청 후 `LAW_OC`에 입력 |

## 데이터 기준

- 기준 시점: 2026-10-01 시행 조례 (`config.BASE_DATE`)
- 대상: `config.TARGET_ORDINANCES`의 서울특별시 조례 6종. 자치구 조례는 대상이 아니며 인용하면 오답입니다.
- `sample_questions.json`은 강사가 원문을 보고 만든 **연습용 예시**입니다. 운영진의 공개 예시 문항이 나오면 같은 형식으로 추가해 쓰세요.
