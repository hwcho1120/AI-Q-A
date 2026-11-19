# 서울특별시 조례 Q&A AI 서비스

서울특별시 조례 6종(주차장·동물보호·자전거·도시공원·청년 기본·금연)을 근거로 시민·공무원의 질문에 **조·항·별표 단위의 정확한 근거, 원문 링크, 시행일**과 함께 답변하고, 근거가 없는 질문에는 **`확인 불가`**로 응답하는 RAG 기반 Q&A API 및 웹 서비스입니다.

```
법제처 OPEN API ──▶ 조·항·호 분리 + 별표(Document Parse) ──▶ 831청크 색인(Solar Embedding 2)
                                                                       │
질문 ─▶ 재작성·HyDE ─▶ RRF 혼합 검색(임베딩+BM25) + 2-hop 참조·링크 확장 ─▶ Solar Pro4 답변 ─▶ 검증·보정 ─▶ JSON / 웹 UI
```

## 1. 실행 방법

### 1) 패키지 설치 및 환경 변수 설정
```bash
pip install -r requirements.txt
cp .env.example .env
```
`.env` 파일에 발급받은 Upstage API 키를 입력합니다 (`.env`는 `.gitignore`에 포함되어 저장소에 올라가지 않습니다).
```env
UPSTAGE_API_KEY=your_upstage_api_key
LAW_OC=test
# (선택) /admin 관리자 페이지 보호용 토큰
# ADMIN_TOKEN=your_admin_token
```

### 2) 검색 색인(임베딩 벡터) 생성
조례 원문(`data/ordinances.json`), 별표 파싱본(`data/annex/`), 청크(`data/chunks_paragraph.json`)는 저장소에 포함되어 있습니다. 아래 명령으로 임베딩 파일(`data/vectors_paragraph.npy`)을 생성합니다.
```bash
python -X utf8 build_index.py
```
*(원천 데이터 수집·별표 파싱부터 다시 수행하려면 `fetch_ordinances.py` → `parse_annex.py` → `build_index.py` 순서로 실행합니다.)*

### 3) API · 웹 서버 실행
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```
- **채점용 API 엔드포인트**: `POST /ask` (`{"question": "..."}`)
- **시민용 질문 웹 화면**: `GET /` ([static/index.html](static/index.html))
- **관리자 검토·승인 화면**: `GET /admin` ([static/admin.html](static/admin.html))
- **헬스 체크**: `GET /health`

### 4) 동작 검증
```bash
# 공개 예시(11문항) / 확장 평가셋 로컬 또는 라이브 서버 점검
python -X utf8 check.py
python -X utf8 check.py --url http://localhost:8000/ask
```

## 2. 시스템 구성 및 단계별 Upstage 모델

| 단계 | 구현 모듈 | 사용 모델 / 방식 |
|---|---|---|
| **수집** | `fetch_ordinances.py` | 법제처 OPEN API (`서울특별시` 조례 6종, 자치구 조례 배제) |
| **문서 파싱 · 청킹** | `parse_annex.py`, `build_index.py` | **Upstage Document Parse** (별표 Markdown 표 구조 복원) · 조/항 본청크 + 호/표행 보조 청크(총 831청크) |
| **검색 · 확장** | `rag.py` (`rewrite_queries`, `retrieve`, `expand_*`) | **Solar Pro4** 질문 재작성/HyDE + **Solar Embedding 2** (`solar-embedding-2-passage` / `solar-embedding-2-query`) + 2-gram BM25 RRF 혼합 + 형제 항·2-hop 참조·별표/정의 링크 확장 |
| **답변 생성** | `rag.py`, `upstage.py` | **Solar Pro4** 구조화 출력(JSON Schema `enum`으로 조항 라벨만 선택, 원문 링크·시행일은 코드 고정 주입) |
| **검증 · 보정** | `rag.py`, `upstage.py` | 점수 하한(`MIN_SCORE`) + 부재/위임·준용 판별 + 수식 `eval` 검산·원문 숫자 대조 + **Solar Pro4** 문장별 Groundedness 검증 + 실패 건 조건부 `reasoning_effort="low"` Repair |
| **서비스 · 관제** | `app.py`, `ui_api.py`, `review.py` | 75초 타임아웃·예외 래핑(`HTTP 200` 보장), API 재시도(`_retry`), Review → Accept / Deny → Approve 관리자 워크플로 |

## 3. 베이스라인 대비 주요 개선 사항

- **검색·문맥 보강**: 일상어를 법률 용어로 바꾸는 다중 쿼리 재작성 + 가상 조문(HyDE) + RRF 합산, `LAW_FOCUS` 조례 집중 필터, 2-hop 조문 참조 및 별표↔본문 양방향 링크 확장.
- **인용 정밀도(Precision/EM) 최적화**: 문맥 제공용 형제 항(`expand_siblings`)은 유지하되 불필요한 형제 항 자동 출력 인용(`CITE_SIBLING_REFS`)을 비활성화하고 `MAX_CITATIONS=3`으로 제한해 과잉 인용 감점을 방지.
- **위임·준용 및 계산 정확도 개선**: 조례 내 시장·규칙·고시 위임 및 타법 준용 조항 안내 규칙 정교화, 날짜 연산 오판(`_DATE_LIT`) 및 대조 숫자 검증 버그 수정, 검증 실패 시에만 `reasoning_effort="low"`로 보정해 평균 응답 속도와 정확도를 동시 확보.
- **오탐·안정성 강화**: 자치구명 포함 정상 시민 질문까지 차단하던 과도한 사전 정규식 제거, 빈 질의 방어, API 일시 오류 재시도 및 서버 시작 시 색인 워밍업(`lifespan`) 적용.
