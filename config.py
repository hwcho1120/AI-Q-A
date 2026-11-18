"""스타터 킷 공통 설정.

여기 값만 바꿔도 시연의 '전/후' 장면을 재현할 수 있습니다.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

# .env 파일이 있으면 읽습니다 (python-dotenv가 없으면 환경변수만 사용)
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass

UPSTAGE_API_KEY = os.environ.get("UPSTAGE_API_KEY", "")
UPSTAGE_BASE_URL = "https://api.upstage.ai/v1"

# 법제처 OPEN API 인증값(OC). 'test'로도 응답하지만, 대회용으로는 open.law.go.kr에서 개인 OC를 신청하세요.
LAW_OC = os.environ.get("LAW_OC", "test")

# ---- 대상: 서울특별시 조례 6종 (자치구 조례는 대상이 아닙니다) ----
TARGET_ORDINANCES = [
    "서울특별시 주차장 설치 및 관리 조례",
    "서울특별시 동물보호 조례",
    "서울특별시 자전거이용 활성화에 관한 조례",
    "서울특별시 도시공원 조례",
    "서울특별시 청년 기본 조례",  # 확인 필요: 대회 운영진이 말한 '청년' 조례가 이것인지
    "서울특별시 금연환경 조성 및 간접흡연 피해방지조례",
]
TARGET_ORG = "서울특별시"      # 지자체기관명이 정확히 이것인 것만 받습니다
BASE_DATE = "20261001"          # 기준 시점: 2026년 10월 1일 시행 조례

# ---- 모델 ----
CHAT_MODEL = "solar-pro4"
EMBED_PASSAGE = "solar-embedding-2-passage"   # 조문 조각용
EMBED_QUERY = "solar-embedding-2-query"       # 질문용 (질문과 조각은 서로 다른 모델)

# ---- 시연 스위치 ----
# 2-4 시연: "article"(조 단위) → "paragraph"(항 단위)로 바꾸면 인용이 항까지 맞기 시작합니다
CHUNK_UNIT = os.environ.get("CHUNK_UNIT", "paragraph")
# 2-2 시연: 별표를 Document Parse 결과로 읽을지(True), 법제처 원문 괘선 텍스트로 읽을지(False)
USE_PARSED_ANNEX = os.environ.get("USE_PARSED_ANNEX", "1") == "1"
# 2-5 시연: 근거 검증 장치를 켤지
USE_VERIFY = os.environ.get("USE_VERIFY", "1") == "1"

TOP_K = 8
W_EMBED = 0.7                 # 혼합 점수에서 임베딩 비중 (나머지는 키워드)
USE_REWRITE = os.environ.get("USE_REWRITE", "1") == "1"   # 질문 재작성(쉬운 말 → 조문 용어)
REWRITE_WEIGHT = 0.95         # 재작성 질문 점수에 곱하는 값 (원 질문을 약간 우대)
SIBLING_FROM_TOP = 3          # 상위 몇 개 근거에 대해 같은 조의 다른 항을 덧붙일지
MAX_CONTEXT = 16              # LLM에 넘기는 근거 최대 개수 (검색 top_k + 참조 조항 + 형제 항)
CITE_ANNEX_PARENT = os.environ.get("CITE_ANNEX_PARENT", "1") == "1"   # 별표 인용 시 그 별표를 정한 본문 조항도 인용
# 인용한 항이 같은 조의 다른 항을 가리키면("제1항에 따라") 그 항도 인용.
# 1,200문항 측정 결과 Recall 개선은 미미(TS3 +0.0024)하고 Precision(-0.065~-0.072)·Exact Match(-10%p) 손실이 커 기본 OFF.
CITE_SIBLING_REFS = os.environ.get("CITE_SIBLING_REFS", "0") == "1"
CITE_SIBLING_MAX_TOTAL = 5    # 위 보강을 포함한 인용 최대 개수
MAX_CITATIONS = 3           # 인용 최대 개수 (여러 개 냈을 때 채점 규칙은 운영진 공지 확인)
USE_REPAIR =os.environ.get("USE_REPAIR", "1") == "1"     # 검증 실패 시 한 번 고쳐 쓰기
# 검색 점수가 이보다 낮으면 '확인 불가' 후보로 봅니다. TODO: 공개 예시로 직접 맞춰 보세요.
MIN_SCORE = 0.25

# 대회 제한은 문항당 90초. 여유를 두고 이 시간을 넘기면 '확인 불가'로 돌려줍니다.
ANSWER_TIMEOUT_SEC = 75
RRF_K = 10                    # 순위 합산(RRF) 상수: 작을수록 각 검색어의 상위권을 더 우대
RRF_DEPTH = 60                # 검색어마다 몇 위까지 합산에 넣을지
USE_HYDE = os.environ.get("USE_HYDE", "1") == "1"         # 가상 조문(HyDE)도 검색어로 사용
LAW_FOCUS = os.environ.get("LAW_FOCUS", "1") == "1"       # 상위권 점수 합이 가장 큰 조례 하나에서만 근거 선택
REF_MAX_PER = 2               # 참조 조항 확장 시 조 하나당 최대 항 수
LAW_FOCUS_RATIO = 0.5         # 1등 조례 점수의 이 비율 이상인 조례는 함께 남김
# 답변 단계 추론 모드: "" 끔 / "low" / "medium" / "high"
# 깨끗한 150문항 A/B(11/17): low → 근거 0.969 그대로, 평균 17.5s·최대 61.8s / 끔 → 근거 0.969, 평균 5.7s·최대 12.8s → 기본값 끔
ANSWER_REASONING = os.environ.get("ANSWER_REASONING", "")
CONSISTENCY_N = int(os.environ.get("CONSISTENCY_N", "1"))   # 답을 동시에 몇 개 만들어 다수결할지 (1이면 끔)
CONSISTENCY_GRACE_SEC = 5.0   # 첫 답 이후 나머지 답을 기다리는 최대 시간 (응답 시간 상한)
REPAIR_BUDGET_SEC = 45.0      # 질문 받은 뒤 이 시간이 지났으면 고쳐 쓰기를 건너뜀 (90초 제한 대비)
USE_ANSWER_FRAMES = os.environ.get("USE_ANSWER_FRAMES", "0") == "1"   # 질문 유형별 답변 틀 (148문항 측정에서 0.920→0.876으로 하락해 끔)
