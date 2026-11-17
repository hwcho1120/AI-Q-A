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

TOP_K = 6
# 근거 후보에 1순위 조례의 '(정의)' 조항을 항상 포함 (정답이 정의 조항인 오인용 대응). 기본 ON.
# 측정(2026-11-17, 정의 문항 89 + 회귀 200): 정의 문항 Recall 0.792→0.927, 회귀 일반문항 0.978→0.958(노이즈 범위, 원인 1건)
USE_DEFINITIONS = os.environ.get("USE_DEFINITIONS", "1") == "1"
# 인용한 항과 같은 조의 다른 항이 검색 결과에 있으면 함께 인용. 기본 OFF.
# 측정: Recall 0.943→0.976 이지만 Precision 0.802→0.562, Exact Match 59.6%→22.8%. 불필요한 인용을 감점하는 채점이면 큰 손해라 OFF
USE_PARA_COMPLETE = os.environ.get("USE_PARA_COMPLETE", "0") == "1"
PARA_COMPLETE_MAX = 2
# 별표를 인용할 때 그 별표가 달린 조·항(별표 제목의 '제N조제M항 관련')도 함께 인용
USE_ANNEX_REFS = os.environ.get("USE_ANNEX_REFS", "1") == "1"
# 답변 후 Groundedness 검증(solar-pro4 판정). 기본 OFF.
# 100문항 샘플 2종(seed 0/1) 측정: 켜면 근거 정확도 -6~7%p, 함정 +8~12%p. 정상 답변을 과하게 막아
# (막힌 13건 모두 검색은 정답) 배점 가정(정확도 40 / 함정 ~10)에서 OFF가 약 1.6점 유리 (2026-11-17).
# 함정 비중이 크다고 판단되면 USE_GROUNDED_CHECK=1 로 켠다.
USE_GROUNDED_CHECK = os.environ.get("USE_GROUNDED_CHECK", "0") == "1"
# 답이 '규정이 없다'고 말하면 found와 무관하게 확인 불가 처리
USE_NORULE_GUARD = os.environ.get("USE_NORULE_GUARD", "1") == "1"
# 검색 개선 스위치 (전/후 비교용)
USE_REWRITE = os.environ.get("USE_REWRITE", "1") == "1"     # 질문을 조문 용어로 다시 써서 함께 검색
# 상위 결과와 같은 조의 다른 항을 추가. 11문항 3회 평균에서 효과 없어 기본 OFF (2026-11-15 측정)
USE_SIBLINGS = os.environ.get("USE_SIBLINGS", "0") == "1"
SIBLING_TOP = 3        # 상위 몇 개 결과의 형제 항을 붙일지
SIBLING_MAX = 4        # 형제 항 최대 추가 개수
# 검색 점수가 이보다 낮으면 '확인 불가' 후보로 봅니다. TODO: 공개 예시로 직접 맞춰 보세요.
MIN_SCORE = 0.25

# 대회 제한은 문항당 90초. 여유를 두고 이 시간을 넘기면 '확인 불가'로 돌려줍니다.
ANSWER_TIMEOUT_SEC = 75
# API 호출 하나의 타임아웃과, 재시도에 쓸 수 있는 총 시간. 한 문항에 호출이 3~4번 들어가므로 합쳐서 90초 안에 끝나게 잡는다
CALL_TIMEOUT_SEC = 30
RETRY_BUDGET_SEC = 25
