# KOSPI200 투트랙 퀀트 투자 전략 파이프라인

KOSPI200 주식을 대상으로 한 **투트랙(Two-Track)** 퀀트 투자 전략 시스템입니다.

## 🎯 프로젝트 핵심 목적

본 프로젝트는 **두 가지 독립적인 트랙**으로 구성되어 이용자에게 정보를 제공합니다:

1. **Track A (랭킹 엔진)**: 피처들로 KOSPI200의 랭킹을 산정하여 이용자에게 제공
2. **Track B (투자 모델)**: 랭킹을 기반으로 다양한 투자모델 예시를 만들어 이용자에게 정보 제공

두 트랙은 **독립적으로 실행 가능**하며, 각각 다른 목적을 가집니다.

## 프로젝트 개요

### 🎯 Track A: 랭킹 엔진 (Ranking Engine)
**목적**: 피처들로 KOSPI200의 랭킹을 산정하여 이용자에게 제공

- **L8**: 랭킹 엔진 실행
  - 피처 기반 랭킹 생성 (단기/장기/통합)
  - 피처 가중치 및 정규화를 통한 종목 랭킹 산정
  - `ranking_daily` 산출물 생성
- **L11**: UI Payload Builder
  - 랭킹 데이터를 UI에서 사용할 수 있는 형태로 변환
  - 투자 성격(방어/균형/민감)별 랭킹 제공

**산출물**: `ranking_daily` (날짜별 종목 랭킹)

### 💼 Track B: 투자 모델 (Investment Model)
**목적**: 랭킹을 기반으로 다양한 투자모델 예시를 만들어 이용자에게 정보 제공

- **L6R**: 랭킹 스코어 변환
  - 랭킹 데이터를 백테스트용 스코어로 변환
  - 단기/장기/통합 랭킹을 투자 신호로 활용
- **L7**: 백테스트 실행
  - BT20 (20일 보유 기간) 전략
  - BT120 (120일 보유 기간) 전략
  - 다양한 투자 모델 예시 제공

**산출물**: `bt_metrics`, `bt_returns`, `bt_equity_curve` (백테스트 성과 지표)

---

## 핵심 개념

### Track A: 랭킹 엔진
- **랭킹**: 피처 기반 종목 랭킹 (단기/장기/통합)
- **피처**: 기술적 지표, 재무 지표, 뉴스 감성, ESG 등
- **투자 성격**: 방어/균형/민감 (α 파라미터)
- **신호 결합**: 단기/장기 랭킹 결합 비중 (γ 파라미터)

### Track B: 투자 모델
- **BT20**: 20일 보유 기간 전략 (단기 투자)
- **BT120**: 120일 보유 기간 전략 (장기 투자)
- **4개 전략**:
  1. BT20 통합 모델 (`l7_bt20_ens`): 단기 보유(20일) + 통합 랭킹
  2. BT20 분리 모델 (`l7_bt20_short`): 단기 보유(20일) + 단기 랭킹만
  3. BT120 통합 모델 (`l7_bt120_ens`): 장기 보유(120일) + 통합 랭킹
  4. BT120 분리 모델 (`l7_bt120_long`): 장기 보유(120일) + 장기 랭킹만
- **포트 배분**: BT20/BT120 포트 자본 배분 비중 (β 파라미터)

### 투트랙 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│              공통 데이터 준비 (Shared Data)                  │
│  L0: 유니버스 구성 (KOSPI200 멤버십)                         │
│  L1: OHLCV 데이터 다운로드 + 기술적 지표 계산                │
│  L2: 재무 데이터 로드 (DART)                                │
│  L3: 패널 병합 (OHLCV + 재무 + 뉴스 + ESG)                  │
│  L4: Walk-Forward CV 분할 및 타겟 생성                      │
└─────────────────────────────────────────────────────────────┘
                        ↓
        ┌───────────────┴───────────────┐
        │                               │
┌───────▼────────┐            ┌─────────▼──────────┐
│ Track A        │            │ Track B             │
│ 랭킹 엔진      │            │ 투자 모델           │
├───────────────┤            ├─────────────────────┤
│ 목적:         │            │ 목적:               │
│ 피처 기반     │            │ 랭킹 기반 투자      │
│ 랭킹 산정     │            │ 모델 예시 제공      │
│               │            │                     │
│ L8: 랭킹 엔진 │            │ L6R: 랭킹 스코어    │
│   - 단기 랭킹 │            │   변환              │
│   - 장기 랭킹 │            │ L7: 백테스트 실행   │
│               │            │   - BT20 (20일)     │
│ L11: UI       │            │   - BT120 (120일)   │
│   Payload     │            │                     │
│   생성        │            │                     │
│               │            │                     │
│ 산출물:       │            │ 산출물:             │
│ - ranking_    │            │ - bt_metrics        │
│   short_daily │            │ - bt_returns        │
│ - ranking_    │            │ - bt_equity_curve    │
│   long_daily  │            │ - bt_positions       │
└───────────────┘            └─────────────────────┘
        │                               │
        └───────────────┬───────────────┘
                        ↓
            [이용자에게 정보 제공]
```

### 투트랙 실행 흐름

1. **공통 데이터 준비** (한 번만 실행)
   ```bash
   python scripts/run_pipeline_l0_l7.py
   ```

2. **Track A 실행** (랭킹만 필요한 경우)
   ```bash
   python -m src.pipeline.track_a_pipeline
   ```
   → 랭킹 데이터 생성 (`ranking_short_daily`, `ranking_long_daily`)

3. **Track B 실행** (투자 모델 예시가 필요한 경우)
   ```bash
   # Track A의 랭킹 데이터를 사용하여 백테스트 실행
   python -m src.pipeline.track_b_pipeline bt20_short
   ```
   → 백테스트 성과 지표 생성 (`bt_metrics`, `bt_returns`, etc.)

---

## 디렉토리 구조

```
03_code/
  src/
    data_collection/          # ⭐ 데이터 수집 모듈 (리팩토링 완료)
      __init__.py             # 모듈 초기화
      collectors.py            # 데이터 수집 함수 (L0~L4)
      pipeline.py              # 데이터 수집 파이프라인 클래스
      ui_interface.py          # UI 인터페이스 함수
    tracks/                   # ⭐ 투트랙 구조의 핵심
      track_a/                # Track A: 랭킹 엔진
        stages/
          ranking/
            l8_dual_horizon.py         # 단기/장기 랭킹 분리 생성
            l8_rank_engine.py           # 랭킹 엔진 실행 (레거시)
            ui_payload_builder.py       # UI Payload 생성
      track_b/                # Track B: 투자 모델
        stages/
          modeling/
            l6r_ranking_scoring.py     # 랭킹 스코어 변환 (Track A → Track B)
          backtest/
            l7_backtest.py             # 백테스트 실행
            l7b_sensitivity.py         # 민감도 분석
            l7c_benchmark.py           # 벤치마크 비교
            l7d_stability.py           # 안정성 분석
      shared/                  # 공통 데이터 처리 (Track A/B 모두 사용)
        stages/
          data/                # L0~L4: 데이터 수집 및 전처리
            l0_universe.py             # 유니버스 구성
            l1_ohlcv.py                # OHLCV 다운로드
            l1_technical_features.py   # 기술적 지표 계산
            l2_fundamentals_dart.py    # 재무 데이터 로드
            l3_panel_merge.py          # 패널 병합
            l3n_news_sentiment.py      # 뉴스 감성 분석
            l3e_esg_sentiment.py       # ESG 감성 분석
            l4_walkforward_split.py    # CV 분할 및 타겟 생성
          regime/              # 시장 국면 분석
            l1d_market_regime.py       # 시장 국면 분류
    pipeline/                 # 파이프라인 엔트리 포인트
      track_a_pipeline.py      # Track A 전체 파이프라인 실행
      track_b_pipeline.py      # Track B 전체 파이프라인 실행
      bt20_pipeline.py         # BT20 투자 모델 파이프라인 (편의 래퍼)
      bt120_pipeline.py        # BT120 투자 모델 파이프라인 (편의 래퍼)
    interfaces/               # UI 연동 인터페이스
      ui_service.py            # Flask에서 사용할 랭킹 조회 함수들
    stages/                    # ⚠️ 레거시 스테이지 (하위 호환성 유지)
      data/                    # 레거시 데이터 처리 (src/tracks/shared로 이동 권장)
      modeling/                # 레거시 모델링 (L5, L6 - Track B에서 선택적 사용)
      ranking/                 # 레거시 랭킹 (src/tracks/track_a로 이동 권장)
      backtest/                # 레거시 백테스트 (src/tracks/track_b로 이동 권장)
    utils/                     # 공통 유틸리티
      config.py                # 설정 파일 로딩
      io.py                    # 아티팩트 저장/로드
      validate.py              # 데이터 검증
      quality.py               # 데이터 품질 체크
    components/                # 공통 컴포넌트
      ranking/                 # 랭킹 관련 컴포넌트
      portfolio/               # 포트폴리오 관련 컴포넌트
      backtest/                # 백테스트 관련 컴포넌트
  configs/
    config.yaml                # 메인 설정 파일 (Track A/B 공통 설정)
  data/
    raw/                       # 원시 데이터
    external/                  # 외부 데이터 (뉴스, ESG 등)
    interim/                   # 중간 산출물 (.parquet, .csv)
      ├── universe_k200_membership_monthly.parquet  # L0 산출물
      ├── ohlcv_daily.parquet                      # L1 산출물
      ├── panel_merged_daily.parquet               # L3 산출물
      ├── dataset_daily.parquet                    # L4 산출물
      ├── ranking_short_daily.parquet              # Track A 산출물
      ├── ranking_long_daily.parquet               # Track A 산출물
      ├── rebalance_scores_from_ranking.parquet   # Track B 산출물
      ├── bt_metrics_bt20_short.parquet            # Track B 산출물
      └── ...
    processed/                 # 최종 산출물
  artifacts/
    models/                    # 학습된 모델 (Track B 선택적 사용)
    rankings/                  # 랭킹 결과 (Track A)
    backtests/                 # 백테스트 결과 (Track B)
    reports/                   # 리포트
  scripts/
    run_pipeline_l0_l7.py       # 공통 데이터 준비 (L0~L4) 실행
  backup/                      # 사용하지 않는 파일 보관
  docs/                        # 문서
```

### 디렉토리 구조 설명

- **`src/data_collection/`**: 데이터 수집 모듈 (리팩토링 완료) ⭐ **새로 추가됨**
  - **`collectors.py`**: L0~L4 데이터 수집 함수 (독립 실행 가능)
  - **`pipeline.py`**: 데이터 수집 파이프라인 클래스 (재현성 보장)
  - **`ui_interface.py`**: UI에서 사용할 수 있는 간단한 인터페이스
  - 기존 데이터는 그대로 유지되며, 새로운 데이터 수집만 수행

- **`src/tracks/`**: 투트랙 구조의 핵심 디렉토리
  - **`track_a/`**: Track A 전용 코드 (랭킹 엔진)
  - **`track_b/`**: Track B 전용 코드 (투자 모델)
  - **`shared/`**: Track A/B 공통 코드 (데이터 준비, 시장 국면 분석 등)

- **`src/pipeline/`**: 파이프라인 실행 엔트리 포인트
  - 각 트랙의 전체 파이프라인을 실행하는 함수 제공

- **`src/stages/`**: 레거시 코드 (하위 호환성 유지)
  - 새로운 코드는 `src/tracks/` 구조를 사용 권장
  - 기존 코드와의 호환성을 위해 유지

---

## 설치 방법

### 1. 의존성 설치

```bash
cd 03_code
pip install -r requirements.txt
```

### 2. 설정 파일 확인

`configs/config.yaml` 파일의 경로 설정을 확인하세요:
```yaml
paths:
  base_dir: C:/Users/seong/OneDrive/Desktop/bootcamp/03_code
```

---

## 실행 방법

### 1단계: 공통 데이터 준비 (L0~L4)

**⚠️ 필수**: Track A와 Track B 모두 실행하기 전에 공통 데이터를 먼저 준비해야 합니다.

#### 방법 1: 새로운 데이터 수집 모듈 사용 (권장) ⭐

**리팩토링 완료**: `src/data_collection` 모듈을 통해 데이터 수집이 완전히 분리되었습니다.

```python
# Python에서 직접 호출
from src.data_collection import collect_all_data

# 전체 데이터 수집 (L0~L4)
result = collect_all_data(
    config_path="configs/config.yaml",
    force_rebuild=False,  # 캐시 사용
)

# 단계별 수집
from src.data_collection import (
    collect_universe,
    collect_ohlcv,
    collect_panel,
    collect_dataset,
)

# L0: 유니버스
universe = collect_universe(
    start_date="2016-01-01",
    end_date="2024-12-31",
    config_path="configs/config.yaml",
)

# L1: OHLCV
ohlcv = collect_ohlcv(
    universe=universe,
    start_date="2016-01-01",
    end_date="2024-12-31",
    config_path="configs/config.yaml",
)

# L3: 패널 병합
panel = collect_panel(
    ohlcv_daily=ohlcv,
    config_path="configs/config.yaml",
)

# L4: CV 분할
dataset = collect_dataset(
    panel_merged_daily=panel,
    config_path="configs/config.yaml",
)
```

**파이프라인 클래스 사용**:

```python
from src.data_collection import DataCollectionPipeline

# 파이프라인 생성
pipeline = DataCollectionPipeline(
    config_path="configs/config.yaml",
    force_rebuild=False,
)

# 전체 실행
result = pipeline.run_all()

# 단계별 실행
pipeline.run_l0()  # 유니버스
pipeline.run_l1()  # OHLCV
pipeline.run_l3()  # 패널 병합
pipeline.run_l4()  # CV 분할

# 아티팩트 조회
artifacts = pipeline.get_artifacts()
```

#### 방법 2: 기존 스크립트 사용 (하위 호환성 유지)

```bash
python scripts/run_pipeline_l0_l7.py
```

이 명령은 다음 단계를 실행합니다:
- **L0**: 유니버스 구성 (KOSPI200 멤버십)
- **L1**: OHLCV 데이터 다운로드 + 기술적 지표 계산
- **L2**: 재무 데이터 로드 (DART)
- **L3**: 패널 병합 (OHLCV + 재무 + 뉴스 + ESG)
- **L4**: Walk-Forward CV 분할 및 타겟 생성

**산출물**:
- `universe_k200_membership_monthly`: KOSPI200 멤버십 정보
- `ohlcv_daily`: 일별 OHLCV 데이터 + 기술적 지표
- `panel_merged_daily`: 병합된 패널 데이터
- `dataset_daily`: CV 분할이 완료된 데이터셋

**참고**: 기존 데이터는 그대로 유지되며, 새로운 데이터 수집 모듈은 기존 데이터를 재사용합니다.

### 2단계: Track A 실행 (랭킹 엔진)

**목적**: 피처 기반으로 KOSPI200 종목의 랭킹을 산정하여 이용자에게 제공

**사용 시나리오**:
- 랭킹 정보만 필요한 경우
- UI에서 랭킹을 표시하고 이용자가 직접 투자 결정하는 경우

```bash
# 랭킹 엔진 실행
python -m src.pipeline.track_a_pipeline

# 또는 Python에서 직접 호출
from src.pipeline.track_a_pipeline import run_track_a_pipeline
result = run_track_a_pipeline()
```

**실행 단계**:
1. 공통 데이터 확인 (L0~L4 산출물)
2. **L8**: 랭킹 엔진 실행
   - 단기 랭킹 생성 (`ranking_short_daily`)
   - 장기 랭킹 생성 (`ranking_long_daily`)
3. **L11**: UI Payload 생성 (선택적)

**산출물**:
- `ranking_short_daily`: 단기 랭킹 (날짜별 종목 랭킹)
- `ranking_long_daily`: 장기 랭킹 (날짜별 종목 랭킹)
- `ui_payload`: UI에서 사용할 수 있는 형태의 랭킹 데이터 (선택적)

### 3단계: Track B 실행 (투자 모델)

**목적**: Track A에서 생성한 랭킹을 기반으로 다양한 투자모델 예시를 만들어 이용자에게 정보 제공

**사용 시나리오**:
- 랭킹 기반 투자 전략의 성과를 확인하고 싶은 경우
- 다양한 투자 모델 예시를 제공하고 싶은 경우
- 백테스트 결과를 통해 투자 전략의 유효성을 검증하고 싶은 경우

**⚠️ 필수 조건**: Track A를 먼저 실행하여 랭킹 데이터를 생성해야 합니다.

#### 방법 1: Track B 파이프라인 직접 실행 (권장)

```bash
# Track B 파이프라인 실행 (4가지 전략 중 선택)
python -m src.pipeline.track_b_pipeline bt20_short   # BT20 단기 랭킹
python -m src.pipeline.track_b_pipeline bt20_ens     # BT20 통합 랭킹
python -m src.pipeline.track_b_pipeline bt120_long   # BT120 장기 랭킹
python -m src.pipeline.track_b_pipeline bt120_ens    # BT120 통합 랭킹
```

#### 방법 2: 편의 래퍼 사용

```bash
# BT20 파이프라인 (20일 보유 기간)
python -m src.pipeline.bt20_pipeline short  # 단기 랭킹만 사용
python -m src.pipeline.bt20_pipeline ens     # 통합 랭킹 사용

# BT120 파이프라인 (120일 보유 기간)
python -m src.pipeline.bt120_pipeline long  # 장기 랭킹만 사용
python -m src.pipeline.bt120_pipeline ens   # 통합 랭킹 사용
```

#### Python에서 직접 호출

```python
from src.pipeline.track_b_pipeline import run_track_b_pipeline

# BT20 전략 (단기 랭킹 사용)
result_bt20_short = run_track_b_pipeline(strategy="bt20_short")

# BT120 전략 (장기 랭킹 사용)
result_bt120_long = run_track_b_pipeline(strategy="bt120_long")
```

**실행 단계**:
1. 공통 데이터 확인 (L0~L4 산출물)
2. Track A 산출물 확인 (`ranking_short_daily`, `ranking_long_daily`)
3. **L6R**: 랭킹 스코어 변환 (랭킹 → 백테스트용 스코어)
4. **L7**: 백테스트 실행
   - 포지션 생성
   - 수익률 계산
   - 성과 지표 계산

**산출물**:
- `rebalance_scores_from_ranking`: 랭킹에서 변환된 스코어
- `bt_positions_{strategy}`: 포지션 정보
- `bt_returns_{strategy}`: 일별 수익률
- `bt_equity_curve_{strategy}`: 자산 곡선
- `bt_metrics_{strategy}`: 백테스트 성과 지표 (Sharpe, MDD, CAGR 등)

---

## 캐시 우선 로직

파이프라인은 **캐시 우선** 방식으로 동작합니다:
- 이미 생성된 중간 산출물(`data/interim/*.parquet`)이 있으면 재사용
- 캐시가 없을 때만 해당 단계를 재실행
- `force_rebuild=True` 옵션으로 캐시 무시 가능

---

## UI 연동

Flask 등 UI 프레임워크에서 랭킹과 데이터를 조회할 수 있는 인터페이스를 제공합니다.

### 데이터 수집 인터페이스 (리팩토링 완료) ⭐

**새로운 UI 인터페이스**: `src/data_collection` 모듈에서 간단한 함수로 데이터를 조회할 수 있습니다.

```python
from src.data_collection import (
    get_universe,
    get_ohlcv,
    get_panel,
    get_dataset,
    check_data_availability,
    collect_data_for_ui,
)

# 데이터 가용성 확인
available = check_data_availability()
print(available)
# {
#     "universe": True,
#     "ohlcv": True,
#     "fundamentals": False,
#     "panel": True,
#     "dataset": True,
#     ...
# }

# 유니버스 조회
universe = get_universe()

# OHLCV 조회
ohlcv = get_ohlcv()

# 패널 조회
panel = get_panel()

# 데이터셋 조회
dataset = get_dataset()

# UI용 통합 데이터 수집
result = collect_data_for_ui()
# {
#     "universe": DataFrame,
#     "ohlcv": DataFrame,
#     "panel": DataFrame,
#     "dataset": DataFrame,
#     "available": Dict[str, bool],
#     ...
# }
```

### Track A: 랭킹 조회

```python
from src.interfaces.ui_service import (
    get_short_term_ranking,
    get_long_term_ranking,
    get_combined_ranking,
)

# 단기 랭킹 조회
short_rankings = get_short_term_ranking("2024-12-31", top_k=20)

# 장기 랭킹 조회
long_rankings = get_long_term_ranking("2024-12-31", top_k=20)

# 통합 랭킹 조회
combined_rankings = get_combined_ranking("2024-12-31", top_k=20)
```

### Track B: 투자 모델 성과 조회

```python
from src.utils.io import load_artifact
from pathlib import Path

# 백테스트 메트릭 로드
bt_metrics = load_artifact(Path("data/interim/bt_metrics_bt20"))
print(bt_metrics[bt_metrics["phase"] == "holdout"])
```

### Flask API 예시

```python
from flask import Flask, jsonify, request
from src.interfaces.ui_service import (
    get_short_term_ranking,
    get_long_term_ranking,
    get_combined_ranking,
)
from src.data_collection import (
    get_universe,
    get_ohlcv,
    get_panel,
    check_data_availability,
)

app = Flask(__name__)

# 데이터 수집 API (리팩토링 완료)
@app.get("/api/data/availability")
def data_availability():
    """데이터 가용성 확인"""
    return jsonify(check_data_availability())

@app.get("/api/data/universe")
def universe_data():
    """유니버스 데이터 조회"""
    df = get_universe()
    return jsonify(df.to_dict(orient="records"))

@app.get("/api/data/ohlcv")
def ohlcv_data():
    """OHLCV 데이터 조회"""
    df = get_ohlcv()
    return jsonify(df.to_dict(orient="records"))

# Track A: 랭킹 API
@app.get("/api/ranking/short")
def short_ranking():
    as_of = request.args.get("as_of", default="2024-12-31")
    top_k = int(request.args.get("top_k", 20))
    return jsonify(get_short_term_ranking(as_of, top_k))

@app.get("/api/ranking/long")
def long_ranking():
    as_of = request.args.get("as_of", default="2024-12-31")
    top_k = int(request.args.get("top_k", 20))
    return jsonify(get_long_term_ranking(as_of, top_k))

@app.get("/api/ranking/combined")
def combined_ranking():
    as_of = request.args.get("as_of", default="2024-12-31")
    top_k = int(request.args.get("top_k", 20))
    return jsonify(get_combined_ranking(as_of, top_k))

# Track B: 투자 모델 성과 API
@app.get("/api/backtest/metrics")
def backtest_metrics():
    strategy = request.args.get("strategy", default="bt20_short")
    # 백테스트 메트릭 반환
    ...
```

---

## 설정 파일

`configs/config.yaml`에서 다음 설정을 관리합니다:

### 공통 설정
- **L4**: CV 파라미터 (step_days, embargo_days 등)
- **L5**: 모델 파라미터 (ridge_alpha, target_transform) - Track B 선택적 사용

### Track A 설정
- **L8**: 랭킹 엔진 설정 (피처 그룹, 가중치 등)
- **L11**: UI Payload 설정

### Track B 설정
- **L6**: 가중치 (weight_short, weight_long)
- **L6R**: 랭킹 스코어 변환 설정
- **L7**: 백테스트 설정 (holding_days, top_k, cost_bps)
- **l7_bt20_ens**: BT20 통합 모델 설정
- **l7_bt20_short**: BT20 분리 모델 설정
- **l7_bt120_ens**: BT120 통합 모델 설정
- **l7_bt120_long**: BT120 분리 모델 설정
- **regime**: 시장 국면 분류 설정
  - `enabled`: 시장 국면 기능 활성화 여부
  - `lookback_days`: 국면 판단을 위한 lookback 기간 (기본값: 60일)
  - `neutral_band`: Neutral 구간 임계값 (기본값: 0.05 = ±5%)
  - `use_volume`: 거래량 지표 사용 여부 (기본값: true)
  - `use_volatility`: 변동성 지표 사용 여부 (기본값: true)
  
  **시장 국면 분류 방식**:
  - 외부 API 호출 없이 `ohlcv_daily` 데이터를 사용하여 자동 분류
  - 가격 수익률, 변동성, 거래량 변화율을 종합하여 Bull/Neutral/Bear 판단
  - 각 rebalance 날짜 기준으로 lookback 기간 동안의 지표를 계산

---

## 성과 지표

### Track A: 랭킹 품질 지표
- 랭킹 일관성
- 피처 중요도
- 랭킹 분포 분석

### Track B: 백테스트 성과 지표

**실제 산출 데이터**: `bt_metrics_{strategy}.parquet` 파일에 저장됩니다.

#### 1. 핵심 성과 (Headline Metrics) - ✅ 실제 산출됨

`bt_metrics` 파일에 포함된 지표:

| 지표명 | 컬럼명 | 설명 | 계산 방식 |
|--------|--------|------|-----------|
| **Net Sharpe Ratio** | `net_sharpe` | 리스크 조정 수익률 (비용 차감) | `(평균 수익률 / 표준편차) * sqrt(252/holding_days)` |
| **Gross Sharpe Ratio** | `gross_sharpe` | 리스크 조정 수익률 (비용 차감 전) | 동일 (Gross 수익률 기준) |
| **Net Total Return** | `net_total_return` | 비용 차감 누적 수익률 | `(최종 자산가치 / 초기 자산가치) - 1` |
| **Gross Total Return** | `gross_total_return` | 비용 차감 전 누적 수익률 | 동일 (비용 차감 전) |
| **Net CAGR** | `net_cagr` | 연평균 복리 수익률 (비용 차감) | `(최종 자산가치 / 초기 자산가치)^(1/년수) - 1` |
| **Gross CAGR** | `gross_cagr` | 연평균 복리 수익률 (비용 차감 전) | 동일 (Gross 기준) |
| **Net MDD** | `net_mdd` | 최대 낙폭 (비용 차감) | `min((equity / peak) - 1.0)` |
| **Gross MDD** | `gross_mdd` | 최대 낙폭 (비용 차감 전) | 동일 (Gross 기준) |
| **Net Calmar Ratio** | `net_calmar_ratio` | 수익성 / 최대낙폭 (비용 차감) | `CAGR / \|MDD\|` |
| **Gross Calmar Ratio** | `gross_calmar_ratio` | 수익성 / 최대낙폭 (비용 차감 전) | 동일 (Gross 기준) |
| **Net Volatility (Annualized)** | `net_vol_ann` | 연환산 변동성 (비용 차감) | `std(수익률) * sqrt(252/holding_days)` |
| **Gross Volatility (Annualized)** | `gross_vol_ann` | 연환산 변동성 (비용 차감 전) | 동일 (Gross 기준) |

**추가 메타데이터**:
- `gross_minus_net_total_return_pct`: Gross와 Net 차이 (거래비용 영향도)
- `avg_cost_pct`: 평균 거래비용 (퍼센트)
- `cost_bps`, `cost_bps_used`, `cost_bps_config`: 거래비용 설정값
- `date_start`, `date_end`: 구간 시작/종료일
- `phase`: 구간 구분 (`dev` 또는 `holdout`)

#### 2. 운용 안정성 (Operational Viability) - ✅ 실제 산출됨

| 지표명 | 컬럼명 | 설명 | 계산 방식 |
|--------|--------|------|-----------|
| **Avg Turnover (Oneway)** | `avg_turnover_oneway` | 평균 일방 회전율 | `평균(매일 포트폴리오 변경 비율)` |
| **Net Hit Ratio** | `net_hit_ratio` | 승률 (비용 차감) | `(수익 > 0인 리밸런싱 수) / 전체 리밸런싱 수` |
| **Gross Hit Ratio** | `gross_hit_ratio` | 승률 (비용 차감 전) | 동일 (Gross 기준) |
| **Net Profit Factor** | `net_profit_factor` | 총 이익 / 총 손실 (비용 차감) | `sum(양수 수익) / abs(sum(음수 수익))` |
| **Gross Profit Factor** | `gross_profit_factor` | 총 이익 / 총 손실 (비용 차감 전) | 동일 (Gross 기준) |
| **Avg Trade Duration** | `avg_trade_duration` | 평균 보유 일수 | `평균(각 종목별 연속 보유 기간)` |
| **Avg N Tickers** | `avg_n_tickers` | 평균 보유 종목 수 | `평균(매 리밸런싱 시 보유 종목 수)` |
| **N Rebalances** | `n_rebalances` | 리밸런싱 횟수 | 전체 리밸런싱 실행 횟수 |

**추가 메타데이터**:
- `top_k`: 선택 종목 수
- `holding_days`: 보유 기간 (일)
- `buffer_k`: 버퍼 종목 수
- `weighting`: 가중치 방식 (`equal` 또는 `softmax`)

#### 최종 백테스트 결과 (2026-01-06 실행)

**4가지 전략의 Holdout 구간 성과**:

| 전략 | Net Sharpe | Net CAGR | Net MDD | Calmar Ratio | Hit Ratio | Profit Factor |
|------|------------|----------|---------|--------------|-----------|---------------|
| **bt120_long** | **0.2163** | **3.65%** | **-8.25%** | **0.4427** | 33.33% | **1.5317** |
| **bt20_short** | **0.1951** | **2.01%** | -10.38% | 0.1933 | **56.52%** | 1.1678 |
| bt20_ens | 0.0921 | 0.32% | -9.56% | 0.0331 | 47.83% | 1.0760 |
| bt120_ens | 0.0210 | -0.60% | -10.84% | -0.0549 | 33.33% | 1.0404 |

**주요 발견사항**:
- **bt120_long** (장기 보유 + 장기 랭킹): 가장 우수한 성과 (Sharpe 0.2163, CAGR 3.65%, MDD -8.25%)
- **bt20_short** (단기 보유 + 단기 랭킹): 가장 높은 Hit Ratio (56.52%)
- 통합 랭킹(ens) 전략은 Holdout 구간에서 일반화 성능이 낮음

**참고**: 모든 지표는 거래비용(cost_bps=10.0)을 반영한 Net 지표이며, 시장 국면 기능은 외부 API 없이 ohlcv_daily 데이터로 자동 분류됩니다.

**참고**: 
- 모든 지표는 `phase` 컬럼으로 구간별(`dev`/`holdout`) 구분됩니다.
- Gross 지표는 거래비용 차감 전 성과를 나타냅니다.
- Net 지표는 거래비용 차감 후 실제 수익을 나타냅니다.

#### 데이터 파일 위치

백테스트 실행 시 다음 파일들이 생성됩니다:

```
data/interim/
├── bt_metrics_{strategy}.parquet          # 백테스트 성과 지표 (메인)
├── bt_positions_{strategy}.parquet        # 포지션 정보
├── bt_returns_{strategy}.parquet          # 일별 수익률
├── bt_equity_curve_{strategy}.parquet     # 자산 곡선
├── bt_regime_metrics_{strategy}.parquet   # 국면별 성과 (조건부)
├── selection_diagnostics_{strategy}.parquet  # 선택 진단 정보
└── bt_returns_diagnostics_{strategy}.parquet # 수익률 진단 정보
```

---

## 투트랙 활용 시나리오

### 시나리오 1: 랭킹만 제공 (Track A만 사용)

**목적**: 이용자에게 종목 랭킹 정보만 제공하고, 이용자가 직접 투자 결정

**실행 순서**:
1. 공통 데이터 준비: `python scripts/run_pipeline_l0_l7.py`
2. Track A 실행: `python -m src.pipeline.track_a_pipeline`
3. UI에서 랭킹 표시 (단기/장기/통합 랭킹)
4. 이용자가 랭킹을 참고하여 직접 투자 결정

**적용 사례**:
- 랭킹 기반 종목 추천 서비스
- 투자 정보 제공 플랫폼

### 시나리오 2: 투자 모델 예시 제공 (Track A + Track B)

**목적**: 랭킹과 함께 다양한 투자 모델의 성과 예시를 제공하여 이용자의 투자 결정 지원

**실행 순서**:
1. 공통 데이터 준비: `python scripts/run_pipeline_l0_l7.py`
2. Track A 실행: `python -m src.pipeline.track_a_pipeline`
3. Track B 실행: `python -m src.pipeline.track_b_pipeline bt20_short` (원하는 전략 선택)
4. UI에서 랭킹 + 투자 모델 성과 표시
5. 이용자가 랭킹과 성과를 함께 참고하여 투자 결정

**적용 사례**:
- 랭킹 기반 포트폴리오 추천 서비스
- 백테스트 결과를 포함한 투자 전략 제안

### 시나리오 3: 통합 제공 (Track A + Track B 모든 전략)

**목적**: 랭킹과 다양한 투자 모델 예시를 모두 제공하여 이용자가 자신의 투자 성향에 맞는 모델 선택

**실행 순서**:
1. 공통 데이터 준비: `python scripts/run_pipeline_l0_l7.py`
2. Track A 실행: `python -m src.pipeline.track_a_pipeline`
3. Track B 실행 (모든 전략):
   ```bash
   python -m src.pipeline.track_b_pipeline bt20_short
   python -m src.pipeline.track_b_pipeline bt20_ens
   python -m src.pipeline.track_b_pipeline bt120_long
   python -m src.pipeline.track_b_pipeline bt120_ens
   ```
4. UI에서 랭킹과 모든 투자 모델 성과 표시
5. 이용자가 자신의 투자 성향에 맞는 모델 선택

**적용 사례**:
- 종합 투자 플랫폼
- 다양한 투자 전략 비교 서비스

---

## 참고 문서

### 핵심 개념
- `docs/TWO_TRACK_ARCHITECTURE.md`: 투트랙 아키텍처 가이드 ⭐ **새로 작성됨**
- `TECH_REPORT_TRACK1_RANKING.md`: Track A 기술 보고서 ⭐ **최상단으로 이동**
- `TECH_REPORT_TRACK2_BACKTEST.md`: Track B 기술 보고서 ⭐ **최상단으로 이동**
- `EASY_TECH_REPORT_FOR_NON_FINANCE.md`: 비금융인을 위한 통합 기술 보고서 ⭐ **최상단으로 이동**
- `DUAL_HORIZON_MODEL_RANKING_UI_NOTES.md`: BT20/BT120 개념 상세 설명
- `docs/TRACK_DEFINITION.md`: Track A/B 정의 및 구조

### 최종 리포트
- `artifacts/reports/FINAL_CONFIG_DETERMINATION_REPORT.md`: 최종 설정 확정 리포트
- `artifacts/reports/FINAL_RANKING_STRATEGY_COMPARISON_REPORT.md`: 랭킹 전략 비교 리포트
- `artifacts/reports/FINAL_METRICS_COMPREHENSIVE_REPORT.md`: 최종 성과 리포트

### 리팩터링 문서
- `REFACTORING_DESIGN.md`: 리팩터링 설계 문서
- `PROJECT_CLEANUP_COMPLETE.md`: 프로젝트 정리 완료 보고서

### 데이터 수집 모듈 (리팩토링 완료) ⭐
- `src/data_collection/`: 데이터 수집 완전 분리 모듈
  - **1단계**: 데이터 수집 완전 분리 (기존 데이터 그대로 유지)
  - **2단계**: 함수/모듈화 (UI에서 import 가능한 형태)
  - **3단계**: 파이프라인 재조립 (재현성 + 실행 간편화)

---

## 재현성

다른 컴퓨터에서도 동일한 결과를 재현하려면:

1. `pip install -r requirements.txt`로 의존성 설치
2. `configs/config.yaml`의 경로 설정 확인
3. 공통 데이터 준비: `python scripts/run_pipeline_l0_l7.py`
4. Track A 실행: `python -m src.pipeline.track_a_pipeline`
5. Track B 실행:
   - `python -m src.pipeline.bt20_pipeline short`
   - `python -m src.pipeline.bt120_pipeline long`

---

## 라이선스

본 프로젝트는 포트폴리오 목적으로 작성되었습니다.
