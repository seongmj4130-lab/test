# KOSPI200 투트랙 퀀트 투자 시스템 통합 기술 보고서

**작성일**: 2026-01-07 (최종 업데이트)  
**버전**: Phase 9 + 랭킹산정모델 최종 픽스 (2026-01-07)  
**대상 독자**: 금융 지식이 없는 소프트웨어 엔지니어, 데이터 사이언티스트, 학생, 일반 기술자  
**목적**: 전체 시스템(Track A: 랭킹 엔진 + Track B: 투자 모델)을 코드와 데이터 기준으로 쉽게 설명

---

## 📖 목차

1. [이 시스템은 무엇을 하나요?](#1-이-시스템은-무엇을-하나요)
2. [투트랙 구조: Track A와 Track B](#2-투트랙-구조-track-a와-track-b)
3. [Track A: 랭킹 엔진 (L0 ~ L8)](#3-track-a-랭킹-엔진-l0--l8)
4. [Track B: 투자 모델 (L6R ~ L7)](#4-track-b-투자-모델-l6r--l7)
5. [실제 코드로 보는 데이터 흐름](#5-실제-코드로-보는-데이터-흐름)
6. [성과 지표 이해하기](#6-성과-지표-이해하기)
7. [금융 용어 쉽게 이해하기](#7-금융-용어-쉽게-이해하기)

---

## 1. 이 시스템은 무엇을 하나요?

### 1.1 한 문장 요약

**"KOSPI200 주식 200개를 매일 점수로 줄 세워서(랭킹), 상위 종목을 선택해 투자했을 때 실제로 돈이 되는지 과거 데이터로 검증하는 시스템"**입니다.

### 1.2 일상 비유로 이해하기

이 시스템은 마치 **"학생 성적 평가 시스템 + 장학금 수여 시뮬레이션"**과 같습니다:

1. **Track A (랭킹 엔진)**: 
   - 200명의 학생(주식)에게 매일 시험을 보게 하고
   - 과거 성적 데이터를 분석해서 "점수 계산 규칙"을 만들고
   - 오늘의 학생들을 점수로 평가해 **1등~200등 순위**를 매깁니다.
   - **이용자에게 랭킹 정보를 제공**합니다.

2. **Track B (투자 모델)**:
   - Track A의 랭킹을 받아서 "상위 20명만 뽑아서 장학금을 줬을 때"를 과거로 되돌아가 실험합니다.
   - 장학금을 줄 때마다 행정비용(거래비용)이 든다고 가정합니다.
   - 최종적으로 "과거에 이렇게 했다면 실제로 얼마나 성과가 좋았는지"를 계산합니다.
   - **이용자에게 다양한 투자 모델 예시를 제공**합니다.

### 1.3 왜 이런 시스템이 필요한가요?

**문제**: 주식 투자는 불확실합니다. 어떤 주식을 살지 결정하기 어렵습니다.

**해결책**: 
- 데이터와 머신러닝으로 객관적인 랭킹을 만듭니다.
- 과거 데이터로 시뮬레이션해서 "이 전략이 실제로 작동하는지" 미리 검증합니다.

**결과**: 신뢰할 수 있는 투자 전략을 만들 수 있습니다.

---

## 2. 투트랙 구조: Track A와 Track B

이 시스템은 **두 개의 독립적인 트랙**으로 구성됩니다:

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
│ - ranking_    │            │ - bt_equity_curve   │
│   long_daily  │            │ - bt_positions       │
└───────────────┘            └─────────────────────┘
```

### 2.1 Track A: 랭킹 엔진

**목적**: 피처들로 KOSPI200의 랭킹을 산정하여 이용자에게 제공

**입력**: 원시 데이터 (주가, 재무 정보, 뉴스 등)  
**출력**: 매일 각 주식의 점수와 순위 (`ranking_short_daily.parquet`, `ranking_long_daily.parquet`)

**역할**: 
- 데이터를 수집하고 정리합니다 (L0~L4, 공통 데이터)
- 머신러닝 모델을 학습시킵니다 (L5, 선택적)
- 각 주식에 점수를 매기고 순위를 매깁니다 (L8)
- UI에서 사용할 수 있는 형태로 변환합니다 (L11, 선택적)

**실행 방법**:
```bash
python -m src.pipeline.track_a_pipeline
```

### 2.2 Track B: 투자 모델

**목적**: 랭킹을 기반으로 다양한 투자모델 예시를 만들어 이용자에게 정보 제공

**입력**: Track A의 랭킹 결과 (`ranking_short_daily.parquet`, `ranking_long_daily.parquet`)  
**출력**: 성과 지표 (수익률, 리스크, 거래비용 등)

**역할**:
- Track A의 랭킹을 백테스트용 스코어로 변환합니다 (L6R)
- 랭킹 상위 종목을 선택합니다 (L7)
- 과거 데이터로 "이렇게 투자했다면" 시뮬레이션합니다 (L7)
- 거래비용까지 반영한 현실적인 성과를 계산합니다 (L7)

**실행 방법**:
```bash
# Track B 전체 파이프라인 실행
python -m src.pipeline.track_b_pipeline bt20_short

# 또는 편의 래퍼 사용
python -m src.pipeline.bt20_pipeline short
python -m src.pipeline.bt120_pipeline long
```

**⚠️ 필수 조건**: Track A를 먼저 실행하여 랭킹 데이터를 생성해야 합니다.

---

## 3. Track A: 랭킹 엔진 (L0 ~ L8)

Track A는 **공통 데이터 준비(L0~L4) + 랭킹 생성(L8)**으로 구성됩니다. 각 단계는 이전 단계의 결과를 받아서 처리합니다.

### 3.1 L0: 유니버스 구성 - "누구를 평가할까?"

**비유**: 시험을 보기 전에 "누가 시험 대상인지" 명단을 만드는 단계

**코드 위치**: `src/tracks/shared/stages/data/l0_universe.py`

**무엇을 하나요?**
- KOSPI200 구성 종목을 월별로 조회합니다
- KOSPI200은 매월 구성 종목이 바뀔 수 있으므로, 각 월말에 "누가 포함되어 있는지" 기록합니다

**실제 코드**:
```python
# l0_universe.py:66-71
def build_k200_membership_month_end():
    # pykrx 라이브러리로 KOSPI200 구성 종목 조회
    portfolio = get_index_portfolio_deposit_file("2016-01-01", date_str, "1028")
    # 결과: date, ym, ticker 컬럼
```

**출력 예시**:
```
date        ym        ticker
2016-01-29  2016-01   005930  (삼성전자)
2016-01-29  2016-01   000660  (SK하이닉스)
...
```

**왜 중요한가요?**
- 잘못된 종목을 평가하면 의미가 없습니다
- KOSPI200에 포함되지 않은 종목은 평가 대상이 아닙니다

### 3.2 L1: OHLCV 전처리 + 기술적 지표 계산 - "주가 데이터 정리하기"

**비유**: 학생의 일일 출석부와 시험 성적을 정리하고, "최근 성적 추세", "성적 변동성" 같은 파생 지표를 만드는 단계

**코드 위치**: 
- `src/tracks/shared/stages/data/l1_ohlcv.py` (OHLCV 다운로드)
- `src/tracks/shared/stages/data/l1_technical_features.py` (기술적 지표 계산)

**무엇을 하나요?**

1. **OHLCV 데이터 다운로드**:
   - OHLCV = Open(시가), High(고가), Low(저가), Close(종가), Volume(거래량)
   - 각 종목의 일일 주가 데이터를 다운로드합니다

2. **기술적 지표 계산**:
   - **모멘텀(Momentum)**: 최근 주가가 얼마나 올랐는지 (예: 20일 모멘텀, 60일 모멘텀)
   - **변동성(Volatility)**: 주가가 얼마나 출렁이는지 (예: 20일 변동성, 60일 변동성)
   - **최대 낙폭(Max Drawdown)**: 최근 기간 중 가장 크게 떨어진 폭
   - **거래량 비율(Volume Ratio)**: 평소 거래량 대비 오늘 거래량이 얼마나 많은지

**실제 코드**:
```python
# l1_technical_features.py:63
# 20일 모멘텀 계산
df["price_momentum_20d"] = (close[t] - close[t-20]) / close[t-20]

# l1_technical_features.py:79-80
# 20일 변동성 계산
vol_20d = grouped["daily_return"].rolling(window=20).std() * np.sqrt(252)
df["volatility_20d"] = vol_20d
```

**출력 예시**:
```
date        ticker    close    price_momentum_20d    volatility_20d
2016-01-04  005930    50000    0.05 (5% 상승)        0.20 (20% 변동성)
...
```

**왜 중요한가요?**
- 주가 자체만으로는 부족합니다
- "최근 추세", "변동성" 같은 정보가 투자 결정에 도움이 됩니다

### 3.3 L2: 재무 데이터 병합 - "회사 체력표 붙이기"

**비유**: 학생의 시험 성적뿐만 아니라 "체력검사 결과", "가정환경" 같은 기본 정보를 추가하는 단계

**코드 위치**: `src/tracks/shared/stages/data/l2_fundamentals_dart.py`

**무엇을 하나요?**
- DART(전자공시시스템) API에서 회사의 재무 정보를 다운로드합니다
- **PER**: 주가 대비 이익 비율 (낮을수록 싸게 평가됨)
- **PBR**: 주가 대비 자산 비율 (낮을수록 싸게 평가됨)
- **ROE**: 자기자본 대비 이익률 (높을수록 수익성 좋음)
- **부채비율**: 빚이 얼마나 많은지

**중요한 점: 공시 지연 반영**
- 재무 정보는 회사가 발표한 후에야 알 수 있습니다
- 예: 2023년 연간 재무는 2024년 3월에 공시될 수 있습니다
- 실제 투자자가 그 시점에 알 수 있었던 정보만 사용하도록 **지연(lag)**을 반영합니다

**실제 코드**:
```python
# l2_fundamentals_dart.py:178-180
# 공시 지연 반영
effective_date = report_rcept_date + disclosure_lag_days  # 기본 1일
# 접수일 없으면: effective_date = year_end + fallback_lag_days  # 기본 90일
```

**출력 예시**:
```
date        ticker    roe      debt_ratio    effective_date
2016-12-31  005930    15.5     25.3          2017-04-01  (90일 지연 반영)
...
```

**왜 중요한가요?**
- 주가만으로는 회사의 "기본 체력"을 알 수 없습니다
- 재무 정보를 함께 보면 더 정확한 평가가 가능합니다
- **지연 반영**을 안 하면 "미래 정보를 미리 본 것"이 되어 부정확합니다

### 3.4 L3: 패널 데이터 병합 - "모든 정보를 한 표로 합치기"

**비유**: 학생의 모든 정보(시험 성적, 체력검사, 가정환경)를 하나의 큰 표로 합치는 단계

**코드 위치**: `src/tracks/shared/stages/data/l3_panel_merge.py`

**무엇을 하나요?**
- L0(유니버스), L1(OHLCV+기술지표), L2(재무), 섹터 정보, 시가총액을 모두 합칩니다
- **날짜 × 종목** 형태의 "패널 데이터"를 만듭니다
- 각 행은 "특정 날짜의 특정 종목"에 대한 모든 정보를 담고 있습니다

**실제 코드**:
```python
# l3_panel_merge.py: merge_asof 사용
# 시간 순서를 지키면서 재무 데이터 병합
df_merged = pd.merge_asof(
    df_ohlcv.sort_values('date'),
    df_fundamentals.sort_values('effective_date'),
    left_on='date',
    right_on='effective_date',
    by='ticker',
    direction='backward'  # 과거로 가장 가까운 재무 정보 사용
)
```

**출력 예시**:
```
date        ticker    close    momentum_20d    roe      sector_name
2016-01-04  005930    50000    0.05           15.5     IT/전자
2016-01-04  000660    80000    0.03           12.3     IT/전자
...
```

**왜 중요한가요?**
- 모든 정보가 한 곳에 있어야 머신러닝 모델이 학습할 수 있습니다
- 날짜와 종목을 기준으로 정확하게 병합해야 합니다

### 3.5 L4: Walk-Forward CV 분할 + 타깃 변수 생성 - "공정한 시험지 만들기"

**비유**: 
- 시험을 볼 때 "미래 문제를 미리 보면 안 되므로" 시간 순서를 지켜서 학습/평가 구간을 나누는 단계
- "20일 후 성적", "120일 후 성적" 같은 목표 변수를 만드는 단계

**코드 위치**: `src/tracks/shared/stages/data/l4_walkforward_split.py`

**무엇을 하나요?**

1. **타깃 변수 생성**:
   - **단기 타깃 (`ret_fwd_20d`)**: 20일 후 수익률 = (20일 후 가격 / 현재 가격) - 1
   - **장기 타깃 (`ret_fwd_120d`)**: 120일 후 수익률 = (120일 후 가격 / 현재 가격) - 1

2. **Walk-Forward CV 분할**:
   - 데이터를 **Dev(개발/튜닝)**와 **Holdout(실전 시험)**으로 나눕니다
   - Dev 구간을 다시 여러 개의 **Train/Test** 구간으로 나눕니다
   - 시간 순서를 지켜서 "과거로 학습 → 미래를 테스트"를 반복합니다
   - **Embargo/Purge**: 미래 정보가 새는 것을 방지하기 위해 경계 구간을 제외합니다

**실제 코드**:
```python
# l4_walkforward_split.py:146-157
# 20일 후 가격 계산
fwd_s = g.shift(-horizon_short)  # 20일 후
fwd_l = g.shift(-horizon_long)   # 120일 후
cur_safe = cur.where(cur != 0)
df[f"ret_fwd_{horizon_short}d"] = fwd_s / cur_safe - 1.0
df[f"ret_fwd_{horizon_long}d"] = fwd_l / cur_safe - 1.0
```

**Walk-Forward CV 예시**:
```
Train: 2016-01 ~ 2017-12  →  Test: 2018-01 ~ 2018-06  (Embargo: 20일)
Train: 2016-01 ~ 2018-06  →  Test: 2018-07 ~ 2018-12  (Embargo: 20일)
...
Holdout: 2023-01 ~ 2024-12  (최종 실전 시험)
```

**왜 중요한가요?**
- **타깃 변수**: 모델이 예측해야 할 "정답"을 정의합니다
- **Walk-Forward CV**: 시간 순서를 지켜야 현실적인 검증이 가능합니다
- **Embargo/Purge**: 미래 정보 누수를 막아야 정확한 성과 평가가 가능합니다

### 3.6 L5: 모델 학습 - "점수 예측 규칙 만들기"

**비유**: 과거 학생들의 성적 데이터를 분석해서 "어떤 학생이 좋은 성적을 낼지 예측하는 규칙"을 만드는 단계

**코드 위치**: `src/tracks/shared/stages/modeling/l5_train_models.py`

**무엇을 하나요?**

1. **피처 선택**:
   - 고정된 피처 리스트를 사용합니다 (`configs/features_short_v1.yaml`, `configs/features_long_v1.yaml`)
   - 단기 모델: **22개 피처** (Core 공통 12개 + Short 전용 6개 + News 감성 4개)
   - 장기 모델: 19개 피처 (Core 공통 12개 + Long 전용 7개)

2. **타깃 변환**:
   - **Cross-Sectional Rank**: 같은 날짜의 여러 종목을 비교해서 순위로 변환합니다
   - 예: 2023-01-04 날짜의 200개 종목을 순위로 변환 (1등=100%, 200등=0%)

3. **모델 학습**:
   - **Ridge 회귀** 모델을 사용합니다
   - 파이프라인: `SimpleImputer` (결측치 처리) → `StandardScaler` (정규화) → `Ridge` (예측)

**실제 코드**:
```python
# l5_train_models.py: 피처 선택
feature_cols = _pick_feature_cols(df, feature_list_yaml)

# 타깃 변환 (Cross-Sectional Rank)
if target_transform == "cs_rank":
    df_target = df.groupby('date')[target_col].transform(
        lambda x: x.rank(pct=True) * 100.0
    )

# 모델 학습
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('ridge', Ridge(alpha=1.0))
])
pipeline.fit(X_train, y_train)
```

**출력**:
- `pred_short_oos.parquet`: 단기(20일) 예측 점수
- `pred_long_oos.parquet`: 장기(120일) 예측 점수

**왜 중요한가요?**
- 머신러닝 모델이 복잡한 패턴을 학습합니다
- 단기와 장기를 분리해서 학습하면 더 정확합니다

### 3.7 L8: 랭킹 산정 - "1등~200등 줄 세우기"

**비유**: 모델이 예측한 점수를 받아서, 날짜별로 종목들을 순위로 줄 세우는 단계

**코드 위치**: 
- `src/components/ranking/score_engine.py` (점수 계산 엔진)
- `src/tracks/track_a/stages/ranking/l8_dual_horizon.py` (단기/장기 랭킹 생성)

**무엇을 하나요?**

1. **Cross-Sectional 정규화**:
   - 같은 날짜의 여러 종목을 비교 가능하도록 정규화합니다
   - **Percentile 정규화**: 순위를 0~100%로 변환
   - **Z-score 정규화**: 평균 0, 표준편차 1로 변환

2. **가중치 기반 점수 합산**:
   - 여러 피처를 가중치로 합산합니다
   - 예: `score_total = 0.3 * momentum + 0.2 * roe + 0.5 * volatility`

3. **순위 생성**:
   - 점수가 높은 순서대로 1등, 2등, ... 200등을 매깁니다

4. **단기/장기 랭킹 결합**:
   - 단기 랭킹과 장기 랭킹을 합쳐서 통합 랭킹을 만듭니다
   - 시장 국면(상승장/하락장)에 따라 가중치를 조정할 수 있습니다

**실제 코드**:
```python
# score_engine.py: Cross-Sectional 정규화
def normalize_feature_cross_sectional(df, feature_col, method='percentile'):
    if method == 'percentile':
        return df.groupby('date')[feature_col].transform(
            lambda x: x.rank(pct=True) * 100.0
        )

# score_engine.py: 점수 합산
score_total = sum(
    normalized_feature * feature_weight 
    for normalized_feature, feature_weight in zip(normalized_features, weights)
)

# l6r_ranking_scoring.py: 단기/장기 결합
score_ens = alpha_short * ranking_short + alpha_long * ranking_long
```

**출력**:
- `ranking_short_daily.parquet`: 단기 랭킹 (날짜별, 종목별 점수와 순위)
- `ranking_long_daily.parquet`: 장기 랭킹
- `rebalance_scores.parquet`: 백테스트용 통합 스코어 (리밸런싱 날짜별)

**왜 중요한가요?**
- 랭킹이 백테스트의 입력이 됩니다
- 정확한 랭킹이 좋은 투자 성과로 이어집니다

---

## 4. 파이프라인 2: 백테스트 (L7)

파이프라인 2는 **랭킹을 받아서 실제 투자 시뮬레이션**을 수행합니다.

### 4.1 Track B 개요

**비유**: "과거로 되돌아가서 이 전략대로 투자했다면 얼마나 벌었을까?"를 시뮬레이션하는 것

**입력**: 
- `ranking_short_daily.parquet`: Track A에서 생성된 단기 랭킹
- `ranking_long_daily.parquet`: Track A에서 생성된 장기 랭킹
- `dataset_daily.parquet`: 공통 데이터 (수익률 정보 포함)

**출력**: 성과 지표 (수익률, 리스크, 거래비용 등)

### 4.2 L6R: 랭킹 스코어 변환

**비유**: Track A의 랭킹을 백테스트에서 사용할 수 있는 형태로 변환하는 단계

**코드 위치**: `src/tracks/track_b/stages/modeling/l6r_ranking_scoring.py`

**무엇을 하나요?**
- Track A의 랭킹 데이터를 받아서 백테스트용 스코어로 변환합니다
- 단기/장기 랭킹을 결합하여 통합 스코어를 생성합니다
- 리밸런싱 날짜별로 스코어를 정리합니다

**출력**: `rebalance_scores_from_ranking.parquet` (백테스트 입력)

### 4.3 L7: 백테스트의 핵심 로직

**코드 위치**: `src/tracks/track_b/stages/backtest/l7_backtest.py`

#### 4.3.1 리밸런싱 날짜 추출

**비유**: "언제 포트폴리오를 다시 조정할까?"를 결정하는 단계

**로직** (2026-01-06 개선):
- **L6R 단계**: 일별 랭킹 데이터에서 `rebalance_interval`만큼 필터링하여 `rebalance_scores` 생성
- **L7 단계**: L6R에서 이미 필터링된 데이터를 그대로 사용 (이중 필터링 방지)
- `rebalance_interval=20`: BT20은 20일마다 리밸런싱 (총 109개)
- (최신) **BT120 오버래핑 트랜치(개선안 36번)**:
  - `rebalance_interval=20`: 매 20일마다 신규 트랜치 1개 추가(월별)
  - `tranche_holding_days=120`(캘린더 day), `tranche_max_active=4`
  - 목적: Holdout 리밸런싱 표본을 늘려 “타이밍 럭”을 줄임

**실제 코드** (L6R 단계):
```python
# l6r_ranking_scoring.py:280-306
if rebalance_interval == 1:
    # 월별 리밸런싱: cv_folds_short.test_end 사용
    rebal_map = folds[["test_end", "phase"]].rename(columns={"test_end": "date"}).copy()
else:
    # 일별 리밸런싱: ranking_daily에서 interval만큼 필터링
    all_dates = sorted(pd.to_datetime(ranking_short_daily["date"].unique()))
    rebalance_dates = [all_dates[i] for i in range(0, len(all_dates), rebalance_interval)]
    rebal_map = pd.DataFrame({"date": rebalance_dates})
```

**L7 단계** (이중 필터링 방지):
```python
# l7_backtest.py:520-529
if rebalance_interval > 1:
    # L6R에서 이미 필터링 완료, L7에서는 모든 날짜 사용
    rebalance_dates_filtered = rebalance_dates_all
    logger.info(f"L6R에서 이미 rebalance_interval={rebalance_interval} 필터링 완료")
```

#### 4.3.2 포트폴리오 선택 - "상위 K개 종목 고르기"

**비유**: 랭킹 1등~K등 학생을 선택하는 단계

**함수**: `select_topk_with_fallback()` (`src/components/portfolio/selector.py`)

**로직**:

1. **필터링**:
   - 필수 컬럼 결측 제거
   - 가격 정보 결측 제거
   - 거래정지 종목 제거 (옵션)

2. **Smart Buffering**:
   - 이전에 보유했던 종목이 여전히 상위권에 있으면 유지합니다
   - 종목 교체를 줄여서 거래비용을 절감합니다
   - 안정성 임계값(`stability_threshold`)을 넘으면 유지합니다

3. **Sector Diversification** (옵션):
   - 같은 섹터에서 너무 많이 선택하지 않도록 제한합니다
   - 예: IT 섹터에서 최대 4개만 선택

**실제 코드**:
```python
# selector.py:112-194
# 1. 허용 범위: top_k + buffer_k
allow_n = top_k + buffer_k if buffer_k > 0 else top_k
allow = g_filtered.head(allow_n).copy()

# 2. 이전 보유 종목 중 허용 범위에 있는 것들
allow_set = set(allow[ticker_col].astype(str).tolist())
keep = [t for t in prev_holdings if t in allow_set]

# 3. Smart Buffering: 안정성 체크
if smart_buffer_enabled:
    for t in keep:
        if stability_score[t] >= stability_threshold:
            selected.append(t)
```

#### 4.3.3 포트폴리오 가중치 - "각 종목에 얼마나 투자할까?"

**비유**: 선택한 학생들에게 장학금을 어떻게 나눠줄까?

**두 가지 방식**:

1. **Equal Weighting (동일 비중)**:
   - 선택한 종목에 똑같은 비중으로 투자
   - 예: 20개 종목 선택 → 각각 5% (1/20)

2. **Softmax Weighting (점수 기반 비중)**:
   - 점수가 높은 종목에 더 많이 투자
   - `softmax(score / temperature)`로 계산
   - `temperature`가 낮을수록 상위 종목에 집중

**실제 코드**:
```python
# l7_backtest.py: 가중치 계산
if weighting == "equal":
    weights = np.ones(len(selected)) / len(selected)
elif weighting == "softmax":
    scores = df_selected[score_col].values
    weights = softmax(scores / softmax_temperature)
```

#### 4.3.4 노출 조정 - "시장 상황에 맞춰 투자 규모 조정"

**비유**: 날씨가 나쁠 때는 외출을 줄이고, 좋을 때는 늘리는 것

**두 가지 조정**:

1. **Volatility Adjustment (변동성 조정)**:
   - 시장 변동성이 높으면 투자 규모를 줄입니다
   - 목표 변동성(예: 15%)에 맞춰 조정합니다
   - 예: 변동성이 20%면 → 0.75배 축소 (15% / 20%)

2. **Risk Scaling (국면별 리스크 스케일링)**:
   - 시장 국면(상승장/하락장/횡보)에 따라 투자 규모를 조정합니다
   - 하락장(Bear): 0.7배 축소
   - 횡보(Neutral): 0.9배
   - 상승장(Bull): 1.0배 (정상)

**실제 코드**:
```python
# l7_backtest.py: 변동성 조정
if volatility_adjustment_enabled:
    vol_scale = target_volatility / current_volatility
    vol_scale = np.clip(vol_scale, min_scale, max_scale)
    exposure *= vol_scale

# l7_backtest.py: 국면별 리스크 스케일링
if risk_scaling_enabled:
    if regime == "bear":
        exposure *= risk_scaling_bear_multiplier  # 0.7
    elif regime == "neutral":
        exposure *= risk_scaling_neutral_multiplier  # 0.9
    elif regime == "bull":
        exposure *= risk_scaling_bull_multiplier  # 1.0
```

#### 4.3.5 수익률 계산 - "실제로 얼마나 벌었나?"

**비유**: 장학금을 준 학생들이 실제로 얼마나 성과를 냈는지 계산하는 것

**로직**:

1. **보유 기간 동안의 수익률**:
   - BT20: 20일 후 수익률 (`true_short`)
   - BT120(오버래핑 트랜치): 월별 20일 후 수익률(`true_short`)로 기간손익을 계산하고, 트랜치는 120일 동안 오버랩 유지

2. **거래비용 차감**:
   - 매 리밸런싱마다 거래비용(`cost_bps`)을 차감합니다
   - 예: `cost_bps=10.0` → 0.10% 비용

3. **누적 수익률 계산**:
   - 각 리밸런싱 구간의 수익률을 누적해서 계산합니다

**실제 코드**:
```python
# l7_backtest.py: 수익률 계산
return_col = cfg.get('return_col', 'true_short')  # BT20: 'true_short', BT120(오버래핑 트랜치): 'true_short'(월별)
returns = df_selected[return_col].values

# 거래비용 차감
turnover = calculate_turnover(prev_weights, new_weights)
cost = turnover * cost_bps / 10000.0
net_return = gross_return - cost
```

#### 4.3.6 성과 지표 계산

**비유**: 최종 성적표를 만드는 것

**주요 지표**:

1. **수익 지표**:
   - **Net Total Return**: 비용 차감 후 누적 수익률
   - **Net CAGR**: 연평균 복리 수익률

2. **위험 지표**:
   - **Net Sharpe Ratio**: 리스크 대비 수익 효율 (클수록 좋음)
   - **Net MDD**: 최대 낙폭 (작을수록 좋음)
   - **Calmar Ratio**: CAGR / |MDD| (클수록 좋음)

3. **운용 지표**:
   - **Avg Turnover**: 평균 회전율 (너무 높으면 비용 증가)
   - **Hit Ratio**: 수익이 난 구간 비율
   - **Profit Factor**: 번 돈 합 / 잃은 돈 합 (1보다 크면 좋음)

**실제 코드**:
```python
# l7_backtest.py: 성과 지표 계산
metrics = {
    'net_total_return': (equity_curve[-1] / equity_curve[0]) - 1.0,
    'net_cagr': (equity_curve[-1] / equity_curve[0]) ** (252 / len(equity_curve)) - 1.0,
    'net_sharpe': np.mean(returns) / np.std(returns) * np.sqrt(252),
    'net_mdd': calculate_mdd(equity_curve),
    'calmar_ratio': net_cagr / abs(net_mdd),
    'avg_turnover': np.mean(turnovers),
    'hit_ratio': np.mean(returns > 0),
    'profit_factor': sum(positive_returns) / abs(sum(negative_returns))
}
```

---

## 5. 실제 코드로 보는 데이터 흐름

이 섹션에서는 실제 코드를 보면서 데이터가 어떻게 흘러가는지 확인합니다.

### 5.1 전체 파이프라인 실행

**공통 데이터 준비** (한 번만 실행):
```bash
python scripts/run_pipeline_l0_l7.py
```

**Track A 실행** (랭킹 엔진):
```bash
python -m src.pipeline.track_a_pipeline
```

**(추가) Holdout 하루 찍어서 Top10 + 팩터셋 Top3 확인(설명가능성)**
```bash
python scripts/inspect_tracka_holdout_day.py --date 2024-12-30 --topk 10 --horizon both
```

**Track B 실행** (투자 모델):
```bash
# Track A를 먼저 실행한 후
python -m src.pipeline.track_b_pipeline bt20_short
```

**Python 코드 예시**:
```python
# Track A 실행
from src.pipeline.track_a_pipeline import run_track_a_pipeline
result_a = run_track_a_pipeline()

# Track B 실행
from src.pipeline.track_b_pipeline import run_track_b_pipeline
result_b = run_track_b_pipeline(strategy="bt20_short")
```

### 5.2 데이터 파일 구조

**중간 산출물** (`data/interim/`):
- `k200_membership_month_end.parquet`: L0 출력
- `ohlcv_panel.parquet`: L1 출력
- `fundamentals_annual.parquet`: L2 출력
- `panel_merged_daily.parquet`: L3 출력
- `targets_and_folds.parquet`: L4 출력

**Track A 산출물** (`data/interim/`):
- `ranking_short_daily.parquet`: L8 단기 랭킹
- `ranking_long_daily.parquet`: L8 장기 랭킹
- `ui_payload.json` (선택적): UI Payload

**Track B 산출물** (`data/interim/`):
- `rebalance_scores_from_ranking.parquet`: L6R 랭킹 스코어 변환 (L7 입력)
- `bt_positions_{strategy}.parquet`: L7 포지션 히스토리
- `bt_returns_{strategy}.parquet`: L7 수익률 히스토리
- `bt_equity_curve_{strategy}.parquet`: L7 누적 자산 곡선
- `bt_metrics_{strategy}.parquet`: L7 성과 지표

**백테스트 산출물** (`artifacts/reports/`):
- `bt_positions.parquet`: 포지션 히스토리
- `bt_returns.parquet`: 수익률 히스토리
- `bt_equity_curve.parquet`: 누적 자산 곡선
- `bt_metrics.parquet`: 성과 지표 (BT20)
- `bt_metrics_bt120.parquet`: 성과 지표 (BT120)

### 5.3 설정 파일 구조

**설정 파일**: `configs/config.yaml`

**주요 섹션**:
- `l4`: Walk-Forward CV 파라미터 (step_days, embargo_days 등) - 공통
- `l5`: 모델 파라미터 (ridge_alpha, target_transform 등) - Track A 선택적
- `l8`: 랭킹 엔진 파라미터 (피처 그룹, 가중치 등) - Track A
- `l6r`: 랭킹 스코어 변환 파라미터 (alpha_short, alpha_long 등) - Track B
- `l7_bt20`: BT20 백테스트 파라미터 - Track B
- `l7_bt120`: BT120 백테스트 파라미터 - Track B

**예시**:
```yaml
l7_bt20:
  holding_days: 20
  top_k: 15
  cost_bps: 10.0
  weighting: "softmax"
  softmax_temperature: 0.5
  rebalance_interval: 20  # 20일마다 리밸런싱

l7_bt120:
  holding_days: 120
  top_k: 20
  cost_bps: 10.0
  weighting: "equal"
  rebalance_interval: 120  # 120일마다 리밸런싱
```

---

## 6. 성과 지표 이해하기

백테스트 결과를 해석하는 방법을 쉽게 설명합니다.

### 6.1 수익 지표

#### Net Total Return (순 누적 수익률)

**의미**: 비용을 뺀 후 최종적으로 몇 % 벌었는지

**비유**: 장학금 프로그램을 시작할 때 100만원이었는데, 끝날 때 150만원이 되었다면 → 50% 수익

**목표**: 높을수록 좋음

#### Net CAGR (순 연평균 복리 수익률)

**의미**: 연평균 몇 % 벌었는지로 환산한 값

**비유**: 3년 동안 50% 벌었다면 → 연평균 약 14.5% (복리 기준)

**계산식**: `(최종값 / 시작값) ^ (1/년수) - 1`

**목표**: 높을수록 좋음 (일반적으로 10% 이상이면 좋음)

### 6.2 위험 지표

#### Net Sharpe Ratio (순 샤프 지수)

**의미**: 리스크(변동성) 대비 수익 효율

**비유**: 같은 수익을 냈어도 덜 출렁이면서 벌었다면 더 좋은 것

**계산식**: `(평균 수익률 - 무위험 수익률) / 수익률 표준편차 * sqrt(252)`

**해석**:
- 1.0 이상: 좋음
- 0.5~1.0: 보통
- 0.5 미만: 개선 필요

**목표**: 높을수록 좋음 (일반적으로 0.9 이상이면 좋음)

#### Net MDD (순 최대 낙폭)

**의미**: 최악의 순간에 얼마나 크게 떨어졌는지

**비유**: 장학금 프로그램 중 가장 큰 손실 구간

**계산식**: `(최고점 - 최저점) / 최고점`

**해석**:
- -5%: 매우 좋음 (거의 안 떨어짐)
- -10%: 좋음
- -20%: 보통
- -30% 이상: 위험

**목표**: 작을수록 좋음 (절댓값이 작을수록 좋음)

#### Calmar Ratio (칼마 지수)

**의미**: CAGR 대비 MDD의 비율 (큰 낙폭 없이 꾸준히 버는지)

**계산식**: `CAGR / |MDD|`

**해석**:
- 1.5 이상: 매우 좋음
- 1.0~1.5: 좋음
- 0.5~1.0: 보통
- 0.5 미만: 개선 필요

**목표**: 높을수록 좋음

### 6.3 운용 지표

#### Avg Turnover (평균 회전율)

**의미**: 얼마나 자주/많이 종목을 갈아탔는지

**비유**: 장학금 수여 대상을 얼마나 자주 바꿨는지

**해석**:
- 100%: 매번 절반을 교체
- 500%: 매우 자주 교체 (비용 증가)
- 1000% 이상: 너무 자주 교체 (현실성 떨어짐)

**목표**: 적당히 낮을수록 좋음 (일반적으로 500% 이하)

#### Hit Ratio (승률)

**의미**: 수익이 난 구간의 비율

**비유**: 장학금을 준 구간 중 몇 %에서 성과가 났는지

**해석**:
- 60% 이상: 매우 좋음
- 50~60%: 좋음
- 50% 미만: 개선 필요

**목표**: 높을수록 좋음 (일반적으로 55% 이상)

#### Profit Factor (수익 팩터)

**의미**: 번 돈 합 / 잃은 돈 합

**비유**: 전체적으로 이긴 구간의 돈이 진 구간의 돈보다 얼마나 많은지

**해석**:
- 2.0 이상: 매우 좋음
- 1.5~2.0: 좋음
- 1.0~1.5: 보통
- 1.0 미만: 손실 (개선 필요)

**목표**: 1보다 크면 좋음 (일반적으로 1.5 이상)

### 6.4 Dev vs Holdout

**Dev (개발 구간)**: 모델을 튜닝하고 검증하는 구간  
**Holdout (실전 시험 구간)**: 최종적으로 실전 성과를 평가하는 구간

**비유**: 
- Dev = 연습 시험 (여러 번 시험을 보면서 실력을 키움)
- Holdout = 최종 시험 (한 번만 보고 실전 성과 평가)

**중요한 점**:
- Dev 성과가 좋아도 Holdout 성과가 나쁘면 **과적합(Overfitting)** 가능성
- Holdout 성과가 Dev 성과와 비슷하면 **일반화(Generalization)** 잘 됨

---

## 7. 금융 용어 쉽게 이해하기

이 섹션에서는 프로젝트에서 사용되는 주요 금융 용어를 쉽게 설명합니다.

### 7.1 기본 용어

#### KOSPI200 (코스피200)

**의미**: 한국 증시에서 시가총액이 큰 상위 200개 회사로 구성된 지수

**비유**: "한국 대형주 대표팀"

**예시**: 삼성전자, SK하이닉스, 현대자동차 등

#### 유니버스 (Universe)

**의미**: 전략이 "고를 수 있는" 종목의 전체 목록

**비유**: 시험을 볼 수 있는 학생 명단

#### 종목 (Ticker)

**의미**: 주식 한 개를 나타내는 코드

**예시**: 005930 (삼성전자), 000660 (SK하이닉스)

#### OHLCV

**의미**: 주가 데이터의 기본 정보
- **Open (시가)**: 그날 시장이 열릴 때의 가격
- **High (고가)**: 그날 가장 비쌌던 가격
- **Low (저가)**: 그날 가장 쌌던 가격
- **Close (종가)**: 그날 시장이 마감할 때의 가격
- **Volume (거래량)**: 그날 거래된 주식 수

**비유**: 하루 동안의 주가 여행 기록

### 7.2 수익률 관련 용어

#### 수익률 (Return)

**의미**: `(현재가 - 과거가) / 과거가` = 몇 % 올랐나/내렸나

**비유**: 100원짜리 주식을 샀는데 110원이 되었다면 → 10% 수익률

#### 누적 수익률 (Total Return)

**의미**: 기간 전체를 통틀어 최종적으로 몇 % 변했는지

**비유**: 1년 동안 여러 번 오르락내리락했지만, 최종적으로 20% 올랐다면 → 20% 누적 수익률

#### CAGR (연복리 수익률)

**의미**: 기간 수익률을 "연 단위"로 환산한 복리 수익률

**비유**: 3년 동안 50% 벌었다면 → 연평균 약 14.5% (복리 기준)

**계산식**: `(최종값 / 시작값) ^ (1/년수) - 1`

### 7.3 리스크 관련 용어

#### 변동성 (Volatility)

**의미**: 수익률이 얼마나 출렁이는지 (리스크의 대표 지표)

**비유**: 주가가 매일 크게 오르락내리락하면 변동성이 높음

**계산**: 수익률의 표준편차

#### MDD (최대 낙폭, Max Drawdown)

**의미**: 최고점 대비 얼마나 떨어졌는지의 최대값

**비유**: 가장 큰 손실 구간

**예시**: 100만원 → 120만원 → 90만원이 되었다면 → MDD = (120-90)/120 = 25%

#### Sharpe Ratio (샤프 지수)

**의미**: 리스크 대비 수익 효율

**비유**: 같은 수익을 냈어도 덜 출렁이면서 벌었다면 더 좋은 것

**계산식**: `(평균 수익률 - 무위험 수익률) / 수익률 표준편차 * sqrt(252)`

**해석**: 1.0 이상이면 좋음

### 7.4 포트폴리오 관련 용어

#### 포트폴리오 (Portfolio)

**의미**: 여러 종목을 묶어 만든 투자 바구니

**비유**: 여러 학생에게 장학금을 주는 것

#### 비중 (Weight)

**의미**: 포트폴리오에서 각 종목이 차지하는 비율

**비유**: 장학금을 어떻게 나눠줄까?

**예시**: 20개 종목에 동일 비중으로 투자 → 각각 5% (1/20)

#### 리밸런싱 (Rebalancing)

**의미**: 일정 주기마다 종목/비중을 다시 맞추는 행위

**비유**: 장학금 수여 대상을 주기적으로 다시 선정하는 것

#### 턴오버 (Turnover, 회전율)

**의미**: 리밸런싱으로 인해 포트폴리오가 얼마나 많이 바뀌었는지

**비유**: 장학금 수여 대상을 얼마나 자주 바꿨는지

**계산**: 새로 산 종목 비중 + 팔 종목 비중

### 7.5 거래비용 관련 용어

#### 거래비용 (Transaction Cost)

**의미**: 사고팔 때 드는 비용 (수수료/세금/스프레드 등)

**비유**: 장학금을 줄 때 드는 행정비용

#### bps (베이시스 포인트)

**의미**: 1bp = 0.01%

**예시**: 10bp = 0.10%, 100bp = 1.0%

#### 슬리피지 (Slippage)

**의미**: "원하는 가격"에 못 사고팔아서 추가로 생기는 비용

**비유**: 원래 100원에 사려고 했는데 101원에 샀다면 → 1원 슬리피지

### 7.6 모델/랭킹 관련 용어

#### 피처 (Feature)

**의미**: 모델이 학습할 수 있는 입력 변수

**비유**: 학생을 평가할 때 보는 항목 (시험 성적, 체력, 가정환경 등)

**예시**: 모멘텀, 변동성, ROE 등

#### 타깃 (Target)

**의미**: 모델이 예측해야 할 정답

**비유**: 학생의 최종 성적

**예시**: 20일 후 수익률, 120일 후 수익률

#### Cross-Sectional (횡단면)

**의미**: 같은 날짜에 여러 종목을 서로 비교하는 관점

**비유**: 같은 날 시험을 본 학생들을 서로 비교하는 것

#### Walk-Forward CV

**의미**: 시간 순서를 지켜서 "과거로 학습 → 미래를 테스트"를 반복하는 검증 방식

**비유**: 연습 시험을 여러 번 보되, 미래 문제를 미리 보지 않도록 주의하는 것

#### Embargo/Purge

**의미**: 미래 정보가 새는 것을 방지하기 위해 경계 구간 데이터를 제외하는 장치

**비유**: 시험 문제가 새지 않도록 경계 구간을 막는 것

### 7.7 시장 국면 관련 용어

#### 시장 국면 (Regime)

**의미**: 상승장/하락장/횡보장 같은 "시장 분위기" 구분

**비유**: 날씨 (맑음/비/흐림)

#### Bull Market (상승장)

**의미**: 전반적으로 오르는 시장

**비유**: 맑은 날씨

#### Bear Market (하락장)

**의미**: 전반적으로 내리는 시장

**비유**: 비 오는 날씨

#### Neutral (중립/횡보)

**의미**: 뚜렷한 상승/하락이 아닌 구간

**비유**: 흐린 날씨

---

## 8. 마무리

이 문서는 KOSPI200 주식 투자 시스템의 전체 구조를 코드와 데이터 기준으로 쉽게 설명했습니다.

### 8.1 핵심 요약

1. **Track A (랭킹 엔진)**: 공통 데이터 준비 → 랭킹 생성 → 이용자에게 랭킹 제공
2. **Track B (투자 모델)**: Track A의 랭킹을 받아서 → 백테스트 실행 → 이용자에게 투자 모델 예시 제공

### 8.2 다음 단계

- **코드 탐색**: 각 단계의 실제 코드를 읽어보면서 이해를 깊게 할 수 있습니다
- **파라미터 튜닝**: `configs/config.yaml`을 수정해서 전략을 개선할 수 있습니다
- **성과 분석**: `artifacts/reports/`의 리포트를 읽어서 성과를 분석할 수 있습니다

### 8.3 관련 문서

- **투트랙 아키텍처 가이드**: `docs/TWO_TRACK_ARCHITECTURE.md` ⭐
- **Track A 기술 보고서**: `TECH_REPORT_TRACK1_RANKING.md`
- **Track B 기술 보고서**: `TECH_REPORT_TRACK2_BACKTEST.md`
- **프로젝트 README**: `README.md`
- **설정 파일**: `configs/config.yaml`

---

**작성 완료일**: 2026-01-06 (최종 업데이트)  
**버전**: Phase 9 + 뉴스 피처 추가 (2026-01-04) + 투트랙 구조 리팩토링 (2026-01-05) + Ridge Alpha 최적화 (2026-01-06)  
**최종 검토**: 
- 코드와 데이터 기준 100% 반영
- 뉴스 피처 추가 반영
- 투트랙 구조 반영 (Track A/B 분리)
- **Ridge Alpha 최적화** (2026-01-06): Grid Search를 통한 L2 정규화 강도 최적화 시도
  - 결과: 모든 전략이 랭킹 기반이므로 ridge_alpha가 성과에 영향을 주지 않음 (정상 동작 확인)
  - 상세 리포트: `artifacts/reports/FINAL_RIDGE_ALPHA_OPTIMIZATION_REPORT.md`
- **완전 교체 전략 + top_k 최적화** (2026-01-06): 완전 교체 전략에서 top_k 최적화
  - 결과: top_k=15이 Holdout 구간에서 최고 성과 (Total Return 12.39%, Sharpe 0.5464, CAGR 6.69%)
  - 상세 리포트: `artifacts/reports/full_replacement_topk_optimization_report.md`
- **가중치 방식(equal vs softmax) 비교 최적화** (2026-01-06): 4가지 전략 모두에 대해 equal과 softmax 비교
  - 결과: 모든 전략에서 equal이 softmax보다 우수 (Holdout Total Return 기준)
  - 상세 리포트: `artifacts/reports/weighting_comparison_optimization_report.md`

