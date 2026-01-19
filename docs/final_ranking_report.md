# KOSPI200 랭킹 엔진 (Track A) 기술 보고서

**작성일**: 2026-01-07 (최종 업데이트)  
**버전**: Phase 9 + 랭킹산정모델 최종 픽스 (2026-01-07)  
**대상**: 사내 퀀트/ML 개발자, 리서처  
**관점**: 코드 기반, 실제 데이터 플로우 기반 설명

---

## 📋 목차

1. [시스템 개요](#1-시스템-개요)
2. [데이터 파이프라인 (L0~L3)](#2-데이터-파이프라인-l0l3)
3. [Walk-Forward CV 분할 (L4)](#3-walk-forward-cv-분할-l4)
4. [모델 학습 (L5, Ridge 회귀)](#4-모델-학습-l5-ridge-회귀)
5. [랭킹 산정 (L8, Score Engine)](#5-랭킹-산정-l8-score-engine)
6. [최종 성과 지표와 검증 로직](#6-최종-성과-지표와-검증-로직)

**⚠️ 참고**: L5는 Track B에서 사용되지만, Track A에서는 L5의 피처 리스트만 사용합니다. Track A만 사용하는 경우 L5는 선택적입니다.

---

## 1. 시스템 개요

### 1.1 Track A: 랭킹 엔진의 역할

**Track A (랭킹 엔진)**는 KOSPI200 유니버스 종목에 대해 다음을 수행합니다:

**핵심 목적**: 피처들로 KOSPI200의 랭킹을 산정하여 이용자에게 제공

**주요 기능**:

1. **피처 생성**: OHLCV, 재무, 기술적 지표, ESG, 뉴스 감성 등 다차원 피처 생성
2. **ML 모델 학습** (선택적): 단기(20일) 및 장기(120일) 수익률 예측을 위한 Ridge 회귀 모델 학습
3. **랭킹 생성**: 단기 랭킹, 장기 랭킹, 통합 랭킹 생성
4. **UI Payload 생성**: UI에서 사용할 수 있는 형태로 랭킹 데이터 변환

**코드 위치**: `src/tracks/track_a/`

**실행 방법**:
```bash
python -m src.pipeline.track_a_pipeline
```

**Track B와의 관계**: Track A는 독립적으로 실행 가능하며, Track B(투자 모델)는 Track A의 랭킹 데이터를 입력으로 사용합니다.

### 1.2 KOSPI200 유니버스 및 타깃 정의

**유니버스**: KOSPI200 구성 종목 (월말 기준 스냅샷)

**타깃 변수**:
- **단기 타깃**: `ret_fwd_20d` (20일 후 수익률)
  - 계산식: `(close[t+20] / close[t]) - 1.0`
  - BT20 백테스트에 사용 (`return_col: "true_short"`)
- **장기 타깃**: `ret_fwd_120d` (120일 후 수익률)
  - 계산식: `(close[t+120] / close[t]) - 1.0`
  - BT120 백테스트에 사용 (`return_col: "true_long"`)

**코드 위치**: `src/tracks/shared/stages/data/l4_walkforward_split.py`

```python
# l4_walkforward_split.py:146-157
fwd_s = g.shift(-horizon_short)  # 20일 후 가격
fwd_l = g.shift(-horizon_long)   # 120일 후 가격
cur_safe = cur.where(cur != 0)
df[f"ret_fwd_{horizon_short}d"] = fwd_s / cur_safe - 1.0
df[f"ret_fwd_{horizon_long}d"] = fwd_l / cur_safe - 1.0
```

### 1.3 주요 산출물 구조

#### `ranking_short_daily.parquet`
- **컬럼**: `date`, `ticker`, `score_total`, `rank_total`, `in_universe` (선택적: `sector_name`)
- **용도**: 단기(20일) 랭킹 신호
- **생성 위치**: `src/components/ranking/score_engine.py::build_ranking_daily()`

#### `ranking_long_daily.parquet`
- **컬럼**: `date`, `ticker`, `score_total`, `rank_total`, `in_universe` (선택적: `sector_name`)
- **용도**: 장기(120일) 랭킹 신호
- **생성 위치**: `src/components/ranking/score_engine.py::build_ranking_daily()`

#### `ranking_short_daily.parquet` / `ranking_long_daily.parquet`
- **컬럼**: `date`, `ticker`, `score_total`, `rank_total`, `in_universe` (선택적: `sector_name`)
- **용도**: 날짜별 종목 랭킹 (이용자에게 제공)
- **생성 위치**: `src/tracks/track_a/stages/ranking/l8_dual_horizon.py`

**참고**: `rebalance_scores.parquet`는 Track B에서 생성되며, Track A의 랭킹 데이터를 백테스트용 스코어로 변환한 것입니다.

---

## 2. 데이터 파이프라인 (L0~L3)

### 2.1 L0: 유니버스 구성

**파일**: `src/tracks/shared/stages/data/l0_universe.py`

**함수**: `build_k200_membership_month_end()`

#### KOSPI200 구성 종목 로딩

**소스**: pykrx 라이브러리 (`get_index_portfolio_deposit_file()`)

**로직**:
1. 거래일 캘린더 생성 (anchor_ticker 기준, 기본값: "005930")
2. 월말 거래일 추출: `groupby(pd.Series(dates).dt.to_period("M")).max()`
3. 각 월말 날짜에 대해 KOSPI200 구성 종목 조회
4. QC: 월별 종목 수 180~220개 범위 검증 (strict 모드)

**산출물 스키마**:
```python
# l0_universe.py:66-71
{
    "date": pd.Timestamp,  # 월말 거래일
    "ym": str,             # "YYYY-MM"
    "ticker": str          # 6자리 종목코드 (zfill(6))
}
```

**예시**:
```
date        ym        ticker
2016-01-29  2016-01   005930
2016-01-29  2016-01   000660
...
```

### 2.2 L1: OHLCV 전처리 및 기술적 지표 계산

**파일**: 
- `src/tracks/shared/stages/data/l1_ohlcv.py`
- `src/tracks/shared/stages/data/l1_technical_features.py`

**함수**: 
- `download_ohlcv_panel()` (OHLCV 다운로드)
- `calculate_technical_features()` (기술적 지표 계산)

#### OHLCV 데이터 다운로드

**소스**: pykrx 라이브러리 (`get_market_ohlcv_by_date()`)

**로직**:
1. 종목별로 OHLCV 데이터 다운로드
2. 컬럼명 정규화 (한글 → 영문): `날짜→date`, `시가→open`, `고가→high`, `저가→low`, `종가→close`, `거래량→volume`, `거래대금→value`
3. `date` 컬럼을 `YYYY-MM-DD` 형식 문자열로 변환
4. `ticker` 컬럼을 6자리로 정규화 (`zfill(6)`)

**기본 컬럼**: `date`, `ticker`, `open`, `high`, `low`, `close`, `volume`, `value` (선택적)

#### 기술적 지표 계산

**함수**: `calculate_technical_features()` (`l1_technical_features.py:12-126`)

**계산되는 피처** (Phase 9 기준):

| 피처명 | 계산 방식 | 윈도우 | 코드 위치 |
|--------|----------|--------|-----------|
| `price_momentum_20d` | `(close[t] - close[t-20]) / close[t-20]` | 20일 | l1_technical_features.py:63 |
| `price_momentum_60d` | `(close[t] - close[t-60]) / close[t-60]` | 60일 | l1_technical_features.py:66 |
| `momentum_3m` | `(close[t] - close[t-90]) / close[t-90]` | 90일 | l1_technical_features.py:69 |
| `momentum_6m` | `(close[t] - close[t-180]) / close[t-180]` | 180일 | l1_technical_features.py:72 |
| `volatility_20d` | `std(daily_return, window=20) * sqrt(252)` | 20일 | l1_technical_features.py:79 |
| `volatility_60d` | `std(daily_return, window=60) * sqrt(252)` | 60일 | l1_technical_features.py:83 |
| `max_drawdown_60d` | `(close - rolling_max) / rolling_max` | 60일 | l1_technical_features.py:90-98 |
| `downside_volatility_60d` | `std(negative_returns, window=60) * sqrt(252)` | 60일 | l1_technical_features.py:102-108 |
| `volume_ratio` | `volume / rolling_mean(volume, window=20)` | 20일 | l1_technical_features.py:111-112 |
| `momentum_reversal` | `momentum_5d - momentum_20d` | 5일 vs 20일 | l1_technical_features.py:116-118 |

**구현 세부사항**:
```python
# l1_technical_features.py:59
df["daily_return"] = grouped[close_col].pct_change()

# l1_technical_features.py:79-80
vol_20d = grouped["daily_return"].rolling(window=20, min_periods=5).std() * np.sqrt(252)
df["volatility_20d"] = vol_20d.reset_index(level=0, drop=True).reindex(df.index)
```

**산출물**: OHLCV + 기술적 지표가 포함된 DataFrame

### 2.3 L2: 재무/펀더멘탈 병합

**파일**: `src/tracks/shared/stages/data/l2_fundamentals_dart.py`

**함수**: `download_annual_fundamentals()`

#### DART API 호출

**소스**: OpenDartReader 라이브러리

**로직**:
1. **corp_code 매핑**: `stock_code(6자리) → corp_code(8자리)` (`_load_corp_map()`)
2. **연간 재무 데이터 조회**: `dart.finstate(corp_code, year, reprt_code="11011", fs_div="CFS")`
   - CFS(연결) 우선, 실패 시 OFS(개별) fallback
3. **공시 지연 반영** (`[Stage 1]`):
   - `report_rcept_date` (접수일) 추출 시도
   - `effective_date = report_rcept_date + disclosure_lag_days` (기본값: 1일)
   - 접수일 없으면: `effective_date = year_end + fallback_lag_days` (기본값: 90일)

**계산 지표**:
- `net_income`: 당기순이익 (`_pick_amount()` 함수로 account_nm 매칭)
- `total_liabilities`: 부채총계
- `equity`: 자본총계
- `debt_ratio`: `(total_liabilities / equity) * 100.0`
- `roe`: `(net_income / equity) * 100.0`

**산출물 스키마**:
```python
# l2_fundamentals_dart.py:348-361
{
    "date": "YYYY-12-31",           # 연말 날짜
    "ticker": str,                   # 6자리 종목코드
    "corp_code": str,                # 8자리 법인코드
    "report_rcept_date": pd.Timestamp | None,  # 접수일 (가능한 경우)
    "effective_date": pd.Timestamp, # 유효일 (공시 지연 반영)
    "lag_source": str,               # "rcept_date" | "year_end_fallback"
    "net_income": float | None,
    "total_liabilities": float | None,
    "equity": float | None,
    "debt_ratio": float | None,
    "roe": float | None
}
```

**API 키 관리**: 환경변수 `DART_API_KEY` 또는 `DART_API_KEYS` (여러 키 지원, 순환 사용)

### 2.4 L3: 패널 데이터 통합

**파일**: `src/tracks/shared/stages/data/l3_panel_merge.py`

**함수**: `build_panel_merged_daily()`

#### 병합 로직

**1단계: OHLCV + 재무 데이터 asof merge**

```python
# l3_panel_merge.py:132-140
merged = pd.merge_asof(
    o_sorted,                    # left: OHLCV (date 기준 정렬)
    f_join,                      # right: 재무 (effective_date 기준 정렬)
    left_on="date",
    right_on="effective_date",
    by="ticker",
    direction="backward",        # 과거 재무 데이터 사용
    allow_exact_matches=True,
)
```

**핵심**: 재무 데이터는 `effective_date` 기준으로 forward-fill (공시 지연 반영)

**2단계: 유니버스 멤버십 매핑**

```python
# l3_panel_merge.py:177-180
merged["ym"] = merged["date"].dt.to_period("M").astype(str)
merged = merged.merge(un_key, on=["ym", "ticker"], how="left", indicator=True)
merged["in_universe"] = merged["_merge"].eq("both")
```

**3단계: 업종 정보 병합** (`[Stage 4]`)

```python
# l3_panel_merge.py:214-222
merged = pd.merge_asof(
    merged_sorted,
    sector_sorted[["date", "ticker", "sector_name"]],
    left_on="date",
    right_on="date",
    by="ticker",
    direction="backward",
    allow_exact_matches=True,
)
```

**4단계: pykrx 재무데이터 병합** (`[L1B]`)

- PER, PBR, EPS, BPS, DIV, market_cap
- `date`, `ticker` 기준 merge
- 0값 처리: PER/EPS/DIV/PBR/BPS의 0 → NaN (손실/무배당 = 결측)

**5단계: 업종 내 상대화 피처 생성** (`[Stage6]`)

```python
# l3_panel_merge.py:360-381
# debt_ratio_sector_z, roe_sector_z 계산
merged[z_col] = merged.groupby(["date", "sector_name"], group_keys=False)[base_col].transform(calc_sector_z)
```

**최종 산출물**: `dataset_daily.parquet`

**필수 컬럼**:
- 식별자: `date`, `ticker`, `in_universe`
- 타깃: `ret_fwd_20d`, `ret_fwd_120d` (L4에서 추가)
- OHLCV: `open`, `high`, `low`, `close`, `volume`, `value`
- 기술적 지표: `price_momentum_20d`, `volatility_20d`, `momentum_3m`, 등
- 재무: `net_income`, `equity`, `debt_ratio`, `roe`, `PER`, `PBR`, 등
- 업종: `sector_name` (선택적)
- 기타: `market_cap`, `turnover_ratio`, 등

---

## 3. Walk-Forward CV 분할 (L4)

**파일**: `src/tracks/shared/stages/data/l4_walkforward_split.py`

**함수**: `build_targets_and_folds()`

### 3.1 타깃 변수 계산

**코드 위치**: `l4_walkforward_split.py:146-157`

```python
g = df.groupby("ticker", sort=False)[px]  # px = "close" 또는 "adj_close"
fwd_s = g.shift(-horizon_short)  # 20일 후 가격
fwd_l = g.shift(-horizon_long)   # 120일 후 가격
cur_safe = cur.where(cur != 0)
df[f"ret_fwd_{horizon_short}d"] = fwd_s / cur_safe - 1.0
df[f"ret_fwd_{horizon_long}d"] = fwd_l / cur_safe - 1.0
```

**Market-Neutral Target** (`[Phase 5]`): 초과 수익률 계산 (옵션)

```python
# l4_walkforward_split.py:161-175
if "in_universe" in df.columns:
    universe_mask = df["in_universe"] == True
    market_ret_short = df.loc[universe_mask].groupby("date")[f"ret_fwd_{horizon_short}d"].mean()
    market_ret_long = df.loc[universe_mask].groupby("date")[f"ret_fwd_{horizon_long}d"].mean()
df[f"ret_fwd_{horizon_short}d_excess"] = df[f"ret_fwd_{horizon_short}d"] - df["date"].map(market_ret_short)
```

**설정**: `config.yaml::l4.market_neutral` (기본값: `false`, 절대 수익률 사용)

### 3.2 Dev/Holdout 분리

**기준**: 연도 기준 (`holdout_years`)

```python
# l4_walkforward_split.py:196-198
overall_end = dates[-1]
holdout_threshold = overall_end - pd.DateOffset(years=holdout_years)
holdout_start = dates[dates.searchsorted(holdout_threshold, side="left")]
```

**설정값**: `config.yaml::l4.holdout_years = 2` (최근 2년 = Holdout)

### 3.3 Walk-Forward CV Fold 생성

**함수**: `_build_folds()` (`l4_walkforward_split.py:200-242`)

**파라미터** (설정값):
- `step_days`: 20 (리밸런싱 간격)
- `test_window_days`: 20 (테스트 윈도우 크기)
- `embargo_days`: 20 (Embargo 기간)
- `rolling_train_years_short`: 3 (단기 모델 학습 기간)
- `rolling_train_years_long`: 5 (장기 모델 학습 기간)

**Fold 생성 로직**:

```python
# l4_walkforward_split.py:212-240
pos = start_pos
while pos <= max_test_start:
    test_start_pos = pos
    test_end_pos = pos + (test_window_days - 1)
    
    train_end_pos = test_start_pos - embargo_days - horizon_days - 1
    train_end = dates[train_end_pos]
    train_start_threshold = train_end - pd.DateOffset(years=train_years)
    train_start_pos = int(dates.searchsorted(train_start_threshold, side="left"))
    train_start = dates[train_start_pos]
    
    folds.append({
        "fold_id": f"{segment}_{fold_i:04d}",
        "segment": segment,  # "dev" | "holdout"
        "train_start": train_start,
        "train_end": train_end,
        "test_start": dates[test_start_pos],
        "test_end": dates[test_end_pos],
        ...
    })
    pos += step_days
```

**Purge/Embargo 로직**:
- **Embargo**: `train_end`와 `test_start` 사이 최소 `embargo_days` 간격
- **Horizon**: `train_end` 이후 `horizon_days` 이후부터 `test_start` 가능
- **Purge**: `train_end_pos = test_start_pos - embargo_days - horizon_days - 1`

**산출물**: `cv_folds_short.parquet`, `cv_folds_long.parquet`

**스키마**:
```python
{
    "fold_id": str,           # "dev_0001", "holdout_0001", 등
    "segment": str,           # "dev" | "holdout"
    "train_start": pd.Timestamp,
    "train_end": pd.Timestamp,
    "test_start": pd.Timestamp,
    "test_end": pd.Timestamp,
    "train_years": int,       # 3 (short) | 5 (long)
    "horizon_days": int,      # 20 (short) | 120 (long)
    "embargo_days": int,      # 20
    "step_days": int,         # 20
    "test_window_days": int   # 20
}
```

---

## 4. 모델 학습 (L5, Ridge 회귀)

**파일**: `src/stages/modeling/l5_train_models.py`

### 4.1 타깃 정의 및 변환

#### Cross-Sectional Rank 변환

**함수**: `_cs_rank_by_date()` (`l5_train_models.py:228-237`)

```python
def _cs_rank_by_date(d: pd.DataFrame, col: str, *, center: bool = True) -> np.ndarray:
    r = d.groupby("date")[col].rank(pct=True)  # 날짜별 percentile rank (0~1)
    if center:
        r = r - 0.5  # [-0.5, 0.5] 범위로 0 중심화
    return r.to_numpy(dtype=np.float32, copy=False)
```

**설정**: `config.yaml::l5.target_transform = "cs_rank"`, `cs_rank_center = true`

**용도**: 절대 수익률 대신 상대 순위로 변환하여 모델 학습 (cross-sectional 비교 강화)

### 4.2 피처 선택

**함수**: `_pick_feature_cols()` (`l5_train_models.py:94-217`)

#### Phase 9 기준 피처 리스트 (고정 모드)

**단기 모델** (`configs/features_short_v1.yaml`): **22개 피처**

| 카테고리 | 피처명 | 개수 |
|---------|--------|------|
| **Core 공통** | `volatility_60d`, `volatility_20d`, `volatility`, `momentum_rank`, `downside_volatility_60d`, `price_momentum_60d`, `price_momentum`, `momentum_6m`, `max_drawdown_60d`, `turnover`, `net_income`, `roe` | 12 |
| **Short 전용** | `price_momentum_20d`, `momentum_3m`, `momentum_reversal`, `ret_daily`, `volume_ratio`, `equity` | 6 |
| **News 감성** | `news_sentiment`, `news_sentiment_ewm5`, `news_sentiment_surprise`, `news_volume` | 4 |

**장기 모델** (`configs/features_long_v1.yaml`): **19개 피처**

| 카테고리 | 피처명 | 개수 |
|---------|--------|------|
| **Core 공통** | `volatility_60d`, `volatility_20d`, `volatility`, `momentum_rank`, `downside_volatility_60d`, `price_momentum_60d`, `price_momentum`, `momentum_6m`, `max_drawdown_60d`, `turnover`, `net_income`, `roe` | 12 |
| **Long 전용** | `total_liabilities`, `debt_ratio`, `esg_score`, `environmental_score`, `social_score`, `governance_score`, `news_sentiment_ewm20` | 7 |

**설정**: `config.yaml::l5.feature_list_short`, `feature_list_long`

**로직**:
```python
# l5_train_models.py:108-140
if horizon == 20 and feature_list_short:
    feature_list_path = feature_list_short
elif horizon == 120 and feature_list_long:
    feature_list_path = feature_list_long

with open(feature_path, 'r', encoding='utf-8') as f:
    feature_config = yaml.safe_load(f) or {}
    fixed_features = feature_config.get("features", [])
    available = [f for f in fixed_features if f in df.columns]
    return available
```

### 4.3 전처리 & 모델

**함수**: `_build_model()` (`l5_train_models.py:249-326`)

#### Pipeline 구성

**Ridge 회귀** (기본 모델):

```python
# l5_train_models.py:265-269
pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),  # 결측치: 중앙값 대체
    ("scaler", StandardScaler(with_mean=True)),     # 표준화 (평균 0, 표준편차 1)
    ("model", Ridge(alpha=ridge_alpha)),            # Ridge 회귀
])
```

**하이퍼파라미터** (2026-01-07 최종 픽스):
- `ridge_alpha`: 8.0 (`config.yaml::l5.ridge_alpha`) - 과적합 방지 강화
- `target_transform`: "cs_rank" (Cross-sectional rank 변환)
- `cs_rank_center`: true (rank - 0.5)
- `min_feature_ic`: -0.1 (모든 피처 사용, 음수 IC 포함)

**다른 모델 옵션** (설정 가능):
- Random Forest: `model_type: "random_forest"`
- XGBoost: `model_type: "xgboost"`

### 4.4 학습 루프 구조

**함수**: `run_L5_train_models()` (주요 로직)

**CV Fold별 학습/예측**:

```python
# 각 fold에 대해:
for fold_spec in fold_specs:
    # 1. Train/Test 분할
    train_mask = (df["date"] >= fold_spec.train_start) & (df["date"] <= fold_spec.train_end)
    test_mask = (df["date"] >= fold_spec.test_start) & (df["date"] <= fold_spec.test_end)
    
    train_data = df[train_mask].copy()
    test_data = df[test_mask].copy()
    
    # 2. 타깃 변환 (cs_rank)
    y_train = _cs_rank_by_date(train_data, target_col, center=True)
    y_test = _cs_rank_by_date(test_data, target_col, center=True)
    
    # 3. 피처 선택
    feature_cols = _pick_feature_cols(train_data, target_col=target_col, cfg=cfg, horizon=horizon)
    X_train = train_data[feature_cols]
    X_test = test_data[feature_cols]
    
    # 4. 모델 학습
    pipe.fit(X_train, y_train)
    
    # 5. OOS 예측
    y_pred = pipe.predict(X_test)
    
    # 6. 결과 저장
    pred_df.append({
        "date": test_data["date"],
        "ticker": test_data["ticker"],
        f"pred_{horizon_name}": y_pred,
        "phase": fold_spec.phase,
        ...
    })
```

### 4.5 예측 산출물 구조

**파일**: `pred_short_oos.parquet`, `pred_long_oos.parquet`

**스키마**:
```python
{
    "date": pd.Timestamp,
    "ticker": str,
    "pred_short": float,      # 단기 예측값 (cs_rank 변환된 타깃 기준)
    "pred_long": float,       # 장기 예측값
    "phase": str,             # "dev" | "holdout"
    "fold_id": str,           # "dev_0001", 등
    "horizon": int,           # 20 | 120
    ...
}
```

**모델 메트릭**: `model_metrics.parquet`

**지표**:
- `ic_rank`: Rank IC (예측 순위와 실제 순위 상관계수)
- `rmse`: Root Mean Squared Error
- `mae`: Mean Absolute Error
- `hit_ratio`: 부호 일치율
- `r2_oos`: Out-of-sample R²

**계산 코드**: `l5_train_models.py:219-247`

```python
def _rank_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    s1 = pd.Series(y_true).rank(pct=True)
    s2 = pd.Series(y_pred).rank(pct=True)
    v = float(s1.corr(s2))
    return 0.0 if np.isnan(v) else v
```

---

## 5. 랭킹 산정 (L8, Score Engine)

**파일**: 
- `src/tracks/track_a/stages/ranking/l8_dual_horizon.py` (단기/장기 랭킹 생성)
- `src/components/ranking/score_engine.py` (스코어 계산 엔진)

### 5.1 피처 정규화

**함수**: `normalize_feature_cross_sectional()` (`score_engine.py:47-124`)

#### 날짜별 Cross-Sectional 정규화

**방법 선택**: `percentile`, `zscore`, 또는 `robust_zscore` (2026-01-07 최종 픽스: `zscore`)

**정규화 방법 비교 결과** (2026-01-07):
- **zscore**: 50.28% Hit Ratio (최고 성과) ✅ **최종 선택**
- **robust_zscore**: 49.05% Hit Ratio
- **percentile**: 45.91% Hit Ratio

**Z-score 정규화** (최종 픽스):
```python
# score_engine.py:92-97
mean_val = np.nanmean(values)
std_val = np.nanstd(values)
if std_val > 1e-8:
    normalized = (values - mean_val) / std_val
```

**Robust Z-score 정규화** (비교 테스트용):
```python
# score_engine.py:102-108
median_val = np.nanmedian(values)
mad_val = np.nanmedian(np.abs(values - median_val))
if mad_val > 1e-8:
    normalized = (values - median_val) / (mad_val * 1.4826)
```

**Percentile 정규화** (비교 테스트용):
```python
# score_engine.py:86-89
ranks = pd.Series(values).rank(pct=True, method="first")  # 0~1 범위
normalized = ranks.values
```

**섹터별 정규화** (`[Stage8]`):
```python
# score_engine.py:81-101
if use_sector_relative:
    # 같은 date, 같은 sector 내에서 정규화
    for (date, sector), group in df.groupby([date_col, sector_col], sort=False):
        # 정규화 수행
```

**설정**: `config.yaml::l8_short.normalization_method = "zscore"`, `l8_long.normalization_method = "zscore"` (2026-01-07 최종 픽스), `use_sector_relative = true`

### 5.2 피처 그룹 & 가중치

**설정 파일**:
- `configs/feature_groups.yaml` (공통, 선택)
- `configs/feature_groups_short.yaml` (단기)
- `configs/feature_groups_long.yaml` (장기)

**그룹 구조** (Phase 9 기준):

| 그룹명 | 피처 목록 | target_weight |
|--------|----------|--------------|
| **value** | `debt_ratio`, `debt_ratio_sector_z` | 0.25 |
| **profitability** | `roe`, `roe_sector_z` | 0.25 |
| **technical** | `volume_ratio`, `price_momentum`, `price_momentum_20d`, `price_momentum_60d`, `momentum_3m`, `momentum_6m`, `volatility`, `volatility_20d`, `volatility_60d`, `turnover`, `momentum_reversal`, `max_drawdown_60d`, `downside_volatility_60d` | 0.50 |
| **other** | `market_cap`, `turnover` | 0.25 |
| **news** | `news_sentiment`, `news_conviction`, `news_volume`, `news_sentiment_ewm5`, `news_sentiment_ewm20`, `news_sentiment_surprise` | 0.10 |

**가중치 계산 로직** (`score_engine.py:179-235`):

```python
# 1. 그룹별 target_weight 합계로 정규화
total_target_weight = sum(groups_with_target.values())
for group_name in group_names:
    group_weights[group_name] = groups_with_target[group_name] / total_target_weight

# 2. 그룹 내 피처별 균등 가중치
for feat in feature_cols:
    for group_name, group_features in feature_groups.items():
        if feat in group_features:
            n_features_in_group = len([f for f in feature_cols if f in group_features])
            feature_weights[feat] = group_weights[group_name] / n_features_in_group
```

**IC 최적화 가중치** (`[IC 최적화]`): `feature_weights_config` 파일에서 최적 가중치 로드 (우선 사용)

**국면별 가중치** (`[국면별 전략]`): `regime_aware_weights_config` 파일에서 국면별 가중치 로드

#### (추가) Holdout 하루 설명가능성: Top10 + 팩터셋(그룹) Top3 기여도

Holdout 기간 중 특정 날짜를 지정하면, 그 날의 **Top10 랭킹**과 함께 각 종목의 `score_total`이
어떤 **팩터셋(그룹)**에서 주로 기여했는지 **Top3**를 출력합니다.

- **코드 위치**:
  - 서비스 함수: `src/tracks/track_a/ranking_service.py::inspect_holdout_day_rankings()`  # [개선안 36번]
  - 계산 로직: `src/tracks/track_a/stages/ranking/holdout_day_inspector.py`  # [개선안 36번]

```python
from src.tracks.track_a.ranking_service import inspect_holdout_day_rankings

out = inspect_holdout_day_rankings(as_of="2024-12-30", topk=10, horizon="both")
df_short = out["short"]  # 단기 Top10 + top_groups
df_long = out["long"]    # 장기 Top10 + top_groups
```

CLI:

```bash
python scripts/inspect_tracka_holdout_day.py --date 2024-12-30 --topk 10 --horizon both
```

### 5.3 스코어 계산

**함수**: `build_score_total()` (`score_engine.py:126-322`)

**계산식**:
```python
# score_engine.py:311-314
score_total = pd.Series(0.0, index=out.index)
for feat, normalized_values in normalized_features.items():
    weight = feature_weights.get(feat, 0.0)
    score_total += weight * normalized_values.fillna(0.0)
```

**수식**: `score_total = Σ (normalized_feature[i] * feature_weight[i])`

**국면별 가중치 적용** (`[국면별 전략]`):
```python
# score_engine.py:252-302
if use_regime_weights:
    for date, group in out.groupby(date_col, sort=False):
        regime = market_regime_df[market_regime_df[date_col] == date].iloc[0]["regime"]
        date_weights = regime_weights_config[regime]  # 국면별 가중치 선택
        # 해당 날짜의 score_total 계산
```

### 5.4 단기/장기/통합 랭킹

**함수**: `build_rank_total()` (`score_engine.py:324-371`)

**랭킹 생성**:
```python
# score_engine.py:356-367
for date, group in out.groupby(date_col, sort=False):
    universe_group = group.loc[group[universe_col] == True]
    ranks = universe_group[score_col].rank(ascending=False, method="first")
    rank_total.loc[universe_group.index] = ranks.values
```

**랭킹 의미**: 높은 `score_total` = 낮은 `rank_total` (1위 = rank_total=1)

#### 단기 랭킹

**입력**: `dataset_daily` (단기 모델 예측값 포함 또는 피처만)

**생성**: `build_ranking_daily()` 호출 (`score_engine.py:373-434`)

**출력**: `ranking_short_daily.parquet` (`date`, `ticker`, `score_total`, `rank_total`, `in_universe`)

#### 장기 랭킹

**입력**: `dataset_daily` (장기 모델 예측값 포함 또는 피처만)

**생성**: `build_ranking_daily()` 호출

**출력**: `ranking_long_daily.parquet` (`date`, `ticker`, `score_total`, `rank_total`, `in_universe`)

#### 통합 랭킹 (Dual Horizon)

**함수**: `run_L8_short_rank_engine()`, `run_L8_long_rank_engine()` (`l8_dual_horizon.py`)

**생성 방식**:
- 단기 랭킹과 장기 랭킹을 각각 독립적으로 생성
- 통합 랭킹은 Track B에서 필요 시 생성 (L6R 단계)

**코드 위치**: `src/tracks/track_a/stages/ranking/l8_dual_horizon.py`

**참고**: 통합 랭킹(`score_ens`)은 Track B의 `l6r_ranking_scoring.py`에서 생성되며, Track A는 단기/장기 랭킹을 각각 제공합니다.

### 5.5 최종 산출물 구조

#### `ranking_short_daily.parquet`

**컬럼**:
- `date`: 날짜
- `ticker`: 종목코드
- `score_total`: 합산 스코어
- `rank_total`: 랭킹 (1~N, 낮을수록 상위)
- `in_universe`: 유니버스 멤버 여부
- `sector_name`: 업종명 (선택적)

#### `ranking_long_daily.parquet`

**컬럼**: `ranking_short_daily`와 동일

#### Track A 산출물 요약

**Track A는 다음 산출물을 생성합니다**:
1. `ranking_short_daily.parquet`: 단기 랭킹 (날짜별 종목 랭킹)
2. `ranking_long_daily.parquet`: 장기 랭킹 (날짜별 종목 랭킹)
3. `ui_payload.json` (선택적): UI에서 사용할 수 있는 형태의 랭킹 데이터

**참고**: `rebalance_scores.parquet`는 Track B에서 생성되며, Track A의 랭킹 데이터를 백테스트용 스코어로 변환한 것입니다.

---

## 6. 최종 성과 지표와 검증 로직

### 6.1 모델 품질 지표

**파일**: `model_metrics.parquet` (L5 산출물)

**지표** (fold별):

| 지표 | 계산 방식 | 코드 위치 |
|------|----------|-----------|
| **IC (Information Coefficient)** | `corr(y_true, y_pred)` | `l5_train_models.py:243` |
| **Rank IC** | `corr(rank(y_true), rank(y_pred))` | `l5_train_models.py:219-223` |
| **ICIR** | `mean(IC) / std(IC)` | (계산식) |
| **RMSE** | `sqrt(mean((y_pred - y_true)²))` | `l5_train_models.py:241` |
| **MAE** | `mean(abs(y_pred - y_true))` | `l5_train_models.py:242` |
| **Hit Ratio** | `mean(sign(y_true) == sign(y_pred))` | `l5_train_models.py:244` |
| **R² OOS** | `r2_score(y_true, y_pred)` | `l5_train_models.py:246` |

**참고(중요)**:
- 본 문서는 Track A(랭킹 엔진) 기술 보고서이며, 전략 성과/AlphaQuality는 Track B(L7) 실행 설정에 따라 달라집니다.
- 2026-01-07 기준 Track B는 **BT120 오버래핑 트랜치(월별 4트랜치)**가 도입되어, 과거 `rebalance_interval=120` 기반의 BT120 성과/ICIR/Long-Short Alpha 숫자와 직접 비교가 어렵습니다.

### 6.2 Alpha Quality 계산

**위치**: 백테스트(L7) 단계에서 계산 (Track B 산출물)

**계산 방식** (참고):
- **IC**: 날짜별 `corr(pred, true_return)` 평균
- **Rank IC**: 날짜별 `corr(rank(pred), rank(true_return))` 평균
- **ICIR**: `mean(IC) / std(IC)`
- **Long/Short Alpha**: 상위 포지션 수익률 - 하위 포지션 수익률

**코드 위치**: `src/tracks/track_b/stages/backtest/l7_backtest.py` (Track B 기술 보고서 참조)

### 6.3 랭킹 트랙 품질 지표 요약

**최신 성과/AlphaQuality는 Track B 리포트를 기준으로 확인**:
- `artifacts/reports/track_b_4strategy_final_summary.md` (4전략 Dev/Holdout + Alpha Quality + Operational + Regime Robustness)
- `artifacts/reports/track_b_backtest_results_after_cost_model_fix.md` (변경 전/후 비교 포함)

---

## 부록: 주요 설정 파일 요약

### `configs/config.yaml` 핵심 설정

```yaml
l4:
  holdout_years: 2
  step_days: 20
  test_window_days: 20
  embargo_days: 20
  horizon_short: 20
  horizon_long: 120
  rolling_train_years_short: 3
  rolling_train_years_long: 5

l5:
  model_type: "ridge"
  target_transform: "cs_rank"
  cs_rank_center: true
  ridge_alpha: 8.0  # [최종 픽스] 과적합 방지 강화
  min_feature_ic: -0.1  # [최종 픽스] 모든 피처 사용
  feature_list_short: "configs/features_short_v1.yaml"
  feature_list_long: "configs/features_long_v1.yaml"
  feature_weights_config_short: "configs/feature_weights_short_hitratio_optimized.yaml"
  feature_weights_config_long: "configs/feature_weights_long_ic_optimized.yaml"

l8_short:
  normalization_method: "zscore"  # [최종 픽스] 정규화 방법 비교 결과 최고 성과
  feature_weights_config: "configs/feature_weights_short_hitratio_optimized.yaml"

l8_long:
  normalization_method: "zscore"  # [최종 픽스] 정규화 방법 비교 결과 최고 성과
  feature_weights_config: "configs/feature_weights_long_ic_optimized.yaml"

l6r:
  alpha_short: 0.5  # 단기/장기 결합 가중치
  regime_alpha:     # 국면별 α 조정 (선택적)
    bull: 0.6
    neutral: 0.5
    bear: 0.4
```

---

## 참고 문서

- **투트랙 아키텍처 가이드**: `docs/TWO_TRACK_ARCHITECTURE.md` ⭐
- **Track B 기술 보고서**: `TECH_REPORT_TRACK2_BACKTEST.md`
- **최종 수치셋 정의**: `docs/FINAL_METRICS_DEFINITION.md`
- **Phase 8 최종 리포트**: `artifacts/reports/phase8_final_metrics_report.md`
- **Phase 9 계획**: `docs/PHASE9_PLAN.md`

---

**문서 버전**: Phase 9 + 랭킹산정모델 최종 픽스 (2026-01-07)  
**최종 업데이트**: 
- 뉴스 감성 피처 4개 추가 (단기 모델 18개→22개)
- 투트랙 구조 반영 (Track A: 랭킹 엔진)
- 코드 경로 업데이트 (`src/tracks/track_a/`)
- **L8-L5 피처셋 통일** (2026-01-07): L8이 L5와 동일한 피처셋 사용 (22개/19개)
- **정규화 방법 최적화** (2026-01-07): percentile/zscore/robust_zscore 비교 → **zscore 최종 선택** (50.28% Hit Ratio)
- **News 피처 가중치 최적화** (2026-01-07): 단기 0.10, 장기 0.03
- **단기 피처 가중치 미세 조정** (2026-01-07): 단기 전용 피처 0.025
- **최종 Hit Ratio 성과** (2026-01-07):
  - 통합 랭킹: 49.58% (전체), **51.06% (Holdout)** ✅ 목표 달성
  - 단기 랭킹: 49.28% (전체), **50.99% (Holdout)** ✅ 목표 달성
  - 장기 랭킹: **50.14% (전체)**, **51.00% (Holdout)** ✅ 목표 달성
- **최종 설정 픽스** (2026-01-07):
  - 정규화 방법: `zscore` (픽스)
  - `ridge_alpha`: 8.0 (픽스)
  - `min_feature_ic`: -0.1 (픽스)
  - 단기 News 피처 가중치: 0.10 (픽스)
  - 장기 News 피처 가중치: 0.03 (픽스)

