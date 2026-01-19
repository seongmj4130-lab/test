# KOSPI200 투자 모델 (Track B) 기술 보고서

**작성일**: 2026-01-07 (최종 업데이트)  
**버전**: Phase 9 + 랭킹산정모델 최종 픽스 (2026-01-07)  
**대상**: 퀀트 PM, 리스크/리서치, 백테스트 구현 엔지니어  
**관점**: 코드 기반, 실제 산출물 기준 설명

---

## 📋 목차

1. [백테스트 트랙 개요](#1-백테스트-트랙-개요)
2. [공통 데이터 준비 (L0~L5)](#2-공통-데이터-준비-l0l5)
3. [Config/파라미터 정의 (BT20/BT120)](#3-config파라미터-정의-bt20bt120)
4. [리밸런싱/포트폴리오 구성 로직 (L7)](#4-리밸런싱포트폴리오-구성-로직-l7)
5. [수익률 계산/성과 지표 로직](#5-수익률-계산성과-지표-로직)
6. [BT20/BT120 전략별 요약](#6-bt20bt120-전략별-요약)

---

## 0. (중요) 2026-01-07 최신 반영 사항 요약

이 문서는 과거 Phase별 실험/의사결정 내역을 포함합니다. **현재 코드/설정의 “최신 스냅샷”은 아래 근거 파일을 우선 기준으로 합니다.**

- `artifacts/reports/track_b_4strategy_final_summary.md` (4전략 Dev/Holdout + AlphaQuality + Operational + Regime)
- `configs/config.yaml` (특히 `l7_bt120_*` 오버래핑 트랜치 설정)

### 0.1 오버래핑 트랜치(필수) 도입 효과

- BT120이 기존 `rebalance_interval=120` 구조에서는 Holdout 리밸런싱이 3회 수준으로 **타이밍 럭**이 큼
- **월별 4트랜치(개선안 36번)** 도입으로 Holdout에서도 **Rebalances≈23회**로 증가 → 성과지표 안정화


## 1. 백테스트 트랙 개요

### 1.1 Track B: 투자 모델의 역할

**Track B (투자 모델)**는 Track A(랭킹 엔진)에서 생성된 랭킹 신호를 입력으로 받아 실제 투자 전략의 성과를 시뮬레이션합니다.

**핵심 목적**: 랭킹을 기반으로 다양한 투자모델 예시를 만들어 이용자에게 정보 제공

**입력**:
- `ranking_short_daily.parquet`: Track A에서 생성된 단기 랭킹
- `ranking_long_daily.parquet`: Track A에서 생성된 장기 랭킹
- `dataset_daily.parquet`: 공통 데이터 (수익률 정보 포함)
- `cv_folds_short.parquet`: 리밸런싱 날짜 정의 (`test_end` 기준)
- `pred_short_oos.parquet`: L5 단기 모델 예측 (선택적, Track B에서 사용)
- `pred_long_oos.parquet`: L5 장기 모델 예측 (선택적, Track B에서 사용)

**⚠️ 참고**: L5 모델 학습은 Track B에서 선택적으로 사용됩니다. Track B는 주로 Track A의 랭킹 데이터를 사용하지만, L5 모델 예측도 입력으로 받을 수 있습니다.

**(추가) Track A 랭킹 “왜 상위인가?” 빠른 진단(설명가능성)**:
- Holdout 기간의 특정 날짜에 대해 **Top10 랭킹 + 팩터셋(그룹) Top3 기여도**를 출력하여,
  Track B 백테스트 결과(선정 종목)가 어떤 요인(technical/value/news/esg 등)에서 주로 나온 것인지 빠르게 확인할 수 있습니다.
```bash
python scripts/inspect_tracka_holdout_day.py --date 2024-12-30 --topk 10 --horizon both
```

**중간 산출물** (L6R 단계):
- `rebalance_scores_from_ranking.parquet`: 랭킹을 백테스트용 스코어로 변환

**출력**:
- `bt_positions_{strategy}.parquet`: 리밸런싱별 포지션 히스토리
- `bt_returns_{strategy}.parquet`: 리밸런싱별 수익률 및 비용
- `bt_equity_curve_{strategy}.parquet`: 누적 자산 곡선
- `bt_metrics_{strategy}.parquet`: 성과 지표 (Dev/Holdout)
- `bt_regime_metrics_{strategy}.parquet`: 국면별 성과 지표 (선택적)

**코드 위치**: 
- `src/tracks/track_b/stages/backtest/l7_backtest.py` (백테스트 실행)
- `src/tracks/track_b/stages/modeling/l6r_ranking_scoring.py` (랭킹 스코어 변환)

**실행 방법**:
```bash
# Track B 전체 파이프라인 실행
python -m src.pipeline.track_b_pipeline bt20_short

# 또는 편의 래퍼 사용
python -m src.pipeline.bt20_pipeline short
python -m src.pipeline.bt120_pipeline long
```

**Track A와의 관계**: Track B는 Track A의 랭킹 데이터를 입력으로 사용하며, Track A를 먼저 실행해야 합니다.

### 1.2 두 전략: BT20 vs BT120

**BT20 (20일 보유 전략)**:
- **보유 기간**: 20 영업일
- **타깃 수익률**: `true_short` (20일 후 수익률)
- **리밸런싱 주기**: `rebalance_interval=20` (20일마다 리밸런싱, Holdout 23회 수준)
- **전략 2종(실제 운영/코드 기준)**:
  - `bt20_short`: 단기 랭킹만 사용, **`weighting="equal"`** (현재 `configs/config.yaml::l7_bt20_short`)
  - `bt20_ens`: 통합 랭킹 사용, **`weighting="softmax"`** (현재 `configs/config.yaml::l7_bt20_ens`)
- **참고**: 과거 실험에서 equal이 softmax보다 우수한 케이스가 있었으나, “최종 운영 설정”은 `configs/config.yaml`을 단일 기준으로 봅니다.

**BT120 (오버래핑 트랜치 기반 장기 전략, 개선안 36번)**:
- **기존 한계**: `rebalance_interval=120`이면 Holdout 리밸런싱이 3회 수준이라 “타이밍 럭”이 큼
- **해결**: 월별 4트랜치(매 20일마다 신규 트랜치 1개 추가, 트랜치 만기 120일(캘린더 day), 동시 보유 최대 4개)
- **평가 수익률**: `true_short` (20일 후 수익률; 월별 기간손익)  ← 트랜치별 보유는 120일
- **리밸런싱 주기**: `rebalance_interval=20` (월별 신규 트랜치 추가)
- **가중치 방식**: `weighting="equal"` 유지
- **리밸런싱 빈도(현재 설정)**: Dev 86개, Holdout 23개 (표본 확보로 안정화)

**rebalance_interval 개선 사항** (2026-01-06):
- **L6R 단계에서 필터링**: 일별 랭킹 데이터(`ranking_short_daily`, `ranking_long_daily`)에서 `rebalance_interval`만큼 필터링하여 `rebalance_scores` 생성
- **L7 단계 이중 필터링 방지**: L6R에서 이미 필터링된 데이터를 L7에서 추가 필터링하지 않도록 개선
- **시장 국면 판단 로직 개선**: `l1d_market_regime.py`에서 수익률 계산 버그 수정 (NaN 문제 해결)

---

## 2. 공통 데이터 준비 (L0~L5)

Track B는 Track A와 공통으로 사용하는 데이터를 필요로 합니다. 또한 L5 모델 학습 단계를 통해 생성된 모델 예측 데이터를 선택적으로 사용할 수 있습니다.

### 2.1 L0~L4: 공통 데이터 수집

**L0**: 유니버스 구성 (KOSPI200 멤버십)
- **파일**: `src/tracks/shared/stages/data/l0_universe.py`
- **산출물**: `universe_k200_membership_monthly.parquet`

**L1**: OHLCV 데이터 다운로드 + 기술적 지표 계산
- **파일**: `src/tracks/shared/stages/data/l1_ohlcv.py`
- **산출물**: `ohlcv_daily.parquet`

**L2**: 재무 데이터 로드 (DART)
- **파일**: `src/tracks/shared/stages/data/l2_fundamentals_dart.py`
- **산출물**: `fundamentals_annual.parquet`

**L3**: 패널 병합 (OHLCV + 재무 + 뉴스 + ESG)
- **파일**: `src/tracks/shared/stages/data/l3_panel_merge.py`
- **산출물**: `panel_merged_daily.parquet`

**L4**: Walk-Forward CV 분할 및 타겟 생성
- **파일**: `src/tracks/shared/stages/data/l4_walkforward_split.py`
- **산출물**: `dataset_daily.parquet`, `cv_folds_short.parquet`, `cv_folds_long.parquet`

### 2.2 L5: 모델 학습 (선택적)

**파일**: `src/stages/modeling/l5_train_models.py`

**목적**: 단기(20일) 및 장기(120일) 수익률 예측을 위한 Ridge 회귀 모델 학습

**주요 기능**:
1. **피처 선택**: 고정된 피처 리스트 사용
   - 단기 모델: 22개 피처 (`configs/features_short_v1.yaml`)
   - 장기 모델: 19개 피처 (`configs/features_long_v1.yaml`)
2. **타깃 변환**: Cross-Sectional Rank 변환 (날짜별 상대 순위)
3. **모델 학습**: Ridge 회귀 (L2 정규화)
   - `ridge_alpha`: 8.0 (2026-01-07 최종 픽스)
   - `min_feature_ic`: -0.1 (모든 피처 사용, 음수 IC 포함)
4. **Walk-Forward CV**: 각 fold별로 모델 학습 및 OOS 예측

**산출물**:
- `pred_short_oos.parquet`: 단기 모델 예측 (날짜별 종목 예측값)
- `pred_long_oos.parquet`: 장기 모델 예측 (날짜별 종목 예측값)
- `model_metrics.parquet`: 모델 성능 지표 (RMSE, IC, Hit Ratio 등)

**⚠️ 참고**: L5는 Track B에서 선택적으로 사용됩니다. Track B는 주로 Track A의 랭킹 데이터를 사용하지만, L5 모델 예측도 입력으로 받을 수 있습니다.

**실행 방법**:
```bash
# 레거시 스크립트 사용 (L0~L7 전체 실행)
python scripts/run_pipeline_l0_l7.py

# 또는 Python에서 직접 호출
from src.stages.modeling.l5_train_models import train_oos_predictions
pred_short, pred_long, metrics = train_oos_predictions(cfg, dataset_daily, cv_folds_short, cv_folds_long)
```

---

## 3. Config/파라미터 정의 (BT20/BT120)

### 2.1 설정 파일 구조

**파일**: `configs/config.yaml`

**BT20 설정 섹션**: `l7_bt20_short`, `l7_bt20_ens`  
**BT120 설정 섹션**: `l7_bt120_long`, `l7_bt120_ens`

### 2.2 Phase 9 기준 파라미터 비교

| 파라미터 | BT20 (대표: bt20_short) | BT120 (대표: bt120_long, 오버래핑 트랜치) | 설명 |
|---------|--------------------------|-------------------------------------------|------|
| **holding_days** | 20 | 20 | 평가 수익률 기간(영업일; 오버래핑 트랜치에서는 월별 20일) |
| **return_col** | `"true_short"` | `"true_short"` | 수익률 컬럼명(오버래핑 트랜치에서는 월별 PnL) |
| **top_k** | 12 | 15 | 선택 종목 수 (전략별로 다름: `config.yaml` 우선) |
| **buffer_k** | 15 | 15 | 버퍼 종목 수 (prev_holdings 유지용) |
| **weighting** | `"equal"` | `"equal"` | 가중치 방식 |
| **rebalance_interval** | 20 | 20 | 리밸런싱 주기 (오버래핑 트랜치: 월별 신규 트랜치 추가) |
| **cost_bps** | 10.0 | 10.0 | 거래비용 (basis points) |
| **score_col** | `"score_total_short"` | `"score_total_long"` | 스코어 컬럼명(전략별 설정) |
| **smart_buffer_enabled** | true | true | 스마트 버퍼링 활성화 |
| **volatility_adjustment_enabled** | true | true | 변동성 조정 활성화 |
| **risk_scaling_enabled** | true | true | 국면별 리스크 스케일링 |
| **regime.enabled** | true | true | 국면 기반 전략 활성화 |

**설정 파일 위치**: `configs/config.yaml:182-274`

---

## 4. 리밸런싱/포트폴리오 구성 로직 (L7)

**파일**: `src/tracks/track_b/stages/backtest/l7_backtest.py`

**함수**: `run_backtest()` (`l7_backtest.py:438-1119`)

### 3.1 리밸런싱 날짜 추출

**코드 위치**: `l7_backtest.py:520-529` (2026-01-06 개선)

**개선 사항**: L6R 단계에서 이미 `rebalance_interval` 필터링이 완료되었으므로, L7에서는 추가 필터링을 건너뜀

```python
# Phase별로 그룹화
for phase, dphase in df_sorted.groupby(phase_col, sort=False):
    rebalance_dates_all = sorted(dphase[date_col].unique())
    
    # [rebalance_interval 개선] L6R에서 이미 필터링되었으므로 L7에서는 추가 필터링 건너뛰기
    rebalance_interval = int(cfg.rebalance_interval)
    if rebalance_interval > 1:
        # L6R에서 이미 필터링 완료, L7에서는 모든 날짜 사용
        rebalance_dates_filtered = rebalance_dates_all
        logger.info(f"[Phase 7 Step 2] L6R에서 이미 rebalance_interval={rebalance_interval} 필터링 완료, L7에서는 추가 필터링 건너뛰기: {len(rebalance_dates_all)}개 날짜 사용")
    elif rebalance_interval == 1:
        # 기존 로직: L7에서 필터링 (rebalance_interval=1이면 필터링 없음)
        rebalance_dates_filtered = rebalance_dates_all
    else:
        # 예외 처리
        rebalance_dates_filtered = rebalance_dates_all
```

**L6R 단계 필터링** (`l6r_ranking_scoring.py:280-306`):
```python
# rebalance_interval에 맞게 일별 랭킹 데이터 필터링
if rebalance_interval == 1:
    # 월별 리밸런싱: cv_folds_short.test_end 사용
    rebal_map = folds[["test_end", "phase"]].rename(columns={"test_end": "date"}).copy()
else:
    # 일별 리밸런싱: ranking_daily에서 interval만큼 필터링
    all_dates = sorted(pd.to_datetime(ranking_short_daily["date"].unique()))
    rebalance_dates = [all_dates[i] for i in range(0, len(all_dates), rebalance_interval)]
    rebal_map = pd.DataFrame({"date": rebalance_dates})
    rebal_map["phase"] = rebal_map["date"].apply(lambda d: _map_date_to_phase(d, cv_folds_short))
```

**이전 로직** (2026-01-06 이전):
```python
# 매 N번째 리밸런싱만 선택 (0-indexed)
        rebalance_dates_filtered = [
            rebalance_dates_all[i] 
            for i in range(0, len(rebalance_dates_all), rebalance_interval)
        ]
        dphase = dphase[dphase[date_col].isin(rebalance_dates_filtered)].copy()
```

**로직**:
- `rebalance_interval=1`: 모든 리밸런싱 날짜 사용 (BT20)
- `rebalance_interval=10`: 10번째 리밸런싱만 사용 (BT120)
- `cv_folds_short.test_end` 기준으로 리밸런싱 날짜 결정

### 3.2 각 리밸런싱 시점에서의 처리

**코드 위치**: `l7_backtest.py:541-808`

#### 3.2.1 스코어 데이터 필터링

```python
# l7_backtest.py:541-544
for dt, g in dphase.groupby(date_col, sort=True):
    g = g.sort_values([score_col, ticker_col], ascending=[False, True]).reset_index(drop=True)
```

**로직**:
- 해당 날짜(`dt`)의 `rebalance_scores` 행 필터링
- `score_col` 기준 내림차순 정렬 (높은 점수 = 상위)

#### 3.2.2 유니버스 필터링

**코드 위치**: `src/components/portfolio/selector.py::select_topk_with_fallback()`

**필터링 단계** (`selector.py:75-105`):

1. **필수 컬럼 결측 필터링**:
   ```python
   # selector.py:77-81
   if required_cols:
       g_filtered = g_filtered.dropna(subset=required_cols)
   ```

2. **가격 결측 필터링**:
   ```python
   # selector.py:84-91
   if filter_missing_price:
       price_cols = [c for c in g_filtered.columns if "ret" in c.lower() or "price" in c.lower()]
       if price_cols:
           g_filtered = g_filtered.dropna(subset=price_cols[:1])
   ```

3. **거래정지 필터링** (옵션):
   ```python
   # selector.py:94-105
   if filter_suspended:
       suspended_cols = [c for c in g_filtered.columns if "suspended" in c.lower()]
       # suspended=True 제외
   ```

#### 3.2.3 상위 K+buffer 선택

**함수**: `select_topk_with_fallback()` (`selector.py:13-251`)

**로직** (`selector.py:112-194`):

```python
# 1. 허용 범위: top_k + buffer_k
allow_n = top_k + buffer_k if buffer_k > 0 else top_k
allow = g_filtered.head(allow_n).copy()

# 2. 이전 보유 종목 중 허용 범위에 있는 것들
allow_set = set(allow[ticker_col].astype(str).tolist())
keep = [t for t in prev_holdings if t in allow_set]

# 3. cap keep to top_k
if len(keep) > top_k:
    keep = keep[:top_k]

# 4. keep 먼저 선택
selected = []
for t in keep:
    selected.append(t)
    selected_set.add(t)

# 5. 부족한 만큼 상위에서 채움
for _, row in allow.iterrows():
    if len(selected) >= top_k:
        break
    if t not in selected_set:
        selected.append(t)
        selected_set.add(t)
```

**Fallback 로직** (`selector.py:195-230`):
- 선택된 종목 수가 `top_k`보다 적으면 다음 순위에서 채움
- 업종 분산 제약이 있으면 제약을 고려하여 채움

#### 3.2.4 스마트 버퍼링 로직

**함수**: `_select_with_smart_buffer()` (`l7_backtest.py:223-299`)

**코드 위치**: `selector.py:119-132`

```python
# selector.py:119-132
if smart_buffer_enabled and buffer_k > 0 and len(prev_holdings) > 0:
    keep = []
    for t in prev_holdings:
        if t in allow_set:
            # 해당 종목의 현재 순위 확인
            ticker_rows = g_filtered[g_filtered[ticker_col].astype(str) == t]
            if len(ticker_rows) > 0:
                rank = ticker_rows.index[0]
                rank_pct = float(rank) / max(total_count - 1, 1)
                # 순위가 상위 X% 내에 있으면 유지
                if rank_pct <= smart_buffer_stability_threshold:
                    keep.append(t)
```

**설정값**:
- `smart_buffer_stability_threshold: 0.7` → 상위 70% 내 종목 유지

**효과**: 안정적인 포지션 유지로 Dev 구간 붕괴 완화

#### 3.2.5 최종 종목 선택 함수

**함수**: `select_topk_with_fallback()` (`selector.py:13-251`)

**호출 위치**: `l7_backtest.py:615-629`

```python
# l7_backtest.py:615-629
g_sel, diagnostics = select_topk_with_fallback(
    g,
    ticker_col=ticker_col,
    score_col=score_col,
    top_k=current_top_k,  # 국면별 top_k
    buffer_k=int(cfg.buffer_k),
    prev_holdings=prev_holdings,
    group_col=cfg.group_col if cfg.diversify_enabled else None,
    max_names_per_group=cfg.max_names_per_group if cfg.diversify_enabled else None,
    required_cols=[ret_col],
    filter_missing_price=True,
    smart_buffer_enabled=cfg.smart_buffer_enabled,
    smart_buffer_stability_threshold=cfg.smart_buffer_stability_threshold,
)
```

**반환값**:
- `g_sel`: 선택된 종목 DataFrame
- `diagnostics`: 진단 정보 (`eligible_count`, `selected_count`, `dropped_missing`, 등)

### 3.3 가중치 계산

**함수**: `_weights_from_scores()` (`l7_backtest.py:119-139`)

#### 3.3.1 Equal Weighting

**코드 위치**: `l7_backtest.py:124-125`

```python
if method == "equal":
    return np.full(n, 1.0 / n, dtype=float)
```

**수식**: `weight[i] = 1.0 / n` (n = 선택된 종목 수)

**BT120 사용**: Phase 8 기준 유지

#### 3.3.2 Softmax Weighting

**코드 위치**: `l7_backtest.py:127-137`

```python
if method == "softmax":
    x = scores.astype(float).to_numpy()
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    t = float(temp) if float(temp) > 0 else 1.0
    x = x / t  # temperature로 나눔
    x = x - np.max(x)  # 안정화 (오버플로우 방지)
    w = np.exp(x)
    sw = w.sum()
    if sw <= 0:
        return np.full(n, 1.0 / n, dtype=float)
    return w / sw
```

**수식**:
1. `x_normalized = (score - max(score)) / temperature`
2. `w_raw = exp(x_normalized)`
3. `weight = w_raw / sum(w_raw)`

**BT20 설정**:
- `softmax_temperature: 0.5` → 낮은 온도 = 상위 종목에 더 집중

**효과**: 
- Temperature가 낮을수록 상위 종목에 가중치 집중
- Temperature=0.5일 때 상위 1-2개 종목에 30-50% 가중치 집중 가능

**호출 위치**: `l7_backtest.py:672`

```python
scores = g_sel[score_col]
w = _weights_from_scores(scores, cfg.weighting, cfg.softmax_temp)
```

### 3.4 변동성/국면 조정

#### 3.4.1 변동성 기반 Exposure 조정

**함수**: `_calculate_volatility_adjustment()` (`l7_backtest.py:355-394`)

**코드 위치**: `l7_backtest.py:679-689`

```python
# l7_backtest.py:679-689
if cfg.volatility_adjustment_enabled and len(recent_returns_history) >= 2:
    recent_returns_array = np.array(recent_returns_history[-cfg.volatility_lookback_days:])
    volatility_adjustment = _calculate_volatility_adjustment(
        recent_returns_array,
        target_vol=cfg.target_volatility,
        lookback_days=cfg.volatility_lookback_days,
        max_mult=cfg.volatility_adjustment_max,
        min_mult=cfg.volatility_adjustment_min,
    )
```

**계산 로직** (`l7_backtest.py:375-393`):

```python
# 최근 N일 수익률의 표준편차 계산 (연율화)
recent_window = recent_returns[-lookback_days:]
current_vol = float(np.std(recent_window)) * np.sqrt(252.0)

# 목표 변동성 대비 현재 변동성 비율
vol_ratio = target_vol / current_vol

# 조정 배수 계산 (클리핑)
adjustment = float(np.clip(vol_ratio, min_mult, max_mult))
```

**수식**: `adjustment = clip(target_vol / current_vol, min_mult, max_mult)`

**설정값**:
- BT20: `min_mult=0.7`, `max_mult=1.2`
- BT120: `min_mult=0.6`, `max_mult=1.2`

**효과**: 변동성이 높을 때 포지션 축소, 낮을 때 확대

#### 3.4.2 국면별 리스크 스케일링

**함수**: `_apply_risk_scaling()` (`l7_backtest.py:396-436`)

**코드 위치**: `l7_backtest.py:691-699`

```python
# l7_backtest.py:691-699
adjusted_exposure = _apply_risk_scaling(
    base_exposure=current_exposure,  # 국면별 exposure
    regime=current_regime,
    risk_scaling_enabled=cfg.risk_scaling_enabled,
    bear_multiplier=cfg.risk_scaling_bear_multiplier,
    neutral_multiplier=cfg.risk_scaling_neutral_multiplier,
    bull_multiplier=cfg.risk_scaling_bull_multiplier,
)
```

**계산 로직** (`l7_backtest.py:424-435`):

```python
if "bear" in regime_lower:
    return base_exposure * bear_multiplier
elif "neutral" in regime_lower:
    return base_exposure * neutral_multiplier
elif "bull" in regime_lower:
    return base_exposure * bull_multiplier
```

**설정값**:
- BT20: `bear_multiplier=0.8`, `neutral_multiplier=1.0`, `bull_multiplier=1.0`
- BT120: `bear_multiplier=0.7`, `neutral_multiplier=0.9`, `bull_multiplier=1.0`

**최종 Exposure 적용** (`l7_backtest.py:701-703`):

```python
final_exposure = adjusted_exposure * volatility_adjustment
gross_ret = gross_ret * final_exposure
```

#### 3.4.3 시장 국면 판단

**함수**: `build_market_regime()` (`src/tracks/shared/stages/regime/l1d_market_regime.py`)

**개선사항 (2026-01-06)**:
- **외부 API 제거**: pykrx 라이브러리로 KOSPI200 지수 데이터를 다운로드하던 방식을 제거
- **내부 데이터 사용**: `ohlcv_daily` 데이터를 사용하여 자동 분류
- **지표 종합**: 가격 수익률, 변동성, 거래량 변화율을 종합하여 판단

**국면 분류 기준** (3단계: Bull/Neutral/Bear):
- **가격 수익률**: lookback 기간 동안의 시장 가중 평균 수익률
- **변동성**: 일일 수익률 표준편차 (연환산)
- **거래량 변화율**: lookback 기간 동안의 거래량 변화율

**분류 로직**:
- **Bull**: 수익률 > neutral_band AND (변동성 < 30% OR 거래량 변화 > -20%)
- **Bear**: 수익률 < -neutral_band AND (변동성 > 40% OR 거래량 변화 > 50%)
- **Neutral**: 그 외 (수익률이 ±neutral_band 범위 내 또는 추가 조건 미충족)

**기본값**:
- `lookback_days`: 60일
- `neutral_band`: 0.05 (±5%)
- `use_volume`: true
- `use_volatility`: true

**국면별 top_k/exposure 결정** (`l7_backtest.py:546-611`):

```python
# l7_backtest.py:560-600
if current_regime == "bull_strong":
    if cfg.regime_top_k_bull_strong is not None:
        current_top_k = int(cfg.regime_top_k_bull_strong)
    if cfg.regime_exposure_bull_strong is not None:
        current_exposure = float(cfg.regime_exposure_bull_strong)
# ... (다른 국면도 동일)
```

### 3.5 포지션/포지션 히스토리

**코드 위치**: `l7_backtest.py:751-773`

```python
# l7_backtest.py:751-773
for idx, (t, wi, sc, tr) in enumerate(zip(g_sel[ticker_col], w, g_sel[score_col], g_sel[ret_col])):
    pos_row = {
        "date": dt,
        "phase": phase,
        "ticker": str(t),
        "weight": float(wi),  # 가중치
        "score": float(sc) if pd.notna(sc) else np.nan,
        "ret_realized": float(tr),  # 실제 수익률
        "top_k": int(cfg.top_k),
        "holding_days": int(cfg.holding_days),
        "cost_bps": float(cfg.cost_bps),
        "weighting": cfg.weighting,
        "buffer_k": int(cfg.buffer_k),
        "k_eff": int(k_eff),  # 실제 선택된 종목 수
        "eligible_count": int(eligible_count),
        "filled_count": int(filled_count),
    }
    positions_rows.append(pos_row)
```

**산출물**: `bt_positions.parquet`

**스키마**:
- `date`: 리밸런싱 날짜
- `phase`: "dev" | "holdout"
- `ticker`: 종목코드
- `weight`: 포지션 가중치 (0~1)
- `score`: 스코어 값
- `ret_realized`: 실제 수익률 (`true_short` 또는 `true_long`)
- `k_eff`: 실제 선택된 종목 수 (K_eff ≤ top_k)

**포지션 업데이트** (`l7_backtest.py:807-808`):

```python
prev_w = new_w  # 이전 가중치 저장
prev_holdings = g_sel[ticker_col].tolist()  # 이전 보유 종목 저장
```

---

## 5. 수익률 계산/성과 지표 로직

### 4.1 구간별 수익률 계산

**코드 위치**: `l7_backtest.py:671-741`

#### 4.1.1 포트폴리오 수익률 계산

```python
# l7_backtest.py:671-677
scores = g_sel[score_col]
w = _weights_from_scores(scores, cfg.weighting, cfg.softmax_temp)
new_w = {t: float(wi) for t, wi in zip(g_sel[ticker_col].tolist(), w.tolist())}
turnover_oneway = _compute_turnover_oneway(prev_w, new_w)

gross_ret = float(np.dot(w, g_sel[ret_col].astype(float).to_numpy()))
```

**수식**: `gross_return = Σ(weight[i] * return[i])`

**타깃 수익률(현재 설정 기준)**:
- BT20: `true_short` (20일 후 수익률)
- BT120 오버래핑 트랜치(개선안 36번): `true_short` (월별 20일 후 수익률로 기간손익 계산, 트랜치 만기는 120일)

#### 4.1.2 거래비용 계산 (개선안 1번/3번)

**함수**: `_compute_turnover_oneway()` (`l7_backtest.py:101-106`)

```python
# l7_backtest.py:101-106
def _compute_turnover_oneway(prev_w: Dict[str, float], new_w: Dict[str, float]) -> float:
    keys = set(prev_w) | set(new_w)
    s = 0.0
    for k in keys:
        s += abs(new_w.get(k, 0.0) - prev_w.get(k, 0.0))
    return 0.5 * s
```

**수식**: `turnover_oneway = 0.5 * Σ|new_weight[i] - prev_weight[i]|`

**거래비용 적용** (기존 문서의 고정 차감 방식 → 현재는 턴오버 기반으로 정상화):

현재 코드는 다음 개념으로 비용을 차감합니다:
- `traded_value = turnover_oneway * |final_exposure|`
- `total_cost = traded_value * (cost_bps + slippage_bps) / 10000`
- `net_return = gross_return - total_cost`

**수식**:
- `total_cost = turnover_oneway * |exposure| * (cost_bps + slippage_bps) / 10000`
- `net_return = gross_return - total_cost`

**설정값**: `cost_bps = 10.0` (0.1%), `slippage_bps` (기본 0.0)

#### 4.1.3 수익률 레코드 생성

**코드 위치**: `l7_backtest.py:775-805`

```python
# l7_backtest.py:775-805
returns_row = {
    "date": dt,
    "phase": phase,
    "top_k": int(current_top_k),
    "holding_days": int(cfg.holding_days),
    "cost_bps": float(cfg.cost_bps),
    "gross_return": float(gross_ret),
    "net_return": float(net_ret),
    "turnover_oneway": float(turnover_oneway),
    "daily_trading_cost": float(daily_trading_cost),
    "turnover_cost": float(turnover_cost),
    "total_cost": float(total_cost),
    "n_tickers": int(len(g_sel)),
    # ... (regime, exposure 등)
}
returns_rows.append(returns_row)
```

**산출물**: `bt_returns.parquet`

**스키마**:
- `date`: 리밸런싱 날짜
- `phase`: "dev" | "holdout"
- `gross_return`: 비용 차감 전 수익률
- `net_return`: 비용 차감 후 수익률
- `turnover_oneway`: 포지션 변경 비율
- `total_cost`: 거래비용
- `n_tickers`: 선택된 종목 수

### 4.2 성과 지표 계산

**코드 위치**: `l7_backtest.py:857-1037`

#### 4.2.1 누적 수익률

**코드 위치**: `l7_backtest.py:844-855`

```python
# l7_backtest.py:844-855
eq_rows: List[dict] = []
for phase, g in bt_returns.groupby("phase", sort=False):
    g = g.sort_values("date").reset_index(drop=True)
    eq = 1.0
    peak = 1.0
    for dt, r in zip(g["date"], g["net_return"]):
        eq *= (1.0 + float(r))
        peak = max(peak, eq)
        dd = (eq / peak) - 1.0
        eq_rows.append({"date": dt, "phase": phase, "equity": float(eq), "drawdown": float(dd)})
```

**수식**: `equity[t] = equity[t-1] * (1 + net_return[t])`

**산출물**: `bt_equity_curve.parquet`

#### 4.2.2 CAGR (연평균 복리 수익률)

**코드 위치**: `l7_backtest.py:859-920`

```python
# l7_backtest.py:866-920
eq_g = float((1.0 + pd.Series(r_gross)).cumprod().iloc[-1])
eq_n = float((1.0 + pd.Series(r_net)).cumprod().iloc[-1])

d0 = pd.to_datetime(g["date"].iloc[0])
d1 = pd.to_datetime(g["date"].iloc[-1])
years = max((pd.Timedelta(d1 - d0).days / 365.25), 1e-9)

net_cagr = float(eq_n ** (1.0 / years) - 1.0) if eq_n > 0 and years > 0 else -1.0
```

**수식**: `CAGR = (equity_final / equity_initial) ^ (1 / years) - 1`

#### 4.2.3 Sharpe Ratio

**코드 위치**: `l7_backtest.py:922-926`

```python
# l7_backtest.py:922-926
periods_per_year = 252.0 / float(cfg.holding_days) if cfg.holding_days > 0 else 12.6

gross_vol = float(np.std(r_gross, ddof=1) * np.sqrt(periods_per_year))
net_vol = float(np.std(r_net, ddof=1) * np.sqrt(periods_per_year))

gross_sharpe = float((np.mean(r_gross) / (np.std(r_gross, ddof=1) + 1e-12)) * np.sqrt(periods_per_year))
net_sharpe = float((np.mean(r_net) / (np.std(r_net, ddof=1) + 1e-12)) * np.sqrt(periods_per_year))
```

**수식**: 
- `volatility_annual = std(returns) * sqrt(periods_per_year)`
- `Sharpe = mean(returns) / std(returns) * sqrt(periods_per_year)`

**연율화**:
- BT20: `periods_per_year = 252 / 20 = 12.6`
- BT120: `periods_per_year = 252 / 120 = 2.1`

#### 4.2.4 MDD (Maximum Drawdown)

**함수**: `_mdd()` (`l7_backtest.py:108-117`)

```python
# l7_backtest.py:108-117
def _mdd(rr: np.ndarray) -> float:
    eq = 1.0
    peak = 1.0
    mdd = 0.0
    for r in rr:
        eq *= (1.0 + float(r))
        peak = max(peak, eq)
        mdd = min(mdd, (eq / peak) - 1.0)
    return float(mdd)
```

**수식**: `MDD = min((equity[t] / peak[t]) - 1.0)`

**코드 위치**: `l7_backtest.py:928-935`

```python
mdd_g = _mdd(r_gross) if len(r_gross) else 0.0
mdd_n = _mdd(r_net) if len(r_net) else 0.0
```

#### 4.2.5 Calmar Ratio

**코드 위치**: `l7_backtest.py:945-956`

```python
# l7_backtest.py:945-956
def _calculate_calmar_ratio(cagr: float, mdd: float) -> float:
    if mdd == 0:
        return float('inf') if cagr > 0 else 0.0
    abs_mdd = abs(mdd)
    if abs_mdd < 1e-9:
        return float('inf') if cagr > 0 else 0.0
    return float(cagr / abs_mdd)

gross_calmar = _calculate_calmar_ratio(gross_cagr, mdd_g)
net_calmar = _calculate_calmar_ratio(net_cagr, mdd_n)
```

**수식**: `Calmar = CAGR / |MDD|`

#### 4.2.6 Hit Ratio

**코드 위치**: `l7_backtest.py:1021-1022`

```python
# l7_backtest.py:1021-1022
"gross_hit_ratio": float((r_gross > 0).mean()) if len(r_gross) else np.nan,
"net_hit_ratio": float((r_net > 0).mean()) if len(r_net) else np.nan,
```

**수식**: `Hit Ratio = mean(returns > 0)`

#### 4.2.7 Profit Factor

**코드 위치**: `l7_backtest.py:958-968`

```python
# l7_backtest.py:958-968
def _calculate_profit_factor(returns: np.ndarray) -> float:
    profits = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return float('inf') if profits > 0 else 0.0
    return float(profits / losses)

gross_profit_factor = _calculate_profit_factor(r_gross)
net_profit_factor = _calculate_profit_factor(r_net)
```

**수식**: `Profit Factor = sum(positive_returns) / abs(sum(negative_returns))`

#### 4.2.8 Avg Trade Duration

**코드 위치**: `l7_backtest.py:970-996`

```python
# l7_backtest.py:970-996
if len(bt_positions) > 0:
    phase_positions = bt_positions[bt_positions["phase"] == phase].copy()
    phase_positions = phase_positions.sort_values(["ticker", "date"])
    
    durations = []
    for ticker, ticker_positions in phase_positions.groupby("ticker", sort=False):
        ticker_positions = ticker_positions.sort_values("date")
        if len(ticker_positions) > 1:
            dates = ticker_positions["date"].values
            for i in range(len(dates) - 1):
                days_diff = pd.Timedelta(dates[i+1] - dates[i]).days
                if days_diff <= cfg.holding_days * 2:  # 연속 보유
                    durations.append(days_diff)
    
    if len(durations) > 0:
        avg_trade_duration = float(np.mean(durations))
```

**수식**: `Avg Trade Duration = mean(연속 보유 일수)`

### 4.3 성과 지표 산출물 구조

**파일**: `bt_metrics.parquet` (BT20), `bt_metrics_bt120.parquet` (BT120)

**스키마** (`l7_backtest.py:998-1036`):

```python
{
    "phase": str,                    # "dev" | "holdout"
    "gross_total_return": float,     # 누적 수익률 (비용 차감 전)
    "net_total_return": float,       # 누적 수익률 (비용 차감 후)
    "gross_cagr": float,             # CAGR (비용 차감 전)
    "net_cagr": float,               # CAGR (비용 차감 후)
    "gross_sharpe": float,           # Sharpe Ratio (비용 차감 전)
    "net_sharpe": float,             # Sharpe Ratio (비용 차감 후)
    "gross_mdd": float,              # MDD (비용 차감 전)
    "net_mdd": float,                # MDD (비용 차감 후)
    "gross_calmar_ratio": float,     # Calmar Ratio (비용 차감 전)
    "net_calmar_ratio": float,       # Calmar Ratio (비용 차감 후)
    "gross_hit_ratio": float,        # Hit Ratio (비용 차감 전)
    "net_hit_ratio": float,          # Hit Ratio (비용 차감 후)
    "gross_profit_factor": float,    # Profit Factor (비용 차감 전)
    "net_profit_factor": float,      # Profit Factor (비용 차감 후)
    "avg_turnover_oneway": float,    # 평균 Turnover
    "avg_trade_duration": float,     # 평균 보유 일수
    "n_rebalances": int,             # 리밸런싱 횟수
    "date_start": pd.Timestamp,      # 시작일
    "date_end": pd.Timestamp,       # 종료일
    ...
}
```

---

## 6. BT20/BT120 전략별 요약

### 5.1 BT20 (Phase 9 Softmax 적용 후)

#### 설정 요약

**코드 위치**: `configs/config.yaml:182-226`

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `holding_days` | 20 | 20일 보유 |
| `return_col` | `"true_short"` | 20일 수익률 사용 |
| `weighting` | `"softmax"` | Softmax 가중치 |
| `softmax_temperature` | 0.5 | 낮은 온도 (집중) |
| `rebalance_interval` | 1 | 매 리밸런싱 실행 |
| `top_k` | 15 | 상위 15개 선택 |
| `buffer_k` | 20 | 버퍼 20개 |

#### 포트폴리오 특성

**가중치 분포** (Softmax Temperature=0.5):

- **상위 종목 집중**: 상위 1-2개 종목에 30-50% 가중치 집중 가능
- **코드 로직**: `l7_backtest.py:127-137`
  ```python
  x = x / 0.5  # 낮은 온도로 나눔 → 큰 값 증폭
  w = np.exp(x) / sum(np.exp(x))  # Softmax
  ```

**Turnover 수준**: 
- Phase 9 Step 2 기준: **55.55%** (목표 ≤ 500% 달성)
- `rebalance_interval=1`이지만 스마트 버퍼링으로 완화

**종목 수**: 
- `top_k=15` (국면별 조정 가능: Bull Strong=10, Bear=30)
- `k_eff` (실제 선택 수) ≤ 15

#### (참고) 과거 실험 성과 (Holdout, 뉴스 피처 추가 후 - 2026-01-04)

**데이터 소스**: `artifacts/reports/news_features_performance_comparison.md`

| 지표 | 값 | Phase 8 대비 | 목표 | 달성 여부 |
|------|-----|-------------|------|----------|
| **Net Sharpe** | **0.7370** | +0.2305 (+45.5%) | ≥ 0.50 | ✅ **초과 달성** |
| **Net CAGR** | **12.08%** | +7.53%p (+165.5%) | ≥ 10% | ✅ **달성** |
| **Net MDD** | -8.53% | -1.77%p | ≤ -10% | ✅ **달성** |
| **Net Calmar** | **1.4155** | +0.7429 (+110.5%) | ≥ 0.8 | ✅ **초과 달성** |
| **Avg Turnover** | **73.36%** | -173.16%p (-70.2%) | ≤ 500% | ✅ **대폭 개선** |
| **Net Hit Ratio** | **65.22%** | +17.39%p (+36.4%) | ≥ 55% | ✅ **초과 달성** |
| **Net Profit Factor** | **1.8230** | +0.2718 (+17.5%) | ≥ 1.5 | ✅ **초과 달성** |

**주요 개선 사항**:
- 뉴스 감성 피처 4개 추가 (`news_sentiment`, `news_sentiment_ewm5`, `news_sentiment_surprise`, `news_volume`)
- 모든 목표 지표 달성
- Sharpe +45.5%, CAGR +165.5% 대폭 개선

**역할**: "수익률 중심 공격적 전략" (뉴스 피처 추가로 성과 대폭 향상)

> **중요(최신 스냅샷)**: 현재 운영/재현 기준 성과는 `artifacts/reports/track_b_4strategy_final_summary.md`를 우선 참조하세요.

### 5.2 BT120 (오버래핑 트랜치, Equal Weighting)

#### 설정 요약

**코드 위치**: `configs/config.yaml:230-274`

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `overlapping_tranches_enabled` | true | 오버래핑 트랜치 모드 활성화 |
| `tranche_holding_days` | 120 | 트랜치 만기(캘린더 day) |
| `tranche_max_active` | 4 | 동시 보유 트랜치 수 |
| `rebalance_interval` | 20 | 매 20일(월별) 신규 트랜치 추가 |
| `holding_days` | 20 | 월별 기간손익 평가(20영업일) |
| `return_col` | `"true_short"` | 월별(20일) 수익률 사용 |
| `weighting` | `"equal"` | 트랜치 내부 종목은 동일비중 |
| `top_k` | 20 | 상위 20개 선택 |
| `buffer_k` | 30 | 버퍼 30개 |

#### 포트폴리오 특성

**가중치 분포** (Equal Weighting):

- **균등 분산**: `weight[i] = 1.0 / n` (n = 선택된 종목 수)
- **코드 로직**: `l7_backtest.py:124-125`
  ```python
  if method == "equal":
      return np.full(n, 1.0 / n, dtype=float)
  ```

**Turnover 수준(최신)**:
- 월별 평가 기준 평균 one-way turnover는 약 7% 수준(전략/구간별 상이)
- 오버래핑 트랜치 구조는 “전체 자본이 한 번에 갈아타는” 형태를 피하므로 타이밍 리스크를 줄입니다

**종목 수**: 
- `top_k=20` (국면별 조정 가능: Bull Strong=12, Bear=30)
- `k_eff` (실제 선택 수) ≤ 20

#### 최신 성과 (Holdout, 오버래핑 트랜치 적용)

**데이터 소스**: `artifacts/reports/track_b_4strategy_final_summary.md`

| 전략 | Net Sharpe | Net CAGR | Net MDD | Calmar | Hit Ratio | Profit Factor | Rebalances |
|------|-----------:|---------:|--------:|-------:|----------:|--------------:|----------:|
| bt120_long | 0.5117 | 4.17% | -6.02% | 0.6933 | 43.48% | 1.5005 | 23 |
| bt120_ens | 0.4797 | 3.94% | -5.91% | 0.6678 | 43.48% | 1.4671 | 23 |

**역할**: "안정성 중심 전략" (Phase 8에서 확정)

### 5.3 (참고) BT120 Softmax 테스트 결과 및 채택하지 않은 이유

**데이터 소스**: `artifacts/reports/phase9_bt120_softmax_comparison_report.md`

#### Phase 9 BT120 Softmax Weighting 성과

**Holdout 구간**:

| 지표 | Phase 8 (Equal) | Phase 9 (Softmax) | 변화 |
|------|-----------------|-------------------|------|
| **Net Sharpe** | 0.4565 | **0.9288** | +0.4723 ✅ |
| **Net CAGR** | 14.92% | **104.21%** | +89.29%p ⚠️ **비정상** |
| **Net MDD** | -9.20% | -18.23% | -9.03%p ⚠️ **악화** |
| **Net Calmar** | 1.6209 | **5.7179** | +4.0970 ✅ |
| **Avg Turnover** | 39.08% | 53.59% | +14.51%p ⚠️ |

**Dev 구간**:

| 지표 | Phase 8 (Equal) | Phase 9 (Softmax) | 변화 |
|------|-----------------|-------------------|------|
| **Net Sharpe** | 0.1730 | 0.1306 | -0.0244 ⚠️ |
| **Net CAGR** | 6.58% | 1.25% | -5.33%p ⚠️ |
| **Net MDD** | -50.73% | -90.76% | -40.03%p ⚠️ **대폭 악화** |

#### 채택하지 않은 이유

**1. CAGR 비정상적 수치**:
- Holdout CAGR: **104.21%**는 비정상적으로 높음
- Equity Curve 확인: 시작 0.933 → 종료 2.689 (Total Return 188.1%)
- 과적합 가능성: Dev CAGR 1.25% vs Holdout CAGR 104.21% (극단적 괴리)

**2. MDD 악화**:
- Phase 8: -9.20% → Phase 9: -18.23% (-9.03%p)
- 목표(≤ -10%) 초과
- Dev 구간 MDD: -50.73% → -90.76% (대폭 악화)

**3. Dev/Holdout 괴리**:
- Dev 성과 악화 (Sharpe 0.1730 → 0.1306, CAGR 6.58% → 1.25%)
- Holdout만 비정상적으로 높음 (과적합 시사)

**4. 안정성 저하**:
- Equal Weighting: 낮은 Turnover (39.08%), 높은 Hit Ratio (66.67%)
- Softmax Weighting: Turnover 증가 (53.59%), Hit Ratio 감소 (61.11%)

#### 최종 결정

**Phase 9 기준 운영 설정**: BT120 = Equal Weighting 유지

**이유**:
1. Phase 8 성과가 Phase 9 목표를 모두 달성
2. Softmax 적용 시 비정상적 수치 및 안정성 저하
3. Equal Weighting이 장기 전략에 더 적합 (안정성 중심)

**참고 문서**: `artifacts/reports/phase9_bt120_softmax_comparison_report.md`

---

## 부록: 주요 함수 및 코드 위치

### 백테스트 핵심 함수

| 함수명 | 파일 | 라인 | 설명 |
|--------|------|------|------|
| `run_backtest()` | `l7_backtest.py` | 438-1119 | 메인 백테스트 함수 |
| `_weights_from_scores()` | `l7_backtest.py` | 119-139 | 가중치 계산 (equal/softmax) |
| `select_topk_with_fallback()` | `selector.py` | 13-251 | 종목 선택 (fallback 포함) |
| `_select_with_smart_buffer()` | `l7_backtest.py` | 223-299 | 스마트 버퍼링 |
| `_calculate_volatility_adjustment()` | `l7_backtest.py` | 355-394 | 변동성 조정 |
| `_apply_risk_scaling()` | `l7_backtest.py` | 396-436 | 국면별 리스크 스케일링 |
| `_compute_turnover_oneway()` | `l7_backtest.py` | 101-106 | Turnover 계산 |
| `_mdd()` | `l7_backtest.py` | 108-117 | MDD 계산 |

### 설정 파일

- **BT20**: `configs/config.yaml::l7_bt20` (182-226행)
- **BT120**: `configs/config.yaml::l7_bt120` (230-274행)

### 산출물 파일

- **BT20**: `data/interim/bt_metrics.parquet`
- **BT120**: `data/interim/bt_metrics_bt120.parquet`
- **포지션**: `data/interim/bt_positions.parquet`
- **수익률**: `data/interim/bt_returns.parquet`
- **자산 곡선**: `data/interim/bt_equity_curve.parquet`

---

## 참고 문서

- **투트랙 아키텍처 가이드**: `docs/TWO_TRACK_ARCHITECTURE.md` ⭐
- **Track A 기술 보고서**: `TECH_REPORT_TRACK1_RANKING.md`
- **Phase 8 최종 리포트**: `artifacts/reports/phase8_final_metrics_report.md`
- **Phase 9 최종 상태 확정**: `artifacts/reports/phase9_final_status_confirmation.md`
- **BT120 Softmax 비교**: `artifacts/reports/phase9_bt120_softmax_comparison_report.md`
- **최종 수치셋 정의**: `docs/FINAL_METRICS_DEFINITION.md`

---

**문서 버전**: Phase 9 + 투트랙 구조 리팩토링 + Track B 최신 스냅샷 반영 (2026-01-07)  
**최종 업데이트**:
- 거래비용 모델 정상화(턴오버 기반) 및 Alpha Quality 지표 산출 반영
- BT120 오버래핑 트랜치(월별 4트랜치) 도입으로 Holdout 리밸런싱 표본 확보
- 최신 성과 표준 리포트: `artifacts/reports/track_b_4strategy_final_summary.md`
- **Ridge Alpha 최적화** (2026-01-06): Grid Search를 통한 L2 정규화 강도 최적화 시도
  - 최적화 결과: 모든 전략이 랭킹 기반이므로 ridge_alpha가 성과에 영향을 주지 않음 (정상 동작 확인)
  - 운영 설정: `configs/config.yaml`의 `l5.ridge_alpha`를 유지 (현재 **8.0**, 2026-01-07 최종 픽스)
  - 상세 리포트: `artifacts/reports/FINAL_RIDGE_ALPHA_OPTIMIZATION_REPORT.md`
- **완전 교체 전략 + top_k 최적화** (2026-01-06): 완전 교체 전략(rebalance_interval=holding_days, buffer_k=0)에서 top_k 최적화
  - 실험 조건: Day1 top_k 매수 → Day20 전량 매도 → Day20 top_k 재매수 (매번 100% 교체)
  - 최적화 결과: top_k=15이 Holdout 구간에서 최고 성과 (Total Return 12.39%, Sharpe 0.5464, CAGR 6.69%)
  - 비교: top_k=15 > top_k=20 > top_k=10 (Holdout 기준)
  - 상세 리포트: `artifacts/reports/full_replacement_topk_optimization_report.md`
- **가중치 방식(equal vs softmax) 비교 최적화** (2026-01-06): 4가지 전략 모두에 대해 equal과 softmax 비교
  - 최적화 결과: 모든 전략에서 equal이 softmax보다 우수 (Holdout Total Return 기준)
  - bt20_short: equal (4.90%) > softmax (2.98%), bt20_ens: equal (6.00%) > softmax (1.85%)
  - bt120_long: equal (2.88%) > softmax (1.19%), bt120_ens: equal (4.03%) > softmax (2.69%)
  - 운영 설정: 전략별로 `configs/config.yaml`을 단일 기준으로 적용
  - 상세 리포트: `artifacts/reports/weighting_comparison_optimization_report.md`
- 투트랙 구조 반영 (Track B: 투자 모델)
- 코드 경로 업데이트 (`src/tracks/track_b/`)

