# 백테스트 결과값에 해당하는 설정값 매핑

**작성일**: 2026-01-07
**결과 기준**: rebalance_interval=20 적용 후 최신 실행 결과

## ⚠️ 결과값 출처 및 시점

**문서에 기록된 기대값**은 `run_backtest_4models.py` 실행 시 **콘솔에 출력된 값**입니다.

### 실행 시점 및 설정

- **실행 시간**: 2026-01-07 13:21:00 (CSV 파일 수정 시간 기준)
- **설정 상태**: `rebalance_interval=20` 수정 후 실행
- **결과 파일**: `artifacts/reports/backtest_4models_comparison.csv`

### 결과값 확인 방법

1. **실행 로그 확인**: `run_backtest_4models.py` 실행 시 콘솔 출력값
2. **CSV 파일 확인**: `artifacts/reports/backtest_4models_comparison.csv` (최신 실행 결과 저장)
   - 수정 시간: 2026-01-07 13:21:00
   - 문서의 기대값과 일치
3. **Parquet 파일**: `data/interim/bt_metrics_{strategy}.parquet` (이전 실행 결과일 수 있음)
   - ⚠️ 주의: 저장된 Parquet 파일은 이전 실행 결과일 수 있으므로, 문서의 기대값과 다를 수 있습니다.

### 중요 사항

- **저장된 Parquet 파일**은 이전 실행 결과일 수 있으므로, 문서의 기대값과 다를 수 있습니다.
- **실제 최신 결과**는 `run_backtest_4models.py` 실행 시 콘솔 출력값 또는 `backtest_4models_comparison.csv` 파일을 확인하세요.
- 문서의 기대값은 **rebalance_interval=20 수정 후** (2026-01-07 13:21:00) 실행된 결과입니다.
- **설정 변경 이력**:
  - `rebalance_interval`을 1에서 20으로 수정
  - BT120 전략에 오버래핑 트랜치 설정 적용 (`tranche_max_active: 4`)
  - `return_col`을 `true_long`에서 `true_short`로 변경 (BT120 전략)

---

## 최종 결과 (Holdout 구간)

| 모델 | Net Sharpe | Net CAGR | Net MDD | Net Calmar Ratio |
|------|------------|----------|---------|------------------|
| **bt20_ens** | **0.6826** | **14.98%** | **-10.98%** | **1.3641** |
| **bt20_short** | **0.6464** | **13.84%** | **-9.09%** | **1.5223** |
| **bt120_ens** | **0.6263** | **11.66%** | **-7.69%** | **1.5156** |
| **bt120_long** | **0.6839** | **13.60%** | **-8.66%** | **1.5700** |

---

## 각 모델별 설정값 (config.yaml)

### 1. bt20_ens (Sharpe: 0.6826, CAGR: 14.98%)

**config.yaml 섹션**: `l7_bt20_ens`

```yaml
l7_bt20_ens:
  holding_days: 20
  top_k: 15
  cost_bps: 10.0
  slippage_bps: 0.0
  buffer_k: 20
  weighting: softmax
  softmax_temperature: 0.5
  score_col: score_ens  # 단기:장기 5:5 결합
  return_col: true_short
  rebalance_interval: 20  # [중요] holding_days와 동일
  smart_buffer_enabled: true
  smart_buffer_stability_threshold: 0.7
  volatility_adjustment_enabled: true
  volatility_lookback_days: 60
  target_volatility: 0.15
  volatility_adjustment_max: 1.2
  volatility_adjustment_min: 0.7
  risk_scaling_enabled: true
  risk_scaling_bear_multiplier: 0.8
  risk_scaling_neutral_multiplier: 1.0
  risk_scaling_bull_multiplier: 1.0
  regime:
    enabled: true
    lookback_days: 60
    threshold_pct: 0.0
    neutral_band: 0.0
    top_k_bull_strong: 10
    top_k_bull_weak: 12
    top_k_bear_strong: 20
    top_k_bear_weak: 20
    top_k_neutral: 20
    exposure_bull_strong: 1.5
    exposure_bull_weak: 1.2
    exposure_bear_strong: 0.6
    exposure_bear_weak: 0.8
    exposure_neutral: 1.0
```

**의존 설정**:
- `l6.weight_short: 0.5`, `l6.weight_long: 0.5` (score_ens 생성용)
- `l5.ridge_alpha: 8.0` (모델 학습)
- `l5.min_feature_ic: -0.1` (피처 필터링)
- `l8_short.normalization_method: zscore` (랭킹 정규화)
- `l8_long.normalization_method: zscore` (랭킹 정규화)

---

### 2. bt20_short (Sharpe: 0.6464, CAGR: 13.84%)

**config.yaml 섹션**: `l7_bt20_short`

```yaml
l7_bt20_short:
  holding_days: 20
  top_k: 12
  cost_bps: 10.0
  slippage_bps: 0.0
  buffer_k: 15
  weighting: equal
  score_col: score_total_short  # 단기 랭킹만 사용
  return_col: true_short
  rebalance_interval: 20  # [중요] holding_days와 동일
  smart_buffer_enabled: true
  smart_buffer_stability_threshold: 0.7
  volatility_adjustment_enabled: true
  volatility_lookback_days: 60
  target_volatility: 0.15
  volatility_adjustment_max: 1.2
  volatility_adjustment_min: 0.7
  risk_scaling_enabled: true
  risk_scaling_bear_multiplier: 0.8
  risk_scaling_neutral_multiplier: 1.0
  risk_scaling_bull_multiplier: 1.0
  regime:
    enabled: true
    lookback_days: 60
    threshold_pct: 0.0
    neutral_band: 0.0
    top_k_bull_strong: 10
    top_k_bull_weak: 12
    top_k_bear_strong: 20
    top_k_bear_weak: 20
    top_k_neutral: 20
    exposure_bull_strong: 1.5
    exposure_bull_weak: 1.2
    exposure_bear_strong: 0.6
    exposure_bear_weak: 0.8
    exposure_neutral: 1.0
```

**의존 설정**:
- `l5.ridge_alpha: 8.0` (모델 학습)
- `l5.min_feature_ic: -0.1` (피처 필터링)
- `l8_short.normalization_method: zscore` (랭킹 정규화)

---

### 3. bt120_ens (Sharpe: 0.6263, CAGR: 11.66%)

**config.yaml 섹션**: `l7_bt120_ens`

```yaml
l7_bt120_ens:
  holding_days: 20  # [오버래핑 트랜치] 월별(20일) 기간수익률로 평가
  top_k: 20
  cost_bps: 10.0
  slippage_bps: 0.0
  buffer_k: 15
  weighting: equal
  score_col: score_ens  # 단기:장기 5:5 결합
  return_col: true_short  # [오버래핑 트랜치] 월별 PnL(20일 fwd)로 계산
  rebalance_interval: 20  # [중요] 월별 리밸런싱(신규 트랜치 추가)
  overlapping_tranches_enabled: true  # [필수] 오버래핑 트랜치 모드
  tranche_holding_days: 120  # 각 트랜치 보유 기간(캘린더 day)
  tranche_max_active: 4  # 월별 4트랜치(동시 보유 최대 4개)
  tranche_allocation_mode: fixed_equal  # 각 트랜치에 1/4 자본 고정 배분
  smart_buffer_enabled: true
  smart_buffer_stability_threshold: 0.7
  volatility_adjustment_enabled: true
  volatility_lookback_days: 60
  target_volatility: 0.15
  volatility_adjustment_max: 1.2
  volatility_adjustment_min: 0.6
  risk_scaling_enabled: true
  risk_scaling_bear_multiplier: 0.7
  risk_scaling_neutral_multiplier: 0.9
  risk_scaling_bull_multiplier: 1.0
  regime:
    enabled: true
    lookback_days: 60
    threshold_pct: 0.0
    neutral_band: 0.05
    top_k_bull_strong: 12
    top_k_bull_weak: 15
    top_k_bear_strong: 30
    top_k_bear_weak: 30
    top_k_neutral: 20
    exposure_bull_strong: 1.3
    exposure_bull_weak: 1.0
    exposure_bear_strong: 0.7
    exposure_bear_weak: 0.9
    exposure_neutral: 1.0
```

**의존 설정**:
- `l6.weight_short: 0.5`, `l6.weight_long: 0.5` (score_ens 생성용)
- `l5.ridge_alpha: 8.0` (모델 학습)
- `l5.min_feature_ic: -0.1` (피처 필터링)
- `l8_short.normalization_method: zscore` (랭킹 정규화)
- `l8_long.normalization_method: zscore` (랭킹 정규화)

---

### 4. bt120_long (Sharpe: 0.6839, CAGR: 13.60%)

**config.yaml 섹션**: `l7_bt120_long`

```yaml
l7_bt120_long:
  holding_days: 20  # [오버래핑 트랜치] 월별(20일) 기간수익률로 평가
  top_k: 15
  cost_bps: 10.0
  slippage_bps: 0.0
  buffer_k: 15
  weighting: equal
  score_col: score_total_long  # 장기 랭킹만 사용
  return_col: true_short  # [오버래핑 트랜치] 월별 PnL(20일 fwd)로 계산
  rebalance_interval: 20  # [중요] 월별 리밸런싱(신규 트랜치 추가)
  overlapping_tranches_enabled: true  # [필수] 오버래핑 트랜치 모드
  tranche_holding_days: 120  # 각 트랜치 보유 기간(캘린더 day)
  tranche_max_active: 4  # 월별 4트랜치(동시 보유 최대 4개)
  tranche_allocation_mode: fixed_equal  # 각 트랜치에 1/4 자본 고정 배분
  smart_buffer_enabled: true
  smart_buffer_stability_threshold: 0.7
  volatility_adjustment_enabled: true
  volatility_lookback_days: 60
  target_volatility: 0.15
  volatility_adjustment_max: 1.2
  volatility_adjustment_min: 0.6
  risk_scaling_enabled: true
  risk_scaling_bear_multiplier: 0.7
  risk_scaling_neutral_multiplier: 0.9
  risk_scaling_bull_multiplier: 1.0
  regime:
    enabled: true
    lookback_days: 60
    threshold_pct: 0.0
    neutral_band: 0.05
    top_k_bull_strong: 12
    top_k_bull_weak: 15
    top_k_bear_strong: 30
    top_k_bear_weak: 30
    top_k_neutral: 20
    exposure_bull_strong: 1.3
    exposure_bull_weak: 1.0
    exposure_bear_strong: 0.7
    exposure_bear_weak: 0.9
    exposure_neutral: 1.0
```

**의존 설정**:
- `l5.ridge_alpha: 8.0` (모델 학습)
- `l5.min_feature_ic: -0.1` (피처 필터링)
- `l8_long.normalization_method: zscore` (랭킹 정규화)

---

## 공통 의존 설정 (모든 전략)

### L4: Walk-Forward CV 분할
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
```

### L5: 모델 학습
```yaml
l5:
  model_type: ridge
  target_transform: cs_rank
  ridge_alpha: 8.0  # [최종 픽스 2026-01-07]
  min_feature_ic: -0.1  # [최종 픽스 2026-01-07]
  filter_features_by_ic: true
  use_rank_ic: true
  feature_list_short: configs/features_short_v1.yaml  # 22개 피처
  feature_list_long: configs/features_long_v1.yaml    # 19개 피처
  feature_weights_config_short: configs/feature_weights_short_hitratio_optimized.yaml
  feature_weights_config_long: configs/feature_weights_long_ic_optimized.yaml
```

### L6: 스코어 결합
```yaml
l6:
  weight_short: 0.5
  weight_long: 0.5
  invert_score_sign: false
```

### L8: 랭킹 엔진
```yaml
l8_short:
  normalization_method: zscore  # [최종 픽스 2026-01-07]
  feature_groups_config: configs/feature_groups_short.yaml
  feature_weights_config: configs/feature_weights_short_hitratio_optimized.yaml
  use_sector_relative: true
  sector_col: sector_name

l8_long:
  normalization_method: zscore  # [최종 픽스 2026-01-07]
  feature_groups_config: configs/feature_groups_long.yaml
  feature_weights_config: configs/feature_weights_long_ic_optimized.yaml
  use_sector_relative: true
  sector_col: sector_name
```

---

## 핵심 설정값 요약

### 필수 설정 (모든 전략)
- `rebalance_interval: 20` ⚠️ **중요**: holding_days와 동일해야 함
- `cost_bps: 10.0`
- `slippage_bps: 0.0`

### 전략별 차이점

| 설정 | bt20_ens | bt20_short | bt120_ens | bt120_long |
|------|----------|-----------|-----------|------------|
| `top_k` | 15 | 12 | 20 | 15 |
| `buffer_k` | 20 | 15 | 15 | 15 |
| `weighting` | softmax | equal | equal | equal |
| `softmax_temperature` | 0.5 | - | - | - |
| `score_col` | score_ens | score_total_short | score_ens | score_total_long |
| `overlapping_tranches_enabled` | false | false | **true** | **true** |
| `tranche_max_active` | - | - | **4** | **4** |

---

## 설정값 변경 시 주의사항

1. **rebalance_interval**: 반드시 `holding_days`와 동일하게 설정
   - `rebalance_interval=1`이면 전략 본질이 변질됨

2. **BT120 오버래핑 트랜치**: 필수 설정
   - `overlapping_tranches_enabled: true`
   - `tranche_holding_days: 120`
   - `tranche_max_active: 4`
   - `return_col: true_short` (월별 PnL 계산)

3. **의존성**: L4, L5, L6, L8 설정이 모두 결과에 영향을 줌
   - L5 모델 학습 결과 → L6 스코어 생성 → L7 백테스트
   - L8 랭킹 생성 → L6R 스코어 변환 → L7 백테스트

---

## 설정 파일 위치

- **03_code**: `configs/config.yaml`
- **06_code22**: `configs/config.yaml` (동일한 설정 적용됨)
