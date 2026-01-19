# 백테스트 설정값 우선순위 문서

**작성일**: 2026-01-08  
**목적**: 백테스트 설정값 적용 우선순위 및 기본값 통일 정책 문서화

---

## 1. 설정값 적용 우선순위

백테스트 설정값은 다음 순서로 적용됩니다:

```
1순위: config.yaml의 전략별 설정값 (l7_bt20_short, l7_bt20_ens, l7_bt120_long, l7_bt120_ens)
2순위: BacktestConfig 클래스 기본값 (src/tracks/track_b/stages/backtest/l7_backtest.py)
```

### 적용 예시

```python
# track_b_pipeline.py에서
default_cfg = BacktestConfig()  # BacktestConfig 기본값 생성
bt_cfg = BacktestConfig(
    top_k=int(l7_cfg.get("top_k", default_cfg.top_k)),  # config.yaml 우선, 없으면 BacktestConfig 기본값
    buffer_k=int(l7_cfg.get("buffer_k", default_cfg.buffer_k)),
    ...
)
```

### 예시 1: config.yaml에 값이 있는 경우

```yaml
# config.yaml
l7_bt20_short:
  top_k: 12
```

**결과**: `top_k = 12` 사용 ✅ (config.yaml 값 우선 적용)

### 예시 2: config.yaml에 값이 없는 경우

```yaml
# config.yaml
l7_bt20_short:
  # top_k 명시 안 됨
```

**결과**: `top_k = 20` 사용 ✅ (BacktestConfig 기본값 사용)

---

## 2. BacktestConfig 기본값 정책

### 2.1. 통일 원칙

**BacktestConfig 기본값은 config.yaml의 전략별 공통값과 통일합니다.**

- **모든 전략에서 동일하게 사용하는 값**: BacktestConfig 기본값으로 설정
- **전략별로 다른 값**: config.yaml에 명시적으로 정의
- **목적**: config.yaml에 설정값이 누락되어도 실제 사용값과 유사한 기본값 적용

### 2.2. 현재 BacktestConfig 기본값

| 설정 | 기본값 | 변경 사유 |
|------|--------|----------|
| `holding_days` | 20 | 모든 전략 공통 |
| `top_k` | 20 | 기본값 (전략별로 다름: 12/15/20) |
| `cost_bps` | 10.0 | 모든 전략 공통 |
| `slippage_bps` | 0.0 | 모든 전략 공통 |
| `score_col` | "score_ens" | 기본값 (전략별로 다름) |
| `ret_col` | "true_short" | 모든 전략 공통 |
| `weighting` | "equal" | 기본값 (bt20_ens는 "softmax") |
| `softmax_temp` | 1.0 | 기본값 |
| `buffer_k` | **15** ⚠️ | **변경: 0 → 15** (3개 전략 공통) |
| `rebalance_interval` | **20** ⚠️ | **변경: 1 → 20** (모든 전략 공통) |
| `smart_buffer_enabled` | **True** ⚠️ | **변경: False → True** (모든 전략 공통) |
| `smart_buffer_stability_threshold` | 0.7 | 모든 전략 공통 |
| `volatility_adjustment_enabled` | **True** ⚠️ | **변경: False → True** (모든 전략 공통) |
| `volatility_lookback_days` | 60 | 모든 전략 공통 |
| `target_volatility` | 0.15 | 모든 전략 공통 |
| `volatility_adjustment_max` | 1.2 | 모든 전략 공통 |
| `volatility_adjustment_min` | 0.6 | 기본값 (bt20은 0.7) |
| `risk_scaling_enabled` | **True** ⚠️ | **변경: False → True** (모든 전략 공통) |
| `risk_scaling_bear_multiplier` | 0.7 | 기본값 (bt20은 0.8) |
| `risk_scaling_neutral_multiplier` | 0.9 | 기본값 (bt20은 1.0) |
| `risk_scaling_bull_multiplier` | 1.0 | 모든 전략 공통 |
| `overlapping_tranches_enabled` | False | 기본값 (bt120만 True) |
| `tranche_holding_days` | 120 | 모든 전략 공통 |
| `tranche_max_active` | 4 | 모든 전략 공통 |
| `tranche_allocation_mode` | "fixed_equal" | 모든 전략 공통 |
| `regime_enabled` | False | 기본값 (모든 전략은 True) |

⚠️ **변경된 기본값**: config.yaml의 실제 사용값과 통일하기 위해 변경되었습니다.

---

## 3. 전략별 config.yaml 설정값

### 3.1. bt20_short

**BacktestConfig 기본값과 다른 설정:**
- `top_k`: 12 (기본값: 20)
- `score_col`: "score_total_short" (기본값: "score_ens")
- `buffer_k`: 15 (기본값: 15) ✅ 통일됨
- `rebalance_interval`: 20 (기본값: 20) ✅ 통일됨
- `smart_buffer_enabled`: True (기본값: True) ✅ 통일됨
- `volatility_adjustment_enabled`: True (기본값: True) ✅ 통일됨
- `volatility_adjustment_min`: 0.7 (기본값: 0.6)
- `risk_scaling_enabled`: True (기본값: True) ✅ 통일됨
- `risk_scaling_bear_multiplier`: 0.8 (기본값: 0.7)
- `risk_scaling_neutral_multiplier`: 1.0 (기본값: 0.9)
- `regime_enabled`: True (기본값: False)

### 3.2. bt20_ens

**BacktestConfig 기본값과 다른 설정:**
- `top_k`: 15 (기본값: 20)
- `weighting`: "softmax" (기본값: "equal")
- `softmax_temp`: 0.5 (기본값: 1.0)
- `buffer_k`: 20 (기본값: 15)
- `rebalance_interval`: 20 (기본값: 20) ✅ 통일됨
- `smart_buffer_enabled`: True (기본값: True) ✅ 통일됨
- `volatility_adjustment_enabled`: True (기본값: True) ✅ 통일됨
- `volatility_adjustment_min`: 0.7 (기본값: 0.6)
- `risk_scaling_enabled`: True (기본값: True) ✅ 통일됨
- `risk_scaling_bear_multiplier`: 0.8 (기본값: 0.7)
- `risk_scaling_neutral_multiplier`: 1.0 (기본값: 0.9)
- `regime_enabled`: True (기본값: False)

### 3.3. bt120_long

**BacktestConfig 기본값과 다른 설정:**
- `top_k`: 15 (기본값: 20)
- `score_col`: "score_total_long" (기본값: "score_ens")
- `buffer_k`: 15 (기본값: 15) ✅ 통일됨
- `rebalance_interval`: 20 (기본값: 20) ✅ 통일됨
- `smart_buffer_enabled`: True (기본값: True) ✅ 통일됨
- `volatility_adjustment_enabled`: True (기본값: True) ✅ 통일됨
- `risk_scaling_enabled`: True (기본값: True) ✅ 통일됨
- `overlapping_tranches_enabled`: True (기본값: False)
- `regime_enabled`: True (기본값: False)

### 3.4. bt120_ens

**BacktestConfig 기본값과 다른 설정:**
- `buffer_k`: 15 (기본값: 15) ✅ 통일됨
- `rebalance_interval`: 20 (기본값: 20) ✅ 통일됨
- `smart_buffer_enabled`: True (기본값: True) ✅ 통일됨
- `volatility_adjustment_enabled`: True (기본값: True) ✅ 통일됨
- `risk_scaling_enabled`: True (기본값: True) ✅ 통일됨
- `overlapping_tranches_enabled`: True (기본값: False)
- `regime_enabled`: True (기본값: False)

---

## 4. 리팩토링 효과

### 4.1. 통일 전 (리팩토링 전)

- **BacktestConfig 기본값**: 보수적인 기본값 (예: `buffer_k=0`, `smart_buffer_enabled=False`)
- **config.yaml 실제 사용값**: 더 적극적인 설정 (예: `buffer_k=15`, `smart_buffer_enabled=True`)
- **문제**: config.yaml에 값이 누락되면 예상과 다른 동작 발생 가능

### 4.2. 통일 후 (리팩토링 후)

- **BacktestConfig 기본값**: config.yaml의 실제 사용값과 통일
- **효과**: config.yaml에 값이 누락되어도 실제 사용값과 유사한 기본값 적용
- **장점**: 예상치 못한 동작 방지, 일관성 확보

---

## 5. 권장 사항

### 5.1. config.yaml에 모든 설정값 명시

설정값을 명시적으로 정의하는 것을 권장합니다:
- 코드 가독성 향상
- 설정 의도 명확화
- 기본값 변경 영향 최소화

### 5.2. BacktestConfig 기본값 변경 시

BacktestConfig 기본값을 변경할 경우:
1. 모든 전략의 config.yaml과 일관성 확인
2. 기본값 변경 영향 분석
3. 문서 업데이트 (본 문서)
4. 백테스트 재실행 및 결과 확인

---

## 6. 참고 파일

- **BacktestConfig 클래스**: `src/tracks/track_b/stages/backtest/l7_backtest.py`
- **파이프라인**: `src/pipeline/track_b_pipeline.py`
- **config.yaml**: `configs/config.yaml`
- **비교 스크립트**: `scripts/analyze_common_configs.py`
- **설정값 표시 스크립트**: `scripts/show_all_backtest_configs.py`

---

## 7. 변경 이력

**2026-01-08**: 
- BacktestConfig 기본값을 config.yaml과 통일
- 변경된 기본값:
  - `buffer_k`: 0 → 15
  - `rebalance_interval`: 1 → 20
  - `smart_buffer_enabled`: False → True
  - `volatility_adjustment_enabled`: False → True
  - `risk_scaling_enabled`: False → True

