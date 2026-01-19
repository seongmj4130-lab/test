# config.yaml vs BacktestConfig 기본값 비교 리포트

**분석 일시**: 2026-01-08
**분석 대상**: 03_code
**분석 목적**: config.yaml 설정값과 BacktestConfig 기본값의 차이 확인

## 요약

**결론**: **14개 설정값이 config.yaml에서 BacktestConfig 기본값과 다르게 설정**되어 있습니다.

이는 **정상적인 동작**입니다:
- **BacktestConfig 기본값**: 설정값이 없을 때 사용되는 기본값
- **config.yaml 설정값**: 실제로 적용되는 값 (우선순위가 높음)
- **리팩토링 효과**: config.yaml에 명시된 값이 있으면 그것을 사용하고, 없으면 BacktestConfig 기본값을 사용

## BacktestConfig 기본값

| 설정 | 기본값 |
|------|--------|
| `holding_days` | 20 |
| `top_k` | **20** |
| `cost_bps` | 10.0 |
| `slippage_bps` | 0.0 |
| `score_col` | **score_ens** |
| `ret_col` | true_short |
| `weighting` | **equal** |
| `softmax_temp` | **1.0** |
| `buffer_k` | **0** |
| `rebalance_interval` | **1** |
| `smart_buffer_enabled` | **False** |
| `volatility_adjustment_enabled` | **False** |
| `volatility_adjustment_min` | **0.6** |
| `risk_scaling_enabled` | **False** |
| `risk_scaling_bear_multiplier` | **0.7** |
| `risk_scaling_neutral_multiplier` | **0.9** |
| `overlapping_tranches_enabled` | **False** |
| `regime_enabled` | **False** |

## 전략별 config.yaml vs BacktestConfig 기본값 차이

### 1. bt20_short

**11개 설정값이 다름:**

| 설정 | BacktestConfig 기본값 | config.yaml | 차이 |
|------|---------------------|-------------|------|
| `top_k` | 20 | **12** | -8 |
| `score_col` | score_ens | **score_total_short** | 변경됨 |
| `buffer_k` | 0 | **15** | +15 |
| `rebalance_interval` | 1 | **20** | +19 |
| `smart_buffer_enabled` | False | **True** | 변경됨 |
| `volatility_adjustment_enabled` | False | **True** | 변경됨 |
| `volatility_adjustment_min` | 0.6 | **0.7** | +0.1 |
| `risk_scaling_enabled` | False | **True** | 변경됨 |
| `risk_scaling_bear_multiplier` | 0.7 | **0.8** | +0.1 |
| `risk_scaling_neutral_multiplier` | 0.9 | **1.0** | +0.1 |
| `regime_enabled` | False | **True** | 변경됨 |

### 2. bt20_ens

**12개 설정값이 다름:**

| 설정 | BacktestConfig 기본값 | config.yaml | 차이 |
|------|---------------------|-------------|------|
| `top_k` | 20 | **15** | -5 |
| `weighting` | equal | **softmax** | 변경됨 |
| `softmax_temp` | 1.0 | **0.5** | -0.5 |
| `buffer_k` | 0 | **20** | +20 |
| `rebalance_interval` | 1 | **20** | +19 |
| `smart_buffer_enabled` | False | **True** | 변경됨 |
| `volatility_adjustment_enabled` | False | **True** | 변경됨 |
| `volatility_adjustment_min` | 0.6 | **0.7** | +0.1 |
| `risk_scaling_enabled` | False | **True** | 변경됨 |
| `risk_scaling_bear_multiplier` | 0.7 | **0.8** | +0.1 |
| `risk_scaling_neutral_multiplier` | 0.9 | **1.0** | +0.1 |
| `regime_enabled` | False | **True** | 변경됨 |

### 3. bt120_long

**9개 설정값이 다름:**

| 설정 | BacktestConfig 기본값 | config.yaml | 차이 |
|------|---------------------|-------------|------|
| `top_k` | 20 | **15** | -5 |
| `score_col` | score_ens | **score_total_long** | 변경됨 |
| `buffer_k` | 0 | **15** | +15 |
| `rebalance_interval` | 1 | **20** | +19 |
| `smart_buffer_enabled` | False | **True** | 변경됨 |
| `volatility_adjustment_enabled` | False | **True** | 변경됨 |
| `risk_scaling_enabled` | False | **True** | 변경됨 |
| `overlapping_tranches_enabled` | False | **True** | 변경됨 |
| `regime_enabled` | False | **True** | 변경됨 |

### 4. bt120_ens

**7개 설정값이 다름:**

| 설정 | BacktestConfig 기본값 | config.yaml | 차이 |
|------|---------------------|-------------|------|
| `buffer_k` | 0 | **15** | +15 |
| `rebalance_interval` | 1 | **20** | +19 |
| `smart_buffer_enabled` | False | **True** | 변경됨 |
| `volatility_adjustment_enabled` | False | **True** | 변경됨 |
| `risk_scaling_enabled` | False | **True** | 변경됨 |
| `overlapping_tranches_enabled` | False | **True** | 변경됨 |
| `regime_enabled` | False | **True** | 변경됨 |

## 전체 통계

### 차이가 나는 설정값 (총 14개)

1. **`top_k`**: BacktestConfig 20 → config.yaml 12/15
2. **`score_col`**: BacktestConfig score_ens → config.yaml score_total_short/long
3. **`buffer_k`**: BacktestConfig 0 → config.yaml 15/20
4. **`rebalance_interval`**: BacktestConfig 1 → config.yaml 20
5. **`smart_buffer_enabled`**: BacktestConfig False → config.yaml True
6. **`volatility_adjustment_enabled`**: BacktestConfig False → config.yaml True
7. **`volatility_adjustment_min`**: BacktestConfig 0.6 → config.yaml 0.7
8. **`risk_scaling_enabled`**: BacktestConfig False → config.yaml True
9. **`risk_scaling_bear_multiplier`**: BacktestConfig 0.7 → config.yaml 0.8
10. **`risk_scaling_neutral_multiplier`**: BacktestConfig 0.9 → config.yaml 1.0
11. **`regime_enabled`**: BacktestConfig False → config.yaml True
12. **`weighting`**: BacktestConfig equal → config.yaml softmax (bt20_ens만)
13. **`softmax_temp`**: BacktestConfig 1.0 → config.yaml 0.5 (bt20_ens만)
14. **`overlapping_tranches_enabled`**: BacktestConfig False → config.yaml True (bt120만)

## 리팩토링 후 동작 방식

### 적용 우선순위

```
config.yaml 설정값 (있으면) → BacktestConfig 기본값 (없으면)
```

### 예시: `top_k`

1. **bt20_short**:
   - config.yaml: `top_k: 12` → **12 사용** ✅
   - BacktestConfig 기본값: 20 → **사용 안 됨**

2. **bt120_ens**:
   - config.yaml: `top_k` 없음 → BacktestConfig 기본값 **20 사용** ✅
   - (실제로는 config.yaml에 명시되어 있을 수도 있음)

### 예시: `smart_buffer_enabled`

1. **모든 전략**:
   - config.yaml: `smart_buffer_enabled: true` → **True 사용** ✅
   - BacktestConfig 기본값: False → **사용 안 됨**

## 결론

### 1. 두 가지가 다른 설정값을 가지고 있나요?

**네, 14개 설정값이 다릅니다.**

- **BacktestConfig 기본값**: 설정값이 없을 때 사용되는 기본값 (더 보수적인 기본값)
- **config.yaml 설정값**: 실제로 적용되는 값 (더 적극적인 설정)

### 2. 이것이 정상적인가요?

**네, 정상입니다.**

- config.yaml에 명시된 값은 **우선순위가 높아서** 실제로 적용됩니다
- BacktestConfig 기본값은 config.yaml에 값이 없을 때만 사용됩니다
- 리팩토링 후에도 **동일한 결과**가 나옵니다 (config.yaml 우선 적용 구조)

### 3. 리팩토링의 효과는?

✅ **일관성 확보**: 모든 기본값이 BacktestConfig에서 단일 소스로 관리됨
✅ **명확한 우선순위**: config.yaml → BacktestConfig 기본값 순서 명확
✅ **유지보수성 향상**: 기본값 변경 시 한 곳만 수정하면 됨

### 4. 주의사항

⚠️ **config.yaml에 모든 설정값 명시 권장**

리팩토링 후에도 동일한 결과가 나오지만, 명시적으로 모든 설정값을 config.yaml에 정의하는 것이 권장됩니다:
- 코드 가독성 향상
- 설정 의도 명확화
- 기본값 변경 영향 최소화

## 참고 파일

- 비교 스크립트: `scripts/compare_config_vs_defaults.py`
- BacktestConfig 클래스: `src/tracks/track_b/stages/backtest/l7_backtest.py`
- config.yaml: `configs/config.yaml`
