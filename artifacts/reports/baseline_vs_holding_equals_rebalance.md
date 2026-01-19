# Baseline vs holding_days=rebalance_interval 비교 리포트

생성 시간: 2026-01-06 17:23:05

## 설정 변경
- **BT20 모델**: holding_days=20, rebalance_interval=20 (이전: holding_days=20, rebalance_interval=1)
- **BT120 모델**: holding_days=120, rebalance_interval=120 (이전: holding_days=120, rebalance_interval=6)

## 결과 비교 (Holdout 구간)

### bt20_ens

| 지표 | Baseline | holding_days=rebalance_interval | 변화율 |
|------|----------|--------------------------------|--------|
| Sharpe | 0.4643 | 0.4486 | -3.40% |
| CAGR | 8.0553% | 7.8335% | -2.75% |
| MDD | -14.4293% | -15.3313% | - |
| Calmar | 0.5583 | 0.5110 | -8.47% |

### bt20_short

| 지표 | Baseline | holding_days=rebalance_interval | 변화율 |
|------|----------|--------------------------------|--------|
| Sharpe | 0.5498 | 0.5498 | 0.00% |
| CAGR | 10.3802% | 10.3802% | -0.00% |
| MDD | -15.2742% | -15.2742% | - |
| Calmar | 0.6796 | 0.6796 | 0.00% |

### bt120_ens

| 지표 | Baseline | holding_days=rebalance_interval | 변화율 |
|------|----------|--------------------------------|--------|
| Sharpe | 0.5932 | 0.5932 | 0.00% |
| CAGR | 50.3884% | 50.3884% | 0.00% |
| MDD | -17.6453% | -17.6453% | - |
| Calmar | 2.8556 | 2.8556 | 0.00% |

### bt120_long

| 지표 | Baseline | holding_days=rebalance_interval | 변화율 |
|------|----------|--------------------------------|--------|
| Sharpe | 0.6833 | 0.6954 | 1.76% |
| CAGR | 55.5829% | 55.8008% | 0.39% |
| MDD | -19.2848% | -16.4825% | - |
| Calmar | 2.8822 | 3.3855 | 17.46% |
