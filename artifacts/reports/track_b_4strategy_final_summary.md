# Track B 4전략 최종 요약표

**근거 파일**: `data/interim/bt_metrics_{strategy}.csv`, `data/interim/bt_regime_metrics_{strategy}.parquet`

> 주의: 모든 전략이 충분한 Rebalances를 확보했습니다.

## 1) 핵심 성과 (Headline Metrics)

| 전략 | 구간 | Net Sharpe | Net Total Return | Net CAGR | Net MDD | Calmar |
|---|---|---|---|---|---|---|
| bt20_short | Dev | 0.498 | 13.084 | 0.502 | -0.253 | 1.983 |
| bt20_short | Holdout | 0.914 | 0.254 | 0.134 | -0.044 | 3.057 |
| bt20_ens | Dev | 0.491 | 8.478 | 0.413 | -0.298 | 1.389 |
| bt20_ens | Holdout | 0.751 | 0.194 | 0.104 | -0.067 | 1.542 |
| bt120_long | Dev | 0.558 | 2.253 | 0.199 | -0.288 | 0.690 |
| bt120_long | Holdout | 0.695 | 0.161 | 0.087 | -0.052 | 1.680 |
| bt120_ens | Dev | 0.557 | 1.846 | 0.175 | -0.294 | 0.595 |
| bt120_ens | Holdout | 0.594 | 0.129 | 0.070 | -0.054 | 1.300 |


## 2) 모델 예측력 (Alpha Quality)

| 전략 | 구간 | IC | Rank IC | ICIR | Rank ICIR | L/S Alpha (ann) |
|---|---|---|---|---|---|---|
| bt20_short | Dev | 0.026 | 0.037 | 0.897 | 1.022 | 1.087 |
| bt20_short | Holdout | 0.021 | 0.059 | 0.373 | 1.076 | 0.080 |
| bt20_ens | Dev | 0.026 | 0.037 | 0.897 | 1.022 | 0.844 |
| bt20_ens | Holdout | 0.021 | 0.059 | 0.373 | 1.076 | 0.044 |
| bt120_long | Dev | 0.026 | 0.037 | 0.897 | 1.022 | 0.844 |
| bt120_long | Holdout | 0.021 | 0.059 | 0.373 | 1.076 | 0.044 |
| bt120_ens | Dev | 0.026 | 0.037 | 0.897 | 1.022 | 0.671 |
| bt120_ens | Holdout | 0.021 | 0.059 | 0.373 | 1.076 | 0.125 |


## 3) 운용 안정성 (Operational Viability)

| 전략 | 구간 | Avg Turnover | Hit Ratio | Profit Factor | Avg Trade Duration | Avg Cost(%) | Rebalances |
|---|---|---|---|---|---|---|---|
| bt20_short | Dev | 0.392 | 0.519 | 6.773 | 29.7 | 0.024 | 81 |
| bt20_short | Holdout | 0.391 | 0.522 | 2.174 | 30.0 | 0.027 | 23 |
| bt20_ens | Dev | 0.388 | 0.543 | 5.316 | 29.7 | 0.024 | 81 |
| bt20_ens | Holdout | 0.361 | 0.609 | 1.883 | 30.0 | 0.024 | 23 |
| bt120_long | Dev | 0.172 | 0.531 | 2.642 | 29.7 | 0.010 | 81 |
| bt120_long | Holdout | 0.163 | 0.609 | 1.726 | 29.9 | 0.011 | 23 |
| bt120_ens | Dev | 0.172 | 0.543 | 2.441 | 29.7 | 0.010 | 81 |
| bt120_ens | Holdout | 0.161 | 0.522 | 1.582 | 29.9 | 0.011 | 23 |


## 4) 국면별 성과 (Regime Robustness)

### bt20_short
_(bt_regime_metrics 아티팩트 없음)_

### bt20_ens
_(bt_regime_metrics 아티팩트 없음)_

### bt120_long
_(bt_regime_metrics 아티팩트 없음)_

### bt120_ens
_(bt_regime_metrics 아티팩트 없음)_

_(regime 데이터가 없어 국면별 표를 생성하지 못했습니다.)_
