# 가중치 방식(equal vs softmax) 비교 최적화 결과

생성 시간: 2026-01-06 21:23:54

## 실험 조건

- **전략**: bt20_short, bt20_ens, bt120_long, bt120_ens (4가지)
- **가중치 방식**: equal, softmax (2가지)
- **softmax_temperature**: 0.5
- **비교 기준**: Holdout 구간 Total Return

## 최적화 결과

### 전략별 최적 가중치 방식

| 전략 | 최적 Weighting | Holdout Total Return | Holdout Sharpe | Holdout CAGR |
|------|---------------|---------------------|----------------|--------------|
| bt20_short | **equal** | 0.0490 | 0.2607 | 2.6866% |
| bt20_ens | **equal** | 0.0600 | 0.3155 | 3.2841% |
| bt120_long | **equal** | 0.0288 | 0.1769 | 2.9224% |
| bt120_ens | **equal** | 0.0403 | 0.2166 | 4.0862% |

## 전체 결과 비교

### bt20_short

| Weighting | Net Sharpe (Holdout) | Net CAGR (Holdout) | Net Total Return (Holdout) | Net Sharpe (Dev) | Net CAGR (Dev) | Net Total Return (Dev) |
|-----------|---------------------|-------------------|---------------------------|----------------|----------------|----------------------|
| equal | 0.2607 | 2.6866% | 0.0490 | 0.2417 | 2.8951% | 0.2181 |
| softmax | 0.1833 | 1.6433% | 0.0298 | 0.2196 | 2.4317% | 0.1807 |

### bt20_ens

| Weighting | Net Sharpe (Holdout) | Net CAGR (Holdout) | Net Total Return (Holdout) | Net Sharpe (Dev) | Net CAGR (Dev) | Net Total Return (Dev) |
|-----------|---------------------|-------------------|---------------------------|----------------|----------------|----------------------|
| equal | 0.3155 | 3.2841% | 0.0600 | 0.2596 | 3.2272% | 0.2455 |
| softmax | 0.1403 | 1.0231% | 0.0185 | 0.2305 | 2.6354% | 0.1970 |

### bt120_long

| Weighting | Net Sharpe (Holdout) | Net CAGR (Holdout) | Net Total Return (Holdout) | Net Sharpe (Dev) | Net CAGR (Dev) | Net Total Return (Dev) |
|-----------|---------------------|-------------------|---------------------------|----------------|----------------|----------------------|
| softmax | 0.1061 | 1.2114% | 0.0119 | 0.2920 | 3.1474% | 0.2172 |
| equal | 0.1769 | 2.9224% | 0.0288 | 0.3207 | 3.4475% | 0.2399 |

### bt120_ens

| Weighting | Net Sharpe (Holdout) | Net CAGR (Holdout) | Net Total Return (Holdout) | Net Sharpe (Dev) | Net CAGR (Dev) | Net Total Return (Dev) |
|-----------|---------------------|-------------------|---------------------------|----------------|----------------|----------------------|
| equal | 0.2166 | 4.0862% | 0.0403 | 0.2843 | 3.1810% | 0.2197 |
| softmax | 0.1652 | 2.7312% | 0.0269 | 0.2605 | 2.8558% | 0.1956 |

## 결론

### 주요 발견사항

- **bt20_short**: equal가 더 우수 (차이: -0.0192, -39.09%)
- **bt20_ens**: equal가 더 우수 (차이: -0.0415, -69.13%)
- **bt120_long**: equal가 더 우수 (차이: -0.0169, -58.54%)
- **bt120_ens**: equal가 더 우수 (차이: -0.0133, -33.15%)

### 권장사항

- 각 전략별로 최적 가중치 방식을 선택하여 config.yaml에 반영 권장
- Holdout 구간 성과를 기준으로 최적화 완료
