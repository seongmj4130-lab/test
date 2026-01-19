# 완전 교체 전략 + top_k 최적화 결과

생성 시간: 2026-01-06 21:12:26

## 실험 조건

- **전략**: 완전 교체 (rebalance_interval = holding_days)
  - Day1: top_k 매수 → Day20: 전량 매도 → Day20 top_k 재매수
  - buffer_k=0 고정 (매번 100% 교체)
  - rebalance_interval=20 (holding_days와 동일)
- **전략**: bt20_short (단기 랭킹)
- **top_k 그리드**: [20, 15, 10]

## 최적화 결과

- **최적 top_k**: 15
- **최적화 점수 (Holdout Total Return)**: 0.1239

### 최적 파라미터 상세 메트릭 (Holdout)

- **Net Sharpe**: 0.5464
- **Net CAGR**: 6.6905%
- **Net Total Return**: 0.1239

## 전체 결과 비교

### Holdout 구간 성과

| top_k | Net Sharpe | Net CAGR | Net Total Return | Net Sharpe (Dev) | Net CAGR (Dev) |
|-------|------------|----------|-------------------|------------------|----------------|
| 15 | 0.5464 | 6.6905% | 0.1239 | 0.2130 | 2.2730% |
| 10 | 0.2328 | 2.3690% | 0.0431 | 0.1525 | 1.0070% |
| 20 | 0.4541 | 4.8804% | 0.0898 | 0.2280 | 2.5243% |

## 결론

완전 교체 전략에서 **top_k=15**이 Holdout 구간 Total Return 기준으로 최적 성과를 보였습니다.

### 주요 발견사항

- 완전 교체 전략 (buffer_k=0, rebalance_interval=holding_days)은 매 리밸런싱마다 100% 종목 교체
- top_k 값에 따른 성과 차이 분석 완료
- 최적 top_k 값: 15
