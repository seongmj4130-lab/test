# 최종 수치셋 정의 (FINAL METRICS DEFINITION)

이 문서는 모든 Phase 보고서/요약/백테스트 결과가 반드시 포함해야 하는 **최종 수치셋**의 정의입니다.

> [개선안 42번] 문서화: “무엇을 어떻게 계산한 값인지”를 고정하여 보고서/리그레션 비교의 기준으로 사용합니다.

---

## 1) 핵심 성과 (Headline Metrics)

- **Net Sharpe Ratio**
  - 정의: 비용 차감 후 일별 수익률 \(r_{net}\)의 연환산 샤프
  - 계산: \(\text{Sharpe} = \frac{\mu(r_{net})}{\sigma(r_{net})} \times \sqrt{252}\)
- **Net Total Return**
  - 정의: 비용 차감 후 누적 수익률
  - 계산: \(\prod_t (1 + r_{net,t}) - 1\)
- **Net CAGR**
  - 정의: 비용 차감 후 연복리 수익률
  - 계산: \((1+\text{Net Total Return})^{1/\text{years}} - 1\)
- **Net MDD**
  - 정의: 비용 차감 후 누적 자산곡선의 최대 낙폭
- **Calmar Ratio**
  - 정의: \(\text{Calmar} = \frac{\text{Net CAGR}}{|\text{Net MDD}|}\)

---

## 2) 모델 예측력 (Alpha Quality)

- **IC (Information Coefficient)**
  - 정의: 날짜별 단면(cross-sectional)에서 score와 미래수익률의 Pearson 상관을 계산 후 평균
- **Rank IC**
  - 정의: 날짜별 단면에서 rank(score)와 rank(ret)의 Pearson 상관(= Spearman과 동등) 후 평균
- **ICIR**
  - 정의: 날짜별 IC 시계열의 평균/표준편차 비율
  - 계산: \(\text{ICIR} = \frac{\mu(IC)}{\sigma(IC)}\)
- **Long/Short Alpha**
  - 정의: 날짜별로 상위 \(k\)개 수익률 평균 − 하위 \(k\)개 수익률 평균 (단면 기준), 이후 평균

---

## 3) 운용 안정성 (Operational Viability)

- **Avg Turnover**
  - 정의: 리밸런싱 시점의 one-way turnover 평균
  - 참고: one-way turnover = \(0.5 \times \sum_i |w_{i,t} - w_{i,t-1}|\)
- **Hit Ratio**
  - 정의: 비용 차감 후 일별 수익률이 양(+)인 비율
- **Profit Factor**
  - 정의: (양(+) 수익률 합) / (음(-) 수익률 합의 절댓값)
- **Avg Trade Duration**
  - 정의: 평균 보유 기간(전략/구현에 따라 추정치로 계산 가능)

---

## 4) 국면별 성과 (Regime Robustness)

시장 국면을 bull/bear/neutral로 분류하여, 각 국면별로 아래 4개 지표를 Dev/Holdout 구간에서 각각 산출합니다.

- **Net Sharpe Ratio**
- **Net Total Return**
- **Net CAGR**
- **Net MDD**

---

## 보고 형식(필수)

- **Dev / Holdout 구간별**로 아래 4개 카테고리 모두 포함
  - Headline Metrics (5)
  - Alpha Quality (4)
  - Operational Viability (4)
  - Regime Robustness (3개 국면 × 4개 지표)
