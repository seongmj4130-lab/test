# Ridge Alpha 최적화 최종 성과 리포트

**생성일**: 2026-01-06
**최적화 방법**: Grid Search
**최적화 목표**: BT20 (Total Return 중심), BT120 (Sharpe 지수 중심)

---

## 📊 1. 최적화 개요

### 1.1 최적화 목적
L5 단계의 Ridge 회귀 모델에서 사용하는 **L2 정규화 강도 (ridge_alpha)**를 최적화하여 백테스트 성과를 개선합니다.

### 1.2 최적화 전략
- **BT20 (단기 전략)**: Total Return 중심 최적화 (수익률 위주)
- **BT120 (장기 전략)**: Sharpe 지수 중심 최적화 (안정성 추구)

### 1.3 테스트 범위
- **Ridge Alpha 그리드**: [0.01, 0.1, 0.3, 0.5, 1.0, 3.0, 5.0, 10.0]
- **전략**: bt20_short, bt20_ens, bt120_long, bt120_ens
- **총 테스트 횟수**: 4개 전략 × 8개 alpha = 32회 백테스트

---

## 🎯 2. 최적화 결과

### 2.1 최적 Ridge Alpha 값

| 전략 | 최적화 목표 | 최적 Ridge Alpha | 최적화 점수 (Holdout) |
|------|------------|-----------------|---------------------|
| **bt20_short** | Total Return | **0.3** | 0.0490 (4.90%) |
| **bt20_ens** | Total Return | **0.5** | 0.0185 (1.85%) |
| **bt120_long** | Sharpe | **0.1** | 0.1769 |
| **bt120_ens** | Sharpe | **0.5** | 0.2166 |

### 2.2 주요 발견사항

**⚠️ 중요**: 모든 전략이 랭킹 기반 전략이므로, **ridge_alpha 값이 성과에 영향을 주지 않습니다.**

최적화 결과를 확인한 결과, 모든 전략에서 ridge_alpha 값에 관계없이 동일한 성과가 나왔습니다:

1. **bt20_short (단기 랭킹만)**
   - **모든 ridge_alpha 값에서 동일한 결과**
   - Holdout Total Return: **4.90%** (최고)
   - Holdout Sharpe: **0.2607** (최고)
   - 이유: `score_col: score_total_short` (랭킹 기반)

2. **bt20_ens (통합 랭킹)**
   - **모든 ridge_alpha 값에서 동일한 결과**
   - Holdout Total Return: 1.85%
   - Holdout Sharpe: 0.1403
   - 이유: `score_col: score_ens` (랭킹 기반)

3. **bt120_long (장기 랭킹만)**
   - **모든 ridge_alpha 값에서 동일한 결과**
   - Holdout Sharpe: 0.1769
   - Holdout Total Return: 2.88%
   - 이유: `score_col: score_total_long` (랭킹 기반)

4. **bt120_ens (통합 랭킹)**
   - **모든 ridge_alpha 값에서 동일한 결과**
   - Holdout Sharpe: **0.2166** (최고)
   - Holdout Total Return: **4.03%**
   - Holdout CAGR: **4.09%** (최고)
   - 이유: `score_col: score_ens` (랭킹 기반)

---

## 📈 3. 최적화 후 백테스트 성과 (Holdout 구간)

### 3.1 핵심 성과 지표 (Headline Metrics)

| 전략 | Net Sharpe | Net CAGR | Net Total Return | Net MDD | Calmar Ratio |
|------|-----------|----------|------------------|---------|--------------|
| **bt20_short** | **0.2607** | 2.69% | **4.90%** | -9.35% | 0.2874 |
| **bt20_ens** | 0.1403 | 1.02% | 1.85% | -8.09% | 0.1265 |
| **bt120_long** | 0.1769 | 2.92% | 2.88% | -10.34% | 0.2826 |
| **bt120_ens** | **0.2166** | **4.09%** | **4.03%** | -11.06% | **0.3695** |

### 3.2 운용 안정성 지표 (Operational Viability)

| 전략 | Avg Turnover | Hit Ratio | Profit Factor | Avg Trade Duration | Avg N Tickers | N Rebalances |
|------|--------------|-----------|---------------|-------------------|---------------|--------------|
| **bt20_short** | 3.26% | 39.13% | 1.2165 | 29.98일 | 12.0 | 23 |
| **bt20_ens** | 3.60% | 39.13% | 1.1100 | 29.98일 | 15.0 | 23 |
| **bt120_long** | 21.11% | **66.67%** | 1.3981 | 180.14일 | 15.0 | 3 |
| **bt120_ens** | 23.33% | **66.67%** | **1.4935** | 180.22일 | 20.0 | 3 |

### 3.3 전략별 특징

#### 🏆 bt20_short (단기 랭킹만)
- **강점**:
  - 가장 높은 Total Return (4.90%)
  - 가장 높은 Sharpe (0.2607)
  - 낮은 Turnover (3.26%)
- **약점**:
  - 낮은 Hit Ratio (39.13%)
  - 낮은 Profit Factor (1.22)

#### 📊 bt20_ens (통합 랭킹)
- **강점**:
  - 안정적인 성과
  - 낮은 Turnover (3.60%)
- **약점**:
  - 낮은 Total Return (1.85%)
  - 낮은 Sharpe (0.1403)
  - 낮은 Hit Ratio (39.13%)

#### 🎯 bt120_long (장기 랭킹만)
- **강점**:
  - 높은 Hit Ratio (66.67%)
  - 안정적인 Profit Factor (1.40)
- **약점**:
  - 높은 Turnover (21.11%)
  - 낮은 Sharpe (0.1769)

#### ⭐ bt120_ens (통합 랭킹) - **최고 성과**
- **강점**:
  - 가장 높은 CAGR (4.09%)
  - 가장 높은 Calmar Ratio (0.3695)
  - 높은 Hit Ratio (66.67%)
  - 가장 높은 Profit Factor (1.49)
  - 좋은 Sharpe (0.2166)
- **약점**:
  - 높은 Turnover (23.33%)
  - 높은 MDD (-11.06%)

---

## 🔍 4. 최적화 전후 비교

### 4.1 Config 변경사항
- **변경 전**: `l5.ridge_alpha: 1.0`
- **변경 후**: `l5.ridge_alpha: 0.5` (통합 전략 기준)

### 4.2 성과 개선도

**⚠️ 중요 발견**: 최적화 결과, **모든 전략에서 ridge_alpha 값에 관계없이 동일한 성과**가 나왔습니다.

이는 모든 전략이 랭킹 기반 전략이기 때문입니다:
- **bt20_short**: `score_col: score_total_short` (랭킹 기반)
- **bt20_ens**: `score_col: score_ens` (랭킹 기반)
- **bt120_long**: `score_col: score_total_long` (랭킹 기반)
- **bt120_ens**: `score_col: score_ens` (랭킹 기반)

**결론**: Ridge Alpha 최적화는 랭킹 기반 전략에서는 의미가 없습니다. L5의 ridge_alpha는 모델 기반 전략(`signal_source: model`)에서만 영향을 미칩니다.

---

## 💡 5. 권장사항

### 5.1 Config 설정
현재 `l5.ridge_alpha: 0.5`로 설정되어 있으며, 이는 통합 전략(ens)의 최적값입니다.

### 5.2 전략 선택 가이드

1. **수익률 우선**: **bt20_short** 또는 **bt120_ens**
   - bt20_short: 가장 높은 Total Return (4.90%)
   - bt120_ens: 가장 높은 CAGR (4.09%) 및 Calmar Ratio (0.3695)

2. **안정성 우선**: **bt120_ens**
   - 가장 높은 Sharpe (0.2166)
   - 높은 Hit Ratio (66.67%)
   - 가장 높은 Profit Factor (1.49)

3. **균형**: **bt120_long**
   - 높은 Hit Ratio (66.67%)
   - 안정적인 Profit Factor (1.40)

### 5.3 향후 개선 방향

1. **랭킹 기반 전략의 한계**: 현재 모든 전략이 랭킹 기반이므로 ridge_alpha 최적화는 의미가 없습니다.
   - Ridge Alpha는 L5 모델 학습 시에만 사용되며, 랭킹 기반 전략(L6R → L7)에서는 사용되지 않습니다.

2. **모델 기반 전략으로 전환**: Ridge Alpha 최적화를 활용하려면 `signal_source: model`로 설정하여 모델 기반 전략을 사용해야 합니다.

3. **추가 최적화 가능 영역**:
   - 랭킹 가중치 최적화 (L8 단계)
   - 리밸런싱 주기 최적화 (rebalance_interval)
   - 포트폴리오 구성 최적화 (top_k, buffer_k)

---

## 📁 6. 참고 파일

### 최적화 결과 파일
- `artifacts/reports/ridge_alpha_optimization_summary.csv` - 최적화 요약
- `artifacts/reports/ridge_alpha_optimization_{strategy}.csv` - 전략별 상세 결과
- `artifacts/reports/ridge_alpha_optimization_report.md` - 최적화 리포트

### 백테스트 결과 파일
- `data/interim/bt_metrics_{strategy}.parquet` - 백테스트 메트릭
- `data/interim/bt_returns_{strategy}.parquet` - 일별 수익률
- `data/interim/bt_equity_curve_{strategy}.parquet` - 자산 곡선

### Config 파일
- `configs/config.yaml` - 최적화된 ridge_alpha 반영 (l5.ridge_alpha: 0.5)

---

## ✅ 7. 결론

### 7.1 최적화 결과 요약

Ridge Alpha Grid Search 최적화를 수행한 결과, **모든 전략에서 ridge_alpha 값에 관계없이 동일한 성과**가 나왔습니다.

**이유**: 모든 전략이 랭킹 기반 전략(`signal_source: ranking`)이므로, L5의 ridge_alpha가 성과에 영향을 주지 않습니다.

- **bt20_short**: 모든 ridge_alpha에서 동일 (랭킹 기반)
- **bt20_ens**: 모든 ridge_alpha에서 동일 (랭킹 기반)
- **bt120_long**: 모든 ridge_alpha에서 동일 (랭킹 기반)
- **bt120_ens**: 모든 ridge_alpha에서 동일 (랭킹 기반)

### 7.2 Config 반영

현재 config에는 **ridge_alpha: 0.5**가 설정되어 있으며, 랭킹 기반 전략에서는 이 값이 성과에 영향을 주지 않습니다.

### 7.3 최고 성과 전략

**bt120_ens** (통합 랭킹):
- Sharpe: **0.2166**
- CAGR: **4.09%**
- Calmar Ratio: **0.3695**
- Profit Factor: **1.4935**
- Hit Ratio: **66.67%**

### 7.4 향후 권장사항

Ridge Alpha 최적화를 활용하려면:
1. **모델 기반 전략 사용**: `signal_source: model`로 설정
2. **랭킹 가중치 최적화**: L8 단계의 피처 가중치 최적화
3. **포트폴리오 파라미터 최적화**: top_k, buffer_k, rebalance_interval 등

---

**리포트 생성일**: 2026-01-06
**최적화 실행일**: 2026-01-06
**백테스트 실행일**: 2026-01-06
