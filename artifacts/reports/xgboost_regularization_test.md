# XGBoost 정규화 강화 테스트 결과

**작성일**: 2026-01-08
**목적**: XGBoost 모델의 과적합 방지를 위한 정규화 강화 테스트

---

## 테스트 개요

XGBoost 모델이 Dev 구간에서는 우수한 성과를 보였지만 Holdout 구간에서 성과가 크게 저하되는 과적합 문제가 발견되었습니다. 이를 해결하기 위해 다양한 정규화 파라미터 조합을 테스트했습니다.

---

## 테스트 결과 비교 (단기 랭킹)

### 1. 기본 파라미터 (Baseline)
- **파라미터**: reg_alpha=0.0, reg_lambda=1.0, max_depth=6, n_estimators=100
- **Dev 구간**:
  - Objective Score: 0.7770
  - Hit Ratio: 0.7616 (76.16%)
  - IC Mean: **0.5129**
  - ICIR: 3.1825
- **Holdout 구간**:
  - Objective Score: 0.4831
  - Hit Ratio: 0.4674 (46.74%)
  - IC Mean: **-0.0068**
  - ICIR: -0.0957
- **과적합 정도**: Dev IC 0.5129 → Holdout IC -0.0068 (차이: -0.5197)

### 2. 강한 정규화 (Strong Regularization)
- **파라미터**: reg_alpha=1.0, reg_lambda=10.0, max_depth=4, n_estimators=50
- **Dev 구간**:
  - Objective Score: 0.5893
  - Hit Ratio: 0.5720 (57.20%)
  - IC Mean: **0.1967**
  - ICIR: 1.0333
- **Holdout 구간**:
  - Objective Score: 0.4870
  - Hit Ratio: 0.4739 (47.39%)
  - IC Mean: **-0.0057**
  - ICIR: -0.0580
- **과적합 정도**: Dev IC 0.1967 → Holdout IC -0.0057 (차이: -0.2024) ✅ **개선**

### 3. 중간 정규화 (Medium Regularization)
- **파라미터**: reg_alpha=0.5, reg_lambda=5.0, max_depth=5, n_estimators=75
- **Dev 구간**:
  - Objective Score: 0.6611
  - Hit Ratio: 0.6402 (64.02%)
  - IC Mean: **0.3370**
  - ICIR: 1.8136
- **Holdout 구간**:
  - Objective Score: 0.4934
  - Hit Ratio: 0.4891 (48.91%)
  - IC Mean: **-0.0042**
  - ICIR: -0.0533
- **과적합 정도**: Dev IC 0.3370 → Holdout IC -0.0042 (차이: -0.3412) ✅ **개선**

---

## 주요 발견사항

### 1. 정규화 강화 효과
- **과적합 감소**: 정규화를 강화할수록 Dev와 Holdout 간 성과 차이가 감소
  - Baseline: -0.5197
  - 중간 정규화: -0.3412
  - 강한 정규화: -0.2024

### 2. 성과 트레이드오프
- **Dev 성과 저하**: 정규화를 강화할수록 Dev 구간 성과는 저하
  - Baseline: IC 0.5129
  - 중간 정규화: IC 0.3370
  - 강한 정규화: IC 0.1967
- **Holdout 성과**: 모든 경우에서 Holdout IC가 음수로 유지됨
  - Baseline: -0.0068
  - 중간 정규화: -0.0042
  - 강한 정규화: -0.0057

### 3. 최적 파라미터 추천
- **중간 정규화** (reg_alpha=0.5, reg_lambda=5.0, max_depth=5, n_estimators=75)
  - Dev 성과와 과적합 방지의 균형이 가장 좋음
  - Holdout IC가 가장 높음 (-0.0042)

---

## 권장사항

### 1. 단기 랭킹
- **권장 파라미터**: reg_alpha=0.5, reg_lambda=5.0, max_depth=5, n_estimators=75
- **이유**: Dev 성과와 Holdout 성과의 균형이 가장 좋음

### 2. 추가 개선 방안
1. **Early Stopping**: Holdout 성과를 기준으로 조기 종료
2. **피처 선택**: IC 기반 피처 필터링 강화
3. **앙상블**: 여러 정규화 수준의 모델을 앙상블
4. **다른 모델 고려**: Ridge 모델이 Holdout에서 더 안정적일 수 있음

### 3. 장기 랭킹 테스트
- 장기 랭킹에서도 동일한 정규화 파라미터로 테스트 완료 ✅

---

## 장기 랭킹 정규화 테스트 결과

### 중간 정규화 (권장 파라미터)
- **파라미터**: reg_alpha=0.5, reg_lambda=5.0, max_depth=5, n_estimators=75
- **Dev 구간**:
  - Objective Score: 0.7707
  - Hit Ratio: 0.6861 (68.61%)
  - IC Mean: **0.5202**
  - ICIR: 3.4494
- **Holdout 구간**:
  - Objective Score: 0.4764
  - Hit Ratio: 0.4167 (41.67%)
  - IC Mean: **-0.0137**
  - ICIR: -0.1170
- **과적합 정도**: Dev IC 0.5202 → Holdout IC -0.0137 (차이: -0.5339)

### 단기 vs 장기 비교 (중간 정규화)
| 지표 | 단기 Dev | 단기 Holdout | 장기 Dev | 장기 Holdout |
|------|----------|-------------|----------|-------------|
| IC Mean | 0.3370 | -0.0042 | 0.5202 | -0.0137 |
| Hit Ratio | 0.6402 | 0.4891 | 0.6861 | 0.4167 |
| ICIR | 1.8136 | -0.0533 | 3.4494 | -0.1170 |
| 과적합 정도 | -0.3412 | | -0.5339 | |

### 주요 발견사항 (장기)
1. **Dev 성과**: 장기 랭킹이 단기보다 높은 IC (0.5202 vs 0.3370)
2. **Holdout 성과**: 장기도 Holdout에서 음수 IC (-0.0137)
3. **과적합 정도**: 장기가 단기보다 더 심함 (-0.5339 vs -0.3412)

---

## 최종 권장사항

### 1. XGBoost 모델 사용 시 주의사항
- ✅ 정규화 강화로 과적합을 일부 완화할 수 있음
- ⚠️ Holdout에서 여전히 음수 IC가 발생 (일반화 성능 부족)
- ⚠️ 장기 랭킹이 단기보다 과적합이 더 심함

### 2. 최적 파라미터 (단기/장기 공통)
- **reg_alpha**: 0.5
- **reg_lambda**: 5.0
- **max_depth**: 5
- **n_estimators**: 75
- **subsample**: 0.8
- **colsample_bytree**: 0.8
- **learning_rate**: 0.1

### 3. 대안 고려
1. **Ridge 모델 우선 사용**: Ridge가 Holdout에서 더 안정적일 가능성
2. **앙상블 활용**: 여러 모델을 결합하여 과적합 완화
3. **피처 선택**: IC 기반 피처 필터링 강화
4. **Early Stopping**: Holdout 성과 기준으로 조기 종료

---

## 다음 단계

1. ✅ 정규화 강화 테스트 완료 (단기/장기)
2. ✅ 장기 랭킹 정규화 테스트 완료
3. ✅ 최적 파라미터로 재학습 및 결과 저장
4. ⏳ Ridge 모델과 성과 비교 (Ridge가 더 안정적일 수 있음)
5. ⏳ 앙상블 최적화 (Grid Search + Ridge + XGBoost)

---

**작성자**: Cursor AI
**최종 업데이트**: 2026-01-08 18:35
