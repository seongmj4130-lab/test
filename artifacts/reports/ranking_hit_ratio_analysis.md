# 랭킹산정모델 Hit Ratio 측정 및 과적합 분석 리포트

**생성일시**: 2025-01-XX
**측정 스크립트**: `scripts/measure_ranking_hit_ratio.py`

---

## 📊 측정 결과 요약

### Hit Ratio 현황

| 구간 | Hit Ratio | 샘플 수 | 목표 대비 |
|------|-----------|---------|-----------|
| **전체** | **45.91%** | 20,921 | ✗ -4.09%p |
| Dev | 45.64% | - | ✗ -4.36%p |
| Holdout | 46.84% | - | ✗ -3.16%p |

**목표**: Hit Ratio ≥ 50%
**현재 상태**: 목표 미달 (전체 45.91%, Holdout 46.84%)

---

## 🔍 과적합 여부 판단

### 판단 결과

✅ **과적합 없음** (정상 범위)

- **Dev Hit Ratio**: 45.64%
- **Holdout Hit Ratio**: 46.84%
- **Gap**: -1.20%p (Holdout이 더 높음)
- **심각도**: Low
- **판단**: Dev와 Holdout 간 차이가 10%p 이내로 정상 범위

### 해석

1. **과적합 없음**: Holdout 성과가 Dev보다 약간 높아 과적합 의심 없음
2. **일관성**: Dev와 Holdout 성과가 유사하여 모델이 안정적
3. **개선 여지**: 전체 Hit Ratio가 목표(50%)보다 낮아 추가 최적화 필요

---

## ⚙️ 현재 적용된 파라미터

### L5 모델 학습 파라미터

| 파라미터 | 현재 값 | 변경 전 | 목적 |
|----------|---------|---------|------|
| `ridge_alpha` | **1.5** | 0.8 | 과적합 방지 강화 |
| `min_feature_ic` | **0.005** | 0.01 | 피처 확대 (12→15개) |
| `filter_features_by_ic` | True | True | IC 필터링 활성화 |
| `use_rank_ic` | True | True | Rank IC 기준 사용 |

### L6R 랭킹 스코어링 파라미터

| 파라미터 | 현재 값 | 변경 전 | 목적 |
|----------|---------|---------|------|
| `alpha_short` | 0.5 | 0.5 | 기본 단기 가중치 |
| `regime_alpha.bull_strong` | **0.6** | 0.8 | Bull 시장 단기 가중치 증가 |
| `regime_alpha.bull_weak` | **0.6** | 0.7 | Bull 시장 단기 가중치 증가 |
| `regime_alpha.neutral` | 0.5 | 0.5 | Neutral 시장 기본값 유지 |
| `regime_alpha.bear_weak` | **0.4** | 0.3 | Bear 시장 단기 가중치 감소 |
| `regime_alpha.bear_strong` | **0.4** | 0.2 | Bear 시장 단기 가중치 감소 |

---

## 📈 개선 효과 분석

### 파라미터 변경 효과

1. **ridge_alpha: 0.8 → 1.5**
   - ✅ 과적합 방지 강화 (정규화 강도 증가)
   - ⚠️ Hit Ratio 개선 효과 제한적 (45.91%로 여전히 목표 미달)

2. **min_feature_ic: 0.01 → 0.005**
   - ✅ 피처 확대 (IC 필터 완화)
   - ⚠️ Hit Ratio 개선 효과 제한적

3. **regime_alpha 조정**
   - ✅ 국면별 가중치 최적화
   - ⚠️ 단일 랭킹 모드에서는 적용되지 않음 (Dual Horizon 모드 필요)

---

## 🎯 추가 최적화 방안

### 우선순위 1: ridge_alpha 추가 증가

**현재**: 1.5
**추천**: 2.0 ~ 3.0

- **예상 효과**: +2~3%p (과적합 추가 감소)
- **리스크**: 과소적합 가능성 (너무 높으면 예측력 저하)

### 우선순위 2: Dual Horizon 모드 활성화

**현재**: 단일 랭킹 모드
**추천**: Dual Horizon 모드 (ranking_short_daily + ranking_long_daily)

- **예상 효과**: +4%p (국면 적응력 향상)
- **필요 작업**: L8 단기/장기 랭킹 생성 후 L6R에서 결합

### 우선순위 3: 피처 가중치 최적화

**현재**: feature_groups 사용
**추천**: IC 기반 최적 가중치 파일 생성

- **예상 효과**: +1~2%p (중요 피처 강조)
- **필요 작업**: `scripts/calculate_feature_ic.py` 실행 후 가중치 최적화

---

## 📋 다음 단계

### 즉시 실행 가능

1. **ridge_alpha 추가 증가**
   ```yaml
   l5:
     ridge_alpha: 2.0  # 또는 2.5, 3.0
   ```

2. **Hit Ratio 재측정**
   ```bash
   python scripts/measure_ranking_hit_ratio.py --force --output artifacts/reports/ranking_hit_ratio_measurement_v2.csv
   ```

### 중기 개선 (1~2일)

1. **Dual Horizon 모드 활성화**
   - L8 단기/장기 랭킹 생성
   - L6R에서 결합 모드 활성화

2. **피처 가중치 최적화**
   - Feature IC 계산
   - 가중치 파일 생성

---

## 📝 결론

### 현재 상태

- ✅ **과적합 없음**: Dev와 Holdout 성과 일관성 유지
- ⚠️ **목표 미달**: Hit Ratio 45.91% (목표 50% 대비 -4.09%p)
- ✅ **안정성**: Holdout 성과가 Dev보다 높아 일반화 성능 양호

### 권장 사항

1. **즉시**: `ridge_alpha`를 2.0~3.0으로 추가 증가
2. **단기**: Dual Horizon 모드 활성화 검토
3. **중기**: 피처 가중치 최적화 및 추가 피처 확장

---

**생성 스크립트**: `scripts/measure_ranking_hit_ratio.py`
**결과 파일**: `artifacts/reports/ranking_hit_ratio_measurement.csv`

