# Hit Ratio 기준 모델 평가 및 최적화 리포트

생성일: 2026-01-06

## 1. 평가 기준

- **목표**: Holdout Hit Ratio ≥ 50%
- **과적합 기준**: Dev - Holdout ≤ 10%
- **최적화 방향**: 과적합 없이 Hit Ratio 50% 이상 달성

## 2. 현재 모델별 Hit Ratio 현황

| 모델 | Dev HR | Holdout HR | 과적합 | 목표달성 | 상태 | 우선순위 |
|------|--------|------------|--------|----------|------|----------|
| **bt20_ens** | 51.22% | **52.17%** | -0.95% | ✅ | ✅ 양호 | 낮음 |
| **bt20_short** | 57.32% | **43.48%** | 13.84% | ❌ | ⚠️ 과적합 | **높음** |
| **bt120_ens** | 38.89% | **61.11%** | -22.22% | ✅ | ✅ 양호 | 낮음 |
| **bt120_long** | 34.72% | **66.67%** | -31.94% | ✅ | ✅ 양호 | 낮음 |

### 주요 발견사항

1. **BT20_ENS**: ✅ 목표 달성 (52.17%), 과적합 없음
2. **BT20_SHORT**: ❌ 목표 미달 (43.48%), 과적합 13.84% → **최적화 필요**
3. **BT120_ENS**: ✅ 목표 달성 (61.11%), 과적합 없음
4. **BT120_LONG**: ✅ 목표 달성 (66.67%), 과적합 없음

## 3. BT20_SHORT 최적화 방안

### 현재 문제점
- Holdout Hit Ratio: 43.48% (< 50% 목표)
- 과적합: 13.84% (Dev 57.32% - Holdout 43.48%)
- 원인: technical 피쳐 그룹이 과적합 유발 가능성

### 최적화 전략

#### 3.1 피쳐 가중치 재조정

**현재 가중치**:
- value: 0.15
- profitability: 0.10
- **technical: 0.60** ← 과적합 유발 가능성
- other: 0.10
- news: 0.05

**최적화 제안**:
- value: 0.15 → **0.20** (재무 지표 강화)
- profitability: 0.10 → **0.15** (수익성 지표 강화)
- **technical: 0.60 → 0.50** (과적합 유발 피쳐 축소)
- other: 0.10 (유지)
- news: 0.05 (유지)

**이유**:
- Technical 피쳐(변동성, 모멘텀)가 단기에서 과적합 유발
- Value/Profitability 피쳐는 안정적 예측력 제공

#### 3.2 정규화 강화

**현재**: `ridge_alpha = 0.5`
**제안**: `ridge_alpha = 0.8 ~ 1.0`

**효과**: 과적합 감소, 일반화 성능 향상

#### 3.3 교차 검증 개선

**현재**: `embargo_days = 20`
**제안**: `embargo_days = 30`

**효과**: Lookahead bias 추가 방지, 과적합 감소

#### 3.4 피쳐 선택

- IC < 0.01인 피쳐 제거
- Rank IC 기준 필터링 활성화
- `filter_features_by_ic: true`
- `min_feature_ic: 0.01`

## 4. 적용 방법

### Step 1: 가중치 파일 생성
✅ 완료: `configs/feature_weights_short_hitratio_optimized.yaml`

### Step 2: config.yaml 수정

```yaml
l5:
  feature_weights_config_short: configs/feature_weights_short_hitratio_optimized.yaml
  ridge_alpha: 0.8  # 0.5 → 0.8 (과적합 감소)
  filter_features_by_ic: true  # false → true
  min_feature_ic: 0.01  # 0.0 → 0.01

l4:
  embargo_days: 30  # 20 → 30 (선택사항)
```

### Step 3: 재학습 및 백테스트

```bash
# 예측 결과 삭제 (재학습 강제)
rm data/interim/pred_short_oos.parquet

# 백테스트 실행
python scripts/run_backtest_4models.py
```

### Step 4: 결과 확인

목표:
- Holdout Hit Ratio ≥ 50%
- 과적합 ≤ 10% (Dev - Holdout)

## 5. 예상 효과

### 가중치 재조정
- Technical 그룹 축소 → 과적합 감소 예상
- Value/Profitability 강화 → 안정적 예측력 향상
- 예상: Holdout Hit Ratio 43.48% → 48~52%

### 정규화 강화
- Ridge alpha 증가 → 모델 복잡도 감소
- 예상: 과적합 13.84% → 8~10%

### 피쳐 선택
- IC 낮은 피쳐 제거 → 노이즈 감소
- 예상: Hit Ratio 추가 1~2% 개선

## 6. 최종 목표

| 지표 | 현재 | 목표 | 개선 필요 |
|------|------|------|-----------|
| Holdout Hit Ratio | 43.48% | ≥ 50% | +6.52%p |
| 과적합 (Dev-Holdout) | 13.84% | ≤ 10% | -3.84%p |

## 7. 다음 단계

1. ✅ 최적화 가중치 파일 생성 완료
2. ⏳ config.yaml 수정 필요
3. ⏳ L5 재학습 및 백테스트 실행
4. ⏳ 결과 확인 및 추가 조정

