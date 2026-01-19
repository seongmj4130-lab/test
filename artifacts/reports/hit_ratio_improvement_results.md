# Hit Ratio 개선안 적용 결과

## 적용된 개선안

### 1. 정규화 방법 개선
- **변경 전**: `normalization_method: percentile`
- **변경 후**: `normalization_method: robust_zscore`
- **적용 위치**: `l8`, `l8_short`, `l8_long`

### 2. 피처 가중치 최적화 (모든 피처 사용)
- **변경 전**: `min_feature_ic: 0.0` (IC > 0만 유지)
- **변경 후**: `min_feature_ic: -0.1` (음수 IC 포함, -0.1 이하만 제거)
- **효과**: 더 많은 피처 사용 가능

## 측정 결과

### 통합 랭킹 Hit Ratio (score_ens vs true_short)

| 구간 | Hit Ratio | 샘플 수 |
|------|-----------|---------|
| **전체** | **41.58%** | 20,921 |
| Dev | 41.16% | - |
| Holdout | 43.08% | - |

### 개별 랭킹 Hit Ratio

#### 단기 랭킹 (score_total_short vs true_short, 20일 수익률)
| 구간 | Hit Ratio | 샘플 수 |
|------|-----------|---------|
| 전체 | 41.58% | 20,921 |
| Dev | 41.16% | - |
| Holdout | 43.08% | - |

#### 장기 랭킹 (score_total_long vs true_long, 120일 수익률)
| 구간 | Hit Ratio | 샘플 수 |
|------|-----------|---------|
| 전체 | 38.72% | 19,449 |
| Dev | 38.13% | - |
| Holdout | 41.45% | - |

## 과적합 판단

- **Dev - Holdout Gap**: -1.93%p
- **심각도**: **low** (정상 범위)
- **판단**: Dev와 Holdout Hit Ratio 차이가 임계값(10%p) 이내로 정상 범위

## 목표 달성 여부

| 지표 | 목표 | 실제 | 달성 여부 |
|------|------|------|----------|
| 전체 Hit Ratio | ≥ 50% | 41.58% | ❌ 미달 (차이: 8.42%p) |
| Holdout Hit Ratio | ≥ 50% | 43.08% | ❌ 미달 (차이: 6.92%p) |

## 현재 설정 파라미터

### L5 모델 학습
- `ridge_alpha`: 1.5
- `min_feature_ic`: -0.1 ✅ (변경됨)
- `filter_features_by_ic`: True
- `use_rank_ic`: True

### L6R 랭킹 스코어링
- `alpha_short`: 0.5
- `alpha_long`: None (자동 계산)
- `regime_alpha`:
  - `bull_strong`: 0.6
  - `bull_weak`: 0.6
  - `neutral`: 0.5
  - `bear_weak`: 0.4
  - `bear_strong`: 0.4

### L8 랭킹 엔진
- `normalization_method`: robust_zscore ✅ (변경됨)
- `use_sector_relative`: True
- `sector_col`: sector_name

## 분석 및 개선 방향

### 현재 상태
1. **과적합 없음**: Dev와 Holdout 차이가 작아 모델이 일반화되고 있음
2. **목표 미달**: Hit Ratio가 50% 목표에 미달 (8.42%p 부족)
3. **Holdout 성과**: Holdout(43.08%)이 Dev(41.16%)보다 높아 긍정적 신호

### 추가 개선 방안

#### 1. 단기/장기 결합 비율 조정 (Priority: High)
- 현재: `alpha_short: 0.5` (5:5)
- 제안: `alpha_short: 0.6` (단기 랭킹 가중치 증가)
- 예상 효과: +2~5%p

#### 2. 정규화 방법 추가 테스트
- `robust_zscore` 외에 `zscore`도 테스트
- 피처별로 최적 정규화 방법 선택

#### 3. 피처 필터링 추가 완화
- `min_feature_ic: -0.1` → `-0.2` (더 많은 피처 포함)
- 또는 IC 필터링 비활성화 (`filter_features_by_ic: false`)

#### 4. 국면별 α 조정 강화
- 현재 국면별 α가 적용되고 있으나, 더 세밀한 조정 필요
- Bull 시장에서 단기 랭킹 가중치를 더 높게 (0.6 → 0.7)

#### 5. 피처 가중치 재최적화
- IC 기반 가중치를 더 강하게 적용
- 국면별 피처 가중치 적용 검토

## 다음 단계

1. ✅ 정규화 방법 개선 (robust_zscore) - 완료
2. ✅ 피처 필터링 완화 (min_feature_ic: -0.1) - 완료
3. ⏳ 단기/장기 결합 비율 조정 테스트
4. ⏳ 추가 정규화 방법 테스트
5. ⏳ 피처 가중치 재최적화

## 측정 일시

- **측정 시간**: 2025-01-XX (스크립트 실행 시점)
- **Config 파일**: `configs/config.yaml`
- **스크립트**: `scripts/measure_ranking_hit_ratio.py`

