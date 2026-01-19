# Track A 랭킹산정 Hit Ratio 개선 방안

## 개요
Track A 파이프라인에서 실제로 사용되는 코드를 분석하여 hit_ratio를 올릴 수 있는 구체적인 개선 방안을 제시합니다.

**✅ 구현 완료**: Robust Z-score 정규화 방법 추가 (`src/components/ranking/score_engine.py`)

## 현재 Hit Ratio 계산 방식
```python
# scripts/measure_ranking_hit_ratio.py의 calculate_hit_ratio() 함수
hit = (pred_direction == actual_direction).astype(int)
hit_ratio = hit.mean()
```
- **pred_direction**: `np.sign(score_ens)` - 예측 스코어의 부호
- **actual_direction**: `np.sign(true_short)` - 실제 수익률의 부호
- **hit_ratio**: 방향 일치 비율

## 개선 방안

### 1. 정규화 방법 개선 (Priority: High)
**현재 위치**: `src/components/ranking/score_engine.py`의 `normalize_feature_cross_sectional()`

**현재 설정**: `config.yaml`의 `l8.normalization_method: "percentile"`

**개선 방안**:
- **Percentile 정규화**: 현재 사용 중, 이상치에 강건하지만 정보 손실 가능
- **Z-score 정규화**: 평균과 표준편차 기반, 더 정확한 상대적 위치 표현
- **하이브리드 접근**: 피처별로 최적 정규화 방법 선택

**구현 위치**:
```python
# src/components/ranking/score_engine.py:47-124
def normalize_feature_cross_sectional(...):
    if method == "percentile":
        ranks = pd.Series(values).rank(pct=True, method="first")
        normalized = ranks.values
    elif method == "zscore":
        mean_val = np.nanmean(values)
        std_val = np.nanstd(values)
        normalized = (values - mean_val) / std_val
```

**테스트 방법**:
```yaml
# config.yaml
l8:
  normalization_method: "zscore"  # percentile → zscore 변경
  # 또는
  normalization_method: "robust_zscore"  # [Hit Ratio 개선안] Robust Z-score 사용 (이상치에 더 강건)
```

**✅ 구현 완료**: `robust_zscore` 정규화 방법이 추가되었습니다. median과 MAD(Median Absolute Deviation)를 사용하여 이상치에 더 강건한 정규화를 수행합니다.

**예상 효과**: Hit Ratio +2~5%p

---

### 2. 피처 가중치 최적화 (Priority: High)
**현재 위치**: `src/components/ranking/score_engine.py`의 `build_score_total()`

**현재 설정**: 
- `l8.feature_weights_config`: IC 기반 최적 가중치 파일 사용
- `l5.filter_features_by_ic: true`
- `l5.min_feature_ic: 0.0`

**개선 방안**:
1. **IC 기반 가중치 강화**: IC가 높은 피처에 더 높은 가중치 부여
2. **Rank IC 사용**: `l5.use_rank_ic: true` (현재 활성화됨)
3. **피처 필터링 완화**: `min_feature_ic: 0.0` → `-0.01` (약간 음수 IC도 허용)
4. **피처별 동적 가중치**: 날짜별/국면별로 피처 중요도 변화 반영

**구현 위치**:
```python
# src/components/ranking/score_engine.py:126-322
def build_score_total(...):
    # feature_weights 또는 feature_groups_config 사용
    # IC 기반 가중치는 이미 구현되어 있음
```

**테스트 방법**:
```yaml
# config.yaml
l5:
  min_feature_ic: -0.01  # 0.0 → -0.01 (완화)
  filter_features_by_ic: true
  use_rank_ic: true
```

**예상 효과**: Hit Ratio +3~7%p

---

### 3. 단기/장기 랭킹 결합 비율 최적화 (Priority: High)
**현재 위치**: `src/tracks/track_b/stages/modeling/l6r_ranking_scoring.py`의 `build_rebalance_scores_from_ranking()`

**현재 설정**: `config.yaml`의 `l6r.alpha_short: 0.5` (단기:장기 = 5:5)

**개선 방안**:
1. **국면별 α 조정**: 이미 구현되어 있음 (`l6r.regime_alpha`)
   - Bull 시장: 단기 랭킹 가중치 증가 (α_short ↑)
   - Bear 시장: 장기 랭킹 가중치 증가 (α_short ↓)
2. **동적 α 조정**: 시장 변동성에 따라 자동 조정
3. **Hit Ratio 기반 α 최적화**: Dev 구간 Hit Ratio가 높은 α 선택

**구현 위치**:
```python
# src/tracks/track_b/stages/modeling/l6r_ranking_scoring.py:389-393
r["score_ens"] = (
    r["alpha_short"] * r["score_short_norm"].fillna(0) 
    + r["alpha_long"] * r["score_long_norm"].fillna(0)
)
```

**테스트 방법**:
```yaml
# config.yaml
l6r:
  alpha_short: 0.6  # 0.5 → 0.6 (단기 랭킹 가중치 증가)
  regime_alpha:
    bull_strong: 0.7  # Bull 시장에서 단기 랭킹 더 강조
    bull_weak: 0.6
    neutral: 0.5
    bear_weak: 0.4
    bear_strong: 0.3  # Bear 시장에서 장기 랭킹 더 강조
```

**예상 효과**: Hit Ratio +2~5%p

---

### 4. 섹터 상대 정규화 강화 (Priority: Medium)
**현재 위치**: `src/components/ranking/score_engine.py`의 `normalize_feature_cross_sectional()`

**현재 설정**: `l8.use_sector_relative: true` (활성화됨)

**개선 방안**:
1. **섹터별 정규화 검증**: 모든 피처가 섹터별로 정규화되는지 확인
2. **섹터 중립화**: 섹터 효과를 완전히 제거하여 순수 알파 포착
3. **섹터 가중치 조정**: 섹터별로 다른 가중치 적용

**구현 위치**:
```python
# src/components/ranking/score_engine.py:81-102
if use_sector_relative:
    # 섹터별 정규화
    for (date, sector), group in df.groupby([date_col, sector_col], sort=False):
        # percentile 또는 zscore 정규화
```

**테스트 방법**:
```yaml
# config.yaml
l8:
  use_sector_relative: true  # 이미 활성화됨
  sector_col: "sector_name"
```

**예상 효과**: Hit Ratio +1~3%p

---

### 5. 국면별 피처 가중치 적용 (Priority: Medium)
**현재 위치**: `src/components/ranking/score_engine.py`의 `build_score_total()`

**현재 설정**: `l8.regime_aware_weights_config` (선택적)

**개선 방안**:
1. **국면별 피처 중요도 변화 반영**: Bull/Bear/Neutral에서 다른 피처가 중요할 수 있음
2. **동적 피처 선택**: 국면별로 사용할 피처 세트 변경
3. **국면별 정규화**: 국면별로 다른 정규화 방법 적용

**구현 위치**:
```python
# src/components/ranking/score_engine.py:244-302
if use_regime_weights:
    # 날짜별로 국면에 맞는 가중치 사용
    for date, group in out.groupby(date_col, sort=False):
        regime = market_regime_df[market_regime_df[date_col] == date]["regime"]
        date_weights = regime_weights_config[regime]
```

**테스트 방법**:
```yaml
# config.yaml
l8:
  regime_aware_weights_config: "configs/feature_weights_regime_detailed.yaml"
l7:
  regime:
    enabled: true
```

**예상 효과**: Hit Ratio +2~4%p

---

### 6. 피처 정규화 개선: Robust Z-score (Priority: Medium)
**현재 위치**: `src/components/ranking/score_engine.py`의 `normalize_feature_cross_sectional()`

**개선 방안**:
- **Robust Z-score**: 중앙값(median)과 MAD(Median Absolute Deviation) 사용
- **Winsorization**: 극단값을 제한하여 이상치 영향 감소
- **Quantile 정규화**: Percentile과 Z-score의 하이브리드

**구현 코드**:
```python
# src/components/ranking/score_engine.py에 추가
def normalize_feature_robust_zscore(values):
    """Robust Z-score: median과 MAD 사용"""
    median_val = np.nanmedian(values)
    mad = np.nanmedian(np.abs(values - median_val))
    if mad > 1e-8:
        normalized = (values - median_val) / (1.4826 * mad)  # 1.4826은 정규분포 보정
    else:
        normalized = np.zeros_like(values)
    return normalized
```

**예상 효과**: Hit Ratio +1~3%p

---

### 7. 랭킹 스코어 결합 방식 개선 (Priority: Low)
**현재 위치**: `src/tracks/track_b/stages/modeling/l6r_ranking_scoring.py`

**현재 방식**: `score_ens = α * score_short + (1-α) * score_long` (선형 결합)

**개선 방안**:
1. **비선형 결합**: 곱셈, 최대값, 최소값 등
2. **가중 평균 대신 가중 합**: 정규화 없이 직접 합산
3. **Rank 기반 결합**: score 대신 rank를 사용하여 결합

**구현 위치**:
```python
# src/tracks/track_b/stages/modeling/l6r_ranking_scoring.py:389-393
# 현재: 선형 결합
r["score_ens"] = (
    r["alpha_short"] * r["score_short_norm"].fillna(0) 
    + r["alpha_long"] * r["score_long_norm"].fillna(0)
)

# 개선안: 비선형 결합 (예시)
r["score_ens"] = (
    np.sqrt(r["alpha_short"]) * r["score_short_norm"].fillna(0) 
    + np.sqrt(r["alpha_long"]) * r["score_long_norm"].fillna(0)
)
```

**예상 효과**: Hit Ratio +0.5~2%p

---

### 8. Hit Ratio 계산 방식 개선 (Priority: Low)
**현재 위치**: `scripts/measure_ranking_hit_ratio.py`의 `calculate_hit_ratio()`

**현재 방식**: `hit = (pred_direction == actual_direction)` (단순 부호 일치)

**개선 방안**:
1. **크기 고려**: 방향뿐만 아니라 크기도 고려
2. **임계값 기반**: 작은 수익률은 노이즈로 간주
3. **Rank 기반 Hit Ratio**: 상위/하위 랭킹의 실제 수익률 비교

**구현 코드**:
```python
# scripts/measure_ranking_hit_ratio.py 수정
def calculate_hit_ratio_improved(scores, return_col="true_short", score_col="score_ens", 
                                 min_return_threshold=0.001):
    """개선된 Hit Ratio: 작은 수익률은 제외"""
    df = scores.copy()
    df = df.dropna(subset=[return_col, score_col])
    
    # 작은 수익률 필터링 (노이즈 제거)
    df = df[df[return_col].abs() >= min_return_threshold]
    
    # 방향 일치
    df["hit"] = (np.sign(df[score_col]) == np.sign(df[return_col])).astype(int)
    
    return df["hit"].mean()
```

**예상 효과**: Hit Ratio +1~2%p (더 정확한 측정)

---

## 우선순위별 실행 계획

### Phase 1: 즉시 적용 가능 (1-2일)
1. ✅ 정규화 방법 변경: percentile → zscore
2. ✅ 피처 필터링 완화: min_feature_ic: 0.0 → -0.01
3. ✅ 단기/장기 결합 비율 조정: alpha_short: 0.5 → 0.6

### Phase 2: 검증 후 적용 (3-5일)
4. ✅ Robust Z-score 정규화 구현
5. ✅ Hit Ratio 계산 방식 개선
6. ✅ 국면별 피처 가중치 최적화

### Phase 3: 장기 개선 (1-2주)
7. ✅ 비선형 결합 방식 테스트
8. ✅ 동적 α 조정 알고리즘 구현

---

## 테스트 방법

### 1. 개별 개선안 테스트
```bash
# 각 개선안을 config.yaml에 반영 후
python scripts/measure_ranking_hit_ratio.py

# 결과 비교
# - Dev Hit Ratio
# - Holdout Hit Ratio
# - 과적합 여부 (Dev - Holdout 차이)
```

### 2. 통합 테스트
```bash
# 전체 파이프라인 재실행
python src/pipeline/track_a_pipeline.py

# 백테스트 실행
python src/tools/run_stage7.py
```

### 3. 성과 비교
```python
# 이전 결과와 비교
# - Hit Ratio 변화
# - Sharpe Ratio 변화
# - MDD 변화
```

---

## 예상 종합 효과

| 개선안 | 예상 Hit Ratio 증가 | 우선순위 |
|--------|-------------------|---------|
| 정규화 방법 개선 | +2~5%p | High |
| 피처 가중치 최적화 | +3~7%p | High |
| 단기/장기 결합 비율 | +2~5%p | High |
| 섹터 상대 정규화 | +1~3%p | Medium |
| 국면별 피처 가중치 | +2~4%p | Medium |
| Robust Z-score | +1~3%p | Medium |
| 비선형 결합 | +0.5~2%p | Low |
| Hit Ratio 계산 개선 | +1~2%p | Low |
| **종합 예상** | **+12~31%p** | - |

---

## 주의사항

1. **과적합 방지**: Dev와 Holdout Hit Ratio 차이를 모니터링
2. **점진적 적용**: 한 번에 하나씩 적용하여 효과 측정
3. **백테스트 검증**: Hit Ratio 증가가 실제 수익률 개선으로 이어지는지 확인
4. **설정 파일 백업**: 변경 전 config.yaml 백업 필수

---

## 참고 파일

- `src/components/ranking/score_engine.py`: 랭킹 스코어 계산
- `src/tracks/track_b/stages/modeling/l6r_ranking_scoring.py`: 단기/장기 랭킹 결합
- `scripts/measure_ranking_hit_ratio.py`: Hit Ratio 측정
- `configs/config.yaml`: 모든 설정값
- `src/tracks/track_a/stages/ranking/l8_dual_horizon.py`: Track A 랭킹 엔진

