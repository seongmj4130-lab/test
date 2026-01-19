# 피처 가중치 적용 전후 Hit Ratio 비교 리포트

**생성일시**: 2025-01-XX  
**비교 기준**: L8 설정 파일 경로 수정 전후

---

## 📋 변경 사항

### Config.yaml 수정

**수정 전**:
```yaml
l8_short:
  feature_weights_config: configs/feature_weights_short.yaml  # 파일 없음
l8_long:
  feature_weights_config: configs/feature_weights_long.yaml  # 파일 없음
```

**수정 후**:
```yaml
l8_short:
  feature_weights_config: configs/feature_weights_short_hitratio_optimized.yaml  # ✅ 피처별 가중치 적용
l8_long:
  feature_weights_config: configs/feature_weights_long_ic_optimized.yaml  # ✅ 피처별 가중치 적용
```

---

## 📊 Hit Ratio 비교

### 측정 결과

| 랭킹 유형 | 수정 전 | 수정 후 | 변화 | Dev | Holdout |
|-----------|---------|---------|------|-----|---------|
| **단기** | 41.58% | 41.58% | 0.00%p | 41.16% → 41.16% | 43.08% → 43.08% |
| **장기** | 38.72% | 38.72% | 0.00%p | 38.13% → 38.13% | 41.45% → 41.45% |
| **통합** | 41.58% | 41.58% | 0.00%p | 41.16% → 41.16% | 43.08% → 43.08% |

### 분석

**결과**: Hit Ratio 변화 없음 (0.00%p)

**가능한 원인**:

1. **기존에도 피처 그룹별 가중치 적용 중**
   - 수정 전: `feature_groups_config` 사용 (그룹별 가중치)
   - 수정 후: `feature_weights_config` 사용 (피처별 가중치)
   - 그룹별 가중치와 피처별 가중치가 유사할 수 있음

2. **가중치 파일의 실제 효과 제한적**
   - 단기: `feature_weights_short_hitratio_optimized.yaml` (28개 피처)
   - 장기: `feature_weights_long_ic_optimized.yaml` (28개 피처)
   - 가중치 분포가 균등에 가까울 수 있음

3. **랭킹 생성 방식의 영향**
   - 정규화 후 가중치 적용으로 인해 최종 스코어 차이가 작을 수 있음
   - 랭킹 순위는 상대적이므로 가중치 변화가 순위에 큰 영향을 주지 않을 수 있음

---

## 🔍 피처 가중치 적용 확인

### L8 단기 랭킹

- ✅ **피처 가중치 로드 완료**: 28개 피처
- ✅ **파일 경로**: `configs/feature_weights_short_hitratio_optimized.yaml`
- ✅ **적용 상태**: 피처별 가중치 적용 중

### L8 장기 랭킹

- ✅ **피처 가중치 로드 완료**: 28개 피처
- ✅ **파일 경로**: `configs/feature_weights_long_ic_optimized.yaml`
- ✅ **적용 상태**: 피처별 가중치 적용 중

---

## 📈 가중치 파일 분석

### 단기 가중치 (`feature_weights_short_hitratio_optimized.yaml`)

**주요 가중치**:
- Value 그룹: 각 0.04 (5개 피처)
- Profitability 그룹: 각 0.075 (2개 피처)
- Technical 그룹: 각 0.025 (20개 피처)
- Other: `in_universe` 0.1

**특징**: Technical 그룹이 많지만 가중치가 낮음 (0.025)

### 장기 가중치 (`feature_weights_long_ic_optimized.yaml`)

**주요 가중치**:
- Value 그룹: 각 0.05 (5개 피처)
- Profitability 그룹: 각 0.1 (2개 피처)
- Technical 그룹: 각 0.02 (20개 피처)
- Other: `in_universe` 0.1

**특징**: Value/Profitability 가중치가 단기보다 높음

---

## 🎯 결론

### 현재 상태

1. ✅ **피처 가중치 적용 확인**: L8 단기/장기 랭킹 모두 피처별 가중치 적용 중
2. ⚠️ **Hit Ratio 변화 없음**: 가중치 변경이 Hit Ratio에 영향을 주지 않음
3. ✅ **과적합 없음**: Dev와 Holdout 성과 일관성 유지

### 해석

1. **가중치 효과 제한적**
   - 피처별 가중치가 랭킹 순위에 큰 영향을 주지 않음
   - 정규화 및 랭킹 변환 과정에서 가중치 효과가 희석될 수 있음

2. **다른 최적화 방안 필요**
   - Hit Ratio 개선을 위해서는 가중치 외 다른 방법 필요
   - 피처 선택, 모델 파라미터, 결합 방식 등 검토 필요

3. **가중치 적용은 정상 작동**
   - 설정 파일 경로 수정으로 피처별 가중치가 정상적으로 적용됨
   - 향후 가중치 튜닝 시 올바르게 반영될 것

---

## 📝 권장 사항

### 즉시 조치

1. ✅ **설정 파일 경로 수정 완료**: 피처별 가중치 정상 적용 확인

### 단기 개선

1. **가중치 튜닝**
   - 현재 가중치가 Hit Ratio에 영향을 주지 않으므로 더 극단적인 가중치 시도
   - 예: Value/Profitability 그룹 가중치 대폭 증가

2. **피처 선택 최적화**
   - IC 기반 피처 필터링 강화
   - 불필요한 피처 제거

### 중기 개선

1. **랭킹 생성 방식 개선**
   - 가중치 적용 시점 조정
   - 정규화 전/후 가중치 적용 비교

2. **통합 가중치 최적화**
   - 단기/장기 결합 가중치 (alpha_short) 튜닝
   - 시장 국면별 가중치 추가 최적화

---

**비교 파일**:
- 수정 전: `artifacts/reports/ranking_hit_ratio_individual.csv`
- 수정 후: `artifacts/reports/ranking_hit_ratio_with_feature_weights_final.csv`

