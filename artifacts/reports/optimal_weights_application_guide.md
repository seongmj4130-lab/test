# 최적 가중치 적용 가이드

**작성일**: 2026-01-08
**목적**: Grid Search 및 Ridge 학습 최적 가중치 적용 방법 안내

---

## 📊 최적 가중치 파일 구분

### Grid Search 최적화 결과 (Phase 2)
- **단기 랭킹**: `configs/feature_groups_short_optimized_grid_20260108_135117.yaml`
- **장기 랭킹**: `configs/feature_groups_long_optimized_grid_20260108_145118.yaml`
- **파일명 패턴**: `feature_groups_{horizon}_optimized_grid_{timestamp}.yaml`

### Ridge 학습 최적화 결과 (Phase 3, 추후)
- **예상 파일명 패턴**: `feature_groups_{horizon}_optimized_ridge_{timestamp}.yaml`
- **예시**: `feature_groups_short_optimized_ridge_20260108_HHMMSS.yaml`

---

## ⚙️ config.yaml 설정

### 현재 설정 (Grid Search 결과 적용)

```yaml
l8_short:
  # [Phase 2 Grid Search 최적화] 그룹별 가중치 Grid Search 결과 (2026-01-08)
  # Objective Score: 0.4121, IC Mean: 0.0200, ICIR: 0.2224
  # 최적 가중치: technical=-0.5, value=0.5, profitability=0.0, news=0.0
  feature_groups_config: configs/feature_groups_short_optimized_grid_20260108_135117.yaml  # [Phase 2] Grid Search 최적화 결과
  # feature_groups_config_ridge: configs/feature_groups_short_optimized_ridge_YYYYMMDD_HHMMSS.yaml  # [Phase 3] Ridge 학습 최적화 결과 (추후 추가)

l8_long:
  # [Phase 2 Grid Search 최적화] 그룹별 가중치 Grid Search 결과 (2026-01-08)
  # Objective Score: 0.4062, IC Mean: 0.0224, ICIR: 0.2556
  # 최적 가중치: technical=-0.5, value=0.5, profitability=0.0, news=0.0
  feature_groups_config: configs/feature_groups_long_optimized_grid_20260108_145118.yaml  # [Phase 2] Grid Search 최적화 결과
  # feature_groups_config_ridge: configs/feature_groups_long_optimized_ridge_YYYYMMDD_HHMMSS.yaml  # [Phase 3] Ridge 학습 최적화 결과 (추후 추가)
```

---

## 🔄 최적 가중치 전환 방법

### Grid Search → Ridge 학습 전환

1. **Ridge 학습 완료 후 파일 생성**
   - 파일명: `feature_groups_{horizon}_optimized_ridge_{timestamp}.yaml`

2. **config.yaml 업데이트**
   ```yaml
   l8_short:
     # Grid Search 결과 (이전)
     # feature_groups_config: configs/feature_groups_short_optimized_grid_20260108_135117.yaml

     # Ridge 학습 결과 (신규)
     feature_groups_config: configs/feature_groups_short_optimized_ridge_20260108_HHMMSS.yaml
   ```

3. **성과 비교**
   - Grid Search 결과와 Ridge 학습 결과 성과 비교
   - 더 우수한 결과 선택

---

## 📝 파일명 규칙

### Grid Search 결과
- **패턴**: `feature_groups_{horizon}_optimized_grid_{timestamp}.yaml`
- **예시**: `feature_groups_short_optimized_grid_20260108_135117.yaml`
- **구분자**: `_grid_`

### Ridge 학습 결과
- **패턴**: `feature_groups_{horizon}_optimized_ridge_{timestamp}.yaml`
- **예시**: `feature_groups_short_optimized_ridge_20260108_150000.yaml`
- **구분자**: `_ridge_`

---

## ✅ 적용 완료 상태

- [x] Grid Search 최적 가중치 파일 생성
- [x] config.yaml 업데이트 (Grid Search 결과 적용)
- [ ] Ridge 학습 최적 가중치 파일 생성 (Phase 3)
- [ ] Ridge 학습 결과 적용 및 성과 비교

---

**작성자**: Cursor AI
**최종 업데이트**: 2026-01-08
