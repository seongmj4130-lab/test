# 피처 가중치 적용 방식 분석 리포트

**생성일시**: 2025-01-XX

---

## 📋 현재 피처 가중치 적용 방식

### 적용 우선순위

코드 기준으로 다음 순서로 가중치가 적용됩니다:

1. **국면별 가중치** (최우선)
   - `regime_aware_weights_config` 사용
   - 시장 국면(Bull/Bear/Neutral)별로 다른 가중치 적용
   - 활성화 조건: `regime_enabled=True` AND `regime_aware_weights_config` 존재

2. **피처별 가중치** (차선)
   - `feature_weights_config`의 `feature_weights` 사용
   - 각 피처마다 개별 가중치 적용
   - 단기/장기 각각 다른 가중치 파일 사용

3. **피처 그룹별 가중치** (최후)
   - `feature_groups_config`의 `target_weight` 사용
   - 피처를 그룹으로 묶어서 그룹별 가중치 적용
   - 그룹 내 피처는 균등 분배

4. **균등 가중치** (기본값)
   - 모든 피처에 동일한 가중치 적용

---

## 🔍 현재 적용 중인 방식

### L8 단기 랭킹 (l8_short)

**설정 파일**: `configs/config.yaml`
```yaml
l8_short:
  feature_weights_config: configs/feature_weights_short.yaml
  feature_groups_config: configs/feature_groups_short.yaml
```

**실제 적용**:
- ❌ `feature_weights_short.yaml` 파일이 **없음**
- ✅ `feature_weights_short_hitratio_optimized.yaml` 파일은 존재하지만 설정에 없음
- → **3순위: 피처 그룹별 가중치 적용** (또는 균등 가중치)

### L8 장기 랭킹 (l8_long)

**설정 파일**: `configs/config.yaml`
```yaml
l8_long:
  feature_weights_config: configs/feature_weights_long.yaml
  feature_groups_config: configs/feature_groups_long.yaml
```

**실제 적용**:
- ❌ `feature_weights_long.yaml` 파일이 **없음**
- ✅ `feature_weights_long_ic_optimized.yaml` 파일은 존재하지만 설정에 없음
- → **3순위: 피처 그룹별 가중치 적용** (또는 균등 가중치)

### L5 모델 학습 (l5)

**설정 파일**: `configs/config.yaml`
```yaml
l5:
  feature_weights_config_short: configs/feature_weights_short_hitratio_optimized.yaml
  feature_weights_config_long: configs/feature_weights_long_ic_optimized.yaml
```

**실제 적용**:
- ✅ **피처별 가중치 적용 중**
- 단기: `feature_weights_short_hitratio_optimized.yaml` 사용
- 장기: `feature_weights_long_ic_optimized.yaml` 사용

---

## 📊 가중치 파일 분석

### 단기 가중치 파일 (`feature_weights_short_hitratio_optimized.yaml`)

**피처별 가중치** (28개 피처):
- Value 그룹: `equity`, `total_liabilities`, `net_income`, `debt_ratio`, `debt_ratio_sector_z` (각 0.04)
- Profitability 그룹: `roe`, `roe_sector_z` (각 0.075)
- Technical 그룹: 모멘텀/변동성 관련 피처들 (각 0.025)
- Other 그룹: `in_universe` (0.1)

**그룹별 가중치** (메타데이터):
- Value: 0.2
- Profitability: 0.15
- Technical: 0.5
- Other: 0.1
- News: 0.05

**주의**: `group_weights`는 메타데이터로만 존재하며, 실제 코드에서는 **사용되지 않음**

### 장기 가중치 파일 (`feature_weights_long_ic_optimized.yaml`)

**피처별 가중치** (28개 피처):
- Value 그룹: 각 0.05 (단기보다 높음)
- Profitability 그룹: 각 0.1 (단기보다 높음)
- Technical 그룹: 각 0.02 (단기보다 낮음)

**그룹별 가중치** (메타데이터):
- Value: 0.25 (단기 0.2보다 높음)
- Profitability: 0.2 (단기 0.15보다 높음)
- Technical: 0.4 (단기 0.5보다 낮음)

---

## ⚠️ 문제점 및 개선 방안

### 문제점

1. **L8 랭킹에서 피처별 가중치 미적용**
   - `feature_weights_short.yaml`, `feature_weights_long.yaml` 파일이 없음
   - 설정 파일 경로와 실제 파일명 불일치
   - 현재는 피처 그룹별 가중치 또는 균등 가중치 적용 중

2. **그룹별 가중치 미사용**
   - `group_weights`는 메타데이터로만 존재
   - 실제 코드에서는 피처별 가중치만 사용
   - 그룹별 가중치를 적용하려면 코드 수정 필요

### 개선 방안

#### 즉시 조치

1. **L8 설정 파일 경로 수정**
   ```yaml
   l8_short:
     feature_weights_config: configs/feature_weights_short_hitratio_optimized.yaml

   l8_long:
     feature_weights_config: configs/feature_weights_long_ic_optimized.yaml
   ```

2. **또는 심볼릭 링크 생성**
   ```bash
   # Windows에서는 mklink 사용
   mklink configs\feature_weights_short.yaml configs\feature_weights_short_hitratio_optimized.yaml
   mklink configs\feature_weights_long.yaml configs\feature_weights_long_ic_optimized.yaml
   ```

#### 중기 개선

1. **그룹별 가중치 적용 로직 추가**
   - `group_weights`를 실제로 사용하도록 코드 수정
   - 그룹별 가중치를 피처별 가중치로 변환하는 로직 추가

2. **가중치 적용 방식 통일**
   - L5와 L8에서 동일한 가중치 파일 사용
   - 단기/장기 각각 일관된 가중치 적용

---

## 📝 결론

### 현재 상태

1. **L5 모델 학습**: ✅ 피처별 가중치 적용 중
   - 단기: `feature_weights_short_hitratio_optimized.yaml`
   - 장기: `feature_weights_long_ic_optimized.yaml`

2. **L8 랭킹 엔진**: ⚠️ 피처별 가중치 미적용
   - 설정 파일 경로와 실제 파일명 불일치
   - 피처 그룹별 가중치 또는 균등 가중치 적용 중

3. **그룹별 가중치**: ❌ 미사용
   - `group_weights`는 메타데이터로만 존재
   - 실제 코드에서는 사용되지 않음

### 권장 사항

1. **즉시**: L8 설정 파일 경로 수정하여 피처별 가중치 적용
2. **단기**: 그룹별 가중치 적용 로직 추가 검토
3. **중기**: L5와 L8 가중치 적용 방식 통일

---

**생성 스크립트**: `scripts/check_feature_weights.py`
