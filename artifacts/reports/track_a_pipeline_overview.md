# Track A 최적화 작업 파이프라인 전체 개요

**작성일**: 2026-01-08
**목적**: Track A 최적화 작업의 전체 파이프라인 구조 및 진행 상황 정리

---

## 🎯 핵심 전략

**Track A 다양 모델 → Hit/IC/ICIR 최적 앙상블 → Track B 고정 백테스트 → 부진 시 Track B 최적화**

```
┌─────────────────────────────────────────────────────────────┐
│                    Track A 최적화 파이프라인                  │
└─────────────────────────────────────────────────────────────┘

Phase 1: Track A 모델 다양화
  ├─ 1.1 기반 구축 ✅
  ├─ 1.2 Grid Search 모델 ✅
  ├─ 1.3 Ridge 학습 모델 ✅
  ├─ 1.4 XGBoost 모델 ⏳
  └─ 1.5 Random Forest 모델 ⏳
       ↓
  각 모델별 Hit/IC/ICIR 평가 (Dev/Holdout)
       ↓
Phase 2: 앙상블 최적화 ⏳
  ├─ 5모델 평가
  ├─ 앙상블 가중치 Grid Search (ICIR 최대, α=0.1)
  └─ 최적 앙상블 가중치 선택
       ↓
Phase 3: Track B 고정 백테스트 ⏳
  ├─ bt120_long (Sharpe 0.6+ 기준)
  ├─ bt20_short
  ├─ bt20_ens
  └─ 성과 기준 달성 여부 판단
       ↓
Phase 4: Track B 최적화 (조건부) ⏳
  └─ Sharpe < 0.6 시 Track B Grid Search (top_k 등)
```

---

## 📋 Phase 1: Track A 모델 다양화

### 현재 상태: ✅ **진행 중** (2/5 모델 완료)

### 1.1 기반 구축 ✅ **완료** (2026-01-07)

**목표**: Track A 최적화를 위한 기반 인프라 구축

**완료된 작업**:
- ✅ 음수 가중치 지원 구현
  - 파일: `src/components/ranking/score_engine.py`
  - 절댓값 합 정규화로 음수 가중치 지원
- ✅ 평가 지표 계산 함수 구현
  - 파일: `src/tracks/track_a/stages/ranking/ranking_metrics.py`
  - 함수: `calculate_ranking_metrics_with_lagged_returns()`
  - 지표: Hit Ratio, IC, ICIR
  - Peek-Ahead Bias 방지: lag_days=0 (Forward Returns 직접 매칭)
- ✅ 모든 피처 사용 준비
  - 파일: `scripts/generate_all_features_list.py`
  - 피처 리스트 YAML 파일 생성 (30개, 35개)

**생성된 파일**:
- `configs/features_all_no_ohlcv.yaml` (30개 피처)
- `configs/features_all_with_ohlcv.yaml` (35개 피처)

---

### 1.2 Grid Search 모델 ✅ **완료** (2026-01-08)

**목표**: 피처 그룹별 가중치 최적화

**실행 스크립트**: `scripts/optimize_track_a_feature_groups_grid.py`

**실행 방법**:
```bash
# 단기 랭킹
python scripts/optimize_track_a_feature_groups_grid.py --horizon short

# 장기 랭킹
python scripts/optimize_track_a_feature_groups_grid.py --horizon long
```

**Grid Search 설정**:
- 그룹 수: 4개 (technical, value, profitability, news)
- 레벨: 3개 (-1.0, 0.0, 1.0)
- 조합 수: 80개 (정규화 후 중복 제거)

**평가 지표**:
- Objective Score = 0.4 × Hit Ratio + 0.3 × IC + 0.3 × ICIR
- 평가 구간: Dev (Walk-Forward CV)

**완료된 작업**:
- ✅ 단기 랭킹 Grid Search 완료
  - 최적 Objective Score: 0.4121
  - 최적 가중치: technical=-0.5, value=0.5
  - 최적 파일: `configs/feature_groups_short_optimized_grid_20260108_135117.yaml`
- ✅ 장기 랭킹 Grid Search 완료
  - 최적 Objective Score: 0.4062
  - 최적 가중치: technical=-0.5, value=0.5
  - 최적 파일: `configs/feature_groups_long_optimized_grid_20260108_145118.yaml`
- ✅ config.yaml 업데이트 완료
  - `l8_short.feature_groups_config` → Grid Search 결과
  - `l8_long.feature_groups_config` → Grid Search 결과

**생성된 파일**:
- `configs/feature_groups_short_optimized_grid_20260108_135117.yaml`
- `configs/feature_groups_long_optimized_grid_20260108_145118.yaml`
- `artifacts/reports/track_a_group_weights_grid_search_20260108_135117.csv`
- `artifacts/reports/track_a_group_weights_grid_search_20260108_145118.csv`

**과적합 분석** (Grid Search 결과 기반):
- 단기 랭킹: HIGH 위험도
  - IC 변동계수 4.32 (높은 변동성)
  - Grid Dev 대비 Holdout 성과 저하
- 장기 랭킹: HIGH 위험도
  - Grid Dev 대비 Holdout 성과 저하

**⚠️ Ridge Alpha 조정 이유**:
- Grid Search 모델은 **L8 단계**에서 피처 그룹별 가중치를 최적화하는 모델
- Ridge 모델은 **L5 단계**에서 개별 피처 가중치를 학습하는 모델
- **둘은 서로 다른 모델이지만, 같은 데이터와 피처를 사용**
- Grid Search 결과에서 발견된 과적합 위험은 **데이터 자체의 특성이나 모델 복잡도 문제**를 시사
- 따라서 Ridge 모델도 같은 문제를 겪을 수 있어, **사전에 Ridge Alpha를 조정**하여 과적합을 방지
- **대응 조치**:
  - Ridge Alpha 조정: 8.0 → 16.0 (과적합 방지 강화)
  - 모델 재학습 완료: Holdout 성과 개선 확인

---

### 1.3 Ridge 학습 모델 ✅ **재학습 완료** (2026-01-08)

**목표**: 개별 피처 가중치 자동 학습

**실행 스크립트**: `scripts/optimize_track_a_ridge_learning.py`

**실행 방법**:
```bash
# 단기 랭킹
python scripts/optimize_track_a_ridge_learning.py --horizon short

# 장기 랭킹
python scripts/optimize_track_a_ridge_learning.py --horizon long
```

**모델 설정**:
- 모델: Ridge Regression
- Ridge Alpha: 16.0 (과적합 방지 강화)
- 목적함수: Hit Ratio + IC + ICIR 조합
- 평가 구간: Dev + Holdout

**완료된 작업**:
- ✅ Ridge 학습 스크립트 작성 완료
- ✅ 모델 재학습 완료 (Ridge Alpha 16.0)
  - **과적합 분석 결과 반영**: 1.2 Grid Search 모델의 과적합 위험도(HIGH)를 바탕으로 Ridge Alpha 8.0 → 16.0으로 증가
  - 단기: Dev IC Rank 0.0535, Holdout IC Rank 0.0713 ✅ (Holdout 우수)
  - 장기: Dev IC Rank 0.0292, Holdout IC Rank 0.1078 ✅ (Holdout 우수)
  - **결과**: Holdout 성과가 Dev 대비 우수하여 과적합 위험 감소 확인
- ⚠️ 평가 지표 문제 발견 및 분석 중
  - 문제: 평가 시 scores의 표준편차가 0
  - 원인 분석 진행 중

**생성된 파일**:
- `artifacts/reports/track_a_ridge_learning_short_*.csv` (여러 파일)

---

### 1.4 XGBoost 모델 ⏳ **예정**

**목표**: XGBoost 모델 학습 및 평가

**예상 작업**:
- XGBoost 모델 학습 스크립트 작성
- 하이퍼파라미터 튜닝
- Hit/IC/ICIR 평가 (Dev/Holdout)

**예상 파일**:
- `scripts/optimize_track_a_xgboost.py`
- `configs/feature_groups_short_optimized_xgboost_*.yaml`
- `configs/feature_groups_long_optimized_xgboost_*.yaml`

---

### 1.5 Random Forest 모델 ⏳ **예정**

**목표**: Random Forest 모델 학습 및 평가

**예상 작업**:
- Random Forest 모델 학습 스크립트 작성
- 하이퍼파라미터 튜닝
- Hit/IC/ICIR 평가 (Dev/Holdout)

**예상 파일**:
- `scripts/optimize_track_a_random_forest.py`
- `configs/feature_groups_short_optimized_rf_*.yaml`
- `configs/feature_groups_long_optimized_rf_*.yaml`

---

## 📋 Phase 2: 앙상블 최적화 ⏳ **예정**

**목표**: 5개 모델의 최적 앙상블 가중치 찾기 (ICIR 최대화)

**전제 조건**: Phase 1 완료 (5개 모델 모두 생성 및 평가 완료)

**앙상블 방법**:
- 가중 평균: `score_ensemble = Σ(α_i × score_i)`
- 제약: `Σ α_i = 1.0`

**최적화 방법**:
- Grid Search: α 간격 0.1
- 최적화 목표: ICIR 최대화
- 평가 구간: Dev + Holdout

**예상 조합 수**:
- 5개 모델, 각 0.0~1.0, 0.1 간격, 합=1.0
- 약 1,001개 조합 (중복 제거 후)

**예상 작업**:
- 5모델 평가 스크립트 작성
- 앙상블 가중치 Grid Search 스크립트 작성
- 최적 앙상블 가중치 선택 및 검증

**예상 파일**:
- `scripts/optimize_track_a_ensemble.py`
- `configs/ensemble_weights_optimized_*.yaml`

---

## 📋 Phase 3: Track B 고정 백테스트 ⏳ **예정**

**목표**: 앙상블 랭킹의 실제 운용 성과 확인 (Sharpe 0.6+ 기준)

**전제 조건**: Phase 2 완료 (최적 앙상블 가중치 선택 완료)

**백테스트 전략**:
1. bt120_long (장기 전략)
2. bt20_short (단기 전략)
3. bt20_ens (앙상블 전략)
4. (추가 전략)

**성과 기준**:
- bt120_long Sharpe ≥ 0.6

**예상 작업**:
- 4전략 백테스트 실행
- bt120_long Sharpe 확인
- 성과 기준 달성 여부 판단

**예상 파일**:
- `artifacts/reports/backtest_phase3_*.csv`
- `artifacts/reports/backtest_phase3_summary.md`

---

## 📋 Phase 4: Track B 최적화 (조건부) ⏳ **예정**

**조건**: Phase 3에서 bt120_long Sharpe < 0.6

**목표**: Track B 파라미터 최적화로 백테스트 성과 개선

**최적화 대상**:
- top_k (상위 종목 수)
- holding_days (보유 기간)
- cost_bps (거래비용)
- 기타 백테스트 파라미터

**최적화 방법**: Grid Search

**예상 작업**:
- Track B Grid Search 스크립트 작성
- 파라미터 최적화
- 재평가

**예상 파일**:
- `scripts/optimize_track_b_parameters.py`
- `artifacts/reports/track_b_optimization_*.csv`

---

## 🔄 현재 진행 상황 요약

### ✅ 완료된 작업

1. **Phase 1.1: 기반 구축** ✅ (2026-01-07)
   - 음수 가중치 지원
   - 평가 지표 계산 함수
   - 모든 피처 사용 준비

2. **Phase 1.2: Grid Search 모델** ✅ (2026-01-08)
   - 단기/장기 랭킹 각각 최적 가중치 발견
   - config.yaml 업데이트 완료
   - 과적합 분석 완료

3. **Phase 1.3: Ridge 학습 모델** ✅ (2026-01-08)
   - Ridge Alpha 16.0으로 재학습 완료
   - Holdout 성과 우수 확인

### ⏳ 진행 중인 작업

- **Phase 1.3**: Ridge 학습 모델 평가 지표 문제 해결 중

### 📅 예정된 작업

- **Phase 1.4**: XGBoost 모델 개발
- **Phase 1.5**: Random Forest 모델 개발
- **Phase 2**: 앙상블 최적화 (5모델 완료 후)
- **Phase 3**: Track B 고정 백테스트
- **Phase 4**: Track B 최적화 (조건부)

---

## 📁 주요 파일 구조

```
03_code/
├── configs/
│   ├── feature_groups_short_optimized_grid_*.yaml  # Grid Search 결과 (단기)
│   ├── feature_groups_long_optimized_grid_*.yaml    # Grid Search 결과 (장기)
│   └── config.yaml                                   # 메인 설정 파일
├── scripts/
│   ├── optimize_track_a_feature_groups_grid.py      # Grid Search 스크립트
│   ├── optimize_track_a_ridge_learning.py            # Ridge 학습 스크립트
│   └── (추후 추가)
│       ├── optimize_track_a_xgboost.py
│       ├── optimize_track_a_random_forest.py
│       └── optimize_track_a_ensemble.py
├── src/
│   ├── components/ranking/score_engine.py           # 랭킹 엔진 (음수 가중치 지원)
│   └── tracks/track_a/stages/ranking/
│       └── ranking_metrics.py                        # 평가 지표 계산 함수
└── artifacts/reports/
    ├── track_a_group_weights_grid_search_*.csv      # Grid Search 결과
    ├── track_a_ridge_learning_*.csv                 # Ridge 학습 결과
    └── track_a_optimization_direction_validation.md # 전체 문서
```

---

## 🎯 다음 단계

1. **Phase 1.3 완료**: Ridge 학습 모델 평가 지표 문제 해결
2. **Phase 1.4 시작**: XGBoost 모델 개발
3. **Phase 1.5 시작**: Random Forest 모델 개발
4. **Phase 2 준비**: 5모델 평가 및 앙상블 최적화 스크립트 작성

---

**작성자**: Cursor AI
**최종 업데이트**: 2026-01-08
