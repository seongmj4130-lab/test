# Track A 전체 파이프라인 및 최적화 작업 위치

**작성일**: 2026-01-08  
**목적**: Track A 전체 파이프라인 구조와 최적화 작업이 이루어지는 단계 파악

---

## 📊 Track A 전체 파이프라인 구조

### 기본 데이터 준비 단계 (순차 실행)

```
┌─────────────────────────────────────────────────────────────────┐
│                    공통 데이터 준비 (L0~L4)                       │
└─────────────────────────────────────────────────────────────────┘

L0: Universe 구성
  └─ KOSPI200 유니버스 구성
       ↓
L1: OHLCV 데이터 전처리
  └─ 가격 데이터 정규화 및 전처리
       ↓
L2: 재무 데이터 병합 (DART)
  └─ 재무제표 데이터 병합 (lag 적용)
       ↓
L3: 패널 데이터 병합
  └─ OHLCV + 재무 데이터 + 뉴스 데이터 병합
       ↓
L4: Walk-Forward CV 분할
  └─ Dev/Holdout 구간 분할 (시간순)
       ↓
┌─────────────────────────────────────────────────────────────────┐
│              Raw 데이터 준비 완료 (병렬 실행 가능)                 │
└─────────────────────────────────────────────────────────────────┘
```

### 최적화 모델 생성 단계 (병렬 독립 실행) ⭐

```
                    Raw 데이터 (L0~L4 완료)
                            │
        ┌───────────────────┴───────────────────┐
        │                                       │
┌───────▼────────────────┐           ┌─────────▼──────────────────┐
│ Grid Search 모델       │           │ ML 모델 (Ridge/XGBoost/RF) │
│ (Baseline 랭킹)        │           │                            │
│                        │           │                            │
│ L8 단계 실행           │           │ L5 단계 실행               │
│ - 피처 그룹별 가중치    │           │ - 개별 피처 가중치 학습     │
│   최적화 (Grid Search) │           │ - Ridge Alpha 튜닝         │
│ - score_baseline 생성  │           │ - score_ml 생성            │
│                        │           │                            │
│ 평가: Hit/IC/ICIR      │           │ 평가: Hit/IC/ICIR          │
│ (Dev/Holdout)          │           │ (Dev/Holdout)              │
└────────────────────────┘           └────────────────────────────┘
        │                                       │
        └───────────────────┬───────────────────┘
                            ↓
                ┌───────────────────────┐
                │ 앙상블 랭킹 구성       │
                │                       │
                │ score_ensemble =      │
                │   α × score_baseline  │
                │   + (1-α) × score_ml  │
                │                       │
                │ 최적화: ICIR 최대     │
                │ (α Grid Search)       │
                └───────────────────────┘
                            ↓
                ┌───────────────────────┐
                │ L6R: 리밸런싱 스코어   │
                │ L7: 백테스트 실행      │
                └───────────────────────┘
```

### 기존 순차 파이프라인 (참고용)

```
L5: 모델 학습 (Ridge Regression) ⭐ 최적화 1
  └─ 개별 피처 가중치 학습
  └─ Ridge Alpha 튜닝 (현재: 16.0)
       ↓
L6: 스코어 생성
  └─ 단기/장기 스코어 결합
       ↓
L6R: 리밸런싱 스코어 변환
  └─ 랭킹 스코어를 리밸런싱 스코어로 변환
       ↓
L7: 백테스트 실행 ⭐ 최적화 2
  └─ 실제 운용 시뮬레이션
  └─ top_k, holding_days, cost_bps 등 파라미터 최적화
       ↓
L8: 랭킹 엔진 실행 ⭐ 최적화 3
  └─ Baseline 랭킹 생성
  └─ 피처 그룹별 가중치 최적화 (Grid Search)
  └─ ML 랭킹 생성 (L5 모델 사용)
  └─ 앙상블 랭킹 생성
```

---

## 🎯 최적화 작업 위치 (병렬 독립 실행 구조)

### ⚠️ 중요: Grid Search 모델과 ML 모델은 병렬 독립 실행 가능

**핵심 아이디어**:
- Raw 데이터 (L0~L4 완료 후)에서 Grid Search 모델과 ML 모델을 **병렬로 독립 실행**
- 각 모델은 같은 raw 데이터를 사용하지만 **서로 독립적**으로 실행 가능
- 각 모델의 랭킹 결과를 앙상블로 결합

**장점**:
1. **병렬 실행 가능**: 두 모델이 독립적이므로 동시에 실행 가능
2. **의존성 제거**: ML 모델이 먼저 실행될 필요 없음
3. **유연성**: 각 모델을 독립적으로 최적화 가능
4. **앙상블 구성**: 각 모델의 결과를 독립적으로 평가 후 앙상블 구성

### ⭐ 최적화 1: L5 단계 - ML 모델 학습 (Ridge/XGBoost/RF)

**위치**: `src/stages/modeling/l5_train_models.py`

**최적화 내용**:
1. **Ridge Alpha 튜닝**
   - 현재 값: 16.0 (과적합 방지 강화)
   - 설정: `configs/config.yaml` → `l5.ridge_alpha`
   - 목적: 과적합 위험 감소, 일반화 성능 향상

2. **피처 선택 및 가중치**
   - 피처 필터링: IC 기반 (`filter_features_by_ic: true`)
   - 피처 가중치: `feature_weights_config_short`, `feature_weights_config_long`
   - 설정 파일:
     - 단기: `configs/feature_weights_short_hitratio_optimized.yaml`
     - 장기: `configs/feature_weights_long_ic_optimized.yaml`

3. **모델 학습**
   - 목표 변수: `ret_fwd_20d` (단기), `ret_fwd_120d` (장기)
   - 변환: `cs_rank` (Cross-Sectional Rank)
   - 평가: Dev/Holdout 구간별 IC, Hit Ratio, ICIR

**최적화 방법**:
- Grid Search (Ridge Alpha)
- Walk-Forward CV (시간순 분할)
- Dev/Holdout 평가

**관련 스크립트**:
- `scripts/run_l5_direct_retrain.py`: L5 모델 재학습

---

### ⭐ 최적화 3: 앙상블 구성 (Grid + ML)

**위치**: `scripts/optimize_track_a_ensemble.py` (예정)

**최적화 내용**:
1. **앙상블 가중치 최적화**
   - Grid Search 모델과 ML 모델의 랭킹 결과 결합
   - 앙상블 가중치: `score_ensemble = α × score_baseline + (1-α) × score_ml`
   - 최적화 목표: ICIR 최대화
   - 최적화 방법: Grid Search (α 간격 0.1)

2. **평가 지표**
   - Hit Ratio, IC, ICIR (Dev/Holdout)
   - 각 모델별 성과 비교

**최적화 방법**:
- Grid Search (앙상블 가중치 α)
- 평가 구간: Dev + Holdout

**관련 스크립트**:
- `scripts/optimize_track_a_ensemble.py`: 앙상블 최적화 (예정)

---

### ⭐ 최적화 4: L7 단계 - 백테스트 파라미터

**위치**: `src/tracks/track_b/stages/backtest/l7_backtest.py`

**최적화 내용**:
1. **백테스트 파라미터**
   - `top_k`: 상위 종목 수 (현재: 12)
   - `holding_days`: 보유 기간 (현재: 20)
   - `cost_bps`: 거래비용 (현재: 10.0)
   - `slippage_bps`: 슬리피지 비용 (현재: 0.0)
   - 설정: `configs/config.yaml` → `l7.*`

2. **백테스트 전략**
   - `l7_bt20_short`: 단기 전략 (20일)
   - `l7_bt120_long`: 장기 전략 (120일)
   - `l7_bt20_ens`: 앙상블 전략 (20일)
   - `l7_bt120_ens`: 앙상블 전략 (120일)

**최적화 방법**:
- Grid Search (Track B 최적화)
- 목표: Sharpe Ratio 0.6+ 달성
- 평가 지표: Sharpe, MDD, Total Return, Hit Ratio

**관련 스크립트**:
- `scripts/run_trackb_backtest_compare_costfix.py`: Track B 백테스트 비교

---

### ⭐ 최적화 2: L8 단계 - Grid Search 모델 (Baseline 랭킹)

**위치**: `src/tracks/track_a/stages/ranking/l8_rank_engine.py`

**최적화 내용**:
1. **Baseline 랭킹 (피처 그룹별 가중치)**
   - Grid Search 모델: 피처 그룹별 가중치 최적화
   - 설정: `configs/config.yaml` → `l8_short.feature_groups_config`, `l8_long.feature_groups_config`
   - 최적화 파일:
     - 단기: `configs/feature_groups_short_optimized_grid_20260108_135117.yaml`
     - 장기: `configs/feature_groups_long_optimized_grid_20260108_145118.yaml`
   - 최적 가중치:
     - technical: -0.5
     - value: 0.5
     - profitability: 0.0
     - news: 0.0

2. **ML 랭킹 (L5 모델 사용)**
   - L5 단계에서 학습한 Ridge 모델 사용
   - 개별 피처 가중치 자동 학습

3. **앙상블 랭킹**
   - Baseline + ML 랭킹 결합
   - 가중치 최적화 (예정: Phase 2)

**최적화 방법**:
- Grid Search (피처 그룹별 가중치)
- Ridge 학습 (개별 피처 가중치)
- 평가 지표: Hit Ratio, IC, ICIR (Dev/Holdout)

**관련 스크립트**:
- `scripts/optimize_track_a_feature_groups_grid.py`: Grid Search 실행
- `scripts/optimize_track_a_ridge_learning.py`: Ridge 학습 실행

---

## 📋 단계별 상세 설명

### L0: Universe 구성
- **목적**: KOSPI200 유니버스 구성
- **최적화**: 없음 (기본 설정)
- **파일**: `src/stages/universe/l0_universe.py`

### L1: OHLCV 데이터 전처리
- **목적**: 가격 데이터 정규화 및 전처리
- **최적화**: 없음 (기본 설정)
- **파일**: `src/stages/data/l1_ohlcv.py`

### L2: 재무 데이터 병합 (DART)
- **목적**: 재무제표 데이터 병합 (lag 적용)
- **최적화**: 없음 (기본 설정)
- **파일**: `src/stages/data/l2_fundamentals_dart.py`
- **설정**: `configs/config.yaml` → `l2.*`

### L3: 패널 데이터 병합
- **목적**: OHLCV + 재무 데이터 + 뉴스 데이터 병합
- **최적화**: 없음 (기본 설정)
- **파일**: `src/stages/data/l3_panel_merge.py`
- **설정**: `configs/config.yaml` → `l3.*`

### L4: Walk-Forward CV 분할
- **목적**: Dev/Holdout 구간 분할 (시간순)
- **최적화**: 없음 (기본 설정)
- **파일**: `src/stages/modeling/l4_walkforward_split.py`
- **설정**: `configs/config.yaml` → `l4.*`
  - `holdout_years: 2`
  - `step_days: 20`
  - `test_window_days: 20`
  - `embargo_days: 20`

### L5: 모델 학습 (Ridge Regression) ⭐ 최적화 1
- **목적**: 개별 피처 가중치 학습
- **최적화**: 
  - Ridge Alpha 튜닝 (현재: 16.0)
  - 피처 선택 및 가중치
- **파일**: `src/stages/modeling/l5_train_models.py`
- **설정**: `configs/config.yaml` → `l5.*`

### L6: 스코어 생성
- **목적**: 단기/장기 스코어 결합
- **최적화**: 없음 (기본 설정)
- **파일**: `src/stages/modeling/l6_scoring.py`
- **설정**: `configs/config.yaml` → `l6.*`
  - `weight_short: 0.5`
  - `weight_long: 0.5`

### L6R: 리밸런싱 스코어 변환
- **목적**: 랭킹 스코어를 리밸런싱 스코어로 변환
- **최적화**: 없음 (기본 설정)
- **파일**: `src/tracks/track_a/stages/ranking/l6r_rebalancing_score.py`
- **설정**: `configs/config.yaml` → `l6r.*`

### L7: 백테스트 실행 ⭐ 최적화 2
- **목적**: 실제 운용 시뮬레이션
- **최적화**: 
  - top_k, holding_days, cost_bps 등 파라미터 최적화
  - Track B Grid Search (조건부)
- **파일**: `src/tracks/track_b/stages/backtest/l7_backtest.py`
- **설정**: `configs/config.yaml` → `l7.*`

### L8: 랭킹 엔진 실행 ⭐ 최적화 3
- **목적**: Baseline 랭킹 생성
- **최적화**: 
  - 피처 그룹별 가중치 최적화 (Grid Search)
  - 개별 피처 가중치 학습 (Ridge)
  - 앙상블 랭킹 최적화 (예정)
- **파일**: `src/tracks/track_a/stages/ranking/l8_rank_engine.py`
- **설정**: `configs/config.yaml` → `l8_short.*`, `l8_long.*`

---

## 🔄 최적화 작업 흐름 (병렬 독립 실행)

### Phase 1: Track A 모델 다양화 (진행 중)

**핵심**: Raw 데이터 (L0~L4) 준비 후, 모든 모델을 **병렬 독립 실행**

```
Raw 데이터 준비 (L0~L4 완료)
        │
        ├─→ Grid Search 모델 (완료) ──┐
        │   └─ L8 단계 실행            │
        │   └─ 피처 그룹별 가중치 최적화│
        │   └─ 스크립트: optimize_track_a_feature_groups_grid.py
        │   └─ 결과: feature_groups_*_optimized_grid_*.yaml
        │                              │
        ├─→ Ridge 학습 모델 (완료) ────┤
        │   └─ L5 단계 실행            │
        │   └─ 개별 피처 가중치 학습   │
        │   └─ 스크립트: optimize_track_a_ridge_learning.py
        │   └─ 설정: l5.ridge_alpha = 16.0
        │                              │
        ├─→ XGBoost 모델 (예정) ───────┤
        │   └─ L5 단계 실행            │
        │   └─ XGBoost 모델 학습       │
        │                              │
        ├─→ Random Forest 모델 (예정) ─┤
        │   └─ L5 단계 실행            │
        │   └─ Random Forest 모델 학습 │
        │                              │
        └─→ (추가 모델) ───────────────┘
                            │
                            ↓
            각 모델별 Hit/IC/ICIR 평가 (Dev/Holdout)
                            │
                            ↓
            앙상블 최적화 (ICIR 최대, α Grid Search)
```

### Phase 2: 앙상블 최적화 (예정)

```
L8 단계: 앙상블 최적화
   └─ 5개 모델의 최적 앙상블 가중치 찾기
   └─ ICIR 최대화
   └─ 스크립트: optimize_track_a_ensemble.py (예정)
```

### Phase 3: Track B 고정 백테스트 (예정)

```
L7 단계: 백테스트 실행
   └─ 4전략 백테스트 (bt120_long, bt20_short, bt20_ens 등)
   └─ Sharpe 0.6+ 기준 확인
```

### Phase 4: Track B 최적화 (조건부, 예정)

```
L7 단계: Track B 파라미터 최적화
   └─ top_k, holding_days, cost_bps 등 Grid Search
   └─ 조건: Sharpe < 0.6
```

---

## 📁 주요 파일 구조

```
03_code/
├── configs/
│   ├── config.yaml                          # 메인 설정 파일
│   ├── feature_groups_short_optimized_grid_*.yaml  # L8 Grid Search 결과 (단기)
│   ├── feature_groups_long_optimized_grid_*.yaml    # L8 Grid Search 결과 (장기)
│   ├── feature_weights_short_hitratio_optimized.yaml  # L5 피처 가중치 (단기)
│   └── feature_weights_long_ic_optimized.yaml        # L5 피처 가중치 (장기)
├── scripts/
│   ├── optimize_track_a_feature_groups_grid.py      # L8 Grid Search
│   ├── optimize_track_a_ridge_learning.py            # L8 Ridge 학습
│   ├── run_l5_direct_retrain.py                      # L5 재학습
│   └── (추후 추가)
│       ├── optimize_track_a_xgboost.py
│       ├── optimize_track_a_random_forest.py
│       └── optimize_track_a_ensemble.py
├── src/
│   ├── stages/
│   │   ├── universe/l0_universe.py                  # L0
│   │   ├── data/l1_ohlcv.py                         # L1
│   │   ├── data/l2_fundamentals_dart.py             # L2
│   │   ├── data/l3_panel_merge.py                   # L3
│   │   ├── modeling/
│   │   │   ├── l4_walkforward_split.py              # L4
│   │   │   ├── l5_train_models.py                  # L5 ⭐ 최적화 1
│   │   │   └── l6_scoring.py                        # L6
│   │   └── ...
│   └── tracks/
│       ├── track_a/stages/ranking/
│       │   ├── l6r_rebalancing_score.py             # L6R
│       │   ├── l8_rank_engine.py                    # L8 ⭐ 최적화 3
│       │   └── ranking_metrics.py                   # 평가 지표
│       └── track_b/stages/backtest/
│           └── l7_backtest.py                       # L7 ⭐ 최적화 2
└── artifacts/
    └── reports/
        ├── track_a_group_weights_grid_search_*.csv  # L8 Grid Search 결과
        ├── track_a_ridge_learning_*.csv             # L8 Ridge 학습 결과
        └── ...
```

---

## 🎯 최적화 작업 요약

| 단계 | 최적화 내용 | 최적화 방법 | 상태 | 관련 파일 |
|------|------------|------------|------|----------|
| **L5** | ML 모델 학습 (Ridge) | Ridge 학습 | ✅ 완료 | `l5_train_models.py` |
| **L5** | Ridge Alpha 튜닝 | Grid Search | ✅ 완료 | `l5_train_models.py` |
| **L5** | 피처 선택 및 가중치 | IC 기반 필터링 | ✅ 완료 | `feature_weights_*.yaml` |
| **L8** | Grid Search 모델 (Baseline) | Grid Search | ✅ 완료 | `optimize_track_a_feature_groups_grid.py` |
| **앙상블** | Grid + ML 앙상블 | Grid Search (α) | ⏳ 예정 | `optimize_track_a_ensemble.py` |
| **L7** | 백테스트 파라미터 | Grid Search | ⏳ 예정 | `l7_backtest.py` |

---

## 📝 실행 순서

### 전체 파이프라인 실행
```bash
# 전체 파이프라인 실행 (L0~L8)
python src/run_all.py
```

### 최적화 작업만 실행 (병렬 독립 실행 가능)

**Raw 데이터 준비** (먼저 실행):
```bash
# L0~L4 실행 (공통 데이터 준비)
python src/tools/run_stage_pipeline.py --stages 0,1,2,3,4
```

**모델 생성** (병렬 독립 실행 가능):
```bash
# Grid Search 모델 (Baseline 랭킹) - 독립 실행
python scripts/optimize_track_a_feature_groups_grid.py --horizon short
python scripts/optimize_track_a_feature_groups_grid.py --horizon long

# ML 모델 (Ridge) - 독립 실행
python scripts/run_l5_direct_retrain.py
python scripts/optimize_track_a_ridge_learning.py --horizon short
python scripts/optimize_track_a_ridge_learning.py --horizon long

# ML 모델 (XGBoost) - 독립 실행 (예정)
python scripts/optimize_track_a_xgboost.py --horizon short
python scripts/optimize_track_a_xgboost.py --horizon long

# ML 모델 (Random Forest) - 독립 실행 (예정)
python scripts/optimize_track_a_random_forest.py --horizon short
python scripts/optimize_track_a_random_forest.py --horizon long
```

**앙상블 구성** (모든 모델 완료 후):
```bash
# 앙상블 최적화 (Grid + ML)
python scripts/optimize_track_a_ensemble.py
```

---

**작성자**: Cursor AI  
**최종 업데이트**: 2026-01-08
