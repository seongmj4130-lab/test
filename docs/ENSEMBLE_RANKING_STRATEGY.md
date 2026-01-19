# 앙상블 랭킹 전략 구현 가이드

## 📋 전략 개요

본 문서는 다음 3단계 랭킹 전략의 구현 가이드를 제공합니다:

1. **Baseline 랭킹**: 피처 가중치 합산 (기존 방식, **설정 수정 금지**)
2. **ML 랭킹**: ML 모델(XGBoost/LightGBM) 예측값을 랭킹으로 변환 (**L5 완전 교체**)
3. **앙상블 랭킹**: Baseline 70% + ML 30% 결합

각 랭킹마다 **동일한 백테스트 로직(Track B)**으로 4개 전략을 실행하여 비교 분석합니다.

## ⚠️ 중요 제약사항

1. **ML 모델 우선순위**: XGBoost → LightGBM (Ridge는 사용 안 함)
2. **L5 완전 교체**: 기존 L5는 ML 모델 전용으로 교체
3. **Baseline/Track B 설정 보존**: 기존 설정 파일 수정 금지
4. **동일한 백테스트 로직**: 모든 랭킹은 Track B의 동일한 백테스트 로직 사용

## 📊 실행 결과 구조

### 총 12개 백테스트 실행

| 랭킹 타입 | 전략 | 설명 |
|----------|------|------|
| **Baseline** | bt20_short | 단기 보유 + Baseline 랭킹 |
| **Baseline** | bt20_ens | 단기 보유 + Baseline 앙상블 |
| **Baseline** | bt120_long | 장기 보유 + Baseline 랭킹 |
| **Baseline** | bt120_ens | 장기 보유 + Baseline 앙상블 |
| **ML** | bt20_short | 단기 보유 + ML 랭킹 |
| **ML** | bt20_ens | 단기 보유 + ML 앙상블 |
| **ML** | bt120_long | 장기 보유 + ML 랭킹 |
| **ML** | bt120_ens | 장기 보유 + ML 앙상블 |
| **Ensemble** | bt20_short | 단기 보유 + 앙상블 랭킹 |
| **Ensemble** | bt20_ens | 단기 보유 + 앙상블 랭킹 |
| **Ensemble** | bt120_long | 장기 보유 + 앙상블 랭킹 |
| **Ensemble** | bt120_ens | 장기 보유 + 앙상블 랭킹 |

**모든 백테스트는 Track B의 동일한 로직으로 실행하여 공정한 비교 분석이 가능합니다.**

---

## 🎯 전략 구조

### 전체 흐름도

```
┌─────────────────────────────────────────────────────────────┐
│                    공통 데이터 준비 (L0~L4)                   │
│  - Universe, OHLCV, 재무 데이터, CV 분할                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
        ┌──────────────────────┴──────────────────────┐
        │                                             │
┌───────▼────────────────┐                   ┌────────▼───────────────────┐
│ 1. Baseline 랭킹        │                   │ 2. ML 랭킹                │
│ (L8: 가중치 합산)        │                   │ (L5: 모델 예측)           │
│                         │                   │                            │
│ 피처들                  │                   │ 피처들                    │
│   ↓                     │                   │   ↓                        │
│ 수동 가중치              │                   │ XGBoost/LightGBM          │
│ (기존 설정 보존)         │                   │ (L5 완전 교체)             │
│   ↓                     │                   │   ↓                        │
│ score_baseline          │                   │ y_pred                    │
│   ↓                     │                   │   ↓                        │
│ rank_baseline           │                   │ rank_ml                   │
└─────────────────────────┘                   └────────────────────────────┘
        │                                             │
        └──────────────────────┬─────────────────────┘
                               ↓
                    ┌──────────────────────┐
                    │ 3. 앙상블 랭킹        │
                    │                       │
                    │ score_ensemble =      │
                    │   0.7 * score_baseline│
                    │   + 0.3 * score_ml    │
                    │                       │
                    │ rank_ensemble         │
                    └───────────────────────┘
                               ↓
        ┌──────────────────────┴──────────────────────┐
        │                                             │
┌───────▼────────────────┐                   ┌────────▼───────────────────┐
│ 4개 백테스트 전략       │                   │ 4개 백테스트 전략           │
│ (Baseline 랭킹)        │                   │ (ML 랭킹)                  │
│                        │                   │                            │
│ - bt20_short           │                   │ - bt20_short               │
│ - bt20_ens             │                   │ - bt20_ens                 │
│ - bt120_long           │                   │ - bt120_long               │
│ - bt120_ens            │                   │ - bt120_ens                │
└────────────────────────┘                   └────────────────────────────┘
        │                                             │
        └──────────────────────┬─────────────────────┘
                               ↓
                    ┌──────────────────────┐
                    │ 4개 백테스트 전략     │
                    │ (앙상블 랭킹)         │
                    │                      │
                    │ - bt20_short          │
                    │ - bt20_ens           │
                    │ - bt120_long         │
                    │ - bt120_ens          │
                    └──────────────────────┘
```

---

## 1. Baseline 랭킹 (기존 가중치 합산)

### 📊 구조

```
피처들 → 수동 가중치 → score_baseline → rank_baseline
```

### ⚠️ 중요: Baseline 설정 보존

**Baseline 랭킹의 모든 설정은 수정하지 않습니다.**
- L8 설정 (`l8_short`, `l8_long`) 보존
- 피처 가중치 파일 보존
- 정규화 방법 보존

### 📈 Phase 2 최적화 결과 (2026-01-08)

**Grid Search 최적화 완료**: 80개 조합 평가 완료

**최적 조합 (Combination ID: 23)**:
- **Objective Score**: 0.4121
- **Hit Ratio**: 49.39%
- **IC Mean**: 0.0200 (양수, 예측력 확인)
- **ICIR**: 0.2224 (양수, 안정성 확인)
- **Rank IC Mean**: 0.0459
- **Rank ICIR**: 0.3753

**최적 그룹별 가중치**:
- `technical`: -0.5 (음수 가중치, 리버스 팩터)
- `value`: 0.5 (양수 가중치, 주요 팩터)
- `profitability`: 0.0 (사용 안 함)
- `news`: 0.0 (사용 안 함)

**최적 가중치 파일**: `configs/feature_groups_short_optimized_grid_20260108_121838.yaml`

**주요 발견사항**:
1. **음수 가중치 효과**: technical 그룹이 음수 가중치일 때 성과 향상
2. **Value 팩터 우수**: value 그룹이 양수 가중치일 때 IC 개선
3. **IC 양수 전환**: 최적 조합에서 IC가 양수로 전환 (예측력 확인)
4. **ICIR 안정화**: ICIR이 양수로 전환 (안정성 확인)

### 🔧 구현 방법

**기존 L8 랭킹 엔진 활용**

```python
# Track A 실행: Baseline 랭킹 생성
python -m src.pipeline.track_a_pipeline

# 결과:
# - ranking_short_daily.parquet (단기)
# - ranking_long_daily.parquet (장기)
# 컬럼: score_total, rank_total
```

**코드 위치:**
- `src/tracks/track_a/stages/ranking/l8_dual_horizon.py`

**설정:**
```yaml
# configs/config.yaml
l8_short:
  normalization_method: zscore
  feature_groups_config: configs/feature_groups_short.yaml  # 또는 최적화된 파일 사용
  feature_weights_config: configs/feature_weights_short_hitratio_optimized.yaml
  use_sector_relative: true
  sector_col: sector_name

l8_long:
  normalization_method: zscore
  feature_groups_config: configs/feature_groups_long.yaml
  feature_weights_config: configs/feature_weights_long_ic_optimized.yaml
  use_sector_relative: true
  sector_col: sector_name
```

**최적화된 설정 사용 (선택사항)**:
```yaml
# Phase 2 Grid Search 최적화 결과 적용
l8_short:
  feature_groups_config: configs/feature_groups_short_optimized_grid_20260108_121838.yaml
  # 최적 가중치: technical=-0.5, value=0.5, profitability=0.0, news=0.0
```

**⚠️ 중요**:
- Baseline 랭킹 기본 설정은 수정하지 않습니다.
- 최적화된 가중치는 선택적으로 적용 가능합니다.

**산출물:**
```
ranking_short_daily.parquet:
  - date, ticker
  - score_total (Baseline 점수)
  - rank_total (Baseline 랭킹)

ranking_long_daily.parquet:
  - date, ticker
  - score_total (Baseline 점수)
  - rank_total (Baseline 랭킹)
```

---

## 2. ML 랭킹 (모델 예측 → 랭킹 변환)

### 📊 구조

```
피처들 → XGBoost/LightGBM → y_pred → rank_ml
```

### ⚠️ 중요: L5 완전 교체

**기존 L5는 ML 모델 전용으로 완전 교체합니다.**
- 기존 Ridge 모델 제거
- XGBoost/LightGBM만 사용
- Baseline 설정은 수정하지 않음

### 🔧 구현 방법

#### Step 1: L5에 LightGBM 추가 및 XGBoost 우선 적용

**L5 모델 타입 우선순위:**
1. **XGBoost** (1순위)
2. **LightGBM** (2순위)
3. Ridge는 사용하지 않음

**L5 코드 수정: `src/stages/modeling/l5_train_models.py`**

```python
# LightGBM 지원 추가
if model_type in ("lgb", "lightgbm"):
    try:
        import lightgbm as lgb
    except Exception as e:
        raise ImportError("lightgbm가 필요합니다. `pip install lightgbm` 후 재실행하세요.") from e

    # 안전한 기본값(과적합 완화 방향)
    n_estimators = int(l5.get("lgb_n_estimators", 600))
    max_depth = int(l5.get("lgb_max_depth", 4))
    learning_rate = float(l5.get("lgb_learning_rate", 0.05))
    subsample = float(l5.get("lgb_subsample", 0.8))
    colsample_bytree = float(l5.get("lgb_colsample_bytree", 0.8))
    reg_lambda = float(l5.get("lgb_reg_lambda", 1.0))
    min_child_weight = float(l5.get("lgb_min_child_weight", 1.0))
    random_state = int(l5.get("random_state", 42))

    model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        min_child_weight=min_child_weight,
        objective="regression",
        n_jobs=-1,
        random_state=random_state,
    )
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ])
    return pipe, f"lightgbm(n_estimators={n_estimators}, depth={max_depth}, lr={learning_rate}, target_transform={tf})"
```

**설정 파일: `configs/config.yaml`**

```yaml
# ML 모델 전용 설정 (기존 l5 설정은 보존, 새로 추가)
l5_ml:
  model_type: xgboost  # 또는 lightgbm
  # XGBoost 설정
  xgb_n_estimators: 600
  xgb_max_depth: 4
  xgb_learning_rate: 0.05
  xgb_subsample: 0.8
  xgb_colsample_bytree: 0.8
  xgb_reg_lambda: 1.0
  xgb_min_child_weight: 1.0
  # LightGBM 설정
  lgb_n_estimators: 600
  lgb_max_depth: 4
  lgb_learning_rate: 0.05
  lgb_subsample: 0.8
  lgb_colsample_bytree: 0.8
  lgb_reg_lambda: 1.0
  lgb_min_child_weight: 1.0
  # 공통 설정
  target_transform: cs_rank
  cs_rank_center: true
  random_state: 42
```

#### Step 2: ML 모델 학습 및 예측

```bash
# XGBoost 실행 (1순위)
python scripts/run_ml_pipeline.py --model-type xgboost

# LightGBM 실행 (2순위, XGBoost 실패 시)
python scripts/run_ml_pipeline.py --model-type lightgbm
```

**산출물:**
```
pred_short_oos.parquet:
  - date, ticker, fold_id, phase
  - y_pred (예측 수익률)
  - y_true (실제 수익률)

pred_long_oos.parquet:
  - date, ticker, fold_id, phase
  - y_pred (예측 수익률)
  - y_true (실제 수익률)
```

#### Step 2: 예측값을 랭킹으로 변환

**새로운 함수 필요: `convert_predictions_to_ranking()`**

```python
# src/stages/modeling/l5_to_ranking.py (신규 파일)

import pandas as pd
import numpy as np

def convert_predictions_to_ranking(
    pred_oos: pd.DataFrame,
    horizon: int,  # 20 or 120
) -> pd.DataFrame:
    """
    L5 모델 예측값(y_pred)을 랭킹으로 변환

    Args:
        pred_oos: L5 산출물 (pred_short_oos 또는 pred_long_oos)
        horizon: 20 (단기) 또는 120 (장기)

    Returns:
        ranking_ml: ML 랭킹 데이터프레임
          - date, ticker
          - score_ml (y_pred 값)
          - rank_ml (랭킹)
    """
    # 1. fold별 예측값 집계 (평균)
    agg = pred_oos.groupby(
        ["date", "ticker", "phase"],
        as_index=False
    ).agg({
        "y_pred": "mean",
        "y_true": "mean",  # 검증용
    })

    # 2. 리밸런싱 날짜 선택 (fold의 test_end)
    # L6의 _pick_rebalance_rows_by_fold_end 로직 활용
    from src.stages.modeling.l6_scoring import _pick_rebalance_rows_by_fold_end

    # fold 정보 필요 (cv_folds에서 가져오기)
    # 여기서는 간단히 date별로 집계
    ranking = agg.groupby(
        ["date", "phase"],
        as_index=False
    ).apply(lambda g: g.nlargest(1, "date")).reset_index(drop=True)

    # 3. 랭킹 계산
    ranking["score_ml"] = ranking["y_pred"]
    ranking["rank_ml"] = ranking.groupby(
        ["date", "phase"]
    )["score_ml"].rank(ascending=False, method="first")

    # 4. 컬럼 정리
    ranking_ml = ranking[[
        "date", "ticker", "phase",
        "score_ml", "rank_ml",
        "y_true"  # 검증용
    ]].copy()

    return ranking_ml
```

**사용 예시:**
```python
# 단기 ML 랭킹
pred_short = pd.read_parquet("data/interim/pred_short_oos.parquet")
ranking_ml_short = convert_predictions_to_ranking(pred_short, horizon=20)

# 장기 ML 랭킹
pred_long = pd.read_parquet("data/interim/pred_long_oos.parquet")
ranking_ml_long = convert_predictions_to_ranking(pred_long, horizon=120)
```

---

## 3. 앙상블 랭킹 (Baseline + ML 결합)

### 📊 구조

```
score_ensemble = 0.7 * score_baseline + 0.3 * score_ml → rank_ensemble
```

### 🔧 구현 방법

**새로운 함수: `build_ensemble_ranking()`**

```python
# src/stages/modeling/ensemble_ranking.py (신규 파일)

import pandas as pd
import numpy as np

def build_ensemble_ranking(
    ranking_baseline: pd.DataFrame,  # L8 산출물
    ranking_ml: pd.DataFrame,        # L5→랭킹 변환 결과
    weight_baseline: float = 0.7,
    weight_ml: float = 0.3,
    horizon: str = "short",  # "short" or "long"
) -> pd.DataFrame:
    """
    Baseline 랭킹과 ML 랭킹을 결합하여 앙상블 랭킹 생성

    Args:
        ranking_baseline: Baseline 랭킹 (L8 산출물)
            - date, ticker, score_total, rank_total
        ranking_ml: ML 랭킹 (L5→랭킹 변환)
            - date, ticker, score_ml, rank_ml
        weight_baseline: Baseline 가중치 (기본 0.7)
        weight_ml: ML 가중치 (기본 0.3)
        horizon: "short" or "long"

    Returns:
        ranking_ensemble: 앙상블 랭킹
            - date, ticker, phase
            - score_baseline, score_ml
            - score_ensemble
            - rank_ensemble
    """
    # 1. 병합 (date, ticker, phase 기준)
    key = ["date", "ticker", "phase"]

    # Baseline 랭킹 준비
    baseline = ranking_baseline[key + ["score_total"]].copy()
    baseline = baseline.rename(columns={"score_total": "score_baseline"})

    # ML 랭킹 준비
    ml = ranking_ml[key + ["score_ml"]].copy()

    # 병합
    merged = baseline.merge(
        ml,
        on=key,
        how="outer",  # outer join (한쪽에만 있어도 포함)
        validate="one_to_one"
    )

    # 2. 가중치 결합
    # NaN 처리: 한쪽에만 있으면 있는 쪽만 사용
    mask_baseline = merged["score_baseline"].notna()
    mask_ml = merged["score_ml"].notna()

    # 정규화된 가중치 계산
    den = (weight_baseline * mask_baseline.astype(float)) + \
          (weight_ml * mask_ml.astype(float))

    num = (weight_baseline * merged["score_baseline"].fillna(0.0)) + \
          (weight_ml * merged["score_ml"].fillna(0.0))

    merged["score_ensemble"] = num / den.replace(0.0, np.nan)

    # 3. 랭킹 계산
    merged["rank_ensemble"] = merged.groupby(
        ["date", "phase"]
    )["score_ensemble"].rank(ascending=False, method="first")

    # 4. 컬럼 정리
    ranking_ensemble = merged[[
        "date", "ticker", "phase",
        "score_baseline", "score_ml",
        "score_ensemble", "rank_ensemble"
    ]].copy()

    return ranking_ensemble
```

**사용 예시:**
```python
# Baseline 랭킹 로드
ranking_baseline_short = pd.read_parquet(
    "data/interim/ranking_short_daily.parquet"
)

# ML 랭킹 로드 (위에서 생성)
ranking_ml_short = convert_predictions_to_ranking(pred_short, horizon=20)

# 앙상블 랭킹 생성
ranking_ensemble_short = build_ensemble_ranking(
    ranking_baseline=ranking_baseline_short,
    ranking_ml=ranking_ml_short,
    weight_baseline=0.7,
    weight_ml=0.3,
    horizon="short"
)
```

---

## 4. 백테스트 전략 실행

### ⚠️ 중요: 동일한 백테스트 로직 사용

**모든 랭킹(Baseline, ML, Ensemble)은 Track B의 동일한 백테스트 로직을 사용합니다.**
- Track B 설정 수정 금지
- 동일한 L7 백테스트 함수 사용
- 비교 분석을 위해 동일한 조건 유지

### 📊 4개 전략 구조

각 랭킹(Baseline, ML, Ensemble)마다 다음 4개 전략을 실행:

1. **bt20_short**: 단기 보유(20일) + 단일 랭킹
2. **bt20_ens**: 단기 보유(20일) + 앙상블 랭킹
3. **bt120_long**: 장기 보유(120일) + 단일 랭킹
4. **bt120_ens**: 장기 보유(120일) + 앙상블 랭킹

**총 12개 백테스트 실행** (3개 랭킹 × 4개 전략)

### 🔧 구현 방법

#### Step 1: 랭킹을 리밸런싱 스코어로 변환 (Track B의 L6R 활용)

```python
# src/stages/modeling/ranking_to_rebalance_scores.py (신규 파일)

from src.tracks.track_b.stages.modeling.l6r_ranking_scoring import (
    build_rebalance_scores_from_ranking,
    RankingRebalanceConfig,
)

def convert_ranking_to_rebalance_scores(
    ranking_daily: pd.DataFrame,
    cv_folds: pd.DataFrame,
    rebalance_interval: int = 1,
    alpha_short: float = 0.5,  # 단기/장기 결합 가중치 (ens 전략용)
) -> pd.DataFrame:
    """
    랭킹을 리밸런싱 스코어로 변환

    Args:
        ranking_daily: 랭킹 데이터 (Baseline/ML/Ensemble)
        cv_folds: CV 분할 정보
        rebalance_interval: 리밸런싱 주기
        alpha_short: 단기 가중치 (ens 전략용)

    Returns:
        rebalance_scores: 리밸런싱 스코어
    """
    # L6R 함수 활용
    config = RankingRebalanceConfig(
        rebalance_interval=rebalance_interval,
        alpha_short=alpha_short,
    )

    # 단기/장기 분리 (ranking_daily가 단일 horizon인 경우)
    # 여기서는 단일 랭킹만 처리하는 것으로 가정
    rebalance_scores = build_rebalance_scores_from_ranking(
        ranking_short_daily=ranking_daily,  # 단일 랭킹
        ranking_long_daily=ranking_daily,   # 동일 (단일 랭킹)
        cv_folds_short=cv_folds,
        cv_folds_long=cv_folds,
        config=config,
    )

    return rebalance_scores
```

#### Step 2: 백테스트 실행 (Track B의 L7 사용, 설정 수정 금지)

**⚠️ 중요: Track B의 기존 백테스트 로직을 그대로 사용합니다.**

```python
# src/pipeline/run_ensemble_backtest.py (신규 파일)

from src.tracks.track_b.stages.backtest.l7_backtest import run_backtest
from src.tracks.track_b.stages.modeling.l6r_ranking_scoring import (
    build_rebalance_scores_from_ranking,
    RankingRebalanceConfig,
)
from src.utils.config import load_config
import pandas as pd

def run_ensemble_backtest_strategies(
    config_path: str = "configs/config.yaml",
    ranking_type: str = "baseline",  # "baseline", "ml", "ensemble"
    force_rebuild: bool = False,
):
    """
    앙상블 랭킹 전략 백테스트 실행

    ⚠️ Track B의 기존 백테스트 로직을 그대로 사용 (설정 수정 금지)

    Args:
        config_path: 설정 파일 경로
        ranking_type: 랭킹 타입 ("baseline", "ml", "ensemble")
        force_rebuild: 재계산 여부
    """
    cfg = load_config(config_path)

    # ⚠️ Track B 설정 그대로 사용 (수정 금지)
    l7_configs = {
        "bt20_short": cfg.get("l7_bt20_short", {}),
        "bt20_ens": cfg.get("l7_bt20_ens", {}),
        "bt120_long": cfg.get("l7_bt120_long", {}),
        "bt120_ens": cfg.get("l7_bt120_ens", {}),
    }

    # 랭킹 로드
    if ranking_type == "baseline":
        ranking_short = pd.read_parquet("data/interim/ranking_short_daily.parquet")
        ranking_long = pd.read_parquet("data/interim/ranking_long_daily.parquet")
    elif ranking_type == "ml":
        from src.stages.modeling.l5_to_ranking import convert_predictions_to_ranking
        pred_short = pd.read_parquet("data/interim/pred_short_oos.parquet")
        pred_long = pd.read_parquet("data/interim/pred_long_oos.parquet")
        ranking_short = convert_predictions_to_ranking(pred_short, horizon=20)
        ranking_long = convert_predictions_to_ranking(pred_long, horizon=120)
    elif ranking_type == "ensemble":
        from src.stages.modeling.ensemble_ranking import build_ensemble_ranking
        baseline_short = pd.read_parquet("data/interim/ranking_short_daily.parquet")
        baseline_long = pd.read_parquet("data/interim/ranking_long_daily.parquet")
        pred_short = pd.read_parquet("data/interim/pred_short_oos.parquet")
        pred_long = pd.read_parquet("data/interim/pred_long_oos.parquet")
        ml_short = convert_predictions_to_ranking(pred_short, horizon=20)
        ml_long = convert_predictions_to_ranking(pred_long, horizon=120)
        ranking_short = build_ensemble_ranking(
            baseline_short, ml_short, weight_baseline=0.7, weight_ml=0.3, horizon="short"
        )
        ranking_long = build_ensemble_ranking(
            baseline_long, ml_long, weight_baseline=0.7, weight_ml=0.3, horizon="long"
        )

    # CV 분할 로드
    cv_folds_short = pd.read_parquet("data/interim/cv_folds_short.parquet")
    cv_folds_long = pd.read_parquet("data/interim/cv_folds_long.parquet")

    # 4개 전략 실행
    strategies = [
        "bt20_short",
        "bt20_ens",
        "bt120_long",
        "bt120_ens",
    ]

    results = {}
    for strategy in strategies:
        l7_cfg = l7_configs[strategy]

        # 리밸런싱 스코어 변환 (Track B의 L6R 사용)
        rebalance_interval = l7_cfg.get("rebalance_interval", 1)
        alpha_short = cfg.get("l6r", {}).get("alpha_short", 0.5)

        config = RankingRebalanceConfig(
            rebalance_interval=rebalance_interval,
            alpha_short=alpha_short if "ens" in strategy else 1.0,  # ens 전략만 결합
        )

        rebalance_scores, _, _, _ = build_rebalance_scores_from_ranking(
            ranking_short_daily=ranking_short if "20" in strategy else ranking_long,
            ranking_long_daily=ranking_long if "ens" in strategy else ranking_short,  # ens만 장기 사용
            cv_folds_short=cv_folds_short,
            cv_folds_long=cv_folds_long,
            config=config,
        )

        # ⚠️ Track B의 기존 백테스트 함수 사용 (설정 수정 금지)
        bt_result = run_backtest(
            rebalance_scores=rebalance_scores,
            config=l7_cfg,  # 전략별 설정 사용
            strategy=strategy,
        )

        results[strategy] = bt_result

    return results
```

---

## 5. 전체 파이프라인 실행 스크립트

### 📝 통합 실행 스크립트

```python
# scripts/run_ensemble_ranking_pipeline.py (신규 파일)

"""
앙상블 랭킹 전략 전체 파이프라인 실행

⚠️ 중요 제약사항:
1. ML 모델: XGBoost 우선, LightGBM 대체
2. L5 완전 교체: 기존 L5는 ML 모델 전용으로 교체
3. Baseline/Track B 설정 수정 금지
4. 동일한 백테스트 로직 사용

실행 순서:
1. Baseline 랭킹 생성 (L8, 설정 수정 금지)
2. ML 랭킹 생성 (L5 ML 모델 → 랭킹 변환)
3. 앙상블 랭킹 생성 (Baseline + ML)
4. 각 랭킹마다 4개 백테스트 전략 실행 (Track B 동일 로직)
"""

import logging
from pathlib import Path
import pandas as pd

from src.pipeline.track_a_pipeline import run_track_a_pipeline
from src.stages.modeling.l5_train_models import train_oos_predictions
from src.stages.modeling.l5_to_ranking import convert_predictions_to_ranking
from src.stages.modeling.ensemble_ranking import build_ensemble_ranking
from src.pipeline.run_ensemble_backtest import run_ensemble_backtest_strategies
from src.utils.config import load_config, get_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_full_ensemble_pipeline(
    config_path: str = "configs/config.yaml",
    force_rebuild: bool = False,
    weight_baseline: float = 0.7,
    weight_ml: float = 0.3,
    ml_model_type: str = "xgboost",  # "xgboost" or "lightgbm"
):
    """
    전체 앙상블 랭킹 파이프라인 실행

    Args:
        config_path: 설정 파일 경로
        force_rebuild: 재계산 여부
        weight_baseline: Baseline 가중치 (기본 0.7)
        weight_ml: ML 가중치 (기본 0.3)
        ml_model_type: ML 모델 타입 ("xgboost" 우선, "lightgbm" 대체)
    """
    logger.info("=" * 80)
    logger.info("앙상블 랭킹 전략 파이프라인 시작")
    logger.info(f"ML 모델: {ml_model_type} (XGBoost 우선, LightGBM 대체)")
    logger.info("=" * 80)

    cfg = load_config(config_path)
    interim_dir = Path(get_path(cfg, "data_interim"))

    # Step 1: Baseline 랭킹 생성 (설정 수정 금지)
    logger.info("[Step 1] Baseline 랭킹 생성 (L8, 설정 수정 금지)")
    run_track_a_pipeline(config_path=config_path, force_rebuild=force_rebuild)

    baseline_short = pd.read_parquet(interim_dir / "ranking_short_daily.parquet")
    baseline_long = pd.read_parquet(interim_dir / "ranking_long_daily.parquet")
    logger.info(f"  ✓ Baseline 단기: {len(baseline_short):,}행")
    logger.info(f"  ✓ Baseline 장기: {len(baseline_long):,}행")

    # Step 2: ML 랭킹 생성 (L5 ML 모델 완전 교체)
    logger.info(f"[Step 2] ML 랭킹 생성 (L5 {ml_model_type} 모델 → 랭킹 변환)")

    # L5 ML 모델 학습 및 예측 (기존 L5 교체)
    # ⚠️ 기존 l5 설정은 보존, l5_ml 설정 사용
    l5_ml_cfg = cfg.get("l5_ml", {})
    l5_ml_cfg["model_type"] = ml_model_type  # XGBoost 우선

    # 데이터 로드
    dataset_daily = pd.read_parquet(interim_dir / "dataset_daily.parquet")
    cv_folds_short = pd.read_parquet(interim_dir / "cv_folds_short.parquet")
    cv_folds_long = pd.read_parquet(interim_dir / "cv_folds_long.parquet")

    # 단기 모델 학습 (20일)
    logger.info(f"  [2-1] 단기 모델 학습 ({ml_model_type}, horizon=20)")
    pred_short, metrics_short, report_short, warns_short = train_oos_predictions(
        dataset_daily=dataset_daily,
        cv_folds=cv_folds_short,
        cfg={**cfg, "l5": l5_ml_cfg},  # l5_ml 설정을 l5로 전달
        target_col="ret_fwd_20d",
        horizon=20,
        interim_dir=interim_dir,
    )
    pred_short.to_parquet(interim_dir / "pred_short_oos_ml.parquet", index=False)
    logger.info(f"    ✓ 단기 예측: {len(pred_short):,}행, IC={report_short.get('dev_ic_rank_mean', 'N/A'):.4f}")

    # 장기 모델 학습 (120일)
    logger.info(f"  [2-2] 장기 모델 학습 ({ml_model_type}, horizon=120)")
    pred_long, metrics_long, report_long, warns_long = train_oos_predictions(
        dataset_daily=dataset_daily,
        cv_folds=cv_folds_long,
        cfg={**cfg, "l5": l5_ml_cfg},  # l5_ml 설정을 l5로 전달
        target_col="ret_fwd_120d",
        horizon=120,
        interim_dir=interim_dir,
    )
    pred_long.to_parquet(interim_dir / "pred_long_oos_ml.parquet", index=False)
    logger.info(f"    ✓ 장기 예측: {len(pred_long):,}행, IC={report_long.get('dev_ic_rank_mean', 'N/A'):.4f}")

    # 예측값을 랭킹으로 변환
    ml_short = convert_predictions_to_ranking(pred_short, horizon=20)
    ml_long = convert_predictions_to_ranking(pred_long, horizon=120)
    ml_short.to_parquet(interim_dir / "ranking_ml_short_daily.parquet", index=False)
    ml_long.to_parquet(interim_dir / "ranking_ml_long_daily.parquet", index=False)
    logger.info(f"  ✓ ML 단기 랭킹: {len(ml_short):,}행")
    logger.info(f"  ✓ ML 장기 랭킹: {len(ml_long):,}행")

    # Step 3: 앙상블 랭킹 생성
    logger.info("[Step 3] 앙상블 랭킹 생성")
    ensemble_short = build_ensemble_ranking(
        ranking_baseline=baseline_short,
        ranking_ml=ml_short,
        weight_baseline=weight_baseline,
        weight_ml=weight_ml,
        horizon="short",
    )
    ensemble_long = build_ensemble_ranking(
        ranking_baseline=baseline_long,
        ranking_ml=ml_long,
        weight_baseline=weight_baseline,
        weight_ml=weight_ml,
        horizon="long",
    )
    logger.info(f"  ✓ 앙상블 단기: {len(ensemble_short):,}행")
    logger.info(f"  ✓ 앙상블 장기: {len(ensemble_long):,}행")

    # Step 4: 백테스트 실행 (Track B 동일 로직 사용, 설정 수정 금지)
    logger.info("[Step 4] 백테스트 실행 (Track B 동일 로직, 설정 수정 금지)")

    # Baseline 랭킹 백테스트
    logger.info("  [4-1] Baseline 랭킹 백테스트 (Track B 동일 로직)")
    baseline_results = run_ensemble_backtest_strategies(
        config_path=config_path,
        ranking_type="baseline",
        force_rebuild=force_rebuild,
    )

    # ML 랭킹 백테스트
    logger.info("  [4-2] ML 랭킹 백테스트 (Track B 동일 로직)")
    ml_results = run_ensemble_backtest_strategies(
        config_path=config_path,
        ranking_type="ml",
        force_rebuild=force_rebuild,
    )

    # 앙상블 랭킹 백테스트
    logger.info("  [4-3] 앙상블 랭킹 백테스트 (Track B 동일 로직)")
    ensemble_results = run_ensemble_backtest_strategies(
        config_path=config_path,
        ranking_type="ensemble",
        force_rebuild=force_rebuild,
    )

    # 결과 요약
    logger.info("=" * 80)
    logger.info("백테스트 결과 요약")
    logger.info("=" * 80)

    for ranking_type, results in [
        ("Baseline", baseline_results),
        ("ML", ml_results),
        ("Ensemble", ensemble_results),
    ]:
        logger.info(f"\n[{ranking_type} 랭킹]")
        for strategy, result in results.items():
            metrics = result.get("metrics", {})
            sharpe = metrics.get("sharpe_ratio", "N/A")
            mdd = metrics.get("mdd", "N/A")
            logger.info(f"  {strategy}: Sharpe={sharpe:.2f}, MDD={mdd:.2%}")

    return {
        "baseline": baseline_results,
        "ml": ml_results,
        "ensemble": ensemble_results,
    }

if __name__ == "__main__":
    # XGBoost 우선 시도
    try:
        results = run_full_ensemble_pipeline(
            config_path="configs/config.yaml",
            force_rebuild=False,
            weight_baseline=0.7,
            weight_ml=0.3,
            ml_model_type="xgboost",  # 1순위
        )
    except Exception as e:
        logger.warning(f"XGBoost 실패: {e}")
        logger.info("LightGBM으로 대체 시도...")
        # LightGBM 대체
        results = run_full_ensemble_pipeline(
            config_path="configs/config.yaml",
            force_rebuild=False,
            weight_baseline=0.7,
            weight_ml=0.3,
            ml_model_type="lightgbm",  # 2순위
        )
```

---

## 6. 설정 파일 예시

### 📝 config.yaml 추가 설정

**⚠️ 중요: Baseline과 Track B 설정은 수정하지 않습니다.**

```yaml
# configs/config.yaml

# ⚠️ 기존 설정 보존 (수정 금지)
# l8_short, l8_long: Baseline 랭킹 설정
# l7_bt20_short, l7_bt20_ens, l7_bt120_long, l7_bt120_ens: Track B 백테스트 설정
# l6r: Track B 리밸런싱 설정

# ML 모델 전용 설정 (신규 추가, 기존 l5는 보존)
l5_ml:
  # 모델 타입: XGBoost 우선, LightGBM 대체
  model_type: xgboost  # 또는 lightgbm

  # XGBoost 설정
  xgb_n_estimators: 600
  xgb_max_depth: 4
  xgb_learning_rate: 0.05
  xgb_subsample: 0.8
  xgb_colsample_bytree: 0.8
  xgb_reg_lambda: 1.0
  xgb_min_child_weight: 1.0

  # LightGBM 설정
  lgb_n_estimators: 600
  lgb_max_depth: 4
  lgb_learning_rate: 0.05
  lgb_subsample: 0.8
  lgb_colsample_bytree: 0.8
  lgb_reg_lambda: 1.0
  lgb_min_child_weight: 1.0

  # 공통 설정
  target_transform: cs_rank
  cs_rank_center: true
  random_state: 42
  export_feature_importance: true

# 앙상블 랭킹 설정 (신규 추가)
ensemble_ranking:
  # Baseline 가중치
  weight_baseline: 0.7
  # ML 가중치
  weight_ml: 0.3

  # ML 모델 우선순위
  ml_model_priority: xgboost  # xgboost 우선, lightgbm 대체
```

---

## 7. 실행 방법

### 📋 단계별 실행

```bash
# Step 1: Baseline 랭킹 생성
python -m src.pipeline.track_a_pipeline

# Step 2: ML 모델 학습 및 예측
python scripts/run_pipeline_l0_l7.py

# Step 3: 전체 앙상블 파이프라인 실행
python scripts/run_ensemble_ranking_pipeline.py
```

### 📋 원클릭 실행

```bash
# 전체 파이프라인 한 번에 실행
python scripts/run_ensemble_ranking_pipeline.py
```

---

## 8. 산출물 구조

### 📊 파일 구조

```
data/interim/
├── ranking_short_daily.parquet          # Baseline 단기
├── ranking_long_daily.parquet           # Baseline 장기
├── pred_short_oos.parquet               # ML 예측 (단기)
├── pred_long_oos.parquet                # ML 예측 (장기)
├── ranking_ml_short_daily.parquet        # ML 랭킹 (단기)
├── ranking_ml_long_daily.parquet        # ML 랭킹 (장기)
├── ranking_ensemble_short_daily.parquet # 앙상블 랭킹 (단기)
└── ranking_ensemble_long_daily.parquet   # 앙상블 랭킹 (장기)

data/processed/
├── bt_metrics_baseline_bt20_short.parquet
├── bt_metrics_baseline_bt20_ens.parquet
├── bt_metrics_baseline_bt120_long.parquet
├── bt_metrics_baseline_bt120_ens.parquet
├── bt_metrics_ml_bt20_short.parquet
├── bt_metrics_ml_bt20_ens.parquet
├── bt_metrics_ml_bt120_long.parquet
├── bt_metrics_ml_bt120_ens.parquet
├── bt_metrics_ensemble_bt20_short.parquet
├── bt_metrics_ensemble_bt20_ens.parquet
├── bt_metrics_ensemble_bt120_long.parquet
└── bt_metrics_ensemble_bt120_ens.parquet
```

---

## 9. 기존 투트랙 모델 백테스트 결과 (Baseline 기준)

**실행 환경**: 06_code22 워크스페이스
**실행 일시**: 2026-01-07
**백테스트 방식**: Track B 파이프라인 (L6R → L7)

### 📊 실제 백테스트 결과 (Dev 구간)

**테스트 기간**: 2016-01-04 ~ 2022-12-29 (Dev 구간)
**리밸런싱 횟수**: 87회

| 전략 | Net Sharpe | Net CAGR | Net MDD | Net Hit Ratio | Rank IC | ICIR | Avg Turnover | Profit Factor | Calmar Ratio |
|------|-----------|----------|---------|--------------|---------|------|-------------|---------------|-------------|
| **bt20_short** | -0.012 | -1.04% | -29.75% | 48.28% | -0.051 | -1.51 | 60.61% | 0.99 | -0.035 |
| **bt20_ens** | 0.143 | 1.03% | -37.04% | 43.68% | -0.052 | -1.44 | 53.20% | 1.12 | 0.028 |
| **bt120_long** | 0.314 | 4.78% | -21.97% | 50.57% | -0.044 | -1.19 | 17.04% | 1.46 | 0.218 |
| **bt120_ens** | 0.355 | 5.79% | -23.03% | 54.02% | -0.052 | -1.44 | 18.76% | 1.55 | 0.251 |

### 📊 실제 백테스트 결과 (Holdout 구간)

**테스트 기간**: 2023-01-31 ~ 2024-11-18 (Holdout 구간)
**리밸런싱 횟수**: 23회

| 전략 | Net Sharpe | Net CAGR | Net MDD | Net Hit Ratio | Rank IC | ICIR | Avg Turnover | Profit Factor | Calmar Ratio |
|------|-----------|----------|---------|--------------|---------|------|-------------|---------------|-------------|
| **bt20_short** | -0.355 | -7.26% | -18.68% | 52.17% | 0.009 | 0.25 | 62.17% | 0.77 | -0.389 |
| **bt20_ens** | -0.161 | -4.60% | -16.95% | 52.17% | 0.014 | 0.35 | 55.59% | 0.89 | -0.271 |
| **bt120_long** | 0.569 | 6.86% | -10.27% | 60.87% | 0.013 | 0.27 | 14.90% | 1.50 | 0.668 |
| **bt120_ens** | 0.460 | 5.04% | -9.65% | 60.87% | 0.014 | 0.35 | 16.77% | 1.38 | 0.522 |

### 📋 주요 설정값 (config.yaml)

#### L6R 설정 (랭킹 스코어 변환)
```yaml
l6r:
  alpha_short: 0.5  # 단기:장기 5:5 결합 (ens 전략용)
  rebalance_interval: 1  # 기본값 (실제로는 l7_* 설정의 rebalance_interval 사용)
  regime_alpha:
    bull_strong: 0.6  # Bull 시장에서 단기 가중치 증가
    bull_weak: 0.6
    neutral: 0.5
    bear_weak: 0.4  # Bear 시장에서 단기 가중치 감소
    bear_strong: 0.4
```

**⚠️ 중요: L6R 설정은 수정하지 않습니다.**

**실제 사용**: 각 전략의 `rebalance_interval` 설정이 우선 적용됨
- bt20_short: rebalance_interval=20
- bt20_ens: rebalance_interval=20
- bt120_long: rebalance_interval=20
- bt120_ens: rebalance_interval=20

#### bt20_short 설정
```yaml
l7_bt20_short:
  holding_days: 20
  top_k: 12
  cost_bps: 10.0
  buffer_k: 15
  weighting: equal
  score_col: score_total_short  # 단기 랭킹만 사용
  return_col: true_short
  rebalance_interval: 20
  regime:
    enabled: true
    neutral_band: 0.0  # Bull/Bear만 사용
    top_k_bull_strong: 10
    top_k_bear_strong: 20
    exposure_bull_strong: 1.5
    exposure_bear_strong: 0.6
```

#### bt20_ens 설정
```yaml
l7_bt20_ens:
  holding_days: 20
  top_k: 15
  cost_bps: 10.0
  buffer_k: 20
  weighting: softmax
  softmax_temperature: 0.5
  score_col: score_ens  # 단기:장기 5:5 결합
  return_col: true_short
  rebalance_interval: 20
  regime:
    enabled: true
    neutral_band: 0.0
    top_k_bull_strong: 10
    top_k_bear_strong: 20
    exposure_bull_strong: 1.5
    exposure_bear_strong: 0.6
```

#### bt120_long 설정
```yaml
l7_bt120_long:
  holding_days: 20  # 오버래핑 트랜치: 월별 평가
  top_k: 15
  cost_bps: 10.0
  buffer_k: 15
  weighting: equal
  score_col: score_total_long  # 장기 랭킹만 사용
  return_col: true_short
  rebalance_interval: 20
  overlapping_tranches_enabled: true
  tranche_holding_days: 120  # 각 트랜치 120일 보유
  tranche_max_active: 4  # 최대 4개 트랜치
  regime:
    enabled: true
    neutral_band: 0.05
    top_k_bull_strong: 12
    top_k_bear_strong: 30
    exposure_bull_strong: 1.3
    exposure_bear_strong: 0.7
```

#### bt120_ens 설정
```yaml
l7_bt120_ens:
  holding_days: 20  # 오버래핑 트랜치: 월별 평가
  top_k: 20
  cost_bps: 10.0
  buffer_k: 15
  weighting: equal
  score_col: score_ens  # 단기:장기 5:5 결합
  return_col: true_short
  rebalance_interval: 20
  overlapping_tranches_enabled: true
  tranche_holding_days: 120
  tranche_max_active: 4
  regime:
    enabled: true
    neutral_band: 0.05
    top_k_bull_strong: 12
    top_k_bear_strong: 30
    exposure_bull_strong: 1.3
    exposure_bear_strong: 0.7
```

### 📊 성과 분석 요약

#### Dev 구간 (2016-2022, 87회 리밸런싱)
- **최고 성과**: bt120_ens (Net Sharpe 0.355, Net CAGR 5.79%, Net Calmar 0.251)
- **안정성**: bt120_long (Net MDD -21.97%, 가장 낮은 MDD)
- **단기 전략**: bt20_ens가 bt20_short보다 우수 (Net Sharpe 0.143 vs -0.012)
- **IC 성과**: 모든 전략에서 음수 IC (예측력 제한적)

#### Holdout 구간 (2023-2024, 23회 리밸런싱)
- **최고 성과**: bt120_long (Net Sharpe 0.569, Net CAGR 6.86%, Net Calmar 0.668)
- **안정성**: bt120_ens (Net MDD -9.65%, 가장 낮은 MDD)
- **단기 전략**: 모두 음수 수익률 (시장 환경 영향)
- **IC 성과**: 양수 IC 확인 (bt20_ens: Rank IC 0.014, bt120_long: Rank IC 0.013, bt120_ens: Rank IC 0.014)

#### 주요 인사이트
1. **장기 전략 우수**: bt120_long과 bt120_ens가 단기 전략보다 성과 우수
2. **Holdout 성과**: Dev 대비 Holdout에서 장기 전략의 성과가 더 우수
3. **Rank IC**: Holdout에서 양수 IC 확인 (예측력 향상)
4. **Turnover**: 장기 전략의 턴오버가 현저히 낮음 (14-19% vs 53-62%)
5. **Profit Factor**: 장기 전략이 1.5 이상으로 우수 (단기 전략은 1.0 미만)
6. **Hit Ratio**: Holdout에서 장기 전략이 60.87%로 우수 (단기 전략은 52.17%)

### 📋 상세 메트릭 (Dev 구간)

| 전략 | Net Total Return | Net CAGR | Net Vol | Net Sharpe | Net MDD | Net Calmar | Hit Ratio | Rank IC | ICIR | Avg Turnover | Profit Factor | Avg Trade Duration |
|------|-----------------|----------|---------|-----------|---------|-----------|-----------|---------|------|-------------|---------------|-------------------|
| **bt20_short** | -7.01% | -1.04% | 13.46% | -0.012 | -29.75% | -0.035 | 48.28% | -0.051 | -1.51 | 60.61% | 0.99 | 29.4일 |
| **bt20_ens** | 7.40% | 1.03% | 18.44% | 0.143 | -37.04% | 0.028 | 43.68% | -0.052 | -1.44 | 53.20% | 1.12 | 29.5일 |
| **bt120_long** | 38.58% | 4.78% | 20.69% | 0.314 | -21.97% | 0.218 | 50.57% | -0.044 | -1.19 | 17.04% | 1.46 | 29.7일 |
| **bt120_ens** | 48.15% | 5.79% | 21.24% | 0.355 | -23.03% | 0.251 | 54.02% | -0.052 | -1.44 | 18.76% | 1.55 | 29.7일 |

### 📋 상세 메트릭 (Holdout 구간)

| 전략 | Net Total Return | Net CAGR | Net Vol | Net Sharpe | Net MDD | Net Calmar | Hit Ratio | Rank IC | ICIR | Avg Turnover | Profit Factor | Avg Trade Duration |
|------|-----------------|----------|---------|-----------|---------|-----------|-----------|---------|------|-------------|---------------|-------------------|
| **bt20_short** | -12.68% | -7.26% | 16.90% | -0.355 | -18.68% | -0.389 | 52.17% | 0.009 | 0.25 | 62.17% | 0.77 | 30.0일 |
| **bt20_ens** | -8.11% | -4.60% | 18.50% | -0.161 | -16.95% | -0.271 | 52.17% | 0.014 | 0.35 | 55.59% | 0.89 | 29.9일 |
| **bt120_long** | 12.68% | 6.86% | 12.93% | 0.569 | -10.27% | 0.668 | 60.87% | 0.013 | 0.27 | 14.90% | 1.50 | 29.9일 |
| **bt120_ens** | 9.24% | 5.04% | 12.06% | 0.460 | -9.65% | 0.522 | 60.87% | 0.014 | 0.35 | 16.77% | 1.38 | 29.9일 |

---

## 10. 구현 체크리스트

### ✅ 필수 구현 항목

#### 1. L5 LightGBM 지원 추가
- [ ] `src/stages/modeling/l5_train_models.py`에 LightGBM 모델 추가
- [ ] XGBoost 우선, LightGBM 대체 로직 구현

#### 2. ML 랭킹 변환 함수
- [ ] `src/stages/modeling/l5_to_ranking.py` 파일 생성
- [ ] `convert_predictions_to_ranking()` 함수 구현

#### 3. 앙상블 랭킹 함수
- [ ] `src/stages/modeling/ensemble_ranking.py` 파일 생성
- [ ] `build_ensemble_ranking()` 함수 구현

#### 4. 백테스트 실행 함수
- [ ] `src/pipeline/run_ensemble_backtest.py` 파일 생성
- [ ] `run_ensemble_backtest_strategies()` 함수 구현
- [ ] Track B의 기존 백테스트 로직 사용 (설정 수정 금지)

#### 5. 통합 파이프라인 스크립트
- [ ] `scripts/run_ensemble_ranking_pipeline.py` 파일 생성
- [ ] `run_full_ensemble_pipeline()` 함수 구현
- [ ] XGBoost 우선, LightGBM 대체 로직 구현

#### 6. 설정 파일
- [ ] `configs/config.yaml`에 `l5_ml` 설정 추가
- [ ] `configs/config.yaml`에 `ensemble_ranking` 설정 추가
- [ ] ⚠️ 기존 Baseline/Track B 설정 수정 금지

#### 7. 테스트 및 검증
- [ ] 단위 테스트 작성
- [ ] 통합 테스트 실행
- [ ] 성과 비교 리포트 생성

### ⚠️ 주의사항

1. **Baseline 설정 수정 금지**: `l8_short`, `l8_long` 설정 보존
2. **Track B 설정 수정 금지**: `l7_bt20_*`, `l7_bt120_*`, `l6r` 설정 보존
3. **동일한 백테스트 로직**: 모든 랭킹은 Track B의 동일한 함수 사용
4. **L5 완전 교체**: 기존 L5는 ML 모델 전용으로 교체 (Ridge 제거)
5. **XGBoost 우선**: XGBoost 실패 시에만 LightGBM 사용

---

---

## 11. 참고: 기존 투트랙 모델 백테스트 실행 방법

### 📋 백테스트 재실행 (06_code22)

**실행 환경**: 06_code22 워크스페이스
**실행 일시**: 2026-01-07
**실행 명령어**:

```bash
# 06_code22 디렉토리에서 실행
cd C:\Users\seong\OneDrive\Desktop\bootcamp\06_code22

# 각 전략별로 실행
python -m src.pipeline.track_b_pipeline bt20_short
python -m src.pipeline.track_b_pipeline bt20_ens
python -m src.pipeline.track_b_pipeline bt120_long
python -m src.pipeline.track_b_pipeline bt120_ens
```

### 📊 결과 확인

백테스트 결과는 다음 위치에 저장됩니다:
- `06_code22/data/interim/bt_metrics_bt20_short.parquet`
- `06_code22/data/interim/bt_metrics_bt20_ens.parquet`
- `06_code22/data/interim/bt_metrics_bt120_long.parquet`
- `06_code22/data/interim/bt_metrics_bt120_ens.parquet`

각 파일에는 Dev/Holdout 구간별 메트릭이 포함되어 있습니다.

### 📋 실행 로그 요약

**실행 성공**: 4개 전략 모두 성공적으로 실행됨
- bt20_short: Dev 87회, Holdout 23회 리밸런싱
- bt20_ens: Dev 87회, Holdout 23회 리밸런싱
- bt120_long: Dev 87회, Holdout 23회 리밸런싱
- bt120_ens: Dev 87회, Holdout 23회 리밸런싱

**시장 국면 분포** (bt20 전략 기준):
- Bull: 70개 (63.1%)
- Neutral: 2개 (1.8%)
- Bear: 39개 (35.1%)

**시장 국면 분포** (bt120 전략 기준):
- Bull: 26개 (23.4%)
- Neutral: 64개 (57.7%)
- Bear: 21개 (18.9%)

---

**작성일**: 2026-01-07
**작성자**: Cursor AI
**버전**: 1.3 (Phase 2 Grid Search 최적화 결과 반영)

**최종 업데이트**: 2026-01-08
- Phase 2 Grid Search 최적화 완료 (80개 조합 평가)
- 최적 그룹별 가중치 확인: technical=-0.5, value=0.5
- IC 양수 전환 확인 (IC Mean: 0.0200, ICIR: 0.2224)
- 최적 가중치 파일 생성: `feature_groups_short_optimized_grid_20260108_121838.yaml`

**이전 업데이트**: 2026-01-07
- 06_code22에서 4개 전략 백테스트 재실행 완료
- Dev/Holdout 구간별 상세 메트릭 반영
- 설정값 상세 반영
