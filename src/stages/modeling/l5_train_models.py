# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/stages/modeling/l5_train_models.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
import yaml
from sklearn.metrics import r2_score

# [Stage6] 피처 그룹 밸런싱 유틸리티
from src.utils.feature_groups import (
    calculate_feature_group_balance,
    load_feature_groups,
)


@dataclass(frozen=True)
class FoldSpec:
    fold_id: str
    phase: str  # "dev" | "holdout"
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="raise")


def _infer_phase(phase_raw: str, fold_id: str) -> str:
    """
    [Phase 6] Phase 추론: dev, validation, holdout 구분
    """
    ph = (phase_raw or "").strip().lower()
    if ph in ("dev", "validation", "holdout"):
        return ph
    fid = (fold_id or "").strip().lower()
    if fid.startswith("dev"):
        return "dev"
    if fid.startswith("validation") or fid.startswith("val"):
        return "validation"
    if fid.startswith("holdout") or fid.startswith("test"):
        return "holdout"
    # phase가 없거나 예상 외면 dev로 통일 (기본값)
    return "dev"


def _standardize_folds(folds: pd.DataFrame) -> list[FoldSpec]:
    if folds is None or not isinstance(folds, pd.DataFrame) or folds.empty:
        raise ValueError("cv_folds is empty or not a DataFrame.")

    phase_col = (
        "phase"
        if "phase" in folds.columns
        else ("segment" if "segment" in folds.columns else None)
    )
    required = ["fold_id", "train_start", "train_end", "test_start", "test_end"]
    missing = [c for c in required if c not in folds.columns]
    if phase_col is None:
        missing.append("phase(or segment)")
    if missing:
        raise ValueError(
            f"cv_folds schema missing columns: {missing}. got={list(folds.columns)}"
        )

    f = folds.copy()
    f["train_start"] = _to_datetime(f["train_start"])
    f["train_end"] = _to_datetime(f["train_end"])
    f["test_start"] = _to_datetime(f["test_start"])
    f["test_end"] = _to_datetime(f["test_end"])

    # 날짜 정합성 체크
    bad = f[(f["train_start"] > f["train_end"]) | (f["test_start"] > f["test_end"])]
    if not bad.empty:
        ex = bad.head(5)
        raise ValueError(f"cv_folds has invalid date ranges. examples:\n{ex}")

    specs: list[FoldSpec] = []
    for _, r in f.iterrows():
        fid = str(r["fold_id"]).strip()
        ph = _infer_phase(str(r[phase_col]), fid)
        specs.append(
            FoldSpec(
                fold_id=fid,
                phase=ph,
                train_start=pd.Timestamp(r["train_start"]),
                train_end=pd.Timestamp(r["train_end"]),
                test_start=pd.Timestamp(r["test_start"]),
                test_end=pd.Timestamp(r["test_end"]),
            )
        )
    return specs


def _pick_feature_cols(
    df: pd.DataFrame, *, target_col: str, cfg: dict = None, horizon: int = None
) -> list[str]:
    # [Phase 6] 피처 고정 모드 우선 적용
    if cfg is not None:
        l5 = (cfg.get("l5", {}) if isinstance(cfg, dict) else {}) or {}
        feature_list_short = l5.get("feature_list_short")
        feature_list_long = l5.get("feature_list_long")

        # horizon이 없으면 target_col에서 추론 시도
        if horizon is None:
            if "20d" in target_col.lower() or "short" in target_col.lower():
                horizon = 20
            elif "120d" in target_col.lower() or "long" in target_col.lower():
                horizon = 120

        # 피처 고정 모드: YAML 파일에서 피처 리스트 읽기
        feature_list_path = None
        if horizon == 20 and feature_list_short:
            feature_list_path = feature_list_short
        elif horizon == 120 and feature_list_long:
            feature_list_path = feature_list_long

        if feature_list_path:
            import logging

            import yaml

            logger = logging.getLogger(__name__)

            try:
                base_dir = Path(cfg.get("paths", {}).get("base_dir", "."))
                feature_path = base_dir / feature_list_path

                if feature_path.exists():
                    with open(feature_path, encoding="utf-8") as f:
                        feature_config = yaml.safe_load(f) or {}
                        fixed_features = feature_config.get("features", [])

                        if not isinstance(fixed_features, list):
                            logger.warning(
                                f"[Phase 6] feature_list format error: expected list, got {type(fixed_features)}"
                            )
                        else:
                            # df에 존재하는 피처만 필터링
                            available = [f for f in fixed_features if f in df.columns]
                            missing = [f for f in fixed_features if f not in df.columns]

                            if len(available) > 0:
                                logger.info(
                                    f"[Phase 6] 피처 고정 모드: {len(available)}/{len(fixed_features)}개 피처 사용 (horizon={horizon})"
                                )
                                if missing:
                                    logger.warning(
                                        f"[Phase 6] 누락된 피처 ({len(missing)}개): {missing[:10]}{'...' if len(missing) > 10 else ''}"
                                    )
                                return available
                            else:
                                logger.warning(
                                    "[Phase 6] 피처 고정 모드: 사용 가능한 피처가 없음. 기존 로직으로 fallback"
                                )
            except Exception as e:
                import logging

                logger = logging.getLogger(__name__)
                logger.warning(
                    f"[Phase 6] 피처 고정 모드 로드 실패: {e}. 기존 로직으로 fallback"
                )

    # [Phase 5 개선안] Market-Neutral: 초과 수익률 컬럼도 피처에서 제외
    exclude = {
        "date",
        "ticker",
        target_col,
        "ret_fwd_20d",
        "ret_fwd_120d",  # 절대 수익률
        "ret_fwd_20d_excess",
        "ret_fwd_120d_excess",  # 초과 수익률
        "market_ret_20d",
        "market_ret_120d",  # 시장 수익률 (참고용)
        "split",
        "phase",
        "segment",
        "fold_id",
    }
    cols: list[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)

    if not cols:
        raise ValueError(
            "No numeric feature columns found after excluding identifiers/targets."
        )

    # [Phase 5 개선안] 피처 IC 필터링: IC 또는 Rank IC < min_feature_ic인 피처 제외
    # [Phase 6] IC 필터링은 피처 고정 모드가 없을 때만 적용
    if cfg is not None:
        l5 = (cfg.get("l5", {}) if isinstance(cfg, dict) else {}) or {}
        filter_by_ic = l5.get("filter_features_by_ic", False)
        use_rank_ic = l5.get(
            "use_rank_ic", False
        )  # [Phase 5 개선안] Rank IC 기준 필터링
        min_feature_ic = float(l5.get("min_feature_ic", 0.01))
        feature_ic_file = l5.get(
            "feature_ic_file", "artifacts/reports/feature_ic_dev.csv"
        )

        if filter_by_ic:
            import logging

            logger = logging.getLogger(__name__)

            # IC 파일 경로 계산
            base_dir = Path(cfg.get("paths", {}).get("base_dir", "."))
            ic_path = base_dir / feature_ic_file

            if ic_path.exists():
                try:
                    ic_df = pd.read_csv(ic_path)
                    logger.info(f"[Phase 5] Loading feature IC from {ic_path}")
                    logger.info(f"[Phase 5] Total features in IC file: {len(ic_df)}")

                    # [Phase 5 개선안] Rank IC 기준 필터링
                    if use_rank_ic:
                        ic_col = "rank_ic"
                        logger.info(
                            f"[Phase 5] Using Rank IC for filtering (Rank IC < {min_feature_ic})"
                        )
                    else:
                        ic_col = "ic"
                        logger.info(
                            f"[Phase 5] Using IC for filtering (IC < {min_feature_ic})"
                        )

                    # IC 또는 Rank IC < min_feature_ic인 피처 제외
                    if ic_col not in ic_df.columns:
                        logger.warning(
                            f"[Phase 5] Column '{ic_col}' not found in IC file. Using 'ic' instead."
                        )
                        ic_col = "ic"

                    bad_features = set(
                        ic_df[ic_df[ic_col] < min_feature_ic]["feature"].tolist()
                    )
                    original_count = len(cols)
                    cols = [c for c in cols if c not in bad_features]
                    filtered_count = len(cols)

                    logger.info(
                        f"[Phase 5] Feature filtering ({ic_col}): {original_count} -> {filtered_count} (removed {original_count - filtered_count} features with {ic_col} < {min_feature_ic})"
                    )

                    if filtered_count == 0:
                        raise ValueError(
                            f"All features were filtered out! Check min_feature_ic={min_feature_ic} and feature_ic_file={feature_ic_file}"
                        )
                except Exception as e:
                    logger.warning(
                        f"[Phase 5] Failed to load/apply feature IC filter: {e}. Using all features."
                    )
            else:
                logger.warning(
                    f"[Phase 5] Feature IC file not found: {ic_path}. IC filtering disabled. Run scripts/calculate_feature_ic.py first."
                )

    return cols


def _rank_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    s1 = pd.Series(y_true).rank(pct=True)
    s2 = pd.Series(y_pred).rank(pct=True)
    v = float(s1.corr(s2))
    return 0.0 if np.isnan(v) else v


def _get_l5_cfg(cfg: dict) -> dict:
    return (cfg.get("l5", {}) if isinstance(cfg, dict) else {}) or {}


def _cs_rank_by_date(d: pd.DataFrame, col: str, *, center: bool = True) -> np.ndarray:
    """
    date별 cross-sectional rank(pct) 변환.
    - NaN은 사전에 dropna되어 있어야 함.
    - center=True면 [-0.5, 0.5] 범위로 0 중심화
    """
    r = d.groupby("date")[col].rank(pct=True)
    if center:
        r = r - 0.5
    return r.to_numpy(dtype=np.float32, copy=False)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    ic = _rank_ic(y_true, y_pred)
    hit = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
    # [Stage 2] r2_oos 추가 (y_true와 y_pred가 동일 스케일에서 계산)
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "ic_rank": ic, "hit_ratio": hit, "r2_oos": r2}


def _build_model(cfg: dict, alpha: float = None) -> tuple[Pipeline, str]:
    l5 = _get_l5_cfg(cfg)

    model_type = str(l5.get("model_type", "ridge")).lower()
    target_transform = str(l5.get("target_transform", "none")).lower()
    cs_rank_center = bool(l5.get("cs_rank_center", True))

    if alpha is None:
        ridge_alpha = float(l5.get("ridge_alpha", 1.0))
    else:
        ridge_alpha = float(alpha)

    tf = f"{target_transform}{'_center' if (target_transform=='cs_rank' and cs_rank_center) else ''}"

    # [개선안 23번] 모델 확장: ridge + random_forest + xgboost (옵션)
    if model_type == "ridge":
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=True)),
                ("model", Ridge(alpha=ridge_alpha)),
            ]
        )
    return pipe, f"ridge(alpha={ridge_alpha}, target_transform={tf})"

    if model_type == "ensemble":
        # [앙상블 모드] 여러 모델 결합
        models = {}

        # Ridge 모델
        ridge_model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=True)),
                ("model", Ridge(alpha=ridge_alpha)),
            ]
        )
        models["ridge"] = ridge_model

        # XGBoost 모델 (사용 가능한 경우)
        if XGBOOST_AVAILABLE:
            xgboost_model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler(with_mean=True)),
                    (
                        "model",
                        XGBRegressor(
                            n_estimators=300,
                            max_depth=5,
                            learning_rate=0.03,
                            random_state=42,
                            subsample=0.9,
                            colsample_bytree=0.9,
                            gamma=0.1,
                        ),
                    ),
                ]
            )
            models["xgboost"] = xgboost_model

        # Grid Search 모델 (간단한 버전)
        grid_model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=True)),
                ("model", Ridge(alpha=ridge_alpha)),  # 일단 Ridge로 구현
            ]
        )
        models["grid"] = grid_model

        # Random Forest 모델
        rf_model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=100, max_depth=5, random_state=42
                    ),
                ),
            ]
        )
        models["rf"] = rf_model

        return models, f"ensemble(models={list(models.keys())}, target_transform={tf})"

    if model_type in ("rf", "random_forest", "randomforest"):
        n_estimators = int(l5.get("rf_n_estimators", 400))
        max_depth = l5.get("rf_max_depth", None)
        max_depth = None if max_depth in (None, "none", "null") else int(max_depth)
        min_samples_leaf = int(l5.get("rf_min_samples_leaf", 5))
        random_state = int(l5.get("random_state", 42))

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1,
            random_state=random_state,
        )
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", model),
            ]
        )
        return (
            pipe,
            f"random_forest(n_estimators={n_estimators}, max_depth={max_depth}, leaf={min_samples_leaf}, target_transform={tf})",
        )

    if model_type in ("xgb", "xgboost"):
        try:
            from xgboost import XGBRegressor
        except Exception as e:
            raise ImportError(
                "xgboost가 필요합니다. `pip install xgboost` 후 재실행하세요."
            ) from e

        # 안전한 기본값(과적합 완화 방향)
        n_estimators = int(l5.get("xgb_n_estimators", 600))
        max_depth = int(l5.get("xgb_max_depth", 4))
        learning_rate = float(l5.get("xgb_learning_rate", 0.05))
        subsample = float(l5.get("xgb_subsample", 0.8))
        colsample_bytree = float(l5.get("xgb_colsample_bytree", 0.8))
        reg_lambda = float(l5.get("xgb_reg_lambda", 1.0))
        min_child_weight = float(l5.get("xgb_min_child_weight", 1.0))
        random_state = int(l5.get("random_state", 42))

        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            objective="reg:squarederror",
            n_jobs=-1,
            random_state=random_state,
        )
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", model),
            ]
        )
        return (
            pipe,
            f"xgboost(n_estimators={n_estimators}, depth={max_depth}, lr={learning_rate}, target_transform={tf})",
        )

    raise ValueError(
        f"Unsupported model_type={model_type}. (supported: ridge, random_forest, xgboost)"
    )


def _select_ridge_alpha_by_time_split(
    *,
    dtrain: pd.DataFrame,
    use_cols: list[str],
    target_col: str,
    cfg: dict,
    horizon: int,
) -> tuple[Optional[float], Optional[float], list[str]]:
    """
    [개선안 19번] alpha 튜닝(옵션) - 단순 Time-split 검증(lookahead 방지)

    정책:
      - fold의 train 기간 내부에서만 검증한다.
      - train의 마지막 N 거래일을 validation으로 사용한다. (N = l4.test_window_days, 기본 20)
      - 목표: validation ic_rank 최대화

    Returns:
      (best_alpha, best_val_ic, warns)
      - best_alpha가 None이면 튜닝 스킵(데이터 부족/설정 없음)
    """
    warns: list[str] = []
    l5 = _get_l5_cfg(cfg)
    if not bool(l5.get("tune_alpha", False)):
        return None, None, warns

    alpha_grid = l5.get("alpha_grid", None)
    if not isinstance(alpha_grid, list) or len(alpha_grid) == 0:
        warns.append("[L5 alpha_tune] tune_alpha=True but alpha_grid is empty -> skip")
        return None, None, warns

    l4 = (
        (cfg.get("l4", {}) if isinstance(cfg, dict) else {})
        or (cfg.get("params", {}).get("l4", {}) if isinstance(cfg, dict) else {})
        or {}
    )
    val_window_days = int(l4.get("test_window_days", 20))
    if val_window_days <= 0:
        warns.append("[L5 alpha_tune] test_window_days<=0 -> skip")
        return None, None, warns

    if "date" not in dtrain.columns:
        warns.append("[L5 alpha_tune] dtrain missing date -> skip")
        return None, None, warns

    # unique train dates (sorted)
    dates = pd.to_datetime(dtrain["date"], errors="coerce").dropna().unique()
    dates = pd.DatetimeIndex(dates).sort_values()
    if len(dates) < (val_window_days + 2):
        warns.append(
            f"[L5 alpha_tune] too few train dates={len(dates)} for val_window_days={val_window_days} -> skip"
        )
        return None, None, warns

    cutoff = dates[-(val_window_days + 1)]
    subtrain = dtrain.loc[pd.to_datetime(dtrain["date"]) <= cutoff].copy()
    val = dtrain.loc[pd.to_datetime(dtrain["date"]) > cutoff].copy()

    if len(subtrain) < 500 or len(val) < 200:
        warns.append(
            f"[L5 alpha_tune] too few rows subtrain={len(subtrain)}, val={len(val)} -> skip"
        )
        return None, None, warns

    target_transform = str(l5.get("target_transform", "none")).lower()
    cs_rank_center = bool(l5.get("cs_rank_center", True))

    X_sub = subtrain[use_cols].to_numpy(dtype=np.float32, copy=False)
    X_val = val[use_cols].to_numpy(dtype=np.float32, copy=False)

    y_sub_raw = subtrain[target_col].to_numpy(dtype=np.float32, copy=False)
    y_val_raw = val[target_col].to_numpy(dtype=np.float32, copy=False)

    if target_transform == "cs_rank":
        y_sub = _cs_rank_by_date(subtrain, target_col, center=cs_rank_center)
        y_val_metric = _cs_rank_by_date(val, target_col, center=cs_rank_center)
    else:
        y_sub = y_sub_raw
        y_val_metric = y_val_raw

    # [Alpha 튜닝 전략 개선] 복합 메트릭 사용: IC + Hit Ratio + Regularization 페널티
    tune_metric = str(l5.get("tune_metric", "ic_rank")).lower()
    use_composite_metric = tune_metric in ("composite", "ic_hit", "ic_penalty")

    best_alpha: Optional[float] = None
    best_score: Optional[float] = None
    alpha_scores = []  # 디버깅용

    for a in alpha_grid:
        try:
            alpha = float(a)
            if not np.isfinite(alpha) or alpha <= 0:
                continue
        except Exception:
            continue

        try:
            model, _ = _build_model(cfg, alpha=alpha)
            model.fit(X_sub, y_sub)
            pred = model.predict(X_val).astype(np.float32)
            ic = _rank_ic(y_val_metric, pred)

            # Hit Ratio 계산
            hit_ratio = float(np.mean(np.sign(y_val_metric) == np.sign(pred)))

            # 복합 메트릭 계산
            if use_composite_metric:
                # Regularization 페널티: Phase 2 기본값(1.0)에서 멀수록 페널티
                penalty_weight = float(l5.get("alpha_penalty_weight", 0.1))
                penalty = abs(np.log(alpha / 1.0)) * penalty_weight

                if tune_metric == "ic_hit":
                    # IC * Hit Ratio 조합
                    composite_score = ic * hit_ratio
                elif tune_metric == "ic_penalty":
                    # IC - Regularization 페널티
                    composite_score = ic - penalty
                else:  # composite (기본)
                    # IC * Hit Ratio - Regularization 페널티
                    composite_score = ic * hit_ratio - penalty

                score = composite_score
            else:
                # 기존 방식: IC만 사용
                score = ic

            alpha_scores.append(
                {"alpha": alpha, "ic": ic, "hit_ratio": hit_ratio, "score": score}
            )

        except Exception as e:
            warns.append(f"[L5 alpha_tune] alpha={a} failed: {type(e).__name__}: {e}")
            continue

        if (best_score is None) or (score > float(best_score)):
            best_score = float(score)
            best_alpha = float(alpha)

    if best_alpha is None:
        warns.append("[L5 alpha_tune] no valid alpha evaluated -> skip")
        return None, None, warns

    # 선택된 alpha의 IC (로깅용)
    best_ic = next((s["ic"] for s in alpha_scores if s["alpha"] == best_alpha), None)
    metric_name = tune_metric if use_composite_metric else "ic_rank"
    warns.append(
        f"[L5 alpha_tune] selected alpha={best_alpha} (val_{metric_name}={best_score:.4f}, val_ic_rank={best_ic:.4f}, horizon={horizon})"
    )

    # 상위 3개 alpha 로깅 (디버깅용)
    if len(alpha_scores) > 0:
        top3 = sorted(alpha_scores, key=lambda x: x["score"], reverse=True)[:3]
        top3_str = ", ".join(
            [f"α={s['alpha']:.1f}(score={s['score']:.4f})" for s in top3]
        )
        warns.append(f"[L5 alpha_tune] top 3 alphas: {top3_str}")

    return best_alpha, best_ic, warns


def _slice_by_date_sorted(
    df: pd.DataFrame, date_arr: np.ndarray, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    # df는 date 기준 오름차순 정렬되어 있어야 함
    left = np.searchsorted(date_arr, np.datetime64(start), side="left")
    right = np.searchsorted(date_arr, np.datetime64(end), side="right")
    if right <= left:
        return df.iloc[0:0]
    return df.iloc[left:right]


def train_oos_predictions(
    *,
    dataset_daily: pd.DataFrame,
    cv_folds: pd.DataFrame,
    cfg: dict,
    target_col: str,
    horizon: int,
    interim_dir: Optional[Path] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, list[str]]:
    warns: list[str] = []

    df = dataset_daily.copy()

    # 필수 컬럼 체크
    for c in ["date", "ticker", target_col]:
        if c not in df.columns:
            raise ValueError(f"dataset_daily missing required column: {c}")

    # date dtype 확정 + 정렬 확정
    df["date"] = pd.to_datetime(df["date"], errors="raise")
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)
    if not df["date"].is_monotonic_increasing:
        # sort 후에도 단조가 아니면 이상 (동일 date는 ticker로만 정렬)
        raise ValueError("dataset_daily date is not monotonic increasing after sort.")

    date_arr = df["date"].to_numpy(dtype="datetime64[ns]", copy=False)

    # [Phase 6] horizon 정보를 _pick_feature_cols에 전달
    feature_cols = _pick_feature_cols(
        df, target_col=target_col, cfg=cfg, horizon=horizon
    )
    fold_specs_all = _standardize_folds(cv_folds)

    # [피쳐 가중치 적용] 피쳐별 가중치 로드 (horizon별)
    feature_weights = None
    l5_cfg = _get_l5_cfg(cfg)

    # horizon에 따라 다른 가중치 파일 사용
    if horizon == 20:
        feature_weights_config = l5_cfg.get("feature_weights_config_short")
    elif horizon == 120:
        feature_weights_config = l5_cfg.get("feature_weights_config_long")
    else:
        feature_weights_config = l5_cfg.get("feature_weights_config")

    if feature_weights_config:
        try:
            base_dir = Path(cfg.get("paths", {}).get("base_dir", "."))
            weights_path = base_dir / feature_weights_config

            if weights_path.exists():
                with open(weights_path, encoding="utf-8") as f:
                    weights_data = yaml.safe_load(f) or {}
                    feature_weights = weights_data.get("feature_weights", {})
                    warns.append(
                        f"[L5 피쳐 가중치] 로드 완료: {len(feature_weights)}개 피쳐, horizon={horizon} ({weights_path})"
                    )
            else:
                warns.append(f"[L5 피쳐 가중치] 파일 없음: {weights_path}")
        except Exception as e:
            warns.append(f"[L5 피쳐 가중치] 로드 실패: {e}")

    l5_cfg = _get_l5_cfg(cfg)
    export_feature_importance = bool(l5_cfg.get("export_feature_importance", False))

    # [원래 상태로 복원] dev/holdout 구분 없이 모든 fold에서 동일하게 학습
    fold_specs = fold_specs_all  # dev/holdout 구분 없이 원래 순서대로 처리

    pred_rows: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    coef_rows: list[dict] = []  # [Stage 2] 계수 저장용

    possible_test_rows = 0
    predicted_rows = 0
    dropped_all_nan_union: set[str] = set()

    for fs in fold_specs:
        dtrain_all = _slice_by_date_sorted(df, date_arr, fs.train_start, fs.train_end)
        dtest_all = _slice_by_date_sorted(df, date_arr, fs.test_start, fs.test_end)

        # [Phase 6] Purge 검증: train_end와 test_start 사이 간격 확인
        gap_days = (fs.test_start - fs.train_end).days
        expected_purge = 20 if horizon == 20 else 120
        if gap_days < expected_purge:
            warns.append(
                f"[L5 PURGE] fold={fs.fold_id}: gap={gap_days} < purge={expected_purge} (horizon={horizon})"
            )

        # target NaN 제거
        dtrain = dtrain_all.dropna(subset=[target_col])
        dtest = dtest_all.dropna(subset=[target_col])

        possible_test_rows += int(dtest.shape[0])

        if dtrain.shape[0] < 2000:
            warns.append(
                f"[L5] fold={fs.fold_id} horizon={horizon}: too few train rows={dtrain.shape[0]}"
            )
            continue
        if dtest.shape[0] < 200:
            warns.append(
                f"[L5] fold={fs.fold_id} horizon={horizon}: too few test rows={dtest.shape[0]}"
            )
            continue

        # ALL-NaN 피처 제거(폴드별)
        use_cols = [c for c in feature_cols if dtrain[c].notna().any()]
        dropped = [c for c in feature_cols if c not in use_cols]
        if dropped:
            dropped_all_nan_union.update(dropped)

        if len(use_cols) < 5:
            warns.append(
                f"[L5] fold={fs.fold_id} horizon={horizon}: too few usable features={len(use_cols)}"
            )
            continue

        l5_cfg = _get_l5_cfg(cfg)
        target_transform = str(l5_cfg.get("target_transform", "none")).lower()
        cs_rank_center = bool(l5_cfg.get("cs_rank_center", True))

        X_train = dtrain[use_cols].to_numpy(dtype=np.float32, copy=False)
        X_test = dtest[use_cols].to_numpy(dtype=np.float32, copy=False)

        # [피쳐 가중치 적용] 피쳐 값에 가중치 곱하기
        if feature_weights:
            weight_array = np.array(
                [feature_weights.get(col, 1.0) for col in use_cols], dtype=np.float32
            )
            # 가중치 정규화 (합=1.0)
            weight_sum = weight_array.sum()
            if weight_sum > 0:
                weight_array = (
                    weight_array / weight_sum * len(use_cols)
                )  # 평균 가중치를 1.0으로 유지
            X_train = X_train * weight_array[np.newaxis, :]
            X_test = X_test * weight_array[np.newaxis, :]

        y_train_raw = dtrain[target_col].to_numpy(dtype=np.float32, copy=False)
        y_test_raw = dtest[target_col].to_numpy(dtype=np.float32, copy=False)

        if target_transform == "cs_rank":
            y_train = _cs_rank_by_date(dtrain, target_col, center=cs_rank_center)
            y_test_metric = _cs_rank_by_date(dtest, target_col, center=cs_rank_center)
        else:
            y_train = y_train_raw
            y_test_metric = y_test_raw

        # [개선안 19번] alpha 튜닝(옵션): fold train 내부 time-split로 선택
        ridge_alpha = float(l5_cfg.get("ridge_alpha", 1.0))
        tuned_alpha, tuned_val_ic, tune_warns = _select_ridge_alpha_by_time_split(
            dtrain=dtrain,
            use_cols=use_cols,
            target_col=target_col,
            cfg=cfg,
            horizon=horizon,
        )
        if tune_warns:
            warns.extend(tune_warns)
        if tuned_alpha is not None:
            ridge_alpha = float(tuned_alpha)

        # 최종 모델 학습 (ridge_alpha 사용)
        model, model_name = _build_model(cfg, alpha=ridge_alpha)

        # [앙상블 모드] 여러 모델 결합
        if isinstance(model, dict) and model_name.startswith("ensemble"):
            # 앙상블 가중치 로드 (l6r에서 로드)
            ensemble_weights_cfg = cfg.get("l6r", {}).get("ensemble_weights", {})
            if horizon == 20:
                ensemble_weights = ensemble_weights_cfg.get("short", {})
            else:  # horizon == 120
                ensemble_weights = ensemble_weights_cfg.get("long", {})

            # 디버그 로그
            warns.append(
                f"[L5 앙상블 디버그] horizon={horizon}, ensemble_weights_cfg 키: {list(ensemble_weights_cfg.keys()) if ensemble_weights_cfg else 'None'}"
            )
            warns.append(f"[L5 앙상블 디버그] 선택된 가중치: {ensemble_weights}")

            # 각 모델 학습 및 예측
            ensemble_predictions = {}
            for model_name_key, model_obj in model.items():
                if (
                    model_name_key in ensemble_weights
                    and ensemble_weights[model_name_key] > 0
                ):
                    try:
                        model_obj.fit(X_train, y_train)
                        pred = model_obj.predict(X_test).astype(np.float32)
                        ensemble_predictions[model_name_key] = pred
                        warns.append(f"[L5 앙상블] {model_name_key} 모델 학습 완료")
                    except Exception as e:
                        warns.append(
                            f"[L5 앙상블] {model_name_key} 모델 학습 실패: {e}"
                        )

            # 가중치 기반 예측 결합
            if ensemble_predictions:
                y_pred = np.zeros_like(list(ensemble_predictions.values())[0])
                total_weight = 0
                weight_details = []
                for model_name_key, pred in ensemble_predictions.items():
                    weight = ensemble_weights.get(model_name_key, 0)
                    y_pred += weight * pred
                    total_weight += weight
                    weight_details.append(f"{model_name_key}:{weight:.2f}")

                if total_weight > 0:
                    y_pred = y_pred / total_weight  # 정규화
                else:
                    # 가중치 합이 0이면 평균 사용
                    y_pred = np.mean(list(ensemble_predictions.values()), axis=0)

                warns.append(
                    f"[L5 앙상블] {len(ensemble_predictions)}개 모델 예측 결합 완료"
                )
                warns.append(
                    f"[L5 앙상블] 가중치 상세: {' + '.join(weight_details)} = {total_weight:.2f}"
                )
                warns.append(
                    f"[L5 앙상블] horizon={horizon}, total_weight={total_weight:.3f}"
                )
            else:
                # 앙상블 예측 실패 시 기본 모델 사용
                default_model = model.get("ridge", list(model.values())[0])
                default_model.fit(X_train, y_train)
                y_pred = default_model.predict(X_test).astype(np.float32)
                warns.append(
                    f"[L5 앙상블] 앙상블 실패로 기본 Ridge 모델 사용 (horizon={horizon})"
                )

            model_for_coef = model.get("ridge", list(model.values())[0])  # 계수 추출용
        else:
            # 기존 단일 모델 로직
            # [Phase 6] 전처리 fit 범위 확인: train만 fit, test는 transform만
            train_date_range = (
                f"{dtrain['date'].min().date()} ~ {dtrain['date'].max().date()}"
            )
            test_date_range = (
                f"{dtest['date'].min().date()} ~ {dtest['date'].max().date()}"
            )
            import logging

            logger = logging.getLogger(__name__)
            logger.debug(
                f"[L5 Phase 6] fold={fs.fold_id}: train={train_date_range}, test={test_date_range}"
            )

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test).astype(np.float32)
            model_for_coef = model

        # [Stage 2] 계수/중요도 추출 및 저장
        if export_feature_importance:
            # 앙상블 모드일 때는 Ridge 모델의 계수 사용
            if isinstance(model, dict):
                base_model = model_for_coef.named_steps.get("model", None)
            else:
                base_model = model.named_steps.get("model", None)

            # [개선안 24번] Ridge 외 모델(RandomForest/XGBoost)은 coef_가 없을 수 있음 -> 안전 분기
            if base_model is None:
                warns.append(
                    "[L5 Stage2] model.named_steps['model'] not found -> skip importance export"
                )
            elif hasattr(base_model, "coef_"):
                coef_array = np.asarray(base_model.coef_, dtype=np.float32)
                for idx, feature_name in enumerate(use_cols):
                    coef_rows.append(
                        {
                            "fold_id": fs.fold_id,
                            "phase": fs.phase,
                            "horizon": int(horizon),
                            "feature": feature_name,
                            "coef": float(coef_array[idx]),
                            "abs_coef": float(np.abs(coef_array[idx])),
                        }
                    )
            elif hasattr(base_model, "feature_importances_"):
                # 트리 계열: feature_importances_를 coef로 저장(호환), coef_sign_stability는 의미가 약함
                imp = np.asarray(base_model.feature_importances_, dtype=np.float32)
                for idx, feature_name in enumerate(use_cols):
                    coef_rows.append(
                        {
                            "fold_id": fs.fold_id,
                            "phase": fs.phase,
                            "horizon": int(horizon),
                            "feature": feature_name,
                            "coef": float(imp[idx]),
                            "abs_coef": float(np.abs(imp[idx])),
                        }
                    )
                warns.append(
                    "[L5 Stage2] tree feature_importances_ exported (stored in coef/abs_coef for compatibility)"
                )
            else:
                warns.append(
                    f"[L5 Stage2] importance export not supported for model={type(base_model).__name__} -> skipped"
                )

        # 지표는 "학습 타깃과 동일 스케일"로 평가
        m = _metrics(y_test_metric, y_pred)

        out = dtest[["date", "ticker"]].copy()
        out["y_true"] = y_test_raw  # raw forward return 유지 (L7 백테스트용)
        out["y_pred"] = y_pred
        if target_transform == "cs_rank":
            out["y_true_rank"] = y_test_metric  # 분석용(선택)
        out["fold_id"] = fs.fold_id
        out["phase"] = fs.phase
        out["horizon"] = int(horizon)
        out["model"] = model_name
        pred_rows.append(out)

        # 메트릭 추가
        metric_row = {
            "fold_id": fs.fold_id,
            "phase": fs.phase,
            "horizon": int(horizon),
            "rmse": m["rmse"],
            "mae": m["mae"],
            "ic_rank": m["ic_rank"],
            "hit_ratio": m["hit_ratio"],
            "r2_oos": m["r2_oos"],  # [Stage 2] r2_oos 추가
            "n_train": int(len(dtrain)),
            "n_test": int(len(dtest)),
            "n_features": int(len(use_cols)),
            # [개선안 19번] alpha 튜닝 진단
            "ridge_alpha_used": float(ridge_alpha),
            "alpha_tuned": bool(tuned_alpha is not None),
            "alpha_val_ic_rank": (
                None if tuned_val_ic is None else float(tuned_val_ic)
            ),
        }
        metric_rows.append(metric_row)

        predicted_rows += int(out.shape[0])

    if not pred_rows:
        raise RuntimeError(f"No OOS predictions generated for horizon={horizon}.")

    pred_oos = pd.concat(pred_rows, ignore_index=True)
    metrics_df = pd.DataFrame(metric_rows)

    # (date,ticker) 유일성 체크
    dup = int(pred_oos.duplicated(subset=["date", "ticker"]).sum())
    if dup > 0:
        raise RuntimeError(f"OOS predictions have duplicate (date,ticker) rows: {dup}")

    coverage = (
        (predicted_rows / possible_test_rows * 100.0) if possible_test_rows > 0 else 0.0
    )

    report: dict = {
        "horizon": int(horizon),
        "target_col": target_col,
        "model": model_name,
        "possible_test_rows": int(possible_test_rows),
        "predicted_rows": int(predicted_rows),
        "oos_coverage_pct": round(float(coverage), 4),
        "folds_total": int(len(fold_specs)),
        "folds_used": (
            int(metrics_df["fold_id"].nunique()) if not metrics_df.empty else 0
        ),
        "dev_folds": (
            int((metrics_df["phase"] == "dev").sum()) if not metrics_df.empty else 0
        ),
        "holdout_folds": (
            int((metrics_df["phase"] == "holdout").sum()) if not metrics_df.empty else 0
        ),
        "dropped_all_nan_features": sorted(list(dropped_all_nan_union)),
    }

    if not metrics_df.empty:
        for ph in ["dev", "holdout"]:
            sub = metrics_df[metrics_df["phase"] == ph]
            if len(sub) > 0:
                report[f"{ph}_rmse_mean"] = round(float(sub["rmse"].mean()), 8)
                report[f"{ph}_ic_rank_mean"] = round(float(sub["ic_rank"].mean()), 8)
                report[f"{ph}_hit_ratio_mean"] = round(
                    float(sub["hit_ratio"].mean()), 8
                )
                # [Stage 2] r2_oos_mean 추가
                if "r2_oos" in sub.columns:
                    report[f"{ph}_r2_oos_mean"] = round(float(sub["r2_oos"].mean()), 8)

    # [Stage 2] 계수 저장
    if export_feature_importance and interim_dir is not None and coef_rows:
        coef_df = pd.DataFrame(coef_rows)
        coef_path = interim_dir / "model_coefs.parquet"

        # 기존 파일이 있으면 병합 (누적 저장)
        if coef_path.exists():
            existing_coef = pd.read_parquet(coef_path)
            # 동일 fold_id, phase, horizon, feature 조합이 있으면 업데이트, 없으면 추가
            merge_keys = ["fold_id", "phase", "horizon", "feature"]
            coef_df = (
                pd.concat([existing_coef, coef_df])
                .drop_duplicates(subset=merge_keys, keep="last")
                .reset_index(drop=True)
            )

        coef_df.to_parquet(coef_path, index=False)
        warns.append(f"[L5 Stage2] Model coefficients saved: {coef_path}")

        # 집계 리포트 생성
        summary_rows = []
        for (h, ph), group in coef_df.groupby(["horizon", "phase"]):
            for feature in group["feature"].unique():
                feat_group = group[group["feature"] == feature]
                coef_values = feat_group["coef"].values
                abs_coef_values = feat_group["abs_coef"].values

                # 부호 일치율 계산 (양수/음수 비율)
                positive_count = (coef_values > 0).sum()
                negative_count = (coef_values < 0).sum()
                total_count = len(coef_values)
                sign_stability = (
                    max(positive_count, negative_count) / total_count
                    if total_count > 0
                    else np.nan
                )

                summary_rows.append(
                    {
                        "horizon": int(h),
                        "phase": ph,
                        "feature": feature,
                        "abs_coef_mean": float(abs_coef_values.mean()),
                        "abs_coef_median": float(np.median(abs_coef_values)),
                        "abs_coef_std": (
                            float(abs_coef_values.std())
                            if len(abs_coef_values) > 1
                            else 0.0
                        ),
                        "coef_mean": float(coef_values.mean()),
                        "coef_std": (
                            float(coef_values.std()) if len(coef_values) > 1 else 0.0
                        ),
                        "coef_sign_stability": float(sign_stability),
                        "n_folds": int(total_count),
                    }
                )

        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_path = interim_dir / "feature_importance_summary.parquet"
            summary_df.to_parquet(summary_path, index=False)
            warns.append(
                f"[L5 Stage2] Feature importance summary saved: {summary_path}"
            )

    # [원래 상태로 복원] Alpha 튜닝 비활성화로 인해 검증 로직 제거

    # [Stage6] 피처 그룹 밸런싱 계산 및 저장
    try:
        feature_groups_config = load_feature_groups()

        # 피처 중요도 추출 (coef_rows에서)
        feature_importance = {}
        if coef_rows:
            # 각 피처별 평균 절대 계수 사용
            coef_df_temp = pd.DataFrame(coef_rows)
            for feature in coef_df_temp["feature"].unique():
                feat_coefs = coef_df_temp[coef_df_temp["feature"] == feature][
                    "abs_coef"
                ].values
                feature_importance[feature] = (
                    float(np.mean(feat_coefs)) if len(feat_coefs) > 0 else 0.0
                )

        # 그룹 밸런싱 계산
        balance_df = calculate_feature_group_balance(
            feature_cols=feature_cols,
            feature_importance=feature_importance if feature_importance else None,
            config=feature_groups_config,
        )

        # feature_group_balance.parquet 저장
        if interim_dir is not None:
            balance_path = interim_dir / "feature_group_balance.parquet"
            balance_df.to_parquet(balance_path, index=False)
            warns.append(
                f"[Stage6] 피처 그룹 밸런싱 저장: {balance_path} ({len(balance_df)} groups)"
            )

            # 리포트에 그룹 밸런싱 정보 추가
            report["feature_groups_total"] = len(balance_df)
            report["feature_groups_balanced"] = int(
                (balance_df["balance_ratio"] > 0.5).sum()
            )
            report["feature_groups_ungrouped"] = int(
                (balance_df["group_name"] == "ungrouped").sum()
            )
        else:
            warns.append(
                "[Stage6] interim_dir이 None이어서 feature_group_balance.parquet를 저장하지 않습니다."
            )
    except Exception as e:
        warns.append(f"[Stage6] 피처 그룹 밸런싱 계산/저장 실패: {e}")
        import traceback

        warns.append(f"[Stage6] 트레이스백: {traceback.format_exc()}")

    return pred_oos, metrics_df, report, warns
