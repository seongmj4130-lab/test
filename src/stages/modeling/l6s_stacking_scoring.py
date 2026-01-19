"""
[개선안 24번] L6S: 스태킹 메타모델로 리밸런싱 스코어(score_ens) 생성

핵심 제약(누수 방지):
  - 메타모델 학습은 **dev phase의 OOS 예측/신호만** 사용한다.
  - holdout은 평가 전용(학습 금지).

입력(도구/파이프라인에서 준비):
  - base_scores: (date,ticker,phase) 단위로 정렬된 베이스 신호 DataFrame들
    예: ridge_score_ens, rf_score_ens, xgb_score_ens, ranking_dual_score_ens
  - true_short: 동일 키로 실현 forward return (L7 ret_col)

출력:
  - rebalance_scores_stacked (date,ticker,phase,score_ens,true_short, in_universe?)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class StackingConfig:
    """
    [개선안 24번] 스태킹 설정
    """

    meta_model: str = "ridge"  # currently only ridge
    ridge_alpha: float = 1.0
    feature_cols: tuple[str, ...] = ("ridge", "rf", "xgb", "ranking_dual")
    phase_train: str = "dev"
    out_score_col: str = "score_ens"
    target_col: str = "true_short"


def _ensure_key(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["date"] = pd.to_datetime(x["date"], errors="raise")
    x["ticker"] = x["ticker"].astype(str).str.zfill(6)
    x["phase"] = x["phase"].astype(str)
    return x


def build_stacked_rebalance_scores(
    *,
    base_frames: dict[str, pd.DataFrame],
    target_frame: pd.DataFrame,
    cfg: StackingConfig,
) -> tuple[pd.DataFrame, dict, list[str]]:
    """
    [개선안 24번] 베이스 신호들을 병합하고, dev OOS로 메타모델 학습 후 score_ens 생성
    """
    warns: list[str] = []

    if not base_frames:
        raise ValueError("base_frames is empty")
    if target_frame is None or target_frame.empty:
        raise ValueError("target_frame is empty")

    ydf = _ensure_key(target_frame)
    if cfg.target_col not in ydf.columns:
        raise KeyError(f"target_frame missing column: {cfg.target_col}")

    # merge base frames on (date,ticker,phase)
    key = ["date", "ticker", "phase"]
    out = ydf[key + [cfg.target_col]].copy()
    for name, df in base_frames.items():
        if df is None or df.empty:
            warns.append(f"[L6S] base frame '{name}' is empty -> skipped")
            continue
        x = _ensure_key(df)
        if "score_ens" not in x.columns:
            raise KeyError(f"base frame '{name}' missing score_ens")
        x = x[key + ["score_ens"]].rename(columns={"score_ens": name})
        out = out.merge(x, on=key, how="left", validate="one_to_one")

    # training split
    train_mask = out["phase"].str.lower().eq(str(cfg.phase_train).lower())
    dtrain = out.loc[train_mask].copy()
    dtest = out.loc[~train_mask].copy()

    # feature matrix
    feat_cols = [c for c in cfg.feature_cols if c in out.columns]
    if len(feat_cols) < 2:
        raise ValueError(
            f"too few stacking features. got={feat_cols}. expected={cfg.feature_cols}"
        )

    X_train = dtrain[feat_cols].to_numpy(dtype=np.float32, copy=False)
    y_train = pd.to_numeric(dtrain[cfg.target_col], errors="coerce").to_numpy(
        dtype=np.float32, copy=False
    )
    X_all = out[feat_cols].to_numpy(dtype=np.float32, copy=False)

    # meta model
    if cfg.meta_model != "ridge":
        raise ValueError(f"unsupported meta_model: {cfg.meta_model}")

    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", Ridge(alpha=float(cfg.ridge_alpha))),
        ]
    )
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_all).astype(np.float32)

    out[cfg.out_score_col] = pred
    # L7 expects true_short column name by default; keep it
    out = out.rename(columns={cfg.target_col: "true_short"})

    report = {
        "meta_model": f"ridge(alpha={float(cfg.ridge_alpha)})",
        "phase_train": cfg.phase_train,
        "features_used": feat_cols,
        "n_train": int(len(dtrain)),
        "n_total": int(len(out)),
        "train_target_mean": float(np.nanmean(y_train)) if len(y_train) else np.nan,
    }

    return out.sort_values(key).reset_index(drop=True), report, warns
