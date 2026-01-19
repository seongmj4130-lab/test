# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/utils/quality.py
from __future__ import annotations

import pandas as pd


def fundamental_coverage_report(df: pd.DataFrame) -> dict:
    """L3: 재무 머지 커버리지(행 기준)"""
    cols = [c for c in ["net_income", "equity", "total_liabilities", "debt_ratio", "roe"] if c in df.columns]
    if not cols:
        return {"coverage_any_pct": None, "covered_rows": 0, "total_rows": int(len(df)), "cols_used": []}

    any_nonnull = df[cols].notna().any(axis=1)
    total = int(len(df))
    covered = int(any_nonnull.sum())
    cov = (covered / total * 100.0) if total else 0.0
    return {
        "coverage_any_pct": round(cov, 4),
        "covered_rows": covered,
        "total_rows": total,
        "cols_used": cols,
    }

def target_coverage_report(dataset: pd.DataFrame, *, horizon_short: int, horizon_long: int) -> dict:
    """L4: 타깃(ret_fwd_*) 결측률 리포트"""
    col_s = f"ret_fwd_{horizon_short}d"
    col_l = f"ret_fwd_{horizon_long}d"

    out = {"target_missing_short_pct": None, "target_missing_long_pct": None}
    if dataset is None or dataset.empty:
        return out

    if col_s in dataset.columns:
        out["target_missing_short_pct"] = round(float(dataset[col_s].isna().mean()) * 100.0, 4)
    if col_l in dataset.columns:
        out["target_missing_long_pct"] = round(float(dataset[col_l].isna().mean()) * 100.0, 4)
    return out

def folds_report(cv_short: pd.DataFrame, cv_long: pd.DataFrame) -> dict:
    """L4: 폴드 개수/세그먼트 분포 리포트"""
    def _rep(cv: pd.DataFrame):
        if cv is None or cv.empty:
            return {"rows": 0, "dev": 0, "holdout": 0}
        seg = cv["segment"] if "segment" in cv.columns else pd.Series([], dtype=str)
        return {
            "rows": int(len(cv)),
            "dev": int((seg == "dev").sum()) if len(seg) else 0,
            "holdout": int((seg == "holdout").sum()) if len(seg) else 0,
        }

    return {"cv_short": _rep(cv_short), "cv_long": _rep(cv_long)}

def walkforward_quality_report(
    dataset: pd.DataFrame | None = None,
    cv_short: pd.DataFrame | None = None,
    cv_long: pd.DataFrame | None = None,
    *,
    # run_all.py 호환용 alias
    dataset_daily: pd.DataFrame | None = None,
    cv_folds_short: pd.DataFrame | None = None,
    cv_folds_long: pd.DataFrame | None = None,

    # run_all.py가 안 넘겨도 죽지 않도록 모두 Optional로
    horizon_short: int | None = None,
    horizon_long: int | None = None,
    step_days: int | None = None,
    test_window_days: int | None = None,
    embargo_days: int | None = None,
    holdout_years: int | None = None,

    # cfg/params에서 값을 끌어올 수 있게
    cfg: dict | None = None,
    params: dict | None = None,

    date_col: str = "date",
    **kwargs,
) -> dict:
    """
    L4 품질지표(메타에 넣을 dict)
    - run_all 호출 인자 누락으로 파이프라인이 죽지 않게 "방어적으로" 작성
    - 가능하면 cfg/params 또는 dataset 컬럼에서 horizon을 추정
    """
    def _safe_int(x):
        return None if x is None else int(x)

    # alias 해소
    ds = dataset_daily if dataset_daily is not None else dataset
    cv_s = cv_short if cv_short is not None else cv_folds_short
    cv_l = cv_long if cv_long is not None else cv_folds_long

    # cfg/params에서 끌어오기
    p = {}
    if isinstance(cfg, dict):
        # cfg가 {"params": {...}} 형태면 params 우선
        p = cfg.get("params", cfg)
    if isinstance(params, dict):
        p = {**p, **params}

    # config에 있는 키를 우선 반영
    horizon_short = horizon_short if horizon_short is not None else p.get("horizon_short")
    horizon_long  = horizon_long  if horizon_long  is not None else p.get("horizon_long")
    step_days     = step_days     if step_days     is not None else p.get("step_days")
    test_window_days = test_window_days if test_window_days is not None else p.get("test_window_days")
    embargo_days  = embargo_days  if embargo_days  is not None else p.get("embargo_days")
    holdout_years = holdout_years if holdout_years is not None else p.get("holdout_years")

    # dataset 컬럼에서 horizon 추정(ret_fwd_{n}d가 있으면)
    if ds is not None and not ds.empty and (horizon_short is None or horizon_long is None):
        import re
        hs = []
        for c in ds.columns:
            m = re.match(r"ret_fwd_(\d+)d$", str(c))
            if m:
                hs.append(int(m.group(1)))
        hs = sorted(set(hs))
        if hs:
            if horizon_short is None:
                horizon_short = hs[0]
            if horizon_long is None:
                horizon_long = hs[-1]

    rep: dict = {}

    # 타깃 결측률: horizon이 확정될 때만 계산(없으면 None 유지)
    if ds is not None and horizon_short is not None and horizon_long is not None:
        rep.update(
            target_coverage_report(
                ds,
                horizon_short=int(horizon_short),
                horizon_long=int(horizon_long),
            )
        )
    else:
        rep.update({"target_missing_short_pct": None, "target_missing_long_pct": None})

    # folds 리포트 (None이어도 folds_report 내부에서 처리 가능)
    rep.update(folds_report(cv_s, cv_l))

    # 메타 파라미터들(없으면 None)
    rep.update({
        "holdout_years": _safe_int(holdout_years),
        "step_days": _safe_int(step_days),
        "test_window_days": _safe_int(test_window_days),
        "embargo_days": _safe_int(embargo_days),
        "horizon_short": _safe_int(horizon_short),
        "horizon_long": _safe_int(horizon_long),
        "date_col": str(date_col),
    })

    return rep
