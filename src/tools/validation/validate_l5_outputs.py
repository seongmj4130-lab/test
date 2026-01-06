# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/validation/validate_l5_outputs.py
from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------
# Path bootstrap (Spyder/Windows 안정화)
# ----------------------------
THIS = Path(__file__).resolve()
ROOT = THIS.parents[2]          # .../03_code
SRC = THIS.parents[1]           # .../03_code/src
CFG_PATH = ROOT / "configs" / "config.yaml"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.utils.config import load_config, get_path
from src.utils.io import load_artifact, artifact_exists

# ----------------------------
# Helpers
# ----------------------------
def _must_exist(base: Path, name: str) -> None:
    if not artifact_exists(base):
        raise FileNotFoundError(f"[FAIL] artifact missing: {name} -> {base}(.parquet/.csv)")

def _load_df(interim: Path, name: str) -> pd.DataFrame:
    base = interim / name
    _must_exist(base, name)
    df = load_artifact(base)
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError(f"[FAIL] artifact empty or not DataFrame: {name}")
    return df

def _load_meta(interim: Path, name: str) -> dict:
    meta_path = interim / f"{name}__meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"[FAIL] meta missing: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))

def _ensure_datetime(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        raise KeyError(f"[FAIL] missing column: {col}")
    if not np.issubdtype(df[col].dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col], errors="raise")

def _ensure_ticker_str(df: pd.DataFrame) -> None:
    if "ticker" not in df.columns:
        raise KeyError("[FAIL] missing column: ticker")
    df["ticker"] = df["ticker"].astype(str).str.zfill(6)

def _basic_checks(df: pd.DataFrame, name: str) -> None:
    for c in ["date", "ticker"]:
        if c not in df.columns:
            raise KeyError(f"[FAIL] {name} missing column: {c}")
    _ensure_datetime(df, "date")
    _ensure_ticker_str(df)

    dup = int(df.duplicated(subset=["date", "ticker"]).sum())
    if dup != 0:
        raise ValueError(f"[FAIL] {name} has duplicate (date,ticker) keys: {dup}")

    if df["date"].isna().any():
        raise ValueError(f"[FAIL] {name} has NaT in date")
    if df["ticker"].isna().any():
        raise ValueError(f"[FAIL] {name} has NA in ticker")

def _pick_pred_col(df: pd.DataFrame) -> str:
    candidates = ["y_pred", "pred", "prediction", "yhat"]
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"[FAIL] prediction column not found. existing={list(df.columns)}")

def _metrics(y: np.ndarray, p: np.ndarray) -> dict:
    mask = np.isfinite(y) & np.isfinite(p)
    if mask.sum() == 0:
        return {"n": 0, "rmse": np.nan, "mae": np.nan, "corr": np.nan, "hit_ratio": np.nan}

    yy = y[mask]
    pp = p[mask]
    rmse = float(np.sqrt(np.mean((pp - yy) ** 2)))
    mae = float(np.mean(np.abs(pp - yy)))
    corr = float(np.corrcoef(pp, yy)[0, 1]) if len(yy) > 1 else np.nan
    hit = float(np.mean((pp > 0) == (yy > 0)))
    return {"n": int(mask.sum()), "rmse": rmse, "mae": mae, "corr": corr, "hit_ratio": hit}

def _merged_intervals(intervals: list[tuple[pd.Timestamp, pd.Timestamp]]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    intervals = sorted(intervals, key=lambda x: x[0])
    out: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    for s, e in intervals:
        if not out:
            out.append((s, e))
            continue
        ps, pe = out[-1]
        if s <= pe:
            out[-1] = (ps, max(pe, e))
        else:
            out.append((s, e))
    return out

def _mask_by_intervals(dates: pd.Series, merged: list[tuple[pd.Timestamp, pd.Timestamp]]) -> pd.Series:
    m = pd.Series(False, index=dates.index)
    for s, e in merged:
        m |= (dates >= s) & (dates <= e)
    return m

def _attach_target_from_dataset(
    pred: pd.DataFrame,
    dataset: pd.DataFrame,
    target_col: str,
    *,
    name: str
) -> pd.DataFrame:
    """
    pred에 target_col이 없으면 dataset_daily에서 (date,ticker)로 조인해 붙인다.
    (검증 용도이며, 누락률/성능 계산을 위해 필요)
    """
    if target_col in pred.columns:
        return pred

    if target_col not in dataset.columns:
        raise KeyError(f"[FAIL] dataset_daily missing target col: {target_col}")

    # 키 중복 방지(앞에서 pred는 이미 체크됨)
    dkey = dataset[["date", "ticker", target_col]].copy()
    _ensure_datetime(dkey, "date")
    _ensure_ticker_str(dkey)
    dup = int(dkey.duplicated(subset=["date", "ticker"]).sum())
    if dup != 0:
        raise ValueError(f"[FAIL] dataset_daily has duplicate (date,ticker) keys: {dup} (cannot attach targets safely)")

    merged = pred.merge(dkey, on=["date", "ticker"], how="left", validate="one_to_one")
    print(f"[INFO] {name}: target '{target_col}' attached from dataset_daily (missing before attach = {merged[target_col].isna().mean()*100:.4f}%)")
    return merged

def _coverage_against_folds(
    dataset: pd.DataFrame,
    pred: pd.DataFrame,
    cv_folds: pd.DataFrame,
    target_col: str,
) -> dict:
    _ensure_datetime(dataset, "date")
    _ensure_ticker_str(dataset)

    if target_col not in dataset.columns:
        raise KeyError(f"[FAIL] dataset_daily missing target col: {target_col}")

    need_cols = ["test_start", "test_end"]
    for c in need_cols:
        if c not in cv_folds.columns:
            raise KeyError(f"[FAIL] cv_folds missing column: {c}")

    starts = pd.to_datetime(cv_folds["test_start"], errors="raise")
    ends = pd.to_datetime(cv_folds["test_end"], errors="raise")
    intervals = [(s, e) for s, e in zip(starts.tolist(), ends.tolist())]
    merged_intv = _merged_intervals(intervals)

    eligible_mask = _mask_by_intervals(dataset["date"], merged_intv) & dataset[target_col].notna()
    eligible = dataset.loc[eligible_mask, ["date", "ticker"]].drop_duplicates()

    pred_keys = pred.loc[pred[target_col].notna(), ["date", "ticker"]].drop_duplicates()

    pred_outside = ~_mask_by_intervals(pred["date"], merged_intv)
    outside_cnt = int(pred_outside.sum())
    if outside_cnt != 0:
        raise ValueError(f"[FAIL] pred_oos contains {outside_cnt} rows outside test windows (leakage or bad slicing).")

    cov = (len(pred_keys) / len(eligible)) * 100.0 if len(eligible) > 0 else np.nan
    return {
        "eligible_rows": int(len(eligible)),
        "pred_rows": int(len(pred_keys)),
        "coverage_pct": float(round(cov, 4)) if np.isfinite(cov) else np.nan,
        "folds": int(len(cv_folds)),
        "test_date_min": str(starts.min().date()),
        "test_date_max": str(ends.max().date()),
    }

def _fold_level_report(pred: pd.DataFrame, pred_col: str, true_col: str) -> pd.DataFrame:
    if "fold_id" not in pred.columns:
        y = pd.to_numeric(pred[true_col], errors="coerce").to_numpy()
        p = pd.to_numeric(pred[pred_col], errors="coerce").to_numpy()
        m = _metrics(y, p)
        return pd.DataFrame([{"fold_id": "ALL", **m}])

    rows = []
    for fid, g in pred.groupby("fold_id", sort=False):
        y = pd.to_numeric(g[true_col], errors="coerce").to_numpy()
        p = pd.to_numeric(g[pred_col], errors="coerce").to_numpy()
        m = _metrics(y, p)
        rows.append({"fold_id": str(fid), **m})
    return pd.DataFrame(rows)

def main():
    print("=== L5 Validation Runner ===")
    print("ROOT:", ROOT)
    print("CFG :", CFG_PATH)

    if not CFG_PATH.exists():
        raise FileNotFoundError(f"[FAIL] config.yaml not found: {CFG_PATH}")

    cfg = load_config(str(CFG_PATH))
    interim = get_path(cfg, "data_interim")
    print("INTERIM:", interim)

    # --- Load artifacts
    ds = _load_df(interim, "dataset_daily")
    cv_s = _load_df(interim, "cv_folds_short")
    cv_l = _load_df(interim, "cv_folds_long")
    ps = _load_df(interim, "pred_short_oos")
    pl = _load_df(interim, "pred_long_oos")
    mm = _load_df(interim, "model_metrics")

    # --- Basic checks
    _basic_checks(ps, "pred_short_oos")
    _basic_checks(pl, "pred_long_oos")

    # --- Print columns (빠른 실체 확인)
    print("\n=== Columns snapshot ===")
    print("pred_short_oos cols:", list(ps.columns))
    print("pred_long_oos  cols:", list(pl.columns))
    print("dataset_daily  cols(head 30):", list(ds.columns)[:30])
    print("cv_folds_short cols:", list(cv_s.columns))
    print("cv_folds_long  cols:", list(cv_l.columns))

    # --- Determine horizon from cv_folds (single-valued)
    hs = int(pd.to_numeric(cv_s["horizon_days"], errors="raise").iloc[0])
    hl = int(pd.to_numeric(cv_l["horizon_days"], errors="raise").iloc[0])

    t_s = f"ret_fwd_{hs}d"
    t_l = f"ret_fwd_{hl}d"

    pred_col_s = _pick_pred_col(ps)
    pred_col_l = _pick_pred_col(pl)

    # --- Attach targets if missing
    ps = _attach_target_from_dataset(ps, ds, t_s, name="pred_short_oos")
    pl = _attach_target_from_dataset(pl, ds, t_l, name="pred_long_oos")

    # --- Missingness
    miss_s = float(round(ps[[t_s, pred_col_s]].isna().any(axis=1).mean() * 100, 6))
    miss_l = float(round(pl[[t_l, pred_col_l]].isna().any(axis=1).mean() * 100, 6))

    # --- Overall metrics
    ms = _metrics(
        pd.to_numeric(ps[t_s], errors="coerce").to_numpy(),
        pd.to_numeric(ps[pred_col_s], errors="coerce").to_numpy(),
    )
    ml = _metrics(
        pd.to_numeric(pl[t_l], errors="coerce").to_numpy(),
        pd.to_numeric(pl[pred_col_l], errors="coerce").to_numpy(),
    )

    # --- Coverage vs folds (no leakage)
    cov_s = _coverage_against_folds(ds, ps, cv_s, t_s)
    cov_l = _coverage_against_folds(ds, pl, cv_l, t_l)

    # --- Fold-level sample report
    rep_s = _fold_level_report(ps, pred_col_s, t_s).sort_values("fold_id").head(5)
    rep_l = _fold_level_report(pl, pred_col_l, t_l).sort_values("fold_id").head(5)

    # --- Meta check
    meta_ps = _load_meta(interim, "pred_short_oos")
    meta_pl = _load_meta(interim, "pred_long_oos")
    meta_mm = _load_meta(interim, "model_metrics")

    q_ps = meta_ps.get("quality", {})
    q_pl = meta_pl.get("quality", {})
    q_mm = meta_mm.get("quality", {})

    print("\n=== [PASS] L5 artifacts loaded and basic checks ok ===")
    print(f"- pred_short_oos rows={len(ps):,} cols={ps.shape[1]} pred_col={pred_col_s} target={t_s}")
    print(f"- pred_long_oos  rows={len(pl):,} cols={pl.shape[1]} pred_col={pred_col_l} target={t_l}")
    print(f"- model_metrics  rows={len(mm):,} cols={mm.shape[1]}")

    print("\n=== Missingness (row has any NA in [target,pred]) ===")
    print(f"- short: {miss_s}%")
    print(f"- long : {miss_l}%")

    print("\n=== Overall OOS Metrics ===")
    print(f"- short: n={ms['n']:,} rmse={ms['rmse']:.6f} mae={ms['mae']:.6f} corr={ms['corr']:.6f} hit={ms['hit_ratio']:.6f}")
    print(f"- long : n={ml['n']:,} rmse={ml['rmse']:.6f} mae={ml['mae']:.6f} corr={ml['corr']:.6f} hit={ml['hit_ratio']:.6f}")

    print("\n=== Coverage vs Eligible rows (test windows ∩ target notna) ===")
    print(f"- short: coverage={cov_s['coverage_pct']}% pred={cov_s['pred_rows']:,} eligible={cov_s['eligible_rows']:,} folds={cov_s['folds']} "
          f"test_range=[{cov_s['test_date_min']} ~ {cov_s['test_date_max']}]")
    print(f"- long : coverage={cov_l['coverage_pct']}% pred={cov_l['pred_rows']:,} eligible={cov_l['eligible_rows']:,} folds={cov_l['folds']} "
          f"test_range=[{cov_l['test_date_min']} ~ {cov_l['test_date_max']}]")

    print("\n=== Fold-level sample (first 5 rows) ===")
    print("[short]\n", rep_s.to_string(index=False))
    print("[long]\n", rep_l.to_string(index=False))

    print("\n=== Meta quality keys (existence only) ===")
    print("- pred_short_oos meta quality keys:", list(q_ps.keys()))
    print("- pred_long_oos  meta quality keys:", list(q_pl.keys()))
    print("- model_metrics  meta quality keys:", list(q_mm.keys()))

    print("\n✅ L5 VALIDATION COMPLETE: All critical checks passed.")
    print("➡️ Next: proceed to L6 (scoring / ranking / rebalance inputs).")

if __name__ == "__main__":
    main()
