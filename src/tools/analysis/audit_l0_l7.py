# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/analysis/audit_l0_l7.py
import json
import sys
from pathlib import Path

from src.utils.config import get_path, load_config
from src.utils.io import artifact_exists, load_artifact

ARTIFACTS = [
    "universe_k200_membership_monthly",
    "ohlcv_daily",
    "fundamentals_annual",
    "panel_merged_daily",
    "dataset_daily",
    "cv_folds_short",
    "cv_folds_long",
    "pred_short_oos",
    "pred_long_oos",
    "model_metrics",
    "rebalance_scores",
    "rebalance_scores_summary",
    "bt_positions",
    "bt_returns",
    "bt_equity_curve",
    "bt_metrics",
]

def _root() -> Path:
    # .../03_code/src/stages/audit_l0_l7.py -> parents[2] == 03_code
    return Path(__file__).resolve().parents[2]

def _must(cond: bool, msg: str):
    if not cond:
        raise SystemExit(f"[FAIL] {msg}")

def _meta_path(interim: Path, name: str) -> Path:
    return interim / f"{name}__meta.json"

def _meta_exists(interim: Path, name: str) -> bool:
    return _meta_path(interim, name).exists()

def _load_meta(interim: Path, name: str) -> dict:
    mp = _meta_path(interim, name)
    _must(mp.exists(), f"missing meta json: {mp}")
    with mp.open("r", encoding="utf-8") as f:
        return json.load(f)

def _load(interim: Path, name: str):
    base = interim / name
    _must(artifact_exists(base), f"artifact missing: {base}")
    return load_artifact(base)

def main():
    print("=== L0~L7 AUDIT RUNNER ===")
    root = _root()
    cfg_path = root / "configs" / "config.yaml"
    _must(cfg_path.exists(), f"config not found: {cfg_path}")

    cfg = load_config(str(cfg_path))
    interim = get_path(cfg, "data_interim")

    print("ROOT  :", root)
    print("CFG   :", cfg_path)
    print("INTERIM:", interim)

    # 1) existence
    for name in ARTIFACTS:
        base = interim / name
        _must(artifact_exists(base), f"missing artifact: {name}")
        _must(_meta_exists(interim, name), f"missing meta: {name}__meta.json")

    print("[PASS] all artifacts/meta exist")

    # 2) minimal schema checks
    uni = _load(interim, "universe_k200_membership_monthly")
    _must(set(["date", "ticker"]).issubset(uni.columns), "L0 schema: need date,ticker")

    ohlcv = _load(interim, "ohlcv_daily")
    _must(set(["date", "ticker", "close"]).issubset(ohlcv.columns), "L1 schema: need date,ticker,close")
    _must(ohlcv.duplicated(["date", "ticker"]).sum() == 0, "L1 duplicate (date,ticker)")

    fa = _load(interim, "fundamentals_annual")
    _must(set(["date", "ticker"]).issubset(fa.columns), "L2 schema: need date,ticker")

    panel = _load(interim, "panel_merged_daily")
    _must(set(["date", "ticker"]).issubset(panel.columns), "L3 schema: need date,ticker")
    _must(panel.duplicated(["date", "ticker"]).sum() == 0, "L3 duplicate (date,ticker)")

    ds = _load(interim, "dataset_daily")
    _must(set(["date", "ticker"]).issubset(ds.columns), "L4 schema: need date,ticker")
    _must(("ret_fwd_20d" in ds.columns) and ("ret_fwd_120d" in ds.columns), "L4 targets missing")

    cv_s = _load(interim, "cv_folds_short")
    cv_l = _load(interim, "cv_folds_long")
    _must("fold_id" in cv_s.columns, "cv_folds_short missing fold_id")
    _must("fold_id" in cv_l.columns, "cv_folds_long missing fold_id")
    for c in ["train_start", "train_end", "test_start", "test_end"]:
        _must(c in cv_s.columns, f"cv_folds_short missing {c}")
        _must(c in cv_l.columns, f"cv_folds_long missing {c}")

    ps = _load(interim, "pred_short_oos")
    pl = _load(interim, "pred_long_oos")
    need_pred_cols = {"date", "ticker", "y_true", "y_pred", "fold_id", "phase", "horizon"}
    _must(need_pred_cols.issubset(ps.columns), f"pred_short_oos schema missing: {sorted(need_pred_cols - set(ps.columns))}")
    _must(need_pred_cols.issubset(pl.columns), f"pred_long_oos schema missing: {sorted(need_pred_cols - set(pl.columns))}")
    _must(ps[["y_true", "y_pred"]].isna().any(axis=1).mean() == 0.0, "pred_short_oos has NA in y_true/y_pred")
    _must(pl[["y_true", "y_pred"]].isna().any(axis=1).mean() == 0.0, "pred_long_oos has NA in y_true/y_pred")

    mm = _load(interim, "model_metrics")
    _must(set(["phase", "horizon", "fold_id", "rmse"]).issubset(mm.columns), "model_metrics schema insufficient")

    rs = _load(interim, "rebalance_scores")
    rss = _load(interim, "rebalance_scores_summary")
    _must(set(["date", "ticker", "phase"]).issubset(rs.columns), "rebalance_scores schema: need date,ticker,phase")
    _must(rs.duplicated(["date", "ticker", "phase"]).sum() == 0, "rebalance_scores duplicate (date,ticker,phase)")
    _must(set(["date", "phase", "coverage_ticker_pct"]).issubset(rss.columns), "rebalance_scores_summary schema insufficient")

    bt_m = _load(interim, "bt_metrics")
    _must(set(["phase", "net_sharpe", "net_mdd", "net_cagr"]).issubset(bt_m.columns), "bt_metrics missing key cols")

    print("[PASS] core schema/duplicate/NA checks")

    # 3) meta quality keys existence
    m3 = _load_meta(interim, "panel_merged_daily")
    _must("fundamental" in (m3.get("quality") or {}), "L3 meta missing quality.fundamental")

    m4 = _load_meta(interim, "dataset_daily")
    _must("walkforward" in (m4.get("quality") or {}), "L4 meta missing quality.walkforward")

    m5s = _load_meta(interim, "pred_short_oos")
    m5l = _load_meta(interim, "pred_long_oos")
    _must("model_oos" in (m5s.get("quality") or {}), "L5 meta missing quality.model_oos (short)")
    _must("model_oos" in (m5l.get("quality") or {}), "L5 meta missing quality.model_oos (long)")

    m6 = _load_meta(interim, "rebalance_scores")
    _must("scoring" in (m6.get("quality") or {}), "L6 meta missing quality.scoring")

    print("[PASS] meta quality keys check")
    print("✅ AUDIT COMPLETE: proceed to B/C/D extensions and final reporting.")

if __name__ == "__main__":
    main()
