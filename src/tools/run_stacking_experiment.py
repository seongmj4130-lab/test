# -*- coding: utf-8 -*-
"""
[개선안 24번] 스태킹 실험 러너

동작:
  - data/interim 산출물 로드
  - 베이스 신호 4종 준비
    - ranking_dual: L6R(in-memory)
    - ridge: Option A 모델 rebalance_scores (기본)
    - rf: L5(model_type=rf) + L6를 in-memory로 재계산 (간소 버전: L6의 weight_short/long을 그대로)
    - xgb: L5(model_type=xgb) + L6 in-memory
  - L6S(meta)로 score_ens 생성 후 L7 백테스트
  - 결과를 finalterm/에 저장:
    - finalterm/35_stacking_results.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config  # noqa: E402
from src.utils.io import load_artifact, artifact_exists  # noqa: E402
from src.stages.modeling.l5_train_models import train_oos_predictions  # noqa: E402
from src.stages.modeling.l6_scoring import build_rebalance_scores  # noqa: E402
from src.stages.modeling.l6r_ranking_scoring import run_L6R_ranking_scoring  # noqa: E402
from src.stages.modeling.l6s_stacking_scoring import StackingConfig, build_stacked_rebalance_scores  # noqa: E402
from src.stages.backtest.l7_backtest import run_backtest  # noqa: E402
from src.stages.backtest.l1d_market_regime import build_market_regime  # noqa: E402


def _safe_read_parquet(p: Path) -> pd.DataFrame:
    if p.exists():
        return pd.read_parquet(p)
    raise FileNotFoundError(p)


def _load_interim_artifact(interim: Path, name: str) -> pd.DataFrame:
    base = interim / name
    if artifact_exists(base):
        return load_artifact(base)
    # parquet fallback
    p = interim / f"{name}.parquet"
    return _safe_read_parquet(p)


def _clone_cfg(cfg: dict) -> dict:
    import copy
    return copy.deepcopy(cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--interim", default="data/interim")
    ap.add_argument("--model-rebalance", default="data/interim/option_a_only_20d/rebalance_scores.parquet")
    ap.add_argument("--out", default="finalterm/35_stacking_results.md")
    ap.add_argument("--meta-alpha", type=float, default=1.0)
    ap.add_argument("--enable-xgb", action="store_true", help="xgboost 설치된 환경에서만 켜세요")
    args = ap.parse_args()

    cfg = load_config(args.config)
    interim = (ROOT / args.interim).resolve()

    dataset_daily = _load_interim_artifact(interim, "dataset_daily")
    cv_s = _load_interim_artifact(interim, "cv_folds_short")
    cv_l = _load_interim_artifact(interim, "cv_folds_long")
    univ = _load_interim_artifact(interim, "universe_k200_membership_monthly")

    # 1) ranking_dual via L6R
    artifacts = {
        "dataset_daily": dataset_daily,
        "cv_folds_short": cv_s,
        "universe_k200_membership_monthly": univ,
        "ranking_short_daily": _safe_read_parquet(interim / "ranking_short_daily.parquet"),
        "ranking_long_daily": _safe_read_parquet(interim / "ranking_long_daily.parquet"),
    }
    dual_out, dual_warns = run_L6R_ranking_scoring(cfg, artifacts, force=True)
    rs_dual = dual_out["rebalance_scores"][["date", "ticker", "phase", "score_ens", "true_short"]].copy()

    # 2) ridge base: model rebalance_scores (default = option_a_only_20d)
    rs_ridge = _safe_read_parquet((ROOT / args.model_rebalance).resolve())
    rs_ridge = rs_ridge[["date", "ticker", "phase", "score_ens", "true_short"]].copy()

    # 3) rf base: recompute L5/L6 in-memory (short+long preds)
    def build_model_rebalance_scores(model_type: str) -> pd.DataFrame:
        c = _clone_cfg(cfg)
        c.setdefault("l5", {})
        c["l5"]["model_type"] = model_type

        # short
        ps, _, _, warns_s = train_oos_predictions(
            dataset_daily=dataset_daily,
            cv_folds=cv_s,
            cfg=c,
            target_col="ret_fwd_20d",
            horizon=int(c.get("l4", {}).get("horizon_short", 20)),
            interim_dir=None,
        )
        # long
        pl, _, _, warns_l = train_oos_predictions(
            dataset_daily=dataset_daily,
            cv_folds=cv_l,
            cfg=c,
            target_col="ret_fwd_120d",
            horizon=int(c.get("l4", {}).get("horizon_long", 120)),
            interim_dir=None,
        )

        # L6 scoring
        l6 = c.get("l6", {}) or {}
        out, _, _, warns6 = build_rebalance_scores(
            pred_short_oos=ps,
            pred_long_oos=pl,
            universe_k200_membership_monthly=univ,
            weight_short=float(l6.get("weight_short", 0.5)),
            weight_long=float(l6.get("weight_long", 0.5)),
            dataset_daily=dataset_daily,
            invert_score_sign=bool(l6.get("invert_score_sign", False)),
        )
        return out[["date", "ticker", "phase", "score_ens", "true_short"]].copy(), (warns_s + warns_l + warns6)

    rs_rf, rf_warns = build_model_rebalance_scores("random_forest")

    rs_xgb = None
    xgb_warns: List[str] = []
    if args.enable_xgb:
        rs_xgb, xgb_warns = build_model_rebalance_scores("xgboost")

    # 4) build stacked scores (dev only train)
    base_frames: Dict[str, pd.DataFrame] = {
        "ranking_dual": rs_dual.rename(columns={"score_ens": "score_ens"})[["date", "ticker", "phase", "score_ens"]],
        "ridge": rs_ridge.rename(columns={"score_ens": "score_ens"})[["date", "ticker", "phase", "score_ens"]],
        "rf": rs_rf.rename(columns={"score_ens": "score_ens"})[["date", "ticker", "phase", "score_ens"]],
    }
    feat_cols = ["ridge", "rf", "ranking_dual"]
    if rs_xgb is not None:
        base_frames["xgb"] = rs_xgb.rename(columns={"score_ens": "score_ens"})[["date", "ticker", "phase", "score_ens"]]
        feat_cols = ["ridge", "rf", "xgb", "ranking_dual"]

    stacked_cfg = StackingConfig(ridge_alpha=float(args.meta_alpha), feature_cols=tuple(feat_cols))
    rs_stacked, report, warns_stack = build_stacked_rebalance_scores(
        base_frames=base_frames,
        target_frame=rs_ridge[["date", "ticker", "phase", "true_short"]],
        cfg=stacked_cfg,
    )

    # 5) L7 backtest for stacked
    l7 = cfg.get("l7", {}) or {}
    bt_cfg = __import__("src.stages.backtest.l7_backtest", fromlist=["BacktestConfig"]).BacktestConfig(
        holding_days=int(l7.get("holding_days", 20)),
        top_k=int(l7.get("top_k", 20)),
        cost_bps=float(l7.get("cost_bps", 10.0)),
        score_col="score_ens",
        ret_col="true_short",
        weighting=str(l7.get("weighting", "equal")),
        buffer_k=int(l7.get("buffer_k", 0)),
        diversify_enabled=bool((l7.get("diversify") or {}).get("enabled", False)),
        group_col=str((l7.get("diversify") or {}).get("group_col", "sector_name")),
        max_names_per_group=int((l7.get("diversify") or {}).get("max_names_per_group", 4)),
        regime_enabled=bool((l7.get("regime") or {}).get("enabled", False)),
    )

    market_regime = None
    if bool((l7.get("regime") or {}).get("enabled", False)):
        p = cfg.get("params", {}) or {}
        market_regime = build_market_regime(
            rebalance_dates=rs_stacked["date"].unique(),
            start_date=str(p.get("start_date", "2016-01-01")),
            end_date=str(p.get("end_date", "2024-12-31")),
            index_code=str(p.get("index_code", "1028")),
            lookback_days=int((l7.get("regime") or {}).get("lookback_days", 60)),
            threshold_pct=float((l7.get("regime") or {}).get("threshold_pct", 0.0)),
        )

    _, _, _, bt_metrics, _, warns7, *_ = run_backtest(
        rebalance_scores=rs_stacked,
        cfg=bt_cfg,
        config_cost_bps=float(l7.get("cost_bps", 10.0)),
        market_regime=market_regime,
    )

    out = (ROOT / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    md = []
    md.append("# 35. 스태킹 결과 (Stacking Results)\n\n")
    md.append(f"- meta_model: {report.get('meta_model')}\n")
    md.append(f"- features_used: {report.get('features_used')}\n")
    md.append(f"- n_train(dev): {report.get('n_train')}\n\n")
    md.append("## 1) Stacked 백테스트 지표(bt_metrics)\n\n")
    md.append(bt_metrics.to_string(index=False))
    md.append("\n\n## 2) 경고(요약)\n\n")
    for w in (dual_warns or [])[:5]:
        md.append(f"- [dual] {w}\n")
    for w in (rf_warns or [])[:5]:
        md.append(f"- [rf] {w}\n")
    for w in (xgb_warns or [])[:5]:
        md.append(f"- [xgb] {w}\n")
    for w in (warns_stack or [])[:5]:
        md.append(f"- [stack] {w}\n")
    for w in (warns7 or [])[:5]:
        md.append(f"- [L7] {w}\n")

    out.write_text("".join(md), encoding="utf-8")
    print(f"[OK] wrote: {out}")


if __name__ == "__main__":
    main()


