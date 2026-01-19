# -*- coding: utf-8 -*-
"""
[개선안 20번] 신호 모드 비교(모델 vs 단일랭킹 vs 듀얼호라이즌) 자동 리포트

출력:
  - reports/dual_horizon_comparison.md

비교 대상(기본):
  - model: data/interim/option_a_only_20d/rebalance_scores.parquet
  - ranking_single: data/interim/stage15_ranking_full_20251223_073900/rebalance_scores.parquet
  - ranking_dual: data/interim 의 ranking_short_daily/long + L6R로 in-memory 생성

비교 기준:
  - 동일 L7 설정(holding_days/top_k/cost_bps/weighting/regime 등)
  - 동일 지표: Net Sharpe, Net MDD, Net CAGR, Net Hit Ratio
  - 벤치마크(멀티): universe_mean / kospi200 / savings (가능한 경우)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
# scripts/* 패턴과 동일하게 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(ROOT))

from src.stages.backtest.l1d_market_regime import build_market_regime  # noqa: E402
from src.stages.backtest.l7_backtest import BacktestConfig, run_backtest  # noqa: E402
from src.stages.backtest.l7c_benchmark import run_l7c_benchmark  # noqa: E402
from src.stages.modeling.l6r_ranking_scoring import (  # noqa: E402
    run_L6R_ranking_scoring,
)
from src.utils.config import load_config  # noqa: E402


def _project_root() -> Path:
    return ROOT


def _safe_read_parquet(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _get_l7_cfg(cfg: dict) -> BacktestConfig:
    l7 = (cfg.get("l7", {}) if isinstance(cfg, dict) else {}) or {}
    reg = (l7.get("regime", {}) or {}) if isinstance(l7.get("regime", {}), dict) else {}
    div = (l7.get("diversify", {}) or {}) if isinstance(l7.get("diversify", {}), dict) else {}
    return BacktestConfig(
        holding_days=int(l7.get("holding_days", 20)),
        top_k=int(l7.get("top_k", 20)),
        cost_bps=float(l7.get("cost_bps", 10.0)),
        score_col=str(l7.get("score_col", "score_ens")),
        ret_col=str(l7.get("return_col", "true_short")),
        weighting=str(l7.get("weighting", "equal")),
        buffer_k=int(l7.get("buffer_k", 0)),
        diversify_enabled=bool(div.get("enabled", False)),
        group_col=str(div.get("group_col", "sector_name")),
        max_names_per_group=int(div.get("max_names_per_group", 4)),
        regime_enabled=bool(reg.get("enabled", False)),
        regime_top_k_bull_strong=reg.get("top_k_bull_strong"),
        regime_top_k_bull_weak=reg.get("top_k_bull_weak"),
        regime_top_k_bear_strong=reg.get("top_k_bear_strong"),
        regime_top_k_bear_weak=reg.get("top_k_bear_weak"),
        regime_top_k_neutral=reg.get("top_k_neutral"),
        regime_exposure_bull_strong=reg.get("exposure_bull_strong"),
        regime_exposure_bull_weak=reg.get("exposure_bull_weak"),
        regime_exposure_bear_strong=reg.get("exposure_bear_strong"),
        regime_exposure_bear_weak=reg.get("exposure_bear_weak"),
        regime_exposure_neutral=reg.get("exposure_neutral"),
        regime_top_k_bull=reg.get("top_k_bull"),
        regime_top_k_bear=reg.get("top_k_bear"),
        regime_exposure_bull=reg.get("exposure_bull"),
        regime_exposure_bear=reg.get("exposure_bear"),
    )


def _compute_market_regime_if_needed(cfg: dict, rebalance_scores: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], List[str]]:
    warns: List[str] = []
    l7 = (cfg.get("l7", {}) if isinstance(cfg, dict) else {}) or {}
    reg = (l7.get("regime", {}) or {}) if isinstance(l7.get("regime", {}), dict) else {}
    if not bool(reg.get("enabled", False)):
        return None, warns
    try:
        dates = rebalance_scores["date"].unique()
        p = cfg.get("params", {}) or {}
        mr = build_market_regime(
            rebalance_dates=dates,
            start_date=str(p.get("start_date", "2016-01-01")),
            end_date=str(p.get("end_date", "2024-12-31")),
            index_code=str(p.get("index_code", "1028")),
            lookback_days=int(reg.get("lookback_days", 60)),
            threshold_pct=float(reg.get("threshold_pct", 0.0)),
        )
        return mr, warns
    except Exception as e:
        warns.append(f"[dual_eval] market_regime build failed -> skip regime: {type(e).__name__}: {e}")
        return None, warns


def _run_one(
    *,
    cfg: dict,
    mode_name: str,
    rebalance_scores: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    warns: List[str] = []
    bt_cfg = _get_l7_cfg(cfg)

    market_regime, w = _compute_market_regime_if_needed(cfg, rebalance_scores)
    warns.extend(w)

    # L7 backtest
    bt_positions, bt_returns, bt_equity_curve, bt_metrics, quality, w2, *_ = run_backtest(
        rebalance_scores=rebalance_scores,
        cfg=bt_cfg,
        config_cost_bps=float((cfg.get("l7", {}) or {}).get("cost_bps", bt_cfg.cost_bps)),
        market_regime=market_regime,
    )
    warns.extend(w2 or [])

    # L7C benchmark (multi)
    out_bench, w3 = run_l7c_benchmark(cfg, {"rebalance_scores": rebalance_scores, "bt_returns": bt_returns}, force=True)
    warns.extend(w3 or [])

    bt_metrics = bt_metrics.copy()
    bt_metrics["mode"] = mode_name
    bt_benchmark_compare_multi = out_bench.get("bt_benchmark_compare_multi", pd.DataFrame()).copy()
    if not bt_benchmark_compare_multi.empty:
        bt_benchmark_compare_multi["mode"] = mode_name

    return bt_metrics, bt_benchmark_compare_multi, bt_returns, warns


def _to_md_table(df: pd.DataFrame, cols: List[str]) -> str:
    if df is None or df.empty:
        return "_(empty)_\n"
    x = df.copy()
    cols2 = [c for c in cols if c in x.columns]
    x = x[cols2].copy()
    # pandas.to_markdown은 tabulate 의존성이 있어(옵션) 직접 Markdown 표를 만든다.
    header = "| " + " | ".join(cols2) + " |\n"
    sep = "| " + " | ".join(["---"] * len(cols2)) + " |\n"
    lines = [header, sep]
    for _, r in x.iterrows():
        row = []
        for c in cols2:
            v = r.get(c, "")
            if pd.isna(v):
                row.append("")
            else:
                row.append(str(v))
        lines.append("| " + " | ".join(row) + " |\n")
    return "".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Compare signal modes (model vs ranking vs dual horizon)")
    ap.add_argument("--config", type=str, default="configs/config.yaml")
    ap.add_argument("--model-rebalance", type=str, default="data/interim/option_a_only_20d/rebalance_scores.parquet")
    ap.add_argument("--ranking-single-rebalance", type=str, default="data/interim/stage15_ranking_full_20251223_073900/rebalance_scores.parquet")
    ap.add_argument("--out", type=str, default="reports/dual_horizon_comparison.md")
    args = ap.parse_args()

    root = _project_root()
    cfg_path = (root / args.config).resolve()
    cfg = load_config(str(cfg_path))

    # Load common artifacts for dual horizon
    interim = root / "data" / "interim"
    artifacts = {
        "dataset_daily": _safe_read_parquet(interim / "dataset_daily.parquet"),
        "cv_folds_short": _safe_read_parquet(interim / "cv_folds_short.parquet"),
        "universe_k200_membership_monthly": _safe_read_parquet(interim / "universe_k200_membership_monthly.parquet"),
        "ranking_short_daily": _safe_read_parquet(interim / "ranking_short_daily.parquet"),
        "ranking_long_daily": _safe_read_parquet(interim / "ranking_long_daily.parquet"),
    }
    missing = [k for k, v in artifacts.items() if v is None or len(v) == 0]
    if missing:
        raise SystemExit(f"[FAIL] missing required interim artifacts for dual horizon: {missing}")

    # Build dual horizon rebalance_scores in-memory via L6R
    dual_outputs, dual_warns = run_L6R_ranking_scoring(cfg, artifacts, force=True)
    rebalance_dual = dual_outputs["rebalance_scores"]

    # Load model + single ranking rebalance_scores
    rebalance_model = _safe_read_parquet((root / args.model_rebalance).resolve())
    rebalance_single = _safe_read_parquet((root / args.ranking_single_rebalance).resolve())
    if rebalance_model is None or rebalance_model.empty:
        raise SystemExit(f"[FAIL] model rebalance_scores not found: {args.model_rebalance}")
    if rebalance_single is None or rebalance_single.empty:
        raise SystemExit(f"[FAIL] ranking_single rebalance_scores not found: {args.ranking_single_rebalance}")

    # Run each mode
    all_metrics = []
    all_bench = []
    warns_all: Dict[str, List[str]] = {"dual_build": dual_warns or []}

    for name, rs in [
        ("model", rebalance_model),
        ("ranking_single", rebalance_single),
        ("ranking_dual", rebalance_dual),
    ]:
        bt_metrics, bench_multi, _, warns = _run_one(cfg=cfg, mode_name=name, rebalance_scores=rs)
        all_metrics.append(bt_metrics)
        if bench_multi is not None and not bench_multi.empty:
            all_bench.append(bench_multi)
        warns_all[name] = warns

    bt_metrics_all = pd.concat(all_metrics, ignore_index=True)
    bench_all = pd.concat(all_bench, ignore_index=True) if all_bench else pd.DataFrame()

    # Suggest alpha_short tuning grid
    l6r = (cfg.get("l6r", {}) if isinstance(cfg, dict) else {}) or {}
    alpha_short = l6r.get("alpha_short", None)
    regime_alpha = l6r.get("regime_alpha", None)

    md = []
    md.append("# 듀얼 호라이즌 비교 리포트 (자동 생성)\n")
    md.append(f"- 생성일: {pd.Timestamp.now(tz='Asia/Seoul').strftime('%Y-%m-%d %H:%M:%S %Z')}\n")
    md.append("\n## 1) 비교 대상\n")
    md.append(f"- model: `{args.model_rebalance}`\n")
    md.append(f"- ranking_single: `{args.ranking_single_rebalance}`\n")
    md.append("- ranking_dual: `data/interim/ranking_short_daily.parquet + ranking_long_daily.parquet` → L6R(in-memory)\n")

    md.append("\n## 2) L7 성과 요약(모드×phase)\n\n")
    cols = ["mode", "phase", "net_sharpe", "net_mdd", "net_cagr", "net_total_return", "net_hit_ratio", "n_rebalances", "avg_turnover_oneway"]
    md.append(_to_md_table(bt_metrics_all, cols))
    md.append("\n")

    md.append("\n## 3) 벤치마크 대비(멀티) 요약\n\n")
    if bench_all is None or bench_all.empty:
        md.append("_(벤치마크 비교 결과 없음)_\n")
    else:
        cols2 = ["mode", "bench_type", "phase", "information_ratio", "tracking_error_ann", "corr_vs_benchmark", "beta_vs_benchmark", "n_rebalances"]
        md.append(_to_md_table(bench_all, cols2))
        md.append("\n")

    md.append("\n## 4) α(단기/장기 결합) 튜닝 제안\n\n")
    md.append(f"- 현재 `l6r.alpha_short`: `{alpha_short}`\n")
    md.append(f"- 현재 `l6r.regime_alpha`: `{regime_alpha}`\n")
    md.append("- 추천 그리드(초기): `alpha_short ∈ {0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}`\n")
    md.append("- 평가 기준: Holdout 기준 Net Sharpe + MDD 동시 개선(동일 cost_bps/holding_days 유지)\n")

    md.append("\n## 5) 경고/메모(상위 5개)\n\n")
    for k, ws in warns_all.items():
        if not ws:
            continue
        md.append(f"- {k}:\n")
        for w in ws[:5]:
            md.append(f"  - {w}\n")

    out_path = (root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(md), encoding="utf-8")
    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()
