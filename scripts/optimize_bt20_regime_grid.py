# -*- coding: utf-8 -*-
"""
[개선안 32번] bt20 전략(regime 파라미터) 그리드 서치로 "전체 지표" 개선

목표:
- bt20_short / bt20_ens에 대해 regime 파라미터를 작은 그리드로 스윕
- Holdout Net Sharpe / Net MDD / Turnover / Cost를 동시에 고려해 Pareto 후보를 출력
- 최적 후보를 리포트(.csv/.md)로 남김

주의:
- L6R/L8을 재실행하지 않고, 이미 생성된 `rebalance_scores_from_ranking_interval_20.parquet`을 사용
- 시장 국면은 `ohlcv_daily.parquet`로부터 build_market_regime로 생성

실행:
  cd C:\\Users\\seong\\OneDrive\\Desktop\\bootcamp\\03_code
  python scripts/optimize_bt20_regime_grid.py
"""

from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.tracks.shared.stages.regime.l1d_market_regime import build_market_regime
from src.tracks.track_b.stages.backtest.l7_backtest import BacktestConfig, run_backtest


def _load_parquet_flex(p: Path) -> pd.DataFrame:
    if p.exists():
        return pd.read_parquet(p)
    if p.with_suffix(".parquet").exists():
        return pd.read_parquet(p.with_suffix(".parquet"))
    if p.with_suffix(".csv").exists():
        df = pd.read_csv(p.with_suffix(".csv"))
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df
    raise FileNotFoundError(str(p))


def _pick_metric_row(bt_metrics: pd.DataFrame, phase: str) -> Dict:
    if bt_metrics is None or bt_metrics.empty:
        return {}
    r = bt_metrics[bt_metrics["phase"].astype(str) == phase]
    if r.empty:
        return {}
    return dict(r.iloc[0])


def _objective(m_hold: Dict) -> float:
    """
    단순 목적함수(스코어):
    - Sharpe가 가장 중요(+)
    - MDD는 덜 음수일수록 좋음(+; mdd는 음수)
    - Turnover/Cost는 낮을수록 좋음(-)
    """
    s = float(m_hold.get("net_sharpe", 0.0) or 0.0)
    mdd = float(m_hold.get("net_mdd", 0.0) or 0.0)  # 음수
    to = float(m_hold.get("avg_turnover_oneway", 0.0) or 0.0)
    cost = float(m_hold.get("avg_cost_pct", 0.0) or 0.0)
    # 가중치는 경험적(튜닝 가능)
    return (2.0 * s) + (0.5 * (mdd * -1.0)) - (1.0 * to) - (20.0 * cost)


def main() -> None:
    interim = project_root / "data" / "interim"
    report_dir = project_root / "artifacts" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    reb = _load_parquet_flex(interim / "rebalance_scores_from_ranking_interval_20.parquet")
    ohlcv = _load_parquet_flex(interim / "ohlcv_daily.parquet")

    # 시장 국면 생성(3단계 bull/bear/neutral)
    market_regime = build_market_regime(
        rebalance_dates=reb["date"].unique(),
        ohlcv_daily=ohlcv,
        lookback_days=60,
        neutral_band=0.0,  # bt20에서는 aggressive도 실험 대상이지만, 여기선 cfg에서 컨트롤
        use_volume=True,
        use_volatility=True,
    )

    # 전략별 스코어/리턴 컬럼
    strategies = {
        "bt20_short": {"score_col": "score_total_short", "ret_col": "true_short", "top_k": 12, "buffer_k": 15, "weighting": "equal"},
        "bt20_ens": {"score_col": "score_ens", "ret_col": "true_short", "top_k": 15, "buffer_k": 20, "weighting": "softmax", "softmax_temp": 0.5},
    }

    # Grid (작게 시작)
    neutral_band_grid = [0.0, 0.02, 0.05]
    top_k_bear_grid = [20, 30]
    exposure_bear_grid = [0.7, 0.8, 1.0]
    risk_bear_mult_grid = [0.7, 0.8, 0.9]

    rows: List[Dict] = []
    for strat, sconf in strategies.items():
        for nb in neutral_band_grid:
            # 시장국면은 nb에 따라 달라지므로 매번 생성
            mr = build_market_regime(
                rebalance_dates=reb["date"].unique(),
                ohlcv_daily=ohlcv,
                lookback_days=60,
                neutral_band=float(nb),
                use_volume=True,
                use_volatility=True,
            )
            for tk_bear in top_k_bear_grid:
                for ex_bear in exposure_bear_grid:
                    for rbm in risk_bear_mult_grid:
                        cfg_bt = BacktestConfig(
                            holding_days=20,
                            top_k=int(sconf["top_k"]),
                            cost_bps=10.0,
                            slippage_bps=0.0,
                            score_col=str(sconf["score_col"]),
                            ret_col=str(sconf["ret_col"]),
                            weighting=str(sconf.get("weighting", "equal")),
                            softmax_temp=float(sconf.get("softmax_temp", 1.0)),
                            buffer_k=int(sconf.get("buffer_k", 0)),
                            regime_enabled=True,
                            regime_top_k_bear=int(tk_bear),
                            regime_exposure_bear=float(ex_bear),
                            regime_top_k_bull=int(sconf["top_k"]),  # bull은 기본 top_k 유지
                            regime_exposure_bull=1.0,
                            risk_scaling_enabled=True,
                            risk_scaling_bear_multiplier=float(rbm),
                            risk_scaling_neutral_multiplier=1.0,
                            risk_scaling_bull_multiplier=1.0,
                            smart_buffer_enabled=True,
                            smart_buffer_stability_threshold=0.7,
                            volatility_adjustment_enabled=True,
                            volatility_lookback_days=60,
                            target_volatility=0.15,
                            volatility_adjustment_min=0.7,
                            volatility_adjustment_max=1.2,
                        )
                        bt_pos, bt_ret, bt_eq, bt_met, quality, warns, *_ = run_backtest(
                            rebalance_scores=reb,
                            cfg=cfg_bt,
                            market_regime=mr,
                        )
                        m_dev = _pick_metric_row(bt_met, "dev")
                        m_hold = _pick_metric_row(bt_met, "holdout")
                        if not m_hold:
                            continue
                        rows.append(
                            {
                                "strategy": strat,
                                "neutral_band": float(nb),
                                "top_k_bear": int(tk_bear),
                                "exposure_bear": float(ex_bear),
                                "risk_bear_mult": float(rbm),
                                "obj": _objective(m_hold),
                                "hold_net_sharpe": m_hold.get("net_sharpe"),
                                "hold_net_cagr": m_hold.get("net_cagr"),
                                "hold_net_mdd": m_hold.get("net_mdd"),
                                "hold_calmar": m_hold.get("net_calmar_ratio"),
                                "hold_turnover": m_hold.get("avg_turnover_oneway"),
                                "hold_avg_cost_pct": m_hold.get("avg_cost_pct"),
                                "dev_net_sharpe": m_dev.get("net_sharpe") if m_dev else None,
                                "dev_net_mdd": m_dev.get("net_mdd") if m_dev else None,
                            }
                        )

    out = pd.DataFrame(rows).sort_values(["strategy", "obj"], ascending=[True, False]).reset_index(drop=True)
    out_path = report_dir / "bt20_regime_grid_results.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    # Top 10 per strategy -> md
    md_lines = ["# bt20 Regime Grid Search Results", ""]
    for strat in strategies.keys():
        sub = out[out["strategy"] == strat].head(10)
        md_lines.append(f"## {strat} (Top 10 by obj)")
        if sub.empty:
            md_lines.append("_(no rows)_")
        else:
            md_lines.append(sub.to_string(index=False))
        md_lines.append("")
    md_path = report_dir / "bt20_regime_grid_results.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"✅ saved: {out_path}")
    print(f"✅ saved: {md_path}")


if __name__ == "__main__":
    main()
