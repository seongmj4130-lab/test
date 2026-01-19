# -*- coding: utf-8 -*-
"""
[개선안 35번] Track B 4전략 최종 요약표(Dev/Holdout + 최종 수치셋) 자동 생성

생성물:
- artifacts/reports/track_b_4strategy_final_summary.md

포함 항목(가능한 범위에서 모두):
1) 핵심 성과(Headline): Net Sharpe, Net Total Return, Net CAGR, Net MDD, Calmar
2) 모델 예측력(Alpha Quality): IC, Rank IC, ICIR, Rank ICIR, Long/Short Alpha(ann)
3) 운용 안정성(Operational): Avg Turnover, Hit Ratio, Profit Factor, Avg Trade Duration, Avg Cost(%)
4) 국면별 성과(Regime Robustness): bull/bear/neutral별 Net Sharpe/Net MDD/Net Total Return (아티팩트 존재 시)

실행:
  cd C:\\Users\\seong\\OneDrive\\Desktop\\bootcamp\\03_code
  python scripts/generate_trackb_4strategy_final_summary.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


STRATEGIES = ["bt20_short", "bt20_ens", "bt120_long", "bt120_ens"]


def _read_csv(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _read_parquet(p: Path) -> pd.DataFrame:
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


def _fmt(x, nd: int = 3) -> str:
    try:
        if x is None:
            return ""
        if isinstance(x, str):
            return x
        fx = float(x)
        if pd.isna(fx):
            return ""
        return f"{fx:.{nd}f}"
    except Exception:
        return str(x)


def _md_table(rows: List[Dict[str, str]], cols: List[str]) -> str:
    if not rows:
        return "_(데이터 없음)_\n"
    out = []
    out.append("| " + " | ".join(cols) + " |")
    out.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        out.append("| " + " | ".join([str(r.get(c, "")) for c in cols]) + " |")
    return "\n".join(out) + "\n"


def _pick_row(df: pd.DataFrame, phase: str) -> Dict:
    if df.empty:
        return {}
    sub = df[df["phase"].astype(str) == phase]
    if sub.empty:
        return {}
    return dict(sub.iloc[0])


def main() -> None:
    interim = PROJECT_ROOT / "data" / "interim"
    out_dir = PROJECT_ROOT / "artifacts" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_by_strategy: Dict[str, pd.DataFrame] = {}
    regime_by_strategy: Dict[str, pd.DataFrame] = {}

    for s in STRATEGIES:
        metrics_by_strategy[s] = _read_csv(interim / f"bt_metrics_{s}.csv")
        regime_by_strategy[s] = _read_parquet(interim / f"bt_regime_metrics_{s}.parquet")

    # ------------------------
    # 1) Headline / Alpha / Operational (Dev/Holdout)
    # ------------------------
    headline_cols = [
        ("net_sharpe", "Net Sharpe"),
        ("net_total_return", "Net Total Return"),
        ("net_cagr", "Net CAGR"),
        ("net_mdd", "Net MDD"),
        ("net_calmar_ratio", "Calmar"),
    ]
    alpha_cols = [
        ("ic", "IC"),
        ("rank_ic", "Rank IC"),
        ("icir", "ICIR"),
        ("rank_icir", "Rank ICIR"),
        ("long_short_alpha_ann", "L/S Alpha (ann)"),
    ]
    op_cols = [
        ("avg_turnover_oneway", "Avg Turnover"),
        ("net_hit_ratio", "Hit Ratio"),
        ("net_profit_factor", "Profit Factor"),
        ("avg_trade_duration", "Avg Trade Duration"),
        ("avg_cost_pct", "Avg Cost(%)"),
        ("n_rebalances", "Rebalances"),
    ]

    def build_section(title: str, cols: List[Tuple[str, str]], nd_map: Optional[Dict[str, int]] = None) -> str:
        rows = []
        for s in STRATEGIES:
            df = metrics_by_strategy[s]
            dev = _pick_row(df, "dev")
            hold = _pick_row(df, "holdout")
            for phase, row in [("Dev", dev), ("Holdout", hold)]:
                r = {"전략": s, "구간": phase}
                for key, label in cols:
                    nd = (nd_map or {}).get(key, 3)
                    r[label] = _fmt(row.get(key) if row else None, nd=nd)
                rows.append(r)
        col_names = ["전략", "구간"] + [label for _, label in cols]
        return f"## {title}\n\n" + _md_table(rows, col_names) + "\n"

    # formatting hints
    nd_map = {
        "net_total_return": 3,
        "net_cagr": 3,
        "net_mdd": 3,
        "avg_trade_duration": 1,
        "avg_cost_pct": 3,
        "n_rebalances": 0,
        "long_short_alpha_ann": 3,
    }

    md = []
    md.append("# Track B 4전략 최종 요약표")
    md.append("")
    md.append("**근거 파일**: `data/interim/bt_metrics_{strategy}.csv`, `data/interim/bt_regime_metrics_{strategy}.parquet`")
    md.append("")
    # [개선안 36번] 오버래핑 트랜치 도입으로 bt120_*도 월별 Rebalances가 가능해짐.
    # 전략별로 실제 holding_days/n_rebalances를 보고 주의 문구를 자동 생성한다.
    caution_lines = []
    for s in STRATEGIES:
        df = metrics_by_strategy.get(s, pd.DataFrame())
        hold = _pick_row(df, "holdout")
        if not hold:
            continue
        hd = hold.get("holding_days")
        nr = hold.get("n_rebalances")
        try:
            hd_i = int(float(hd))
        except Exception:
            hd_i = None
        try:
            nr_i = int(float(nr))
        except Exception:
            nr_i = None
        if nr_i is not None and nr_i <= 5:
            caution_lines.append(f"- `{s}` Holdout은 Rebalances={nr_i}로 표본이 매우 작아 지표 변동이 큽니다.")
        if hd_i is not None and hd_i >= 60:
            caution_lines.append(f"- `{s}`는 holding_days={hd_i}로 긴 주기 전략이라 구간별 표본이 작을 수 있습니다.")
    if caution_lines:
        md.append("> 주의:")
        md.extend(caution_lines)
    else:
        md.append("> 주의: 모든 전략이 충분한 Rebalances를 확보했습니다.")
    md.append("")

    md.append(build_section("1) 핵심 성과 (Headline Metrics)", headline_cols, nd_map=nd_map))
    md.append(build_section("2) 모델 예측력 (Alpha Quality)", alpha_cols, nd_map=nd_map))
    md.append(build_section("3) 운용 안정성 (Operational Viability)", op_cols, nd_map=nd_map))

    # ------------------------
    # 4) Regime Robustness
    # ------------------------
    md.append("## 4) 국면별 성과 (Regime Robustness)\n")
    any_regime = False
    for s in STRATEGIES:
        df = regime_by_strategy[s]
        md.append(f"### {s}")
        if df is None or df.empty:
            md.append("_(bt_regime_metrics 아티팩트 없음)_\n")
            continue
        any_regime = True
        # keep only essential columns
        cols = ["phase", "regime", "n_rebalances", "net_total_return", "net_sharpe", "net_mdd", "net_cagr", "net_hit_ratio", "date_start", "date_end"]
        cols = [c for c in cols if c in df.columns]
        sub = df[cols].copy()
        # formatting
        for c in ["net_total_return", "net_sharpe", "net_mdd", "net_cagr", "net_hit_ratio"]:
            if c in sub.columns:
                sub[c] = pd.to_numeric(sub[c], errors="coerce").round(4)
        md.append(sub.to_string(index=False))
        md.append("")

    if not any_regime:
        md.append("_(regime 데이터가 없어 국면별 표를 생성하지 못했습니다.)_\n")

    out_path = out_dir / "track_b_4strategy_final_summary.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"✅ 저장 완료: {out_path}")


if __name__ == "__main__":
    main()
