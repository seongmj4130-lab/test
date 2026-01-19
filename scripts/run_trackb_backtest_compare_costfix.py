# -*- coding: utf-8 -*-
"""
[개선안 1번][개선안 3번] Track B 백테스트 재실행 + 수정 전/후 성과 비교

목표:
- L7 비용 모델(턴오버 기반) 수정이 실제 백테스트 성과지표에 어떤 영향을 주는지 자동 비교
- 기존 파일(bt_metrics_*.csv)이 존재하면 "수정 전"으로 읽고, 재실행 후 "수정 후"와 비교표를 생성

실행:
  cd C:\\Users\\seong\\OneDrive\\Desktop\\bootcamp\\03_code
  python scripts/run_trackb_backtest_compare_costfix.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.tracks.track_b.backtest_service import run_backtest_strategy

STRATEGIES = ["bt20_short", "bt20_ens", "bt120_long", "bt120_ens"]


def _read_metrics_csv(interim_dir: Path, strategy: str) -> pd.DataFrame:
    p = interim_dir / f"bt_metrics_{strategy}.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def _pick_cols(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "phase",
        "net_sharpe",
        "net_total_return",
        "net_cagr",
        "net_mdd",
        "net_calmar_ratio",
        "net_profit_factor",
        "net_hit_ratio",
        "avg_turnover_oneway",
        "avg_trade_duration",
        "avg_cost_pct",
        "gross_minus_net_total_return_pct",
        "n_rebalances",
        "date_start",
        "date_end",
    ]
    avail = [c for c in cols if c in df.columns]
    return df[avail].copy()


def _to_md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(데이터 없음)_\n"
    # pandas.to_markdown은 tabulate 의존성이 있어(옵션) 프로젝트 환경에서 실패할 수 있음
    # -> 의존성 없이 최소한의 markdown table 출력
    cols = list(df.columns)
    lines = []
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, r in df.iterrows():
        vals = []
        for c in cols:
            v = r.get(c)
            if v is None or (isinstance(v, float) and pd.isna(v)):
                vals.append("")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    interim_dir = project_root / "data" / "interim"
    report_dir = project_root / "artifacts" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    before: Dict[Tuple[str, str], dict] = {}
    after: Dict[Tuple[str, str], dict] = {}

    # 1) 수정 전(현재 파일) 읽기
    for s in STRATEGIES:
        df0 = _read_metrics_csv(interim_dir, s)
        if df0.empty:
            continue
        for _, r in df0.iterrows():
            key = (s, str(r.get("phase")))
            before[key] = dict(r)

    # 2) 백테스트 재실행 (L6R 캐시 사용, L7은 항상 재계산/저장)
    for s in STRATEGIES:
        run_backtest_strategy(strategy=s, config_path="configs/config.yaml", force_rebuild=False)

    # 3) 수정 후 읽기
    for s in STRATEGIES:
        df1 = _read_metrics_csv(interim_dir, s)
        if df1.empty:
            continue
        for _, r in df1.iterrows():
            key = (s, str(r.get("phase")))
            after[key] = dict(r)

    # 4) 비교표 생성
    rows = []
    metrics = [
        ("net_sharpe", "Net Sharpe Ratio"),
        ("net_total_return", "Net Total Return"),
        ("net_cagr", "Net CAGR"),
        ("net_mdd", "Net MDD"),
        ("net_calmar_ratio", "Calmar Ratio"),
        ("net_profit_factor", "Profit Factor"),
        ("net_hit_ratio", "Hit Ratio"),
        ("avg_turnover_oneway", "Avg Turnover (one-way)"),
        ("avg_trade_duration", "Avg Trade Duration"),
        ("avg_cost_pct", "Avg Cost (%)"),
        ("gross_minus_net_total_return_pct", "Gross - Net Total Return"),
    ]

    for s in STRATEGIES:
        for phase in ["dev", "holdout"]:
            key = (s, phase)
            b = before.get(key, None)
            a = after.get(key, None)
            for col, label in metrics:
                rows.append(
                    {
                        "strategy": s,
                        "phase": phase,
                        "metric": label,
                        "before": None if b is None else b.get(col),
                        "after": None if a is None else a.get(col),
                    }
                )

    comp = pd.DataFrame(rows)
    # delta 계산(숫자일 때만)
    comp["delta"] = pd.to_numeric(comp["after"], errors="coerce") - pd.to_numeric(comp["before"], errors="coerce")

    # 5) 리포트 저장
    md_lines = []
    md_lines.append("# Track B 백테스트 결과 (거래비용 모델 수정 전/후 비교)")
    md_lines.append("")
    md_lines.append("**변경 요약**: L7의 거래비용 차감이 `턴오버 기반`으로 적용되도록 수정 (고정 10bp 차감 제거) + slippage_bps 옵션 추가")
    md_lines.append("")
    md_lines.append("## 전략별 원본 bt_metrics (수정 후)")
    for s in STRATEGIES:
        md_lines.append(f"### {s}")
        df_now = _pick_cols(_read_metrics_csv(interim_dir, s))
        md_lines.append(_to_md_table(df_now))
        md_lines.append("")

    md_lines.append("## 수정 전/후 비교표")
    md_lines.append(_to_md_table(comp))
    md_lines.append("")
    md_lines.append("## 다음 액션(추천)")
    md_lines.append("- `slippage_bps`를 0.0→5.0 등으로 올려 현실성 점검(성과 과대추정 여부 확인)")
    md_lines.append("- bt20의 MDD가 여전히 크면: `regime.exposure_bear_*` 추가 축소 또는 `volatility_adjustment_min/max` 튜닝")
    md_lines.append("")

    out_path = report_dir / "track_b_backtest_results_after_cost_model_fix.md"
    out_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"✅ 저장 완료: {out_path}")


if __name__ == "__main__":
    main()
