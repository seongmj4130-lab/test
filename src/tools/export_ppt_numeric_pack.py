"""
[개선안 12번] 최종 PPT 숫자 산출물(100% 근거 기반) Exporter

절대 규칙:
- 추정/가정 금지. 파일/코드/아티팩트에서 읽히는 값만 사용.
- 파일이 없으면 '알 수 없습니다(근거 파일 없음)'로 기록하고 계속 진행.
- 모든 출력은 reports/extract/ppt/ 아래에 저장.
- RUN_TAG/BASELINE_TAG 두 개를 모두 처리(있으면).

실행:
  python src/tools/export_ppt_numeric_pack.py
  python src/tools/export_ppt_numeric_pack.py --run-tag stage14_checklist_final_20251222_141500 --baseline-tag stage12_final_export_20251221_013411
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

UNKNOWN = "알 수 없습니다(근거 파일 없음)"

DEFAULT_RUN_TAG = "stage14_checklist_final_20251222_141500"
DEFAULT_BASELINE_TAG = "stage12_final_export_20251221_013411"


@dataclass(frozen=True)
class FoundArtifact:
    path: Optional[Path]
    source_kind: str  # extract|tag|root|missing
    note: str


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, text: str) -> None:
    _ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8", errors="replace")


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fieldnames})


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _parse_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip()
        if not s or s in ("N/A", "nan", "NaN", "None"):
            return None
        return float(s)
    except Exception:
        return None


def _safe_num_or_unknown(x: Optional[float]) -> Any:
    return x if x is not None and not (isinstance(x, float) and (math.isnan(x) or math.isinf(x))) else UNKNOWN


def _ratio_classify(a: Optional[float], b: Optional[float]) -> str:
    """추정 금지: 스케일(1x/100x/unknown)만 분류."""
    if a is None or b is None:
        return "unit_unknown"
    if abs(a) < 1e-12 or abs(b) < 1e-12:
        return "unit_unknown"
    r = a / b
    if abs(r - 1.0) < 0.05:
        return "unit_same_scale"
    if abs(abs(r) - 100.0) < 5.0:
        return "unit_mixed_percent_vs_decimal"
    if abs(abs(r) - 0.01) < 0.001:
        return "unit_mixed_percent_vs_decimal"
    return "unit_unknown"


def find_project_root_by_config(cwd: Path) -> Path:
    """
    configs/config.yaml이 있는 폴더를 자동 탐색해서 루트로 사용.
    없으면 cwd 반환(근거 부족).
    """
    for p in [cwd, *cwd.parents[:8]]:
        if (p / "configs" / "config.yaml").exists():
            return p
    return cwd


def find_artifact(project_root: Path, extract_dir: Path, tag: str, name: str) -> FoundArtifact:
    """
    아티팩트 탐색 우선순위:
    1) reports/extract/charts/{tag}__{name}.csv
    2) data/interim/{tag}/{name}.parquet|csv
    3) data/interim/{name}.parquet|csv
    """
    p1 = extract_dir / "charts" / f"{tag}__{name}.csv"
    if p1.exists():
        return FoundArtifact(path=p1, source_kind="extract", note=str(p1.relative_to(project_root)))
    base_tag = project_root / "data" / "interim" / tag
    for ext in (".parquet", ".csv"):
        p2 = base_tag / f"{name}{ext}"
        if p2.exists():
            return FoundArtifact(path=p2, source_kind="tag", note=str(p2.relative_to(project_root)))
    base_root = project_root / "data" / "interim"
    for ext in (".parquet", ".csv"):
        p3 = base_root / f"{name}{ext}"
        if p3.exists():
            return FoundArtifact(path=p3, source_kind="root", note=str(p3.relative_to(project_root)))
    return FoundArtifact(path=None, source_kind="missing", note=UNKNOWN)


def load_df(art: FoundArtifact) -> Tuple[Optional[pd.DataFrame], str]:
    if art.path is None:
        return None, art.note
    p = art.path
    try:
        if p.suffix.lower() == ".csv":
            return pd.read_csv(p), art.note
        if p.suffix.lower() == ".parquet":
            return pd.read_parquet(p), art.note
        return None, art.note
    except Exception:
        return None, art.note


def inventory_sources(project_root: Path, extract_dir: Path, ppt_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    [1] PPT 숫자 산출물 목록(Inventory) 생성
    - README.md, charts_manifest.csv, kpi_snapshot.csv를 읽어 숫자 소스 파일 목록화
    - PPT 항목 카테고리별(source_file, source_columns, derive_method) 매핑
    """
    sources: List[Dict[str, Any]] = []

    # 1) 존재 파일 목록화
    for rel in ["README.md", "charts_manifest.csv", "kpi_snapshot.csv", "integrity_report.md", "backtest_contract.md", "features_final.json"]:
        p = extract_dir / rel
        sources.append({"source_file": str(p.relative_to(project_root)), "exists": p.exists()})

    charts_manifest = extract_dir / "charts_manifest.csv"
    if charts_manifest.exists():
        cm = pd.read_csv(charts_manifest)
        for _, r in cm.iterrows():
            exported = str(r.get("exported_csv") or "")
            src_path = str(r.get("source_path") or "")
            sources.append({"source_file": exported if exported else src_path, "exists": True})

    # unique
    seen = set()
    src_rows = []
    for r in sources:
        k = (r.get("source_file"), r.get("exists"))
        if k in seen:
            continue
        seen.add(k)
        src_rows.append(r)

    # 2) PPT 숫자 항목 고정 매핑(근거 파일이 없으면 그대로 UNKNOWN)
    items: List[Dict[str, Any]] = []
    def add(cat: str, item: str, source_file: str, source_cols: str, derive: str) -> None:
        items.append(
            {
                "category": cat,
                "item": item,
                "source_file": source_file,
                "source_columns": source_cols,
                "derive_method": derive,
            }
        )

    add("executive_kpi", "holdout_kpi_core", "reports/extract/kpi_snapshot.csv", "net_total_return,net_cagr,net_sharpe,net_mdd,cost_bps_used,n_rebalances", "direct (holdout columns in md table parsed)")
    add("executive_kpi", "avg_turnover_oneway/avg_total_cost", "data/interim/{tag}/bt_metrics.parquet or bt_returns.csv", "avg_turnover_oneway + total_cost", "bt_metrics holdout preferred; else mean(bt_returns.total_cost) / mean(bt_returns.turnover_oneway)")
    add("executive_kpi", "bench_total_return/excess_total_return", "bt_benchmark_returns + bt_vs_benchmark", "bench_return/excess_return", "cumprod(1+r)-1 by phase")

    add("data_snapshot", "dataset_daily_shape", "data/interim/{tag}/dataset_daily.parquet or data/interim/dataset_daily.parquet", "rows,cols", "direct")
    add("data_snapshot", "unique_tickers/dates/dup", "dataset_daily.parquet", "ticker,date", "nunique + duplicated(ticker,date)")
    add("data_snapshot", "feature_stats", "dataset_daily.parquet + reports/extract/features_final.json", "features_final", "NaN%, min/p01/median/p99/max")

    add("model_validation", "model_metrics", "data/interim/{tag}/model_metrics.parquet or data/interim/model_metrics.parquet", "rmse/mae/ic_rank/hit_ratio/...", "direct export if exists")
    add("model_validation", "pred_based_metrics", "pred_short_oos/pred_long_oos", "y_true/y_pred", "recompute per phase (ic_rank, corr, rmse, hit_ratio)")

    add("ranking_quality", "top_bottom_spread", "rebalance_scores or ranking_daily", "score + return", "top_k mean - bottom_k mean by date")
    add("ranking_quality", "daily_ic", "rebalance_scores or ranking_daily", "score + return", "Spearman corr(score, return) by date")

    add("backtest_performance", "return_distribution", "bt_returns.csv", "net_return", "p10/p50/p90/min/max by phase")
    add("backtest_performance", "monthly_returns", "bt_returns + bt_benchmark_returns/bt_vs_benchmark", "net_return/bench_return/excess_return", "monthly compounded within month")

    add("risk_drawdown", "drawdown_key_events", "bt_drawdown_events.csv", "peak_date,trough_date,drawdown,length_days", "top 3 most negative drawdowns")
    add("stability", "yearly_metrics", "bt_yearly_metrics.csv", "year,net_total_return,net_sharpe,net_mdd,...", "direct export")
    add("stability", "rolling_sharpe_summary", "data/interim/bt_rolling_sharpe.parquet", "rolling_sharpe", "min/max/mean + min date by phase")

    add("sensitivity", "grid", "bt_sensitivity.csv", "grid_* + net_sharpe", "direct export")
    add("sensitivity", "top10pct_sharpe_ranges", "bt_sensitivity.csv", "net_sharpe + grid", "holdout, threshold=90th percentile; min/max of grid params")

    add("benchmark_fairness", "benchmark_type", "reports/extract/backtest_contract.md", "L7C build_universe_benchmark_returns", "code contract: mean(universe returns) benchmark")
    add("benchmark_fairness", "benchmark_perf", "bt_benchmark_returns.csv", "bench_return/bench_equity", "CAGR/Sharpe/MDD computed from series")
    add("benchmark_fairness", "config_mismatch_table", "reports/extract/integrity_report.md + config_snapshot.md", "cost_bps_used vs config", "direct parse")

    add("repro_runtime", "cli_options", "reports/extract/cli_run_all.md", "argparse table", "direct")
    add("repro_runtime", "commands_log", "reports/extract/repro_commands.md", "Command lines", "direct")

    # write inventory outputs
    inv_md = []
    inv_md.append("# PPT Numeric Inventory")
    inv_md.append("")
    inv_md.append("## 숫자 소스 파일 목록(발견된 것)")
    inv_md.append("")
    inv_md.append("| source_file | exists |")
    inv_md.append("|---|---|")
    for r in src_rows:
        inv_md.append(f"| `{r['source_file']}` | {r['exists']} |")
    inv_md.append("")
    inv_md.append("## PPT 숫자 항목 매핑")
    inv_md.append("")
    inv_md.append("| category | item | source_file | source_columns | derive_method |")
    inv_md.append("|---|---|---|---|---|")
    for it in items:
        inv_md.append(
            f"| {it['category']} | {it['item']} | `{it['source_file']}` | `{it['source_columns']}` | {it['derive_method']} |"
        )
    inv_md.append("")
    _write_text(ppt_dir / "ppt_numeric_inventory.md", "\n".join(inv_md))
    _write_csv(
        ppt_dir / "ppt_numeric_inventory.csv",
        items,
        fieldnames=["category", "item", "source_file", "source_columns", "derive_method"],
    )
    return src_rows, items


def compute_total_return_from_series(r: pd.Series) -> Optional[float]:
    try:
        x = r.astype(float).to_numpy()
        eq = (1.0 + pd.Series(x)).cumprod()
        return float(eq.iloc[-1] - 1.0) if len(eq) else None
    except Exception:
        return None


def compute_mean_total_cost(bt_returns: pd.DataFrame) -> Optional[float]:
    if bt_returns is None or "total_cost" not in bt_returns.columns:
        return None
    try:
        return float(bt_returns["total_cost"].astype(float).mean())
    except Exception:
        return None


def compute_mean_turnover(bt_returns: pd.DataFrame) -> Optional[float]:
    if bt_returns is None or "turnover_oneway" not in bt_returns.columns:
        return None
    try:
        return float(bt_returns["turnover_oneway"].astype(float).mean())
    except Exception:
        return None


def compute_bench_metrics(bench_returns: pd.DataFrame, holding_days: Optional[float]) -> Dict[str, Optional[float]]:
    """
    bench_returns: columns phase, date, bench_return, bench_equity
    """
    out: Dict[str, Optional[float]] = {"bench_total_return": None, "bench_cagr": None, "bench_sharpe": None, "bench_mdd": None}
    if bench_returns is None:
        return out
    if "bench_return" not in bench_returns.columns or "bench_equity" not in bench_returns.columns:
        return out
    try:
        r = bench_returns["bench_return"].astype(float)
        out["bench_total_return"] = compute_total_return_from_series(r)
    except Exception:
        pass
    # sharpe/periods_per_year
    if holding_days is not None and holding_days > 0 and "bench_return" in bench_returns.columns:
        try:
            periods_per_year = 252.0 / float(holding_days)
            x = bench_returns["bench_return"].astype(float).to_numpy()
            if len(x) > 1:
                out["bench_sharpe"] = float((np.mean(x) / (np.std(x, ddof=1) + 1e-12)) * np.sqrt(periods_per_year))
        except Exception:
            pass
    # mdd from equity
    try:
        eq = bench_returns["bench_equity"].astype(float)
        peak = eq.cummax()
        dd = (eq / peak) - 1.0
        out["bench_mdd"] = float(dd.min()) if len(dd) else None
    except Exception:
        pass
    # cagr from dates
    try:
        d0 = pd.to_datetime(bench_returns["date"].iloc[0])
        d1 = pd.to_datetime(bench_returns["date"].iloc[-1])
        years = max(((d1 - d0).days / 365.25), 1e-9)
        if out["bench_total_return"] is not None:
            eq_end = 1.0 + float(out["bench_total_return"])
            out["bench_cagr"] = float(eq_end ** (1.0 / years) - 1.0)
    except Exception:
        pass
    return out


def build_executive_kpi(project_root: Path, extract_dir: Path, ppt_dir: Path, run_tag: str, baseline_tag: str) -> None:
    """
    [2] Executive KPI 테이블(태그별 1행) 생성
    - kpi_snapshot.csv 우선
    - 보강: bt_metrics / bt_returns / bt_vs_benchmark / bt_benchmark_returns
    - unit_flag 추가
    """
    kpi_path = extract_dir / "kpi_snapshot.csv"
    kpi_df = pd.read_csv(kpi_path) if kpi_path.exists() else pd.DataFrame()

    out_rows: List[Dict[str, Any]] = []
    unit_md_lines: List[str] = []
    unit_md_lines.append("# PPT KPI Units Check")
    unit_md_lines.append("")
    unit_md_lines.append(f"- source_kpi_snapshot: `{str(kpi_path.relative_to(project_root)) if kpi_path.exists() else UNKNOWN}`")
    unit_md_lines.append("")

    for tag in [run_tag, baseline_tag]:
        row: Dict[str, Any] = {
            "tag": tag,
            "phase": "holdout",
            "net_total_return": UNKNOWN,
            "net_cagr": UNKNOWN,
            "net_sharpe": UNKNOWN,
            "net_mdd": UNKNOWN,
            "cost_bps_used": UNKNOWN,
            "n_rebalances": UNKNOWN,
            "avg_turnover_oneway": UNKNOWN,
            "avg_total_cost": UNKNOWN,
            "bench_total_return": UNKNOWN,
            "excess_total_return": UNKNOWN,
            "unit_flag": "unit_unknown",
        }

        # kpi_snapshot 우선
        if not kpi_df.empty and "tag" in kpi_df.columns:
            rr = kpi_df[kpi_df["tag"].astype(str) == str(tag)]
            if len(rr) > 0:
                r0 = rr.iloc[0].to_dict()
                for k in ["net_total_return", "net_cagr", "net_sharpe", "net_mdd", "cost_bps_used", "n_rebalances"]:
                    if k in r0 and r0[k] is not None and str(r0[k]) != "nan":
                        row[k] = r0[k]

        # bt_returns (holdout) for derived total return / avg turnover / avg total cost
        bt_ret_art = find_artifact(project_root, extract_dir, tag, "bt_returns")
        bt_returns, bt_returns_src = load_df(bt_ret_art)
        bt_h = None
        holding_days = None
        if bt_returns is not None and "phase" in bt_returns.columns:
            bt_h = bt_returns[bt_returns["phase"].astype(str) == "holdout"].copy()
            if len(bt_h) > 0 and "holding_days" in bt_h.columns:
                holding_days = _parse_float(bt_h["holding_days"].iloc[0])
        if bt_h is not None and len(bt_h) > 0:
            # derive net_total_return if missing
            if row["net_total_return"] == UNKNOWN and "net_return" in bt_h.columns:
                v = compute_total_return_from_series(bt_h["net_return"])
                row["net_total_return"] = _safe_num_or_unknown(v)
            # n_rebalances
            if row["n_rebalances"] == UNKNOWN:
                row["n_rebalances"] = int(len(bt_h))
            # avg turnover
            v_turn = compute_mean_turnover(bt_h)
            row["avg_turnover_oneway"] = _safe_num_or_unknown(v_turn)
            # avg total cost
            v_cost = compute_mean_total_cost(bt_h)
            row["avg_total_cost"] = _safe_num_or_unknown(v_cost)

        # bt_metrics (holdout) for missing core metrics and avg_turnover_oneway (preferred)
        bt_met_art = find_artifact(project_root, extract_dir, tag, "bt_metrics")
        bt_metrics, bt_metrics_src = load_df(bt_met_art)
        if bt_metrics is not None and "phase" in bt_metrics.columns:
            mh = bt_metrics[bt_metrics["phase"].astype(str) == "holdout"]
            if len(mh) > 0:
                m0 = mh.iloc[0].to_dict()
                for k in ["net_total_return", "net_cagr", "net_sharpe", "net_mdd", "cost_bps_used", "n_rebalances", "avg_turnover_oneway"]:
                    if row.get(k) == UNKNOWN and k in m0:
                        row[k] = m0[k]
                # avg_total_cost from avg_cost_pct? only if available and direct total_cost not available
                if row["avg_total_cost"] == UNKNOWN and "avg_cost_pct" in m0:
                    # avg_cost_pct is percent of equity; cannot convert without assumption -> do not compute
                    row["avg_total_cost"] = UNKNOWN

        # bt_vs_benchmark / bt_benchmark_returns for bench/excess totals
        bt_vs_art = find_artifact(project_root, extract_dir, tag, "bt_vs_benchmark")
        bt_vs, bt_vs_src = load_df(bt_vs_art)
        if bt_vs is not None and "phase" in bt_vs.columns:
            hv = bt_vs[bt_vs["phase"].astype(str) == "holdout"].copy()
            if len(hv) > 0 and "excess_return" in hv.columns:
                row["excess_total_return"] = _safe_num_or_unknown(compute_total_return_from_series(hv["excess_return"]))
            if holding_days is None and len(hv) > 0 and "holding_days" in hv.columns:
                holding_days = _parse_float(hv["holding_days"].iloc[0])

        bench_art = find_artifact(project_root, extract_dir, tag, "bt_benchmark_returns")
        bench_df, bench_src = load_df(bench_art)
        if bench_df is not None and "phase" in bench_df.columns:
            hb = bench_df[bench_df["phase"].astype(str) == "holdout"].copy()
            if len(hb) > 0:
                bm = compute_bench_metrics(hb, holding_days)
                if bm.get("bench_total_return") is not None:
                    row["bench_total_return"] = bm["bench_total_return"]

        # unit_flag: kpi_snapshot(net_total_return) vs derived(bt_returns net_total_return)
        kpi_nt = _parse_float(row["net_total_return"])  # may be percent-scale or decimal
        derived_nt = None
        if bt_h is not None and len(bt_h) > 0 and "net_return" in bt_h.columns:
            derived_nt = compute_total_return_from_series(bt_h["net_return"])
        row["unit_flag"] = _ratio_classify(kpi_nt, derived_nt)

        unit_md_lines.append(f"## {tag}")
        unit_md_lines.append("")
        unit_md_lines.append(f"- bt_returns_source: `{bt_returns_src}`")
        unit_md_lines.append(f"- bt_metrics_source: `{bt_metrics_src}`")
        unit_md_lines.append(f"- bt_vs_benchmark_source: `{bt_vs_src}`")
        unit_md_lines.append(f"- bt_benchmark_returns_source: `{bench_src}`")
        unit_md_lines.append(f"- unit_flag: `{row['unit_flag']}`")
        if kpi_nt is None or derived_nt is None:
            unit_md_lines.append(f"- unit_check_detail: `{UNKNOWN}`")
        else:
            unit_md_lines.append(f"- unit_check_detail: `net_total_return kpi={kpi_nt}, derived={derived_nt}, ratio={kpi_nt/derived_nt if abs(derived_nt)>1e-12 else 'inf'}`")
        unit_md_lines.append("")

        out_rows.append(row)

    _write_csv(
        ppt_dir / "ppt_executive_kpi.csv",
        out_rows,
        fieldnames=[
            "tag",
            "phase",
            "net_total_return",
            "net_cagr",
            "net_sharpe",
            "net_mdd",
            "cost_bps_used",
            "n_rebalances",
            "avg_turnover_oneway",
            "avg_total_cost",
            "bench_total_return",
            "excess_total_return",
            "unit_flag",
        ],
    )
    _write_text(ppt_dir / "ppt_kpi_units_check.md", "\n".join(unit_md_lines).rstrip() + "\n")


def build_data_snapshot(project_root: Path, extract_dir: Path, ppt_dir: Path, run_tag: str, baseline_tag: str) -> None:
    """
    [3] Data Snapshot 생성(dataset_daily 기준) + phase별 start/end (bt_returns)
    """
    # dataset_daily: RUN_TAG 우선, 없으면 root fallback
    ds_art = FoundArtifact(path=None, source_kind="missing", note=UNKNOWN)
    tag_dir = project_root / "data" / "interim" / run_tag
    p_tag = tag_dir / "dataset_daily.parquet"
    if p_tag.exists():
        ds_art = FoundArtifact(path=p_tag, source_kind="tag", note=str(p_tag.relative_to(project_root)))
    else:
        p_root = project_root / "data" / "interim" / "dataset_daily.parquet"
        if p_root.exists():
            ds_art = FoundArtifact(path=p_root, source_kind="root", note=str(p_root.relative_to(project_root)))

    ds_df = None
    if ds_art.path is not None:
        try:
            ds_df = pd.read_parquet(ds_art.path)
        except Exception:
            ds_df = None

    # features list
    feats_path = extract_dir / "features_final.json"
    feats: List[str] = []
    if feats_path.exists():
        try:
            j = json.loads(_read_text(feats_path))
            feats = list(j.get("features_final", []) or [])
        except Exception:
            feats = []

    # phase date ranges from bt_returns (both tags)
    phase_rows: List[Dict[str, Any]] = []
    for tag in [run_tag, baseline_tag]:
        art = find_artifact(project_root, extract_dir, tag, "bt_returns")
        df, src = load_df(art)
        if df is None or "phase" not in df.columns or "date" not in df.columns:
            phase_rows.append({"tag": tag, "phase": UNKNOWN, "date_min": "", "date_max": "", "n_rows": "", "source": src})
            continue
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        for ph, g in d.groupby("phase", sort=False):
            dmin = str(g["date"].min().date()) if g["date"].notna().any() else ""
            dmax = str(g["date"].max().date()) if g["date"].notna().any() else ""
            phase_rows.append({"tag": tag, "phase": str(ph), "date_min": dmin, "date_max": dmax, "n_rows": int(len(g)), "source": src})

    # snapshot row
    snap: Dict[str, Any] = {
        "dataset_path": ds_art.note,
        "rows": UNKNOWN,
        "cols": UNKNOWN,
        "unique_tickers": UNKNOWN,
        "unique_dates": UNKNOWN,
        "ticker_date_dup_count": UNKNOWN,
    }
    if ds_df is not None:
        snap["rows"] = int(ds_df.shape[0])
        snap["cols"] = int(ds_df.shape[1])
        if "ticker" in ds_df.columns:
            snap["unique_tickers"] = int(ds_df["ticker"].astype(str).nunique())
        if "date" in ds_df.columns:
            snap["unique_dates"] = int(pd.to_datetime(ds_df["date"], errors="coerce").nunique())
        if "ticker" in ds_df.columns and "date" in ds_df.columns:
            tmp = ds_df[["ticker", "date"]].copy()
            tmp["ticker"] = tmp["ticker"].astype(str)
            tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
            snap["ticker_date_dup_count"] = int(tmp.duplicated(subset=["ticker", "date"]).sum())

    _write_csv(
        ppt_dir / "ppt_data_snapshot.csv",
        [snap, *phase_rows],
        fieldnames=["dataset_path", "rows", "cols", "unique_tickers", "unique_dates", "ticker_date_dup_count", "tag", "phase", "date_min", "date_max", "n_rows", "source"],
    )

    # feature missingness/stats
    feat_rows: List[Dict[str, Any]] = []
    if ds_df is None:
        feat_rows.append({"feature": UNKNOWN, "nan_pct": UNKNOWN, "min": "", "p01": "", "median": "", "p99": "", "max": "", "source": ds_art.note})
    else:
        for f in feats:
            if f not in ds_df.columns:
                feat_rows.append({"feature": f, "nan_pct": UNKNOWN, "min": "", "p01": "", "median": "", "p99": "", "max": "", "source": ds_art.note})
                continue
            s = pd.to_numeric(ds_df[f], errors="coerce").astype("float64")
            nan_pct = float(s.isna().mean() * 100.0)
            v = s.dropna()
            if len(v) == 0:
                feat_rows.append({"feature": f, "nan_pct": nan_pct, "min": "", "p01": "", "median": "", "p99": "", "max": "", "source": ds_art.note})
                continue
            feat_rows.append(
                {
                    "feature": f,
                    "nan_pct": nan_pct,
                    "min": float(v.min()),
                    "p01": float(v.quantile(0.01)),
                    "median": float(v.quantile(0.50)),
                    "p99": float(v.quantile(0.99)),
                    "max": float(v.max()),
                    "source": ds_art.note,
                }
            )
    _write_csv(
        ppt_dir / "ppt_feature_missingness.csv",
        feat_rows,
        fieldnames=["feature", "nan_pct", "min", "p01", "median", "p99", "max", "source"],
    )


def build_model_validation(project_root: Path, extract_dir: Path, ppt_dir: Path, run_tag: str, baseline_tag: str) -> None:
    """
    [4] Model Validation
    - model_metrics 아티팩트가 있으면 그대로 export.
    - 없으면 pred_short_oos/pred_long_oos로부터 phase별 ic_rank(스피어만), corr, rmse, hit_ratio 재계산.
    - alpha_selected는 없으면 UNKNOWN.
    """
    out_rows: List[Dict[str, Any]] = []

    def compute_from_preds(df: pd.DataFrame, *, tag: str, horizon_label: str) -> List[Dict[str, Any]]:
        rows = []
        if df is None:
            return rows
        # column candidates
        cols = set(df.columns)
        y_true_col = "y_true" if "y_true" in cols else ("true" if "true" in cols else None)
        y_pred_col = "y_pred" if "y_pred" in cols else ("pred" if "pred" in cols else None)
        phase_col = "phase" if "phase" in cols else None
        date_col = "date" if "date" in cols else None
        if y_true_col is None or y_pred_col is None or phase_col is None:
            return rows
        d = df.copy()
        if date_col and date_col in d.columns:
            d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
        for ph, g in d.groupby(phase_col, sort=False):
            yt = pd.to_numeric(g[y_true_col], errors="coerce")
            yp = pd.to_numeric(g[y_pred_col], errors="coerce")
            m = (~yt.isna()) & (~yp.isna())
            yt = yt[m]
            yp = yp[m]
            if len(yt) == 0:
                continue
            # metrics
            rmse = float(np.sqrt(np.mean((yp.to_numpy() - yt.to_numpy()) ** 2)))
            corr = float(np.corrcoef(yt.to_numpy(), yp.to_numpy())[0, 1]) if len(yt) > 1 else np.nan
            # spearman via rank corr
            ic_rank = float(pd.Series(yt).rank(pct=True).corr(pd.Series(yp).rank(pct=True)))
            hit = float(np.mean(np.sign(yt.to_numpy()) == np.sign(yp.to_numpy())))
            rows.append(
                {
                    "tag": tag,
                    "horizon": horizon_label,
                    "phase": str(ph),
                    "rmse": rmse,
                    "corr": corr,
                    "ic_rank": ic_rank if not (isinstance(ic_rank, float) and math.isnan(ic_rank)) else np.nan,
                    "hit_ratio": hit,
                    "alpha_selected": UNKNOWN,
                    "source": UNKNOWN,
                }
            )
        return rows

    for tag in [run_tag, baseline_tag]:
        # 1) model_metrics
        art = find_artifact(project_root, extract_dir, tag, "model_metrics")
        df, src = load_df(art)
        if df is not None:
            df2 = df.copy()
            df2["tag"] = tag
            df2["source"] = src
            # alpha_selected column not guaranteed
            if "alpha_selected" not in df2.columns:
                df2["alpha_selected"] = UNKNOWN
            out_rows.extend(df2.to_dict(orient="records"))
            continue

        # 2) recompute from preds
        pred_s, src_s = load_df(find_artifact(project_root, extract_dir, tag, "pred_short_oos"))
        pred_l, src_l = load_df(find_artifact(project_root, extract_dir, tag, "pred_long_oos"))
        if pred_s is not None:
            rows = compute_from_preds(pred_s, tag=tag, horizon_label="short")
            for r in rows:
                r["source"] = src_s
            out_rows.extend(rows)
        if pred_l is not None:
            rows = compute_from_preds(pred_l, tag=tag, horizon_label="long")
            for r in rows:
                r["source"] = src_l
            out_rows.extend(rows)

        if pred_s is None and pred_l is None:
            out_rows.append({"tag": tag, "horizon": UNKNOWN, "phase": UNKNOWN, "rmse": UNKNOWN, "corr": UNKNOWN, "ic_rank": UNKNOWN, "hit_ratio": UNKNOWN, "alpha_selected": UNKNOWN, "source": UNKNOWN})

    # normalize columns
    # if direct export had different schema, just export union columns
    df_out = pd.DataFrame(out_rows)
    if df_out.empty:
        _write_csv(ppt_dir / "ppt_model_validation.csv", [{"tag": UNKNOWN, "source": UNKNOWN}], fieldnames=["tag", "source"])
        return
    df_out.to_csv(ppt_dir / "ppt_model_validation.csv", index=False, encoding="utf-8")


def build_ranking_quality(project_root: Path, extract_dir: Path, ppt_dir: Path, run_tag: str, baseline_tag: str) -> None:
    """
    [5] Ranking Quality
    - rebalance_scores 또는 ranking_daily가 있으면 날짜별 Top-K 평균 수익, Bottom-K 평균 수익, 스프레드, IC(스피어만) 계산
    - 없으면 md에 UNKNOWN만 기록
    """
    md_lines: List[str] = []
    md_lines.append("# Ranking Quality (Top/Bottom spread & IC)")
    md_lines.append("")

    out_rows: List[Dict[str, Any]] = []
    for tag in [run_tag, baseline_tag]:
        src_used = UNKNOWN
        df = None
        # prefer rebalance_scores
        for name in ["rebalance_scores", "ranking_daily"]:
            art = find_artifact(project_root, extract_dir, tag, name)
            df, src_used = load_df(art)
            if df is not None:
                break
        if df is None:
            md_lines.append(f"- {tag}: {UNKNOWN}")
            out_rows.append({"tag": tag, "date": "", "phase": "", "top_k_mean_ret": UNKNOWN, "bottom_k_mean_ret": UNKNOWN, "spread": UNKNOWN, "ic_spearman": UNKNOWN, "source": UNKNOWN})
            continue

        # columns
        cols = set(df.columns)
        date_col = "date" if "date" in cols else None
        phase_col = "phase" if "phase" in cols else None
        score_col = "score_ens" if "score_ens" in cols else ("score" if "score" in cols else None)
        ret_col = "true_short" if "true_short" in cols else ("ret_fwd_20d" if "ret_fwd_20d" in cols else ("y_true" if "y_true" in cols else None))
        if date_col is None or phase_col is None or score_col is None or ret_col is None:
            md_lines.append(f"- {tag}: 필요한 컬럼이 없어 계산 불가 ({src_used})")
            out_rows.append({"tag": tag, "date": "", "phase": "", "top_k_mean_ret": UNKNOWN, "bottom_k_mean_ret": UNKNOWN, "spread": UNKNOWN, "ic_spearman": UNKNOWN, "source": src_used})
            continue

        # choose K from config_snapshot if available; else use 20 as UNKNOWN? can't assume -> UNKNOWN, then default to 20? Not allowed.
        # So: compute for K=20 only if we can read l7.top_k from reports/extract/config_snapshot.md (evidence-based).
        k = None
        cfg_md = extract_dir / "config_snapshot.md"
        if cfg_md.exists():
            for line in _read_text(cfg_md).splitlines():
                if line.startswith("- **top_k**:"):
                    try:
                        k = int(line.split("`")[1])
                    except Exception:
                        k = None
                    break
        if k is None:
            md_lines.append(f"- {tag}: top_k를 근거로 확정할 수 없어 계산 스킵 ({src_used})")
            out_rows.append({"tag": tag, "date": "", "phase": "", "top_k_mean_ret": UNKNOWN, "bottom_k_mean_ret": UNKNOWN, "spread": UNKNOWN, "ic_spearman": UNKNOWN, "source": src_used})
            continue

        d = df.copy()
        d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
        d[phase_col] = d[phase_col].astype(str)
        d["__score__"] = pd.to_numeric(d[score_col], errors="coerce")
        d["__ret__"] = pd.to_numeric(d[ret_col], errors="coerce")

        # by (phase,date)
        for (ph, dt), g in d.dropna(subset=["__score__", "__ret__"]).groupby([phase_col, date_col], sort=False):
            g2 = g.sort_values("__score__", ascending=False)
            top = g2.head(int(k))
            bottom = g2.tail(int(k))
            top_mean = float(top["__ret__"].mean()) if len(top) else np.nan
            bot_mean = float(bottom["__ret__"].mean()) if len(bottom) else np.nan
            spread = float(top_mean - bot_mean) if (not math.isnan(top_mean) and not math.isnan(bot_mean)) else np.nan
            ic = float(g2["__score__"].rank(pct=True).corr(g2["__ret__"].rank(pct=True))) if len(g2) > 1 else np.nan
            out_rows.append(
                {
                    "tag": tag,
                    "phase": str(ph),
                    "date": str(pd.to_datetime(dt).date()) if pd.notna(dt) else "",
                    "top_k": int(k),
                    "top_k_mean_ret": top_mean,
                    "bottom_k_mean_ret": bot_mean,
                    "spread": spread,
                    "ic_spearman": ic,
                    "source": src_used,
                }
            )

    if out_rows:
        pd.DataFrame(out_rows).to_csv(ppt_dir / "ppt_ranking_quality.csv", index=False, encoding="utf-8")
    else:
        _write_csv(ppt_dir / "ppt_ranking_quality.csv", [{"tag": UNKNOWN, "source": UNKNOWN}], fieldnames=["tag", "source"])
    _write_text(ppt_dir / "ppt_ranking_quality.md", "\n".join(md_lines).rstrip() + "\n")


def build_backtest_performance(project_root: Path, extract_dir: Path, ppt_dir: Path, run_tag: str, baseline_tag: str) -> None:
    """
    [6] Backtest Performance
    - bt_returns: 리밸런싱 단위 분포(p10/p50/p90/min/max) by phase
    - bt_vs_benchmark: excess_return 동일 통계 (가능하면)
    - 월별(month) compounded 테이블
    """
    dist_rows: List[Dict[str, Any]] = []
    monthly_rows: List[Dict[str, Any]] = []

    def add_dist(tag: str, name: str, df: pd.DataFrame, col: str, src: str) -> None:
        if df is None or col not in df.columns or "phase" not in df.columns:
            dist_rows.append({"tag": tag, "metric": name, "phase": UNKNOWN, "p10": UNKNOWN, "p50": UNKNOWN, "p90": UNKNOWN, "min": UNKNOWN, "max": UNKNOWN, "source": src})
            return
        for ph, g in df.groupby("phase", sort=False):
            x = pd.to_numeric(g[col], errors="coerce").dropna()
            if len(x) == 0:
                dist_rows.append({"tag": tag, "metric": name, "phase": str(ph), "p10": UNKNOWN, "p50": UNKNOWN, "p90": UNKNOWN, "min": UNKNOWN, "max": UNKNOWN, "source": src})
                continue
            dist_rows.append(
                {
                    "tag": tag,
                    "metric": name,
                    "phase": str(ph),
                    "p10": float(x.quantile(0.10)),
                    "p50": float(x.quantile(0.50)),
                    "p90": float(x.quantile(0.90)),
                    "min": float(x.min()),
                    "max": float(x.max()),
                    "source": src,
                }
            )

    for tag in [run_tag, baseline_tag]:
        bt_ret, bt_ret_src = load_df(find_artifact(project_root, extract_dir, tag, "bt_returns"))
        if bt_ret is None:
            add_dist(tag, "net_return", None, "net_return", bt_ret_src)
        else:
            add_dist(tag, "net_return", bt_ret, "net_return", bt_ret_src)

            # monthly compounded net_return (and bench/excess if available)
            d = bt_ret.copy()
            if "date" in d.columns:
                d["date"] = pd.to_datetime(d["date"], errors="coerce")
            else:
                d["date"] = pd.NaT
            d["month"] = d["date"].dt.to_period("M").astype(str)

            # merge bench_return if available
            bench, bench_src = load_df(find_artifact(project_root, extract_dir, tag, "bt_benchmark_returns"))
            if bench is not None and set(["phase", "date", "bench_return"]).issubset(set(bench.columns)):
                b = bench.copy()
                b["date"] = pd.to_datetime(b["date"], errors="coerce")
                d = d.merge(b[["phase", "date", "bench_return"]], on=["phase", "date"], how="left")
            # merge excess_return if available
            vs, vs_src = load_df(find_artifact(project_root, extract_dir, tag, "bt_vs_benchmark"))
            if vs is not None and set(["phase", "date", "excess_return"]).issubset(set(vs.columns)):
                v = vs.copy()
                v["date"] = pd.to_datetime(v["date"], errors="coerce")
                d = d.merge(v[["phase", "date", "excess_return"]], on=["phase", "date"], how="left")

            for (ph, m), g in d.groupby(["phase", "month"], sort=False):
                nr = pd.to_numeric(g.get("net_return"), errors="coerce")
                br = pd.to_numeric(g.get("bench_return"), errors="coerce") if "bench_return" in g.columns else None
                er = pd.to_numeric(g.get("excess_return"), errors="coerce") if "excess_return" in g.columns else None

                def comp(s: Optional[pd.Series]) -> Any:
                    if s is None:
                        return UNKNOWN
                    s2 = s.dropna()
                    if len(s2) == 0:
                        return UNKNOWN
                    eq = (1.0 + s2.astype(float)).prod()
                    return float(eq - 1.0)

                monthly_rows.append(
                    {
                        "tag": tag,
                        "phase": str(ph),
                        "month": str(m),
                        "net_return_compounded": comp(nr),
                        "bench_return_compounded": comp(br),
                        "excess_return_compounded": comp(er),
                        "source_bt_returns": bt_ret_src,
                        "source_bench": bench_src,
                        "source_vs": vs_src,
                    }
                )

        bt_vs, bt_vs_src = load_df(find_artifact(project_root, extract_dir, tag, "bt_vs_benchmark"))
        if bt_vs is None:
            add_dist(tag, "excess_return", None, "excess_return", bt_vs_src)
        else:
            add_dist(tag, "excess_return", bt_vs, "excess_return", bt_vs_src)

    pd.DataFrame(dist_rows).to_csv(ppt_dir / "ppt_bt_return_distribution.csv", index=False, encoding="utf-8")
    pd.DataFrame(monthly_rows).to_csv(ppt_dir / "ppt_monthly_returns.csv", index=False, encoding="utf-8")


def build_drawdown_events(project_root: Path, extract_dir: Path, ppt_dir: Path, run_tag: str, baseline_tag: str) -> None:
    """
    [7] Risk/Drawdown key events
    """
    rows: List[Dict[str, Any]] = []
    for tag in [run_tag, baseline_tag]:
        dd_df, src = load_df(find_artifact(project_root, extract_dir, tag, "bt_drawdown_events"))
        if dd_df is None:
            rows.append({"tag": tag, "phase": UNKNOWN, "peak_date": "", "trough_date": "", "drawdown": UNKNOWN, "length_days": UNKNOWN, "source": src})
            continue
        if "drawdown" not in dd_df.columns:
            rows.append({"tag": tag, "phase": UNKNOWN, "peak_date": "", "trough_date": "", "drawdown": UNKNOWN, "length_days": UNKNOWN, "source": src})
            continue
        d = dd_df.copy()
        d["drawdown"] = pd.to_numeric(d["drawdown"], errors="coerce")
        d = d.dropna(subset=["drawdown"]).sort_values("drawdown", ascending=True)
        top3 = d.head(3)
        for _, r in top3.iterrows():
            rows.append(
                {
                    "tag": tag,
                    "phase": str(r.get("phase", "")),
                    "peak_date": str(r.get("peak_date", "")),
                    "trough_date": str(r.get("trough_date", "")),
                    "drawdown": r.get("drawdown"),
                    "length_days": r.get("length_days", ""),
                    "source": src,
                }
            )
    pd.DataFrame(rows).to_csv(ppt_dir / "ppt_drawdown_key_events.csv", index=False, encoding="utf-8")


def build_stability(project_root: Path, extract_dir: Path, ppt_dir: Path, run_tag: str, baseline_tag: str) -> None:
    """
    [8] Stability: yearly_metrics export + rolling sharpe summary(있으면)
    """
    # yearly export
    all_yearly: List[pd.DataFrame] = []
    for tag in [run_tag, baseline_tag]:
        y, src = load_df(find_artifact(project_root, extract_dir, tag, "bt_yearly_metrics"))
        if y is None:
            all_yearly.append(pd.DataFrame([{"tag": tag, "source": src, "note": UNKNOWN}]))
        else:
            y2 = y.copy()
            y2["tag"] = tag
            y2["source"] = src
            all_yearly.append(y2)
    pd.concat(all_yearly, ignore_index=True).to_csv(ppt_dir / "ppt_yearly_metrics.csv", index=False, encoding="utf-8")

    # rolling sharpe summary (root only typical)
    rs_art = find_artifact(project_root, extract_dir, run_tag, "bt_rolling_sharpe")  # tag doesn't matter due to find order
    rs_df, rs_src = load_df(rs_art)
    rs_rows: List[Dict[str, Any]] = []
    if rs_df is None or "phase" not in rs_df.columns:
        rs_rows.append({"source": rs_src, "phase": UNKNOWN, "rolling_sharpe_min": UNKNOWN, "rolling_sharpe_max": UNKNOWN, "rolling_sharpe_mean": UNKNOWN, "min_date": UNKNOWN})
    else:
        d = rs_df.copy()
        if "date" in d.columns:
            d["date"] = pd.to_datetime(d["date"], errors="coerce")
        if "rolling_sharpe" not in d.columns:
            rs_rows.append({"source": rs_src, "phase": UNKNOWN, "rolling_sharpe_min": UNKNOWN, "rolling_sharpe_max": UNKNOWN, "rolling_sharpe_mean": UNKNOWN, "min_date": UNKNOWN})
        else:
            d["rolling_sharpe"] = pd.to_numeric(d["rolling_sharpe"], errors="coerce")
            for ph, g in d.dropna(subset=["rolling_sharpe"]).groupby("phase", sort=False):
                rs_rows.append(
                    {
                        "source": rs_src,
                        "phase": str(ph),
                        "rolling_sharpe_min": float(g["rolling_sharpe"].min()) if len(g) else UNKNOWN,
                        "rolling_sharpe_max": float(g["rolling_sharpe"].max()) if len(g) else UNKNOWN,
                        "rolling_sharpe_mean": float(g["rolling_sharpe"].mean()) if len(g) else UNKNOWN,
                        "min_date": str(g.loc[g["rolling_sharpe"].idxmin()]["date"].date()) if ("date" in g.columns and len(g)) else UNKNOWN,
                    }
                )
    pd.DataFrame(rs_rows).to_csv(ppt_dir / "ppt_rolling_sharpe_summary.csv", index=False, encoding="utf-8")


def build_sensitivity(project_root: Path, extract_dir: Path, ppt_dir: Path, run_tag: str, baseline_tag: str) -> None:
    """
    [9] Sensitivity: grid export + summary (holdout 상위 10% sharpe 조건 범위)
    """
    md_lines: List[str] = []
    md_lines.append("# Sensitivity Summary (holdout top 10% Sharpe)")
    md_lines.append("")
    grids: List[pd.DataFrame] = []

    for tag in [run_tag, baseline_tag]:
        g, src = load_df(find_artifact(project_root, extract_dir, tag, "bt_sensitivity"))
        if g is None:
            grids.append(pd.DataFrame([{"tag": tag, "source": src, "note": UNKNOWN}]))
            md_lines.append(f"- {tag}: {UNKNOWN} ({src})")
            continue
        g2 = g.copy()
        g2["tag"] = tag
        g2["source"] = src
        grids.append(g2)

        if "phase" not in g2.columns or "net_sharpe" not in g2.columns:
            md_lines.append(f"- {tag}: 필요한 컬럼 부족 ({src})")
            continue
        h = g2[g2["phase"].astype(str) == "holdout"].copy()
        if len(h) == 0:
            md_lines.append(f"- {tag}: holdout 행 없음 ({src})")
            continue
        h["net_sharpe"] = pd.to_numeric(h["net_sharpe"], errors="coerce")
        h2 = h.dropna(subset=["net_sharpe"])
        if len(h2) == 0:
            md_lines.append(f"- {tag}: net_sharpe 결측 ({src})")
            continue
        thr = float(h2["net_sharpe"].quantile(0.90))
        top = h2[h2["net_sharpe"] >= thr]
        md_lines.append(f"## {tag}")
        md_lines.append("")
        md_lines.append(f"- source: `{src}`")
        md_lines.append(f"- sharpe_p90_threshold: `{thr}`")
        for col in ["grid_top_k", "grid_cost_bps", "grid_buffer_k"]:
            if col in top.columns:
                v = pd.to_numeric(top[col], errors="coerce").dropna()
                if len(v) > 0:
                    md_lines.append(f"- {col}_range_in_top10pct: `{float(v.min())}` ~ `{float(v.max())}`")
                else:
                    md_lines.append(f"- {col}_range_in_top10pct: `{UNKNOWN}`")
            else:
                md_lines.append(f"- {col}_range_in_top10pct: `{UNKNOWN}`")
        md_lines.append("")

    pd.concat(grids, ignore_index=True).to_csv(ppt_dir / "ppt_sensitivity_grid.csv", index=False, encoding="utf-8")
    _write_text(ppt_dir / "ppt_sensitivity_summary.md", "\n".join(md_lines).rstrip() + "\n")


def build_benchmark_fairness(project_root: Path, extract_dir: Path, ppt_dir: Path, run_tag: str, baseline_tag: str) -> None:
    """
    [10] Benchmark/Fairness
    - 벤치마크 타입 근거(backtest_contract.md) 요약
    - 벤치 성과 계산 가능한 경우 계산
    - 비용 조건 차이(무결성) 표
    """
    md: List[str] = []
    md.append("# Benchmark / Fairness")
    md.append("")

    contract = extract_dir / "backtest_contract.md"
    if contract.exists():
        md.append("## 벤치마크 타입 근거")
        md.append("")
        md.append(f"- source: `{contract.relative_to(project_root)}`")
        md.append("- 근거 문서에 `build_universe_benchmark_returns` 및 (phase,date) 평균 수익률로 bench_return를 구성하는 코드 스니펫이 포함되어 있습니다.")
        md.append("")
    else:
        md.append(f"## 벤치마크 타입 근거: {UNKNOWN}")
        md.append("")

    # bench performance
    md.append("## 벤치마크 성과(계산 가능 범위)")
    md.append("")
    perf_rows: List[Dict[str, Any]] = []
    for tag in [run_tag, baseline_tag]:
        bench, src = load_df(find_artifact(project_root, extract_dir, tag, "bt_benchmark_returns"))
        if bench is None or "phase" not in bench.columns:
            perf_rows.append({"tag": tag, "phase": UNKNOWN, "bench_total_return": UNKNOWN, "bench_cagr": UNKNOWN, "bench_sharpe": UNKNOWN, "bench_mdd": UNKNOWN, "source": src})
            continue
        # holding_days: try from bt_returns
        bt, _bt_src = load_df(find_artifact(project_root, extract_dir, tag, "bt_returns"))
        hold_days = None
        if bt is not None and "holding_days" in bt.columns:
            hold_days = _parse_float(bt["holding_days"].iloc[0])
        for ph, g in bench.groupby("phase", sort=False):
            bm = compute_bench_metrics(g.sort_values("date"), hold_days)
            perf_rows.append(
                {
                    "tag": tag,
                    "phase": str(ph),
                    "bench_total_return": _safe_num_or_unknown(bm.get("bench_total_return")),
                    "bench_cagr": _safe_num_or_unknown(bm.get("bench_cagr")),
                    "bench_sharpe": _safe_num_or_unknown(bm.get("bench_sharpe")),
                    "bench_mdd": _safe_num_or_unknown(bm.get("bench_mdd")),
                    "source": src,
                }
            )
    pd.DataFrame(perf_rows).to_csv(ppt_dir / "ppt_benchmark_perf.csv", index=False, encoding="utf-8")
    md.append(f"- benchmark_perf_csv: `reports/extract/ppt/ppt_benchmark_perf.csv`")
    md.append("")

    # config mismatch table: cost_bps used vs config from extract outputs
    cfg_md = extract_dir / "config_snapshot.md"
    cfg_cost = None
    cfg_top_k = None
    if cfg_md.exists():
        for line in _read_text(cfg_md).splitlines():
            if line.startswith("- **cost_bps**:"):
                try:
                    cfg_cost = float(line.split("`")[1])
                except Exception:
                    cfg_cost = None
            if line.startswith("- **top_k**:"):
                try:
                    cfg_top_k = int(line.split("`")[1])
                except Exception:
                    cfg_top_k = None

    mismatch_rows: List[Dict[str, Any]] = []
    kpi_path = extract_dir / "kpi_snapshot.csv"
    kpi = pd.read_csv(kpi_path) if kpi_path.exists() else pd.DataFrame()
    for tag in [run_tag, baseline_tag]:
        used_cost = None
        if not kpi.empty and "tag" in kpi.columns and "cost_bps_used" in kpi.columns:
            rr = kpi[kpi["tag"].astype(str) == str(tag)]
            if len(rr) > 0:
                used_cost = _parse_float(rr.iloc[0]["cost_bps_used"])
        # top_k_used from bt_returns if present
        bt, bt_src = load_df(find_artifact(project_root, extract_dir, tag, "bt_returns"))
        top_k_used = None
        if bt is not None and "top_k" in bt.columns:
            try:
                top_k_used = int(pd.to_numeric(bt[bt["phase"].astype(str) == "holdout"]["top_k"].iloc[0], errors="coerce"))
            except Exception:
                top_k_used = None
        mismatch_rows.append(
            {
                "tag": tag,
                "config_cost_bps": cfg_cost if cfg_cost is not None else UNKNOWN,
                "cost_bps_used": used_cost if used_cost is not None else UNKNOWN,
                "cost_bps_mismatch": (cfg_cost != used_cost) if (cfg_cost is not None and used_cost is not None) else UNKNOWN,
                "config_top_k": cfg_top_k if cfg_top_k is not None else UNKNOWN,
                "top_k_used": top_k_used if top_k_used is not None else UNKNOWN,
                "top_k_mismatch": (cfg_top_k != top_k_used) if (cfg_top_k is not None and top_k_used is not None) else UNKNOWN,
                "source_kpi": str(kpi_path.relative_to(project_root)) if kpi_path.exists() else UNKNOWN,
                "source_bt_returns": bt_src,
            }
        )
    pd.DataFrame(mismatch_rows).to_csv(ppt_dir / "ppt_config_mismatch_table.csv", index=False, encoding="utf-8")
    md.append(f"- config_mismatch_csv: `reports/extract/ppt/ppt_config_mismatch_table.csv`")
    md.append("")

    _write_text(ppt_dir / "ppt_benchmark_fairness.md", "\n".join(md).rstrip() + "\n")


def build_slide_mapping_readme(ppt_dir: Path) -> None:
    """
    [11] 슬라이드-파일 매핑 README
    """
    md: List[str] = []
    md.append("# PPT Numeric Pack README")
    md.append("")
    md.append("## 슬라이드-파일 매핑(제안)")
    md.append("")
    md.append("| slide | purpose | file | key columns | unit notes |")
    md.append("|---:|---|---|---|---|")
    md.append("| 1 | Executive KPI | `ppt_executive_kpi.csv` | net_total_return, net_sharpe, net_mdd, cost_bps_used | unit_flag 참고 |")
    md.append("| 2 | KPI 단위 점검 | `ppt_kpi_units_check.md` | (text) | % vs decimal 혼용 주의 |")
    md.append("| 3 | Data Snapshot | `ppt_data_snapshot.csv` | rows, cols, unique_tickers/dates, phase date range | - |")
    md.append("| 4 | Feature Missingness | `ppt_feature_missingness.csv` | nan_pct, p01/p99 | - |")
    md.append("| 5 | Model Validation | `ppt_model_validation.csv` | ic_rank, rmse, hit_ratio | - |")
    md.append("| 6 | Ranking Quality | `ppt_ranking_quality.csv` | top/bottom/spread/ic | bottom은 비교 지표(거래 구현 여부 별도) |")
    md.append("| 7 | Backtest Return Dist | `ppt_bt_return_distribution.csv` | p10/p50/p90/min/max | - |")
    md.append("| 8 | Monthly Returns | `ppt_monthly_returns.csv` | net/bench/excess compounded | - |")
    md.append("| 9 | Drawdown Events | `ppt_drawdown_key_events.csv` | peak/trough/drawdown/length | drawdown scale 주의(unit_flag와 별도) |")
    md.append("| 10 | Stability | `ppt_yearly_metrics.csv` | yearly net_total_return, net_sharpe, net_mdd | - |")
    md.append("| 11 | Rolling Sharpe | `ppt_rolling_sharpe_summary.csv` | rolling_sharpe min/max/mean | - |")
    md.append("| 12 | Sensitivity | `ppt_sensitivity_grid.csv` | grid_* + net_sharpe | - |")
    md.append("| 13 | Sensitivity Summary | `ppt_sensitivity_summary.md` | (text) | - |")
    md.append("| 14 | Benchmark/Fairness | `ppt_benchmark_fairness.md` | (text) | - |")
    md.append("| 15 | Config mismatch | `ppt_config_mismatch_table.csv` | cost_bps/top_k mismatch | - |")
    md.append("")
    md.append("## '알 수 없습니다' 항목")
    md.append("")
    md.append("- 각 CSV에서 값이 `알 수 없습니다(근거 파일 없음)`인 경우, 해당 아티팩트/컬럼이 발견되지 않았음을 의미합니다.")
    md.append("")
    _write_text(ppt_dir / "README.md", "\n".join(md).rstrip() + "\n")


def bundle_all_outputs_to_single_file(ppt_dir: Path, *, out_name: str = "ppt_numeric_pack_all_in_one.md") -> None:
    """
    [개선안 12번] PPT 산출물을 1개 파일로 번들링.
    - 추정 금지: ppt_dir에 실제 존재하는 파일만 읽어서 합친다.
    - CSV는 그대로 code block으로 포함해 복붙 가능하게 한다.
    """
    order = [
        "ppt_numeric_inventory.md",
        "ppt_numeric_inventory.csv",
        "ppt_executive_kpi.csv",
        "ppt_kpi_units_check.md",
        "ppt_data_snapshot.csv",
        "ppt_feature_missingness.csv",
        "ppt_model_validation.csv",
        "ppt_ranking_quality.md",
        "ppt_ranking_quality.csv",
        "ppt_bt_return_distribution.csv",
        "ppt_monthly_returns.csv",
        "ppt_drawdown_key_events.csv",
        "ppt_yearly_metrics.csv",
        "ppt_rolling_sharpe_summary.csv",
        "ppt_sensitivity_summary.md",
        "ppt_sensitivity_grid.csv",
        "ppt_benchmark_fairness.md",
        "ppt_benchmark_perf.csv",
        "ppt_config_mismatch_table.csv",
        "README.md",
    ]

    lines: List[str] = []
    lines.append("# PPT Numeric Pack (All-in-One)")
    lines.append("")
    lines.append("> 본 파일은 `reports/extract/ppt/`의 산출물을 **그대로** 이어붙인 번들입니다.")
    lines.append("")

    for fname in order:
        p = ppt_dir / fname
        lines.append(f"## {fname}")
        lines.append("")
        if not p.exists():
            lines.append(UNKNOWN)
            lines.append("")
            continue
        text = _read_text(p).rstrip() + "\n"
        if p.suffix.lower() == ".csv":
            lines.append("```csv")
            lines.append(text.rstrip("\n"))
            lines.append("```")
            lines.append("")
        else:
            # md는 그대로 포함 (이미 markdown)
            lines.append(text.rstrip("\n"))
            lines.append("")

    _write_text(ppt_dir / out_name, "\n".join(lines).rstrip() + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="최종 PPT 숫자 산출물 Exporter (reports/extract/ppt)")
    ap.add_argument("--run-tag", default=DEFAULT_RUN_TAG)
    ap.add_argument("--baseline-tag", default=DEFAULT_BASELINE_TAG)
    args = ap.parse_args(argv)

    project_root = find_project_root_by_config(Path.cwd())
    extract_dir = project_root / "reports" / "extract"
    ppt_dir = extract_dir / "ppt"
    _ensure_dir(ppt_dir)

    # [1] inventory
    inventory_sources(project_root, extract_dir, ppt_dir)

    # [2] executive KPI
    build_executive_kpi(project_root, extract_dir, ppt_dir, args.run_tag, args.baseline_tag)

    # [3] data snapshot
    build_data_snapshot(project_root, extract_dir, ppt_dir, args.run_tag, args.baseline_tag)

    # [4] model validation
    build_model_validation(project_root, extract_dir, ppt_dir, args.run_tag, args.baseline_tag)

    # [5] ranking quality
    build_ranking_quality(project_root, extract_dir, ppt_dir, args.run_tag, args.baseline_tag)

    # [6] backtest performance
    build_backtest_performance(project_root, extract_dir, ppt_dir, args.run_tag, args.baseline_tag)

    # [7] drawdown events
    build_drawdown_events(project_root, extract_dir, ppt_dir, args.run_tag, args.baseline_tag)

    # [8] stability
    build_stability(project_root, extract_dir, ppt_dir, args.run_tag, args.baseline_tag)

    # [9] sensitivity
    build_sensitivity(project_root, extract_dir, ppt_dir, args.run_tag, args.baseline_tag)

    # [10] benchmark/fairness
    build_benchmark_fairness(project_root, extract_dir, ppt_dir, args.run_tag, args.baseline_tag)

    # [11] slide mapping README
    build_slide_mapping_readme(ppt_dir)
    # [번들] all-in-one 파일 생성
    bundle_all_outputs_to_single_file(ppt_dir)

    print(f"[export_ppt_numeric_pack] done. ppt_dir={ppt_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
