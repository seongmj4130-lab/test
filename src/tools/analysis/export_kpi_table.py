# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/analysis/export_kpi_table.py
"""
KPI 테이블 추출 스크립트
산출물 parquet/csv에서 KPI를 추출하여 1장짜리 테이블로 저장
"""
import argparse
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

warnings.filterwarnings("ignore")

def normalize_unit(metric: str, value: Any, unit_hint: str = "") -> Tuple[Any, str]:
    """
    단위 변환 표준화
    Returns: (normalized_value, unit)
    """
    if value is None or pd.isna(value):
        return value, unit_hint

    # Series인 경우 처리
    if isinstance(value, pd.Series):
        # Series 전체에 대해 변환 (나중에 집계)
        return value, unit_hint

    # pct_metrics: returns, turnover, mdd, cagr, vol, tracking_error, coverage -> *100 해서 %로 저장
    pct_keywords = ["return", "turnover", "mdd", "cagr", "vol", "tracking_error", "coverage", "nan_pct", "nonnull_pct", "cost_pct"]
    if any(kw in metric.lower() for kw in pct_keywords):
        if isinstance(value, (int, float)) and abs(value) < 10:  # ratio로 읽힌 경우
            return value * 100, "%"
        return value, "%"

    # ratio_metrics: sharpe, ic_rank, hit_ratio, information_ratio, beta, corr -> ratio 그대로
    ratio_keywords = ["sharpe", "ic_rank", "hit_ratio", "information_ratio", "beta", "corr"]
    if any(kw in metric.lower() for kw in ratio_keywords):
        return value, "ratio"

    # bps_metrics: cost_bps -> bps 그대로
    if "cost_bps" in metric.lower():
        return value, "bps"

    # 기본값
    if unit_hint:
        return value, unit_hint

    return value, "count"

def resolve_artifact(name: str, root: Path, tag: Optional[str] = None) -> Tuple[Optional[Path], Optional[str]]:
    """
    아티팩트 파일 탐색 (유연하게)
    Returns: (file_path, format) or (None, None)
    """
    candidates = []

    # tag가 있으면 tag 폴더 우선
    if tag:
        candidates.extend([
            root / "data" / "interim" / tag / f"{name}.parquet",
            root / "data" / "interim" / tag / f"{name}.csv",
            root / "data" / "processed" / tag / f"{name}.parquet",
            root / "data" / "processed" / tag / f"{name}.csv",
        ])

    # 기본 경로
    candidates.extend([
        root / "data" / "interim" / f"{name}.parquet",
        root / "data" / "interim" / f"{name}.csv",
        root / "data" / "processed" / f"{name}.parquet",
        root / "data" / "processed" / f"{name}.csv",
    ])

    for path in candidates:
        if path.exists():
            fmt = "parquet" if path.suffix == ".parquet" else "csv"
            return path, fmt

    return None, None

def get_row_count_meta(path: Path) -> int:
    """메타데이터만 읽어서 행 수 반환"""
    if path.suffix == ".parquet":
        try:
            pf = pq.ParquetFile(path)
            return pf.metadata.num_rows
        except Exception:
            return 0
    else:
        # CSV는 대략적으로만 (전체 읽지 않음)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return sum(1 for _ in f) - 1  # 헤더 제외
        except Exception:
            return 0

def load_partial(df_path: Path, columns: List[str], fmt: str) -> pd.DataFrame:
    """필요한 컬럼만 부분 로딩"""
    try:
        if fmt == "parquet":
            return pd.read_parquet(df_path, columns=columns)
        else:
            return pd.read_csv(df_path, usecols=columns, low_memory=False)
    except Exception as e:
        return pd.DataFrame()

def add_kpi_row(
    rows: List[Dict[str, Any]],
    section: str,
    metric: str,
    dev_value: Any,
    holdout_value: Any,
    unit: str,
    source_file: str,
    source_cols: str,
    note: str = "",
):
    """KPI 행 추가"""
    rows.append({
        "section": section,
        "metric": metric,
        "dev_value": dev_value if pd.notna(dev_value) else None,
        "holdout_value": holdout_value if pd.notna(holdout_value) else None,
        "unit": unit,
        "source_file": source_file,
        "source_cols": source_cols,
        "note": note,
    })

def extract_data_kpis(root: Path, tag: Optional[str]) -> List[Dict[str, Any]]:
    """DATA 섹션 KPI 추출"""
    rows = []

    # L0: universe_k200_membership_monthly
    path, fmt = resolve_artifact("universe_k200_membership_monthly", root, tag)
    if path:
        try:
            df = pd.read_parquet(path) if fmt == "parquet" else pd.read_csv(path, low_memory=False)
            months_count = df["ym"].nunique() if "ym" in df.columns else df["date"].dt.to_period("M").nunique()
            tickers_per_month = df.groupby("ym" if "ym" in df.columns else df["date"].dt.to_period("M"))["ticker"].nunique()
            duplicates_count = df.duplicated(["date", "ticker"]).sum()

            add_kpi_row(rows, "DATA", "months_count", months_count, None, "count", str(path), "ym or date", "")
            add_kpi_row(rows, "DATA", "tickers_per_month_min", tickers_per_month.min(), None, "count", str(path), "ticker grouped by ym", "")
            add_kpi_row(rows, "DATA", "tickers_per_month_mean", tickers_per_month.mean(), None, "count", str(path), "ticker grouped by ym", "")
            add_kpi_row(rows, "DATA", "tickers_per_month_max", tickers_per_month.max(), None, "count", str(path), "ticker grouped by ym", "")
            add_kpi_row(rows, "DATA", "duplicates_count", duplicates_count, None, "count", str(path), "date, ticker", "")
        except Exception as e:
            add_kpi_row(rows, "DATA", "universe_k200_membership_monthly", None, None, "count", str(path), "", f"ERROR: {e}")
    else:
        add_kpi_row(rows, "DATA", "universe_k200_membership_monthly", None, None, "count", "", "", "MISSING ARTIFACT")

    # L1: ohlcv_daily
    path, fmt = resolve_artifact("ohlcv_daily", root, tag)
    ohlcv_rows = None
    if path:
        try:
            ohlcv_rows = get_row_count_meta(path)
            # date/ticker만 부분 로딩
            df_sample = load_partial(path, ["ticker", "date"], fmt)
            n_tickers = df_sample["ticker"].nunique() if "ticker" in df_sample.columns else None
            n_dates = df_sample["date"].nunique() if "date" in df_sample.columns else None

            add_kpi_row(rows, "DATA", "ohlcv_rows_total", ohlcv_rows, None, "count", str(path), "metadata", "")
            add_kpi_row(rows, "DATA", "ohlcv_n_tickers", n_tickers, None, "count", str(path), "ticker", "")
            add_kpi_row(rows, "DATA", "ohlcv_n_dates", n_dates, None, "count", str(path), "date", "")

            # value 컬럼 확인
            df_value_check = load_partial(path, ["value"], fmt)
            if "value" in df_value_check.columns:
                value_nonnull_pct = df_value_check["value"].notna().mean() * 100
                add_kpi_row(rows, "DATA", "ohlcv_value_nonnull_pct", value_nonnull_pct, None, "%", str(path), "value", "")
            else:
                add_kpi_row(rows, "DATA", "ohlcv_value_nonnull_pct", None, None, "%", str(path), "", "MISSING COL: value")
        except Exception as e:
            add_kpi_row(rows, "DATA", "ohlcv_daily", None, None, "count", str(path), "", f"ERROR: {e}")
    else:
        add_kpi_row(rows, "DATA", "ohlcv_daily", None, None, "count", "", "", "MISSING ARTIFACT")

    # L2: fundamentals_annual
    path, fmt = resolve_artifact("fundamentals_annual", root, tag)
    if path:
        try:
            rows_total = get_row_count_meta(path)
            df_sample = load_partial(path, ["ticker", "date"], fmt)
            n_tickers = df_sample["ticker"].nunique() if "ticker" in df_sample.columns else None
            years_count = df_sample["date"].dt.year.nunique() if "date" in df_sample.columns else None
            coverage_ratio = rows_total / (n_tickers * years_count) if (n_tickers and years_count) else None

            add_kpi_row(rows, "DATA", "fundamentals_rows_total", rows_total, None, "count", str(path), "metadata", "")
            add_kpi_row(rows, "DATA", "fundamentals_n_tickers", n_tickers, None, "count", str(path), "ticker", "")
            add_kpi_row(rows, "DATA", "fundamentals_years_count", years_count, None, "count", str(path), "date.year", "")
            if coverage_ratio is not None:
                add_kpi_row(rows, "DATA", "fundamentals_coverage_ratio", coverage_ratio * 100, None, "%", str(path), "rows / (tickers * years)", "")

            # [Stage 1] lag_source 비율 계산
            df_lag = load_partial(path, ["lag_source"], fmt)
            if "lag_source" in df_lag.columns:
                lag_source_counts = df_lag["lag_source"].value_counts()
                total = len(df_lag)
                if total > 0:
                    rcept_pct = (lag_source_counts.get("rcept_date", 0) / total) * 100
                    fallback_pct = (lag_source_counts.get("year_end_fallback", 0) / total) * 100
                    add_kpi_row(rows, "DATA", "fundamentals_lag_source_rcept_pct", rcept_pct, None, "%", str(path), "lag_source", "")
                    add_kpi_row(rows, "DATA", "fundamentals_lag_source_fallback_pct", fallback_pct, None, "%", str(path), "lag_source", "")
        except Exception as e:
            add_kpi_row(rows, "DATA", "fundamentals_annual", None, None, "count", str(path), "", f"ERROR: {e}")
    else:
        add_kpi_row(rows, "DATA", "fundamentals_annual", None, None, "count", "", "", "MISSING ARTIFACT")

    # L3: panel_merged_daily
    path, fmt = resolve_artifact("panel_merged_daily", root, tag)
    panel_rows = None
    if path:
        try:
            panel_rows = get_row_count_meta(path)
            # ohlcv_rows는 위에서 이미 계산했을 수 있음
            ohlcv_path, _ = resolve_artifact("ohlcv_daily", root, tag)
            ohlcv_rows_for_check = get_row_count_meta(ohlcv_path) if ohlcv_path else None

            df_sample = load_partial(path, ["net_income", "equity", "debt_ratio", "roe"], fmt)

            note = ""
            if ohlcv_rows_for_check and panel_rows and ohlcv_rows_for_check > 0:
                diff_pct = abs(panel_rows - ohlcv_rows_for_check) / ohlcv_rows_for_check * 100
                if diff_pct > 1.0:
                    note = f"WARNING: row mismatch (possible mixed artifacts) - diff={diff_pct:.2f}%"

            add_kpi_row(rows, "DATA", "panel_rows_total", panel_rows, None, "count", str(path), "metadata", note)

            if ohlcv_rows_for_check:
                keep_ratio = (panel_rows / ohlcv_rows_for_check * 100) if ohlcv_rows_for_check > 0 else None
                add_kpi_row(rows, "DATA", "panel_keep_ratio_vs_ohlcv", keep_ratio, None, "%", str(path), "panel_rows / ohlcv_rows", "")

            for col in ["net_income", "equity", "debt_ratio", "roe"]:
                if col in df_sample.columns:
                    nan_pct = df_sample[col].isna().mean() * 100
                    add_kpi_row(rows, "DATA", f"panel_nan_pct_{col}", nan_pct, None, "%", str(path), col, "")
                else:
                    add_kpi_row(rows, "DATA", f"panel_nan_pct_{col}", None, None, "%", str(path), "", f"MISSING COL: {col}")
        except Exception as e:
            add_kpi_row(rows, "DATA", "panel_merged_daily", None, None, "count", str(path), "", f"ERROR: {e}")
    else:
        add_kpi_row(rows, "DATA", "panel_merged_daily", None, None, "count", "", "", "MISSING ARTIFACT")

    # L6: rebalance_scores_summary
    path, fmt = resolve_artifact("rebalance_scores_summary", root, tag)
    if path:
        try:
            n_rebalances = get_row_count_meta(path)
            df = pd.read_parquet(path) if fmt == "parquet" else pd.read_csv(path, low_memory=False)

            add_kpi_row(rows, "DATA", "rebalance_n_rebalances", n_rebalances, None, "count", str(path), "metadata", "")

            if "n_tickers" in df.columns:
                add_kpi_row(rows, "DATA", "rebalance_n_tickers_min", df["n_tickers"].min(), None, "count", str(path), "n_tickers", "")
                add_kpi_row(rows, "DATA", "rebalance_n_tickers_mean", df["n_tickers"].mean(), None, "count", str(path), "n_tickers", "")
                add_kpi_row(rows, "DATA", "rebalance_n_tickers_max", df["n_tickers"].max(), None, "count", str(path), "n_tickers", "")

            if "coverage_vs_universe_pct" in df.columns:
                add_kpi_row(rows, "DATA", "rebalance_coverage_vs_universe_pct_mean", df["coverage_vs_universe_pct"].mean(), None, "%", str(path), "coverage_vs_universe_pct", "")

            score_missing_cols = [c for c in df.columns if "missing" in c.lower() and "score" in c.lower()]
            if score_missing_cols:
                # total이면 sum(정수), 평균이면 missing_mean_per_rebalance로 이름 변경
                if all("total" in c.lower() or "sum" in c.lower() for c in score_missing_cols):
                    score_missing_total = df[score_missing_cols].sum(axis=1).sum()
                    add_kpi_row(rows, "DATA", "rebalance_score_missing_total", int(score_missing_total), None, "count", str(path), ",".join(score_missing_cols), "")
                else:
                    # 평균인 경우
                    score_missing_mean = df[score_missing_cols].mean(axis=1).mean()
                    add_kpi_row(rows, "DATA", "rebalance_score_missing_mean_per_rebalance", score_missing_mean, None, "count", str(path), ",".join(score_missing_cols), "")
        except Exception as e:
            add_kpi_row(rows, "DATA", "rebalance_scores_summary", None, None, "count", str(path), "", f"ERROR: {e}")
    else:
        add_kpi_row(rows, "DATA", "rebalance_scores_summary", None, None, "count", "", "", "MISSING ARTIFACT")

    return rows

def extract_model_kpis(root: Path, tag: Optional[str]) -> List[Dict[str, Any]]:
    """MODEL 섹션 KPI 추출"""
    rows = []

    path, fmt = resolve_artifact("model_metrics", root, tag)
    if not path:
        add_kpi_row(rows, "MODEL", "model_metrics", None, None, "", "", "", "MISSING ARTIFACT")
        return rows

    try:
        df = pd.read_parquet(path) if fmt == "parquet" else pd.read_csv(path, low_memory=False)

        if df.empty:
            add_kpi_row(rows, "MODEL", "model_metrics", None, None, "", str(path), "", "EMPTY")
            return rows

        required_cols = ["horizon", "phase", "rmse", "mae", "ic_rank", "hit_ratio"]
        # [Stage 2] r2_oos는 선택적 (없어도 계속 진행)
        has_r2_oos = "r2_oos" in df.columns
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            add_kpi_row(rows, "MODEL", "model_metrics", None, None, "", str(path), "", f"MISSING COLS: {','.join(missing_cols)}")
            return rows

        # horizon별로 처리하되, dev/holdout을 한 행에 합침
        for horizon in [20, 120]:
            df_horizon = df[df["horizon"] == horizon]
            df_dev = df_horizon[df_horizon["phase"] == "dev"]
            df_holdout = df_horizon[df_horizon["phase"] == "holdout"]

            metrics = {
                "rmse": ["mean", "median", "p10", "p90"],
                "mae": ["mean", "median"],
                "ic_rank": ["mean", "median", "p10", "p90"],
                "hit_ratio": ["mean", "median"],
            }
            # [Stage 2] r2_oos 추가
            if has_r2_oos:
                metrics["r2_oos"] = ["mean", "median"]

            for metric_col, stats in metrics.items():
                if metric_col not in df_horizon.columns:
                    continue

                for stat in stats:
                    dev_val = None
                    holdout_val = None

                    if not df_dev.empty:
                        if stat == "mean":
                            dev_val = df_dev[metric_col].mean()
                        elif stat == "median":
                            dev_val = df_dev[metric_col].median()
                        elif stat == "p10":
                            dev_val = df_dev[metric_col].quantile(0.1)
                        elif stat == "p90":
                            dev_val = df_dev[metric_col].quantile(0.9)

                    if not df_holdout.empty:
                        if stat == "mean":
                            holdout_val = df_holdout[metric_col].mean()
                        elif stat == "median":
                            holdout_val = df_holdout[metric_col].median()
                        elif stat == "p10":
                            holdout_val = df_holdout[metric_col].quantile(0.1)
                        elif stat == "p90":
                            holdout_val = df_holdout[metric_col].quantile(0.9)

                    if dev_val is None and holdout_val is None:
                        continue

                    metric_name = f"{metric_col}_{stat}__{horizon}d"
                    unit = "ratio"
                    add_kpi_row(rows, "MODEL", metric_name, dev_val, holdout_val, unit, str(path), f"{metric_col} by horizon={horizon}", "")

            # n_folds
            dev_n_folds = df_dev["fold_id"].nunique() if not df_dev.empty and "fold_id" in df_dev.columns else None
            holdout_n_folds = df_holdout["fold_id"].nunique() if not df_holdout.empty and "fold_id" in df_holdout.columns else None
            if dev_n_folds is not None or holdout_n_folds is not None:
                metric_name = f"n_folds__{horizon}d"
                add_kpi_row(rows, "MODEL", metric_name, dev_n_folds, holdout_n_folds, "count", str(path), "fold_id unique count", "")

            # n_features
            if "n_features" in df_horizon.columns:
                dev_feat_mean = df_dev["n_features"].mean() if not df_dev.empty else None
                dev_feat_min = df_dev["n_features"].min() if not df_dev.empty else None
                dev_feat_max = df_dev["n_features"].max() if not df_dev.empty else None
                holdout_feat_mean = df_holdout["n_features"].mean() if not df_holdout.empty else None
                holdout_feat_min = df_holdout["n_features"].min() if not df_holdout.empty else None
                holdout_feat_max = df_holdout["n_features"].max() if not df_holdout.empty else None

                for stat in ["mean", "min", "max"]:
                    metric_name = f"n_features_{stat}__{horizon}d"
                    dev_val = dev_feat_mean if stat == "mean" else (dev_feat_min if stat == "min" else dev_feat_max)
                    holdout_val = holdout_feat_mean if stat == "mean" else (holdout_feat_min if stat == "min" else holdout_feat_max)
                    add_kpi_row(rows, "MODEL", metric_name, dev_val, holdout_val, "count", str(path), "n_features", "")

            # [Stage 2] r2_oos_mean 명시적 추가 (short/long)
            if has_r2_oos:
                horizon_name = "short" if horizon == 20 else "long"
                dev_r2_oos_mean = df_dev["r2_oos"].mean() if not df_dev.empty else None
                holdout_r2_oos_mean = df_holdout["r2_oos"].mean() if not df_holdout.empty else None
                metric_name = f"model_r2_oos_mean_{horizon_name}"
                add_kpi_row(rows, "MODEL", metric_name, dev_r2_oos_mean, holdout_r2_oos_mean, "ratio", str(path), f"r2_oos mean for horizon={horizon}", "")

    except Exception as e:
        add_kpi_row(rows, "MODEL", "model_metrics", None, None, "", str(path), "", f"ERROR: {e}")

    # [Stage 2] feature_importance_summary에서 feature_sign_stability_top5 추출
    path_summary, fmt_summary = resolve_artifact("feature_importance_summary", root, tag)
    if path_summary:
        try:
            df_summary = pd.read_parquet(path_summary) if fmt_summary == "parquet" else pd.read_csv(path_summary, low_memory=False)

            if not df_summary.empty and "coef_sign_stability" in df_summary.columns:
                # horizon별로 처리
                for horizon in [20, 120]:
                    horizon_name = "short" if horizon == 20 else "long"
                    df_h = df_summary[df_summary["horizon"] == horizon]

                    if not df_h.empty:
                        # dev와 holdout 각각 처리
                        for phase in ["dev", "holdout"]:
                            df_phase = df_h[df_h["phase"] == phase]
                            if not df_phase.empty:
                                # abs_coef_mean 기준 상위 5개 선택
                                top5 = df_phase.nlargest(5, "abs_coef_mean")
                                # feature 이름과 sign_stability를 조합
                                top5_list = [
                                    f"{row['feature']}({row['coef_sign_stability']:.2f})"
                                    for _, row in top5.iterrows()
                                ]
                                top5_str = ", ".join(top5_list)

                                metric_name = f"feature_sign_stability_top5_{horizon_name}_{phase}"
                                value = top5_str if top5_str else None
                                if phase == "dev":
                                    add_kpi_row(rows, "MODEL", metric_name, value, None, "str", str(path_summary), f"top5 features by abs_coef_mean (horizon={horizon}, phase={phase})", "")
                                else:
                                    # holdout은 기존 행 업데이트 또는 새 행 추가
                                    existing_idx = None
                                    for i, r in enumerate(rows):
                                        if r.get("metric") == metric_name:
                                            existing_idx = i
                                            break
                                    if existing_idx is not None:
                                        rows[existing_idx]["holdout_value"] = value
                                    else:
                                        add_kpi_row(rows, "MODEL", metric_name, None, value, "str", str(path_summary), f"top5 features by abs_coef_mean (horizon={horizon}, phase={phase})", "")
        except Exception as e:
            # feature_importance_summary가 없어도 계속 진행 (선택적)
            pass

    return rows

def extract_backtest_kpis(root: Path, tag: Optional[str], config_cost_bps: Optional[float] = None) -> List[Dict[str, Any]]:
    """BACKTEST 섹션 KPI 추출"""
    rows = []

    # bt_metrics
    path, fmt = resolve_artifact("bt_metrics", root, tag)
    if not path:
        add_kpi_row(rows, "BACKTEST", "bt_metrics", None, None, "", "", "", "MISSING ARTIFACT")
        return rows

    try:
        df = pd.read_parquet(path) if fmt == "parquet" else pd.read_csv(path, low_memory=False)

        if "phase" not in df.columns:
            add_kpi_row(rows, "BACKTEST", "bt_metrics", None, None, "", str(path), "", "MISSING COL: phase")
            return rows

        metrics_to_extract = [
            "net_total_return", "net_cagr", "net_vol_ann", "net_sharpe", "net_mdd",
            "net_hit_ratio", "avg_turnover_oneway", "n_rebalances", "avg_n_tickers",
            "gross_total_return", "date_start", "date_end",
        ]

        # dev와 holdout을 한 번에 처리
        df_dev = df[df["phase"] == "dev"]
        df_holdout = df[df["phase"] == "holdout"]

        row_dev = df_dev.iloc[0] if not df_dev.empty else None
        row_holdout = df_holdout.iloc[0] if not df_holdout.empty else None

        for metric in metrics_to_extract:
            dev_val = None
            holdout_val = None

            if row_dev is not None and metric in row_dev.index:
                val = row_dev[metric]
                if pd.notna(val):
                    dev_val = val

            if row_holdout is not None and metric in row_holdout.index:
                val = row_holdout[metric]
                if pd.notna(val):
                    holdout_val = val

            if dev_val is None and holdout_val is None:
                continue

            # 단위 변환
            if metric == "date_start" or metric == "date_end":
                unit = "date"
                if dev_val is not None:
                    dev_val = str(dev_val)
                if holdout_val is not None:
                    holdout_val = str(holdout_val)
            else:
                dev_val, unit = normalize_unit(metric, dev_val)
                holdout_val, _ = normalize_unit(metric, holdout_val)

            add_kpi_row(rows, "BACKTEST", metric, dev_val, holdout_val, unit, str(path), metric, "")

        # cost_bps_used (bt_metrics에서)
        dev_cost_bps_used = row_dev.get("cost_bps") if row_dev is not None and "cost_bps" in row_dev.index else None
        holdout_cost_bps_used = row_holdout.get("cost_bps") if row_holdout is not None and "cost_bps" in row_holdout.index else None
        if dev_cost_bps_used is not None or holdout_cost_bps_used is not None:
            add_kpi_row(rows, "BACKTEST", "cost_bps_used", dev_cost_bps_used, holdout_cost_bps_used, "bps", str(path), "cost_bps", "")

        # mismatch_flag (config_cost_bps와 비교)
        if config_cost_bps is not None:
            dev_mismatch = (abs(dev_cost_bps_used - config_cost_bps) > 0.01) if (dev_cost_bps_used is not None) else None
            holdout_mismatch = (abs(holdout_cost_bps_used - config_cost_bps) > 0.01) if (holdout_cost_bps_used is not None) else None
            if dev_mismatch is not None or holdout_mismatch is not None:
                add_kpi_row(rows, "BACKTEST", "cost_bps_mismatch_flag", dev_mismatch, holdout_mismatch, "bool", str(path), f"cost_bps_used vs config({config_cost_bps})", "")

        # avg_cost_pct (bt_returns에서 계산)
        # gross_minus_net_total_return_pct
        if row_dev is not None and "gross_total_return" in row_dev.index and "net_total_return" in row_dev.index:
            gross_dev = row_dev["gross_total_return"]
            net_dev = row_dev["net_total_return"]
            if pd.notna(gross_dev) and pd.notna(net_dev):
                diff_dev = (gross_dev - net_dev) * 100  # % 변환
                add_kpi_row(rows, "BACKTEST", "gross_minus_net_total_return_pct", diff_dev, None, "%", str(path), "gross_total_return - net_total_return", "")

        if row_holdout is not None and "gross_total_return" in row_holdout.index and "net_total_return" in row_holdout.index:
            gross_holdout = row_holdout["gross_total_return"]
            net_holdout = row_holdout["net_total_return"]
            if pd.notna(gross_holdout) and pd.notna(net_holdout):
                diff_holdout = (gross_holdout - net_holdout) * 100  # % 변환
                # 기존 행 업데이트 또는 새 행 추가
                existing_idx = None
                for i, r in enumerate(rows):
                    if r.get("metric") == "gross_minus_net_total_return_pct":
                        existing_idx = i
                        break
                if existing_idx is not None:
                    rows[existing_idx]["holdout_value"] = diff_holdout
                else:
                    add_kpi_row(rows, "BACKTEST", "gross_minus_net_total_return_pct", None, diff_holdout, "%", str(path), "gross_total_return - net_total_return", "")

    except Exception as e:
        add_kpi_row(rows, "BACKTEST", "bt_metrics", None, None, "", str(path), "", f"ERROR: {e}")

    # bt_returns (보조)
    path, fmt = resolve_artifact("bt_returns", root, tag)
    if path:
        try:
            df = pd.read_parquet(path) if fmt == "parquet" else pd.read_csv(path, low_memory=False)
            if "phase" in df.columns and "net_return" in df.columns:
                df_dev = df[df["phase"] == "dev"]
                df_holdout = df[df["phase"] == "holdout"]

                # ratio로 읽히면 *100 해서 %로 저장
                dev_return_mean_raw = df_dev["net_return"].mean() if not df_dev.empty else None
                holdout_return_mean_raw = df_holdout["net_return"].mean() if not df_holdout.empty else None
                dev_return_std_raw = df_dev["net_return"].std() if not df_dev.empty else None
                holdout_return_std_raw = df_holdout["net_return"].std() if not df_holdout.empty else None

                dev_return_mean, _ = normalize_unit("net_return_mean", dev_return_mean_raw)
                holdout_return_mean, _ = normalize_unit("net_return_mean", holdout_return_mean_raw)
                dev_return_std, _ = normalize_unit("net_return_std", dev_return_std_raw)
                holdout_return_std, _ = normalize_unit("net_return_std", holdout_return_std_raw)

                add_kpi_row(rows, "BACKTEST", "bt_returns_net_return_mean", dev_return_mean, holdout_return_mean, "%", str(path), "net_return", "")
                add_kpi_row(rows, "BACKTEST", "bt_returns_net_return_std", dev_return_std, holdout_return_std, "%", str(path), "net_return", "")

                if "turnover_oneway" in df.columns:
                    dev_turnover_raw = df_dev["turnover_oneway"].mean() if not df_dev.empty else None
                    holdout_turnover_raw = df_holdout["turnover_oneway"].mean() if not df_holdout.empty else None
                    dev_turnover, _ = normalize_unit("turnover_oneway_mean", dev_turnover_raw)
                    holdout_turnover, _ = normalize_unit("turnover_oneway_mean", holdout_turnover_raw)
                    add_kpi_row(rows, "BACKTEST", "bt_returns_turnover_oneway_mean", dev_turnover, holdout_turnover, "%", str(path), "turnover_oneway", "")

                # avg_cost_pct 계산 (cost 컬럼이 있으면, 없으면 gross - net로 계산)
                if "cost" in df.columns:
                    dev_avg_cost = df_dev["cost"].mean() if not df_dev.empty else None
                    holdout_avg_cost = df_holdout["cost"].mean() if not df_holdout.empty else None
                    dev_avg_cost_pct, _ = normalize_unit("avg_cost_pct", dev_avg_cost)
                    holdout_avg_cost_pct, _ = normalize_unit("avg_cost_pct", holdout_avg_cost)
                    add_kpi_row(rows, "BACKTEST", "avg_cost_pct", dev_avg_cost_pct, holdout_avg_cost_pct, "%", str(path), "cost", "")
                elif "gross_return" in df.columns and "net_return" in df.columns:
                    # cost 컬럼이 없으면 gross - net로 계산
                    dev_cost = (df_dev["gross_return"] - df_dev["net_return"]).mean() if not df_dev.empty else None
                    holdout_cost = (df_holdout["gross_return"] - df_holdout["net_return"]).mean() if not df_holdout.empty else None
                    dev_avg_cost_pct, _ = normalize_unit("avg_cost_pct", dev_cost)
                    holdout_avg_cost_pct, _ = normalize_unit("avg_cost_pct", holdout_cost)
                    add_kpi_row(rows, "BACKTEST", "avg_cost_pct", dev_avg_cost_pct, holdout_avg_cost_pct, "%", str(path), "gross_return - net_return", "")
        except Exception as e:
            pass  # bt_returns는 보조이므로 실패해도 계속

    # [Stage13] selection_diagnostics KPI 추가
    path_diag, fmt_diag = resolve_artifact("selection_diagnostics", root, tag)
    if path_diag:
        try:
            df_diag = pd.read_parquet(path_diag) if fmt_diag == "parquet" else pd.read_csv(path_diag, low_memory=False)
            if "phase" in df_diag.columns:
                df_dev_diag = df_diag[df_diag["phase"] == "dev"]
                df_holdout_diag = df_diag[df_diag["phase"] == "holdout"]

                # k_eff_mean (selected_count의 평균)
                if "selected_count" in df_diag.columns:
                    dev_k_eff_mean = df_dev_diag["selected_count"].mean() if not df_dev_diag.empty else None
                    holdout_k_eff_mean = df_holdout_diag["selected_count"].mean() if not df_holdout_diag.empty else None
                    add_kpi_row(rows, "BACKTEST", "k_eff_mean_dev", dev_k_eff_mean, None, "count", str(path_diag), "selected_count", "")
                    add_kpi_row(rows, "BACKTEST", "k_eff_mean_holdout", None, holdout_k_eff_mean, "count", str(path_diag), "selected_count", "")

                # k_eff_fill_rate_mean (=k_eff/top_k)
                if "selected_count" in df_diag.columns and "top_k" in df_diag.columns:
                    dev_fill_rate = (df_dev_diag["selected_count"] / df_dev_diag["top_k"]).mean() if not df_dev_diag.empty else None
                    holdout_fill_rate = (df_holdout_diag["selected_count"] / df_holdout_diag["top_k"]).mean() if not df_holdout_diag.empty else None
                    dev_fill_rate_pct = dev_fill_rate * 100 if dev_fill_rate is not None else None
                    holdout_fill_rate_pct = holdout_fill_rate * 100 if holdout_fill_rate is not None else None
                    add_kpi_row(rows, "BACKTEST", "k_eff_fill_rate_mean_dev", dev_fill_rate_pct, None, "%", str(path_diag), "selected_count/top_k", "")
                    add_kpi_row(rows, "BACKTEST", "k_eff_fill_rate_mean_holdout", None, holdout_fill_rate_pct, "%", str(path_diag), "selected_count/top_k", "")

                # eligible_count_mean
                if "eligible_count" in df_diag.columns:
                    dev_eligible_mean = df_dev_diag["eligible_count"].mean() if not df_dev_diag.empty else None
                    holdout_eligible_mean = df_holdout_diag["eligible_count"].mean() if not df_holdout_diag.empty else None
                    add_kpi_row(rows, "BACKTEST", "eligible_count_mean_dev", dev_eligible_mean, None, "count", str(path_diag), "eligible_count", "")
                    add_kpi_row(rows, "BACKTEST", "eligible_count_mean_holdout", None, holdout_eligible_mean, "count", str(path_diag), "eligible_count", "")

                # dropped_by_missing_mean
                if "dropped_missing" in df_diag.columns:
                    dev_dropped_missing_mean = df_dev_diag["dropped_missing"].mean() if not df_dev_diag.empty else None
                    holdout_dropped_missing_mean = df_holdout_diag["dropped_missing"].mean() if not df_holdout_diag.empty else None
                    add_kpi_row(rows, "BACKTEST", "dropped_by_missing_mean_dev", dev_dropped_missing_mean, None, "count", str(path_diag), "dropped_missing", "")
                    add_kpi_row(rows, "BACKTEST", "dropped_by_missing_mean_holdout", None, holdout_dropped_missing_mean, "count", str(path_diag), "dropped_missing", "")

                # dropped_by_filter_mean
                if "dropped_filter" in df_diag.columns:
                    dev_dropped_filter_mean = df_dev_diag["dropped_filter"].mean() if not df_dev_diag.empty else None
                    holdout_dropped_filter_mean = df_holdout_diag["dropped_filter"].mean() if not df_holdout_diag.empty else None
                    add_kpi_row(rows, "BACKTEST", "dropped_by_filter_mean_dev", dev_dropped_filter_mean, None, "count", str(path_diag), "dropped_filter", "")
                    add_kpi_row(rows, "BACKTEST", "dropped_by_filter_mean_holdout", None, holdout_dropped_filter_mean, "count", str(path_diag), "dropped_filter", "")

                # dropped_by_sectorcap_mean
                if "dropped_sectorcap" in df_diag.columns:
                    dev_dropped_sectorcap_mean = df_dev_diag["dropped_sectorcap"].mean() if not df_dev_diag.empty else None
                    holdout_dropped_sectorcap_mean = df_holdout_diag["dropped_sectorcap"].mean() if not df_holdout_diag.empty else None
                    add_kpi_row(rows, "BACKTEST", "dropped_by_sectorcap_mean_dev", dev_dropped_sectorcap_mean, None, "count", str(path_diag), "dropped_sectorcap", "")
                    add_kpi_row(rows, "BACKTEST", "dropped_by_sectorcap_mean_holdout", None, holdout_dropped_sectorcap_mean, "count", str(path_diag), "dropped_sectorcap", "")
        except Exception as e:
            pass  # selection_diagnostics는 선택적이므로 실패해도 계속

    # [Stage13] bt_returns_diagnostics KPI 추가
    path_bt_diag, fmt_bt_diag = resolve_artifact("bt_returns_diagnostics", root, tag)
    if path_bt_diag:
        try:
            df_bt_diag = pd.read_parquet(path_bt_diag) if fmt_bt_diag == "parquet" else pd.read_csv(path_bt_diag, low_memory=False)
            if "phase" in df_bt_diag.columns:
                df_dev_bt_diag = df_bt_diag[df_bt_diag["phase"] == "dev"]
                df_holdout_bt_diag = df_bt_diag[df_bt_diag["phase"] == "holdout"]

                # regime 컬럼 non-null 비율
                if "regime" in df_bt_diag.columns:
                    dev_regime_nonnull_pct = (df_dev_bt_diag["regime"].notna().sum() / len(df_dev_bt_diag) * 100) if not df_dev_bt_diag.empty else None
                    holdout_regime_nonnull_pct = (df_holdout_bt_diag["regime"].notna().sum() / len(df_holdout_bt_diag) * 100) if not df_holdout_bt_diag.empty else None
                    add_kpi_row(rows, "BACKTEST", "bt_returns_diag_regime_nonnull_pct_dev", dev_regime_nonnull_pct, None, "%", str(path_bt_diag), "regime non-null %", "")
                    add_kpi_row(rows, "BACKTEST", "bt_returns_diag_regime_nonnull_pct_holdout", None, holdout_regime_nonnull_pct, "%", str(path_bt_diag), "regime non-null %", "")

                # exposure 컬럼 non-null 비율
                if "exposure" in df_bt_diag.columns:
                    dev_exposure_nonnull_pct = (df_dev_bt_diag["exposure"].notna().sum() / len(df_dev_bt_diag) * 100) if not df_dev_bt_diag.empty else None
                    holdout_exposure_nonnull_pct = (df_holdout_bt_diag["exposure"].notna().sum() / len(df_holdout_bt_diag) * 100) if not df_holdout_bt_diag.empty else None
                    add_kpi_row(rows, "BACKTEST", "bt_returns_diag_exposure_nonnull_pct_dev", dev_exposure_nonnull_pct, None, "%", str(path_bt_diag), "exposure non-null %", "")
                    add_kpi_row(rows, "BACKTEST", "bt_returns_diag_exposure_nonnull_pct_holdout", None, holdout_exposure_nonnull_pct, "%", str(path_bt_diag), "exposure non-null %", "")

                # bt_returns_diagnostics 저장 여부
                add_kpi_row(rows, "BACKTEST", "bt_returns_diag_saved", True, True, "bool", str(path_bt_diag), "bt_returns_diagnostics saved", "")

                # bt_returns_core 컬럼 수
                path_bt_core, fmt_bt_core = resolve_artifact("bt_returns", root, tag)
                if path_bt_core:
                    try:
                        df_bt_core = pd.read_parquet(path_bt_core) if fmt_bt_core == "parquet" else pd.read_csv(path_bt_core, low_memory=False)
                        core_cols = ",".join(sorted(df_bt_core.columns.tolist()))
                        core_cols_count = len(df_bt_core.columns)
                        add_kpi_row(rows, "BACKTEST", "bt_returns_core_cols_count", core_cols_count, core_cols_count, "count", str(path_bt_core), "core columns count", "")
                        add_kpi_row(rows, "BACKTEST", "bt_returns_core_cols", core_cols, core_cols, "str", str(path_bt_core), "core columns", "")
                    except Exception:
                        pass
        except Exception as e:
            pass  # bt_returns_diagnostics는 선택적이므로 실패해도 계속

    return rows

def extract_benchmark_kpis(root: Path, tag: Optional[str]) -> List[Dict[str, Any]]:
    """BENCHMARK 섹션 KPI 추출"""
    rows = []

    path, fmt = resolve_artifact("bt_benchmark_compare", root, tag)
    if not path:
        add_kpi_row(rows, "BENCHMARK", "bt_benchmark_compare", None, None, "", "", "", "MISSING ARTIFACT")
        return rows

    try:
        df = pd.read_parquet(path) if fmt == "parquet" else pd.read_csv(path, low_memory=False)

        if "phase" not in df.columns:
            add_kpi_row(rows, "BENCHMARK", "bt_benchmark_compare", None, None, "", str(path), "", "MISSING COL: phase")
            return rows

        metrics_to_extract = ["tracking_error_ann", "information_ratio", "corr_vs_benchmark", "beta_vs_benchmark"]

        df_dev = df[df["phase"] == "dev"]
        df_holdout = df[df["phase"] == "holdout"]

        row_dev = df_dev.iloc[0] if not df_dev.empty else None
        row_holdout = df_holdout.iloc[0] if not df_holdout.empty else None

        for metric in metrics_to_extract:
            dev_val = None
            holdout_val = None

            if row_dev is not None and metric in row_dev.index:
                val = row_dev[metric]
                if pd.notna(val):
                    dev_val = val

            if row_holdout is not None and metric in row_holdout.index:
                val = row_holdout[metric]
                if pd.notna(val):
                    holdout_val = val

            if dev_val is None and holdout_val is None:
                continue

            # 단위 변환
            dev_val, unit = normalize_unit(metric, dev_val)
            holdout_val, _ = normalize_unit(metric, holdout_val)

            add_kpi_row(rows, "BENCHMARK", metric, dev_val, holdout_val, unit, str(path), metric, "")

    except Exception as e:
        add_kpi_row(rows, "BENCHMARK", "bt_benchmark_compare", None, None, "", str(path), "", f"ERROR: {e}")

    return rows

def extract_stability_kpis(root: Path, tag: Optional[str]) -> List[Dict[str, Any]]:
    """STABILITY 섹션 KPI 추출"""
    rows = []

    # bt_yearly_metrics
    path, fmt = resolve_artifact("bt_yearly_metrics", root, tag)
    if path:
        try:
            df = pd.read_parquet(path) if fmt == "parquet" else pd.read_csv(path, low_memory=False)
            if "phase" in df.columns:
                df_dev = df[df["phase"] == "dev"]
                df_holdout = df[df["phase"] == "holdout"]

                if "net_total_return" in df.columns:
                    returns_dev = df_dev["net_total_return"] if not df_dev.empty else None
                    returns_holdout = df_holdout["net_total_return"] if not df_holdout.empty else None
                    if returns_dev is not None and len(returns_dev) > 0:
                        # Series를 직접 처리
                        returns_dev_pct = returns_dev * 100 if isinstance(returns_dev, pd.Series) and returns_dev.abs().max() < 10 else returns_dev
                        add_kpi_row(rows, "STABILITY", "yearly_return_min", returns_dev_pct.min() if isinstance(returns_dev_pct, pd.Series) else None, None, "%", str(path), "net_total_return", "")
                        add_kpi_row(rows, "STABILITY", "yearly_return_max", returns_dev_pct.max() if isinstance(returns_dev_pct, pd.Series) else None, None, "%", str(path), "net_total_return", "")
                        add_kpi_row(rows, "STABILITY", "yearly_return_mean", returns_dev_pct.mean() if isinstance(returns_dev_pct, pd.Series) else None, None, "%", str(path), "net_total_return", "")
                    if returns_holdout is not None and len(returns_holdout) > 0:
                        returns_holdout_pct = returns_holdout * 100 if isinstance(returns_holdout, pd.Series) and returns_holdout.abs().max() < 10 else returns_holdout
                        # 기존 행 업데이트
                        for metric_name, val in zip(["yearly_return_min", "yearly_return_max", "yearly_return_mean"],
                                                     [returns_holdout_pct.min() if isinstance(returns_holdout_pct, pd.Series) else None,
                                                      returns_holdout_pct.max() if isinstance(returns_holdout_pct, pd.Series) else None,
                                                      returns_holdout_pct.mean() if isinstance(returns_holdout_pct, pd.Series) else None]):
                            existing_idx = None
                            for i, r in enumerate(rows):
                                if r.get("metric") == metric_name:
                                    existing_idx = i
                                    break
                            if existing_idx is not None:
                                rows[existing_idx]["holdout_value"] = val
                            else:
                                add_kpi_row(rows, "STABILITY", metric_name, None, val, "%", str(path), "net_total_return", "")

                if "net_sharpe" in df.columns:
                    sharpe_dev = df_dev["net_sharpe"] if not df_dev.empty else None
                    sharpe_holdout = df_holdout["net_sharpe"] if not df_holdout.empty else None
                    if sharpe_dev is not None:
                        add_kpi_row(rows, "STABILITY", "yearly_sharpe_min", sharpe_dev.min(), None, "ratio", str(path), "net_sharpe", "")
                        add_kpi_row(rows, "STABILITY", "yearly_sharpe_max", sharpe_dev.max(), None, "ratio", str(path), "net_sharpe", "")
                        add_kpi_row(rows, "STABILITY", "yearly_sharpe_mean", sharpe_dev.mean(), None, "ratio", str(path), "net_sharpe", "")
                    if sharpe_holdout is not None:
                        for metric_name, val in zip(["yearly_sharpe_min", "yearly_sharpe_max", "yearly_sharpe_mean"],
                                                     [sharpe_holdout.min(), sharpe_holdout.max(), sharpe_holdout.mean()]):
                            existing_idx = None
                            for i, r in enumerate(rows):
                                if r.get("metric") == metric_name:
                                    existing_idx = i
                                    break
                            if existing_idx is not None:
                                rows[existing_idx]["holdout_value"] = val
                            else:
                                add_kpi_row(rows, "STABILITY", metric_name, None, val, "ratio", str(path), "net_sharpe", "")

                if "net_mdd" in df.columns:
                    mdd_dev = df_dev["net_mdd"] if not df_dev.empty else None
                    mdd_holdout = df_holdout["net_mdd"] if not df_holdout.empty else None
                    if mdd_dev is not None and len(mdd_dev) > 0:
                        mdd_dev_pct = mdd_dev * 100 if isinstance(mdd_dev, pd.Series) and mdd_dev.abs().max() < 10 else mdd_dev
                        add_kpi_row(rows, "STABILITY", "yearly_mdd_min", mdd_dev_pct.min() if isinstance(mdd_dev_pct, pd.Series) else None, None, "%", str(path), "net_mdd", "")
                    if mdd_holdout is not None and len(mdd_holdout) > 0:
                        mdd_holdout_pct = mdd_holdout * 100 if isinstance(mdd_holdout, pd.Series) and mdd_holdout.abs().max() < 10 else mdd_holdout
                        existing_idx = None
                        for i, r in enumerate(rows):
                            if r.get("metric") == "yearly_mdd_min":
                                existing_idx = i
                                break
                        if existing_idx is not None:
                            rows[existing_idx]["holdout_value"] = mdd_holdout_pct.min() if isinstance(mdd_holdout_pct, pd.Series) else None
                        else:
                            add_kpi_row(rows, "STABILITY", "yearly_mdd_min", None, mdd_holdout_pct.min() if isinstance(mdd_holdout_pct, pd.Series) else None, "%", str(path), "net_mdd", "")

                if "year" in df.columns:
                    years_dev = df_dev["year"].nunique() if not df_dev.empty else None
                    years_holdout = df_holdout["year"].nunique() if not df_holdout.empty else None
                    add_kpi_row(rows, "STABILITY", "years_covered", years_dev, years_holdout, "count", str(path), "year", "")
        except Exception as e:
            pass

    # bt_rolling_sharpe
    path, fmt = resolve_artifact("bt_rolling_sharpe", root, tag)
    if path:
        try:
            df = pd.read_parquet(path) if fmt == "parquet" else pd.read_csv(path, low_memory=False)
            if "phase" in df.columns and "net_rolling_sharpe" in df.columns:
                # net_rolling_n >= window_rebalances 조건 확인 (컬럼이 있으면)
                # window_rebalances는 데이터에 맞게 조정 (median 또는 적절한 값)
                if "net_rolling_n" in df.columns:
                    median_n = df["net_rolling_n"].median()
                    window_rebalances = max(10, int(median_n * 0.8))  # median의 80% 또는 최소 10
                    df_valid = df[df["net_rolling_n"] >= window_rebalances].copy()
                else:
                    df_valid = df.copy()  # 컬럼이 없으면 전체 사용

                df_dev = df_valid[df_valid["phase"] == "dev"]
                df_holdout = df_valid[df_valid["phase"] == "holdout"]

                if not df_dev.empty and len(df_dev) > 0:
                    sharpe_dev = df_dev["net_rolling_sharpe"]
                    add_kpi_row(rows, "STABILITY", "rolling_sharpe_min", sharpe_dev.min(), None, "ratio", str(path), "net_rolling_sharpe", "")
                    add_kpi_row(rows, "STABILITY", "rolling_sharpe_p05", sharpe_dev.quantile(0.05), None, "ratio", str(path), "net_rolling_sharpe", "")
                    add_kpi_row(rows, "STABILITY", "rolling_sharpe_median", sharpe_dev.median(), None, "ratio", str(path), "net_rolling_sharpe", "")
                    add_kpi_row(rows, "STABILITY", "rolling_sharpe_p95", sharpe_dev.quantile(0.95), None, "ratio", str(path), "net_rolling_sharpe", "")
                    add_kpi_row(rows, "STABILITY", "rolling_sharpe_max", sharpe_dev.max(), None, "ratio", str(path), "net_rolling_sharpe", "")

                if not df_holdout.empty and len(df_holdout) > 0:
                    sharpe_holdout = df_holdout["net_rolling_sharpe"]
                    # 기존 행 업데이트 또는 새 행 추가
                    metrics_to_update = ["rolling_sharpe_min", "rolling_sharpe_p05", "rolling_sharpe_median", "rolling_sharpe_p95", "rolling_sharpe_max"]
                    for metric_name, val in zip(metrics_to_update, [sharpe_holdout.min(), sharpe_holdout.quantile(0.05), sharpe_holdout.median(), sharpe_holdout.quantile(0.95), sharpe_holdout.max()]):
                        existing_idx = None
                        for i, r in enumerate(rows):
                            if r.get("metric") == metric_name:
                                existing_idx = i
                                break
                        if existing_idx is not None:
                            rows[existing_idx]["holdout_value"] = val
                        else:
                            add_kpi_row(rows, "STABILITY", metric_name, None, val, "ratio", str(path), "net_rolling_sharpe", "")
        except Exception as e:
            import traceback
            print(f"ERROR in rolling_sharpe: {e}")
            traceback.print_exc()
            pass

    # bt_drawdown_events
    path, fmt = resolve_artifact("bt_drawdown_events", root, tag)
    if path:
        try:
            df = pd.read_parquet(path) if fmt == "parquet" else pd.read_csv(path, low_memory=False)
            if "phase" in df.columns and "drawdown" in df.columns:
                df_dev = df[df["phase"] == "dev"]
                df_holdout = df[df["phase"] == "holdout"]

                if not df_dev.empty:
                    worst_dd_dev = df_dev["drawdown"].min()
                    worst_dd_dev_pct = worst_dd_dev * 100 if isinstance(worst_dd_dev, (int, float)) and abs(worst_dd_dev) < 10 else worst_dd_dev
                    worst_idx_dev = df_dev["drawdown"].idxmin()
                    worst_length_dev = df_dev.loc[worst_idx_dev, "length_days"] if "length_days" in df_dev.columns else None
                    add_kpi_row(rows, "STABILITY", "worst_drawdown", worst_dd_dev_pct, None, "%", str(path), "drawdown", "")
                    if worst_length_dev is not None:
                        add_kpi_row(rows, "STABILITY", "worst_drawdown_length_days", worst_length_dev, None, "days", str(path), "length_days", "")

                if not df_holdout.empty:
                    worst_dd_holdout = df_holdout["drawdown"].min()
                    worst_dd_holdout_pct = worst_dd_holdout * 100 if isinstance(worst_dd_holdout, (int, float)) and abs(worst_dd_holdout) < 10 else worst_dd_holdout
                    worst_idx_holdout = df_holdout["drawdown"].idxmin()
                    worst_length_holdout = df_holdout.loc[worst_idx_holdout, "length_days"] if "length_days" in df_holdout.columns else None
                    # 기존 행 업데이트
                    existing_idx = None
                    for i, r in enumerate(rows):
                        if r.get("metric") == "worst_drawdown":
                            existing_idx = i
                            break
                    if existing_idx is not None:
                        rows[existing_idx]["holdout_value"] = worst_dd_holdout_pct
                    else:
                        add_kpi_row(rows, "STABILITY", "worst_drawdown", None, worst_dd_holdout_pct, "%", str(path), "drawdown", "")

                    if worst_length_holdout is not None:
                        existing_idx = None
                        for i, r in enumerate(rows):
                            if r.get("metric") == "worst_drawdown_length_days":
                                existing_idx = i
                                break
                        if existing_idx is not None:
                            rows[existing_idx]["holdout_value"] = worst_length_holdout
                        else:
                            add_kpi_row(rows, "STABILITY", "worst_drawdown_length_days", None, worst_length_holdout, "days", str(path), "length_days", "")
        except Exception as e:
            pass

    return rows

def extract_settings_kpis(root: Path, config_path: Path, tag: str) -> List[Dict[str, Any]]:
    """SETTINGS 섹션 KPI 추출"""
    rows = []

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as e:
        add_kpi_row(rows, "SETTINGS", "config", None, None, "", str(config_path), "", f"ERROR: {e}")
        return rows

    # tag
    add_kpi_row(rows, "SETTINGS", "tag", tag, None, "str", str(config_path), "", "")

    # l4
    l4 = cfg.get("l4", {})
    l4_params = ["holdout_years", "step_days", "test_window_days", "embargo_days", "horizon_short", "horizon_long", "rolling_train_years_short", "rolling_train_years_long"]
    for param in l4_params:
        val = l4.get(param)
        add_kpi_row(rows, "SETTINGS", f"l4_{param}", val, None, "count" if "years" in param or "days" in param else "count", str(config_path), f"l4.{param}", "")

    # l5
    l5 = cfg.get("l5", {})
    model_type = l5.get("model_type", "ridge")
    add_kpi_row(rows, "SETTINGS", "l5_model_type", model_type, None, "str", str(config_path), "l5.model_type", "")
    ridge_alpha = l5.get("ridge_alpha")
    if ridge_alpha is not None:
        add_kpi_row(rows, "SETTINGS", "l5_ridge_alpha", ridge_alpha, None, "ratio", str(config_path), "l5.ridge_alpha", "")

    # l6
    l6 = cfg.get("l6", {})
    weight_short = l6.get("weight_short")
    weight_long = l6.get("weight_long")
    if weight_short is not None:
        add_kpi_row(rows, "SETTINGS", "l6_weight_short", weight_short, None, "ratio", str(config_path), "l6.weight_short", "")
    if weight_long is not None:
        add_kpi_row(rows, "SETTINGS", "l6_weight_long", weight_long, None, "ratio", str(config_path), "l6.weight_long", "")

    # l7
    l7 = cfg.get("l7", {})
    l7_params = ["holding_days", "top_k", "cost_bps", "buffer_k", "weighting", "score_col", "return_col"]
    for param in l7_params:
        val = l7.get(param)
        unit = "bps" if "bps" in param else "count" if param in ["holding_days", "top_k", "buffer_k"] else "str"
        add_kpi_row(rows, "SETTINGS", f"l7_{param}", val, None, unit, str(config_path), f"l7.{param}", "")

    return rows

def format_value(val: Any, unit: str) -> str:
    """값 포맷팅 (단위에 따라 다르게)"""
    if val is None or pd.isna(val):
        return "N/A"

    if isinstance(val, bool):
        return str(val)

    if isinstance(val, str):
        # 날짜 포맷팅
        if "date" in unit.lower() or "time" in str(val).lower():
            try:
                dt = pd.to_datetime(val)
                return dt.strftime("%Y-%m-%d")
            except:
                return str(val)
        return str(val)

    if isinstance(val, (int, float)):
        # 단위별 포맷팅
        if unit == "%":
            return f"{val:.2f}"
        elif unit == "bps":
            return f"{val:.1f}"
        elif unit == "ratio":
            return f"{val:.4f}"
        elif unit == "count":
            if val == int(val):
                return f"{int(val)}"
            return f"{val:.1f}"
        else:
            return f"{val:.4f}"

    return str(val)

def format_markdown(df: pd.DataFrame) -> str:
    """Markdown 1장 포맷팅"""
    lines = ["# KPI Table", ""]

    sections = ["DATA", "MODEL", "BACKTEST", "BENCHMARK", "STABILITY", "SETTINGS"]

    for section in sections:
        df_section = df[df["section"] == section].copy()

        if df_section.empty:
            continue

        lines.append(f"## {section}")
        lines.append("")

        if section == "MODEL":
            # horizon별로 행 펼치기
            horizons = []
            for metric in df_section["metric"].unique():
                if "__20d" in str(metric):
                    horizons.append(20)
                elif "__120d" in str(metric):
                    horizons.append(120)

            for horizon in sorted(set(horizons)):
                lines.append(f"### Horizon {horizon}d")
                lines.append("")
                df_h = df_section[df_section["metric"].str.contains(f"__{horizon}d", na=False)].copy()

                # metric에서 horizon 제거
                df_h["metric_short"] = df_h["metric"].str.replace(f"__{horizon}d", "", regex=False)

                # 정렬: mean, median, p10, p90, min, max 순서
                metric_order = ["rmse", "mae", "ic_rank", "hit_ratio", "n_folds", "n_features"]
                df_h["sort_key"] = df_h["metric_short"].apply(
                    lambda x: next((i for i, m in enumerate(metric_order) if m in str(x)), 999)
                )
                df_h = df_h.sort_values(["sort_key", "metric_short"]).drop(columns=["sort_key"])

                lines.append("| Metric | Dev | Holdout | Unit |")
                lines.append("|---|---|---|---|")
                for _, row in df_h.iterrows():
                    dev_val = format_value(row['dev_value'], row['unit'])
                    holdout_val = format_value(row['holdout_value'], row['unit'])
                    lines.append(f"| {row['metric_short']} | {dev_val} | {holdout_val} | {row['unit']} |")
                lines.append("")
        else:
            # 정렬: 알파벳 순서 또는 논리적 순서
            if section == "BACKTEST":
                priority_order = [
                    "net_total_return", "net_cagr", "net_sharpe", "net_mdd", "net_vol_ann",
                    "net_hit_ratio", "gross_total_return", "gross_minus_net_total_return_pct",
                    "cost_bps_used", "cost_bps_mismatch_flag", "avg_cost_pct",
                    "avg_turnover_oneway", "n_rebalances", "avg_n_tickers",
                    "date_start", "date_end",
                    "bt_returns_net_return_mean", "bt_returns_net_return_std", "bt_returns_turnover_oneway_mean",
                ]
                df_section["sort_key"] = df_section["metric"].apply(
                    lambda x: next((i for i, m in enumerate(priority_order) if m in str(x)), 999)
                )
                df_section = df_section.sort_values(["sort_key", "metric"]).drop(columns=["sort_key"])
            elif section == "DATA":
                priority_order = [
                    "months_count", "tickers_per_month", "duplicates_count",
                    "ohlcv_rows_total", "ohlcv_n_tickers", "ohlcv_n_dates", "ohlcv_value_nonnull_pct",
                    "fundamentals_rows_total", "fundamentals_n_tickers", "fundamentals_years_count", "fundamentals_coverage_ratio",
                    "fundamentals_lag_source",  # [Stage 1] lag_source 비율
                    "panel_rows_total", "panel_keep_ratio_vs_ohlcv", "panel_nan_pct",
                    "rebalance_n_rebalances", "rebalance_n_tickers", "rebalance_coverage", "rebalance_score_missing",
                ]
                df_section["sort_key"] = df_section["metric"].apply(
                    lambda x: next((i for i, m in enumerate(priority_order) if m in str(x)), 999)
                )
                df_section = df_section.sort_values(["sort_key", "metric"]).drop(columns=["sort_key"])
            elif section == "STABILITY":
                priority_order = [
                    "yearly_return", "yearly_sharpe", "yearly_mdd", "years_covered",
                    "rolling_sharpe", "worst_drawdown", "worst_drawdown_length_days",
                ]
                df_section["sort_key"] = df_section["metric"].apply(
                    lambda x: next((i for i, m in enumerate(priority_order) if m in str(x)), 999)
                )
                df_section = df_section.sort_values(["sort_key", "metric"]).drop(columns=["sort_key"])

            lines.append("| Metric | Dev | Holdout | Unit |")
            lines.append("|---|---|---|---|")
            for _, row in df_section.iterrows():
                dev_val = format_value(row['dev_value'], row['unit'])
                holdout_val = format_value(row['holdout_value'], row['unit'])

                # Note가 있으면 메트릭명에 표시
                metric_display = row['metric']
                if pd.notna(row['note']) and row['note']:
                    if "WARNING" in str(row['note']):
                        metric_display = f"{metric_display} ⚠️"
                    elif "MISSING" in str(row['note']):
                        metric_display = f"{metric_display} ❌"

                lines.append(f"| {metric_display} | {dev_val} | {holdout_val} | {row['unit']} |")
            lines.append("")

            # Note가 있는 행은 하단에 요약
            notes_df = df_section[df_section["note"].notna() & (df_section["note"] != "")]
            if not notes_df.empty:
                lines.append("**Notes:**")
                for _, note_row in notes_df.iterrows():
                    lines.append(f"- `{note_row['metric']}`: {note_row['note']}")
                lines.append("")

        # Sources
        sources = df_section["source_file"].dropna().unique()
        if len(sources) > 0:
            lines.append("**Sources:**")
            for src in sources:
                if src:
                    # 경로를 상대 경로로 변환 (가독성)
                    try:
                        src_path = Path(src)
                        if len(src) > 80 and len(src_path.parts) > 3:
                            src_display = "..." + "/".join(src_path.parts[-3:])
                        else:
                            src_display = src
                    except:
                        src_display = src
                    lines.append(f"- `{src_display}`")
            lines.append("")

    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Export KPI Table from Artifacts")
    parser.add_argument("--root", type=str, default=None)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--out-dir", type=str, default="reports/kpi")
    parser.add_argument("--format", type=str, default="both", choices=["csv", "md", "both"],
                       help="Output format: csv, md, or both (default: both)")
    parser.add_argument("--no-md", action="store_true", default=False,
                       help="[TASK A-1] MD 렌더링 비활성화 (CSV만 생성)")
    args = parser.parse_args()

    # [TASK A-1] --no-md 옵션이면 format을 csv로 변경
    if args.no_md:
        args.format = "csv"

    # 루트 경로 결정
    if args.root:
        root = Path(args.root)
    else:
        root = Path(__file__).resolve().parents[2]

    config_path = root / args.config
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[KPI Export] Root: {root}")
    print(f"[KPI Export] Tag: {args.tag}")
    print(f"[KPI Export] Config: {config_path}")

    # config에서 cost_bps 읽기
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f) or {}
        config_cost_bps = cfg.get("l7", {}).get("cost_bps")
    except Exception:
        config_cost_bps = None

    # KPI 추출
    all_rows = []
    all_rows.extend(extract_data_kpis(root, args.tag))
    all_rows.extend(extract_model_kpis(root, args.tag))
    all_rows.extend(extract_backtest_kpis(root, args.tag, config_cost_bps))
    all_rows.extend(extract_benchmark_kpis(root, args.tag))
    all_rows.extend(extract_stability_kpis(root, args.tag))
    all_rows.extend(extract_settings_kpis(root, config_path, args.tag))

    df = pd.DataFrame(all_rows)

    # 중복 제거: metric별로 groupby해서 dev/holdout을 한 행에 합침
    if len(df) > 0:
        df_grouped = df.groupby(["section", "metric"], as_index=False).agg({
            "dev_value": lambda x: x.dropna().iloc[0] if x.dropna().size > 0 else None,
            "holdout_value": lambda x: x.dropna().iloc[0] if x.dropna().size > 0 else None,
            "unit": "first",
            "source_file": lambda x: ", ".join(x.dropna().unique()) if x.dropna().size > 0 else "",
            "source_cols": "first",
            "note": lambda x: "; ".join(x.dropna().unique()) if x.dropna().size > 0 else "",
        })
        df = df_grouped

    # 검증
    if len(df) < 40:
        print(f"WARNING: KPI table has only {len(df)} rows (expected >= 40)", file=sys.stderr)

    backtest_dev = df[(df["section"] == "BACKTEST") & (df["metric"] == "net_total_return")]["dev_value"].iloc[0] if len(df[(df["section"] == "BACKTEST") & (df["metric"] == "net_total_return")]) > 0 else None
    backtest_holdout = df[(df["section"] == "BACKTEST") & (df["metric"] == "net_total_return")]["holdout_value"].iloc[0] if len(df[(df["section"] == "BACKTEST") & (df["metric"] == "net_total_return")]) > 0 else None

    if pd.isna(backtest_dev) and pd.isna(backtest_holdout):
        print("ERROR: Both dev and holdout net_total_return are missing", file=sys.stderr)
        sys.exit(2)

    # 저장
    if args.format in ["csv", "both"]:
        csv_path = out_dir / f"kpi_table__{args.tag}.csv"
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"[KPI Export] CSV saved: {csv_path}")

    if args.format in ["md", "both"]:
        md_path = out_dir / f"kpi_table__{args.tag}.md"
        md_content = format_markdown(df)
        md_path.write_text(md_content, encoding="utf-8")
        print(f"[KPI Export] Markdown saved: {md_path}")

    # 핵심 KPI 요약 출력
    print("\n=== Key Performance Indicators ===")
    if pd.notna(backtest_dev):
        print(f"Dev Net Total Return: {backtest_dev:.2f}%")
    if pd.notna(backtest_holdout):
        print(f"Holdout Net Total Return: {backtest_holdout:.2f}%")

    dev_sharpe = df[(df["section"] == "BACKTEST") & (df["metric"] == "net_sharpe")]["dev_value"].iloc[0] if len(df[(df["section"] == "BACKTEST") & (df["metric"] == "net_sharpe")]) > 0 else None
    holdout_sharpe = df[(df["section"] == "BACKTEST") & (df["metric"] == "net_sharpe")]["holdout_value"].iloc[0] if len(df[(df["section"] == "BACKTEST") & (df["metric"] == "net_sharpe")]) > 0 else None

    if pd.notna(dev_sharpe):
        print(f"Dev Sharpe Ratio: {dev_sharpe:.4f}")
    if pd.notna(holdout_sharpe):
        print(f"Holdout Sharpe Ratio: {holdout_sharpe:.4f}")

    print(f"\n[KPI Export] Completed. Total KPIs: {len(df)}")

if __name__ == "__main__":
    main()
