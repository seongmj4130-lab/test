# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/stages/export/l8_generate_report_outputs.py
"""
L8: Generate report artifacts
- Equity curve (strategy vs benchmark)
- Monthly returns table + annual performance
- Rolling beta vs benchmark
- Sector weight time series (if holdings + sector map available)
- Gross vs net comparison (if both available)
- Turnover distribution (if holdings available)
- Factor exposure regression (if factor returns file provided)

Run example:
python src/stages/l8_generate_report_outputs.py --tag today_L7C_fix01 --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -------------------------
# Helpers
# -------------------------
def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"파일이 없습니다: {path}")
    suf = path.suffix.lower()
    if suf in [".parquet"]:
        return pd.read_parquet(path)
    if suf in [".csv"]:
        return pd.read_csv(path)
    if suf in [".feather"]:
        return pd.read_feather(path)
    raise ValueError(f"지원하지 않는 확장자: {suf} ({path})")

def _try_import_yaml():
    try:
        import yaml  # type: ignore
        return yaml
    except Exception:
        return None

def _load_config_paths(base_dir: Path, config_path: Optional[Path]) -> Dict[str, Path]:
    """
    Minimal YAML loader.
    Expected keys are flexible; we will fallback to base_dir/data/processed, raw, interim.
    """
    paths = {
        "BASE_DIR": base_dir,
        "RAW_DIR": base_dir / "data" / "raw",
        "INTERIM_DIR": base_dir / "data" / "interim",
        "PROCESSED_DIR": base_dir / "data" / "processed",
        "REPORT_DIR": base_dir / "reports",
    }

    if config_path is None:
        return paths

    yaml = _try_import_yaml()
    if yaml is None:
        print("[WARN] PyYAML이 없어 config.yaml 파싱을 건너뜁니다. 기본 경로로 진행합니다.")
        return paths

    if not config_path.exists():
        print(f"[WARN] config 파일이 없습니다: {config_path}. 기본 경로로 진행합니다.")
        return paths

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    # 여러 프로젝트에서 PATH/paths 구조가 다를 수 있으니 최대한 방어적으로 처리
    cand = None
    for key in ["PATH", "paths", "path", "Paths"]:
        if isinstance(cfg, dict) and key in cfg and isinstance(cfg[key], dict):
            cand = cfg[key]
            break

    if isinstance(cand, dict):
        for k, v in cand.items():
            try:
                p = Path(v)
                if not p.is_absolute():
                    p = base_dir / p
                paths[str(k).upper()] = p
            except Exception:
                pass

    # 최소한 자주 쓰는 키를 보정
    if "RAW_DIR" not in paths:
        paths["RAW_DIR"] = base_dir / "data" / "raw"
    if "PROCESSED_DIR" not in paths:
        paths["PROCESSED_DIR"] = base_dir / "data" / "processed"
    if "INTERIM_DIR" not in paths:
        paths["INTERIM_DIR"] = base_dir / "data" / "interim"
    if "REPORT_DIR" not in paths:
        paths["REPORT_DIR"] = base_dir / "reports"

    return paths

def _ensure_date_index(df: pd.DataFrame, date_col: Optional[str] = None) -> pd.DataFrame:
    cands = [date_col] if date_col else []
    cands += ["date", "dt", "trade_date", "datetime", "asof", "rebalance_date"]
    found = None
    for c in cands:
        if c and c in df.columns:
            found = c
            break
    if found is None:
        raise ValueError(f"날짜 컬럼을 찾지 못했습니다. columns={list(df.columns)[:20]} ...")

    out = df.copy()
    out[found] = pd.to_datetime(out[found])
    out = out.sort_values(found).set_index(found)
    return out

def _pick_col(df: pd.DataFrame, preferred: List[str], required: bool = True) -> Optional[str]:
    for c in preferred:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"필요 컬럼을 찾지 못했습니다. 후보={preferred}, 실제={list(df.columns)[:50]}")
    return None

def _normalize_return_scale(s: pd.Series) -> Tuple[pd.Series, str]:
    """
    If returns look like percent (e.g., 2.3 meaning 2.3%), convert to decimals.
    Heuristic only.
    """
    s2 = s.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(s2) == 0:
        return s.astype(float), "unknown"
    max_abs = float(np.nanmax(np.abs(s2)))
    # if returns frequently exceed 1.5, it's likely percent units
    if max_abs > 1.5:
        return (s.astype(float) / 100.0), "percent->decimal"
    return s.astype(float), "decimal"

def _equity_curve(ret: pd.Series) -> pd.Series:
    r = ret.fillna(0.0)
    return (1.0 + r).cumprod()

def _max_drawdown(eq: pd.Series) -> float:
    peak = eq.cummax()
    dd = eq / peak - 1.0
    return float(dd.min())

def _annualized_stats(ret: pd.Series, freq: int = 252) -> Dict[str, float]:
    r = ret.dropna().astype(float)
    if len(r) == 0:
        return {"cagr": np.nan, "vol": np.nan, "sharpe": np.nan}
    mu = float(r.mean())
    sd = float(r.std(ddof=1))
    ann_mu = mu * freq
    ann_sd = sd * np.sqrt(freq)
    sharpe = (ann_mu / ann_sd) if ann_sd > 0 else np.nan
    return {"ann_mean": ann_mu, "ann_vol": ann_sd, "sharpe": sharpe}

def _compound_return(ret: pd.Series) -> float:
    r = ret.dropna().astype(float)
    if len(r) == 0:
        return np.nan
    return float((1.0 + r).prod() - 1.0)

def _monthly_table(ret: pd.Series) -> pd.DataFrame:
    r = ret.dropna().astype(float)
    if len(r) == 0:
        return pd.DataFrame()
    m = (1.0 + r).groupby(pd.Grouper(freq="M")).prod() - 1.0
    out = m.to_frame("monthly_return")
    out["year"] = out.index.year
    out["month"] = out.index.month
    pivot = out.pivot_table(index="year", columns="month", values="monthly_return", aggfunc="first")
    pivot = pivot.sort_index()
    pivot.columns = [f"{int(c):02d}" for c in pivot.columns]
    pivot["YTD"] = pivot.apply(lambda row: float((1.0 + row.dropna()).prod() - 1.0), axis=1)
    return pivot

def _annual_performance(ret: pd.Series, freq: int = 252) -> pd.DataFrame:
    r = ret.dropna().astype(float)
    if len(r) == 0:
        return pd.DataFrame()
    years = sorted(r.index.year.unique())
    rows = []
    for y in years:
        ry = r[r.index.year == y]
        eq = _equity_curve(ry)
        dd = _max_drawdown(eq)
        stats = _annualized_stats(ry, freq=freq)
        rows.append({
            "year": y,
            "total_return": _compound_return(ry),
            "ann_mean": stats["ann_mean"],
            "ann_vol": stats["ann_vol"],
            "sharpe": stats["sharpe"],
            "max_drawdown": dd,
            "n_obs": int(len(ry)),
        })
    return pd.DataFrame(rows).set_index("year")

def _rolling_beta(strat: pd.Series, bench: pd.Series, window: int = 252, min_periods: int = 60) -> pd.Series:
    df = pd.concat([strat.rename("s"), bench.rename("b")], axis=1).dropna()
    if len(df) == 0:
        return pd.Series(dtype=float)
    cov = df["s"].rolling(window, min_periods=min_periods).cov(df["b"])
    var = df["b"].rolling(window, min_periods=min_periods).var()
    beta = cov / var
    return beta.rename("rolling_beta")

def _find_by_glob(search_dirs: List[Path], patterns: List[str]) -> List[Path]:
    hits: List[Path] = []
    for d in search_dirs:
        if not d.exists():
            continue
        for pat in patterns:
            hits.extend(list(d.rglob(pat)))
    # remove duplicates
    uniq = []
    seen = set()
    for p in hits:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq

def _pick_latest(paths: List[Path]) -> Optional[Path]:
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)

# -------------------------
# Loaders
# -------------------------
def load_returns_auto(paths: Dict[str, Path], tag: str, returns_path: Optional[Path]) -> pd.DataFrame:
    if returns_path is not None:
        df = _read_table(returns_path)
        return df

    search_dirs = [paths["PROCESSED_DIR"], paths["INTERIM_DIR"], paths["BASE_DIR"] / "data"]
    patterns = [
        f"*{tag}*return*.parquet",
        f"*{tag}*bt*.parquet",
        f"*bt*{tag}*.parquet",
        f"*{tag}*return*.csv",
        f"*{tag}*bt*.csv",
    ]
    cands = _find_by_glob(search_dirs, patterns)
    picked = _pick_latest(cands)
    if picked is None:
        raise FileNotFoundError(
            "returns 파일을 자동으로 찾지 못했습니다. "
            "직접 --returns-path 로 지정해 주세요."
        )
    print(f"[INFO] returns 자동 선택: {picked}")
    return _read_table(picked)

def load_holdings_auto(paths: Dict[str, Path], tag: str, holdings_path: Optional[Path]) -> Optional[pd.DataFrame]:
    if holdings_path is not None:
        return _read_table(holdings_path)

    search_dirs = [paths["PROCESSED_DIR"], paths["INTERIM_DIR"], paths["BASE_DIR"] / "data"]
    patterns = [
        f"*{tag}*holding*.parquet",
        f"*{tag}*position*.parquet",
        f"*{tag}*portfolio*.parquet",
        f"*{tag}*holding*.csv",
        f"*{tag}*position*.csv",
        f"*{tag}*portfolio*.csv",
    ]
    cands = _find_by_glob(search_dirs, patterns)
    picked = _pick_latest(cands)
    if picked is None:
        print("[WARN] holdings 파일을 찾지 못했습니다. 섹터/턴오버 관련 산출물을 건너뜁니다.")
        return None
    print(f"[INFO] holdings 자동 선택: {picked}")
    return _read_table(picked)

def load_universe_auto(paths: Dict[str, Path], universe_path: Optional[Path]) -> Optional[pd.DataFrame]:
    if universe_path is not None:
        return _read_table(universe_path)

    raw_dir = paths.get("RAW_DIR", paths["BASE_DIR"] / "data" / "raw")
    if not raw_dir.exists():
        return None

    patterns = [
        "kospi200_universe*.parquet",
        "universe*.parquet",
        "kospi200_universe*.csv",
        "universe*.csv",
    ]
    cands = _find_by_glob([raw_dir], patterns)
    picked = _pick_latest(cands)
    if picked is None:
        print("[WARN] universe(섹터맵) 파일을 찾지 못했습니다. 섹터 비중 산출을 건너뜁니다.")
        return None
    print(f"[INFO] universe 자동 선택: {picked}")
    return _read_table(picked)

# -------------------------
# Report generators
# -------------------------
def make_equity_and_tables(
    df_ret: pd.DataFrame,
    outdir: Path,
    date_col: Optional[str],
    strat_net_col: Optional[str],
    strat_gross_col: Optional[str],
    bench_col: Optional[str],
    freq: int,
    beta_window: int,
    beta_minp: int,
) -> Dict[str, pd.Series]:
    outdir.mkdir(parents=True, exist_ok=True)

    dfi = _ensure_date_index(df_ret, date_col)

    # pick columns
    net_col = strat_net_col or _pick_col(dfi, ["net_return", "strategy_net_return", "ret_net", "pnl_net"])
    bmk_col = bench_col or _pick_col(dfi, ["bench_return", "benchmark_return", "bm_return", "index_return"])
    gross_col = strat_gross_col
    if gross_col is None:
        gross_col = _pick_col(dfi, ["gross_return", "strategy_gross_return", "ret_gross", "pnl_gross"], required=False)

    strat_net, net_scale = _normalize_return_scale(dfi[net_col])
    bench, bench_scale = _normalize_return_scale(dfi[bmk_col])
    if gross_col is not None:
        strat_gross, gross_scale = _normalize_return_scale(dfi[gross_col])
    else:
        strat_gross, gross_scale = (pd.Series(index=dfi.index, dtype=float), "missing")

    print(f"[INFO] return scale: net={net_scale}, bench={bench_scale}, gross={gross_scale}")

    # align
    df = pd.concat(
        [
            strat_net.rename("strategy_net"),
            bench.rename("benchmark"),
            strat_gross.rename("strategy_gross") if gross_col is not None else None,
        ],
        axis=1
    )
    df = df.dropna(subset=["strategy_net", "benchmark"], how="any")

    # equity curves
    eq_s = _equity_curve(df["strategy_net"])
    eq_b = _equity_curve(df["benchmark"])

    # plots: equity curve
    plt.figure()
    plt.plot(eq_s.index, eq_s.values, label="strategy_net")
    plt.plot(eq_b.index, eq_b.values, label="benchmark")
    plt.title("Equity Curve (Cumulative)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "equity_curve.png", dpi=160)
    plt.close()

    # save time series
    equity_df = pd.DataFrame({"strategy_net_equity": eq_s, "benchmark_equity": eq_b})
    equity_df.to_csv(outdir / "equity_curve.csv", encoding="utf-8-sig")

    # monthly table
    mtab_s = _monthly_table(df["strategy_net"])
    mtab_b = _monthly_table(df["benchmark"])
    mtab_s.to_csv(outdir / "monthly_returns_strategy.csv", encoding="utf-8-sig")
    mtab_b.to_csv(outdir / "monthly_returns_benchmark.csv", encoding="utf-8-sig")

    # annual performance
    ann_s = _annual_performance(df["strategy_net"], freq=freq)
    ann_b = _annual_performance(df["benchmark"], freq=freq)
    ann_s.to_csv(outdir / "annual_performance_strategy.csv", encoding="utf-8-sig")
    ann_b.to_csv(outdir / "annual_performance_benchmark.csv", encoding="utf-8-sig")

    # rolling beta
    beta = _rolling_beta(df["strategy_net"], df["benchmark"], window=beta_window, min_periods=beta_minp)
    beta.to_csv(outdir / "rolling_beta.csv", encoding="utf-8-sig")

    plt.figure()
    plt.plot(beta.index, beta.values)
    plt.title(f"Rolling Beta (window={beta_window})")
    plt.tight_layout()
    plt.savefig(outdir / "rolling_beta.png", dpi=160)
    plt.close()

    # gross vs net / cost impact
    if gross_col is not None and "strategy_gross" in df.columns:
        comp = pd.DataFrame({
            "gross": df["strategy_gross"],
            "net": df["strategy_net"],
            "cost_impact": df["strategy_gross"] - df["strategy_net"],
        })
        comp.to_csv(outdir / "gross_vs_net.csv", encoding="utf-8-sig")

        eq_g = _equity_curve(comp["gross"])
        eq_n = _equity_curve(comp["net"])

        plt.figure()
        plt.plot(eq_g.index, eq_g.values, label="gross_equity")
        plt.plot(eq_n.index, eq_n.values, label="net_equity")
        plt.title("Gross vs Net Equity (Transaction Cost Impact)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "gross_vs_net_equity.png", dpi=160)
        plt.close()

    return {"strategy_net": df["strategy_net"], "benchmark": df["benchmark"]}

def make_sector_and_turnover(
    df_hold: pd.DataFrame,
    df_univ: pd.DataFrame,
    outdir: Path,
    date_col_hold: Optional[str],
    ticker_col: Optional[str],
    weight_col: Optional[str],
    sector_col: Optional[str],
):
    """
    holdings: columns expected (date, ticker, weight)
    universe: columns expected (ticker, sector/industry)
    """
    outdir.mkdir(parents=True, exist_ok=True)

    h = df_hold.copy()
    h = _ensure_date_index(h, date_col_hold)

    tcol = ticker_col or _pick_col(h, ["ticker", "code", "symbol", "asset", "secid"])
    wcol = weight_col or _pick_col(h, ["weight", "w", "port_weight", "allocation", "value_weight"])

    u = df_univ.copy()
    ut = _pick_col(u, ["ticker", "code", "symbol"], required=True)

    # sector column candidates
    scol = sector_col
    if scol is None:
        scol = _pick_col(u, ["sector", "industry", "sector_name", "industry_name", "gics_sector"], required=False)
    if scol is None:
        raise ValueError("universe에서 섹터 컬럼을 찾지 못했습니다. --sector-col 로 지정해 주세요.")

    u = u[[ut, scol]].drop_duplicates()
    u = u.rename(columns={ut: "ticker_u", scol: "sector_u"})

    hh = h.reset_index().rename(columns={tcol: "ticker_u", wcol: "weight"})
    hh = hh.merge(u, on="ticker_u", how="left")

    if hh["sector_u"].isna().mean() > 0.3:
        print("[WARN] 섹터 매칭이 많이 비었습니다. ticker 포맷(6자리) / universe 매핑을 확인하세요.")

    # sector weights time series
    sector_ts = hh.groupby(["date", "sector_u"])["weight"].sum().unstack("sector_u").fillna(0.0)
    sector_ts.to_csv(outdir / "sector_weights_timeseries.csv", encoding="utf-8-sig")

    # plot stacked area (default colors)
    plt.figure(figsize=(12, 6))
    plt.stackplot(sector_ts.index, sector_ts.T.values, labels=sector_ts.columns)
    plt.title("Sector Weights (Time Series)")
    plt.legend(loc="upper left", ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "sector_weights_timeseries.png", dpi=160)
    plt.close()

    # turnover distribution: 0.5*sum|w_t - w_{t-1}|
    # build pivot ticker weights
    w_pivot = hh.pivot_table(index="date", columns="ticker_u", values="weight", aggfunc="sum").fillna(0.0)
    w_diff = w_pivot.diff().abs().sum(axis=1) * 0.5
    w_diff = w_diff.dropna()
    w_diff.rename("turnover_oneway").to_csv(outdir / "turnover_series.csv", encoding="utf-8-sig")

    plt.figure()
    plt.hist(w_diff.values, bins=30)
    plt.title("Turnover Distribution (One-way)")
    plt.tight_layout()
    plt.savefig(outdir / "turnover_hist.png", dpi=160)
    plt.close()

def factor_exposure_regression(
    strat_ret: pd.Series,
    factor_returns_path: Path,
    outdir: Path,
    date_col: Optional[str] = None,
    rf_col: Optional[str] = None,
):
    """
    Time-series regression:
      (strategy - rf) = alpha + sum(beta_i * factor_i) + eps
    factor_returns file should have date + factor columns (and optionally rf).
    """
    outdir.mkdir(parents=True, exist_ok=True)
    f = _read_table(factor_returns_path)
    f = _ensure_date_index(f, date_col)

    # pick rf
    if rf_col is None:
        rf_col = _pick_col(f, ["rf", "risk_free", "riskfree", "r_f"], required=False)

    # build X factors
    cols = [c for c in f.columns if c not in ([rf_col] if rf_col else [])]
    if len(cols) == 0:
        raise ValueError("팩터 컬럼이 없습니다. factor_returns 파일 컬럼을 확인하세요.")

    df = pd.concat([strat_ret.rename("strategy"), f[cols]], axis=1).dropna()
    if rf_col and rf_col in f.columns:
        df = pd.concat([df, f[rf_col].rename("rf")], axis=1).dropna()
        y = (df["strategy"] - df["rf"]).astype(float)
    else:
        y = df["strategy"].astype(float)

    X = df[cols].astype(float)
    # add intercept
    X_ = np.column_stack([np.ones(len(X)), X.values])
    y_ = y.values.reshape(-1, 1)

    # OLS (numpy)
    beta = np.linalg.lstsq(X_, y_, rcond=None)[0].flatten()
    alpha = beta[0]
    betas = beta[1:]

    # diagnostics
    y_hat = X_ @ beta.reshape(-1, 1)
    resid = (y_ - y_hat).flatten()
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_.flatten() - y_.flatten().mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    out = pd.DataFrame({
        "coef": [alpha] + list(betas),
    }, index=["alpha"] + cols)
    out.loc["R2", "coef"] = r2
    out.to_csv(outdir / "factor_exposure_ols.csv", encoding="utf-8-sig")

    # residual plot
    plt.figure()
    plt.plot(df.index, resid)
    plt.title("Factor Regression Residuals (OLS)")
    plt.tight_layout()
    plt.savefig(outdir / "factor_regression_residuals.png", dpi=160)
    plt.close()

# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default=None, help="프로젝트 루트. 미지정 시 현재 작업 디렉토리")
    parser.add_argument("--config", type=str, default=None, help="configs/config.yaml 경로(선택)")
    parser.add_argument("--tag", type=str, required=True, help="실행 tag (예: today_L7C_fix01)")

    # input paths override
    parser.add_argument("--returns-path", type=str, default=None, help="returns 파일 경로(parquet/csv)")
    parser.add_argument("--holdings-path", type=str, default=None, help="holdings 파일 경로(parquet/csv)")
    parser.add_argument("--universe-path", type=str, default=None, help="universe(섹터맵) 파일 경로(parquet/csv)")
    parser.add_argument("--factor-returns-path", type=str, default=None, help="팩터 수익률 파일 경로(parquet/csv)")

    # column overrides
    parser.add_argument("--date-col", type=str, default=None, help="returns 날짜 컬럼명(선택)")
    parser.add_argument("--strat-net-col", type=str, default=None, help="전략 net return 컬럼명(선택)")
    parser.add_argument("--strat-gross-col", type=str, default=None, help="전략 gross return 컬럼명(선택)")
    parser.add_argument("--bench-col", type=str, default=None, help="벤치마크 return 컬럼명(선택)")

    parser.add_argument("--hold-date-col", type=str, default=None, help="holdings 날짜 컬럼명(선택)")
    parser.add_argument("--hold-ticker-col", type=str, default=None, help="holdings ticker 컬럼명(선택)")
    parser.add_argument("--hold-weight-col", type=str, default=None, help="holdings weight 컬럼명(선택)")
    parser.add_argument("--univ-sector-col", type=str, default=None, help="universe 섹터 컬럼명(선택)")

    # stats params
    parser.add_argument("--freq", type=int, default=252, help="연환산 기준 빈도(일봉=252)")
    parser.add_argument("--beta-window", type=int, default=252, help="rolling beta window")
    parser.add_argument("--beta-minp", type=int, default=60, help="rolling beta min periods")

    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve() if args.base_dir else Path.cwd().resolve()
    config_path = Path(args.config).resolve() if args.config else None
    paths = _load_config_paths(base_dir, config_path)

    outdir = paths["REPORT_DIR"] / args.tag
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] output dir: {outdir}")

    # 1) returns based artifacts
    df_ret = load_returns_auto(
        paths, args.tag,
        Path(args.returns_path).resolve() if args.returns_path else None
    )
    series = make_equity_and_tables(
        df_ret=df_ret,
        outdir=outdir,
        date_col=args.date_col,
        strat_net_col=args.strat_net_col,
        strat_gross_col=args.strat_gross_col,
        bench_col=args.bench_col,
        freq=args.freq,
        beta_window=args.beta_window,
        beta_minp=args.beta_minp,
    )

    # 2) holdings + universe -> sector weights & turnover
    df_hold = load_holdings_auto(
        paths, args.tag,
        Path(args.holdings_path).resolve() if args.holdings_path else None
    )
    df_univ = load_universe_auto(
        paths,
        Path(args.universe_path).resolve() if args.universe_path else None
    )
    if df_hold is not None and df_univ is not None:
        try:
            make_sector_and_turnover(
                df_hold=df_hold,
                df_univ=df_univ,
                outdir=outdir,
                date_col_hold=args.hold_date_col,
                ticker_col=args.hold_ticker_col,
                weight_col=args.hold_weight_col,
                sector_col=args.univ_sector_col,
            )
        except Exception as e:
            print(f"[WARN] 섹터/턴오버 산출 실패: {e}")

    # 3) factor exposure regression (optional)
    if args.factor_returns_path:
        try:
            factor_exposure_regression(
                strat_ret=series["strategy_net"],
                factor_returns_path=Path(args.factor_returns_path).resolve(),
                outdir=outdir,
                date_col=None,
                rf_col=None,
            )
        except Exception as e:
            print(f"[WARN] 팩터 노출 회귀 실패: {e}")

    print("[DONE] report artifacts generated.")

if __name__ == "__main__":
    main()
