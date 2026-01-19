# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/stages/backtest/l7c_benchmark.py
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _require_pykrx():
    """
    [개선안 17번] KOSPI200 벤치마크 계산을 위해 pykrx 로딩
    """
    try:
        from pykrx import stock

        return stock
    except Exception as e:
        raise ImportError(
            "pykrx가 필요합니다. `pip install pykrx` 후 재실행하세요."
        ) from e


def _to_yyyymmdd(s: pd.Timestamp) -> str:
    return pd.to_datetime(s).strftime("%Y%m%d")


def _resolve_section(cfg: dict, name: str) -> dict:
    if not isinstance(cfg, dict):
        return {}
    params = cfg.get("params", {}) or {}
    if isinstance(params, dict):
        sec = params.get(name, None)
        if isinstance(sec, dict):
            return sec
    sec2 = cfg.get(name, None)
    return sec2 if isinstance(sec2, dict) else {}


def _pick_strategy_return_series(bt_returns: pd.DataFrame) -> tuple[pd.Series, str]:
    cols = set(bt_returns.columns)
    net_candidates = [
        "net_return",
        "ret_net",
        "net_ret",
        "return_net",
        "net",
        "net_r",
        "portfolio_net_return",
    ]
    for c in net_candidates:
        if c in cols:
            return bt_returns[c].astype(float), c

    gross_candidates = ["gross_return", "ret_gross", "gross_ret", "return_gross"]
    gross_col: Optional[str] = None
    for c in gross_candidates:
        if c in cols:
            gross_col = c
            break

    if gross_col is not None:
        g = bt_returns[gross_col].astype(float)
        cost_candidates = ["cost", "trade_cost", "tcost", "cost_rate"]
        for cc in cost_candidates:
            if cc in cols:
                return (g - bt_returns[cc].astype(float)), f"{gross_col}-({cc})"

        turnover_candidates = ["turnover_oneway", "turnover", "oneway_turnover"]
        cost_bps_candidates = ["cost_bps", "tcost_bps"]
        tcol = next((c for c in turnover_candidates if c in cols), None)
        cbps = next((c for c in cost_bps_candidates if c in cols), None)
        if tcol is not None and cbps is not None:
            cost = (
                bt_returns[tcol].astype(float)
                * bt_returns[cbps].astype(float)
                / 10000.0
            )
            return (g - cost), f"{gross_col}-({tcol}*{cbps}/10000)"
        return g, gross_col

    fallback = ["return", "ret", "r", "pnl", "strategy_return", "portfolio_return"]
    for c in fallback:
        if c in cols:
            return bt_returns[c].astype(float), c
    raise KeyError(f"bt_returns missing return column. cols={sorted(list(cols))}")


def build_universe_benchmark_returns(
    rebalance_scores: pd.DataFrame,
    *,
    ret_col_candidates=None,
    date_col: str = "date",
    ticker_col: str = "ticker",
    phase_col: str = "phase",
) -> pd.DataFrame:
    if ret_col_candidates is None:
        ret_col_candidates = ["true_short", "y_true", "ret_fwd_20d", "ret", "return"]

    for c in [date_col, phase_col, ticker_col]:
        if c not in rebalance_scores.columns:
            raise KeyError(f"rebalance_scores missing required column: {c}")

    ret_col = None
    for c in ret_col_candidates:
        if c in rebalance_scores.columns:
            ret_col = c
            break
    if ret_col is None:
        raise KeyError(
            f"no benchmark return col in rebalance_scores. tried={ret_col_candidates}"
        )

    df = rebalance_scores[[date_col, phase_col, ticker_col, ret_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df[phase_col] = df[phase_col].astype(str)
    df[ticker_col] = df[ticker_col].astype(str)

    bench = (
        df.dropna(subset=[ret_col])
        .groupby([phase_col, date_col], sort=False)[ret_col]
        .mean()
        .reset_index()
        .rename(columns={ret_col: "bench_return"})
        .sort_values([phase_col, date_col])
        .reset_index(drop=True)
    )

    bench["bench_equity"] = 1.0
    for phase, g in bench.groupby(phase_col, sort=False):
        idx = g.index
        bench.loc[idx, "bench_equity"] = (
            (1.0 + g["bench_return"].astype(float)).cumprod().values
        )

    return bench


def build_savings_benchmark_returns(
    bt_returns: pd.DataFrame,
    *,
    savings_apr: float = 0.03,
    holding_days: int,
) -> pd.DataFrame:
    """
    [개선안 17번] 적금(고정금리) 벤치마크를 bt_returns의 (phase,date) 그리드에 맞춰 생성
    - holding_days 기준으로 '리밸런싱 1회' 수익률로 환산
    """
    df = bt_returns[["phase", "date"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["phase"] = df["phase"].astype(str)
    df = df.dropna(subset=["date"])
    if len(df) == 0:
        return pd.DataFrame(columns=["phase", "date", "bench_return", "bench_type"])
    apr = float(savings_apr)
    per = (1.0 + apr) ** (float(holding_days) / 252.0) - 1.0
    df["bench_return"] = float(per)
    df["bench_type"] = "savings"
    return df.sort_values(["phase", "date"]).reset_index(drop=True)


def build_kospi200_benchmark_returns(
    bt_returns: pd.DataFrame,
    *,
    holding_days: int,
    index_code: str = "1028",
) -> pd.DataFrame:
    """
    [개선안 17번] KOSPI200 지수 'forward holding_days' 수익률을 bt_returns의 (phase,date)에 맞춰 생성
    - pykrx의 일별 지수 close를 받아서 trading-day 기준 shift(-holding_days)로 forward return 계산
    """
    df = bt_returns[["phase", "date"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["phase"] = df["phase"].astype(str)
    df = df.dropna(subset=["date"])
    if len(df) == 0:
        return pd.DataFrame(columns=["phase", "date", "bench_return", "bench_type"])

    dates = pd.DatetimeIndex(df["date"].unique()).sort_values()
    stock = _require_pykrx()
    s = _to_yyyymmdd(dates.min())
    e = _to_yyyymmdd(dates.max())
    idx = stock.get_index_ohlcv_by_date(s, e, str(index_code))
    if idx is None or len(idx) == 0:
        raise RuntimeError(
            f"지수 데이터가 비어있습니다 (index_code={index_code}, {s}~{e})"
        )

    idx = idx.reset_index()
    rename_map = {"날짜": "date", "종가": "close"}
    for old, new in rename_map.items():
        if old in idx.columns:
            idx = idx.rename(columns={old: new})
    if "date" not in idx.columns:
        idx = idx.rename(columns={idx.columns[0]: "date"})
    if "close" not in idx.columns:
        close_candidates = ["종가", "close", "Close", "CLOSE"]
        close_col = next((c for c in close_candidates if c in idx.columns), None)
        if close_col is None:
            raise RuntimeError(
                f"지수 데이터에 close(종가) 컬럼이 없습니다. cols={list(idx.columns)}"
            )
        idx = idx.rename(columns={close_col: "close"})

    idx["date"] = pd.to_datetime(idx["date"], errors="coerce")
    idx["close"] = pd.to_numeric(idx["close"], errors="coerce")
    idx = (
        idx.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    )

    # trading-day forward return
    hd = int(holding_days)
    idx["close_fwd"] = idx["close"].shift(-hd)
    idx["bench_return"] = (idx["close_fwd"] / idx["close"]) - 1.0
    idx = idx.dropna(subset=["bench_return"])
    idx_ret = idx[["date", "bench_return"]].copy()

    out = df.merge(idx_ret, on="date", how="left", validate="many_to_one")
    out["bench_return"] = out["bench_return"].fillna(0.0)
    out["bench_type"] = "kospi200"
    return out.sort_values(["phase", "date"]).reset_index(drop=True)


def compare_strategy_vs_benchmark(
    bt_returns: pd.DataFrame,
    bench_returns: pd.DataFrame,
    *,
    holding_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, list[str]]:
    warns: list[str] = []

    br = bt_returns.copy()
    br["date"] = pd.to_datetime(br["date"])
    br["phase"] = br["phase"].astype(str)

    bench = bench_returns.copy()
    bench["date"] = pd.to_datetime(bench["date"])
    bench["phase"] = bench["phase"].astype(str)

    strat_ret, used_col = _pick_strategy_return_series(br)
    br["_strategy_return_"] = strat_ret

    m = br.merge(
        bench[["phase", "date", "bench_return"]], on=["phase", "date"], how="inner"
    )
    if len(m) == 0:
        raise ValueError("no overlapping dates between bt_returns and benchmark")

    m["excess_return"] = m["_strategy_return_"].astype(float) - m[
        "bench_return"
    ].astype(float)
    periods_per_year = 252.0 / float(holding_days) if holding_days > 0 else 12.6

    rows = []
    for phase, g in m.groupby("phase", sort=False):
        ex = g["excess_return"].astype(float).to_numpy()
        te = (
            float(np.std(ex, ddof=1) * np.sqrt(periods_per_year))
            if len(ex) > 1
            else 0.0
        )
        ir = (
            float(
                (np.mean(ex) / (np.std(ex, ddof=1) + 1e-12)) * np.sqrt(periods_per_year)
            )
            if len(ex) > 1
            else 0.0
        )

        strat = g["_strategy_return_"].astype(float).to_numpy()
        b = g["bench_return"].astype(float).to_numpy()

        corr = float(np.corrcoef(strat, b)[0, 1]) if len(ex) > 1 else np.nan
        beta = (
            float(np.cov(strat, b, ddof=1)[0, 1] / (np.var(b, ddof=1) + 1e-12))
            if len(ex) > 1
            else np.nan
        )

        rows.append(
            {
                "phase": phase,
                "n_rebalances": int(len(g)),
                "tracking_error_ann": te,
                "information_ratio": ir,
                "corr_vs_benchmark": corr,
                "beta_vs_benchmark": beta,
                "date_start": g["date"].min(),
                "date_end": g["date"].max(),
                "strategy_return_col_used": used_col,
            }
        )

    compare_metrics = pd.DataFrame(rows)
    quality = {
        "benchmark": {
            "holding_days": int(holding_days),
            "rows_overlap": int(len(m)),
            "strategy_return_col_used": used_col,
        }
    }
    return (
        m.sort_values(["phase", "date"]).reset_index(drop=True),
        compare_metrics,
        quality,
        warns,
    )


def compare_strategy_vs_benchmarks(
    bt_returns: pd.DataFrame,
    bench_multi: pd.DataFrame,
    *,
    holding_days: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, list[str]]:
    """
    [개선안 17번] 멀티 벤치마크 비교(롱 포맷)
    bench_multi schema: phase,date,bench_return,bench_type
    """
    warns: list[str] = []
    req = {"phase", "date", "bench_return", "bench_type"}
    if not req.issubset(set(bench_multi.columns)):
        raise KeyError(
            f"bench_multi missing cols: {sorted(list(req - set(bench_multi.columns)))}"
        )

    rows = []
    metrics_rows = []
    quality = {"benchmark_multi": {"holding_days": int(holding_days)}}

    for btype, bdf in bench_multi.groupby("bench_type", sort=False):
        m, met, q, w = compare_strategy_vs_benchmark(
            bt_returns=bt_returns,
            bench_returns=bdf.rename(columns={"bench_return": "bench_return"})[
                ["phase", "date", "bench_return"]
            ],
            holding_days=holding_days,
        )
        m["bench_type"] = str(btype)
        met["bench_type"] = str(btype)
        rows.append(m)
        metrics_rows.append(met)
        warns.extend(w)
        quality["benchmark_multi"][str(btype)] = q.get("benchmark", {})

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    metrics = (
        pd.concat(metrics_rows, ignore_index=True) if metrics_rows else pd.DataFrame()
    )
    return out, metrics, quality, warns


# [핵심 수정] 재귀 호출을 없애고 로직을 직접 실행
def run_l7c_benchmark(cfg, artifacts, *, force=False):
    warns: list[str] = []  # [개선안 17번] 멀티 벤치마크 과정 경고 수집
    # 1. 설정 로드
    p = cfg.get("params", {}) or {}
    l7 = p.get("l7", None)
    if not isinstance(l7, dict) or not l7:
        l7 = cfg.get("l7", {}) or {}

    holding_days = int(l7.get("holding_days", 20))

    # 2. 데이터 추출
    if "rebalance_scores" not in artifacts:
        raise KeyError("run_l7c_benchmark requires 'rebalance_scores' in artifacts")
    if "bt_returns" not in artifacts:
        raise KeyError("run_l7c_benchmark requires 'bt_returns' in artifacts")

    rebalance_scores = artifacts["rebalance_scores"]
    bt_returns = artifacts["bt_returns"]

    # [개선안 17번] 벤치마크 3종 표준화 (기본: universe_mean + kospi200 + savings)
    l7c = _resolve_section(cfg, "l7c")
    bench_types = l7c.get("benchmark_types", None)
    if not isinstance(bench_types, list) or len(bench_types) == 0:
        bench_types = ["universe_mean", "kospi200", "savings"]

    savings_apr = float(l7c.get("savings_apr", 0.03))
    index_code = str((cfg.get("params", {}) or {}).get("index_code", "1028"))

    bench_multi_parts = []
    bench_universe = None
    if "universe_mean" in [str(x).lower() for x in bench_types]:
        bench_universe = build_universe_benchmark_returns(rebalance_scores)
        b = bench_universe.rename(columns={"bench_return": "bench_return"}).copy()
        b["bench_type"] = "universe_mean"
        bench_multi_parts.append(b[["phase", "date", "bench_return", "bench_type"]])

    if "savings" in [str(x).lower() for x in bench_types]:
        bench_multi_parts.append(
            build_savings_benchmark_returns(
                bt_returns, savings_apr=savings_apr, holding_days=holding_days
            )
        )

    if "kospi200" in [str(x).lower() for x in bench_types]:
        try:
            bench_multi_parts.append(
                build_kospi200_benchmark_returns(
                    bt_returns, holding_days=holding_days, index_code=index_code
                )
            )
        except Exception as e:
            warns.append(f"[L7C] kospi200 benchmark skipped: {type(e).__name__}: {e}")

    bench_multi = (
        pd.concat(bench_multi_parts, ignore_index=True)
        if bench_multi_parts
        else pd.DataFrame(columns=["phase", "date", "bench_return", "bench_type"])
    )

    # 단일(기존 호환): universe_mean 우선
    if bench_universe is None:
        bench_universe = build_universe_benchmark_returns(rebalance_scores)
    bt_vs_bench, metrics, quality, warns2 = compare_strategy_vs_benchmark(
        bt_returns=bt_returns,
        bench_returns=bench_universe,
        holding_days=holding_days,
    )
    warns.extend(warns2)

    # 멀티(신규)
    (
        bt_vs_bench_multi,
        metrics_multi,
        quality_multi,
        warns3,
    ) = compare_strategy_vs_benchmarks(
        bt_returns=bt_returns,
        bench_multi=bench_multi,
        holding_days=holding_days,
    )
    warns.extend(warns3)

    # 4. 결과 반환
    outputs = {
        # 기존 키(하위호환)
        "bt_benchmark_returns": bench_universe,
        "bt_vs_benchmark": bt_vs_bench,
        "bt_benchmark_compare": metrics,
        # 신규 키(멀티)
        "bt_benchmark_returns_multi": bench_multi,
        "bt_vs_benchmark_multi": bt_vs_bench_multi,
        "bt_benchmark_compare_multi": metrics_multi,
    }

    # quality는 메타데이터 용도지만 여기서는 outputs 위주로 반환,
    # 필요하다면 artifacts["_l7c_quality"] = quality 같은 처리를 runner에서 수행
    return outputs, warns
