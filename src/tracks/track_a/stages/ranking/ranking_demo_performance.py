# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/stages/ranking/ranking_demo_performance.py
"""
[Stage11] Ranking Demo Performance
Top20 equal-weight 포트폴리오 성과 곡선 계산

입력:
- ranking_daily.parquet (date, ticker, rank_total, ...)
- ohlcv_daily.parquet (date, ticker, close, ...) - 수익률 계산용
- (있으면) KOSPI200 지수 시계열

출력:
- ui_equity_curves.parquet (date, strategy_ret, bench_ret, strategy_equity, bench_equity, excess_equity)
- ui_metrics.csv (total_return, cagr, vol, sharpe, mdd)
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def _require_pykrx():
    """
    [개선안 17번] KOSPI200 지수 벤치마크 계산을 위해 pykrx 로딩
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


def build_savings_benchmark_returns(
    *,
    dates: pd.Series,
    savings_apr: float = 0.03,
    periods_per_year: float = 252.0,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    [개선안 17번] 적금(고정금리) 벤치마크 일별 수익률 생성

    Args:
        dates: 전략과 동일한 trading date 시계열
        savings_apr: 연 이율(예: 0.03 = 3%)
        periods_per_year: 연 환산(기본 252 trading days)

    Returns:
        DataFrame(date, bench_ret)
    """
    d = pd.to_datetime(pd.Series(dates).dropna().unique())
    d = pd.DatetimeIndex(d).sort_values()
    if len(d) == 0:
        return pd.DataFrame(columns=[date_col, "bench_ret"])

    apr = float(savings_apr)
    ppy = float(periods_per_year) if float(periods_per_year) > 0 else 252.0
    daily_ret = (1.0 + apr) ** (1.0 / ppy) - 1.0
    return pd.DataFrame({date_col: d, "bench_ret": float(daily_ret)})


def build_kospi200_benchmark_returns(
    *,
    dates: pd.Series,
    index_code: str = "1028",
    date_col: str = "date",
) -> pd.DataFrame:
    """
    [개선안 17번] KOSPI200 지수 벤치마크 일별 수익률 생성 (pykrx)

    Args:
        dates: 전략과 동일한 trading date 시계열
        index_code: pykrx index_code (기본 1028=KOSPI200)

    Returns:
        DataFrame(date, bench_ret)
    """
    d = pd.to_datetime(pd.Series(dates).dropna().unique())
    d = pd.DatetimeIndex(d).sort_values()
    if len(d) == 0:
        return pd.DataFrame(columns=[date_col, "bench_ret"])

    stock = _require_pykrx()
    s = _to_yyyymmdd(d.min())
    e = _to_yyyymmdd(d.max())

    # pykrx 지수 OHLCV
    idx = stock.get_index_ohlcv_by_date(s, e, str(index_code))
    if idx is None or len(idx) == 0:
        raise RuntimeError(
            f"KOSPI200 지수 데이터가 비어있습니다 (index_code={index_code}, {s}~{e})"
        )

    idx = idx.reset_index()
    rename_map = {"날짜": "date", "종가": "close"}
    for old, new in rename_map.items():
        if old in idx.columns:
            idx = idx.rename(columns={old: new})
    if "date" not in idx.columns:
        idx = idx.rename(columns={idx.columns[0]: "date"})
    if "close" not in idx.columns:
        # 일부 환경에서 한글/영문 혼재
        close_candidates = ["종가", "close", "Close", "CLOSE"]
        close_col = next((c for c in close_candidates if c in idx.columns), None)
        if close_col is None:
            raise RuntimeError(
                f"지수 데이터에 close(종가) 컬럼이 없습니다. cols={list(idx.columns)}"
            )
        idx = idx.rename(columns={close_col: "close"})

    idx["date"] = pd.to_datetime(idx["date"], errors="coerce")
    idx = (
        idx.dropna(subset=["date", "close"]).sort_values("date").reset_index(drop=True)
    )
    idx["bench_ret"] = pd.to_numeric(idx["close"], errors="coerce").pct_change()
    idx = idx.dropna(subset=["bench_ret"])

    out = idx[["date", "bench_ret"]].copy()
    # 전략 날짜로 필터 (inner join이 누락을 만들 수 있어 left 후 결측은 0 처리)
    out = pd.DataFrame({date_col: d}).merge(
        out, left_on=date_col, right_on="date", how="left"
    )
    out = out.drop(columns=["date"])
    out["bench_ret"] = out["bench_ret"].fillna(0.0)
    return out[[date_col, "bench_ret"]].copy()


def calculate_returns(
    ohlcv_daily: pd.DataFrame,
    date_col: str = "date",
    ticker_col: str = "ticker",
    price_col: str = "close",
) -> pd.DataFrame:
    """
    일별 수익률 계산

    Args:
        ohlcv_daily: OHLCV 데이터
        date_col: 날짜 컬럼명
        ticker_col: 티커 컬럼명
        price_col: 가격 컬럼명 (기본 "close")

    Returns:
        returns DataFrame (date, ticker, return)
    """
    df = ohlcv_daily[[date_col, ticker_col, price_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([ticker_col, date_col])

    # 티커별 수익률 계산
    df["return"] = df.groupby(ticker_col)[price_col].pct_change()

    # 첫날 수익률 제거 (NaN)
    df = df[df["return"].notna()].copy()

    return df[[date_col, ticker_col, "return"]]


def build_top20_equal_weight_returns(
    ranking_daily: pd.DataFrame,
    returns: pd.DataFrame,
    top_k: int = 20,
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    """
    Top20 equal-weight 포트폴리오 일별 수익률 계산

    Args:
        ranking_daily: ranking_daily DataFrame
        returns: 수익률 DataFrame (date, ticker, return)
        top_k: 상위 종목 수 (기본 20)
        date_col: 날짜 컬럼명
        ticker_col: 티커 컬럼명

    Returns:
        strategy_returns DataFrame (date, strategy_ret)
    """
    # 날짜 정규화
    ranking_daily = ranking_daily.copy()
    ranking_daily[date_col] = pd.to_datetime(ranking_daily[date_col])

    returns = returns.copy()
    returns[date_col] = pd.to_datetime(returns[date_col])

    # in_universe 필터링 (있으면)
    if "in_universe" in ranking_daily.columns:
        ranking_daily = ranking_daily[ranking_daily["in_universe"]].copy()

    # rank_total 기준 정렬
    ranking_daily = ranking_daily[ranking_daily["rank_total"].notna()].copy()
    ranking_daily = ranking_daily.sort_values(
        [date_col, "rank_total"], ascending=[True, True]
    )

    # 날짜별 Top K 선택
    strategy_returns = []

    for date, group in ranking_daily.groupby(date_col):
        # Top K 티커 선택
        top_tickers = group.head(top_k)[ticker_col].tolist()

        # 해당 날짜의 수익률 필터링
        date_returns = returns[returns[date_col] == date]
        top_returns = date_returns[date_returns[ticker_col].isin(top_tickers)]["return"]

        if len(top_returns) > 0:
            # Equal-weight 평균 수익률
            strategy_ret = float(top_returns.mean())
        else:
            strategy_ret = 0.0

        strategy_returns.append(
            {
                date_col: date,
                "strategy_ret": strategy_ret,
                "n_tickers": len(top_tickers),
            }
        )

    df = pd.DataFrame(strategy_returns)
    df = df.sort_values(date_col).reset_index(drop=True)

    return df


def build_benchmark_returns(
    ohlcv_daily: pd.DataFrame,
    benchmark_type: str = "universe_mean",
    *,
    cfg: Optional[dict] = None,
    date_col: str = "date",
    ticker_col: str = "ticker",
    price_col: str = "close",
) -> pd.DataFrame:
    """
    벤치마크 수익률 계산

    Args:
        ohlcv_daily: OHLCV 데이터
        benchmark_type: "universe_mean" (유니버스 평균) 또는 "kospi200" (KOSPI200 지수)
        date_col: 날짜 컬럼명
        ticker_col: 티커 컬럼명
        price_col: 가격 컬럼명

    Returns:
        benchmark_returns DataFrame (date, bench_ret)
    """
    btype = str(benchmark_type).strip().lower()

    if btype == "universe_mean":
        # 유니버스 평균 수익률
        returns = calculate_returns(ohlcv_daily, date_col, ticker_col, price_col)

        # 날짜별 평균 수익률
        bench_returns = returns.groupby(date_col)["return"].mean().reset_index()
        bench_returns.columns = [date_col, "bench_ret"]

        return bench_returns

    elif btype == "kospi200":
        # [개선안 17번] 실제 KOSPI200 지수 수익률 사용
        dates = pd.to_datetime(ohlcv_daily[date_col], errors="coerce")
        index_code = "1028"
        if isinstance(cfg, dict):
            index_code = str(
                (cfg.get("params", {}) or {}).get("index_code", index_code)
            )
        return build_kospi200_benchmark_returns(
            dates=dates, index_code=index_code, date_col=date_col
        )

    elif btype == "savings":
        # [개선안 17번] 적금(고정금리) 벤치마크
        dates = pd.to_datetime(ohlcv_daily[date_col], errors="coerce")
        savings_apr = 0.03
        if isinstance(cfg, dict):
            l11 = (cfg.get("l11", {}) if isinstance(cfg, dict) else {}) or {}
            savings_apr = float(l11.get("savings_apr", savings_apr))
        return build_savings_benchmark_returns(
            dates=dates, savings_apr=savings_apr, date_col=date_col
        )

    else:
        raise ValueError(
            f"Unknown benchmark_type: {benchmark_type}. expected: universe_mean|kospi200|savings"
        )


def build_equity_curves_multi(
    *,
    strategy_returns: pd.DataFrame,
    benchmark_returns_by_type: dict[str, pd.DataFrame],
    date_col: str = "date",
    initial_value: float = 100.0,
) -> pd.DataFrame:
    """
    [개선안 17번] 멀티 벤치마크용 누적 곡선(롱 포맷)
    """
    rows = []
    for btype, bench_df in benchmark_returns_by_type.items():
        eq = build_equity_curves(
            strategy_returns, bench_df, date_col=date_col, initial_value=initial_value
        )
        eq["benchmark_type"] = str(btype)
        rows.append(eq)
    if not rows:
        return pd.DataFrame(
            columns=[
                date_col,
                "strategy_ret",
                "bench_ret",
                "strategy_equity",
                "bench_equity",
                "excess_equity",
                "benchmark_type",
            ]
        )
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["benchmark_type", date_col]).reset_index(drop=True)


def calculate_performance_metrics_multi(
    equity_curves_multi: pd.DataFrame,
    *,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    [개선안 17번] 멀티 벤치마크별 성과 지표 계산
    """
    if equity_curves_multi is None or len(equity_curves_multi) == 0:
        return pd.DataFrame(
            columns=["benchmark_type", "total_return", "cagr", "vol", "sharpe", "mdd"]
        )
    if "benchmark_type" not in equity_curves_multi.columns:
        m = calculate_performance_metrics(equity_curves_multi, date_col=date_col)
        return pd.DataFrame([{"benchmark_type": "single", **m}])
    rows = []
    for btype, g in equity_curves_multi.groupby("benchmark_type", sort=False):
        m = calculate_performance_metrics(g, date_col=date_col)
        rows.append({"benchmark_type": str(btype), **m})
    return pd.DataFrame(rows)


def build_equity_curves(
    strategy_returns: pd.DataFrame,
    benchmark_returns: pd.DataFrame,
    date_col: str = "date",
    initial_value: float = 100.0,
) -> pd.DataFrame:
    """
    누적 곡선 계산

    Args:
        strategy_returns: 전략 수익률 DataFrame (date, strategy_ret)
        benchmark_returns: 벤치마크 수익률 DataFrame (date, bench_ret)
        date_col: 날짜 컬럼명
        initial_value: 초기값 (기본 100.0)

    Returns:
        equity_curves DataFrame (date, strategy_ret, bench_ret, strategy_equity, bench_equity, excess_equity)
    """
    # 병합
    df = pd.merge(
        strategy_returns,
        benchmark_returns,
        on=date_col,
        how="inner",
    )

    df = df.sort_values(date_col).reset_index(drop=True)

    # 누적 곡선 계산
    df["strategy_equity"] = (1 + df["strategy_ret"]).cumprod() * initial_value
    df["bench_equity"] = (1 + df["bench_ret"]).cumprod() * initial_value
    df["excess_equity"] = df["strategy_equity"] - df["bench_equity"]

    return df


def calculate_performance_metrics(
    equity_curves: pd.DataFrame,
    date_col: str = "date",
    strategy_equity_col: str = "strategy_equity",
    bench_equity_col: str = "bench_equity",
    strategy_ret_col: str = "strategy_ret",
    bench_ret_col: str = "bench_ret",
) -> dict[str, float]:
    """
    성과 지표 계산

    Args:
        equity_curves: 누적 곡선 DataFrame
        date_col: 날짜 컬럼명
        strategy_equity_col: 전략 누적값 컬럼명
        bench_equity_col: 벤치마크 누적값 컬럼명
        strategy_ret_col: 전략 수익률 컬럼명
        bench_ret_col: 벤치마크 수익률 컬럼명

    Returns:
        metrics 딕셔너리
    """
    if len(equity_curves) == 0:
        return {
            "total_return": 0.0,
            "cagr": 0.0,
            "vol": 0.0,
            "sharpe": 0.0,
            "mdd": 0.0,
        }

    df = equity_curves.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # 총 수익률
    initial_value = df[strategy_equity_col].iloc[0]
    final_value = df[strategy_equity_col].iloc[-1]
    total_return = (final_value / initial_value - 1.0) * 100.0  # %

    # 기간 계산 (연 단위)
    date_start = df[date_col].min()
    date_end = df[date_col].max()
    years = (date_end - date_start).days / 365.25

    # CAGR
    if years > 0:
        cagr = ((final_value / initial_value) ** (1.0 / years) - 1.0) * 100.0  # %
    else:
        cagr = 0.0

    # 변동성 (연율화)
    vol = df[strategy_ret_col].std() * np.sqrt(252) * 100.0  # %

    # Sharpe Ratio (무위험 수익률 = 0 가정)
    if vol > 0:
        sharpe = (cagr / 100.0) / (vol / 100.0)
    else:
        sharpe = 0.0

    # MDD (Maximum Drawdown)
    running_max = df[strategy_equity_col].cummax()
    drawdown = (df[strategy_equity_col] - running_max) / running_max * 100.0  # %
    mdd = float(drawdown.min())

    return {
        "total_return": total_return,
        "cagr": cagr,
        "vol": vol,
        "sharpe": sharpe,
        "mdd": mdd,
    }


def run_L11_demo_performance(
    cfg: dict,
    artifacts: dict,
    *,
    force: bool = False,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """
    [Stage11] Ranking Demo Performance 실행

    Args:
        cfg: 설정 딕셔너리
        artifacts: 이전 스테이지 산출물 딕셔너리
        force: 강제 재생성 플래그

    Returns:
        (outputs, warnings) 튜플
        - outputs: {"ui_equity_curves": DataFrame, "ui_metrics": DataFrame}
        - warnings: 경고 메시지 리스트
    """
    warns: list[str] = []

    # 입력 데이터 확인
    ranking_daily = artifacts.get("ranking_daily")
    ohlcv_daily = artifacts.get("ohlcv_daily")

    if ranking_daily is None:
        raise ValueError("L11 requires 'ranking_daily' in artifacts")

    if ohlcv_daily is None:
        raise ValueError("L11 requires 'ohlcv_daily' in artifacts")

    # 설정 읽기
    l11 = cfg.get("l11", {}) or {}
    top_k = int(l11.get("top_k", 20))
    benchmark_type = l11.get("benchmark_type", "universe_mean")

    # 수익률 계산
    returns = calculate_returns(ohlcv_daily)

    # Top20 equal-weight 수익률 계산
    strategy_returns = build_top20_equal_weight_returns(
        ranking_daily,
        returns,
        top_k=top_k,
    )

    # 벤치마크 수익률 계산
    benchmark_returns = build_benchmark_returns(
        ohlcv_daily,
        benchmark_type=benchmark_type,
    )

    # 누적 곡선 계산
    equity_curves = build_equity_curves(
        strategy_returns,
        benchmark_returns,
    )

    # 성과 지표 계산
    metrics = calculate_performance_metrics(equity_curves)
    metrics_df = pd.DataFrame([metrics])

    outputs = {
        "ui_equity_curves": equity_curves,
        "ui_metrics": metrics_df,
    }

    return outputs, warns
