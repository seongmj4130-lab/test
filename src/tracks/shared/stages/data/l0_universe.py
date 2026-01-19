# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/stages/data/l0_universe.py
from __future__ import annotations

import pandas as pd


def _require_pykrx():
    try:
        from pykrx import stock
        return stock
    except Exception as e:
        raise ImportError("pykrx가 필요합니다. `pip install pykrx` 후 재실행하세요.") from e

def _to_yyyymmdd(s: str) -> str:
    return pd.to_datetime(s).strftime("%Y%m%d")

def build_k200_membership_month_end(
    *,
    start_date: str,
    end_date: str,
    index_code: str = "1028",     # KOSPI200
    anchor_ticker: str = "005930",
    strict: bool = True,
) -> pd.DataFrame:
    """
    출력: date(월말 거래일, datetime64), ym(YYYY-MM), ticker(6자리)
    - 월말 거래일 기준으로 KOSPI200 구성종목 스냅샷 생성
    - strict=True: 월 누락/구성종목 수 이상치 즉시 실패
    """
    stock = _require_pykrx()

    s = _to_yyyymmdd(start_date)
    e = _to_yyyymmdd(end_date)

    # trading calendar (anchor_ticker 기준)
    cal = stock.get_market_ohlcv_by_date(s, e, anchor_ticker)
    if cal is None or len(cal) == 0:
        raise RuntimeError("거래일 캘린더 생성 실패: start/end/anchor_ticker 확인 필요")

    dates = pd.to_datetime(cal.index)
    month_end_dates = (
        pd.Series(dates)
        .groupby(pd.Series(dates).dt.to_period("M"))
        .max()
        .sort_values()
        .tolist()
    )
    expected_ym = [d.to_period("M").strftime("%Y-%m") for d in month_end_dates]

    records: list[dict] = []
    for d in month_end_dates:
        ymd = d.strftime("%Y%m%d")

        tickers = None
        # pykrx 버전차 대응: named param / positional 모두 시도
        try:
            tickers = stock.get_index_portfolio_deposit_file(index_code, date=ymd)
        except TypeError:
            tickers = stock.get_index_portfolio_deposit_file(index_code, ymd)

        if tickers is None or len(tickers) == 0:
            raise RuntimeError(f"KOSPI200 구성종목 조회 실패: index={index_code}, date={ymd}")

        for t in tickers:
            records.append(
                {
                    "date": pd.to_datetime(d),
                    "ym": d.to_period("M").strftime("%Y-%m"),
                    "ticker": str(t).zfill(6),
                }
            )

    df = pd.DataFrame(records).drop_duplicates(["date", "ticker"]).sort_values(["date", "ticker"]).reset_index(drop=True)

    # ---------- QC ----------
    counts = df.groupby("ym")["ticker"].nunique().sort_index()
    missing = sorted(set(expected_ym) - set(counts.index.tolist()))
    # KOSPI200은 통상 200개. (일시적 이슈 대비 완충)
    bad = counts[(counts < 180) | (counts > 220)]

    if strict:
        if missing:
            raise ValueError(f"[L0 universe] missing months in membership table: {missing[:10]} (total={len(missing)})")
        if len(bad) > 0:
            raise ValueError(f"[L0 universe] abnormal #tickers for months:\n{bad.to_string()}")

    return df

# END OF FILE: l0_universe.py
################################################################################
