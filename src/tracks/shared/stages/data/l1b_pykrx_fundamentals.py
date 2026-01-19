"""
L1B: pykrx 재무데이터 다운로드

PER, PBR, EPS, BPS, DIV, 시가총액을 pykrx에서 다운로드합니다.
"""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _require_pykrx():
    try:
        from pykrx import stock

        return stock
    except Exception as e:
        raise ImportError(
            "pykrx가 필요합니다. `pip install pykrx` 후 재실행하세요."
        ) from e


def _to_yyyymmdd(s: str) -> str:
    """날짜를 YYYYMMDD 형식으로 변환"""
    return pd.to_datetime(s).strftime("%Y%m%d")


def download_pykrx_fundamentals_daily(
    *,
    tickers: list[str],
    start_date: str,
    end_date: str,
    sleep_sec: float = 0.1,
    log_every: int = 50,
) -> pd.DataFrame:
    """
    pykrx로 일별 펀더멘털 데이터 다운로드

    출력:
      - date, ticker, PER, PBR, EPS, BPS, DIV, market_cap

    Args:
        tickers: 종목 코드 리스트 (6자리)
        start_date: 시작일 (YYYY-MM-DD)
        end_date: 종료일 (YYYY-MM-DD)
        sleep_sec: API 호출 간격 (초)
        log_every: 로그 출력 빈도

    Returns:
        DataFrame with columns: date, ticker, PER, PBR, EPS, BPS, DIV, market_cap
    """
    stock = _require_pykrx()

    s = _to_yyyymmdd(start_date)
    e = _to_yyyymmdd(end_date)

    records: list[dict[str, Any]] = []
    total_tickers = len(tickers)
    success_count = 0
    error_count = 0

    logger.info(
        f"[L1B] pykrx 재무데이터 다운로드 시작: {total_tickers}개 종목, {start_date} ~ {end_date}"
    )

    for idx, ticker in enumerate(sorted(set(tickers)), 1):
        try:
            # 펀더멘털 데이터 (PER, PBR, EPS, BPS, DIV)
            df_fund = stock.get_market_fundamental_by_date(s, e, ticker)

            if df_fund is None or df_fund.empty:
                error_count += 1
                if idx % log_every == 0:
                    logger.info(
                        f"[L1B] 진행: {idx}/{total_tickers}, 성공: {success_count}, 실패: {error_count}"
                    )
                time.sleep(sleep_sec)
                continue

            # 시가총액 데이터
            df_cap = stock.get_market_cap_by_date(s, e, ticker)

            # 데이터 정규화
            df_fund = df_fund.reset_index()
            df_fund = df_fund.rename(
                columns={
                    "날짜": "date",
                    "PER": "PER",
                    "PBR": "PBR",
                    "EPS": "EPS",
                    "BPS": "BPS",
                    "DIV": "DIV",
                }
            )

            # date 컬럼이 없으면 첫 번째 컬럼을 date로 사용
            if "date" not in df_fund.columns:
                if len(df_fund.columns) > 0:
                    df_fund = df_fund.rename(columns={df_fund.columns[0]: "date"})

            df_fund["date"] = pd.to_datetime(df_fund["date"], errors="coerce")
            df_fund["ticker"] = str(ticker).zfill(6)

            # 시가총액 병합
            if df_cap is not None and not df_cap.empty:
                df_cap = df_cap.reset_index()
                df_cap = df_cap.rename(
                    columns={
                        "날짜": "date",
                        "시가총액": "market_cap",
                    }
                )

                if "date" not in df_cap.columns:
                    if len(df_cap.columns) > 0:
                        df_cap = df_cap.rename(columns={df_cap.columns[0]: "date"})

                df_cap["date"] = pd.to_datetime(df_cap["date"], errors="coerce")
                df_cap["ticker"] = str(ticker).zfill(6)

                # 병합
                df_fund = df_fund.merge(
                    df_cap[["date", "ticker", "market_cap"]],
                    on=["date", "ticker"],
                    how="left",
                )
            else:
                df_fund["market_cap"] = np.nan

            # 필요한 컬럼만 선택
            keep_cols = [
                "date",
                "ticker",
                "PER",
                "PBR",
                "EPS",
                "BPS",
                "DIV",
                "market_cap",
            ]
            available_cols = [c for c in keep_cols if c in df_fund.columns]
            df_fund = df_fund[available_cols].copy()

            # NaN을 None으로 변환 (JSON 직렬화 대비)
            for col in df_fund.columns:
                if col not in ["date", "ticker"]:
                    df_fund[col] = df_fund[col].replace([np.inf, -np.inf], np.nan)

            records.append(df_fund)
            success_count += 1

            if idx % log_every == 0:
                logger.info(
                    f"[L1B] 진행: {idx}/{total_tickers}, 성공: {success_count}, 실패: {error_count}"
                )

            time.sleep(sleep_sec)

        except Exception as e:
            error_count += 1
            logger.warning(f"[L1B] 종목 {ticker} 다운로드 실패: {e}")
            if idx % log_every == 0:
                logger.info(
                    f"[L1B] 진행: {idx}/{total_tickers}, 성공: {success_count}, 실패: {error_count}"
                )
            time.sleep(sleep_sec)
            continue

    if not records:
        logger.warning("[L1B] 다운로드된 데이터가 없습니다.")
        # 빈 DataFrame 반환 (스키마는 유지)
        return pd.DataFrame(
            columns=["date", "ticker", "PER", "PBR", "EPS", "BPS", "DIV", "market_cap"]
        )

    out = pd.concat(records, ignore_index=True)
    out = out.sort_values(["date", "ticker"]).reset_index(drop=True)

    # 날짜 형식 통일
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")

    logger.info(
        f"[L1B] 다운로드 완료: {len(out):,}행, 성공: {success_count}/{total_tickers}, 실패: {error_count}"
    )

    return out
