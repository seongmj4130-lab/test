# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/stages/export/market_regime.py
"""
[Stage10] 시장 국면(Regime) 지표 계산

기능:
- pykrx로 KOSPI200 지수(1028) 시계열 다운로드
- 또는 universe 종목 동일가중 수익률로 시장 proxy 생성
- regime_score (0~100), regime_label (BULL/BEAR/NEUTRAL) 계산
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# pykrx import를 위한 경로 추가
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

def _require_pykrx():
    """pykrx 모듈 import"""
    try:
        from pykrx import stock
        return stock
    except ImportError as e:
        raise ImportError("pykrx가 필요합니다. `pip install pykrx` 후 재실행하세요.") from e

def _to_yyyymmdd(s: str) -> str:
    """날짜를 YYYYMMDD 형식으로 변환"""
    return pd.to_datetime(s).strftime("%Y%m%d")

def download_kospi200_index(
    start_date: str,
    end_date: str,
    cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    pykrx로 KOSPI200 지수(1028) 시계열 다운로드

    Args:
        start_date: 시작일 (YYYY-MM-DD)
        end_date: 종료일 (YYYY-MM-DD)
        cache_dir: 캐시 디렉토리 (None이면 캐시 안 함)

    Returns:
        DataFrame with columns: date, close, volume (optional)
    """
    stock = _require_pykrx()

    s = _to_yyyymmdd(start_date)
    e = _to_yyyymmdd(end_date)

    # 캐시 확인
    if cache_dir:
        cache_file = cache_dir / f"kospi200_index_{s}_{e}.parquet"
        if cache_file.exists():
            try:
                cached_df = pd.read_parquet(cache_file)
                print(f"[Market Regime] 캐시에서 로드: {cache_file}")
                return cached_df
            except Exception as e:
                print(f"[Market Regime] 캐시 로드 실패, 재다운로드: {e}")

    # pykrx로 지수 데이터 다운로드
    try:
        # KOSPI200 지수 코드: 1028
        index_code = "1028"

        # pykrx의 get_index_ohlcv_by_date 사용
        df = stock.get_index_ohlcv_by_date(s, e, index_code)

        if df is None or len(df) == 0:
            raise RuntimeError(f"KOSPI200 지수 데이터 다운로드 실패: {s} ~ {e}")

        # 컬럼명 정규화 (한글 -> 영어)
        rename_map = {
            "날짜": "date",
            "시가": "open",
            "고가": "high",
            "저가": "low",
            "종가": "close",
            "거래량": "volume",
            "거래대금": "value",
        }

        df = df.reset_index()
        if "날짜" in df.columns:
            df = df.rename(columns={"날짜": "date"})
        elif df.index.name == "날짜":
            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: "date"})

        # 컬럼명 영어로 변환
        for k, v in rename_map.items():
            if k in df.columns:
                df = df.rename(columns={k: v})

        # date 컬럼 정규화
        if "date" not in df.columns:
            df = df.rename(columns={df.columns[0]: "date"})

        df["date"] = pd.to_datetime(df["date"])

        # 필수 컬럼만 선택
        output_cols = ["date", "close"]
        if "volume" in df.columns:
            output_cols.append("volume")

        result = df[output_cols].copy()
        result = result.sort_values("date").reset_index(drop=True)

        # 캐시 저장
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            result.to_parquet(cache_file, index=False)
            print(f"[Market Regime] 캐시 저장: {cache_file}")

        return result

    except Exception as e:
        raise RuntimeError(f"KOSPI200 지수 다운로드 실패: {e}") from e

def build_market_proxy_from_universe(
    ohlcv_daily: pd.DataFrame,
    universe_daily: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    universe 종목의 동일가중 수익률로 시장 proxy 생성

    Args:
        ohlcv_daily: OHLCV 데이터 (date, ticker, close 포함)
        universe_daily: 유니버스 멤버십 데이터 (date, ticker, in_universe 포함)
        date_col: 날짜 컬럼명

    Returns:
        DataFrame with columns: date, close (시장 proxy 수익률 누적)
    """
    # 유니버스 멤버십 필터링
    if "in_universe" in universe_daily.columns:
        universe_tickers = universe_daily[universe_daily["in_universe"]]["ticker"].unique()
    else:
        universe_tickers = universe_daily["ticker"].unique()

    # 유니버스 종목의 OHLCV만 선택
    universe_ohlcv = ohlcv_daily[ohlcv_daily["ticker"].isin(universe_tickers)].copy()

    if len(universe_ohlcv) == 0:
        raise ValueError("유니버스 종목의 OHLCV 데이터가 없습니다.")

    # 날짜별 수익률 계산
    universe_ohlcv = universe_ohlcv.sort_values([date_col, "ticker"])
    universe_ohlcv["ret"] = universe_ohlcv.groupby("ticker")["close"].pct_change()

    # 날짜별 동일가중 평균 수익률
    daily_ret = universe_ohlcv.groupby(date_col)["ret"].mean()

    # 누적 수익률 (시작점 = 100)
    cumulative_ret = (1 + daily_ret).cumprod() * 100

    # 결과 DataFrame 생성
    result = pd.DataFrame({
        "date": pd.to_datetime(daily_ret.index),
        "close": cumulative_ret.values,
    })

    return result.sort_values("date").reset_index(drop=True)

def calculate_regime_score(
    index_data: pd.DataFrame,
    lookback_days: int = 60,
    ma_short: int = 20,
    ma_long: int = 60,
) -> pd.DataFrame:
    """
    시장 국면 점수 계산

    Args:
        index_data: 지수 데이터 (date, close 포함)
        lookback_days: 국면 판단을 위한 lookback 기간
        ma_short: 단기 이동평균 기간
        ma_long: 장기 이동평균 기간

    Returns:
        DataFrame with columns: date, regime_score, regime_label
    """
    df = index_data.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # 수익률 계산
    df["ret"] = df["close"].pct_change()

    # 이동평균 계산
    df["ma_short"] = df["close"].rolling(window=ma_short, min_periods=1).mean()
    df["ma_long"] = df["close"].rolling(window=ma_long, min_periods=1).mean()

    # 모멘텀 지표 (최근 lookback_days 누적 수익률)
    df["momentum"] = df["ret"].rolling(window=lookback_days, min_periods=1).sum()

    # 변동성 지표 (최근 lookback_days 변동성)
    df["volatility"] = df["ret"].rolling(window=lookback_days, min_periods=1).std()

    # 국면 점수 계산 (0~100)
    # 1. 이동평균 관계 (40점)
    # 골든크로스: +40점, 데드크로스: 0점
    ma_score = np.where(
        df["ma_short"] > df["ma_long"],
        40.0,
        0.0,
    )

    # 2. 모멘텀 (30점)
    # Sharpe-like ratio: momentum / volatility
    # 정규화하여 0~30점으로 변환
    sharpe_like = df["momentum"] / df["volatility"].replace(0, np.nan)
    sharpe_like = np.nan_to_num(sharpe_like, nan=0.0)

    # -2 ~ +2 범위를 0~30으로 매핑
    momentum_score = np.clip((sharpe_like + 2) / 4 * 30, 0, 30)

    # 3. 가격 위치 (30점) - 최근 lookback_days 고점 대비 현재 위치
    rolling_max = df["close"].rolling(window=lookback_days, min_periods=1).max()
    rolling_min = df["close"].rolling(window=lookback_days, min_periods=1).min()

    # 고점 대비 위치 (0~1)
    price_range = rolling_max - rolling_min
    price_position_ratio = (df["close"] - rolling_min) / price_range.replace(0, np.nan)
    price_position_ratio = np.nan_to_num(price_position_ratio, nan=0.5)  # NaN이면 중간값

    # 0~30점으로 변환
    price_position = price_position_ratio * 30

    # 총점 (0~100)
    regime_score = ma_score + momentum_score + price_position
    regime_score = np.clip(regime_score, 0, 100)

    # 레이블 분류
    regime_label = pd.Series("NEUTRAL", index=df.index)
    regime_label[regime_score >= 60] = "BULL"
    regime_label[regime_score <= 40] = "BEAR"

    result = pd.DataFrame({
        "date": df["date"],
        "regime_score": regime_score,
        "regime_label": regime_label,
    })

    return result

def build_market_regime_daily(
    start_date: str,
    end_date: str,
    ohlcv_daily: Optional[pd.DataFrame] = None,
    universe_daily: Optional[pd.DataFrame] = None,
    use_pykrx: bool = True,
    cache_dir: Optional[Path] = None,
    lookback_days: int = 60,
) -> pd.DataFrame:
    """
    시장 국면 일자별 데이터 생성

    Args:
        start_date: 시작일
        end_date: 종료일
        ohlcv_daily: OHLCV 데이터 (pykrx 실패 시 사용)
        universe_daily: 유니버스 멤버십 데이터 (pykrx 실패 시 사용)
        use_pykrx: pykrx 사용 여부 (우선순위 A)
        cache_dir: 캐시 디렉토리
        lookback_days: 국면 판단 lookback 기간

    Returns:
        DataFrame with columns: date, regime_score, regime_label
    """
    index_data = None

    # 우선순위 A: pykrx로 KOSPI200 지수 다운로드
    if use_pykrx:
        try:
            index_data = download_kospi200_index(start_date, end_date, cache_dir)
            print(f"[Market Regime] pykrx로 KOSPI200 지수 다운로드 성공: {len(index_data)} rows")
        except Exception as e:
            print(f"[Market Regime] pykrx 다운로드 실패: {e}")
            print(f"[Market Regime] 우선순위 B로 전환: universe 동일가중 수익률 사용")
            use_pykrx = False

    # 우선순위 B: universe 동일가중 수익률로 시장 proxy 생성
    if not use_pykrx or index_data is None:
        if ohlcv_daily is None or universe_daily is None:
            raise ValueError(
                "pykrx 다운로드 실패 시 ohlcv_daily와 universe_daily가 필요합니다."
            )

        index_data = build_market_proxy_from_universe(ohlcv_daily, universe_daily)
        print(f"[Market Regime] universe 동일가중 수익률로 시장 proxy 생성: {len(index_data)} rows")

    # 국면 점수 계산
    regime_df = calculate_regime_score(index_data, lookback_days=lookback_days)

    return regime_df
