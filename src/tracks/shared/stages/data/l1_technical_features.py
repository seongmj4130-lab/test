# -*- coding: utf-8 -*-
# src/tracks/shared/stages/data/l1_technical_features.py
# [FEATURESET_COMPLETE] OHLCV 기반 기술적 지표 계산 함수
# 문서: docs/FEATURESET_COMPLETE.md

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_technical_features(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    close_col: str = "close",
    volume_col: str = "volume",
    high_col: str = "high",
    low_col: str = "low",
) -> pd.DataFrame:
    """
    OHLCV 데이터로부터 기술적 지표 계산
    
    Args:
        df: OHLCV 데이터프레임 (date, ticker, open, high, low, close, volume 포함)
        date_col: 날짜 컬럼명
        ticker_col: 종목코드 컬럼명
        close_col: 종가 컬럼명
        volume_col: 거래량 컬럼명
        high_col: 고가 컬럼명
        low_col: 저가 컬럼명
    
    Returns:
        원본 데이터프레임에 기술적 지표 컬럼이 추가된 데이터프레임
    
    계산되는 피처:
    - price_momentum_20d: 20일 모멘텀
    - price_momentum_60d: 60일 모멘텀
    - momentum_3m: 3개월(90일) 모멘텀 [Phase 5 전략1]
    - momentum_6m: 6개월(180일) 모멘텀 [Phase 5 전략1]
    - price_momentum: 일반 모멘텀 (20일 기준)
    - volatility_20d: 20일 변동성 (연율화)
    - volatility_60d: 60일 변동성 (연율화)
    - volatility: 일반 변동성 (20일 기준)
    - max_drawdown_60d: 60일 최대 낙폭
    - downside_volatility_60d: 60일 하방 변동성
    - volume_ratio: 거래량 비율 (이동평균 대비)
    - momentum_reversal: 모멘텀 반전 지표
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([ticker_col, date_col]).reset_index(drop=True)
    
    # 종목별 그룹화
    grouped = df.groupby(ticker_col, sort=False)
    
    # 1. 일일 수익률 계산
    df["daily_return"] = grouped[close_col].pct_change()
    
    # 2. price_momentum_20d: 20일 모멘텀
    # (close[t] - close[t-20]) / close[t-20]
    df["price_momentum_20d"] = grouped[close_col].pct_change(periods=20)
    
    # 3. price_momentum_60d: 60일 모멘텀
    df["price_momentum_60d"] = grouped[close_col].pct_change(periods=60)
    
    # [Phase 5 전략1] momentum_3m: 3개월(90일) 모멘텀
    df["momentum_3m"] = grouped[close_col].pct_change(periods=90)
    
    # [Phase 5 전략1] momentum_6m: 6개월(180일) 모멘텀
    df["momentum_6m"] = grouped[close_col].pct_change(periods=180)
    
    # 4. price_momentum: 일반 모멘텀 (20일 기준, price_momentum_20d와 동일)
    df["price_momentum"] = df["price_momentum_20d"]
    
    # 5. volatility_20d: 20일 rolling 변동성 (연율화)
    # std(일일 수익률, window=20) × sqrt(252)
    vol_20d = grouped["daily_return"].rolling(window=20, min_periods=5).std() * np.sqrt(252)
    df["volatility_20d"] = vol_20d.reset_index(level=0, drop=True).reindex(df.index)
    
    # 6. volatility_60d: 60일 rolling 변동성 (연율화)
    vol_60d = grouped["daily_return"].rolling(window=60, min_periods=10).std() * np.sqrt(252)
    df["volatility_60d"] = vol_60d.reset_index(level=0, drop=True).reindex(df.index)
    
    # 7. volatility: 일반 변동성 (20일 기준, volatility_20d와 동일)
    df["volatility"] = df["volatility_20d"]
    
    # 8. max_drawdown_60d: 60일 rolling window에서 최대 낙폭
    def _calculate_max_drawdown(group: pd.DataFrame) -> pd.Series:
        """60일 rolling window에서 최대 낙폭 계산"""
        prices = group[close_col]
        rolling_max = prices.rolling(window=60, min_periods=10).max()
        drawdown = (prices - rolling_max) / (rolling_max + 1e-10)
        return drawdown
    
    max_dd_series = grouped.apply(lambda x: _calculate_max_drawdown(x)).reset_index(level=0, drop=True)
    df["max_drawdown_60d"] = max_dd_series.reindex(df.index)
    
    # 9. downside_volatility_60d: 60일 하방 변동성
    # 음수 수익률만 사용하여 변동성 계산
    def _calculate_downside_volatility(group: pd.DataFrame) -> pd.Series:
        """하방 변동성 계산"""
        downside = group["daily_return"].where(group["daily_return"] < 0, 0)
        return downside.rolling(window=60, min_periods=10).std() * np.sqrt(252)
    
    downside_vol_series = grouped.apply(lambda x: _calculate_downside_volatility(x)).reset_index(level=0, drop=True)
    df["downside_volatility_60d"] = downside_vol_series.reindex(df.index)
    
    # 10. volume_ratio: 거래량 비율 (20일 이동평균 대비)
    volume_ma = grouped[volume_col].rolling(window=20, min_periods=5).mean()
    df["volume_ratio"] = df[volume_col] / (volume_ma.reset_index(level=0, drop=True).reindex(df.index) + 1e-10)  # 0으로 나누기 방지
    
    # 11. momentum_reversal: 모멘텀 반전 지표
    # 단기 모멘텀(5일)과 장기 모멘텀(20일)의 차이
    short_momentum = grouped[close_col].pct_change(periods=5)
    long_momentum = grouped[close_col].pct_change(periods=20)
    df["momentum_reversal"] = short_momentum - long_momentum
    
    # daily_return은 중간 계산용이므로 제거
    df = df.drop(columns=["daily_return"], errors="ignore")
    
    # 날짜를 다시 문자열로 변환 (원본 형식 유지)
    df[date_col] = df[date_col].dt.strftime("%Y-%m-%d")
    
    return df


def calculate_turnover(
    df: pd.DataFrame,
    market_cap_df: pd.DataFrame,
    *,
    date_col: str = "date",
    ticker_col: str = "ticker",
    volume_col: str = "volume",
) -> pd.DataFrame:
    """
    회전율(turnover) 계산: volume / market_cap
    
    Args:
        df: OHLCV 데이터프레임
        market_cap_df: 시가총액 데이터프레임 (date, ticker, market_cap 포함)
        date_col: 날짜 컬럼명
        ticker_col: 종목코드 컬럼명
        volume_col: 거래량 컬럼명
    
    Returns:
        df에 turnover 컬럼이 추가된 데이터프레임
    """
    df = df.copy()
    
    # 시가총액 데이터와 병합
    merged = df.merge(
        market_cap_df[[date_col, ticker_col, "market_cap"]],
        on=[date_col, ticker_col],
        how="left",
    )
    
    # turnover = volume / market_cap
    merged["turnover"] = merged[volume_col] / (merged["market_cap"] + 1e-10)
    
    # market_cap 컬럼은 제거 (필요시 유지 가능)
    merged = merged.drop(columns=["market_cap"], errors="ignore")
    
    return merged

