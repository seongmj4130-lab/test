# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/stages/backtest/l1d_market_regime.py
"""
[Stage5] 시장 국면(regime) 계산 모듈
- rebalance 날짜 기준으로 시장 국면(bull/neutral/bear) 계산
- ohlcv_daily 데이터를 사용하여 가격/거래량/변동성 지표로 자동 분류
- 외부 API 호출 없이 내부 데이터만 사용
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_market_regime(
    *,
    rebalance_dates: pd.Series | list[str] | pd.DataFrame,
    ohlcv_daily: pd.DataFrame,  # OHLCV 데이터 (필수)
    lookback_days: int = 60,
    neutral_band: float = 0.05,  # ±5% 범위를 neutral로 분류 (예: 0.05 = ±5%)
    use_volume: bool = True,  # 거래량 지표 사용 여부
    use_volatility: bool = True,  # 변동성 지표 사용 여부
) -> pd.DataFrame:
    """
    [Stage5] rebalance 날짜 기준으로 시장 국면(regime) 계산

    외부 API 없이 ohlcv_daily 데이터를 사용하여 가격/거래량/변동성 지표로 자동 분류합니다.

    Args:
        rebalance_dates: rebalance 날짜 리스트 또는 DataFrame (date 컬럼 포함)
        ohlcv_daily: OHLCV 일별 데이터 (date, ticker, close, volume 컬럼 필요)
        lookback_days: 국면 판단을 위한 lookback 기간 (일수, 기본값: 60일)
        neutral_band: ±범위를 neutral로 분류 (기본값: 0.05 = ±5%)
                     예: 0.05면 [-5%, +5%] 범위가 neutral, 그 외는 bull/bear
        use_volume: 거래량 지표 사용 여부 (기본값: True)
        use_volatility: 변동성 지표 사용 여부 (기본값: True)

    Returns:
        DataFrame with columns:
          - date
          - regime (3단계: bull/neutral/bear)
          - lookback_return_pct
          - lookback_volatility_pct (변동성, use_volatility=True일 때)
          - lookback_volume_change_pct (거래량 변화율, use_volume=True일 때)

    국면 판단 로직:
    1. 가격 수익률: lookback 기간 동안의 시장 가중 평균 수익률
    2. 변동성: lookback 기간 동안의 일일 수익률 표준편차 (연환산)
    3. 거래량: lookback 기간 동안의 거래량 변화율
    4. 종합 판단:
       - Bull: 수익률 > neutral_band AND (변동성 낮음 OR 거래량 증가)
       - Bear: 수익률 < -neutral_band AND (변동성 높음 OR 거래량 감소)
       - Neutral: 그 외 (수익률이 ±neutral_band 범위 내)
    """
    # rebalance_dates 처리
    if isinstance(rebalance_dates, pd.DataFrame):
        if "date" not in rebalance_dates.columns:
            raise ValueError("rebalance_dates DataFrame에 'date' 컬럼이 없습니다.")
        dates = pd.to_datetime(rebalance_dates["date"]).unique()
    elif isinstance(rebalance_dates, pd.Series):
        dates = pd.to_datetime(rebalance_dates).unique()
    else:
        dates = pd.to_datetime(rebalance_dates).unique()

    dates = sorted(dates)

    if len(dates) == 0:
        raise ValueError("rebalance_dates가 비어있습니다.")

    # ohlcv_daily 데이터 확인
    if ohlcv_daily is None or len(ohlcv_daily) == 0:
        raise ValueError("ohlcv_daily 데이터가 비어있습니다.")

    required_cols = ["date", "ticker", "close"]
    missing_cols = [col for col in required_cols if col not in ohlcv_daily.columns]
    if missing_cols:
        raise ValueError(f"ohlcv_daily에 필수 컬럼이 없습니다: {missing_cols}")

    # 데이터 준비
    ohlcv = ohlcv_daily.copy()
    ohlcv["date"] = pd.to_datetime(ohlcv["date"])
    ohlcv = ohlcv.sort_values(["date", "ticker"]).reset_index(drop=True)

    logger.info(
        f"[L1D] 시장 국면 계산 시작: {len(dates)}개 rebalance 날짜, {len(ohlcv)}개 OHLCV 레코드"
    )

    # 각 rebalance 날짜에 대해 국면 계산
    regime_rows = []

    for rebal_date in dates:
        rebal_date = pd.to_datetime(rebal_date)

        # lookback 시작일 계산
        lookback_start = rebal_date - pd.Timedelta(days=lookback_days)

        # 해당 기간의 OHLCV 데이터 필터링
        period_data = ohlcv[
            (ohlcv["date"] >= lookback_start) & (ohlcv["date"] <= rebal_date)
        ].copy()

        if len(period_data) == 0:
            logger.warning(
                f"[L1D] {rebal_date.strftime('%Y-%m-%d')}: 데이터 부족, neutral로 분류"
            )
            regime_rows.append(
                {
                    "date": rebal_date,
                    "regime": "neutral",
                    "lookback_return_pct": np.nan,
                    "lookback_volatility_pct": np.nan,
                    "lookback_volume_change_pct": np.nan,
                }
            )
            continue

        # [수정] 전체 period_data에서 종목별로 날짜 순서대로 정렬 후 전일 종가 계산
        period_data_sorted = period_data.sort_values(["ticker", "date"]).copy()
        period_data_sorted["prev_close"] = period_data_sorted.groupby("ticker")[
            "close"
        ].shift(1)
        period_data_sorted["daily_return"] = (
            period_data_sorted["close"] - period_data_sorted["prev_close"]
        ) / period_data_sorted["prev_close"]

        # 일별 시장 가중 평균 수익률 계산 (시가총액 가중 또는 동일 가중)
        daily_returns = []
        daily_volumes = []

        for date in sorted(period_data_sorted["date"].unique()):
            day_data = period_data_sorted[period_data_sorted["date"] == date]

            # 유효한 수익률만 사용
            valid_returns = day_data["daily_return"].dropna()
            if len(valid_returns) > 0:
                # 동일 가중 평균 수익률
                daily_returns.append(valid_returns.mean())

            # 거래량 집계
            if "volume" in day_data.columns:
                valid_volumes = day_data["volume"].dropna()
                if len(valid_volumes) > 0:
                    daily_volumes.append(valid_volumes.sum())

        if len(daily_returns) < 2:
            logger.warning(
                f"[L1D] {rebal_date.strftime('%Y-%m-%d')}: 수익률 데이터 부족, neutral로 분류"
            )
            regime_rows.append(
                {
                    "date": rebal_date,
                    "regime": "neutral",
                    "lookback_return_pct": np.nan,
                    "lookback_volatility_pct": np.nan,
                    "lookback_volume_change_pct": np.nan,
                }
            )
            continue

        # 1. 가격 수익률: 전체 기간 누적 수익률
        returns_series = pd.Series(daily_returns)
        total_return_pct = ((1 + returns_series).prod() - 1) * 100.0

        # 2. 변동성: 일일 수익률 표준편차 (연환산)
        volatility_pct = (
            returns_series.std() * np.sqrt(252) * 100.0 if use_volatility else np.nan
        )

        # 3. 거래량 변화율: 초기 vs 최종 거래량
        volume_change_pct = np.nan
        if use_volume and len(daily_volumes) >= 2:
            start_volume = daily_volumes[0]
            end_volume = daily_volumes[-1]
            if start_volume > 0:
                volume_change_pct = ((end_volume - start_volume) / start_volume) * 100.0

        # 4. 국면 분류
        neutral_threshold_pct = abs(neutral_band) * 100.0  # 예: 0.05 → ±5%

        # 기본 분류: 수익률 기준
        if abs(total_return_pct) <= neutral_threshold_pct:
            regime = "neutral"
        elif total_return_pct > neutral_threshold_pct:
            # Bull 후보: 추가 확인
            if use_volatility and not pd.isna(volatility_pct):
                # 변동성이 너무 높으면(>30%) neutral로 조정
                if volatility_pct > 30.0:
                    regime = "neutral"
                else:
                    regime = "bull"
            elif use_volume and not pd.isna(volume_change_pct):
                # 거래량이 크게 감소하면(-20% 이상) neutral로 조정
                if volume_change_pct < -20.0:
                    regime = "neutral"
                else:
                    regime = "bull"
            else:
                regime = "bull"
        else:  # total_return_pct < -neutral_threshold_pct
            # Bear 후보: 추가 확인
            if use_volatility and not pd.isna(volatility_pct):
                # 변동성이 매우 높으면(>40%) bear 확정
                if volatility_pct > 40.0:
                    regime = "bear"
                else:
                    regime = "bear"
            elif use_volume and not pd.isna(volume_change_pct):
                # 거래량이 크게 증가하면(+50% 이상) panic selling 가능성, bear 확정
                if volume_change_pct > 50.0:
                    regime = "bear"
                else:
                    regime = "bear"
            else:
                regime = "bear"

        regime_rows.append(
            {
                "date": rebal_date,
                "regime": regime,
                "lookback_return_pct": total_return_pct,
                "lookback_volatility_pct": volatility_pct,
                "lookback_volume_change_pct": volume_change_pct,
            }
        )

        logger.debug(
            f"[L1D] {rebal_date.strftime('%Y-%m-%d')}: {regime} (수익률={total_return_pct:.2f}%, 변동성={volatility_pct:.2f}%, 거래량변화={volume_change_pct:.2f}%)"
        )

    df = pd.DataFrame(regime_rows)
    df = df.sort_values("date").reset_index(drop=True)

    # [Phase 2 수정] 3단계 분류 완료 (bull, neutral, bear)
    regime_counts = df["regime"].value_counts()

    logger.info(f"[L1D] 국면 계산 완료: 총 {len(df)}개 rebalance 날짜")
    logger.info("[L1D] 국면 분포 (3단계):")
    for regime in ["bull", "neutral", "bear"]:
        count = regime_counts.get(regime, 0)
        logger.info(f"[L1D]   {regime}: {count}개 ({count/len(df)*100:.1f}%)")

    return df
