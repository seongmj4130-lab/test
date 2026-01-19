#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
일별 mark-to-market 백테스트 테스트 스크립트
"""

import numpy as np
import pandas as pd

from src.tracks.track_b.stages.backtest.l7_backtest import (
    BacktestConfig,
    _calculate_daily_portfolio_returns,
    _convert_daily_to_monthly_returns,
)


def test_daily_backtest():
    """일별 백테스트 기본 기능 테스트"""

    # 테스트 데이터 생성
    dates = pd.date_range('2023-01-01', '2023-01-31', freq='D')
    tickers = ['000030', '000050', '000070']

    # 일별 가격 데이터 생성 (단순 상승 추세 + 노이즈)
    np.random.seed(42)
    daily_prices = []
    for date in dates:
        for ticker in tickers:
            # 기본 추세: 매일 10원 상승
            base_price = 1000 + (date - pd.Timestamp('2023-01-01')).days * 10
            # 랜덤 노이즈 추가
            price_noise = np.random.normal(0, 20)
            close_price = base_price + price_noise
            daily_prices.append({
                'date': date,
                'ticker': ticker,
                'close': max(close_price, 100)  # 최소 가격 보장
            })

    daily_prices_df = pd.DataFrame(daily_prices)

    # 리밸런싱 포지션 데이터 (중간에 한 번 리밸런싱)
    rebalance_dates = [pd.Timestamp('2023-01-01'), pd.Timestamp('2023-01-15')]
    positions_at_rebalance = {
        pd.Timestamp('2023-01-01'): {'000030': 0.5, '000050': 0.3, '000070': 0.2},
        pd.Timestamp('2023-01-15'): {'000030': 0.4, '000050': 0.4, '000070': 0.2},
    }

    print("=== 테스트 데이터 생성 완료 ===")
    print(f"가격 데이터: {len(daily_prices_df)} 행")
    print(f"리밸런싱 날짜: {len(rebalance_dates)} 개")
    print(f"포지션 데이터: {len(positions_at_rebalance)} 개 날짜")

    # 일별 포트 수익률 계산
    daily_returns = _calculate_daily_portfolio_returns(
        rebalance_dates=rebalance_dates,
        daily_prices=daily_prices_df,
        positions_at_rebalance=positions_at_rebalance,
        cost_bps=10.0,
        slippage_bps=0.0,
        volatility_adjustment_enabled=False,
        volatility_lookback_days=60,
        target_volatility=0.15,
        volatility_adjustment_max=1.2,
        volatility_adjustment_min=0.6,
        risk_scaling_enabled=False,
        risk_scaling_bear_multiplier=0.7,
        risk_scaling_neutral_multiplier=0.9,
        risk_scaling_bull_multiplier=1.0,
    )

    # 월별 누적수익률 변환
    monthly_returns = _convert_daily_to_monthly_returns(daily_returns)

    print("\n=== 일별 포트 수익률 샘플 ===")
    print(daily_returns.head(10)[['date', 'portfolio_return', 'trading_cost', 'is_rebalance_date', 'n_positions']].to_string())

    print(f"\n=== 일별 수익률 통계 ===")
    print(f"총 일수: {len(daily_returns)}")
    print(f"리밸런싱 일수: {daily_returns['is_rebalance_date'].sum()}")
    print(f"평균 일별 수익률: {daily_returns['portfolio_return'].mean():.6f}")
    print(f"총 포트 수익률: {daily_returns['portfolio_return'].sum():.6f}")
    print(f"총 거래 비용: {daily_returns['trading_cost'].sum():.6f}")

    print("\n=== 월별 누적수익률 ===")
    print(monthly_returns[['year_month', 'monthly_return', 'cumulative_return', 'n_trading_days']].to_string())

    print("\n=== 테스트 성공 ===")
    return True

if __name__ == "__main__":
    test_daily_backtest()
