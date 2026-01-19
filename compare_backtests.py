#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
기존 백테스트 vs 일별 mark-to-market 백테스트 비교
"""

import pandas as pd
import numpy as np
from src.tracks.track_b.stages.backtest.l7_backtest import BacktestConfig, run_backtest

def compare_backtests():
    """기존 vs 일별 백테스트 비교"""

    print("=== 백테스트 비교 테스트 ===")

    # 테스트 데이터 생성 (간단한 예시)
    dates = pd.date_range('2023-01-01', '2023-01-31', freq='10D')  # 10일마다 리밸런싱
    tickers = ['005930', '000660', '035420', '207940']

    rebalance_data = []
    for date in dates:
        for i, ticker in enumerate(tickers):
            rebalance_data.append({
                'date': date,
                'ticker': ticker,
                'phase': 'test',
                'score_short': 1.0 - i * 0.1,
                'true_short': np.random.normal(0.01, 0.02),
                'score_ens': 1.0 - i * 0.1,
            })

    rebalance_scores = pd.DataFrame(rebalance_data)

    # 일별 가격 데이터 생성
    price_dates = pd.date_range('2023-01-01', '2023-01-31', freq='D')
    price_data = []
    np.random.seed(123)

    for date in price_dates:
        for ticker in tickers:
            base_price = 100000 + (date - pd.Timestamp('2023-01-01')).days * 50
            noise = np.random.normal(0, 1000)
            price_data.append({
                'date': date,
                'ticker': ticker,
                'close': max(base_price + noise, 50000),
            })

    daily_prices = pd.DataFrame(price_data)

    print(f"리밸런싱 데이터: {rebalance_scores.shape}")
    print(f"일별 가격 데이터: {daily_prices.shape}")

    # 1. 기존 백테스트 (포워드 수익률 기반)
    print("\n--- 기존 백테스트 실행 ---")
    cfg_traditional = BacktestConfig(
        holding_days=10,
        top_k=3,
        cost_bps=10.0,
        daily_backtest_enabled=False,  # 기존 방식
        weighting='equal',
    )

    bt_pos_trad, bt_ret_trad, bt_eq_trad, bt_met_trad, quality_trad, warns_trad = run_backtest(
        rebalance_scores=rebalance_scores,
        cfg=cfg_traditional,
        date_col='date',
        ticker_col='ticker',
        phase_col='phase',
    )

    print("기존 백테스트 결과:")
    print(f"  - 포지션: {len(bt_pos_trad)} 행")
    print(f"  - 수익률: {len(bt_ret_trad)} 행")
    print(f"  - 총 수익률: {bt_met_trad['net_total_return'].iloc[0]:.4f}")
    print(f"  - CAGR: {bt_met_trad['net_cagr'].iloc[0]:.4f}")

    # 2. 일별 mark-to-market 백테스트
    print("\n--- 일별 백테스트 실행 ---")
    cfg_daily = BacktestConfig(
        holding_days=10,
        top_k=3,
        cost_bps=10.0,
        daily_backtest_enabled=True,  # 새로운 방식
        weighting='equal',
    )

    bt_pos_daily, bt_ret_daily, bt_eq_daily, bt_met_daily, quality_daily, warns_daily = run_backtest(
        rebalance_scores=rebalance_scores,
        daily_prices=daily_prices,
        cfg=cfg_daily,
        date_col='date',
        ticker_col='ticker',
        phase_col='phase',
    )

    # 월별 누적수익률은 quality에 포함되어 있음
    bt_monthly = pd.DataFrame(quality_daily.get('backtest', {}).get('monthly_returns', []))

    print("일별 백테스트 결과:")
    print(f"  - 포지션: {len(bt_pos_daily)} 행")
    print(f"  - 일별 수익률: {len(bt_ret_daily)} 행")
    print(f"  - 월별 누적수익률: {len(bt_monthly)} 행")
    print(f"  - 총 수익률: {bt_met_daily['net_total_return'].iloc[0]:.4f}")
    print(f"  - CAGR: {bt_met_daily['net_cagr'].iloc[0]:.4f}")

    # 3. 비교 분석
    print("\n=== 비교 분석 ===")
    trad_total = bt_met_trad['net_total_return'].iloc[0]
    daily_total = bt_met_daily['net_total_return'].iloc[0]

    print(f"기존 방식 총 수익률: {trad_total:.4f}")
    print(f"일별 방식 총 수익률: {daily_total:.4f}")
    print(f"차이: {daily_total - trad_total:.4f}")

    if len(bt_monthly) > 0:
        print("\n월별 누적수익률 상세:")
        print(bt_monthly[['year_month', 'monthly_return', 'cumulative_return']].to_string())

    print("\n=== 백테스트 비교 완료 ===")
    return True

if __name__ == "__main__":
    compare_backtests()