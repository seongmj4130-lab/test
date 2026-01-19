#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
실제 프로젝트 데이터로 일별 mark-to-market 백테스트 테스트
"""

import pandas as pd
import numpy as np
from src.tracks.track_b.stages.backtest.l7_backtest import BacktestConfig, _run_daily_backtest

def test_real_daily_backtest():
    """실제 프로젝트 데이터로 일별 백테스트 테스트"""

    print("=== 실제 데이터 로드 ===")

    # rebalance_scores 데이터 로드 (샘플)
    rebalance_scores = pd.read_parquet('data/interim/rebalance_scores.parquet')
    print(f"rebalance_scores 로드 완료: {rebalance_scores.shape}")

    # 2023년 데이터만 사용 (테스트용으로 소규모)
    rebalance_scores_2023 = rebalance_scores[
        (rebalance_scores['date'] >= '2023-01-01') &
        (rebalance_scores['date'] <= '2023-03-31')
    ].copy()
    print(f"2023년 데이터 필터링: {rebalance_scores_2023.shape}")

    # dev phase만 사용
    rebalance_scores_dev = rebalance_scores_2023[rebalance_scores_2023['phase'] == 'dev'].copy()
    print(f"dev phase 필터링: {rebalance_scores_dev.shape}")

    if len(rebalance_scores_dev) == 0:
        print("dev phase 데이터가 없습니다. 임의 데이터 생성으로 테스트합니다.")
        return test_mock_real_data()

    # daily_prices 데이터 로드
    daily_prices = pd.read_parquet('data/interim/dataset_daily.parquet')
    daily_prices_2023 = daily_prices[
        (daily_prices['date'] >= '2023-01-01') &
        (daily_prices['date'] <= '2023-03-31')
    ].copy()
    print(f"일별 가격 데이터 로드 완료: {daily_prices_2023.shape}")

    # BacktestConfig 생성 (일별 백테스트 활성화)
    cfg = BacktestConfig(
        holding_days=20,
        top_k=10,
        cost_bps=10.0,
        daily_backtest_enabled=True,  # 일별 백테스트 활성화
        weighting='equal',
        buffer_k=5,
    )

    print("\n=== 일별 백테스트 실행 ===")

    try:
        # 일별 백테스트 실행
        bt_positions, bt_returns, bt_equity_curve, bt_metrics, quality, warns, _, _, _, bt_monthly_returns = _run_daily_backtest(
            rebalance_scores=rebalance_scores_dev,
            daily_prices=daily_prices_2023,
            cfg=cfg,
            date_col='date',
            ticker_col='ticker',
            phase_col='phase',
        )

        print("✅ 일별 백테스트 성공!")
        print(f"포지션 데이터: {len(bt_positions)} 행")
        print(f"일별 수익률 데이터: {len(bt_returns)} 행")
        print(f"월별 누적수익률 데이터: {len(bt_monthly_returns)} 행")

        # 결과 요약
        print("\n=== 결과 요약 ===")
        if len(bt_monthly_returns) > 0:
            print("월별 누적수익률:")
            print(bt_monthly_returns[['year_month', 'monthly_return', 'cumulative_return']].to_string())

        if len(bt_metrics) > 0:
            print("\n전체 성과 지표:")
            print(bt_metrics[['phase', 'net_cagr', 'net_total_return']].to_string())

        if warns:
            print(f"\n경고사항 ({len(warns)}개):")
            for warn in warns[:5]:  # 처음 5개만 표시
                print(f"- {warn}")

        return True

    except Exception as e:
        print(f"❌ 일별 백테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mock_real_data():
    """실제 데이터가 없을 경우 모의 데이터로 테스트"""
    print("=== 모의 실제 데이터 테스트 ===")

    # 2023년 3개월간의 가상 rebalance_scores 생성
    dates = pd.date_range('2023-01-01', '2023-03-31', freq='20D')  # 20일마다 리밸런싱
    tickers = ['005930', '000660', '035420', '207940', '051910', '006400', '000270', '005380', '068270', '096770']

    rebalance_data = []
    for date in dates:
        for i, ticker in enumerate(tickers):
            rebalance_data.append({
                'date': date,
                'ticker': ticker,
                'phase': 'dev',
                'score_short': 1.0 - i * 0.1,  # 랭킹 기반 스코어
                'true_short': np.random.normal(0.02, 0.05),  # 임의 수익률
                'score_ens': 1.0 - i * 0.1,
            })

    rebalance_scores = pd.DataFrame(rebalance_data)

    # 일별 가격 데이터 생성
    price_dates = pd.date_range('2023-01-01', '2023-03-31', freq='D')
    price_data = []
    np.random.seed(123)

    for date in price_dates:
        for ticker in tickers:
            # 기본 가격 추세 + 랜덤 노이즈
            day_offset = (date - pd.Timestamp('2023-01-01')).days
            base_price = 100000 + day_offset * 100  # 매일 100원 상승 추세
            noise = np.random.normal(0, 2000)  # 일일 변동성
            close_price = base_price + noise
            price_data.append({
                'date': date,
                'ticker': ticker,
                'close': max(close_price, 10000),  # 최소 가격 보장
            })

    daily_prices = pd.DataFrame(price_data)

    print(f"모의 rebalance_scores: {rebalance_scores.shape}")
    print(f"모의 daily_prices: {daily_prices.shape}")

    # BacktestConfig
    cfg = BacktestConfig(
        holding_days=20,
        top_k=5,
        cost_bps=10.0,
        daily_backtest_enabled=True,
        weighting='equal',
        buffer_k=2,
    )

    print("\n=== 모의 일별 백테스트 실행 ===")

    try:
        bt_positions, bt_returns, bt_equity_curve, bt_metrics, quality, warns, _, _, _, bt_monthly_returns = _run_daily_backtest(
            rebalance_scores=rebalance_scores,
            daily_prices=daily_prices,
            cfg=cfg,
            date_col='date',
            ticker_col='ticker',
            phase_col='phase',
        )

        print("✅ 모의 일별 백테스트 성공!")
        print(f"포지션 데이터: {len(bt_positions)} 행")
        print(f"일별 수익률 데이터: {len(bt_returns)} 행")
        print(f"월별 누적수익률 데이터: {len(bt_monthly_returns)} 행")

        # 결과 요약
        print("\n=== 결과 요약 ===")
        if len(bt_monthly_returns) > 0:
            print("월별 누적수익률:")
            print(bt_monthly_returns[['year_month', 'monthly_return', 'cumulative_return']].to_string())

        if len(bt_metrics) > 0:
            print("\n전체 성과 지표:")
            print(bt_metrics[['phase', 'net_cagr', 'net_total_return']].to_string())

        return True

    except Exception as e:
        print(f"❌ 모의 일별 백테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_real_daily_backtest()
    if success:
        print("\n🎉 테스트 성공!")
    else:
        print("\n❌ 테스트 실패!")