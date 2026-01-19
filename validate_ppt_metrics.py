import pandas as pd


def validate_ppt_metrics():
    print("=== PPT 보고서 성과 지표 재계산 및 비교 ===")
    print()

    # 1. Track A 성과 지표 비교
    print("1. Track A 성과 지표 (Hit Ratio)")
    try:
        track_a_df = pd.read_csv("data/track_a_performance_metrics.csv")
        print("현재 데이터:")
        for idx, row in track_a_df.iterrows():
            if "Holdout" in row["metric"]:
                print(f'  {row["metric"]:15}: {row["value"]} (목표: {row["target"]})')
    except Exception as e:
        print(f"Track A 데이터 로드 실패: {e}")

    print()

    # 2. Track B Holdout 성과 지표 비교
    print("2. Track B Holdout 성과 지표")
    try:
        holdout_df = pd.read_csv("data/holdout_performance_metrics.csv")
        strategies = [
            "BT20 단기 (20일)",
            "BT20 앙상블 (20일)",
            "BT120 장기 (120일)",
            "BT120 앙상블 (120일)",
        ]

        for strategy in strategies:
            row = holdout_df[holdout_df["strategy"] == strategy].iloc[0]
            sharpe = row["sharpe_ratio"]
            cagr = row["cagr"] * 100  # 퍼센트로 변환
            mdd = row["mdd"] * 100
            calmar = row["calmar_ratio"]
            hit_ratio = row["hit_ratio"] * 100

            print(f"{strategy}:")
            print(f"  Sharpe: {sharpe:.3f}")
            print(f"  CAGR: {cagr:.1f}%")
            print(f"  MDD: {mdd:.1f}%")
            print(f"  Calmar: {calmar:.3f}")
            print(f"  Hit Ratio: {hit_ratio:.1f}%")
            print()
    except Exception as e:
        print(f"Track B Holdout 데이터 로드 실패: {e}")

    # 3. 누적 수익률 최종 값 비교
    print("3. 최종 누적 수익률 (2024-10-31 기준)")
    try:
        cumulative_df = pd.read_csv(
            "data/strategies_kospi200_monthly_cumulative_returns.csv"
        )
        final_row = cumulative_df.iloc[-1]

        strategies_cum = [
            "BT20 단기 (20일)",
            "BT20 앙상블 (20일)",
            "BT120 장기 (120일)",
            "BT120 앙상블 (120일)",
            "KOSPI200",
        ]
        for strategy in strategies_cum:
            if strategy in final_row:
                final_return = (final_row[strategy] - 1) * 100  # 누적 수익률을 퍼센트로
                print(f"{strategy}: {final_return:.1f}%")
    except Exception as e:
        print(f"누적 수익률 데이터 로드 실패: {e}")

    print()

    # 4. 월별 승률 계산
    print("4. 월별 승률 분석")
    try:
        monthly_df = pd.read_csv("data/strategies_kospi200_monthly_returns.csv")

        win_rates = {}
        strategies_cum = [
            "BT20 단기 (20일)",
            "BT20 앙상블 (20일)",
            "BT120 장기 (120일)",
            "BT120 앙상블 (120일)",
            "KOSPI200",
        ]

        for strategy in strategies_cum:
            monthly_col = f"{strategy}_monthly_return"
            if monthly_col in monthly_df.columns:
                monthly_returns = monthly_df[monthly_col]
                win_rate = (monthly_returns > 0).mean() * 100
                win_rates[strategy] = win_rate

        for strategy, win_rate in win_rates.items():
            print(f"{strategy}: {win_rate:.1f}%")
    except Exception as e:
        print(f"월별 수익률 데이터 로드 실패: {e}")

    print()

    # 5. 최고/최악 월 수익률
    print("5. 최고/최악 월 수익률")
    try:
        monthly_df = pd.read_csv("data/strategies_kospi200_monthly_returns.csv")

        strategies_cum = [
            "BT20 단기 (20일)",
            "BT20 앙상블 (20일)",
            "BT120 장기 (120일)",
            "BT120 앙상블 (120일)",
            "KOSPI200",
        ]

        for strategy in strategies_cum:
            monthly_col = f"{strategy}_monthly_return"
            if monthly_col in monthly_df.columns:
                monthly_returns = monthly_df[monthly_col]
                max_month = monthly_returns.max() * 100
                min_month = monthly_returns.min() * 100
                print(f"{strategy}: 최고 {max_month:.1f}%, 최악 {min_month:.1f}%")
    except Exception as e:
        print(f"월별 수익률 데이터 로드 실패: {e}")

    print()
    print("=== PPT 보고서 vs 현재 데이터 비교 ===")
    print()

    # PPT 보고서의 값들과 비교
    print("PPT 보고서 값 (참조):")
    print("Track A Holdout Hit Ratios: 50.99% ~ 51.06%")
    print("BT20 단기 Sharpe: 0.914, CAGR: 13.4%, MDD: -4.4%")
    print(
        "최종 누적 수익률 (요약): BT20 단기 26.2%, BT20 앙상블 22.4%, BT120 장기 17.9%, BT120 앙상블 13.5%, KOSPI200 6.4%"
    )
    print(
        "월별 승률: BT20 단기 77.3%, BT20 앙상블 68.2%, BT120 장기 54.5%, BT120 앙상블 59.1%, KOSPI200 50.0%"
    )


if __name__ == "__main__":
    validate_ppt_metrics()
