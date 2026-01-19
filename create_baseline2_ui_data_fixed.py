import numpy as np
import pandas as pd


def create_baseline2_ui_data_fixed():
    """Baseline2 기준 KOSPI200 TR vs 4전략 비교 데이터 생성 (기존 데이터 활용)"""

    print("📊 Baseline2 UI 데이터 생성 (기존 데이터 활용)")
    print("=" * 70)

    # 기존 UI 데이터 로드
    try:
        existing_data = pd.read_csv("data/ui_monthly_log_returns_data.csv")
        print("✅ 기존 UI 데이터 로드됨")
    except:
        print("❌ 기존 UI 데이터 없음")
        return

    # KOSPI200 TR 데이터 생성 (기존 데이터의 KOSPI200를 TR로 변환)
    print("\n🏛️ KOSPI200 → KOSPI200 TR 변환")

    # 기존 KOSPI200 데이터를 TR로 조정 (연 2.5% 배당 가정)
    monthly_dividend_yield = 0.025 / 12  # 월별 배당 수익률

    # TR = Price Return + Dividend Return
    existing_data["kospi_tr_monthly_return"] = (
        existing_data["kospi200_monthly_return"] + monthly_dividend_yield
    )
    existing_data["kospi_tr_cumulative_return"] = (
        1 + existing_data["kospi_tr_monthly_return"]
    ).cumprod() - 1
    existing_data["kospi_tr_log_cumulative_return"] = np.log(
        1 + existing_data["kospi_tr_cumulative_return"]
    )

    print("✅ KOSPI200 TR 데이터 생성 완료")

    # 전략별 데이터는 기존 데이터 유지 (bt20_short, bt20_ensemble, bt120_long, bt120_ensemble)
    # 컬럼명 정리
    baseline2_data = existing_data.copy()

    # 컬럼명 변경 (bt20_ensemble → bt20_앙상블 등)
    column_mapping = {
        "bt20_ensemble_monthly_return": "bt20_앙상블_monthly_return",
        "bt20_ensemble_cumulative_return": "bt20_앙상블_cumulative_return",
        "bt20_ensemble_log_cumulative_return": "bt20_앙상블_log_cumulative_return",
        "bt120_ensemble_monthly_return": "bt120_앙상블_monthly_return",
        "bt120_ensemble_cumulative_return": "bt120_앙상블_cumulative_return",
        "bt120_ensemble_log_cumulative_return": "bt120_앙상블_log_cumulative_return",
    }

    baseline2_data = baseline2_data.rename(columns=column_mapping)

    # 필요한 컬럼만 선택
    required_columns = [
        "year_month",
        "kospi_tr_monthly_return",
        "kospi_tr_cumulative_return",
        "kospi_tr_log_cumulative_return",
        "bt20_short_monthly_return",
        "bt20_short_cumulative_return",
        "bt20_short_log_cumulative_return",
        "bt20_앙상블_monthly_return",
        "bt20_앙상블_cumulative_return",
        "bt20_앙상블_log_cumulative_return",
        "bt120_long_monthly_return",
        "bt120_long_cumulative_return",
        "bt120_long_log_cumulative_return",
        "bt120_앙상블_monthly_return",
        "bt120_앙상블_cumulative_return",
        "bt120_앙상블_log_cumulative_return",
    ]

    baseline2_data = baseline2_data[required_columns]

    print(
        f"✅ 데이터 정리 완료: {len(baseline2_data)}행 × {len(baseline2_data.columns)}열"
    )

    # 성과 지표 계산
    print("\n📊 성과 지표 계산")

    performance_metrics = {}

    # KOSPI200 TR 성과
    kospi_returns = baseline2_data["kospi_tr_monthly_return"].values
    kospi_total_return = baseline2_data["kospi_tr_cumulative_return"].iloc[-1]
    kospi_cagr = (1 + kospi_total_return) ** (12 / len(baseline2_data)) - 1
    kospi_volatility = np.std(kospi_returns) * np.sqrt(12)
    kospi_sharpe = kospi_cagr / kospi_volatility if kospi_volatility != 0 else 0

    # MDD 계산
    cumulative_returns = baseline2_data["kospi_tr_cumulative_return"]
    kospi_mdd = np.min(cumulative_returns - np.maximum.accumulate(cumulative_returns))

    performance_metrics["KOSPI200 TR"] = {
        "총수익률": kospi_total_return,
        "연평균수익률": kospi_cagr,
        "MDD": kospi_mdd,
        "Sharpe": kospi_sharpe,
        "Hit_Ratio": None,
    }

    # 전략별 성과 계산
    strategies = ["bt20_short", "bt20_앙상블", "bt120_long", "bt120_앙상블"]
    strategy_names = {
        "bt20_short": "BT20 단기",
        "bt20_앙상블": "BT20 앙상블",
        "bt120_long": "BT120 장기",
        "bt120_앙상블": "BT120 앙상블",
    }

    for strategy in strategies:
        monthly_col = f"{strategy}_monthly_return"
        cumulative_col = f"{strategy}_cumulative_return"

        if monthly_col in baseline2_data.columns:
            returns = baseline2_data[monthly_col].values
            total_return = baseline2_data[cumulative_col].iloc[-1]
            cagr = (1 + total_return) ** (12 / len(baseline2_data)) - 1
            volatility = np.std(returns) * np.sqrt(12)
            sharpe = cagr / volatility if volatility != 0 else 0

            # MDD 계산
            cumulative_returns = baseline2_data[cumulative_col]
            mdd = np.min(cumulative_returns - np.maximum.accumulate(cumulative_returns))

            # Hit Ratio
            hit_ratio = (returns > 0).mean()

            performance_metrics[strategy_names[strategy]] = {
                "총수익률": total_return,
                "연평균수익률": cagr,
                "MDD": mdd,
                "Sharpe": sharpe,
                "Hit_Ratio": hit_ratio,
            }

    # 데이터 저장
    print("\n💾 데이터 저장")

    # 월별 데이터 CSV
    monthly_csv_path = "data/ui_baseline2_monthly_log_returns.csv"
    baseline2_data.to_csv(monthly_csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ 월별 데이터: {monthly_csv_path}")

    # 성과 지표 CSV
    metrics_df = pd.DataFrame.from_dict(performance_metrics, orient="index")
    metrics_csv_path = "data/ui_baseline2_performance_metrics.csv"
    metrics_df.to_csv(metrics_csv_path, encoding="utf-8-sig")
    print(f"✅ 성과 지표: {metrics_csv_path}")

    # 결과 요약 출력
    print("\n📋 최종 결과 요약")
    print("-" * 60)

    print("월별 데이터 컬럼:")
    for col in baseline2_data.columns:
        print(f"  • {col}")

    print("\n성과 지표:")
    for name, metrics in performance_metrics.items():
        print(f"  • {name}:")
        print(".2%")
        print(".3f")
        if metrics["Hit_Ratio"] is not None:
            print(".1%")

    print("\n🎯 Baseline2 UI 데이터 생성 완료!")
    print("   - KOSPI200 TR 로그 누적 수익률 그래프")
    print("   - 4개 전략 로그 누적 수익률 비교")
    print("   - 월별 데이터 기반 UI 구현 가능")


if __name__ == "__main__":
    create_baseline2_ui_data_fixed()
