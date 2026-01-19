import os

import matplotlib.pyplot as plt
import pandas as pd

# 스타일 설정
plt.style.use("default")
plt.rcParams["figure.figsize"] = (14, 8)
plt.rcParams["font.family"] = "Malgun Gothic" if os.name == "nt" else "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


def load_cumulative_comparison_data():
    """전략별 누적 비교 데이터 로드"""

    df = pd.read_csv("data/ui_strategies_cumulative_comparison.csv")

    print("📊 전략별 누적 비교 데이터 로드 완료")
    print(f"   • 데이터 기간: {len(df)}개월")
    print(f"   • 컬럼 수: {len(df.columns)}개")
    print(f"   • 시작: {df['year_month'].iloc[0]}")
    print(f"   • 종료: {df['year_month'].iloc[-1]}")

    return df


def show_cumulative_columns(df):
    """누적 비교 데이터 컬럼 설명"""

    print("\n📋 전략별 누적 비교 그래프용 데이터 컬럼")
    print("=" * 60)

    columns_description = {
        "year_month": "연월 (X축용 날짜)",
        "kospi_tr_cumulative_log_return": "KOSPI TR 누적 로그 수익률 (%) - 배당 포함 총수익지수",
        "bt20_단기_cumulative_log_return": "BT20 단기 누적 로그 수익률 (%) - 20일 리밸런싱, 롱숏 전략",
        "bt20_앙상블_cumulative_log_return": "BT20 앙상블 누적 로그 수익률 (%) - 20일 리밸런싱, 롱온리 전략",
        "bt120_장기_cumulative_log_return": "BT120 장기 누적 로그 수익률 (%) - 120일 리밸런싱, 롱온리 전략",
        "bt120_앙상블_cumulative_log_return": "BT120 앙상블 누적 로그 수익률 (%) - 120일 리밸런싱, 롱온리 전략",
    }

    for col, desc in columns_description.items():
        if col in df.columns:
            values = df[col]
            print(f"• {col}: {desc}")
            print(".2f")
            print(".2f")
            print(".3f")
            print()


def create_strategies_cumulative_comparison_chart(df, output_path):
    """전략별 누적 로그 수익률 비교 그래프 생성"""

    plt.figure(figsize=(14, 8))

    # 색상 설정
    colors = {
        "KOSPI TR": "#FF6B6B",  # Red
        "BT20 단기": "#4ECDC4",  # Teal
        "BT20 앙상블": "#45B7D1",  # Light Blue
        "BT120 장기": "#96CEB4",  # Mint Green
        "BT120 앙상블": "#FECA57",  # Yellow
    }

    # KOSPI TR 먼저 그리기
    plt.plot(
        df["year_month"],
        df["kospi_tr_cumulative_log_return"],
        label="KOSPI TR",
        color=colors["KOSPI TR"],
        linewidth=3,
        alpha=0.9,
    )

    # 전략들 그리기
    strategy_mapping = {
        "bt20_단기_cumulative_log_return": "BT20 단기",
        "bt20_앙상블_cumulative_log_return": "BT20 앙상블",
        "bt120_장기_cumulative_log_return": "BT120 장기",
        "bt120_앙상블_cumulative_log_return": "BT120 앙상블",
    }

    for col, display_name in strategy_mapping.items():
        plt.plot(
            df["year_month"],
            df[col],
            label=display_name,
            color=colors[display_name],
            linewidth=2.5,
            alpha=0.9,
        )

    # 0선 추가
    plt.axhline(y=0, color="black", linestyle="--", alpha=0.8, linewidth=1)

    # 그래프 설정
    plt.title(
        "전략별 누적 로그 수익률 비교 (2023-2024)", fontsize=16, fontweight="bold"
    )
    plt.ylabel("누적 수익률 (%)", fontsize=12)
    plt.xlabel("기간", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.legend(loc="upper left", fontsize=10, frameon=True, framealpha=0.7)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    # 저장
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"✅ 전략별 누적 비교 그래프 생성: {output_path}")


def create_performance_summary(df):
    """성과 요약 생성"""

    print("\n📊 전략별 누적 수익률 성과 요약")
    print("=" * 80)

    strategy_names = {
        "kospi_tr_cumulative_log_return": "KOSPI TR",
        "bt20_단기_cumulative_log_return": "BT20 단기",
        "bt20_앙상블_cumulative_log_return": "BT20 앙상블",
        "bt120_장기_cumulative_log_return": "BT120 장기",
        "bt120_앙상블_cumulative_log_return": "BT120 앙상블",
    }

    summary_data = []

    for col, name in strategy_names.items():
        start_val = df[col].iloc[0]
        end_val = df[col].iloc[-1]
        total_return = end_val - start_val

        # 최대/최소 값
        max_val = df[col].max()
        min_val = df[col].min()

        # 변동성 (표준편차)
        volatility = df[col].std()

        # 승률 (양수 개월 비율)
        monthly_returns = df[col].diff().dropna()
        win_rate = (monthly_returns > 0).mean() * 100

        summary_data.append(
            {
                "전략": name,
                "시작값": start_val,
                "종료값": end_val,
                "총수익률": total_return,
                "최고값": max_val,
                "최저값": min_val,
                "변동성": volatility,
                "승률": win_rate,
            }
        )

    # DataFrame 생성
    summary_df = pd.DataFrame(summary_data)

    # 출력
    for _, row in summary_df.iterrows():
        print(f"\n🏆 {row['전략']}")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".3f")
        print(".1f")

    # CSV로 저장
    summary_df.to_csv(
        "results/strategies_cumulative_performance_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    print(
        "\n✅ 성과 요약 CSV 저장: results/strategies_cumulative_performance_summary.csv"
    )

    return summary_df


def create_comparison_table(df):
    """전략 비교 테이블 생성"""

    print("\n📋 전략별 성과 비교표")
    print("=" * 80)

    strategy_names = {
        "kospi_tr_cumulative_log_return": "KOSPI TR",
        "bt20_단기_cumulative_log_return": "BT20 단기",
        "bt20_앙상블_cumulative_log_return": "BT20 앙상블",
        "bt120_장기_cumulative_log_return": "BT120 장기",
        "bt120_앙상블_cumulative_log_return": "BT120 앙상블",
    }

    # 테이블 헤더
    print("<25")
    print("-" * 125)

    for col, name in strategy_names.items():
        start_val = df[col].iloc[0]
        end_val = df[col].iloc[-1]
        total_return = end_val - start_val
        max_drawdown = (
            start_val - min_val if (min_val := df[col].min()) < start_val else 0
        )

        print("<25")

    print()


def main():
    """메인 실행 함수"""

    print("🎯 전략별 누적 비교 그래프 생성 시작")
    print("=" * 50)

    # 데이터 로드
    df = load_cumulative_comparison_data()

    # 컬럼 설명
    show_cumulative_columns(df)

    # 성과 요약
    performance_summary = create_performance_summary(df)

    # 비교 테이블
    create_comparison_table(df)

    # 그래프 생성
    output_path = "results/strategies_cumulative_comparison_updated.png"
    create_strategies_cumulative_comparison_chart(df, output_path)

    print("\n🎉 모든 그래프 생성 완료!")
    print(f"   • 메인 그래프: {output_path}")
    print("   • 성과 요약: results/strategies_cumulative_performance_summary.csv")


if __name__ == "__main__":
    main()
