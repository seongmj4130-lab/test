#!/usr/bin/env python3
"""
정정된 누적 수익률 데이터로 3전략 6구간 비교 그래프 생성
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

# 한글 폰트 설정
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def plot_strategies_comparison_corrected():
    """정정된 누적 수익률 데이터로 전략 비교 그래프 생성"""

    # 정정된 데이터 로드
    df = pd.read_csv("data/ui_strategies_cumulative_comparison_corrected.csv")
    df["month"] = pd.to_datetime(df["month"])

    print("=== 정정된 누적 수익률 데이터 그래프 생성 ===")
    print(f"데이터 기간: {df['month'].min()} ~ {df['month'].max()}")

    # 전략별 컬럼 그룹화
    strategies = {
        "BT20 단기 전략": [col for col in df.columns if col.startswith("bt20_short_")],
        "BT120 장기 전략": [col for col in df.columns if col.startswith("bt120_long_")],
        "BT20 앙상블 전략": [col for col in df.columns if col.startswith("bt20_ens_")],
    }

    # 기간 레이블
    periods = ["20일", "40일", "60일", "80일", "100일", "120일"]
    period_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    # 서브플롯 생성 (3행 1열)
    fig, axes = plt.subplots(3, 1, figsize=(15, 18))
    fig.suptitle(
        "3전략 6구간 월별 누적 수익률 비교 (정정된 데이터)\n(월별 수익률 → cumprod(1+r) 재계산)",
        fontsize=16,
        fontweight="bold",
        y=0.95,
    )

    for idx, (strategy_name, cols) in enumerate(strategies.items()):
        ax = axes[idx]

        # 각 기간별 라인 그래프
        for i, (col, period, color) in enumerate(zip(cols, periods, period_colors)):
            ax.plot(
                df["month"],
                df[col],
                label=period,
                color=color,
                linewidth=2,
                marker="o",
                markersize=3,
                alpha=0.8,
            )

        # KOSPI200 벤치마크 추가
        ax.plot(
            df["month"],
            df["kospi200"],
            label="KOSPI200",
            color="black",
            linewidth=2.5,
            linestyle="--",
            alpha=0.8,
        )

        # 그래프 꾸미기
        ax.set_title(f"{strategy_name}", fontsize=14, fontweight="bold", pad=20)
        ax.set_ylabel("누적 수익률 (%)", fontsize=12)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=10)
        ax.grid(True, alpha=0.3)

        # X축 날짜 포맷팅
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        # 0선 추가
        ax.axhline(y=0, color="red", linestyle="-", alpha=0.3, linewidth=1)

    # 전체 레이아웃 조정
    plt.tight_layout()

    # 저장
    output_path = "3_strategies_6_periods_comparison_corrected.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"✅ 정정된 그래프가 '{output_path}' 파일로 저장되었습니다.")

    # 최종 성과 요약 (정정된 데이터 기준)
    print("\n=== 최종 성과 요약 (정정된 데이터, 2024년 12월) ===")
    final_row = df.iloc[-1]

    for strategy_name, cols in strategies.items():
        print(f"\n{strategy_name}:")
        for period, col in zip(periods, cols):
            value = final_row[col]
            kospi = final_row["kospi200"]
            excess = value - kospi
            print(
                f"  {period}: {value:.1f}% (KOSPI200: {kospi:.1f}%, 초과: {excess:+.1f}%)"
            )


def plot_period_comparison_corrected():
    """정정된 데이터로 기간별 전략 비교 그래프"""

    # 정정된 데이터 로드
    df = pd.read_csv("data/ui_strategies_cumulative_comparison_corrected.csv")
    df["month"] = pd.to_datetime(df["month"])

    # 기간별로 그룹화
    periods = ["20", "40", "60", "80", "100", "120"]
    period_labels = ["20일", "40일", "60일", "80일", "100일", "120일"]
    strategy_colors = {
        "BT20 단기": "#1f77b4",
        "BT120 장기": "#ff7f0e",
        "BT20 앙상블": "#2ca02c",
    }

    # 서브플롯 생성 (2행 3열)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "6구간별 3전략 월별 누적 수익률 비교 (정정된 데이터)\n(월별 수익률 → cumprod(1+r) 재계산)",
        fontsize=16,
        fontweight="bold",
        y=0.95,
    )

    axes = axes.flatten()

    for idx, (period, period_label) in enumerate(zip(periods, period_labels)):
        ax = axes[idx]

        # 각 전략별 라인
        for strategy_name, color in strategy_colors.items():
            if strategy_name == "BT20 단기":
                col_name = f"bt20_short_{period}"
            elif strategy_name == "BT120 장기":
                col_name = f"bt120_long_{period}"
            else:  # BT20 앙상블
                col_name = f"bt20_ens_{period}"

            ax.plot(
                df["month"],
                df[col_name],
                label=strategy_name,
                color=color,
                linewidth=2,
                marker="o",
                markersize=3,
            )

        # KOSPI200 추가
        ax.plot(
            df["month"],
            df["kospi200"],
            label="KOSPI200",
            color="black",
            linewidth=2,
            linestyle="--",
            alpha=0.8,
        )

        # 그래프 꾸미기
        ax.set_title(f"{period_label} 보유 기간", fontsize=12, fontweight="bold")
        ax.set_ylabel("누적 수익률 (%)", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # X축 날짜 포맷팅
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        # 0선 추가
        ax.axhline(y=0, color="red", linestyle="-", alpha=0.3, linewidth=1)

    # 전체 레이아웃 조정
    plt.tight_layout()

    # 저장
    output_path = "6_periods_3_strategies_comparison_corrected.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"✅ 기간별 비교 그래프가 '{output_path}' 파일로 저장되었습니다.")


if __name__ == "__main__":
    plot_strategies_comparison_corrected()
    plot_period_comparison_corrected()
