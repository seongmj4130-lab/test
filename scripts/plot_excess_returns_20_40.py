"""
20일과 40일 트렌치의 초과수익률 그래프 생성
단기랭킹, 장기랭킹, 통합랭킹(5:5) 전략 비교
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 한글 폰트 설정
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def load_and_process_data():
    """데이터 로드 및 가공"""
    file_path = (
        Path(__file__).parent
        / "data"
        / "dummy_excess_return_monthly_2023_2024_by_horizon_and_rank.csv"
    )
    df = pd.read_csv(file_path)

    # 20일과 40일 데이터만 필터링
    df_filtered = df[df["horizon_days"].isin([20, 40])].copy()

    # 날짜 형식 변환
    df_filtered["date"] = pd.to_datetime(df_filtered["month"])
    df_filtered = df_filtered.sort_values(["horizon_days", "date"])

    return df_filtered


def create_comparison_chart(df):
    """20일 vs 40일 비교 차트 생성"""

    # 컬러 팔레트 설정
    colors = {
        "단기랭킹": "#1f77b4",  # 파란색
        "장기랭킹": "#ff7f0e",  # 주황색
        "통합랭킹(5:5)": "#2ca02c",  # 초록색
    }

    # 2x1 서브플롯 생성
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    fig.suptitle(
        "20일 vs 40일 트렌치 초과수익률 비교\n(KOSPI200 대비)",
        fontsize=16,
        fontweight="bold",
    )

    horizons = [20, 40]

    for idx, horizon in enumerate(horizons):
        ax = axes[idx]

        # 해당 트렌치 데이터 필터링
        horizon_data = df[df["horizon_days"] == horizon]

        # 각 전략별로 라인 플롯
        for strategy in ["단기랭킹", "장기랭킹", "통합랭킹(5:5)"]:
            strategy_data = horizon_data[horizon_data["rank_strategy"] == strategy]

            if not strategy_data.empty:
                ax.plot(
                    strategy_data["date"],
                    strategy_data["excess_return_pct"],
                    label=strategy,
                    color=colors[strategy],
                    linewidth=2.5,
                    marker="o",
                    markersize=4,
                    alpha=0.8,
                )

        # 그래프 설정
        ax.set_title(f"{horizon}일 트렌치", fontsize=14, fontweight="bold")
        ax.set_xlabel("날짜", fontsize=12)
        ax.set_ylabel("초과수익률 (%)", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=11)

        # y축 범위 설정 (데이터에 맞게 자동 조정하되, 0선 표시)
        y_min = horizon_data["excess_return_pct"].min()
        y_max = horizon_data["excess_return_pct"].max()
        margin = (y_max - y_min) * 0.1
        ax.set_ylim(y_min - margin, y_max + margin)

        # 0선 추가
        ax.axhline(y=0, color="black", linestyle="--", alpha=0.5, linewidth=1)

        # x축 날짜 포맷팅
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    return fig


def create_overlay_chart(df):
    """20일과 40일을 같은 차트에 오버레이"""

    # 컬러 팔레트 설정 (선 스타일로 구분)
    strategies = ["단기랭킹", "장기랭킹", "통합랭킹(5:5)"]
    horizons = [20, 40]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]  # 파랑, 주황, 초록
    line_styles = ["-", "--"]  # solid, dashed

    fig, ax = plt.subplots(figsize=(15, 8))

    for s_idx, strategy in enumerate(strategies):
        for h_idx, horizon in enumerate(horizons):
            strategy_data = df[
                (df["rank_strategy"] == strategy) & (df["horizon_days"] == horizon)
            ]

            if not strategy_data.empty:
                label = f"{strategy} ({horizon}일)"
                ax.plot(
                    strategy_data["date"],
                    strategy_data["excess_return_pct"],
                    label=label,
                    color=colors[s_idx],
                    linestyle=line_styles[h_idx],
                    linewidth=2.5,
                    marker="o" if h_idx == 0 else "s",
                    markersize=4,
                    alpha=0.8,
                )

    ax.set_title(
        "20일 vs 40일 트렌치 초과수익률 오버레이 비교", fontsize=16, fontweight="bold"
    )
    ax.set_xlabel("날짜", fontsize=12)
    ax.set_ylabel("초과수익률 (%)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=10, bbox_to_anchor=(1.05, 1))

    # 0선 추가
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5, linewidth=1)

    # x축 날짜 포맷팅
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    return fig


def calculate_summary_stats(df):
    """요약 통계 계산"""
    summary = {}

    for horizon in [20, 40]:
        horizon_data = df[df["horizon_days"] == horizon]
        summary[horizon] = {}

        for strategy in ["단기랭킹", "장기랭킹", "통합랭킹(5:5)"]:
            strategy_data = horizon_data[horizon_data["rank_strategy"] == strategy]

            if not strategy_data.empty:
                excess_returns = strategy_data["excess_return_pct"].values
                summary[horizon][strategy] = {
                    "최종_초과수익률": excess_returns[-1],
                    "평균_초과수익률": np.mean(excess_returns),
                    "표준편차": np.std(excess_returns),
                    "최대_초과수익률": np.max(excess_returns),
                    "최소_초과수익률": np.min(excess_returns),
                    "양의_개월_비율": (excess_returns > 0).mean() * 100,
                }

    return summary


def print_summary_stats(summary):
    """요약 통계 출력"""
    print("\n" + "=" * 80)
    print("초과수익률 요약 통계 (2023-01 ~ 2024-12)")
    print("=" * 80)

    for horizon in [20, 40]:
        print(f"\n{horizon}일 트렌치:")
        print("-" * 40)

        for strategy, stats in summary[horizon].items():
            print(f"\n{strategy}:")
            print(".2f")
            print(".2f")
            print(".2f")
            print(".2f")
            print(".2f")
            print(".1f")


def main():
    """메인 실행 함수"""

    # 데이터 로드 및 가공
    df = load_and_process_data()
    print(f"데이터 로드 완료: {len(df)} 행")
    print(f"트렌치: {sorted(df['horizon_days'].unique())}")
    print(f"전략: {sorted(df['rank_strategy'].unique())}")
    print(f"기간: {df['date'].min()} ~ {df['date'].max()}")

    # 요약 통계 계산 및 출력
    summary = calculate_summary_stats(df)
    print_summary_stats(summary)

    # 그래프 생성
    print("\n그래프 생성 중...")

    # 개별 비교 차트
    fig1 = create_comparison_chart(df)
    chart1_path = (
        Path(__file__).parent / "results" / "excess_returns_20_40_comparison.png"
    )
    chart1_path.parent.mkdir(exist_ok=True)
    fig1.savefig(chart1_path, dpi=300, bbox_inches="tight")
    print(f"개별 비교 차트 저장: {chart1_path}")

    # 오버레이 차트
    fig2 = create_overlay_chart(df)
    chart2_path = Path(__file__).parent / "results" / "excess_returns_20_40_overlay.png"
    fig2.savefig(chart2_path, dpi=300, bbox_inches="tight")
    print(f"오버레이 차트 저장: {chart2_path}")

    print("\n그래프 생성 완료!")
    print("차트 파일이 results 폴더에 저장되었습니다.")


if __name__ == "__main__":
    main()
