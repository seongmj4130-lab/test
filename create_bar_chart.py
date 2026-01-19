from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# KOSPI200 벤치마크 수익률 확인
perf_path = Path("data/track_b_performance_metrics.parquet")
if perf_path.exists():
    df_perf = pd.read_parquet(perf_path)

    # KOSPI200 관련 데이터 찾기
    kospi_data = df_perf[
        df_perf["strategy"].str.contains(
            "kospi|KOSPI|benchmark|BENCHMARK", case=False, na=False
        )
    ]
    if len(kospi_data) > 0:
        print("KOSPI200 벤치마크 데이터:")
        print(kospi_data.to_string(index=False))
    else:
        print("KOSPI200 데이터가 별도로 존재하지 않습니다.")

# 일별 수익률 데이터에서 누적 계산
returns_path = Path("data/strategies_daily_returns_holdout.csv")
if returns_path.exists():
    df_returns = pd.read_csv(returns_path)

    # KOSPI200 누적 수익률 계산 (일별 수익률 합계)
    kospi_cum = df_returns["KOSPI200"].sum() * 100  # 백분율 변환
    print(f"\nKOSPI200 홀드아웃 기간 누적 수익률: {kospi_cum:.2f}%")

    # 전략별 누적 수익률
    strategies = {
        "bt20_short": df_returns.get(
            "BT20 단기 (20일)", pd.Series([0] * len(df_returns))
        ).sum()
        * 100,
        "bt120_long": df_returns.get(
            "BT120 장기 (120일)", pd.Series([0] * len(df_returns))
        ).sum()
        * 100,
        "bt120_ens": df_returns.get(
            "BT120 앙상블 (120일)", pd.Series([0] * len(df_returns))
        ).sum()
        * 100,
    }

    print("\n전략별 누적 수익률:")
    for name, cum_return in strategies.items():
        excess = cum_return - kospi_cum
        print(f"{name}: {cum_return:.2f}% (KOSPI200 대비 {excess:+.2f}%)")

    # 그래프 데이터 준비
    chart_data = {
        "KOSPI200": kospi_cum,
        "BT20 Short": strategies["bt20_short"],
        "BT120 Long": strategies["bt120_long"],
        "BT120 Ensemble": strategies["bt120_ens"],
    }

    # 막대 그래프 생성
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(
        chart_data.keys(),
        chart_data.values(),
        color=["lightcoral", "skyblue", "lightgreen", "gold"],
        edgecolor="black",
        linewidth=1,
    )

    # 값 표시
    for bar, value in zip(bars, chart_data.values()):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + (0.5 if height >= 0 else -1.5),
            f"{value:.1f}%",
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontweight="bold",
            fontsize=10,
        )

    # 그래프 설정
    ax.set_title(
        "Track B 전략 vs KOSPI200 벤치마크 비교\n(홀드아웃 기간: 2023-01-01 ~ 2024-10-31)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_ylabel("총 수익률 (%)", fontsize=12)
    ax.set_xlabel("전략", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    # 0선 표시
    ax.axhline(y=0, color="black", linestyle="-", alpha=0.5)

    plt.tight_layout()
    plt.savefig("track_b_vs_kospi200_comparison.png", dpi=300, bbox_inches="tight")
    print("\n📊 그래프가 생성되었습니다: track_b_vs_kospi200_comparison.png")

    # PPT용 텍스트 설명
    print("\n=== PPT 슬라이드용 그래프 설명 ===")
    print("그래프 제목: Track B 전략 vs KOSPI200 벤치마크 비교")
    print("기간: 홀드아웃 (2023.01.01 ~ 2024.10.31)")
    print()
    print("주요 인사이트:")
    for name, value in chart_data.items():
        if name == "KOSPI200":
            print(f"- {name}: {value:.1f}% (벤치마크)")
        else:
            excess = value - kospi_cum
            status = "초과" if excess > 0 else "미달"
            print(f"- {name}: {value:.1f}% (벤치마크 대비 {excess:+.1f}% {status})")
else:
    print("수익률 데이터 파일 없음")
