import pandas as pd


def display_total_returns_simple():
    """통일 파라미터 총수익률 결과를 간단하게 표시"""

    print("📊 통일 파라미터 총수익률 결과 (Holdout 기간: 23개월)")
    print("=" * 70)

    # 결과 파일 읽기
    try:
        df = pd.read_csv("results/total_returns_unified_parameters.csv")
    except FileNotFoundError:
        print("❌ 결과 파일을 찾을 수 없습니다.")
        return

    # 결과 표시
    print("<12")
    print("-" * 70)

    for _, row in df.iterrows():
        strategy = row["전략"]
        cagr = row["CAGR"] * 100  # 퍼센트로 변환
        total_return = row["총수익률"] * 100  # 퍼센트로 변환
        mdd = row["MDD"] * 100  # 퍼센트로 변환
        sharpe = row["Sharpe"]
        calmar = row["Calmar"]

        print("<12")

    print()

    print("🔥 핵심 성과 요약")
    print("-" * 30)

    # BT120 평균
    bt120_avg_return = df[df["전략"].str.contains("BT120")]["총수익률"].mean() * 100
    bt120_avg_sharpe = df[df["전략"].str.contains("BT120")]["Sharpe"].mean()

    # BT20 평균
    bt20_avg_return = df[df["전략"].str.contains("BT20")]["총수익률"].mean() * 100
    bt20_avg_sharpe = df[df["전략"].str.contains("BT20")]["Sharpe"].mean()

    print(".1f")
    print(".1f")
    print()

    print("💡 투자 추천")
    print("-" * 20)

    # Sharpe 기준 최고 전략
    best_strategy = df.loc[df["Sharpe"].idxmax(), "전략"]
    best_sharpe = df["Sharpe"].max()

    print(f"🏆 최고 전략: {best_strategy}")
    print(".3f")
    print()
    print("📋 추천 포트폴리오:")
    print("• BT120 전략군 60% + BT20 전략군 40%")
    print("• (안정성과 수익성 균형)")


if __name__ == "__main__":
    display_total_returns_simple()
