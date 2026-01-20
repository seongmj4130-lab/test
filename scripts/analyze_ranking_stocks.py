import pandas as pd


def analyze_ranking_stocks():
    """단기, 장기, 통합 랭킹의 종목 차이 분석"""

    print("🔍 단기 vs 장기 vs 통합 랭킹 종목 비교")
    print("=" * 60)

    # 데이터 로드 (두 파일 모두 같은 데이터)
    try:
        df = pd.read_csv("data/daily_holdout_short_ranking_top20.csv")
        print("✅ 랭킹 데이터 로드됨")
    except:
        print("❌ 랭킹 데이터 파일 없음")
        return

    # 특정 날짜(2023-01-02)의 데이터 추출
    target_date = "2023-01-02"
    df_date = df[df["date"] == target_date].copy()

    if df_date.empty:
        print(f"❌ {target_date} 데이터 없음")
        return

    print(f"📅 분석 날짜: {target_date}")
    print(f"📊 총 종목 수: {len(df_date)}")
    print()

    # 단기 랭킹 top10 (score_short 기준)
    short_top10 = df_date.nlargest(10, "score_short")[
        ["ranking", "ticker", "score_short", "score_long", "score_ens"]
    ]
    short_tickers = set(short_top10["ticker"].astype(str).values)

    # 장기 랭킹 top10 (score_long 기준)
    long_top10 = df_date.nlargest(10, "score_long")[
        ["ranking", "ticker", "score_short", "score_long", "score_ens"]
    ]
    long_tickers = set(long_top10["ticker"].astype(str).values)

    # 통합 랭킹 top10 (score_ens 기준)
    ens_top10 = df_date.nlargest(10, "score_ens")[
        ["ranking", "ticker", "score_short", "score_long", "score_ens"]
    ]
    ens_tickers = set(ens_top10["ticker"].astype(str).values)

    print("🏆 단기 랭킹 Top 10")
    print("-" * 70)
    for _, row in short_top10.iterrows():
        ticker = str(row["ticker"])
        print("<6")

    print()

    print("🏆 장기 랭킹 Top 10")
    print("-" * 70)
    for _, row in long_top10.iterrows():
        ticker = str(row["ticker"])
        print("<6")

    print()

    print("🏆 통합 랭킹 Top 10 (단기+장기 5:5)")
    print("-" * 70)
    for _, row in ens_top10.iterrows():
        ticker = str(row["ticker"])
        print("<6")

    print()

    # 종목 overlap 분석
    print("🔄 종목 Overlap 분석")
    print("-" * 40)

    short_long_overlap = short_tickers & long_tickers
    short_ens_overlap = short_tickers & ens_tickers
    long_ens_overlap = long_tickers & ens_tickers
    all_overlap = short_tickers & long_tickers & ens_tickers

    print(f"단기 ↔ 장기 overlap: {len(short_long_overlap)} 종목")
    print(f"단기 ↔ 통합 overlap: {len(short_ens_overlap)} 종목")
    print(f"장기 ↔ 통합 overlap: {len(long_ens_overlap)} 종목")
    print(f"단기 ↔ 장기 ↔ 통합 overlap: {len(all_overlap)} 종목")
    print()

    # 상관계수 분석
    print("📈 점수 상관성 분석")
    print("-" * 30)

    correlation_short_long = df_date["score_short"].corr(df_date["score_long"])
    correlation_short_ens = df_date["score_short"].corr(df_date["score_ens"])
    correlation_long_ens = df_date["score_long"].corr(df_date["score_ens"])

    print(".4f")
    print(".4f")
    print(".4f")
    print()

    # 평균 점수 비교
    print("📊 평균 점수 비교")
    print("-" * 25)

    avg_short = df_date["score_short"].mean()
    avg_long = df_date["score_long"].mean()
    avg_ens = df_date["score_ens"].mean()

    print(".6f")
    print(".6f")
    print(".6f")
    print()

    # 변동성 분석
    print("📉 점수 변동성 분석")
    print("-" * 25)

    std_short = df_date["score_short"].std()
    std_long = df_date["score_long"].std()
    std_ens = df_date["score_ens"].std()

    print(".6f")
    print(".6f")
    print(".6f")
    print()

    # 주요 차이점 분석
    print("🎯 주요 차이점 분석")
    print("-" * 30)

    # 단기 랭킹에서만 있는 종목
    only_short = short_tickers - long_tickers - ens_tickers
    print(f"단기 랭킹 only: {len(only_short)} 종목")
    if only_short:
        print(f"  종목: {list(only_short)}")

    # 장기 랭킹에서만 있는 종목
    only_long = long_tickers - short_tickers - ens_tickers
    print(f"장기 랭킹 only: {len(only_long)} 종목")
    if only_long:
        print(f"  종목: {list(only_long)}")

    # 통합 랭킹에서만 있는 종목
    only_ens = ens_tickers - short_tickers - long_tickers
    print(f"통합 랭킹 only: {len(only_ens)} 종목")
    if only_ens:
        print(f"  종목: {list(only_ens)}")

    print()

    # 결론
    print("🎉 결론")
    print("-" * 15)

    if len(all_overlap) >= 7:  # 70% 이상 overlap
        print("✅ 단기/장기/통합 랭킹이 매우 유사함")
        print("   → 5:5 결합이 효과적으로 작동")
        print("   → 전략 차별화에 한계 존재")
    else:
        print("⚠️ 랭킹 간 차이가 있음")
        print("   → 전략별 특성 활용 가능")

    print()
    print("💡 통합 랭킹은 단기+장기 균형을 잘 반영하고 있지만,")
    print("   전략별 차별화를 위해서는 파라미터 다양화 필요!")


if __name__ == "__main__":
    analyze_ranking_stocks()
