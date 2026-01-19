import pandas as pd
import numpy as np

def compare_specific_date_rankings(date_str='2023-06-21'):
    """2023년 6월 21일 특정 날짜의 랭킹 상세 비교"""
    print(f"=== {date_str} 랭킹 상세 비교 분석 ===\n")

    # Holdout 데이터 로드
    holdout_short = pd.read_csv('data/holdout_daily_ranking_short_top20.csv')
    holdout_long = pd.read_csv('data/holdout_daily_ranking_long_top20.csv')
    holdout_integrated = pd.read_csv('data/holdout_daily_ranking_integrated_top20.csv')

    # 일간 데이터 로드
    daily_short = pd.read_csv('data/daily_all_business_days_short_ranking_top20.csv')
    daily_long = pd.read_csv('data/daily_all_business_days_long_ranking_top20.csv')

    # 날짜 필터링 (컬럼명 차이 처리)
    holdout_short_filtered = holdout_short[holdout_short['날짜'] == date_str]
    holdout_long_filtered = holdout_long[holdout_long['날짜'] == date_str]
    holdout_integrated_filtered = holdout_integrated[holdout_integrated['날짜'] == date_str]
    daily_short_filtered = daily_short[daily_short['date'] == date_str]
    daily_long_filtered = daily_long[daily_long['date'] == date_str]

    print(f"📊 데이터 건수:")
    print(f"  Holdout 단기: {len(holdout_short_filtered)}개")
    print(f"  Holdout 장기: {len(holdout_long_filtered)}개")
    print(f"  Holdout 통합: {len(holdout_integrated_filtered)}개")
    print(f"  일간 단기: {len(daily_short_filtered)}개")
    print(f"  일간 장기: {len(daily_long_filtered)}개\n")

    # 1등 종목 비교
    print("🥇 1등 종목 비교:")
    comparisons = []

    if not holdout_short_filtered.empty and not daily_short_filtered.empty:
        h_short_1st = holdout_short_filtered.iloc[0]
        d_short_1st = daily_short_filtered.iloc[0]
        short_match = h_short_1st['종목명(ticker)'] == str(d_short_1st['ticker'])
        comparisons.append(('단기', h_short_1st['종목명(ticker)'], str(d_short_1st['ticker']), short_match))

    if not holdout_long_filtered.empty and not daily_long_filtered.empty:
        h_long_1st = holdout_long_filtered.iloc[0]
        d_long_1st = daily_long_filtered.iloc[0]
        long_match = h_long_1st['종목명(ticker)'] == str(d_long_1st['ticker'])
        comparisons.append(('장기', h_long_1st['종목명(ticker)'], str(d_long_1st['ticker']), long_match))

    for strategy, holdout_ticker, daily_ticker, match in comparisons:
        status = "✅ 일치" if match else "❌ 불일치"
        print(f"  {strategy}: Holdout={holdout_ticker} | 일간={daily_ticker} | {status}")

    print()

    # Top5 일치도 분석
    print("📊 Top5 종목 일치도 분석:")

    for strategy, holdout_df, daily_df in [
        ('단기', holdout_short_filtered, daily_short_filtered),
        ('장기', holdout_long_filtered, daily_long_filtered)
    ]:
        if not holdout_df.empty and not daily_df.empty:
            holdout_top5 = set(holdout_df.head(5)['종목명(ticker)'].tolist())
            daily_top5 = set(daily_df.head(5)['ticker'].tolist())

            intersection = holdout_top5 & daily_top5
            union = holdout_top5 | daily_top5
            jaccard = len(intersection) / len(union) if union else 0

            print(f"  {strategy} 전략:")
            print(f"    Holdout Top5: {sorted(holdout_top5)}")
            print(f"    일간 Top5: {sorted(daily_top5)}")
            print(".1f")

    # 점수 분포 비교
    print("\n📈 점수 분포 비교:")

    for strategy, holdout_df, daily_df, score_col, daily_score_col in [
        ('단기', holdout_short_filtered, daily_short_filtered, 'score', 'score_short'),
        ('장기', holdout_long_filtered, daily_long_filtered, 'score', 'score_long')
    ]:
        if not holdout_df.empty and not daily_df.empty:
            h_scores = holdout_df.head(5)[score_col]
            d_scores = daily_df.head(5)[daily_score_col]

            print(f"  {strategy} 전략 Top5 점수:")
            print(f"    Holdout: {h_scores.mean():.6f} (min: {h_scores.min():.6f}, max: {h_scores.max():.6f})")
            print(f"    일간: {d_scores.mean():.6f} (min: {d_scores.min():.6f}, max: {d_scores.max():.6f})")
            print(f"    차이: {abs(h_scores.mean() - d_scores.mean()):.6f}")

    # 피처 그룹 비교 (간단 버전)
    print("\n🎯 피처 그룹 비교:")

    for strategy, holdout_df, daily_df in [
        ('단기', holdout_short_filtered, daily_short_filtered),
        ('장기', holdout_long_filtered, daily_long_filtered)
    ]:
        if not holdout_df.empty and not daily_df.empty:
            # Holdout 피처 그룹
            h_features_raw = holdout_df.head(5)['top3 피쳐그룹'].tolist()
            print(f"  {strategy} 전략 Top5 피처 그룹:")
            print(f"    Holdout: {h_features_raw[:3]}...")  # 처음 3개만 표시
            print(f"    일간: top1/top2/top3_feature_group 컬럼 사용")

    print("\n" + "="*60)
    print("🎯 최종 분석 결과:")
    print("1. 1등 종목 일치 여부:")
    for strategy, holdout_ticker, daily_ticker, match in comparisons:
        if match:
            print(f"   ✅ {strategy} 전략: 완전 일치 ({holdout_ticker})")
        else:
            print(f"   ❌ {strategy} 전략: 불일치 (Holdout: {holdout_ticker} vs 일간: {daily_ticker})")

    print("\n2. 데이터 차이점:")
    print("   - Holdout: 실제 백테스트 결과 (실전용)")
    print("   - 일간: 모든 영업일 계산 결과 (개발용)")

    print("\n3. UI 사용 권장사항:")
    print("   - 실시간 서비스: Holdout 데이터 사용")
    print("   - 과거 분석: 일간 데이터 활용")
    print("   - 개발 단계: 두 데이터 비교 검증")

if __name__ == "__main__":
    compare_specific_date_rankings('2023-06-21')