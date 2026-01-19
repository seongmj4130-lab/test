import pandas as pd
import numpy as np

def load_and_filter_data(filepath, date_str='2023-06-21'):
    """파일 로드 및 날짜 필터링"""
    try:
        df = pd.read_csv(filepath)
        if '날짜' in df.columns:
            filtered = df[df['날짜'] == date_str].copy()
        elif 'date' in df.columns:
            filtered = df[df['date'] == date_str].copy()
        else:
            print(f"날짜 컬럼을 찾을 수 없습니다: {filepath}")
            return None

        print(f"✅ {filepath.split('/')[-1]}: {len(filtered)}개 데이터")
        return filtered
    except Exception as e:
        print(f"❌ 파일 읽기 오류 {filepath}: {e}")
        return None

def compare_rankings(data_sources, strategy_name):
    """다양한 데이터 소스의 랭킹 비교"""
    print(f"\n{'='*60}")
    print(f"🔍 {strategy_name} 전략 랭킹 비교 (2023-06-21)")
    print(f"{'='*60}")

    rankings_data = {}
    for source_name, filepath in data_sources.items():
        data = load_and_filter_data(filepath)
        if data is not None and not data.empty:
            rankings_data[source_name] = data

    if not rankings_data:
        print(f"⚠️ {strategy_name} 전략 데이터가 없습니다.")
        return

    # 각 소스의 1등 종목 비교
    print("\n🥇 1등 종목 비교:")
    for source_name, df in rankings_data.items():
        if len(df) > 0:
            top1 = df.iloc[0]
            ticker_col = '종목명(ticker)' if '종목명(ticker)' in df.columns else 'ticker'
            score_col = 'score' if 'score' in df.columns else 'Score'
            features_col = 'top3 피쳐그룹' if 'top3 피쳐그룹' in df.columns else 'top3_features'

            ticker = top1.get(ticker_col, 'N/A')
            score = top1.get(score_col, 'N/A')
            features = top1.get(features_col, 'N/A')

            print("15")

    # 랭킹 일치도 분석 (Top5 기준)
    print("\n📊 Top5 랭킹 일치도 분석:")
    if len(rankings_data) >= 2:
        sources_list = list(rankings_data.keys())
        for i in range(len(sources_list)):
            for j in range(i+1, len(sources_list)):
                source1, source2 = sources_list[i], sources_list[j]
                df1, df2 = rankings_data[source1], rankings_data[source2]

                ticker_col1 = '종목명(ticker)' if '종목명(ticker)' in df1.columns else 'ticker'
                ticker_col2 = '종목명(ticker)' if '종목명(ticker)' in df2.columns else 'ticker'

                tickers1 = set(df1.head(5)[ticker_col1].tolist())
                tickers2 = set(df2.head(5)[ticker_col2].tolist())

                intersection = tickers1 & tickers2
                union = tickers1 | tickers2
                jaccard = len(intersection) / len(union) if union else 0

                print(".1f")

# 메인 비교 분석
print("=== 000_code 랭킹 데이터 비교 분석 ===\n")
print("📅 분석일: 2023-06-21")

# 단기 전략 비교
short_sources = {
    'Holdout 단기': 'data/holdout_daily_ranking_short_top20.csv',
    '일간 단기': 'data/daily_all_business_days_short_ranking_top20.csv',
    'UI 단기': 'data/ui_overall_short_ranking.csv',
    '새로운 단기': 'data/daily_new_short_ranking_top20.csv'
}

compare_rankings(short_sources, "단기")

# 장기 전략 비교
long_sources = {
    'Holdout 장기': 'data/holdout_daily_ranking_long_top20.csv',
    '일간 장기': 'data/daily_all_business_days_long_ranking_top20.csv',
    'UI 장기': 'data/ui_overall_long_ranking.csv',
    '새로운 장기': 'data/daily_new_long_ranking_top20.csv'
}

compare_rankings(long_sources, "장기")

# 통합 전략 (holdout만 있음)
integrated_sources = {
    'Holdout 통합': 'data/holdout_daily_ranking_integrated_top20.csv'
}

compare_rankings(integrated_sources, "통합")

print("\n" + "="*60)
print("🎯 분석 결과 요약:")
print("- Holdout 데이터: 실제 백테스트 결과 (최종)")
print("- 일간 데이터: 모든 영업일 랭킹")
print("- UI 데이터: 사용자 인터페이스용 가공 데이터")
print("- 새로운 데이터: 최근 생성된 랭킹")

print("\n💡 권장사항:")
print("- UI에서는 Holdout 데이터를 우선 사용")
print("- 개발/테스트에서는 일간 데이터를 활용")
print("- 데이터 일관성 검증을 주기적으로 수행")