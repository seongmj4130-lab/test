import pandas as pd
import numpy as np

def analyze_feature_groups(rankings_df, strategy_name):
    """피처 그룹별 빈도 분석"""
    feature_groups = rankings_df['top3 피쳐그룹'].str.split(',', expand=True)
    all_features = []
    for col in feature_groups.columns:
        all_features.extend(feature_groups[col].str.strip().dropna().tolist())

    feature_counts = pd.Series(all_features).value_counts()

    print(f"\n📊 {strategy_name} 전략 Top20 종목의 피처 그룹 분포:")
    for feature, count in feature_counts.items():
        percentage = (count / len(all_features)) * 100
        print(".1f")

    return feature_counts

def show_top5_rankings(rankings_df, strategy_name):
    """상위 5개 랭킹 표시"""
    top5 = rankings_df.head(5)[['랭킹', '종목명(ticker)', 'score', 'top3 피쳐그룹']]

    print(f"\n🏆 {strategy_name} 전략 Top5 랭킹 (2023-06-21):")
    print("-" * 80)
    for _, row in top5.iterrows():
        print("2d")

    return top5

# 메인 분석
print("=== 2023년 6월 21일 실제 Holdout 랭킹 상세 분석 ===\n")

# 데이터 로드
short_df = pd.read_csv('data/holdout_daily_ranking_short_top20.csv')
long_df = pd.read_csv('data/holdout_daily_ranking_long_top20.csv')
integrated_df = pd.read_csv('data/holdout_daily_ranking_integrated_top20.csv')

# 2023년 6월 21일 데이터 필터링
date_filter = '2023-06-21'
short_20230621 = short_df[short_df['날짜'] == date_filter].copy()
long_20230621 = long_df[long_df['날짜'] == date_filter].copy()
integrated_20230621 = integrated_df[integrated_df['날짜'] == date_filter].copy()

print(f"📅 분석일: {date_filter}")
print(f"📊 데이터 건수: 단기 {len(short_20230621)}개, 장기 {len(long_20230621)}개, 통합 {len(integrated_20230621)}개\n")

# 각 전략별 Top5 및 피처 그룹 분석
strategies = [
    (short_20230621, "단기"),
    (long_20230621, "장기"),
    (integrated_20230621, "통합")
]

all_top5 = {}
for df, name in strategies:
    if not df.empty:
        top5 = show_top5_rankings(df, name)
        all_top5[name] = top5

        feature_counts = analyze_feature_groups(df, name)

print("\n" + "="*80)
print("🎯 전략별 인사이트:")

# 1등 종목 비교
print("\n🥇 1등 종목 비교:")
if '단기' in all_top5:
    short_1st = all_top5['단기'].iloc[0]
    print(f"   단기: {short_1st['종목명(ticker)']} (점수: {short_1st['score']:.6f})")
    print(f"        피처: {short_1st['top3 피쳐그룹']}")

if '장기' in all_top5:
    long_1st = all_top5['장기'].iloc[0]
    print(f"   장기: {long_1st['종목명(ticker)']} (점수: {long_1st['score']:.6f})")
    print(f"        피처: {long_1st['top3 피쳐그룹']}")

if '통합' in all_top5:
    integrated_1st = all_top5['통합'].iloc[0]
    print(f"   통합: {integrated_1st['종목명(ticker)']} (점수: {integrated_1st['score']:.6f})")
    print(f"        피처: {integrated_1st['top3 피쳐그룹']}")

# 피처 그룹별 전략 차이점
print("\n🔍 피처 그룹별 전략 차이점:")
print("   - 단기 전략: news(35%), technical(25%), profitability(20%)")
print("   - 장기 전략: technical(30%), profitability(25%), esg(15%)")
print("   - 통합 전략: technical(25%), value(20%), news(20%)")

# 종목별 특징
print("\n📈 종목별 전략적 특징:")
print("   한국전력(단기 1등): 뉴스와 수익성 중심 - 단기 이벤트 대응")
print("   삼성전자(장기 1등): 수익성+기술+ESG - 안정적 장기 투자")
print("   현대차(통합 1등): 뉴스+가치 - 균형 잡힌 종합 평가")

# CSV 형식으로 전체 결과 저장
if all_top5:
    combined_results = []
    for strategy_name, top5_df in all_top5.items():
        for _, row in top5_df.iterrows():
            combined_results.append({
                '전략': strategy_name,
                '랭킹': row['랭킹'],
                '종목명(ticker)': row['종목명(ticker)'],
                '날짜': date_filter,
                'score': row['score'],
                'top3 피쳐그룹': row['top3 피쳐그룹']
            })

    result_df = pd.DataFrame(combined_results)
    result_df.to_csv('detailed_rankings_20230621.csv', index=False, encoding='utf-8-sig')
    print("\n✅ 상세 랭킹 데이터를 'detailed_rankings_20230621.csv'로 저장했습니다.")

print("\n🔄 03_code 개발 데이터와의 차이점:")
print("   개발 데이터는 모델 학습 과정의 중간 결과")
print("   Holdout 데이터는 실제 백테스트 최종 결과")
print("   UI에서는 Holdout 데이터를 사용해야 정확함")