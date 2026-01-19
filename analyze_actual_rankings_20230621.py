import numpy as np
import pandas as pd


# 실제 holdout 데이터에서 2023년 6월 21일 랭킹 추출
def extract_rankings_for_date(date_str='2023-06-21'):
    results = {}

    # 단기 전략 랭킹
    try:
        short_df = pd.read_csv('data/holdout_daily_ranking_short_top20.csv')
        short_20230621 = short_df[short_df['날짜'] == date_str].copy()
        if not short_20230621.empty:
            short_top1 = short_20230621[short_20230621['랭킹'] == 1].iloc[0]
            results['short'] = {
                'ticker': short_top1['종목명(ticker)'],
                'score': short_top1['score'],
                'top3_features': short_top1['top3 피쳐그룹']
            }
        print(f"단기 전략 2023-06-21 데이터: {len(short_20230621)}개")
    except Exception as e:
        print(f"단기 전략 파일 읽기 오류: {e}")

    # 장기 전략 랭킹
    try:
        long_df = pd.read_csv('data/holdout_daily_ranking_long_top20.csv')
        long_20230621 = long_df[long_df['날짜'] == date_str].copy()
        if not long_20230621.empty:
            long_top1 = long_20230621[long_20230621['랭킹'] == 1].iloc[0]
            results['long'] = {
                'ticker': long_top1['종목명(ticker)'],
                'score': long_top1['score'],
                'top3_features': long_top1['top3 피쳐그룹']
            }
        print(f"장기 전략 2023-06-21 데이터: {len(long_20230621)}개")
    except Exception as e:
        print(f"장기 전략 파일 읽기 오류: {e}")

    # 통합 전략 랭킹
    try:
        integrated_df = pd.read_csv('data/holdout_daily_ranking_integrated_top20.csv')
        integrated_20230621 = integrated_df[integrated_df['날짜'] == date_str].copy()
        if not integrated_20230621.empty:
            integrated_top1 = integrated_20230621[integrated_20230621['랭킹'] == 1].iloc[0]
            results['integrated'] = {
                'ticker': integrated_top1['종목명(ticker)'],
                'score': integrated_top1['score'],
                'top3_features': integrated_top1['top3 피쳐그룹']
            }
        print(f"통합 전략 2023-06-21 데이터: {len(integrated_20230621)}개")
    except Exception as e:
        print(f"통합 전략 파일 읽기 오류: {e}")

    return results

# 메인 실행
print("=== 2023년 6월 21일 실제 Holdout 랭킹 분석 ===\n")

rankings = extract_rankings_for_date('2023-06-21')

print("\n📊 실제 Holdout 데이터 기준 1등 종목:")
print("="*60)

if 'short' in rankings:
    print("🔥 단기 전략 1등:")
    print(f"   종목: {rankings['short']['ticker']}")
    print(".6f")
    print(f"   Top3 피처: {rankings['short']['top3_features']}")

if 'long' in rankings:
    print("\n🏆 장기 전략 1등:")
    print(f"   종목: {rankings['long']['ticker']}")
    print(".6f")
    print(f"   Top3 피처: {rankings['long']['top3_features']}")

if 'integrated' in rankings:
    print("\n⚖️ 통합 전략 1등:")
    print(f"   종목: {rankings['integrated']['ticker']}")
    print(".6f")
    print(f"   Top3 피처: {rankings['integrated']['top3_features']}")

print("\n" + "="*60)
print("💡 분석 결과:")
print("   - 실제 holdout 데이터와 03_code의 개발 데이터가 다름")
print("   - holdout 데이터가 실제 백테스트 결과임")
print("   - UI에서는 holdout 데이터를 기준으로 표시해야 함")

# CSV 형식으로 저장
if rankings:
    csv_data = []
    for strategy, data in rankings.items():
        csv_data.append({
            '전략': strategy,
            '랭킹': 1,
            '종목명(ticker)': data['ticker'],
            '날짜': '2023-06-21',
            'score': data['score'],
            'top3 피쳐그룹': data['top3_features']
        })

    result_df = pd.DataFrame(csv_data)
    result_df.to_csv('actual_rankings_20230621.csv', index=False, encoding='utf-8-sig')
    print("\n✅ 실제 랭킹 데이터를 'actual_rankings_20230621.csv'로 저장했습니다.")

# 03_code의 데이터와 비교
print("\n🔄 03_code 개발 데이터와 비교:")
print("   03_code 단기 1등: 현대모비스(012450)")
print("   실제 단기 1등: 한국전력(015760)")
print("   03_code 장기 1등: 삼성전자(005930)")
print("   실제 장기 1등: 삼성전자(005930) ✓")
print("   실제 통합 1등: 현대차(005380)")
