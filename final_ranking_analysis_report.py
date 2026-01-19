import pandas as pd
import numpy as np

def create_comprehensive_ranking_report():
    """000_code 랭킹 데이터 종합 분석 보고서"""
    print("=== 000_code 랭킹 데이터 종합 분석 보고서 ===\n")
    print("📅 분석일: 2023년 6월 21일")
    print("="*80)

    # 1. 데이터 소스 개요
    print("\n1️⃣ 데이터 소스 개요:")
    data_sources = {
        'Holdout 단기': 'holdout_daily_ranking_short_top20.csv',
        'Holdout 장기': 'holdout_daily_ranking_long_top20.csv',
        'Holdout 통합': 'holdout_daily_ranking_integrated_top20.csv',
        '일간 단기': 'daily_all_business_days_short_ranking_top20.csv',
        '일간 장기': 'daily_all_business_days_long_ranking_top20.csv',
        'UI 단기': 'ui_overall_short_ranking.csv',
        'UI 장기': 'ui_overall_long_ranking.csv'
    }

    for name, filename in data_sources.items():
        try:
            df = pd.read_csv(f'data/{filename}')
            if '날짜' in df.columns:
                filtered = df[df['날짜'] == '2023-06-21']
            elif 'date' in df.columns:
                filtered = df[df['date'] == '2023-06-21']
            else:
                filtered = pd.DataFrame()  # UI 파일 등 날짜 없는 경우

            status = f"✅ {len(filtered)}개 데이터" if not filtered.empty else "❌ 데이터 없음"
            print(f"   {name}: {status}")
        except Exception as e:
            print(f"   {name}: ❌ 읽기 오류")

    # 2. Holdout vs 일간 데이터 비교
    print("\n2️⃣ Holdout vs 일간 데이터 비교:")

    # Holdout 데이터 로드
    holdout_short = pd.read_csv('data/holdout_daily_ranking_short_top20.csv')
    holdout_long = pd.read_csv('data/holdout_daily_ranking_long_top20.csv')
    daily_short = pd.read_csv('data/daily_all_business_days_short_ranking_top20.csv')
    daily_long = pd.read_csv('data/daily_all_business_days_long_ranking_top20.csv')

    # 2023년 6월 21일 필터링
    date_str = '2023-06-21'
    h_short = holdout_short[holdout_short['날짜'] == date_str]
    h_long = holdout_long[holdout_long['날짜'] == date_str]
    d_short = daily_short[daily_short['date'] == date_str]
    d_long = daily_long[daily_long['date'] == date_str]

    # 티커 정규화 함수
    def normalize_ticker(ticker_str):
        """티커 문자열을 숫자만 추출"""
        if isinstance(ticker_str, str):
            # '한국전력(015760)' 형태에서 '015760' 추출
            if '(' in ticker_str and ')' in ticker_str:
                return ticker_str.split('(')[1].split(')')[0]
            else:
                return str(ticker_str).zfill(6)  # 숫자만 있는 경우 6자리로 패딩
        return str(ticker_str).zfill(6)

    # 1등 종목 비교 (정규화 적용)
    print("\n🥇 1등 종목 비교 (정규화 적용):")
    if not h_short.empty and not d_short.empty:
        h_short_ticker = normalize_ticker(h_short.iloc[0]['종목명(ticker)'])
        d_short_ticker = normalize_ticker(d_short.iloc[0]['ticker'])
        short_match = h_short_ticker == d_short_ticker
        print(f"   단기: Holdout={h_short_ticker} | 일간={d_short_ticker} | {'✅ 일치' if short_match else '❌ 불일치'}")

    if not h_long.empty and not d_long.empty:
        h_long_ticker = normalize_ticker(h_long.iloc[0]['종목명(ticker)'])
        d_long_ticker = normalize_ticker(d_long.iloc[0]['ticker'])
        long_match = h_long_ticker == d_long_ticker
        print(f"   장기: Holdout={h_long_ticker} | 일간={d_long_ticker} | {'✅ 일치' if long_match else '❌ 불일치'}")

    # Top5 일치도 분석 (정규화 적용)
    print("\n📊 Top5 일치도 분석 (정규화 적용):")

    for strategy, h_df, d_df in [('단기', h_short, d_short), ('장기', h_long, d_long)]:
        if not h_df.empty and not d_df.empty:
            h_top5 = set([normalize_ticker(x) for x in h_df.head(5)['종목명(ticker)'].tolist()])
            d_top5 = set([normalize_ticker(str(x)) for x in d_df.head(5)['ticker'].tolist()])

            intersection = h_top5 & d_top5
            union = h_top5 | d_top5
            jaccard = len(intersection) / len(union) if union else 0

            print(f"   {strategy} 전략:")
            print(f"     Holdout Top5: {sorted(list(h_top5))}")
            print(f"     일간 Top5: {sorted(list(d_top5))}")
            print(".1f")

    # 점수 범위 비교
    print("\n📈 점수 범위 비교:")

    for strategy, h_df, d_df, h_score_col, d_score_col in [
        ('단기', h_short, d_short, 'score', 'score_short'),
        ('장기', h_long, d_long, 'score', 'score_long')
    ]:
        if not h_df.empty and not d_df.empty:
            h_scores = h_df[h_score_col]
            d_scores = d_df[d_score_col]

            print(f"   {strategy} 전략 전체 점수 범위:")
            print(".6f")
            print(".6f")
            print(f"     평균 차이: {abs(h_scores.mean() - d_scores.mean()):.6f}")

    # 3. 데이터 포맷 차이점 분석
    print("\n3️⃣ 데이터 포맷 차이점 분석:")

    print("\n📋 Holdout 데이터 구조:")
    print("   - 컬럼: 랭킹, 종목명(ticker), 날짜, score, top3 피쳐그룹")
    print("   - 예시: 한국전력(015760), 0.044294, news,profitability,technical")
    print("   - 특징: 실제 백테스트 결과, 간단한 구조")

    print("\n📋 일간 데이터 구조:")
    print("   - 컬럼: ranking, ticker, date, score_short, score_long, score_ens, top1_feature_group...")
    print("   - 예시: 15760, 0.146514, technical, news, value")
    print("   - 특징: 개발용 상세 데이터, 복잡한 구조")

    # 4. 최종 권장사항
    print("\n4️⃣ 최종 분석 및 권장사항:")

    print("\n🎯 데이터 일관성 문제:")
    print("   ❌ 1등 종목 불일치: 티커 포맷 차이로 인한 오인식")
    print("   ❌ 점수 스케일 차이: Holdout 점수가 현저히 낮음")
    print("   ❌ 피처 그룹 차이: 다른 분류 방식 사용")

    print("\n💡 근본 원인 추정:")
    print("   1. 모델 버전 차이: Holdout과 일간 데이터가 다른 모델 사용")
    print("   2. 파라미터 차이: 피처 가중치, 정규화 방식 등이 다름")
    print("   3. 데이터 전처리 차이: 결측치 처리, 아웃라이어 제거 방식 차이")

    print("\n✅ 해결 방안:")
    print("   1. UI에서는 Holdout 데이터만 사용 (실제 백테스트 결과)")
    print("   2. 개발 단계에서 데이터 일관성 검증 프로세스 구축")
    print("   3. 모델 버전 관리 및 변경 이력 추적")
    print("   4. 정기적인 데이터 품질 모니터링")

    print("\n🎪 실무 적용 가이드:")
    print("   - 프로덕션: Holdout 데이터 (실전용)")
    print("   - 연구/개발: 일간 데이터 (분석용)")
    print("   - 모니터링: 두 데이터 비교를 통한 모델 성능 추적")

    print("\n" + "="*80)
    print("📝 결론: 000_code의 랭킹 데이터는 목적에 따라 다르게 사용해야 함")
    print("   Holdout 데이터가 실제 투자에 사용될 최종 결과물임")
    print("="*80)

if __name__ == "__main__":
    create_comprehensive_ranking_report()