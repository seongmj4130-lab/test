import csv

# 종목명 매핑 (주요 KOSPI200 종목들)
ticker_to_name = {
    '005930': '삼성전자',
    '000660': 'SK하이닉스',
    '035420': 'NAVER',
    '207940': '삼성바이오로직스',
    '005380': '현대차',
    '000270': '기아',
    '068270': '셀트리온',
    '035720': '카카오',
    '005490': 'POSCO홀딩스',
    '051910': 'LG화학',
    '012330': '현대모비스',
    '055550': '신한지주',
    '032830': '삼성생명',
    '003550': 'LG',
    '006400': '삼성SDI',
    '086790': '하나금융지주',
    '138040': '메리츠금융지주',
    '036570': '엔씨소프트',
    '000810': '삼성화재',
    '009150': '삼성전기',
    '034730': 'SK',
    '352820': '하이브',
    '011200': 'HMM',
    '010130': '고려아연',
    '009830': '한화솔루션',
    '241560': '두산밥캣',
    '137310': '에스디바이오센서',
    '003240': '태광산업',
    '034730': 'SK텔레콤',
    '005380': '현대자동차',
    '035250': '강원랜드',
    '003240': '태광산업',
    '097950': 'CJ제일제당',
    '017670': 'SK텔레콤',
    '028260': '삼성물산',
    '326030': 'LG유플러스',
    '001040': 'CJ',
    '000240': '한국앤컴퍼니',
    '030200': 'KT',
    '000120': '현대건설',
    '241560': '두산밥캣',
    '015760': '한국전력',
    '029780': '삼성카드',
    '036460': '한국가스공사',
    '012330': '현대모비스',
    '000100': '유한양행'
}

# 파일 읽고 쓰기
with open('data/daily_all_business_days_long_ranking_top20.csv', 'r') as f:
    reader = csv.DictReader(f)

    # 새로운 파일 생성
    with open('data/daily_all_business_days_long_ranking_top20_formatted.csv', 'w', newline='') as out_f:
        fieldnames = ['ranking', 'ticker_formatted', 'company_name', 'date', 'score_short', 'score_long', 'score_ens',
                     'top1_feature_group', 'top2_feature_group', 'top3_feature_group',
                     'top1_features', 'top2_features', 'top3_features', 'original_ranking']
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            ticker = str(row['ticker'])
            # 6자리로 포맷팅
            ticker_formatted = f'{int(ticker):06d}'

            # 종목명 찾기
            company_name = ticker_to_name.get(ticker_formatted, f'Unknown({ticker_formatted})')

            writer.writerow({
                'ranking': row['ranking'],
                'ticker_formatted': ticker_formatted,
                'company_name': company_name,
                'date': row['date'],
                'score_short': row['score_short'],
                'score_long': row['score_long'],
                'score_ens': row['score_ens'],
                'top1_feature_group': row['top1_feature_group'],
                'top2_feature_group': row['top2_feature_group'],
                'top3_feature_group': row['top3_feature_group'],
                'top1_features': row['top1_features'],
                'top2_features': row['top2_features'],
                'top3_features': row['top3_features'],
                'original_ranking': row['original_ranking']
            })

print('포맷팅된 파일이 생성되었습니다: data/daily_all_business_days_long_ranking_top20_formatted.csv')