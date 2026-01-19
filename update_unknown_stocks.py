#!/usr/bin/env python3
"""
Unknown 종목명을 수동으로 추가하여 100% 매칭 확인
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List
import sys

# 프로젝트 경로 설정
project_root = Path("C:/Users/seong/OneDrive/Desktop/bootcamp/000_code")

# 사용자가 제공한 종목명 매핑 데이터
provided_mappings = """
강원랜드(035250),LONG
농심(004370),LONG
CJ(001040),LONG
SK텔레콤(017670),LONG
삼성카드(029780),LONG
삼성물산(028260),LONG
LG유플러스(032640),LONG
KT(030200),LONG
CJ제일제당(097950),LONG
휠라홀딩스(030000),LONG
한국앤컴퍼니(000240),LONG
유한양행(000100),LONG
한국가스공사(036460),LONG
한국전력(015760),LONG
SK스퀘어(326030),LONG
GS(078930),LONG
한샘(009240),LONG
한화(000880),LONG
하이트진로(000080),LONG
HD현대(375500),LONG
녹십자(006280),LONG
SK이노베이션(096770),LONG
대웅제약(003090),LONG
현대백화점(069960),LONG
대웅(069620),LONG
대한제당(001440),LONG
KCC(002380),LONG
SNT모티브(139480),LONG
롯데지주(004990),LONG
한국금융지주(071050),LONG
대림비앤코(000210),LONG
보령(280360),LONG
롯데쇼핑(023530),LONG
이노션(241590),LONG
호텔신라(026960),LONG
CJ대한통운(000120),LONG
대한항공(003490),LONG
신세계(031430),LONG
롯데칠성(032350),LONG
현대두산인프라코어(267250),LONG
제일기획(251270),LONG
LIG넥스원(079550),LONG
게임빌(039130),LONG
아모레퍼시픽(090430),LONG
신세계푸드(004170),LONG
오뚜기(007310),LONG
하이트진로홀딩스(284740),LONG
기업은행(024110),LONG
GS건설(006360),LONG
강원랜드(192080),LONG
에스원(007070),LONG
효성(004800),LONG
BNK금융지주(013890),LONG
강원랜드(020000),LONG
에이비엘바이오(294870),LONG
영풍(000670),LONG
LG에너지솔루션(373220),LONG
금호석유(011170),LONG
BGF리테일(282330),LONG
HD현대일렉트릭(336260),LONG
롯데제과(005300),LONG
씨에스윈드(112610),LONG
KB금융(105560),LONG
대우조선해양(042660),LONG
현대제철(004020),LONG
솔루엠(381970),LONG
현대홈쇼핑(057050),LONG
우리금융지주(316140),LONG
BNK금융지주(138930),LONG
DB손해보험(005830),LONG
LG전자(066570),LONG
삼양홀딩스(001800),LONG
쌍용C&E(003410),LONG
한화생명(088350),LONG
미래에셋증권(006800),LONG
제일기획(300720),LONG
아이에스동서(010780),LONG
하나제약(093370),LONG
씨에스베어링(178920),LONG
동북아12호선박투자(114090),LONG
효성티앤씨(298020),LONG
메리츠금융지주(069260),LONG
호텔신라(008770),LONG
대우건설(047040),LONG
코오롱인더(120110),LONG
현대비앤지스틸(016380),LONG
OCI(009900),LONG
동양(000990),LONG
롯데정밀화학(004000),LONG
SK바이오사이언스(302440),LONG
SK케미칼(285130),LONG
KT&G(033780),LONG
SK네트웍스(001740),LONG
DGB금융지주(139130),LONG
OCI(010060),LONG
현대글로비스(011210),LONG
S-Oil(010950),LONG
오리온(271560),LONG
삼성에스디에스(018260),LONG
현대건설(000720),LONG
영원무역(009970),LONG
한국전력(015760),SHORT
한국가스공사(036460),SHORT
CJ(001040),SHORT
CJ제일제당(097950),SHORT
SK텔레콤(017670),SHORT
KT(030200),SHORT
휠라홀딩스(030000),SHORT
농심(004370),SHORT
삼성물산(028260),SHORT
강원랜드(035250),SHORT
삼성카드(029780),SHORT
LG유플러스(032640),SHORT
유한양행(000100),SHORT
한국앤컴퍼니(000240),SHORT
대웅(069620),SHORT
하이트진로(000080),SHORT
HD현대(375500),SHORT
GS(078930),SHORT
SK이노베이션(096770),SHORT
한샘(009240),SHORT
SK스퀘어(326030),SHORT
대웅제약(003090),SHORT
현대백화점(069960),SHORT
녹십자(006280),SHORT
한화(000880),SHORT
대한제당(001440),SHORT
KCC(002380),SHORT
롯데쇼핑(023530),SHORT
한국금융지주(071050),SHORT
롯데지주(004990),SHORT
대림비앤코(000210),SHORT
이노션(241590),SHORT
보령(280360),SHORT
SNT모티브(139480),SHORT
신세계(031430),SHORT
롯데칠성(032350),SHORT
CJ대한통운(000120),SHORT
호텔신라(026960),SHORT
대한항공(003490),SHORT
제일기획(251270),SHORT
LIG넥스원(079550),SHORT
현대두산인프라코어(267250),SHORT
신세계푸드(004170),SHORT
게임빌(039130),SHORT
하이트진로홀딩스(284740),SHORT
아모레퍼시픽(090430),SHORT
오뚜기(007310),SHORT
에스원(007070),SHORT
GS건설(006360),SHORT
효성(004800),SHORT
BNK금융지주(013890),SHORT
강원랜드(192080),SHORT
에이비엘바이오(294870),SHORT
강원랜드(020000),SHORT
기업은행(024110),SHORT
금호석유(011170),SHORT
영풍(000670),SHORT
LG에너지솔루션(373220),SHORT
BGF리테일(282330),SHORT
롯데제과(005300),SHORT
씨에스윈드(112610),SHORT
HD현대일렉트릭(336260),SHORT
대우조선해양(042660),SHORT
KB금융(105560),SHORT
솔루엠(381970),SHORT
현대홈쇼핑(057050),SHORT
현대제철(004020),SHORT
우리금융지주(316140),SHORT
BNK금융지주(138930),SHORT
DB손해보험(005830),SHORT
쌍용C&E(003410),SHORT
삼양홀딩스(001800),SHORT
LG전자(066570),SHORT
제일기획(300720),SHORT
아이에스동서(010780),SHORT
하나제약(093370),SHORT
씨에스베어링(178920),SHORT
동북아12호선박투자(114090),SHORT
효성티앤씨(298020),SHORT
메리츠금융지주(069260),SHORT
호텔신라(008770),SHORT
대우건설(047040),SHORT
코오롱인더(120110),SHORT
현대비앤지스틸(016380),SHORT
OCI(009900),SHORT
동양(000990),SHORT
롯데정밀화학(004000),SHORT
SK바이오사이언스(302440),SHORT
SK네트웍스(001740),SHORT
SK케미칼(285130),SHORT
KT&G(033780),SHORT
DGB금융지주(139130),SHORT
OCI(010060),SHORT
현대글로비스(011210),SHORT
S-Oil(010950),SHORT
오리온(271560),SHORT
삼성에스디에스(018260),SHORT
영원무역(009970),SHORT
현대건설(000720),SHORT
"""


def parse_mappings_to_dict() -> Dict[str, str]:
    """제공된 매핑 데이터를 딕셔너리로 파싱"""
    mappings = {}
    lines = [line.strip() for line in provided_mappings.strip().split('\n') if line.strip()]

    for line in lines:
        if line and ',' in line:
            company_ticker, strategy = line.split(',', 1)
            company_ticker = company_ticker.strip()

            # 회사명과 티커 분리
            if '(' in company_ticker and ')' in company_ticker:
                company_name = company_ticker.split('(')[0].strip()
                ticker = company_ticker.split('(')[1].split(')')[0].strip()

                # 티커를 6자리로 포맷팅
                ticker_formatted = f"{int(ticker):06d}"
                key = f"Unknown({ticker_formatted})"

                mappings[key] = company_name

    return mappings


def update_unknown_stocks():
    """Unknown 종목명을 업데이트하고 100% 매칭 확인"""

    print("=== Unknown 종목명 업데이트 시작 ===")

    # 제공된 매핑 데이터 파싱
    provided_mappings = parse_mappings_to_dict()
    print(f"제공된 매핑 수: {len(provided_mappings)}")

    # 기존 Unknown 목록 로드
    unknown_file = project_root / "data" / "all_unknown_stocks_for_hardcoding.csv"
    unknown_df = pd.read_csv(unknown_file)
    print(f"기존 Unknown 종목 수: {len(unknown_df)}")

    # 중복 제거된 Unknown 목록 생성
    unique_unknown = unknown_df[['종목명(ticker)']].drop_duplicates()
    print(f"고유 Unknown 종목 수: {len(unique_unknown)}")

    # 매칭 확인
    matched = 0
    unmatched = []
    mapping_results = []

    for _, row in unique_unknown.iterrows():
        unknown_key = row['종목명(ticker)']

        if unknown_key in provided_mappings:
            company_name = provided_mappings[unknown_key]
            matched += 1
            mapping_results.append({
                '원본': unknown_key,
                '매핑결과': f"{company_name}({unknown_key.split('(')[1].split(')')[0]})",
                '상태': '✅ 매칭됨'
            })
        else:
            unmatched.append(unknown_key)
            mapping_results.append({
                '원본': unknown_key,
                '매핑결과': '매핑되지 않음',
                '상태': '❌ 미매칭'
            })

    # 결과 출력
    print(f"\n=== 매칭 결과 ===")
    print(f"매칭된 종목: {matched}/{len(unique_unknown)} ({matched/len(unique_unknown)*100:.1f}%)")

    if unmatched:
        print(f"미매칭 종목: {len(unmatched)}개")
        print("\n미매칭 목록:")
        for item in unmatched[:10]:  # 처음 10개만
            print(f"  - {item}")
        if len(unmatched) > 10:
            print(f"  ... 외 {len(unmatched)-10}개")
    else:
        print("🎉 모든 Unknown 종목이 100% 매칭되었습니다!")

    # 상세 결과 저장
    results_df = pd.DataFrame(mapping_results)
    results_file = project_root / "data" / "unknown_stocks_mapping_results.csv"
    results_df.to_csv(results_file, index=False, encoding='utf-8-sig')
    print(f"\n상세 결과 파일: {results_file}")

    # 100% 매칭 시 최종 업데이트된 종목명 딕셔너리 생성
    if matched == len(unique_unknown):
        print("\n=== 최종 종목명 딕셔너리 생성 ===")

        # 기존 티커 매핑에 Unknown 매핑 추가
        final_ticker_mapping = {
            '005930': '삼성전자', '000660': 'SK하이닉스', '035420': 'NAVER', '034730': 'SK텔레콤',
            '005380': '현대차', '000270': '기아', '035720': '카카오', '005490': 'POSCO홀딩스',
            '051910': 'LG화학', '012330': '현대모비스', '055550': '신한지주', '032830': '삼성생명',
            '003550': 'LG', '006400': '삼성SDI', '086790': '하나금융지주', '138040': '메리츠금융지주',
            '036570': '엔씨소프트', '000810': '삼성화재', '009150': '삼성전기', '034730': 'SK',
            '352820': '하이브', '011200': 'HMM', '010130': '고려아연', '009830': '한화솔루션',
            '241560': '두산밥캣', '137310': '에스디바이오센서', '003240': '태광산업'
        }

        # Unknown 매핑 추가
        for unknown_key, company_name in provided_mappings.items():
            ticker = unknown_key.split('(')[1].split(')')[0]
            final_ticker_mapping[ticker] = company_name

        # 딕셔너리 파일로 저장
        dict_file = project_root / "data" / "final_ticker_mapping.py"
        with open(dict_file, 'w', encoding='utf-8') as f:
            f.write("# 최종 티커-종목명 매핑 딕셔너리 (100% 완성)\n")
            f.write("ticker_to_name = {\n")
            for ticker, name in sorted(final_ticker_mapping.items()):
                f.write(f"    '{ticker}': '{name}',\n")
            f.write("}\n")

        print(f"최종 매핑 딕셔너리 파일 생성: {dict_file}")
        print(f"총 매핑 종목 수: {len(final_ticker_mapping)}")

        # Holdout 파일들 업데이트
        update_holdout_files(final_ticker_mapping)


def update_holdout_files(ticker_mapping: Dict[str, str]):
    """Holdout 랭킹 파일들을 업데이트된 종목명으로 교체"""

    print("\n=== Holdout 파일 업데이트 시작 ===")

    holdout_files = [
        'holdout_daily_ranking_long_top20.csv',
        'holdout_daily_ranking_short_top20.csv'
    ]

    for filename in holdout_files:
        file_path = project_root / "data" / filename

        if file_path.exists():
            print(f"업데이트 중: {filename}")

            # 파일 로드
            df = pd.read_csv(file_path)

            # 종목명 업데이트
            updated_count = 0
            for idx, row in df.iterrows():
                company_ticker = row['종목명(ticker)']
                if 'Unknown(' in company_ticker:
                    ticker = company_ticker.split('(')[1].split(')')[0]
                    if ticker in ticker_mapping:
                        company_name = ticker_mapping[ticker]
                        df.at[idx, '종목명(ticker)'] = f"{company_name}({ticker})"
                        updated_count += 1

            # 파일 저장
            df.to_csv(file_path, index=False, encoding='utf-8-sig')
            print(f"  {updated_count}개 종목명 업데이트됨")

    print("Holdout 파일 업데이트 완료!")


if __name__ == "__main__":
    update_unknown_stocks()