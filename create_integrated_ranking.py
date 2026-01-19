#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
통합 랭킹 생성 스크립트
단기 랭킹과 장기 랭킹을 5:5 비율로 결합하여 통합 랭킹 생성
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_ranking_data(short_path: str, long_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """단기 및 장기 랭킹 데이터 로드"""
    logger.info(f"단기 랭킹 파일 로드: {short_path}")
    short_df = pd.read_csv(short_path)
    logger.info(f"장기 랭킹 파일 로드: {long_path}")
    long_df = pd.read_csv(long_path)

    logger.info(f"단기 랭킹 shape: {short_df.shape}")
    logger.info(f"장기 랭킹 shape: {long_df.shape}")

    return short_df, long_df

def merge_rankings(short_df: pd.DataFrame, long_df: pd.DataFrame) -> pd.DataFrame:
    """단기와 장기 랭킹을 날짜와 티커 기준으로 병합"""
    # 티커 추출을 위한 컬럼 생성
    short_df['ticker'] = short_df['종목명(ticker)'].str.extract(r'\((\d+)\)')
    long_df['ticker'] = long_df['종목명(ticker)'].str.extract(r'\((\d+)\)')

    # 날짜와 티커 기준으로 병합
    merged_df = pd.merge(
        short_df[['날짜', 'ticker', '종목명(ticker)', 'score', 'top3 피쳐그룹']],
        long_df[['날짜', 'ticker', 'score', 'top3 피쳐그룹']],
        on=['날짜', 'ticker'],
        how='outer',
        suffixes=('_short', '_long')
    )

    logger.info(f"병합된 데이터 shape: {merged_df.shape}")
    return merged_df

def calculate_integrated_score(row) -> float:
    """단기와 장기 스코어를 5:5 비율로 결합"""
    short_score = row['score_short']
    long_score = row['score_long']

    # 둘 다 유효한 경우 5:5 결합
    if pd.notna(short_score) and pd.notna(long_score):
        return short_score * 0.5 + long_score * 0.5
    # 하나만 유효한 경우 해당 값 사용
    elif pd.notna(short_score):
        return short_score * 0.5  # 가중치 반영
    elif pd.notna(long_score):
        return long_score * 0.5  # 가중치 반영
    else:
        return np.nan

def combine_feature_groups(row) -> str:
    """단기와 장기의 피쳐 그룹을 결합"""
    short_groups = row['top3 피쳐그룹_short']
    long_groups = row['top3 피쳐그룹_long']

    all_groups = []

    # 단기 그룹 추가
    if pd.notna(short_groups):
        all_groups.extend([g.strip() for g in short_groups.split(',')])

    # 장기 그룹 추가 (중복 제거)
    if pd.notna(long_groups):
        long_list = [g.strip() for g in long_groups.split(',')]
        for group in long_list:
            if group not in all_groups:
                all_groups.append(group)

    # 최대 3개까지만 선택
    return ','.join(all_groups[:3])

def create_integrated_ranking(merged_df: pd.DataFrame) -> pd.DataFrame:
    """통합 랭킹 생성"""
    # 통합 스코어 계산
    merged_df['score'] = merged_df.apply(calculate_integrated_score, axis=1)

    # 피쳐 그룹 결합
    merged_df['top3 피쳐그룹'] = merged_df.apply(combine_feature_groups, axis=1)

    # 필요한 컬럼만 선택
    result_df = merged_df[['날짜', 'ticker', '종목명(ticker)', 'score', 'top3 피쳐그룹']].copy()

    # NaN 스코어 제거
    result_df = result_df.dropna(subset=['score'])

    # 날짜별로 스코어 기준 내림차순 정렬 및 top20 선정
    result_df['날짜'] = pd.to_datetime(result_df['날짜'])
    result_df = result_df.sort_values(['날짜', 'score'], ascending=[True, False])

    # 날짜별 top20 선정
    top20_dfs = []
    for date, group in result_df.groupby('날짜'):
        top20 = group.head(20).copy()
        top20['랭킹'] = range(1, len(top20) + 1)
        top20_dfs.append(top20)

    final_df = pd.concat(top20_dfs, ignore_index=True)

    # 최종 컬럼 순서 조정
    final_df = final_df[['랭킹', '종목명(ticker)', '날짜', 'score', 'top3 피쳐그룹']]

    # 날짜 포맷 조정
    final_df['날짜'] = final_df['날짜'].dt.strftime('%Y-%m-%d')

    return final_df

def main():
    """메인 함수"""
    try:
        # 파일 경로 설정
        data_dir = Path('data')
        short_path = data_dir / 'holdout_daily_ranking_short_top20.csv'
        long_path = data_dir / 'holdout_daily_ranking_long_top20.csv'
        output_path = data_dir / 'holdout_daily_ranking_integrated_top20.csv'

        # 데이터 로드
        short_df, long_df = load_ranking_data(short_path, long_path)

        # 데이터 병합
        merged_df = merge_rankings(short_df, long_df)

        # 통합 랭킹 생성
        integrated_df = create_integrated_ranking(merged_df)

        # 결과 저장
        integrated_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        logger.info(f"통합 랭킹 파일 저장 완료: {output_path}")
        logger.info(f"최종 shape: {integrated_df.shape}")

        # 검증
        logger.info("=== 검증 결과 ===")
        logger.info(f"총 날짜 수: {integrated_df['날짜'].nunique()}")
        logger.info(f"총 종목 수: {integrated_df['종목명(ticker)'].nunique()}")
        logger.info(f"랭킹 분포: {integrated_df['랭킹'].value_counts().sort_index()}")

        # 샘플 출력
        logger.info("\n=== 샘플 데이터 ===")
        sample_date = integrated_df['날짜'].min()
        sample_data = integrated_df[integrated_df['날짜'] == sample_date].head(5)
        for _, row in sample_data.iterrows():
            logger.info(f"랭킹 {row['랭킹']}: {row['종목명(ticker)']} (score: {row['score']:.6f}) - {row['top3 피쳐그룹']}")

    except Exception as e:
        logger.error(f"오류 발생: {e}")
        raise

if __name__ == "__main__":
    main()