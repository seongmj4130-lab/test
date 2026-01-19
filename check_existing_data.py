#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
기존 랭킹 데이터 확인
"""

import pandas as pd
import os

def check_existing_data():
    """기존 랭킹 데이터 확인"""

    # 기존 랭킹 데이터 확인
    ranking_files = [
        'data/interim/ranking_short_daily.parquet',
        'data/interim/ranking_long_daily.parquet',
        'results/final_track_a_performance_results.csv'
    ]

    for file_path in ranking_files:
        if os.path.exists(file_path):
            print(f'\n{file_path} 존재!')
            try:
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_parquet(file_path)

                print(f'크기: {len(df):,}행 × {len(df.columns)}열')
                print('컬럼:', list(df.columns))

                if len(df) > 0:
                    print('샘플 데이터:')
                    print(df.head(2).to_string())

                    # 날짜 범위 확인
                    if 'date' in df.columns:
                        print(f'날짜 범위: {df["date"].min()} ~ {df["date"].max()}')

                    # 기본 통계
                    if 'score' in df.columns:
                        print(f'스코어 통계: 평균 {df["score"].mean():.4f}, '
                              f'표준편차 {df["score"].std():.4f}, '
                              f'범위 {df["score"].min():.4f} ~ {df["score"].max():.4f}')

            except Exception as e:
                print(f'데이터 로드 오류: {e}')

        else:
            print(f'{file_path} 없음')

if __name__ == '__main__':
    check_existing_data()