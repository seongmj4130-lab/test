# -*- coding: utf-8 -*-
"""L5/L6/L8 산출물 상세 일치 여부 확인"""
import pandas as pd
import numpy as np

print('=' * 100)
print('L5/L6/L8 산출물 상세 일치 여부 확인')
print('=' * 100)

# 파일 로드
l5_short = pd.read_parquet('data/interim/pred_short_oos.parquet')
l5_long = pd.read_parquet('data/interim/pred_long_oos.parquet')
l8_short = pd.read_parquet('data/interim/ranking_short_daily.parquet')
l8_long = pd.read_parquet('data/interim/ranking_long_daily.parquet')
l6_scores = pd.read_parquet('data/interim/rebalance_scores.parquet')
l6_scores_interval20 = pd.read_parquet('data/interim/rebalance_scores_from_ranking_interval_20.parquet')

# 1. L6가 L5 예측값을 포함하는지 확인
print('\n[1] L6가 L5 예측값을 포함하는지 확인')
print('-' * 100)

# L6의 특정 날짜 선택 (L5에 존재하는 날짜)
common_dates = set(l6_scores['date'].unique()) & set(l5_short['date'].unique())
if common_dates:
    test_date = sorted(common_dates)[0]
    print(f'\n테스트 날짜: {test_date}')
    
    l6_date = l6_scores[l6_scores['date'] == test_date].copy()
    l5_short_date = l5_short[l5_short['date'] == test_date].copy()
    
    print(f'  L6 행 수: {len(l6_date)}')
    print(f'  L5 Short 행 수: {len(l5_short_date)}')
    
    # ticker 기준으로 병합하여 비교
    merged = l6_date.merge(
        l5_short_date[['ticker', 'y_pred', 'y_true']],
        on='ticker',
        how='left',
        suffixes=('_l6', '_l5')
    )
    
    # score_short와 y_pred 비교
    if 'score_short' in merged.columns and 'y_pred' in merged.columns:
        valid = merged[merged['y_pred'].notna() & merged['score_short'].notna()]
        if len(valid) > 0:
            corr = valid['score_short'].corr(valid['y_pred'])
            print(f'  score_short vs y_pred 상관계수: {corr:.4f}')
            print(f'  일치하는 종목 수: {len(valid)}/{len(merged)}')
            
            # 샘플 비교
            print(f'\n  샘플 비교 (상위 5개):')
            sample = valid[['ticker', 'score_short', 'y_pred']].head()
            print(sample.to_string(index=False))

# 2. L6가 L8 랭킹을 포함하는지 확인
print('\n\n[2] L6가 L8 랭킹을 포함하는지 확인')
print('-' * 100)

# L6 interval20이 L8를 사용하는지 확인
common_dates_l8 = set(l6_scores_interval20['date'].unique()) & set(l8_short['date'].unique())
if common_dates_l8:
    test_date = sorted(common_dates_l8)[0]
    print(f'\n테스트 날짜: {test_date}')
    
    l6_date = l6_scores_interval20[l6_scores_interval20['date'] == test_date].copy()
    l8_short_date = l8_short[l8_short['date'] == test_date].copy()
    
    print(f'  L6 Interval20 행 수: {len(l6_date)}')
    print(f'  L8 Short 행 수: {len(l8_short_date)}')
    
    # ticker 기준으로 병합하여 비교
    merged = l6_date.merge(
        l8_short_date[['ticker', 'score_total', 'rank_total']],
        on='ticker',
        how='left',
        suffixes=('_l6', '_l8')
    )
    
    # score_total 비교
    print(f'  L6 컬럼: {list(l6_date.columns)}')
    print(f'  L8 컬럼: {list(l8_short_date.columns)}')
    
    # 컬럼명 확인
    l6_score_col = 'score_total' if 'score_total' in l6_date.columns else None
    l8_score_col = 'score_total' if 'score_total' in l8_short_date.columns else None
    
    if l6_score_col and l8_score_col:
        # 병합 시 suffixes로 인해 컬럼명이 변경됨
        merged_col_l6 = 'score_total' if 'score_total' in merged.columns else None
        merged_col_l8 = 'score_total_l8' if 'score_total_l8' in merged.columns else 'score_total'
        
        if merged_col_l6 and merged_col_l8 in merged.columns:
            valid = merged[merged[merged_col_l8].notna()]
            if len(valid) > 0:
                corr = valid[merged_col_l6].corr(valid[merged_col_l8])
                print(f'  score_total vs L8 score_total 상관계수: {corr:.4f}')
                print(f'  일치하는 종목 수: {len(valid)}/{len(merged)}')
                
                # 샘플 비교
                print(f'\n  샘플 비교 (상위 5개):')
                sample_cols = ['ticker', merged_col_l6, merged_col_l8]
                sample_cols = [c for c in sample_cols if c in valid.columns]
                if sample_cols:
                    print(valid[sample_cols].head().to_string(index=False))

# 3. L6 스코어 계산 확인
print('\n\n[3] L6 스코어 계산 확인')
print('-' * 100)

# score_ens = weight_short * score_short + weight_long * score_long 확인
if 'score_ens' in l6_scores.columns and 'score_short' in l6_scores.columns and 'score_long' in l6_scores.columns:
    valid = l6_scores[
        l6_scores['score_short'].notna() & 
        l6_scores['score_long'].notna() & 
        l6_scores['score_ens'].notna()
    ].copy()
    
    if len(valid) > 0:
        # weight_short = 0.5, weight_long = 0.5 가정
        expected_ens = 0.5 * valid['score_short'] + 0.5 * valid['score_long']
        actual_ens = valid['score_ens']
        
        diff = (expected_ens - actual_ens).abs()
        max_diff = diff.max()
        mean_diff = diff.mean()
        
        print(f'  score_ens 계산 확인 (weight_short=0.5, weight_long=0.5):')
        print(f'    검증 종목 수: {len(valid)}')
        print(f'    최대 차이: {max_diff:.6f}')
        print(f'    평균 차이: {mean_diff:.6f}')
        print(f'    일치 여부: {"✅ 일치" if max_diff < 0.0001 else "⚠️ 불일치"}')
        
        if max_diff >= 0.0001:
            print(f'\n    차이가 큰 샘플 (상위 5개):')
            valid['diff'] = diff
            print(valid.nlargest(5, 'diff')[['ticker', 'date', 'score_short', 'score_long', 'score_ens', 'diff']].to_string(index=False))

# 4. 날짜 범위 요약
print('\n\n[4] 날짜 범위 요약')
print('-' * 100)

print(f'\nL5 Short: {l5_short["date"].min()} ~ {l5_short["date"].max()} ({len(l5_short):,}행)')
print(f'L5 Long:  {l5_long["date"].min()} ~ {l5_long["date"].max()} ({len(l5_long):,}행)')
print(f'L8 Short: {l8_short["date"].min()} ~ {l8_short["date"].max()} ({len(l8_short):,}행)')
print(f'L8 Long:  {l8_long["date"].min()} ~ {l8_long["date"].max()} ({len(l8_long):,}행)')
print(f'L6 기본: {l6_scores["date"].min()} ~ {l6_scores["date"].max()} ({len(l6_scores):,}행, {l6_scores["date"].nunique()}개 날짜)')
print(f'L6 Interval20: {l6_scores_interval20["date"].min()} ~ {l6_scores_interval20["date"].max()} ({len(l6_scores_interval20):,}행, {l6_scores_interval20["date"].nunique()}개 날짜)')

# 5. 종목 수 비교
print('\n\n[5] 종목 수 비교')
print('-' * 100)

print(f'\nL5 Short 종목 수: {l5_short["ticker"].nunique()}')
print(f'L5 Long 종목 수:  {l5_long["ticker"].nunique()}')
print(f'L8 Short 종목 수: {l8_short["ticker"].nunique()}')
print(f'L8 Long 종목 수:  {l8_long["ticker"].nunique()}')
print(f'L6 기본 종목 수: {l6_scores["ticker"].nunique()}')
print(f'L6 Interval20 종목 수: {l6_scores_interval20["ticker"].nunique()}')

print('\n' + '=' * 100)
print('확인 완료')
print('=' * 100)

