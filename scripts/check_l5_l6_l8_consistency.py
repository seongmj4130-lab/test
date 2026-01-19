# -*- coding: utf-8 -*-
"""L5/L6/L8 산출물 일치 여부 확인"""
import pandas as pd
import os
from datetime import datetime
from pathlib import Path

print('=' * 100)
print('L5/L6/L8 산출물 일치 여부 확인')
print('=' * 100)

# 파일 경로
l5_short = 'data/interim/pred_short_oos.parquet'
l5_long = 'data/interim/pred_long_oos.parquet'
l8_short = 'data/interim/ranking_short_daily.parquet'
l8_long = 'data/interim/ranking_long_daily.parquet'
l6_scores = 'data/interim/rebalance_scores.parquet'
l6_scores_interval20 = 'data/interim/rebalance_scores_from_ranking_interval_20.parquet'

results = {}

# L5 예측값 확인
print('\n[L5] 모델 예측값')
print('-' * 100)
for name, path in [('Short', l5_short), ('Long', l5_long)]:
    if os.path.exists(path):
        df = pd.read_parquet(path)
        mtime = os.path.getmtime(path)
        mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        
        print(f'\n{name} 예측값 ({path}):')
        print(f'  수정 시간: {mtime_str}')
        print(f'  행 수: {len(df):,}')
        print(f'  컬럼: {list(df.columns)}')
        
        if 'date' in df.columns:
            print(f'  날짜 범위: {df["date"].min()} ~ {df["date"].max()}')
        if 'ticker' in df.columns:
            print(f'  종목 수: {df["ticker"].nunique()}')
        
        results[f'l5_{name.lower()}'] = {
            'exists': True,
            'rows': len(df),
            'cols': list(df.columns),
            'date_min': df['date'].min() if 'date' in df.columns else None,
            'date_max': df['date'].max() if 'date' in df.columns else None,
            'tickers': df['ticker'].nunique() if 'ticker' in df.columns else None,
        }
    else:
        print(f'\n{name} 예측값: 파일 없음')
        results[f'l5_{name.lower()}'] = {'exists': False}

# L8 랭킹 확인
print('\n\n[L8] 랭킹 데이터')
print('-' * 100)
for name, path in [('Short', l8_short), ('Long', l8_long)]:
    if os.path.exists(path):
        df = pd.read_parquet(path)
        mtime = os.path.getmtime(path)
        mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        
        print(f'\n{name} 랭킹 ({path}):')
        print(f'  수정 시간: {mtime_str}')
        print(f'  행 수: {len(df):,}')
        print(f'  컬럼: {list(df.columns)}')
        
        if 'date' in df.columns:
            print(f'  날짜 범위: {df["date"].min()} ~ {df["date"].max()}')
        if 'ticker' in df.columns:
            print(f'  종목 수: {df["ticker"].nunique()}')
        
        results[f'l8_{name.lower()}'] = {
            'exists': True,
            'rows': len(df),
            'cols': list(df.columns),
            'date_min': df['date'].min() if 'date' in df.columns else None,
            'date_max': df['date'].max() if 'date' in df.columns else None,
            'tickers': df['ticker'].nunique() if 'ticker' in df.columns else None,
        }
    else:
        print(f'\n{name} 랭킹: 파일 없음')
        results[f'l8_{name.lower()}'] = {'exists': False}

# L6 스코어 확인
print('\n\n[L6] 스코어 데이터')
print('-' * 100)
for name, path in [('기본', l6_scores), ('Interval 20', l6_scores_interval20)]:
    if os.path.exists(path):
        df = pd.read_parquet(path)
        mtime = os.path.getmtime(path)
        mtime_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        
        print(f'\n{name} 스코어 ({path}):')
        print(f'  수정 시간: {mtime_str}')
        print(f'  행 수: {len(df):,}')
        print(f'  컬럼: {list(df.columns)}')
        
        if 'date' in df.columns:
            print(f'  날짜 범위: {df["date"].min()} ~ {df["date"].max()}')
            print(f'  고유 날짜 수: {df["date"].nunique()}')
        if 'ticker' in df.columns:
            print(f'  종목 수: {df["ticker"].nunique()}')
        
        # 스코어 컬럼 확인
        score_cols = [c for c in df.columns if 'score' in c.lower()]
        if score_cols:
            print(f'  스코어 컬럼: {score_cols}')
        
        results[f'l6_{name.lower().replace(" ", "_")}'] = {
            'exists': True,
            'rows': len(df),
            'cols': list(df.columns),
            'date_min': df['date'].min() if 'date' in df.columns else None,
            'date_max': df['date'].max() if 'date' in df.columns else None,
            'date_count': df['date'].nunique() if 'date' in df.columns else None,
            'tickers': df['ticker'].nunique() if 'ticker' in df.columns else None,
        }
    else:
        print(f'\n{name} 스코어: 파일 없음')
        results[f'l6_{name.lower().replace(" ", "_")}'] = {'exists': False}

# 일치 여부 확인
print('\n\n[일치 여부 확인]')
print('-' * 100)

# 날짜 범위 비교
if results.get('l5_short', {}).get('exists') and results.get('l8_short', {}).get('exists'):
    l5_date_min = results['l5_short']['date_min']
    l8_date_min = results['l8_short']['date_min']
    l5_date_max = results['l5_short']['date_max']
    l8_date_max = results['l8_short']['date_max']
    
    print(f'\nL5 Short vs L8 Short 날짜 범위:')
    print(f'  L5: {l5_date_min} ~ {l5_date_max}')
    print(f'  L8: {l8_date_min} ~ {l8_date_max}')
    print(f'  일치 여부: {"✅ 일치" if l5_date_min == l8_date_min and l5_date_max == l8_date_max else "⚠️ 불일치"}')

if results.get('l5_long', {}).get('exists') and results.get('l8_long', {}).get('exists'):
    l5_date_min = results['l5_long']['date_min']
    l8_date_min = results['l8_long']['date_min']
    l5_date_max = results['l5_long']['date_max']
    l8_date_max = results['l8_long']['date_max']
    
    print(f'\nL5 Long vs L8 Long 날짜 범위:')
    print(f'  L5: {l5_date_min} ~ {l5_date_max}')
    print(f'  L8: {l8_date_min} ~ {l8_date_max}')
    print(f'  일치 여부: {"✅ 일치" if l5_date_min == l8_date_min and l5_date_max == l8_date_max else "⚠️ 불일치"}')

# L6가 L5/L8를 포함하는지 확인
if results.get('l6_기본', {}).get('exists'):
    l6_df = pd.read_parquet(l6_scores)
    print(f'\nL6 스코어가 L5/L8 데이터를 포함하는지 확인:')
    
    # L6에 필요한 컬럼이 있는지 확인
    required_cols = ['date', 'ticker']
    has_required = all(col in l6_df.columns for col in required_cols)
    print(f'  필수 컬럼 (date, ticker): {"✅ 있음" if has_required else "❌ 없음"}')
    
    # 스코어 컬럼 확인
    score_cols = [c for c in l6_df.columns if 'score' in c.lower()]
    print(f'  스코어 컬럼: {score_cols if score_cols else "없음"}')
    
    # 샘플 데이터 확인
    if len(l6_df) > 0:
        print(f'\n  샘플 데이터 (첫 5행):')
        print(l6_df.head().to_string())

print('\n' + '=' * 100)
print('확인 완료')
print('=' * 100)

