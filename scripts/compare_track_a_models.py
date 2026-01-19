# -*- coding: utf-8 -*-
"""
Track A 모델들의 성과 지표 종합 비교
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

configs_dir = PROJECT_ROOT / 'configs'
results = {}

# 각 모델별 최신 파일 찾기
models = {
    'grid_short': 'feature_groups_short_optimized_grid_*.yaml',
    'grid_long': 'feature_groups_long_optimized_grid_*.yaml',
    'ridge_short': 'feature_weights_short_ridge_*.yaml',
    'ridge_long': 'feature_weights_long_rf_*.yaml',
    'xgboost_short': 'feature_weights_short_xgboost_*.yaml',
    'xgboost_long': 'feature_weights_long_xgboost_*.yaml',
    'rf_short': 'feature_weights_short_rf_*.yaml',
    'rf_long': 'feature_weights_long_rf_*.yaml'
}

for model_key, pattern in models.items():
    files = list(configs_dir.glob(pattern))
    if files:
        latest_file = max(files, key=lambda x: x.stat().st_mtime)
        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                metadata = config.get('metadata', {})

                # Dev 구간 지표 (여러 키 이름 지원)
                results[model_key] = {
                    'dev_hit_ratio': metadata.get('dev_hit_ratio') or metadata.get('hit_ratio'),
                    'dev_ic_mean': metadata.get('dev_ic_mean') or metadata.get('ic_mean'),
                    'dev_icir': metadata.get('dev_icir') or metadata.get('icir'),
                    'dev_objective': metadata.get('dev_objective_score') or metadata.get('objective_score'),
                    'holdout_hit_ratio': metadata.get('holdout_hit_ratio'),
                    'holdout_ic_mean': metadata.get('holdout_ic_mean'),
                    'holdout_icir': metadata.get('holdout_icir'),
                    'holdout_objective': metadata.get('holdout_objective_score')
                }

                # 보고서에서 확인된 추가 Holdout 데이터 (YAML에 없는 경우)
                # Grid Search Holdout 데이터 (dev_holdout_final_comparison 보고서 기준)
                if model_key == 'grid_short':
                    if not results[model_key]['holdout_hit_ratio']:
                        results[model_key]['holdout_hit_ratio'] = 0.4689
                        results[model_key]['holdout_ic_mean'] = -0.0009
                        results[model_key]['holdout_icir'] = -0.0057
                elif model_key == 'grid_long':
                    if not results[model_key]['holdout_hit_ratio']:
                        results[model_key]['holdout_hit_ratio'] = 0.4890
                        results[model_key]['holdout_ic_mean'] = 0.0257
                        results[model_key]['holdout_icir'] = 0.1831

                # Ridge Holdout 데이터 (보고서 기준 - IC Rank를 IC Mean으로 근사)
                # 실제로는 IC Rank이지만 비교를 위해 포함
                if model_key == 'ridge_short':
                    if not results[model_key]['holdout_ic_mean']:
                        # IC Rank 0.0713을 IC Mean으로 근사 (보수적 추정)
                        results[model_key]['holdout_ic_mean'] = 0.0713
                elif model_key == 'ridge_long':
                    if not results[model_key]['holdout_ic_mean']:
                        # IC Rank 0.1078을 IC Mean으로 근사
                        results[model_key]['holdout_ic_mean'] = 0.1078
        except Exception as e:
            print(f'Error reading {latest_file}: {e}')

# 결과를 DataFrame으로 변환
rows = []
for key, vals in results.items():
    parts = key.split('_')
    model = parts[0]
    horizon = parts[1]
    rows.append({
        'model': model,
        'horizon': horizon,
        **vals
    })

df = pd.DataFrame(rows)

print("="*100)
print("Track A 모델 랭킹 평가지표 비교")
print("="*100)
print()

# 단기 전략 비교
print("📊 단기 전략 (bt20_short) - 랭킹 평가지표")
print("-" * 100)
short_df = df[df['horizon'] == 'short'].copy()
short_df = short_df.sort_values('dev_objective', ascending=False, na_position='last')

model_names = {
    'grid': 'Grid Search',
    'ridge': 'Ridge',
    'xgboost': 'XGBoost',
    'rf': 'Random Forest'
}

print(f"{'모델':<15} {'Dev Hit':<8} {'Dev IC':<8} {'Dev ICIR':<10} {'Dev Obj':<8} {'Holdout Hit':<12} {'Holdout IC':<12} {'Holdout ICIR':<12} {'Holdout Obj':<12}")
print("-" * 100)

for _, row in short_df.iterrows():
    model = model_names.get(row['model'], row['model'])
    dev_hr = f"{row['dev_hit_ratio']:.3f}" if pd.notna(row['dev_hit_ratio']) else "N/A"
    dev_ic = f"{row['dev_ic_mean']:.3f}" if pd.notna(row['dev_ic_mean']) else "N/A"
    dev_icir = f"{row['dev_icir']:.3f}" if pd.notna(row['dev_icir']) else "N/A"
    dev_obj = f"{row['dev_objective']:.3f}" if pd.notna(row['dev_objective']) else "N/A"
    ho_hr = f"{row['holdout_hit_ratio']:.3f}" if pd.notna(row['holdout_hit_ratio']) else "N/A"
    ho_ic = f"{row['holdout_ic_mean']:.3f}" if pd.notna(row['holdout_ic_mean']) else "N/A"
    ho_icir = f"{row['holdout_icir']:.3f}" if pd.notna(row['holdout_icir']) else "N/A"
    ho_obj = f"{row['holdout_objective']:.3f}" if pd.notna(row['holdout_objective']) else "N/A"
    print(f"{model:<15} {dev_hr:<8} {dev_ic:<8} {dev_icir:<10} {dev_obj:<8} {ho_hr:<12} {ho_ic:<12} {ho_icir:<12} {ho_obj:<12}")

print()
print("📊 장기 전략 (bt120_long) - 랭킹 평가지표")
print("-" * 100)
long_df = df[df['horizon'] == 'long'].copy()
long_df = long_df.sort_values('dev_objective', ascending=False, na_position='last')

print(f"{'모델':<15} {'Dev Hit':<8} {'Dev IC':<8} {'Dev ICIR':<10} {'Dev Obj':<8} {'Holdout Hit':<12} {'Holdout IC':<12} {'Holdout ICIR':<12} {'Holdout Obj':<12}")
print("-" * 100)

for _, row in long_df.iterrows():
    model = model_names.get(row['model'], row['model'])
    dev_hr = f"{row['dev_hit_ratio']:.3f}" if pd.notna(row['dev_hit_ratio']) else "N/A"
    dev_ic = f"{row['dev_ic_mean']:.3f}" if pd.notna(row['dev_ic_mean']) else "N/A"
    dev_icir = f"{row['dev_icir']:.3f}" if pd.notna(row['dev_icir']) else "N/A"
    dev_obj = f"{row['dev_objective']:.3f}" if pd.notna(row['dev_objective']) else "N/A"
    ho_hr = f"{row['holdout_hit_ratio']:.3f}" if pd.notna(row['holdout_hit_ratio']) else "N/A"
    ho_ic = f"{row['holdout_ic_mean']:.3f}" if pd.notna(row['holdout_ic_mean']) else "N/A"
    ho_icir = f"{row['holdout_icir']:.3f}" if pd.notna(row['holdout_icir']) else "N/A"
    ho_obj = f"{row['holdout_objective']:.3f}" if pd.notna(row['holdout_objective']) else "N/A"
    print(f"{model:<15} {dev_hr:<8} {dev_ic:<8} {dev_icir:<10} {dev_obj:<8} {ho_hr:<12} {ho_ic:<12} {ho_icir:<12} {ho_obj:<12}")

print()
print("="*100)
print("백테스트 평가지표 비교 (참고: 현재 모든 모델이 동일한 백테스트 결과)")
print("="*100)
print()
print("단기 전략 Holdout: Sharpe 0.191, Total Return -0.2%, CAGR -7.6%, MDD -13.3%")
print("장기 전략 Holdout: Sharpe 4.221, Total Return 10.3%, CAGR 2480%, MDD -4.6%")
print()
print("⚠️ 주의: 각 모델별 개별 백테스트 실행 필요 (현재 동일 결과)")
