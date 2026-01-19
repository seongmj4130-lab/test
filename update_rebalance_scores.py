#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
앙상블 가중치 최적화를 위한 rebalance_scores 업데이트
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def update_rebalance_scores():
    """앙상블 가중치를 변경하여 rebalance_scores 업데이트"""

    print("=== 앙상블 가중치 최적화 ===")

    # config에서 가중치 읽기
    config_path = Path('configs/config.yaml')
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        weight_long = config.get('l6', {}).get('weight_long', 0.5)
        weight_short = config.get('l6', {}).get('weight_short', 0.5)
        print(f"Config에서 읽은 가중치 - weight_long: {weight_long}, weight_short: {weight_short}")
    else:
        # 기본값 사용
        weight_long = 0.8
        weight_short = 0.2
        print(f"기본 가중치 사용 - weight_long: {weight_long}, weight_short: {weight_short}")

    # 기존 rebalance_scores 로드
    rebalance_path = Path('data/interim/rebalance_scores.parquet')
    if not rebalance_path.exists():
        print("❌ rebalance_scores.parquet 파일이 존재하지 않습니다.")
        return

    df_old = pd.read_parquet(rebalance_path)
    print(f"기존 rebalance_scores: {len(df_old)}행")

    # 새로운 앙상블 가중치 적용
    df_new = df_old.copy()

    if 'score_short' in df_new.columns and 'score_long' in df_new.columns:
        print("앙상블 스코어 재계산 중...")
        # score_ens = weight_long * score_long + weight_short * score_short
        df_new['score_ens'] = weight_long * df_new['score_long'] + weight_short * df_new['score_short']

        # 새로운 파일로 저장
        new_path = Path('data/interim/rebalance_scores_optimized.parquet')
        df_new.to_parquet(new_path)
        print(f"✅ 최적화된 rebalance_scores 저장: {new_path}")

        print("새로운 score_ens 통계:")
        print(df_new['score_ens'].describe())

        # 기존 vs 새로운 score_ens 비교
        print("\nscore_ens 비교:")
        old_weight_long = 0.35
        old_weight_short = 0.65
        df_old_calc = old_weight_long * df_old['score_long'] + old_weight_short * df_old['score_short']
        print(f"기존 (weight_long={old_weight_long}, weight_short={old_weight_short}):")
        print(df_old_calc.describe())
        print(f"\n신규 (weight_long={weight_long}, weight_short={weight_short}):")
        print(df_new['score_ens'].describe())

    else:
        print("❌ score_short 또는 score_long 컬럼이 없음")
        print("사용 가능한 컬럼:", df_new.columns.tolist())

if __name__ == '__main__':
    update_rebalance_scores()
