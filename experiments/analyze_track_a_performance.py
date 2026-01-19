# -*- coding: utf-8 -*-
"""
Track A 성과 상세 분석 스크립트
"""

from pathlib import Path
from src.utils.config import load_config
from src.utils.io import load_artifact
import pandas as pd
import numpy as np


def analyze_track_a_performance():
    """
    Track A 성과 상세 분석
    """
    cfg = load_config('configs/config.yaml')
    interim_dir = Path(cfg['paths']['base_dir']) / 'data' / 'interim'

    print('🔍 Track A 상세 성과 분석')
    print('='*60)

    # 랭킹 데이터 로드
    try:
        ranking_short = load_artifact(interim_dir / 'ranking_short_daily.parquet')
        ranking_long = load_artifact(interim_dir / 'ranking_long_daily.parquet')

        print(f'단기 랭킹 데이터: {len(ranking_short):,}행')
        print(f'장기 랭킹 데이터: {len(ranking_long):,}행')

        # IC 계산 함수
        def calculate_ic(ranking_df, score_col, true_col):
            ic_by_date = ranking_df.groupby('date').apply(
                lambda x: x[score_col].corr(x[true_col], method='spearman')
            ).dropna()

            ic_mean = ic_by_date.mean()
            ic_std = ic_by_date.std()
            icir = ic_mean / ic_std if ic_std > 0 else 0

            return {
                'ic_mean': ic_mean,
                'ic_std': ic_std,
                'icir': icir,
                'n_periods': len(ic_by_date)
            }

        # 단기 랭킹 IC
        short_ic = calculate_ic(ranking_short, 'score_total_short', 'true_short')
        print('📊 단기 랭킹 IC 분석:')
        print('.4f')
        print('.4f')
        print('.4f')
        print(f'  • 유효 기간: {short_ic["n_periods"]}개')

        # 장기 랭킹 IC
        long_ic = calculate_ic(ranking_long, 'score_total_long', 'true_long')
        print('
📊 장기 랭킹 IC 분석:'        print('.4f'        print('.4f'        print('.4f'        print(f'  • 유효 기간: {long_ic["n_periods"]}개')

        # 피처 분석
        print('
📊 피처 분석:'        numeric_cols = ranking_short.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in ['score_total_short', 'true_short', 'ret_fwd_20d', 'date', 'ticker']]

        print(f'  • 전체 피처 수: {len(feature_cols)}개')

        # 상위 IC 피처들 (샘플)
        feature_ic = {}
        for feature in feature_cols[:20]:  # 상위 20개 샘플링
            try:
                ic = ranking_short[feature].corr(ranking_short['true_short'], method='spearman')
                if not pd.isna(ic) and abs(ic) > 0.01:  # 의미있는 IC만
                    feature_ic[feature] = ic
            except:
                continue

        print(f'  • 상위 IC 피처 (Top 10):')
        sorted_features = sorted(feature_ic.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        for i, (feature, ic) in enumerate(sorted_features, 1):
            print('.4f')

        # 개선 효과 평가
        print('
🎯 개선 효과 평가:'        print(f'  • 목표 IC: ≥ 0.03')
        print('.4f'        print('.4f'
        print(f'  • 목표 ICIR: ≥ 0.5')
        print('.4f'        print('.4f'
        # 종합 평가
        short_score = (1 if short_ic['ic_mean'] >= 0.03 else 0) + (1 if short_ic['icir'] >= 0.5 else 0)
        long_score = (1 if long_ic['ic_mean'] >= 0.03 else 0) + (1 if long_ic['icir'] >= 0.5 else 0)

        print('
🏆 종합 평가:'        print(f'  • 단기 랭킹 점수: {short_score}/2')
        print(f'  • 장기 랭킹 점수: {long_score}/2')
        print(f'  • 전체 점수: {short_score + long_score}/4')

        total_score = short_score + long_score
        if total_score >= 3:
            rating = "⭐⭐⭐⭐⭐ EXCELLENT"
        elif total_score >= 2:
            rating = "⭐⭐⭐⭐ GOOD"
        else:
            rating = "⭐⭐ FAIR"

        print(f'  • 평가 등급: {rating}')

        # 결과 저장
        results = {
            'short_ic': short_ic,
            'long_ic': long_ic,
            'top_features': sorted_features[:5],
            'total_score': total_score,
            'rating': rating
        }

        return results

    except Exception as e:
        print(f'분석 중 오류 발생: {e}')
        return None


if __name__ == "__main__":
    results = analyze_track_a_performance()

    if results:
        print('
✅ Track A 성과 분석 완료!'        print(f'최종 평가: {results["rating"]}')