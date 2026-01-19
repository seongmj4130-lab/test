# -*- coding: utf-8 -*-
"""
앙상블 가중치 최적화 스크립트 (고속화 버전)

실제 데이터 기반 과적합 분석 결과를 활용하여 최적 앙상블 가중치를 탐색합니다.
Grid Search 방식으로 각 모델의 가중치를 최적화합니다.
"""
from __future__ import annotations

import sys
import warnings
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.components.ranking.score_engine import build_score_total
from src.utils.config import load_config
from src.utils.io import load_artifact


# 평가 지표 계산 함수들
def calculate_hit_ratio(scores: pd.Series, returns: pd.Series, top_k: int = 20) -> float:
    """Hit Ratio: 상위 top_k개 종목의 승률"""
    if len(scores) == 0 or len(returns) == 0:
        return np.nan
    top_k_idx = scores.nlargest(top_k).index
    top_k_returns = returns.loc[top_k_idx]
    hit_ratio = (top_k_returns > 0).mean()
    return float(hit_ratio) if not np.isnan(hit_ratio) else np.nan

def calculate_ic(scores: pd.Series, returns: pd.Series) -> float:
    """IC (Information Coefficient): Pearson 상관계수"""
    if len(scores) == 0 or len(returns) == 0:
        return np.nan
    valid_idx = scores.notna() & returns.notna()
    if valid_idx.sum() < 2:
        return np.nan
    s = pd.to_numeric(scores[valid_idx], errors='coerce')
    r = pd.to_numeric(returns[valid_idx], errors='coerce')
    final_valid = s.notna() & r.notna()
    if final_valid.sum() < 2:
        return np.nan
    s = s[final_valid]
    r = r[final_valid]
    if s.std() == 0 or r.std() == 0:
        return np.nan
    corr = s.corr(r)
    return float(corr) if not np.isnan(corr) else np.nan

def calculate_icir(ic_series: pd.Series) -> float:
    """ICIR: IC의 안정성 (mean / std)"""
    if len(ic_series) == 0:
        return np.nan
    ic_valid = ic_series.dropna()
    if len(ic_valid) == 0:
        return np.nan
    ic_mean = ic_valid.mean()
    ic_std = ic_valid.std()
    if ic_std == 0 or np.isnan(ic_std) or np.isnan(ic_mean):
        return np.nan
    icir = ic_mean / ic_std
    return float(icir) if not np.isnan(icir) else np.nan

def calculate_forward_returns(panel_data: pd.DataFrame, horizon: str) -> pd.DataFrame:
    """미래 수익률 계산"""
    df = panel_data.copy()
    periods = 20 if horizon == 'short' else 120

    # 종목별로 그룹화하여 미래 수익률 계산
    def calc_fwd_ret(group):
        prices = group['close'].pct_change(periods).shift(-periods)
        return prices

    df[f'ret_fwd_{periods}d'] = df.groupby('ticker').apply(calc_fwd_ret).reset_index(level=0, drop=True)

    return df


def calculate_objective_score(
    hit_ratio: float,
    ic_mean: float,
    icir: float,
    horizon: str = 'short'
) -> float:
    """
    목적함수 계산 (단기/장기별 가중치 적용)
    """
    if horizon == 'short':
        # 단기: Hit Ratio 40% + IC Mean 30% + ICIR 30%
        weights = {'hit': 0.4, 'ic': 0.3, 'icir': 0.3}
    else:
        # 장기: IC Mean 50% + ICIR 30% + Hit Ratio 20%
        weights = {'hit': 0.2, 'ic': 0.5, 'icir': 0.3}

    # NaN 처리
    hit_ratio = hit_ratio if not np.isnan(hit_ratio) else 0.0
    ic_mean = ic_mean if not np.isnan(ic_mean) else 0.0
    icir = icir if not np.isnan(icir) else 0.0

    objective = (
        weights['hit'] * hit_ratio +
        weights['ic'] * max(0, ic_mean) +  # IC는 양수만 고려
        weights['icir'] * max(0, icir)    # ICIR도 양수만 고려
    )

    return float(objective)

def generate_ensemble_ranking_fast(
    model_rankings: Dict[str, pd.DataFrame],
    weights: Dict[str, float]
) -> pd.DataFrame:
    """
    앙상블 랭킹 생성 (고속 벡터화 버전)

    Args:
        model_rankings: 모델별 랭킹 점수 {'grid': df, 'ridge': df, 'xgboost': df, 'rf': df}
        weights: 모델별 가중치 {'grid': 0.4, 'ridge': 0.3, 'xgboost': 0.2, 'rf': 0.1}

    Returns:
        앙상블 랭킹 DataFrame (date, ticker, score_ensemble)
    """
    # 가중치 정규화
    total_weight = sum(weights.values())
    if abs(total_weight) > 1e-10:
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
    else:
        n_models = len(weights)
        normalized_weights = {k: 1.0 / n_models for k in weights.keys()}

    # Pivot 테이블 생성 및 가중 합산 (벡터화)
    weighted_scores = []
    for model_name, weight in normalized_weights.items():
        if model_name in model_rankings and weight > 0:
            df = model_rankings[model_name].copy()
            df['weighted_score'] = df['score'] * weight
            weighted_scores.append(df[['date', 'ticker', 'weighted_score']])

    if not weighted_scores:
        return pd.DataFrame(columns=['date', 'ticker', 'score_ensemble'])

    # 병합 및 합산
    ensemble_df = weighted_scores[0].rename(columns={'weighted_score': 'score_ensemble'})
    for df in weighted_scores[1:]:
        ensemble_df = ensemble_df.merge(
            df, on=['date', 'ticker'], how='outer', suffixes=('', '_temp')
        )
        ensemble_df['score_ensemble'] = ensemble_df['score_ensemble'].fillna(0) + ensemble_df['weighted_score'].fillna(0)
        ensemble_df = ensemble_df.drop(columns=['weighted_score'])

    ensemble_df['score_ensemble'] = ensemble_df['score_ensemble'].fillna(0.5)

    return ensemble_df

def evaluate_ensemble(
    ensemble_ranking: pd.DataFrame,
    panel_data: pd.DataFrame,
    cv_folds: pd.DataFrame,
    horizon: str = 'short'
) -> Dict[str, float]:
    """앙상블 성과 평가"""
    target_col = 'ret_fwd_20d' if horizon == 'short' else 'ret_fwd_120d'

    # Dev/Holdout 구간 분리
    if 'segment' in cv_folds.columns:
        dev_folds = cv_folds[cv_folds['segment'] == 'dev']
        holdout_folds = cv_folds[cv_folds['segment'] == 'holdout']
    else:
        dev_folds = cv_folds[~cv_folds['fold_id'].str.startswith('holdout')]
        holdout_folds = cv_folds[cv_folds['fold_id'].str.startswith('holdout')]

    dev_dates = dev_folds['test_end'].unique()
    holdout_dates = holdout_folds['test_end'].unique()

    # 평가 함수
    def evaluate_dates(dates):
        ics, hits = [], []
        for date in dates:
            date_data = panel_data[panel_data['date'] == date]
            ranking_data = ensemble_ranking[ensemble_ranking['date'] == date]

            if len(ranking_data) < 20:
                continue

            merged = date_data.merge(ranking_data, on=['date', 'ticker'], how='inner')
            if len(merged) < 20:
                continue

            ic = calculate_ic(merged['score_ensemble'], merged[target_col])
            hit = calculate_hit_ratio(merged['score_ensemble'], merged[target_col], top_k=20)

            if not np.isnan(ic):
                ics.append(ic)
            if not np.isnan(hit):
                hits.append(hit)

        return ics, hits

    # Dev 평가
    dev_ics, dev_hits = evaluate_dates(dev_dates)
    dev_ic_mean = np.mean(dev_ics) if len(dev_ics) > 0 else np.nan
    dev_icir = calculate_icir(pd.Series(dev_ics)) if len(dev_ics) > 0 else np.nan
    dev_hit_ratio = np.mean(dev_hits) if len(dev_hits) > 0 else np.nan

    # Holdout 평가
    holdout_ics, holdout_hits = evaluate_dates(holdout_dates)
    holdout_ic_mean = np.mean(holdout_ics) if len(holdout_ics) > 0 else np.nan
    holdout_icir = calculate_icir(pd.Series(holdout_ics)) if len(holdout_ics) > 0 else np.nan
    holdout_hit_ratio = np.mean(holdout_hits) if len(holdout_hits) > 0 else np.nan

    # 목적함수
    dev_objective = calculate_objective_score(dev_hit_ratio, dev_ic_mean, dev_icir, horizon)
    holdout_objective = calculate_objective_score(holdout_hit_ratio, holdout_ic_mean, holdout_icir, horizon)

    return {
        'dev_ic_mean': dev_ic_mean,
        'dev_icir': dev_icir,
        'dev_hit_ratio': dev_hit_ratio,
        'dev_objective': dev_objective,
        'holdout_ic_mean': holdout_ic_mean,
        'holdout_icir': holdout_icir,
        'holdout_hit_ratio': holdout_hit_ratio,
        'holdout_objective': holdout_objective,
        'ic_diff': holdout_ic_mean - dev_ic_mean if not (np.isnan(holdout_ic_mean) or np.isnan(dev_ic_mean)) else np.nan,
        'objective_diff': holdout_objective - dev_objective if not (np.isnan(holdout_objective) or np.isnan(dev_objective)) else np.nan
    }

def generate_model_rankings(
    panel_data: pd.DataFrame,
    horizon: str = 'short'
) -> Dict[str, pd.DataFrame]:
    """
    각 모델별 랭킹 점수 생성
    """
    cfg = load_config('configs/config.yaml')
    base_dir = Path(cfg['paths']['base_dir'])
    configs_dir = base_dir / 'configs'

    # 모델 설정 파일들
    model_configs = {
        'grid': None,  # 최신 파일 찾기
        'ridge': None,
        'xgboost': None,
        'rf': configs_dir / f'feature_weights_{horizon}_rf_20260108_204232.yaml'
    }

    # Grid Search 최신 파일 찾기
    grid_pattern = f'feature_groups_{horizon}_optimized_grid_*.yaml'
    grid_files = list(configs_dir.glob(grid_pattern))
    if grid_files:
        model_configs['grid'] = max(grid_files, key=lambda x: x.stat().st_mtime)
        print(f"  Grid Search 최신 파일: {model_configs['grid'].name}")
    else:
        print("  Grid Search 파일을 찾을 수 없음")

    # 최신 파일 찾기
    for key in ['ridge', 'xgboost']:
        pattern = f'feature_weights_{horizon}_{key}_*.yaml'
        files = list(configs_dir.glob(pattern))
        print(f"  {key.upper()} 파일 검색 패턴: {pattern}")
        print(f"  발견된 파일 수: {len(files)}")
        if files:
            latest_file = max(files, key=lambda x: x.stat().st_mtime)
            model_configs[key] = latest_file
            print(f"  최신 파일: {latest_file.name}")
        else:
            print(f"  {key.upper()} 파일을 찾을 수 없음")

    rankings = {}

    # Grid Search 랭킹
    if model_configs['grid'].exists():
        try:
            rankings['grid'] = build_score_total(
                panel_data,
                feature_groups_config=model_configs['grid'],
                normalization_method='percentile',
                date_col='date'
            )
            rankings['grid'] = rankings['grid'][['date', 'ticker', 'score_total']].rename(columns={'score_total': 'score'})
            print(f"  Grid Search 랭킹 생성: {len(rankings['grid'])}개")
        except Exception as e:
            print(f"  ⚠️ Grid Search 랭킹 생성 실패: {e}")

    # ML 모델들 랭킹 직접 생성
    for model_name in ['ridge', 'xgboost', 'rf']:
        if model_configs[model_name] and model_configs[model_name].exists():
            try:
                # 피처 가중치 로드
                with open(model_configs[model_name], 'r', encoding='utf-8') as f:
                    weights_config = yaml.safe_load(f)

                # 랭킹 생성
                ranking_df = generate_ml_model_ranking(panel_data, weights_config, horizon)
                rankings[model_name] = ranking_df
                print(f"  {model_name.upper()} 랭킹 생성: {len(rankings[model_name])}개")
            except Exception as e:
                print(f"  ⚠️ {model_name.upper()} 랭킹 생성 실패: {e}")

    print(f"최종 랭킹 딕셔너리: {list(rankings.keys())}, 길이: {len(rankings)}")
    if rankings:
        print("✅ rankings 딕셔너리가 비어있지 않습니다")
        return rankings
    else:
        print("❌ rankings 딕셔너리가 비어있습니다")
        return None

def generate_ml_model_ranking(panel_data: pd.DataFrame, weights_config: dict, horizon: str) -> pd.DataFrame:
    """
    ML 모델 랭킹 생성
    """
    # 필요한 피처 선택 (OHLCV 포함)
    all_cols = [col for col in panel_data.columns
                if col not in ['date', 'ticker', 'ret_fwd_20d', 'ret_fwd_120d', 'split', 'phase', 'segment', 'fold_id', 'in_universe', 'ym', 'corp_code']
                and panel_data[col].dtype in [np.float64, np.float32, np.int64, np.int32]]

    if not all_cols:
        raise ValueError("사용 가능한 피처가 없습니다")

    # 가중치 추출
    if 'weights' in weights_config:
        weights = weights_config['weights']
    elif 'feature_weights' in weights_config:
        weights = weights_config['feature_weights']
    else:
        # config에서 직접 가중치 찾기
        weights = {}
        for key, value in weights_config.items():
            if isinstance(value, (int, float)):
                weights[key] = value

    if not weights:
        raise ValueError("가중치 정보를 찾을 수 없습니다")

    # 점수 계산
    scores = np.zeros(len(panel_data))
    valid_features = []

    for feature, weight in weights.items():
        if feature in all_cols and feature in panel_data.columns:
            feature_values = panel_data[feature].fillna(0)
            scores += feature_values * weight
            valid_features.append(feature)

    if not valid_features:
        raise ValueError("유효한 피처가 없습니다")

    # 결과 DataFrame 생성
    result_df = panel_data[['date', 'ticker']].copy()
    result_df['score'] = scores

    # 정규화 (랭킹 목적)
    result_df['score'] = (result_df['score'] - result_df['score'].mean()) / result_df['score'].std()

    return result_df

def optimize_ensemble_weights(
    horizon: str = 'short',
    weight_step: float = 0.1,
    max_weight: float = 1.0,
    max_combinations: int = 200
) -> pd.DataFrame:
    """
    앙상블 가중치 최적화 (Grid Search)

    Args:
        horizon: 'short' 또는 'long'
        weight_step: 가중치 간격
        max_weight: 최대 가중치
        max_combinations: 최대 평가 조합 수

    Returns:
        최적화 결과 DataFrame
    """
    print("="*100)
    print(f"🚀 앙상블 가중치 최적화 ({horizon.upper()} 전략) - {max_combinations}개 조합")
    print("="*100)

    # 데이터 로드
    cfg = load_config('configs/config.yaml')
    base_dir = Path(cfg['paths']['base_dir'])
    interim_dir = base_dir / 'data' / 'interim'

    panel_data = load_artifact(interim_dir / 'panel_merged_daily')
    cv_folds = load_artifact(interim_dir / f'cv_folds_{horizon}')

    print(f"📊 데이터: 패널 {len(panel_data):,}행, CV folds {len(cv_folds)}개")

    # 미래 수익률 계산 (없는 경우)
    target_col = 'ret_fwd_20d' if horizon == 'short' else 'ret_fwd_120d'
    if target_col not in panel_data.columns:
        print(f"⚠️ {target_col} 컬럼이 없어 계산 중...")
        panel_data = calculate_forward_returns(panel_data, horizon)
        print(f"✅ {target_col} 컬럼 계산 완료")

    # 모델별 랭킹 생성
    print("\n[1/3] 모델별 랭킹 생성 중...")
    model_rankings = generate_model_rankings(panel_data, horizon)

    if model_rankings is None or not model_rankings:
        print("⚠️ 사용 가능한 모델 랭킹이 없음")
        return pd.DataFrame()

    available_models = list(model_rankings.keys())
    print(f"✅ 생성 완료: {available_models}")

    # 가중치 조합 생성
    print("\n[2/3] 가중치 조합 생성 중...")
    weight_values = np.arange(0, max_weight + weight_step, weight_step)

    combinations = []
    for w in product(weight_values, repeat=len(available_models)):
        if sum(w) > 0:
            total = sum(w)
            normalized = tuple(wi / total for wi in w)
            combinations.append(normalized)

    # 중복 제거
    combinations = list(set(combinations))
    print(f"총 {len(combinations):,}개 조합 생성 → {min(max_combinations, len(combinations))}개 평가")

    # 평가 실행
    print("\n[3/3] 앙상블 평가 중...")
    print("="*100)

    results = []
    import time
    start_time = time.time()

    # tqdm 진행률 바
    pbar = tqdm(total=min(max_combinations, len(combinations)), desc="평가 진행", ncols=100)

    for i, weights_tuple in enumerate(combinations[:max_combinations]):
        iteration_start = time.time()

        weights = dict(zip(available_models, weights_tuple))

        try:
            # 앙상블 랭킹 생성 (고속)
            ensemble_ranking = generate_ensemble_ranking_fast(model_rankings, weights)

            if len(ensemble_ranking) == 0:
                pbar.update(1)
                continue

            # 앙상블 평가
            metrics = evaluate_ensemble(ensemble_ranking, panel_data, cv_folds, horizon)

            result = {
                'horizon': horizon,
                **{f'weight_{model}': weight for model, weight in weights.items()},
                **metrics
            }
            results.append(result)

            # 진행률 업데이트
            holdout_obj = metrics.get('holdout_objective', 0)
            pbar.set_postfix({
                'HoldoutObj': f"{holdout_obj:.4f}",
                'IC': f"{metrics.get('holdout_ic_mean', 0):.4f}",
                '시간/조합': f"{time.time() - iteration_start:.1f}s"
            })
            pbar.update(1)

            # 중간 저장 (매 50번마다)
            if (i + 1) % 50 == 0 and results:
                temp_df = pd.DataFrame(results)
                temp_file = base_dir / 'artifacts' / 'reports' / f'ensemble_optimization_{horizon}_intermediate_{i+1:04d}.csv'
                temp_df.to_csv(temp_file, index=False, encoding='utf-8-sig')
                tqdm.write(f"💾 중간 저장: {temp_file.name}")

        except Exception as e:
            tqdm.write(f"⚠️ 조합 {i+1} 평가 실패: {str(e)[:80]}")
            pbar.update(1)
            continue

    pbar.close()

    print("\n✅ 앙상블 평가 완료!")
    total_time = time.time() - start_time
    print(f"⏱️  총 소요시간: {total_time/60:.1f}분 ({total_time:.1f}초)")

    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        print("⚠️ 평가 결과가 없음")
        return pd.DataFrame()

    # 최적 결과 선택
    best_result = results_df.loc[results_df['holdout_objective'].idxmax()]

    print("\n" + "="*100)
    print("🏆 최적 앙상블 가중치:")
    print("="*100)
    for col in results_df.columns:
        if col.startswith('weight_'):
            model_name = col.replace('weight_', '').upper()
            weight = best_result[col]
            print(f"  {model_name:12s}: {weight:.3f}")

    print("\n📊 최적 성과:")
    print(f"  Holdout Obj  : {best_result['holdout_objective']:.4f}")
    print(f"  Holdout IC   : {best_result['holdout_ic_mean']:.4f}")
    print(f"  Holdout ICIR : {best_result['holdout_icir']:.4f}")
    print(f"  Holdout Hit  : {best_result['holdout_hit_ratio']:.1%}")
    print(f"  IC Diff      : {best_result['ic_diff']:.4f}")

    # 결과 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = base_dir / 'artifacts' / 'reports' / f'ensemble_optimization_{horizon}_{timestamp}.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 결과 저장: {output_file}")

    return results_df

def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='앙상블 가중치 최적화')
    parser.add_argument('--horizon', choices=['short', 'long', 'both'], default='short',
                       help='전략 유형 (기본: short)')
    parser.add_argument('--weight-step', type=float, default=0.1,
                       help='가중치 간격 (기본: 0.1)')
    parser.add_argument('--max-weight', type=float, default=1.0,
                       help='최대 가중치 (기본: 1.0)')
    parser.add_argument('--combinations', type=int, default=200,
                       help='최대 평가 조합 수 (기본: 200)')

    args = parser.parse_args()

    # 단기 전략
    if args.horizon in ['short', 'both']:
        optimize_ensemble_weights('short', args.weight_step, args.max_weight, args.combinations)

    # 장기 전략
    if args.horizon in ['long', 'both']:
        optimize_ensemble_weights('long', args.weight_step, args.max_weight, args.combinations)

def test_specific_weights(horizon: str, weight_sets: List[Dict[str, float]]):
    """특정 가중치 조합 테스트 (과적합 개선용)"""
    print(f"\n{'='*80}")
    print(f"🧪 특정 가중치 조합 테스트 ({horizon.upper()} 전략)")
    print(f"{'='*80}")

    # 데이터 로드
    cfg = load_config('configs/config.yaml')
    base_dir = Path(cfg['paths']['base_dir'])
    interim_dir = base_dir / 'data' / 'interim'
    results = []

    print("\n[1/3] 데이터 로드 중...")
    panel_data = load_artifact(interim_dir / 'panel_merged_daily')
    cv_folds = load_artifact(interim_dir / f'cv_folds_{horizon}')

    if panel_data is None or cv_folds is None:
        print("❌ 데이터 로드 실패")
        return pd.DataFrame()

    print(f"📊 데이터: 패널 {len(panel_data):,d}행, CV folds {len(cv_folds)}개")

    # 미래 수익률 계산 (없는 경우)
    target_col = 'ret_fwd_20d' if horizon == 'short' else 'ret_fwd_120d'
    if target_col not in panel_data.columns:
        print(f"⚠️ {target_col} 컬럼이 없어 계산 중...")
        panel_data = calculate_forward_returns(panel_data, horizon)
        print(f"✅ {target_col} 컬럼 계산 완료")

    # 모델 랭킹 생성
    print("\n[2/3] 모델별 랭킹 생성 중...")
    model_rankings = generate_model_rankings(panel_data, horizon)
    if model_rankings is None or not model_rankings:
        print("❌ 모델 랭킹 생성 실패")
        return pd.DataFrame()

    print(f"✅ 생성 완료: {list(model_rankings.keys())}")

    # 특정 가중치 조합 테스트
    print(f"\n[3/3] {len(weight_sets)}개 가중치 조합 평가 중...")
    for i, weights in enumerate(tqdm(weight_sets, desc="평가 진행")):
        try:
            # 앙상블 랭킹 생성
            ensemble_ranking = generate_ensemble_ranking_fast(model_rankings, weights)

            if len(ensemble_ranking) == 0:
                continue

            # 앙상블 평가
            metrics = evaluate_ensemble(ensemble_ranking, panel_data, cv_folds, horizon)

            result = {
                'horizon': horizon,
                'test_set': f'improved_{i+1}',
                **{f'weight_{model}': weight for model, weight in weights.items()},
                **metrics
            }
            results.append(result)

        except Exception as e:
            tqdm.write(f"⚠️ 조합 {i+1} 평가 실패: {str(e)[:80]}")
            continue

    print("\n✅ 특정 가중치 조합 테스트 완료!")

    results_df = pd.DataFrame(results)

    if len(results_df) == 0:
        print("⚠️ 평가 결과가 없음")
        return pd.DataFrame()

    # 결과 출력
    print(f"\n{'='*80}")
    print("🧪 테스트 결과:")
    print(f"{'='*80}")

    for idx, row in results_df.iterrows():
        print(f"\n테스트 세트 {row['test_set']}:")
        for col in results_df.columns:
            if col.startswith('weight_'):
                model_name = col.replace('weight_', '').upper()
                weight = row[col]
                print(f"  {model_name:12s}: {weight:.3f}")
        print(f"  Holdout Obj  : {row['holdout_objective']:.4f}")
        print(f"  Holdout IC   : {row['holdout_ic_mean']:.4f}")
        print(f"  Holdout ICIR : {row['holdout_icir']:.4f}")
        print(f"  Holdout Hit  : {row['holdout_hit_ratio']:.1%}")
        print(f"  IC Diff      : {row['ic_diff']:.4f}")

    # 결과 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = base_dir / 'artifacts' / 'reports' / f'ensemble_improved_weights_{horizon}_{timestamp}.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 결과 저장: {output_file}")

    return results_df

def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='앙상블 가중치 최적화')
    parser.add_argument('--horizon', choices=['short', 'long', 'both'], default='short',
                       help='전략 유형 (기본: short)')
    parser.add_argument('--weight-step', type=float, default=0.1,
                       help='가중치 간격 (기본: 0.1)')
    parser.add_argument('--max-weight', type=float, default=1.0,
                       help='최대 가중치 (기본: 1.0)')
    parser.add_argument('--combinations', type=int, default=200,
                       help='최대 평가 조합 수 (기본: 200)')
    parser.add_argument('--test-improved', action='store_true',
                       help='과적합 개선된 가중치 조합 테스트')

    args = parser.parse_args()

    if args.test_improved:
        # 과적합 개선된 가중치 조합 테스트
        improved_weights = {
            'short': [
                {'grid': 0.35, 'ridge': 0.57, 'xgboost': 0.08, 'rf': 0.00},  # XGBoost 감소, Ridge 증가
                {'grid': 0.30, 'ridge': 0.60, 'xgboost': 0.10, 'rf': 0.00},  # 추가 옵션
                {'grid': 0.40, 'ridge': 0.50, 'xgboost': 0.10, 'rf': 0.00},  # 보수적 옵션
            ],
            'long': [
                {'grid': 0.10, 'ridge': 0.20, 'xgboost': 0.70, 'rf': 0.00},  # XGBoost 단독 → 앙상블
                {'grid': 0.15, 'ridge': 0.25, 'xgboost': 0.60, 'rf': 0.00},  # 보수적 옵션
                {'grid': 0.05, 'ridge': 0.15, 'xgboost': 0.80, 'rf': 0.00},  # 공격적 옵션
            ]
        }

        if args.horizon in ['short', 'both']:
            test_specific_weights('short', improved_weights['short'])

        if args.horizon in ['long', 'both']:
            test_specific_weights('long', improved_weights['long'])

    else:
        # 기존 최적화 실행
        if args.horizon in ['short', 'both']:
            optimize_ensemble_weights('short', args.weight_step, args.max_weight, args.combinations)

        if args.horizon in ['long', 'both']:
            optimize_ensemble_weights('long', args.weight_step, args.max_weight, args.combinations)

if __name__ == "__main__":
    main()
