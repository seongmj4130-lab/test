# -*- coding: utf-8 -*-
"""
정규화 방법 비교 테스트: percentile, zscore, robust_zscore
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.tracks.track_a.stages.ranking.l8_dual_horizon import run_L8_short_rank_engine, run_L8_long_rank_engine
from src.utils.config import load_config
from src.utils.io import save_artifact, load_artifact
import pandas as pd
from src.stages.modeling.l6r_ranking_scoring import run_L6R_ranking_scoring

def test_normalization_method(method: str, cfg: dict, artifacts: dict):
    """정규화 방법 테스트"""
    print(f"\n{'='*60}")
    print(f"[테스트] {method} 정규화 방법")
    print(f"{'='*60}")
    
    # 정규화 방법 설정
    cfg['l8_short']['normalization_method'] = method
    cfg['l8_long']['normalization_method'] = method
    
    # L8 랭킹 생성
    print(f"  → 단기 랭킹 생성 중...")
    outputs_short, _ = run_L8_short_rank_engine(cfg, artifacts, force=True)
    print(f"  → 장기 랭킹 생성 중...")
    outputs_long, _ = run_L8_long_rank_engine(cfg, artifacts, force=True)
    
    # 임시 저장
    interim_dir = Path("data/interim")
    save_artifact(outputs_short['ranking_short_daily'], interim_dir / f"ranking_short_daily_{method}", force=True)
    save_artifact(outputs_long['ranking_long_daily'], interim_dir / f"ranking_long_daily_{method}", force=True)
    
    # L6R 실행을 위해 랭킹 데이터 교체
    artifacts_test = artifacts.copy()
    artifacts_test['ranking_short_daily'] = outputs_short['ranking_short_daily']
    artifacts_test['ranking_long_daily'] = outputs_long['ranking_long_daily']
    
    # L6R 실행
    print(f"  → L6R 랭킹 스코어링 실행 중...")
    outputs_l6r, _ = run_L6R_ranking_scoring(cfg, artifacts_test, force=True)
    rebalance_scores = outputs_l6r.get('rebalance_scores')
    
    if rebalance_scores is None or len(rebalance_scores) == 0:
        print(f"  ✗ {method}: rebalance_scores 생성 실패")
        return None
    
    # Hit Ratio 계산
    def calculate_hit_ratio(scores, return_col, score_col):
        df = scores.dropna(subset=[return_col, score_col])
        if len(df) == 0:
            return None
        pred_direction = (df[score_col] > 0).astype(int)
        actual_direction = (df[return_col] > 0).astype(int)
        hit = (pred_direction == actual_direction).astype(int)
        return float(hit.mean())
    
    # 통합 랭킹 Hit Ratio
    hr_ensemble = calculate_hit_ratio(rebalance_scores, 'true_short', 'score_ens')
    
    # 단기 랭킹 Hit Ratio
    hr_short = calculate_hit_ratio(rebalance_scores, 'true_short', 'score_total_short') if 'score_total_short' in rebalance_scores.columns else None
    
    # 장기 랭킹 Hit Ratio
    hr_long = calculate_hit_ratio(rebalance_scores, 'true_long', 'score_total_long') if 'score_total_long' in rebalance_scores.columns else None
    
    result = {
        'method': method,
        'hit_ratio_ensemble': hr_ensemble,
        'hit_ratio_short': hr_short,
        'hit_ratio_long': hr_long,
        'n_samples': len(rebalance_scores)
    }
    
    print(f"  ✓ {method} 완료:")
    print(f"    - 통합 랭킹 Hit Ratio: {hr_ensemble:.2%}" if hr_ensemble else "    - 통합 랭킹 Hit Ratio: N/A")
    print(f"    - 단기 랭킹 Hit Ratio: {hr_short:.2%}" if hr_short else "    - 단기 랭킹 Hit Ratio: N/A")
    print(f"    - 장기 랭킹 Hit Ratio: {hr_long:.2%}" if hr_long else "    - 장기 랭킹 Hit Ratio: N/A")
    
    return result

def main():
    # Config 로드
    cfg = load_config('configs/config.yaml')
    
    # Artifacts 로드
    interim_dir = Path("data/interim")
    artifacts = {
        'dataset_daily': load_artifact(interim_dir / 'dataset_daily'),
        'panel_merged_daily': load_artifact(interim_dir / 'panel_merged_daily'),
        'cv_folds_short': load_artifact(interim_dir / 'cv_folds_short'),
        'universe_k200_membership_monthly': load_artifact(interim_dir / 'universe_k200_membership_monthly'),
        'ohlcv_daily': load_artifact(interim_dir / 'ohlcv_daily'),
    }
    
    # 정규화 방법 테스트
    methods = ['percentile', 'zscore', 'robust_zscore']
    results = []
    
    for method in methods:
        try:
            result = test_normalization_method(method, cfg.copy(), artifacts)
            if result:
                results.append(result)
        except Exception as e:
            print(f"  ✗ {method} 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
    
    # 결과 비교
    print(f"\n{'='*60}")
    print("[정규화 방법 비교 결과]")
    print(f"{'='*60}")
    
    if results:
        # 통합 랭킹 Hit Ratio 기준 정렬
        results_sorted = sorted(results, key=lambda x: x['hit_ratio_ensemble'] or 0, reverse=True)
        
        print("\n[통합 랭킹 Hit Ratio 기준]")
        for i, r in enumerate(results_sorted, 1):
            print(f"  {i}. {r['method']:15s}: {r['hit_ratio_ensemble']:.2%}" if r['hit_ratio_ensemble'] else f"  {i}. {r['method']:15s}: N/A")
        
        # 최고 성과 방법
        best = results_sorted[0]
        print(f"\n[최고 성과 방법]")
        print(f"  방법: {best['method']}")
        print(f"  통합 랭킹 Hit Ratio: {best['hit_ratio_ensemble']:.2%}" if best['hit_ratio_ensemble'] else "  통합 랭킹 Hit Ratio: N/A")
        print(f"  단기 랭킹 Hit Ratio: {best['hit_ratio_short']:.2%}" if best['hit_ratio_short'] else "  단기 랭킹 Hit Ratio: N/A")
        print(f"  장기 랭킹 Hit Ratio: {best['hit_ratio_long']:.2%}" if best['hit_ratio_long'] else "  장기 랭킹 Hit Ratio: N/A")
        
        return best['method']
    else:
        print("  테스트 결과가 없습니다.")
        return None

if __name__ == "__main__":
    best_method = main()
    if best_method:
        print(f"\n[권장] {best_method} 정규화 방법 사용")

