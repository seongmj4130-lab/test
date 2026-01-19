import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.stages.modeling.l5_train_models import train_oos_predictions
from src.stages.modeling.l6_scoring import build_rebalance_scores
from src.tracks.track_b.stages.backtest.l7_backtest import BacktestConfig, run_backtest
from src.utils.config import get_path, load_config
from src.utils.io import artifact_exists, load_artifact, save_artifact


def test_holding_days_variations():
    """통합 전략에서 holding_days를 40, 60, 80, 100으로 변경해서 백테스트 실행"""

    print("🔬 통합 전략 holding_days 변화 백테스트")
    print("=" * 60)

    # 기본 설정 로드
    cfg = load_config()

    # L5, L6 실행 (재사용)
    print("📊 L5/L6 데이터 준비...")

    # 기존 데이터 사용
    base_dir = get_path(cfg, 'base_dir')
    interim_dir = Path(base_dir) / 'data' / 'interim'

    # rebalance_scores 로드
    rebalance_scores_path = interim_dir / 'rebalance_scores_fixed.csv'
    if not rebalance_scores_path.exists():
        print("❌ rebalance_scores_fixed.csv 없음")
        return

    rebalance_scores = pd.read_csv(rebalance_scores_path)
    print(f"✅ rebalance_scores 로드: {len(rebalance_scores)}행")

    # 테스트할 holding_days 값들
    holding_days_options = [40, 60, 80, 100]
    results = []

    # 각 holding_days에 대해 백테스트 실행
    for holding_days in holding_days_options:
        print(f"\n🏃 holding_days = {holding_days} 테스트")
        print("-" * 40)

        # 전략별 설정
        strategies = [
            {
                'name': f'bt20_ens_h{holding_days}',
                'config_section': 'l7_bt20_ens',
                'holding_days': holding_days,
                'score_col': 'score_ens'
            },
            {
                'name': f'bt120_ens_h{holding_days}',
                'config_section': 'l7_bt120_ens',
                'holding_days': holding_days,
                'score_col': 'score_ens'
            }
        ]

        for strategy in strategies:
            try:
                print(f"  📈 {strategy['name']} 백테스트 실행...")

                # BacktestConfig 생성 (holding_days만 변경)
                base_config = cfg.get(strategy['config_section'], {})
                bt_config = BacktestConfig(
                    holding_days=strategy['holding_days'],
                    top_k=base_config.get('top_k', 15),
                    cost_bps=base_config.get('cost_bps', 10.0),
                    slippage_bps=base_config.get('slippage_bps', 5.0),
                    score_col=strategy['score_col'],
                    ret_col=base_config.get('return_col', 'true_short'),
                    weighting=base_config.get('weighting', 'equal'),
                    buffer_k=base_config.get('buffer_k', 10),
                    rebalance_interval=base_config.get('rebalance_interval', 20)
                )

                # 백테스트 실행
                bt_positions, bt_returns, bt_metrics, warns = run_backtest(rebalance_scores, bt_config)

                # Holdout 결과 추출
                holdout = bt_metrics[bt_metrics['phase'] == 'holdout']
                if len(holdout) > 0:
                    result = holdout.iloc[0]
                    result_dict = {
                        'strategy': strategy['name'],
                        'holding_days': holding_days,
                        'net_sharpe': result['net_sharpe'],
                        'net_cagr': result['net_cagr'],
                        'net_mdd': result['net_mdd'],
                        'net_calmar_ratio': result['net_calmar_ratio'],
                        'net_total_return': result['net_total_return']
                    }
                    results.append(result_dict)

                    print(f"    ✅ 완료: Sharpe {result['net_sharpe']:.3f}, CAGR {result['net_cagr']:.1%}")

                else:
                    print(f"    ❌ Holdout 데이터 없음")

            except Exception as e:
                print(f"    ❌ {strategy['name']} 실패: {e}")
                continue

    # 결과 정리 및 저장
    if results:
        results_df = pd.DataFrame(results)
        results_file = f'results/holding_days_variations_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        results_df.to_csv(results_file, index=False, encoding='utf-8-sig')

        print("\n📊 최종 결과")
        print("-" * 50)
        print(results_df.to_string(index=False))

        print(f"\n💾 결과 저장: {results_file}")

        # holding_days별 분석
        print("\n📈 holding_days별 분석")
        print("-" * 40)

        for hd in holding_days_options:
            hd_results = [r for r in results if r['holding_days'] == hd]
            if hd_results:
                print(f"\nholding_days = {hd}:")
                for result in hd_results:
                    print(".1f")

    else:
        print("❌ 테스트 결과 없음")

if __name__ == "__main__":
    test_holding_days_variations()
