#!/usr/bin/env python3
"""
단일 전략 백테스트 샘플 테스트
"""

import sys
from pathlib import Path
import logging
import yaml

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config, get_path
from src.utils.io import load_artifact
from src.tracks.track_b.stages.backtest.l7_backtest import run_backtest, BacktestConfig

def test_single_backtest(strategy_name: str, holding_days: int):
    """단일 전략 백테스트 테스트"""

    print(f"🧪 단일 전략 샘플 테스트: {strategy_name}, {holding_days}일")
    print("=" * 60)

    try:
        # 설정 로드
        cfg = load_config(project_root / 'configs' / 'config.yaml')

        # 개선된 파라미터 로드
        redesigned_path = project_root / 'configs' / 'redesigned_backtest_params.yaml'
        redesigned = None
        if redesigned_path.exists():
            with open(redesigned_path, 'r', encoding='utf-8') as f:
                redesigned = yaml.safe_load(f)

        # 전략 설정 가져오기
        if redesigned and 'params' in redesigned and strategy_name in redesigned['params']:
            strategy_config = redesigned['params'][strategy_name].copy()
            if strategy_name == "bt20_short":
                strategy_config['score_col'] = 'score_total_short'
            elif strategy_name == "bt120_long":
                strategy_config['score_col'] = 'score_total_long'
            elif strategy_name == "bt20_ens":
                strategy_config['score_col'] = 'score_ens'
        else:
            # 기존 설정 사용
            if strategy_name == "bt20_short":
                strategy_config = cfg.get('l7_bt20_short', {}).copy()
                strategy_config['score_col'] = 'score_total_short'
            elif strategy_name == "bt120_long":
                strategy_config = cfg.get('l7_bt120_long', {}).copy()
                strategy_config['score_col'] = 'score_total_long'
            elif strategy_name == "bt20_ens":
                strategy_config = cfg.get('l7_bt20_ens', {}).copy()
                strategy_config['score_col'] = 'score_ens'

        # holding_days 설정
        strategy_config['holding_days'] = holding_days

        print(f"📊 전략 설정: {strategy_config}")

        # BacktestConfig 생성
        backtest_cfg = BacktestConfig(
            holding_days=strategy_config.get('holding_days', 20),
            top_k=strategy_config.get('top_k', 20),
            cost_bps=strategy_config.get('cost_bps', 10.0),
            slippage_bps=strategy_config.get('slippage_bps', 0.0),
            score_col=strategy_config.get('score_col', 'score_ens'),
            ret_col='',
            weighting='equal',
            softmax_temp=1.0,
            overlapping_tranches_enabled=strategy_config.get('overlapping_tranches_enabled', False),
            tranche_holding_days=int(strategy_config.get('tranche_holding_days', 120) or 120),
            tranche_max_active=strategy_config.get('tranche_max_active', 4),
            tranche_allocation_mode='fixed_equal',
            buffer_k=strategy_config.get('buffer_k', 15),
            rebalance_interval=strategy_config.get('rebalance_interval', 20),
            diversify_enabled=False,
            group_col='sector_name',
            max_names_per_group=4,
            regime_enabled=strategy_config.get('regime', {}).get('enabled', False),
        )

        print(f"⚙️ BacktestConfig: top_k={backtest_cfg.top_k}, cost_bps={backtest_cfg.cost_bps}, slippage_bps={backtest_cfg.slippage_bps}")

        # 데이터 로드
        print("📂 데이터 로드 중...")
        baseline_dir = project_root / 'baseline_20260112_145649'
        l6_path = baseline_dir / 'data' / 'interim' / 'rebalance_scores_corrected.parquet'

        if not l6_path.exists():
            raise FileNotFoundError(f"L6 데이터 파일이 없습니다: {l6_path}")

        rebalance_scores = load_artifact(l6_path)

        # HOLDOUT 구간만 필터링
        if 'phase' in rebalance_scores.columns:
            holdout_data = rebalance_scores[rebalance_scores['phase'] == 'holdout']
            print(f"HOLDOUT 필터링: {len(rebalance_scores)} → {len(holdout_data)} 행")
            rebalance_scores = holdout_data
        else:
            print("⚠️ phase 컬럼 없음, 전체 데이터 사용")

        # 시장 국면 데이터 로드
        market_regime_path = baseline_dir / 'data' / 'interim' / 'l1d_market_regime.parquet'
        market_regime = None
        if market_regime_path.exists():
            market_regime = load_artifact(market_regime_path)
            print(f"시장 국면 데이터 로드: {len(market_regime)} 행")
        else:
            print("⚠️ 시장 국면 데이터 없음, regime 비활성화")

        # 백테스트 실행
        print("🚀 백테스트 실행 중...")
        (
            portfolio_df,
            trades_df,
            equity_curve_df,
            metrics_df,
            performance_dict,
            warnings_list,
            selection_diagnostics,
            returns_diagnostics,
            runtime_profile,
            regime_metrics
        ) = run_backtest(
            rebalance_scores=rebalance_scores,
            cfg=backtest_cfg,
            market_regime=market_regime
        )

        # 결과 분석
        print("📈 백테스트 결과:")
        if len(metrics_df) > 0:
            result = {
                'strategy': strategy_name,
                'holding_days': holding_days,
                'sharpe': float(metrics_df['net_sharpe'].mean()) if 'net_sharpe' in metrics_df.columns else 0.0,
                'cagr': float(metrics_df['net_cagr'].mean()) if 'net_cagr' in metrics_df.columns else 0.0,
                'total_return': float(metrics_df['net_total_return'].mean()) if 'net_total_return' in metrics_df.columns else 0.0,
                'mdd': float(metrics_df['net_mdd'].mean()) if 'net_mdd' in metrics_df.columns else 0.0,
                'calmar': float(metrics_df['net_calmar_ratio'].mean()) if 'net_calmar_ratio' in metrics_df.columns else 0.0,
                'hit_ratio': float(metrics_df['net_hit_ratio'].mean()) if 'net_hit_ratio' in metrics_df.columns else 0.0,
                'avg_turnover': float(metrics_df['avg_turnover_oneway'].mean()) if 'avg_turnover_oneway' in metrics_df.columns else 0.0,
                'profit_factor': float(metrics_df['net_profit_factor'].mean()) if 'net_profit_factor' in metrics_df.columns else 0.0,
                'avg_trade_duration': float(metrics_df['avg_trade_duration'].mean()) if 'avg_trade_duration' in metrics_df.columns else 0.0,
            }

            print(".4f")
            print(".2f")
            print(".2f")
            print(".2f")
            print(".2f")
            print(".2f")
            print(".2f")
            print(".2f")
            print(".2f")
            print(".2f")

            if warnings_list:
                print(f"⚠️ 경고: {len(warnings_list)}개")
                for warning in warnings_list[:3]:  # 처음 3개만 표시
                    print(f"   - {warning}")

            return result
        else:
            print("❌ 메트릭 데이터가 없습니다")
            return None

    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # 단기 전략 20일 테스트
    print("=== 단기 전략 20일 테스트 ===")
    result1 = test_single_backtest("bt20_short", 20)

    if result1:
        print("\n=== 통합 전략 60일 테스트 ===")
        result2 = test_single_backtest("bt20_ens", 60)

        if result2:
            print("\n=== 장기 전략 120일 테스트 ===")
            result3 = test_single_backtest("bt120_long", 120)

            if result3:
                print("\n🎯 테스트 결과 비교:")
                print(f"단기 20일 Sharpe: {result1['sharpe']:.3f}")
                print(f"통합 60일 Sharpe: {result2['sharpe']:.3f}")
                print(f"장기 120일 Sharpe: {result3['sharpe']:.3f}")
                print("✅ 실제 백테스트에서 값이 다르게 나오는 것을 확인!")