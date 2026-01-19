#!/usr/bin/env python3
"""
단일 백테스트 테스트
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.utils.config import load_config, get_path
from src.utils.io import load_artifact
from src.tracks.track_b.stages.backtest.l7_backtest import run_backtest, BacktestConfig

def test_single_backtest():
    """bt20_short, 20일 단일 백테스트"""

    print("🔬 단일 백테스트 테스트")
    print("=" * 50)

    # 설정 로드
    config_path = project_root / 'configs' / 'config.yaml'
    cfg = load_config(config_path)

    # 전략 설정 - bt20_short, 20일
    strategy_config = cfg.get('l7_bt20_short', {}).copy()
    strategy_config['score_col'] = 'score_total_short'
    strategy_config['holding_days'] = 20

    print("📋 Strategy config from YAML:")
    for key, value in strategy_config.items():
        print(f"  {key}: {value}")

    # BacktestConfig 생성
    backtest_cfg = BacktestConfig(
        holding_days=strategy_config.get('holding_days', 20),
        top_k=strategy_config.get('top_k', 20),
        cost_bps=strategy_config.get('cost_bps', 10.0),
        slippage_bps=strategy_config.get('slippage_bps', 0.0),
        score_col=strategy_config.get('score_col', 'score_ens'),
        ret_col=strategy_config.get('return_col', ''),
        weighting=strategy_config.get('weighting', 'equal'),
        softmax_temp=strategy_config.get('softmax_temperature', 1.0),
        overlapping_tranches_enabled=strategy_config.get('overlapping_tranches_enabled', False),
        tranche_holding_days=int(strategy_config.get('tranche_holding_days', 120) or 120),  # 강제 int 변환
        tranche_max_active=strategy_config.get('tranche_max_active', 4),
        tranche_allocation_mode=strategy_config.get('tranche_allocation_mode', 'fixed_equal'),
        buffer_k=strategy_config.get('buffer_k', 15),
        rebalance_interval=strategy_config.get('rebalance_interval', 20),
        diversify_enabled=strategy_config.get('diversify', {}).get('enabled', False),
        group_col=strategy_config.get('diversify', {}).get('group_col', 'sector_name'),
        max_names_per_group=strategy_config.get('diversify', {}).get('max_names_per_group', 4),
        regime_enabled=strategy_config.get('regime', {}).get('enabled', False),
        smart_buffer_enabled=strategy_config.get('smart_buffer_enabled', True),
        smart_buffer_stability_threshold=strategy_config.get('smart_buffer_stability_threshold', 0.7),
        volatility_adjustment_enabled=strategy_config.get('volatility_adjustment_enabled', True),
        volatility_lookback_days=strategy_config.get('volatility_lookback_days', 60),
        target_volatility=strategy_config.get('target_volatility', 0.15),
        volatility_adjustment_max=strategy_config.get('volatility_adjustment_max', 1.2),
        volatility_adjustment_min=strategy_config.get('volatility_adjustment_min', 0.7),
        risk_scaling_enabled=strategy_config.get('risk_scaling_enabled', True),
        risk_scaling_bear_multiplier=strategy_config.get('risk_scaling_bear_multiplier', 0.7),
        risk_scaling_neutral_multiplier=strategy_config.get('risk_scaling_neutral_multiplier', 0.9),
        risk_scaling_bull_multiplier=strategy_config.get('risk_scaling_bull_multiplier', 1.0),
    )

    print("📊 BacktestConfig:")
    print(f"  holding_days: {backtest_cfg.holding_days}")
    print(f"  score_col: {backtest_cfg.score_col}")
    print(f"  ret_col: {backtest_cfg.ret_col}")
    print(f"  rebalance_interval: {backtest_cfg.rebalance_interval}")
    print(f"  tranche_holding_days: {backtest_cfg.tranche_holding_days}")
    print(f"  overlapping_tranches_enabled: {backtest_cfg.overlapping_tranches_enabled}")

    # 데이터 로드
    print("\n📊 데이터 로드 중...")
    baseline_dir = project_root / 'baseline_20260112_145649'
    l6_path = baseline_dir / 'data' / 'interim' / 'rebalance_scores.parquet'

    if not l6_path.exists():
        print(f"❌ L6 데이터 파일이 없습니다: {l6_path}")
        return

    rebalance_scores = load_artifact(l6_path)
    print(f"L6 데이터 로드 완료: {len(rebalance_scores)} 행")

    # 샘플 데이터 확인
    print("\n🔍 L6 데이터 샘플:")
    print(rebalance_scores.head(3))

    # return 컬럼 확인
    print(f"\n🎯 Return 컬럼들: {[col for col in rebalance_scores.columns if 'true' in col.lower() or 'ret' in col.lower()]}")

    # 백테스트 실행
    print("\n⚡ 백테스트 실행 중...")
    try:
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
            market_regime=None
        )

        print("✅ 백테스트 완료!")

        # 결과 확인
        print(f"\n📊 성과 지표:")
        print(f"  Sharpe: {performance_dict.get('sharpe_ratio', 'N/A')}")
        print(f"  CAGR: {performance_dict.get('cagr', 'N/A')}")
        print(f"  Total Return: {performance_dict.get('total_return', 'N/A')}")
        print(f"  MDD: {performance_dict.get('max_drawdown', 'N/A')}")

        print(f"\n📋 Metrics DF shape: {metrics_df.shape}")
        if len(metrics_df) > 0:
            print("Metrics DF 샘플:")
            print(metrics_df.head(2))

            # 주요 성과 지표 확인
            for _, row in metrics_df.iterrows():
                phase = row.get('phase', 'unknown')
                sharpe = row.get('net_sharpe', 'N/A')
                cagr = row.get('net_cagr', 'N/A')
                mdd = row.get('net_mdd', 'N/A')
                total_return = row.get('net_total_return', 'N/A')
                print(f"  {phase.upper()}: Sharpe={sharpe}, CAGR={cagr}, MDD={mdd}, Total Return={total_return}")

    except Exception as e:
        print(f"❌ 백테스트 실패: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_single_backtest()