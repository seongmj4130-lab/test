#!/usr/bin/env python3
"""
부분 백테스트 실행: 전략별로 나누어 실행
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

sys.stdout.reconfigure(encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

from src.tracks.track_b.stages.backtest.l7_backtest import BacktestConfig, run_backtest
from src.utils.config import get_path, load_config
from src.utils.io import load_artifact, save_artifact


def get_strategy_config(cfg: dict, strategy_name: str, holding_days: int) -> dict:
    """전략별 설정을 가져와서 holding_days에 맞게 수정"""

    # [개선안 적용] redesigned_backtest_params.yaml 우선 사용
    redesigned_path = project_root / 'configs' / 'redesigned_backtest_params.yaml'
    if redesigned_path.exists():
        with open(redesigned_path, 'r', encoding='utf-8') as f:
            redesigned = yaml.safe_load(f)

        if 'params' in redesigned and strategy_name in redesigned['params']:
            config_section = redesigned['params'][strategy_name].copy()
            # score_col 설정
            if strategy_name == "bt20_short":
                config_section['score_col'] = 'score_total_short'
            elif strategy_name == "bt120_long":
                config_section['score_col'] = 'score_total_long'
            elif strategy_name == "bt20_ens":
                config_section['score_col'] = 'score_ens'
            logger.info(f"✅ {strategy_name} 전략에 개선 파라미터 적용")
        else:
            logger.warning(f"❌ {strategy_name} 전략 redesigned 파라미터 없음, 기존 설정 사용")
            config_section = {}
    else:
        config_section = {}

    # 개선 파라미터가 없으면 기존 config.yaml 사용
    if not config_section:
        logger.info(f"📋 {strategy_name} 전략: config.yaml에서 설정 로드")
        if strategy_name == "bt20_short":
            config_section = cfg.get('l7_bt20_short', {}).copy()
            config_section['score_col'] = 'score_total_short'
        elif strategy_name == "bt120_long":
            config_section = cfg.get('l7_bt120_long', {}).copy()
            config_section['score_col'] = 'score_total_long'
        elif strategy_name == "bt20_ens":
            config_section = cfg.get('l7_bt20_ens', {}).copy()
            config_section['score_col'] = 'score_ens'
        else:
            logger.warning(f"Unknown strategy: {strategy_name}, using default config")
            config_section = cfg.get('l7', {}).copy()

        logger.info(f"✅ {strategy_name} 설정: top_k={config_section.get('top_k')}, cost_bps={config_section.get('cost_bps')}, slippage_bps={config_section.get('slippage_bps')}")

    # holding_days 적용 (동적 파라미터가 자동 적용됨)
    config_section['holding_days'] = holding_days

    return config_section

def run_single_backtest(cfg: dict, strategy_name: str, holding_days: int) -> dict:
    """단일 전략, 단일 기간에 대한 백테스트 실행"""

    logger.info(f"🏃 {strategy_name} 전략, {holding_days}일 기간 백테스트 시작")

    try:
        # 전략 설정 가져오기
        strategy_config = get_strategy_config(cfg, strategy_name, holding_days)

        # BacktestConfig 생성
        backtest_cfg = BacktestConfig(
            holding_days=strategy_config.get('holding_days', 20),
            top_k=strategy_config.get('top_k', 20),
            cost_bps=strategy_config.get('cost_bps', 10.0),
            slippage_bps=strategy_config.get('slippage_bps', 0.0),
            score_col=strategy_config.get('score_col', 'score_ens'),
            target_volatility=strategy_config.get('target_volatility', 0.15),
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

        # 데이터 로드
        logger.info("📊 데이터 로드 중...")

        baseline_dir = project_root / 'baseline_20260112_145649'
        l6_path = baseline_dir / 'data' / 'interim' / 'rebalance_scores_corrected.parquet'
        if not l6_path.exists():
            raise FileNotFoundError(f"L6 데이터 파일이 없습니다: {l6_path}")

        rebalance_scores = load_artifact(l6_path)
        logger.info(f"L6 데이터 로드 완료: {len(rebalance_scores)} 행")

        # HOLDOUT 구간만 필터링
        if 'phase' in rebalance_scores.columns:
            holdout_data = rebalance_scores[rebalance_scores['phase'] == 'holdout']
            logger.info(f"HOLDOUT 구간 필터링: 전체 {len(rebalance_scores)} → HOLDOUT {len(holdout_data)} 행")
            rebalance_scores = holdout_data
        else:
            logger.warning("phase 컬럼이 없어 HOLDOUT 필터링을 수행할 수 없습니다.")

        # 시장 국면 데이터 로드
        market_regime_path = baseline_dir / 'data' / 'interim' / 'l1d_market_regime.parquet'
        market_regime = None
        if market_regime_path.exists():
            market_regime = load_artifact(market_regime_path)
            logger.info(f"시장 국면 데이터 로드 완료: {len(market_regime)} 행")
        else:
            logger.warning("market_regime 데이터가 없어 regime 기능을 비활성화합니다")

        # 실제 백테스트 실행
        logger.info("⚡ 백테스트 실행 중...")
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

        # 결과 정리
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
                'warnings': warnings_list,
                'timestamp': datetime.now().isoformat()
            }
            logger.info(f"✅ {strategy_name} {holding_days}일 백테스트 완료")
            logger.info(f"   Sharpe: {result['sharpe']:.4f}, CAGR: {result['cagr']:.4f}, MDD: {result['mdd']:.4f}")
            return result
        else:
            logger.warning(f"메트릭 데이터가 없습니다: {strategy_name} {holding_days}일")
            return None

    except Exception as e:
        logger.error(f"❌ {strategy_name} {holding_days}일 백테스트 실패: {e}")
        return None

def run_strategy_batch(cfg: dict, strategy_name: str, holding_days_list: list):
    """특정 전략의 여러 기간을 배치 실행"""

    print(f"🚀 {strategy_name} 전략 배치 실행 시작")
    print(f"   실행할 기간: {holding_days_list}")
    print("=" * 50)

    results = []

    for hd in holding_days_list:
        result = run_single_backtest(cfg, strategy_name, hd)
        if result:
            results.append(result)
        else:
            print(f"⚠️ {strategy_name} {hd}일 실행 실패")

    # 결과 저장
    if results:
        df = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = project_root / 'results' / f'backtest_{strategy_name}_{timestamp}.csv'
        df.to_csv(output_file, index=False)
        print(f"💾 {strategy_name} 결과 저장: {output_file}")

        # 요약 출력
        print(f"\n📊 {strategy_name} 전략 결과 요약:")
        for result in results:
            hd = result['holding_days']
            sharpe = result['sharpe']
            cagr = result['cagr']
            print(f"   {hd}일: Sharpe {sharpe:.3f}, CAGR {cagr:.2f}%")

    return results

def main():
    """메인 실행 함수"""

    # 설정 로드
    cfg = load_config(project_root / 'configs' / 'config.yaml')

    # 실행할 전략과 기간들
    strategies = {
        'bt20_short': [20, 40, 60, 80, 100, 120],
        'bt20_ens': [20, 40, 60, 80, 100, 120],
        'bt120_long': [20, 40, 60, 80, 100, 120]
    }

    all_results = []

    # 전략별 배치 실행
    for strategy_name, holding_days_list in strategies.items():
        results = run_strategy_batch(cfg, strategy_name, holding_days_list)
        all_results.extend(results)

    # 전체 결과 저장
    if all_results:
        df = pd.DataFrame(all_results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = project_root / 'results' / f'backtest_all_strategies_{timestamp}.csv'
        df.to_csv(output_file, index=False)
        print(f"\n💾 전체 결과 저장: {output_file}")

        # 최종 요약
        print("\n🎯 전체 백테스트 완료 요약:")
        print(f"   총 케이스: {len(all_results)}")
        print(f"   평균 Sharpe: {df['sharpe'].mean():.3f}")
        print(f"   최고 Sharpe: {df['sharpe'].max():.3f} ({df.loc[df['sharpe'].idxmax(), 'strategy']} {df.loc[df['sharpe'].idxmax(), 'holding_days']}일)")
        print(f"   평균 CAGR: {df['cagr'].mean():.2f}%")

if __name__ == "__main__":
    main()
