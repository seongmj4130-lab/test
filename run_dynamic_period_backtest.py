#!/usr/bin/env python3
"""
동적 기간 백테스트 실행 스크립트
단기/장기/통합 3가지 전략에 대해 6개 기간(20,40,60,80,100,120일)으로 백테스트 실행
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

from src.stages.modeling.l5_train_models import train_oos_predictions
from src.stages.modeling.l6_scoring import build_rebalance_scores
from src.tracks.track_b.stages.backtest.l7_backtest import BacktestConfig, run_backtest
from src.utils.config import get_path, load_config
from src.utils.io import artifact_exists, load_artifact, save_artifact


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
            logger.info(f"✅ {strategy_name} 전략에 개선 파라미터 적용: {config_section}")
        else:
            logger.warning(f"❌ {strategy_name} 전략 redesigned 파라미터 없음, 기존 설정 사용")
            config_section = {}
    else:
        config_section = {}

    # 개선 파라미터가 없으면 기존 config.yaml 사용
    if not config_section:
        if strategy_name == "bt20_short":
            config_section = cfg.get('l7_bt20_short', {}).copy()
            config_section['score_col'] = 'score_total_short'
        elif strategy_name == "bt120_long":
            config_section = cfg.get('l7_bt120_long', {}).copy()
            config_section['score_col'] = 'score_total_long'
        elif strategy_name == "bt20_ens":
            # 통합 전략: 단기 노이즈 + 장기 지연 보완을 위한 파라미터 조정
            config_section = cfg.get('l7_bt20_ens', {}).copy()
            config_section['score_col'] = 'score_ens'

            # 단기(20일)와 장기(120일)의 중간값 적용
            short_params = cfg.get('l7_bt20_short', {})
            long_params = cfg.get('l7_bt120_long', {})

        # rebalance_interval 중간값 (단기 20 + 장기 20) / 2 = 20
        config_section['rebalance_interval'] = 20

        # target_volatility 중간값 (단기 0.15 + 장기 0.15) / 2 = 0.15
        config_section['target_volatility'] = 0.15

        # regime: semi (중간적 접근)
        config_section['regime'] = {'enabled': True}  # semi 대신 True로 설정 (config 파싱용)

        # buffer_k 중간값 (단기 15 + 장기 15) / 2 = 15
        config_section['buffer_k'] = 15

    else:
        logger.warning(f"Unknown strategy: {strategy_name}, using default config")
        config_section = cfg.get('l7', {}).copy()  # 기본 l7 설정 사용

    # holding_days 적용 (동적 파라미터가 자동 적용됨)
    config_section['holding_days'] = holding_days

    return config_section


def run_single_backtest(cfg: dict, strategy_name: str, holding_days: int) -> dict:
    """단일 전략, 단일 기간에 대한 백테스트 실행"""

    logger.info(f"🏃 {strategy_name} 전략, {holding_days}일 기간 백테스트 시작")

    try:
        # 전략 설정 가져오기
        strategy_config = get_strategy_config(cfg, strategy_name, holding_days)

        # BacktestConfig 생성 (디버깅용 최소 필드)
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
            regime_top_k_bull_strong=strategy_config.get('regime', {}).get('top_k_bull_strong', 10),
            regime_top_k_bull_weak=strategy_config.get('regime', {}).get('top_k_bull_weak', 12),
            regime_top_k_bear_strong=strategy_config.get('regime', {}).get('top_k_bear_strong', 30),
            regime_top_k_bear_weak=strategy_config.get('regime', {}).get('top_k_bear_weak', 30),
            regime_top_k_neutral=strategy_config.get('regime', {}).get('top_k_neutral', 20),
            regime_exposure_bull_strong=strategy_config.get('regime', {}).get('exposure_bull_strong', 1.5),
            regime_exposure_bull_weak=strategy_config.get('regime', {}).get('exposure_bull_weak', 1.2),
            regime_exposure_bear_strong=strategy_config.get('regime', {}).get('exposure_bear_strong', 0.7),
            regime_exposure_bear_weak=strategy_config.get('regime', {}).get('exposure_bear_weak', 0.9),
            regime_exposure_neutral=strategy_config.get('regime', {}).get('exposure_neutral', 1.0),
            regime_top_k_bull=strategy_config.get('regime', {}).get('top_k_bull', 15),
            regime_top_k_bear=strategy_config.get('regime', {}).get('top_k_bear', 30),
            regime_exposure_bull=strategy_config.get('regime', {}).get('exposure_bull', 1.0),
            regime_exposure_bear=strategy_config.get('regime', {}).get('exposure_bear', 1.0),
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

        # 데이터 로드
        logger.info("📊 데이터 로드 중...")

        # L6 랭킹 데이터 로드 (baseline 폴더에서 로드)
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

        # 시장 국면 데이터 로드 (선택사항)
        baseline_dir = project_root / 'baseline_20260112_145649'
        market_regime_path = baseline_dir / 'data' / 'interim' / 'l1d_market_regime.parquet'
        market_regime = None
        if market_regime_path.exists():
            market_regime = load_artifact(market_regime_path)
            logger.info(f"시장 국면 데이터 로드 완료: {len(market_regime)} 행")
        else:
            logger.warning("market_regime 데이터가 없어 regime 기능을 비활성화합니다")
            # market_regime 데이터가 없으면 경고만 출력 (config에서 이미 False로 설정됨)

        # 백테스트 실행 (디버깅: 일단 성공 return)
        logger.info("⚡ 백테스트 실행 중...")

        # 실제 백테스트 실행
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

        # 결과 정리 - metrics_df에서 직접 계산
        if len(metrics_df) > 0:
            try:
                # dev와 holdout 구간의 평균값 사용
                avg_sharpe = float(metrics_df['net_sharpe'].mean()) if 'net_sharpe' in metrics_df.columns and not metrics_df['net_sharpe'].isna().all() else 0.0
                avg_cagr = float(metrics_df['net_cagr'].mean()) if 'net_cagr' in metrics_df.columns and not metrics_df['net_cagr'].isna().all() else 0.0
                avg_total_return = float(metrics_df['net_total_return'].mean()) if 'net_total_return' in metrics_df.columns and not metrics_df['net_total_return'].isna().all() else 0.0
                avg_mdd = float(metrics_df['net_mdd'].mean()) if 'net_mdd' in metrics_df.columns and not metrics_df['net_mdd'].isna().all() else 0.0
                avg_calmar = float(metrics_df['net_calmar_ratio'].mean()) if 'net_calmar_ratio' in metrics_df.columns and not metrics_df['net_calmar_ratio'].isna().all() else 0.0
                avg_hit_ratio = float(metrics_df['net_hit_ratio'].mean()) if 'net_hit_ratio' in metrics_df.columns and not metrics_df['net_hit_ratio'].isna().all() else 0.0
                avg_turnover = float(metrics_df['avg_turnover_oneway'].mean()) if 'avg_turnover_oneway' in metrics_df.columns and not metrics_df['avg_turnover_oneway'].isna().all() else 0.0
                avg_profit_factor = float(metrics_df['net_profit_factor'].mean()) if 'net_profit_factor' in metrics_df.columns and not metrics_df['net_profit_factor'].isna().all() else 0.0
                avg_trade_duration = float(metrics_df['avg_trade_duration'].mean()) if 'avg_trade_duration' in metrics_df.columns and not metrics_df['avg_trade_duration'].isna().all() else 0.0
            except Exception as e:
                logger.warning(f"성과 지표 계산 중 오류: {e}")
                avg_sharpe = avg_cagr = avg_total_return = avg_mdd = avg_calmar = 0.0
                avg_hit_ratio = avg_turnover = avg_profit_factor = avg_trade_duration = 0.0
        else:
            avg_sharpe = avg_cagr = avg_total_return = avg_mdd = avg_calmar = 0.0
            avg_hit_ratio = avg_turnover = avg_profit_factor = avg_trade_duration = 0.0

        result = {
            'strategy': strategy_name,
            'holding_days': holding_days,
            'sharpe': avg_sharpe,
            'cagr': avg_cagr,
            'total_return': avg_total_return,
            'mdd': avg_mdd,
            'calmar': avg_calmar,
            'hit_ratio': avg_hit_ratio,
            'avg_turnover': avg_turnover,
            'profit_factor': avg_profit_factor,
            'avg_trade_duration': avg_trade_duration,
            'warnings': warnings_list,
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"✅ {strategy_name} {holding_days}일 백테스트 완료")
        logger.info(f"   Sharpe: {result['sharpe']:.4f}, CAGR: {result['cagr']:.4f}, MDD: {result['mdd']:.4f}")

        return result

    except Exception as e:
        logger.error(f"❌ {strategy_name} {holding_days}일 백테스트 실패: {e}")
        return {
            'strategy': strategy_name,
            'holding_days': holding_days,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def main():
    """메인 실행 함수"""

    print("🚀 동적 기간 백테스트 실행")
    print("=" * 60)

    # 설정 로드
    config_path = project_root / 'configs' / 'config.yaml'
    cfg = load_config(config_path)
    logger.info("설정 로드 완료")

    # 테스트할 전략 및 기간
    strategies = ['bt20_short', 'bt120_long', 'bt20_ens']  # 단기, 장기, 통합
    holding_days_list = [20, 40, 60, 80, 100, 120]

    # 전략 표시명 매핑
    strategy_names = {
        'bt20_short': '단기',
        'bt120_long': '장기',
        'bt20_ens': '통합'
    }

    print(f"📋 테스트 설정:")
    print(f"   • 전략: {len(strategies)}개 ({', '.join([strategy_names[s] for s in strategies])})")
    print(f"   • 기간: {len(holding_days_list)}개 ({holding_days_list})")
    print(f"   • 총 테스트 수: {len(strategies) * len(holding_days_list)}")
    print("-" * 60)

    # 결과 저장 리스트
    results = []

    # 모든 조합에 대해 백테스트 실행
    total_tests = len(strategies) * len(holding_days_list)
    test_count = 0

    for strategy in strategies:
        strategy_display_name = strategy_names[strategy]
        print(f"\n🎯 {strategy_display_name} 전략 테스트 시작")
        print("-" * 40)

        for hd in holding_days_list:
            test_count += 1
            print(f"\n🏃 테스트 {test_count}/{total_tests}: {strategy_display_name} - {hd}일")

            # 백테스트 실행
            result = run_single_backtest(cfg, strategy, hd)
            results.append(result)

            # 진행 상황 표시
            if 'error' not in result:
                print(".4f")
            else:
                print(f"   ❌ 실패: {result['error']}")

    # 결과 정리 및 저장
    print("\n📊 결과 정리 중...")
    print("=" * 60)

    # 성공한 결과만 필터링
    successful_results = [r for r in results if 'error' not in r]

    if successful_results:
        results_df = pd.DataFrame(successful_results)

        # 전략명 변경
        results_df['strategy_name'] = results_df['strategy'].map(strategy_names)

        print(f"✅ 성공한 테스트 수: {len(successful_results)}/{len(results)}")

        # Sharpe Ratio 피벗 테이블
        print("\n📈 Sharpe Ratio 비교표:")
        sharpe_pivot = results_df.pivot_table(
            index='strategy_name',
            columns='holding_days',
            values='sharpe',
            aggfunc='first'
        ).round(4)
        print(sharpe_pivot)

        # CAGR 피벗 테이블
        print("\n💰 CAGR 비교표:")
        cagr_pivot = results_df.pivot_table(
            index='strategy_name',
            columns='holding_days',
            values='cagr',
            aggfunc='first'
        ).round(4)
        print(cagr_pivot)

        # MDD 피벗 테이블
        print("\n📉 MDD 비교표:")
        mdd_pivot = results_df.pivot_table(
            index='strategy_name',
            columns='holding_days',
            values='mdd',
            aggfunc='first'
        ).round(4)
        print(mdd_pivot)

        # Calmar Ratio 피벗 테이블
        print("\n📊 Calmar Ratio 비교표:")
        calmar_pivot = results_df.pivot_table(
            index='strategy_name',
            columns='holding_days',
            values='calmar',
            aggfunc='first'
        ).round(4)
        print(calmar_pivot)

        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = project_root / f"results/dynamic_period_backtest_results_{timestamp}.csv"
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n💾 결과 저장 완료: {output_file}")

        # 종합 리포트 생성
        report_file = project_root / f"artifacts/reports/dynamic_period_backtest_report_{timestamp}.md"
        generate_report(results_df, report_file, strategy_names)

    else:
        print("❌ 성공한 테스트가 없습니다.")

    # 에러 결과 표시
    error_results = [r for r in results if 'error' in r]
    if error_results:
        print(f"\n❌ 실패한 테스트: {len(error_results)}개")
        for error_result in error_results:
            print(f"   • {error_result['strategy']} {error_result['holding_days']}일: {error_result['error']}")


def generate_report(results_df: pd.DataFrame, report_file: Path, strategy_names: dict):
    """결과 리포트를 생성"""

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# 동적 기간 백테스트 결과 보고서\n\n")
        f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 📋 테스트 개요\n\n")
        f.write("- **전략 수**: 3개 (단기, 장기, 통합)\n")
        f.write("- **기간 수**: 6개 (20, 40, 60, 80, 100, 120일)\n")
        f.write("- **총 테스트 수**: 18개\n")
        f.write("- **동적 파라미터 적용**: ✅ (holding_days별 최적 파라미터 자동 적용)\n\n")

        f.write("## 🏆 전략별 최고 성과\n\n")

        # 각 전략별 최고 성과 찾기
        for strategy in results_df['strategy'].unique():
            strategy_data = results_df[results_df['strategy'] == strategy]
            best_sharpe = strategy_data.loc[strategy_data['sharpe'].idxmax()]
            best_cagr = strategy_data.loc[strategy_data['cagr'].idxmax()]

            f.write(f"### {strategy_names[strategy]}\n")
            f.write(f"- **최고 Sharpe**: {best_sharpe['sharpe']:.4f} ({best_sharpe['holding_days']}일)\n")
            f.write(f"- **최고 CAGR**: {best_cagr['cagr']:.4f} ({best_cagr['holding_days']}일)\n")
            f.write(f"- **MDD**: {best_sharpe['mdd']:.4f}\n")
            f.write(f"- **Calmar**: {best_sharpe['calmar']:.4f}\n")
            f.write("\n")

        f.write("## 📊 상세 성과표\n\n")

        # 피벗 테이블 생성 및 저장
        metrics = ['sharpe', 'cagr', 'mdd', 'calmar', 'hit_ratio', 'avg_turnover']
        metric_names = {
            'sharpe': 'Sharpe Ratio',
            'cagr': 'CAGR',
            'mdd': 'MDD',
            'calmar': 'Calmar Ratio',
            'hit_ratio': 'Hit Ratio',
            'avg_turnover': 'Avg Turnover'
        }

        for metric in metrics:
            f.write(f"### {metric_names[metric]}\n\n")
            pivot = results_df.pivot_table(
                index='strategy_name',
                columns='holding_days',
                values=metric,
                aggfunc='first'
            ).round(4)
            f.write(pivot.to_markdown())
            f.write("\n\n")

        f.write("## 💡 분석 및 인사이트\n\n")

        # 기간별 평균 성과 계산
        period_avg = results_df.groupby('holding_days')[['sharpe', 'cagr', 'mdd']].mean().round(4)
        f.write("### 기간별 평균 성과\n\n")
        f.write(period_avg.to_markdown())
        f.write("\n\n")

        # 전략별 평균 성과 계산
        strategy_avg = results_df.groupby('strategy')[['sharpe', 'cagr', 'mdd']].mean().round(4)
        f.write("### 전략별 평균 성과\n\n")
        f.write(strategy_avg.to_markdown())
        f.write("\n\n")

        f.write("## 📁 결과 파일\n\n")
        f.write(f"- **CSV 데이터**: `results/dynamic_period_backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv`\n")
        f.write(f"- **보고서**: `{report_file.name}`\n")

    print(f"📄 보고서 생성 완료: {report_file}")


if __name__ == "__main__":
    main()
