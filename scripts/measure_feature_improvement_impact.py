# -*- coding: utf-8 -*-
"""
피쳐 개선 효과 측정 스크립트

개선 전후 IC 성과를 종합적으로 비교 분석합니다.
"""

import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.feature_unit_tester import FeatureUnitTester
from src.utils.config import load_config
from src.utils.io import load_artifact, save_artifact


def measure_baseline_performance():
    """
    개선 전 기준 성과 측정

    Returns:
        기준 성과 딕셔너리
    """
    print("📊 기준 성과 측정 시작...")

    # 설정 로드
    cfg = load_config('configs/config.yaml')
    interim_dir = Path(cfg['paths']['base_dir']) / 'data' / 'interim'

    # CV folds 로드
    cv_folds = load_artifact(interim_dir / 'cv_folds_short')

    # 기본 피쳐들로 단위 테스트
    baseline_features = [
        'close', 'volume', 'price_momentum', 'price_momentum_60d',
        'momentum_3m', 'momentum_6m', 'volatility_60d', 'volatility_20d',
        'net_income', 'roe', 'debt_ratio', 'turnover'
    ]

    # 테스트용 데이터 준비
    panel_df = load_artifact(interim_dir / 'panel_merged_daily')
    rebalance_df = load_artifact(interim_dir / 'rebalance_scores_from_ranking')

    if panel_df is None or rebalance_df is None:
        return None

    # 실제 존재하는 피쳐들만
    available_features = [f for f in baseline_features if f in panel_df.columns]

    feature_data = panel_df[['date', 'ticker'] + available_features].copy()
    target_data = rebalance_df[['date', 'ticker', 'true_short']].copy()
    target_data = target_data.rename(columns={'true_short': 'ret_fwd_20d'})

    # 단위 테스트 실행
    tester = FeatureUnitTester()
    baseline_results = tester.test_feature_set(
        feature_data, target_data, cv_folds, available_features, 'short'
    )

    if len(baseline_results) > 0:
        # 종합 IC 계산
        avg_ic = baseline_results['holdout_ic_mean'].mean()
        avg_hit = baseline_results['holdout_hit_ratio'].mean()

        baseline_performance = {
            'features_tested': len(baseline_results),
            'avg_ic': avg_ic,
            'avg_hit_ratio': avg_hit,
            'top_feature_ic': baseline_results['holdout_ic_mean'].max(),
            'feature_details': baseline_results.to_dict('records')
        }

        print(".4f")
        print(".1%")
        return baseline_performance

    return None


def measure_improved_performance():
    """
    개선 후 성과 측정

    Returns:
        개선 성과 딕셔너리
    """
    print("🚀 개선 성과 측정 시작...")

    # 설정 로드
    cfg = load_config('configs/config.yaml')
    interim_dir = Path(cfg['paths']['base_dir']) / 'data' / 'interim'

    # CV folds 로드
    cv_folds = load_artifact(interim_dir / 'cv_folds_short')

    # 개선된 피쳐들
    improved_features = [
        # 기존 피쳐들
        'close', 'volume', 'price_momentum', 'price_momentum_60d',
        'momentum_3m', 'momentum_6m', 'volatility_60d', 'volatility_20d',
        'net_income', 'roe', 'debt_ratio', 'turnover',
        # 새로 추가된 피쳐들
        'close_to_52w_high', 'close_to_52w_low', 'intraday_price_position',
        'momentum_3m_ewm', 'momentum_6m_ewm', 'momentum_3m_vol_adj',
        'volatility_asymmetry', 'tail_risk_5pct',
        'news_intensity', 'news_trend'
    ]

    # 테스트용 데이터 준비
    panel_df = load_artifact(interim_dir / 'panel_merged_daily')
    rebalance_df = load_artifact(interim_dir / 'rebalance_scores_from_ranking')

    if panel_df is None or rebalance_df is None:
        return None

    # 실제 존재하는 피쳐들만
    available_features = [f for f in improved_features if f in panel_df.columns]

    feature_data = panel_df[['date', 'ticker'] + available_features].copy()
    target_data = rebalance_df[['date', 'ticker', 'true_short']].copy()
    target_data = target_data.rename(columns={'true_short': 'ret_fwd_20d'})

    # 단위 테스트 실행
    tester = FeatureUnitTester()
    improved_results = tester.test_feature_set(
        feature_data, target_data, cv_folds, available_features, 'short'
    )

    if len(improved_results) > 0:
        # 종합 IC 계산
        avg_ic = improved_results['holdout_ic_mean'].mean()
        avg_hit = improved_results['holdout_hit_ratio'].mean()

        improved_performance = {
            'features_tested': len(improved_results),
            'avg_ic': avg_ic,
            'avg_hit_ratio': avg_hit,
            'top_feature_ic': improved_results['holdout_ic_mean'].max(),
            'feature_details': improved_results.to_dict('records')
        }

        print(".4f")
        print(".1%")
        return improved_performance

    return None


def compare_before_after(baseline: Dict, improved: Dict) -> Dict:
    """
    개선 전후 비교 분석

    Args:
        baseline: 기준 성과
        improved: 개선 성과

    Returns:
        비교 분석 결과
    """
    comparison = {
        'baseline_features': baseline['features_tested'],
        'improved_features': improved['features_tested'],
        'new_features_added': improved['features_tested'] - baseline['features_tested'],
        'ic_improvement': improved['avg_ic'] - baseline['avg_ic'],
        'ic_improvement_pct': ((improved['avg_ic'] - baseline['avg_ic']) / abs(baseline['avg_ic'])) * 100 if baseline['avg_ic'] != 0 else 0,
        'hit_ratio_improvement': improved['avg_hit_ratio'] - baseline['avg_hit_ratio'],
        'hit_ratio_improvement_pct': ((improved['avg_hit_ratio'] - baseline['avg_hit_ratio']) / baseline['avg_hit_ratio']) * 100 if baseline['avg_hit_ratio'] != 0 else 0,
        'top_ic_improvement': improved['top_feature_ic'] - baseline['top_feature_ic']
    }

    # 개선 평가
    if comparison['ic_improvement'] > 0.005:  # IC 0.005 이상 개선
        comparison['overall_assessment'] = 'EXCELLENT'
    elif comparison['ic_improvement'] > 0.002:  # IC 0.002 이상 개선
        comparison['overall_assessment'] = 'GOOD'
    elif comparison['ic_improvement'] > 0:
        comparison['overall_assessment'] = 'MODERATE'
    else:
        comparison['overall_assessment'] = 'NEEDS_IMPROVEMENT'

    return comparison


def generate_improvement_report(baseline: Dict, improved: Dict, comparison: Dict) -> str:
    """
    개선 효과 보고서 생성

    Args:
        baseline: 기준 성과
        improved: 개선 성과
        comparison: 비교 분석

    Returns:
        보고서 문자열
    """
    report = []
    report.append("# 피쳐 개선 효과 측정 보고서")
    report.append("")
    report.append(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")

    # 개요
    report.append("## 📋 분석 개요")
    report.append("")
    report.append("- **개선 전 피쳐 수**: {}개".format(comparison['baseline_features']))
    report.append("- **개선 후 피쳐 수**: {}개".format(comparison['improved_features']))
    report.append("- **추가된 피쳐 수**: {}개".format(comparison['new_features_added']))
    report.append("")

    # 성과 비교
    report.append("## 📊 성과 비교")
    report.append("")
    report.append("| 구분 | 개선 전 | 개선 후 | 개선량 | 개선율 |")
    report.append("|------|--------|--------|--------|--------|")
    report.append("| 평균 IC | {:.4f} | {:.4f} | {:.4f} | {:.1f}% |".format(
        baseline['avg_ic'], improved['avg_ic'],
        comparison['ic_improvement'], comparison['ic_improvement_pct']
    ))
    report.append("| 평균 Hit Ratio | {:.1%} | {:.1%} | {:.1%} | {:.1f}% |".format(
        baseline['avg_hit_ratio'], improved['avg_hit_ratio'],
        comparison['hit_ratio_improvement'], comparison['hit_ratio_improvement_pct']
    ))
    report.append("")

    # 평가
    report.append("## 🎯 종합 평가")
    report.append("")
    assessment = comparison['overall_assessment']
    if assessment == 'EXCELLENT':
        report.append("**⭐ EXCELLENT**: IC 개선이 매우 우수합니다 (0.005+).")
    elif assessment == 'GOOD':
        report.append("**✅ GOOD**: IC 개선이 양호합니다 (0.002-0.005).")
    elif assessment == 'MODERATE':
        report.append("**⚠️ MODERATE**: IC 개선이 미미합니다 (0-0.002).")
    else:
        report.append("**❌ NEEDS IMPROVEMENT**: 개선이 필요합니다.")
    report.append("")

    # 상세 분석
    report.append("## 🔍 상세 분석")
    report.append("")

    # 상위 개선 피쳐들
    if baseline.get('feature_details') and improved.get('feature_details'):
        # 개선된 피쳐들의 IC 비교
        baseline_df = pd.DataFrame(baseline['feature_details'])
        improved_df = pd.DataFrame(improved['feature_details'])

        # 공통 피쳐들 비교
        common_features = set(baseline_df['feature_name']) & set(improved_df['feature_name'])
        if common_features:
            report.append("### 기존 피쳐들의 개선 효과")
            report.append("")
            report.append("| 피쳐명 | 개선 전 IC | 개선 후 IC | 개선량 |")
            report.append("|--------|-----------|-----------|--------|")

            for feature in list(common_features)[:10]:  # 상위 10개만
                baseline_ic = baseline_df[baseline_df['feature_name'] == feature]['holdout_ic_mean'].iloc[0]
                improved_ic = improved_df[improved_df['feature_name'] == feature]['holdout_ic_mean'].iloc[0]
                improvement = improved_ic - baseline_ic

                report.append("| {} | {:.4f} | {:.4f} | {:.4f} |".format(
                    feature, baseline_ic, improved_ic, improvement
                ))

            report.append("")

        # 새로운 피쳐들의 성과
        new_features = set(improved_df['feature_name']) - set(baseline_df['feature_name'])
        if new_features:
            report.append("### 새로운 피쳐들의 성과")
            report.append("")
            report.append("| 피쳐명 | IC | Hit Ratio | 품질 점수 |")
            report.append("|--------|----|-----------|----------|")

            new_feature_results = improved_df[improved_df['feature_name'].isin(new_features)]
            top_new = new_feature_results.nlargest(10, 'quality_score')

            for _, row in top_new.iterrows():
                report.append("| {} | {:.4f} | {:.1%} | {:.1f} |".format(
                    row['feature_name'], row['holdout_ic_mean'],
                    row['holdout_hit_ratio'], row['quality_score']
                ))

            report.append("")

    # 결론 및 권장사항
    report.append("## 💡 결론 및 권장사항")
    report.append("")

    if comparison['ic_improvement'] > 0:
        report.append("✅ **피쳐 개선이 긍정적인 효과를 보였습니다.**")
        report.append("- IC 평균: {:.1f}% 개선".format(comparison['ic_improvement_pct']))
        report.append("- Hit Ratio: {:.1f}% 개선".format(comparison['hit_ratio_improvement_pct']))
        report.append("")
        report.append("**추천 사항**:")
        report.append("1. 개선된 피쳐셋을 정식으로 채택")
        report.append("2. 상위 성과 피쳐들을 우선 활용")
        report.append("3. 추가 피쳐 엔지니어링 고려")
    else:
        report.append("⚠️ **피쳐 개선 효과가 제한적입니다.**")
        report.append("- 추가적인 피쳐 엔지니어링 필요")
        report.append("- 다른 개선 방향 탐색 권장")

    return "\n".join(report)


def main():
    """
    메인 실행 함수
    """
    print("🎯 피쳐 개선 효과 측정 시작")
    print("="*50)

    # 개선 전 성과 측정
    print("\n[1/3] 개선 전 기준 성과 측정...")
    baseline = measure_baseline_performance()

    if baseline is None:
        print("❌ 기준 성과 측정 실패")
        return

    # 개선 후 성과 측정
    print("\n[2/3] 개선 후 성과 측정...")
    improved = measure_improved_performance()

    if improved is None:
        print("❌ 개선 성과 측정 실패")
        return

    # 비교 분석
    print("\n[3/3] 개선 전후 비교 분석...")
    comparison = compare_before_after(baseline, improved)

    # 결과 출력
    print("\n" + "="*50)
    print("📊 최종 결과 요약")
    print("="*50)
    print(f"기존 피쳐 수: {comparison['baseline_features']}")
    print(f"개선 피쳐 수: {comparison['improved_features']}")
    print(f"추가 피쳐 수: {comparison['new_features_added']}")
    print(f"IC 개선: {comparison['ic_improvement']:.4f}")
    print(f"Hit Ratio 개선: {comparison['hit_ratio_improvement']:.1%}")
    print(f"최고 IC 개선: {comparison['top_ic_improvement']:.4f}")
    print(f"평가: {comparison['overall_assessment']}")

    # 보고서 생성 및 저장
    report = generate_improvement_report(baseline, improved, comparison)

    # 저장
    cfg = load_config('configs/config.yaml')
    reports_dir = Path(cfg['paths']['base_dir']) / 'artifacts' / 'reports'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = reports_dir / f'feature_improvement_impact_{timestamp}.md'

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n💾 보고서 저장: {report_file}")

    print("\n✅ 피쳐 개선 효과 측정 완료!")
    print("="*50)


if __name__ == "__main__":
    main()
