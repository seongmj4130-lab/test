# -*- coding: utf-8 -*-
"""
Track A/B 백테스트 재현성 검증 스크립트

3번 반복 실행하여 Track A/B 백테스트의 재현성을 검증합니다.
L0~L6 데이터는 고정시키고 L7 백테스트만 반복 실행합니다.
"""

from pathlib import Path
import shutil
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime
import yaml
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

def clear_backtest_cache():
    """백테스트 캐시 삭제 (L7 결과만)"""
    print("🧹 백테스트 캐시 데이터 삭제 중...")

    # L7 백테스트 결과 파일만 삭제
    interim_dir = PROJECT_ROOT / 'data' / 'interim'
    reports_dir = PROJECT_ROOT / 'artifacts' / 'reports'

    # 삭제할 파일 패턴들 (L7 결과만)
    patterns_to_remove = [
        'bt_metrics_*.parquet',
        'bt_metrics_*.csv',
        'bt_*_metrics*.parquet',
        'bt_*_metrics*.csv'
    ]

    # interim 폴더에서 삭제
    if interim_dir.exists():
        for pattern in patterns_to_remove:
            for file_path in interim_dir.glob(f'**/{pattern}'):
                if file_path.is_file():
                    file_path.unlink()
                    print(f"  삭제: {file_path.name}")

    # reports 폴더에서 비교 파일 삭제
    if reports_dir.exists():
        comparison_file = reports_dir / 'backtest_4models_comparison.csv'
        if comparison_file.exists():
            comparison_file.unlink()
            print(f"  삭제: {comparison_file.name}")

    print("✅ 백테스트 캐시 데이터 삭제 완료")

def run_backtest_iteration(iteration_num):
    """단일 백테스트 실행 (L0~L6 고정, L7만 재실행)"""
    print(f"\n{'='*60}")
    print(f"🔄 백테스트 반복 실행 #{iteration_num}")
    print(f"{'='*60}")

    try:
        # 4개 모델 백테스트 실행 (L0~L6 데이터는 유지)
        result = subprocess.run([
            sys.executable, 'scripts/run_backtest_4models.py'
        ], capture_output=True, text=True, cwd=PROJECT_ROOT)

        if result.returncode != 0:
            print(f"❌ 백테스트 #{iteration_num} 실패:")
            print(result.stderr)
            return None

        print(f"✅ 백테스트 #{iteration_num} 완료")

        # 결과 파일 읽기
        result_file = PROJECT_ROOT / 'artifacts' / 'reports' / 'backtest_4models_comparison.csv'
        if result_file.exists():
            df = pd.read_csv(result_file)
            df['iteration'] = iteration_num
            return df
        else:
            print(f"⚠️ 결과 파일을 찾을 수 없음: {result_file}")
            return None

    except Exception as e:
        print(f"❌ 백테스트 #{iteration_num} 오류: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def analyze_backtest_reproducibility(results_df):
    """백테스트 재현성 분석"""
    print("\n🔍 Track A/B 백테스트 재현성 분석")
    print("="*80)

    if results_df is None or len(results_df) == 0:
        print("❌ 분석할 데이터가 없습니다.")
        return

    # 전략별 분석
    strategies = results_df['strategy'].unique()

    for strategy in strategies:
        strategy_data = results_df[results_df['strategy'] == strategy]

        print(f"\n🎯 전략: {strategy}")
        print("-" * 40)

        # 주요 메트릭의 통계
        key_metrics = ['net_sharpe', 'net_cagr', 'net_mdd', 'net_calmar_ratio']

        reproducibility_issues = []

        for metric in key_metrics:
            if metric in strategy_data.columns:
                values = strategy_data[metric].values
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = std_val / abs(mean_val) if mean_val != 0 else 0

                print("12.4f"
                      "6.4f"
                      "8.1%")

                # 재현성 평가 (Sharpe는 5%, 다른 지표는 10% 변동 허용)
                threshold = 0.05 if 'sharpe' in metric else 0.10
                if cv > threshold:
                    reproducibility_issues.append(f"{metric}: {cv:.1%}")

        # 재현성 평가
        if not reproducibility_issues:
            reproducibility = "⭐⭐⭐⭐⭐ EXCELLENT"
        elif len(reproducibility_issues) <= 1:
            reproducibility = "⭐⭐⭐⭐ GOOD"
        elif len(reproducibility_issues) <= 2:
            reproducibility = "⭐⭐⭐ OK"
        else:
            reproducibility = "⚠️ POOR"

        print(f"재현성 평가: {reproducibility} (문제 지표: {len(reproducibility_issues)}개)")
        if reproducibility_issues:
            for issue in reproducibility_issues:
                print(f"  - {issue}")

    # 전체 요약
    print("\n📊 전체 재현성 요약")
    print("="*50)

    total_runs = len(results_df)
    unique_results = len(results_df.drop_duplicates(subset=['strategy', 'net_sharpe', 'net_cagr', 'net_mdd']))

    print(f"총 실행 횟수: {total_runs}")
    print(f"고유 결과 수: {unique_results}")
    print(f"결과 일관성: {unique_results}/{total_runs} ({unique_results/total_runs:.1%})")

    if unique_results == total_runs:
        print("결론: ✅ 완벽한 재현성 (모든 실행에서 동일 결과)")
    elif unique_results >= total_runs * 0.9:
        print("결론: ⚠️ 양호한 재현성 (90% 이상 일관성)")
    elif unique_results >= total_runs * 0.7:
        print("결론: ❌ 재현성 문제 (70-90% 일관성)")
    else:
        print("결론: ❌❌ 심각한 재현성 문제 (70% 미만 일관성)")

def run_reproducibility_test(n_iterations=3):
    """Track A/B 백테스트 재현성 검증 메인 함수"""
    print("🔬 Track A/B 백테스트 재현성 검증 시작")
    print(f"반복 횟수: {n_iterations}회")
    print("="*80)

    all_results = []

    for i in range(1, n_iterations + 1):
        # 백테스트 캐시 데이터 삭제
        clear_backtest_cache()

        # 백테스트 실행
        result_df = run_backtest_iteration(i)

        if result_df is not None:
            all_results.append(result_df)
            print(f"✅ #{i} 실행 결과 수집 완료")
        else:
            print(f"❌ #{i} 실행 실패")

        print(f"\n⏳ 다음 실행까지 5초 대기...")
        import time
        time.sleep(5)

    # 결과 분석
    if len(all_results) == 0:
        print("❌ 모든 실행이 실패했습니다.")
        return None

    # 결과 통합
    combined_results = pd.concat(all_results, ignore_index=True)

    # 결과 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = PROJECT_ROOT / 'artifacts' / 'reports' / f'backtest_reproducibility_test_{timestamp}.csv'
    combined_results.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"\n✅ 재현성 검증 완료!")
    print(f"📊 결과 파일: {output_file}")
    print(f"📈 총 결과 수: {len(combined_results)}개")

    return combined_results

if __name__ == "__main__":
    # 재현성 검증 실행
    results = run_reproducibility_test(n_iterations=3)

    # 결과 분석
    analyze_backtest_reproducibility(results)