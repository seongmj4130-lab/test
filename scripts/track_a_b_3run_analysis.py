# -*- coding: utf-8 -*-
"""
Track A/B 3번 실행 분석 스크립트

현재 설정 기반으로 Track A/B 백테스트를 3번 반복 실행하고 결과를 분석합니다.
"""

from pathlib import Path
import shutil
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

def clear_l7_cache():
    """L7 백테스트 캐시만 삭제"""
    print("🧹 L7 백테스트 캐시 삭제 중...")

    interim_dir = PROJECT_ROOT / 'data' / 'interim'
    reports_dir = PROJECT_ROOT / 'artifacts' / 'reports'

    # L7 결과 파일 삭제
    l7_patterns = [
        'bt_metrics_*.parquet',
        'bt_metrics_*.csv',
        'bt_*_metrics*.parquet',
        'bt_*_metrics*.csv'
    ]

    for pattern in l7_patterns:
        for file_path in interim_dir.glob(f'**/{pattern}'):
            if file_path.is_file():
                file_path.unlink()
                print(f"  삭제: {file_path.name}")

    # 비교 결과 파일 삭제
    comparison_file = reports_dir / 'backtest_4models_comparison.csv'
    if comparison_file.exists():
        comparison_file.unlink()
        print(f"  삭제: {comparison_file.name}")

    print("✅ L7 캐시 삭제 완료")

def run_single_backtest(iteration_num):
    """단일 백테스트 실행"""
    print(f"\n{'='*50}")
    print(f"🔄 실행 #{iteration_num}")
    print(f"{'='*50}")

    try:
        # 4개 모델 백테스트 실행
        result = subprocess.run([
            sys.executable, 'scripts/run_backtest_4models.py'
        ], capture_output=True, text=True, cwd=PROJECT_ROOT)

        if result.returncode != 0:
            print(f"❌ 실행 #{iteration_num} 실패:")
            print(result.stderr[-300:])  # 마지막 300자만 출력
            return None

        print(f"✅ 실행 #{iteration_num} 완료")

        # 결과 파일 읽기
        result_file = PROJECT_ROOT / 'artifacts' / 'reports' / 'backtest_4models_comparison.csv'
        if result_file.exists():
            df = pd.read_csv(result_file)
            df['iteration'] = iteration_num
            return df
        else:
            print("⚠️ 결과 파일을 찾을 수 없음")
            return None

    except Exception as e:
        print(f"❌ 실행 #{iteration_num} 오류: {str(e)}")
        return None

def run_3_iterations():
    """3번 반복 실행"""
    print("🎯 Track A/B 3번 반복 실행 분석")
    print("="*70)
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = []

    for i in range(1, 4):  # 1, 2, 3번 실행
        # L7 캐시 삭제
        clear_l7_cache()

        # 백테스트 실행
        result_df = run_single_backtest(i)

        if result_df is not None:
            all_results.append(result_df)
            print(f"✅ #{i} 결과 수집 완료 ({len(result_df)}개 전략)")
        else:
            print(f"❌ #{i} 결과 수집 실패")

        # 실행 간격
        if i < 3:  # 마지막 실행이 아니면
            print("⏳ 다음 실행까지 3초 대기...")
            import time
            time.sleep(3)

    return all_results

def analyze_results(results_list):
    """결과 분석"""
    print(f"\n📊 Track A/B 3번 실행 결과 분석")
    print("="*70)

    if len(results_list) == 0:
        print("❌ 분석할 결과가 없습니다.")
        return

    # 결과를 하나로 합치기
    combined_df = pd.concat(results_list, ignore_index=True)

    # 전략별 분석
    strategies = combined_df['strategy'].unique()

    print(f"총 실행 횟수: {len(results_list)}")
    print(f"고유 전략 수: {len(strategies)}")
    print(f"총 결과 수: {len(combined_df)}")

    for strategy in strategies:
        strategy_data = combined_df[combined_df['strategy'] == strategy].copy()

        print(f"\n🎯 전략: {strategy}")
        print("-" * 40)

        # 각 실행의 결과 출력
        for iteration in [1, 2, 3]:
            iter_data = strategy_data[strategy_data['iteration'] == iteration]
            if len(iter_data) > 0:
                row = iter_data.iloc[0]
                print(f"실행 {iteration}: Sharpe {row['net_sharpe']:.4f}, "
                      f"CAGR {row['net_cagr']:.4f}, "
                      f"MDD {row['net_mdd']:.4f}, "
                      f"Calmar {row['net_calmar_ratio']:.4f}")

        # 통계 분석
        metrics = ['net_sharpe', 'net_cagr', 'net_mdd', 'net_calmar_ratio']
        print(f"\n통계 분석 (3번 실행):")

        for metric in metrics:
            if metric in strategy_data.columns:
                values = strategy_data[metric].values
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = std_val / abs(mean_val) if mean_val != 0 else 0
                min_val = np.min(values)
                max_val = np.max(values)

                print(f"  {metric}: 평균 {mean_val:.4f}, "
                      f"표준편차 {std_val:.4f}, "
                      f"CV {cv:.1%}, "
                      f"범위 [{min_val:.4f}, {max_val:.4f}]")

        # 재현성 평가
        sharpe_cv = np.std(strategy_data['net_sharpe']) / abs(np.mean(strategy_data['net_sharpe']))

        if sharpe_cv < 0.01:
            reproducibility = "⭐⭐⭐⭐⭐ EXCELLENT (완벽한 재현성)"
        elif sharpe_cv < 0.05:
            reproducibility = "⭐⭐⭐⭐ GOOD (우수한 재현성)"
        elif sharpe_cv < 0.10:
            reproducibility = "⭐⭐⭐ OK (양호한 재현성)"
        else:
            reproducibility = "⚠️ POOR (재현성 문제)"

        print(f"재현성 평가: {reproducibility}")

    # 전체 비교 테이블
    print(f"\n📋 전체 비교 테이블")
    print("="*70)

    pivot_table = combined_df.pivot_table(
        index='strategy',
        columns='iteration',
        values=['net_sharpe', 'net_cagr', 'net_mdd', 'net_calmar_ratio'],
        aggfunc='first'
    )

    print(pivot_table.to_string(float_format='%.4f'))

    # 결과 저장
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = PROJECT_ROOT / 'artifacts' / 'reports' / f'track_a_b_3run_analysis_{timestamp}.csv'
    combined_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"\n💾 결과 저장: {output_file}")

def main():
    """메인 함수"""
    # 3번 반복 실행
    results = run_3_iterations()

    # 결과 분석
    analyze_results(results)

    print(f"\n🏆 분석 완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()