# -*- coding: utf-8 -*-
"""
재현성 검증을 위한 3번 완전 재실행

L5-L7 파이프라인을 3번 완전히 재실행하여 결과 일관성 검증
"""

from pathlib import Path
import pandas as pd
import numpy as np
import shutil
import time
from datetime import datetime
import sys
import subprocess

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

def backup_existing_results():
    """기존 결과 백업"""
    print("📋 기존 결과 백업 중...")

    interim_dir = PROJECT_ROOT / 'data' / 'interim'
    backup_dir = PROJECT_ROOT / 'data' / 'backup_before_reproducibility_test'
    backup_dir.mkdir(parents=True, exist_ok=True)

    # 백업할 파일들
    files_to_backup = [
        'pred_short_oos.parquet',
        'pred_long_oos.parquet',
        'rebalance_scores.parquet',
        'rebalance_scores_original.parquet',
        'bt_metrics_bt20_ens.parquet',
        'bt_metrics_bt20_short.parquet',
        'bt_metrics_bt120_ens.parquet',
        'bt_metrics_bt120_long.parquet'
    ]

    for file in files_to_backup:
        src = interim_dir / file
        if src.exists():
            dst = backup_dir / f"{file}.backup"
            shutil.copy2(src, dst)
            print(f"  ✅ {file} 백업 완료")

    print("📋 백업 완료\n")

def clear_ml_cache():
    """ML 캐시 및 중간 결과 삭제"""
    print("🗑️ ML 캐시 및 중간 결과 삭제 중...")

    interim_dir = PROJECT_ROOT / 'data' / 'interim'

    # 삭제할 파일들 (L5-L7 관련)
    files_to_delete = [
        'pred_short_oos.parquet',
        'pred_long_oos.parquet',
        'rebalance_scores.parquet',
        'bt_metrics_bt20_ens.parquet',
        'bt_metrics_bt20_short.parquet',
        'bt_metrics_bt120_ens.parquet',
        'bt_metrics_bt120_long.parquet'
    ]

    for file in files_to_delete:
        file_path = interim_dir / file
        if file_path.exists():
            file_path.unlink()
            print(f"  ✅ {file} 삭제 완료")

    print("🗑️ 캐시 정리 완료\n")

def run_single_pipeline_run(run_id, seed=None):
    """단일 파이프라인 실행"""
    print(f"🚀 재실행 #{run_id} 시작")
    print("="*60)

    start_time = time.time()

    try:
        # L5-L7 파이프라인 실행
        cmd = [sys.executable, 'scripts/run_l5_l7_pipeline.py']
        if seed:
            cmd.extend(['--seed', str(seed)])

        result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)

        if result.returncode != 0:
            print(f"❌ 실행 #{run_id} 실패")
            print(f"stdout: {result.stdout}")
            print(f"stderr: {result.stderr}")
            return None

        # 백테스트 결과 로드
        interim_dir = PROJECT_ROOT / 'data' / 'interim'
        results = {}

        bt_files = [
            'bt_metrics_bt20_ens.parquet',
            'bt_metrics_bt20_short.parquet',
            'bt_metrics_bt120_ens.parquet',
            'bt_metrics_bt120_long.parquet'
        ]

        for bt_file in bt_files:
            file_path = interim_dir / bt_file
            if file_path.exists():
                df = pd.read_parquet(file_path)
                results[bt_file.replace('.parquet', '')] = df
            else:
                print(f"⚠️ {bt_file} 생성 실패")
                return None

        execution_time = time.time() - start_time
        print(".2f"
        return results, execution_time

    except Exception as e:
        print(f"❌ 실행 #{run_id} 예외 발생: {str(e)}")
        return None

def save_run_results(run_results, run_id):
    """실행 결과 저장"""
    results_dir = PROJECT_ROOT / 'artifacts' / 'reports' / 'reproducibility_test_results'
    results_dir.mkdir(parents=True, exist_ok=True)

    for strategy, df in run_results.items():
        filename = f"{strategy}_run_{run_id}.parquet"
        filepath = results_dir / filename
        df.to_parquet(filepath, index=False)
        print(f"  💾 {filename} 저장 완료")

def run_reproducibility_test():
    """재현성 테스트 실행"""
    print("🔬 재현성 검증 테스트 시작")
    print("="*80)
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. 기존 결과 백업
    backup_existing_results()

    # 2. 캐시 정리
    clear_ml_cache()

    # 3. 3번 재실행
    all_results = {}
    execution_times = {}

    for run_id in range(1, 4):
        print(f"\n{'='*80}")
        print(f"🔄 RUN {run_id}/3")
        print('='*80)

        # 서로 다른 시드로 재현성 테스트
        seed = 42 + run_id  # 43, 44, 45

        result = run_single_pipeline_run(run_id, seed)
        if result:
            run_results, exec_time = result
            all_results[f'run_{run_id}'] = run_results
            execution_times[f'run_{run_id}'] = exec_time

            # 결과 저장
            save_run_results(run_results, run_id)

            # 다음 실행 전 캐시 정리 (완전한 재현성 보장)
            if run_id < 3:
                clear_ml_cache()
        else:
            print(f"❌ RUN {run_id} 실패로 테스트 중단")
            return None

    return all_results, execution_times

def analyze_reproducibility_results(all_results, execution_times):
    """재현성 결과 분석"""
    print("
📊 재현성 분석 결과"    print("="*80)

    if not all_results:
        print("❌ 분석할 결과가 없습니다.")
        return

    # 각 전략별로 실행 간 차이 분석
    strategies = ['bt_metrics_bt20_ens', 'bt_metrics_bt20_short', 'bt_metrics_bt120_ens', 'bt_metrics_bt120_long']

    reproducibility_metrics = {}

    for strategy in strategies:
        print(f"\n🎯 {strategy} 재현성 분석")
        print("-" * 50)

        run_values = []
        for run_id in range(1, 4):
            run_key = f'run_{run_id}'
            if run_key in all_results and strategy in all_results[run_key]:
                df = all_results[run_key][strategy]
                if len(df) > 0:
                    # Holdout 결과만 사용 (더 안정적)
                    holdout_data = df[df['phase'] == 'holdout']
                    if len(holdout_data) > 0:
                        metrics = {
                            'sharpe': holdout_data['net_sharpe'].iloc[0],
                            'cagr': holdout_data['net_cagr'].iloc[0],
                            'mdd': holdout_data['net_mdd'].iloc[0],
                            'calmar': holdout_data['net_calmar_ratio'].iloc[0]
                        }
                        run_values.append(metrics)
                        print(f"  RUN {run_id}: Sharpe={metrics['sharpe']:.4f}, CAGR={metrics['cagr']:.4f}, MDD={metrics['mdd']:.4f}")

        if len(run_values) == 3:
            # 각 지표별 변동성 계산
            sharpe_values = [v['sharpe'] for v in run_values]
            cagr_values = [v['cagr'] for v in run_values]
            mdd_values = [v['mdd'] for v in run_values]
            calmar_values = [v['calmar'] for v in run_values]

            reproducibility_metrics[strategy] = {
                'sharpe_std': np.std(sharpe_values),
                'sharpe_cv': np.std(sharpe_values) / np.mean(sharpe_values) if np.mean(sharpe_values) != 0 else 0,
                'cagr_std': np.std(cagr_values),
                'cagr_cv': np.std(cagr_values) / np.mean(cagr_values) if np.mean(cagr_values) != 0 else 0,
                'mdd_std': np.std(mdd_values),
                'mdd_cv': np.std(mdd_values) / abs(np.mean(mdd_values)) if np.mean(mdd_values) != 0 else 0,
                'calmar_std': np.std(calmar_values),
                'calmar_cv': np.std(calmar_values) / np.mean(calmar_values) if np.mean(calmar_values) != 0 else 0
            }

            print(".6f"            print(".4f"            print(".6f"            print(".4f"            print(".6f"            print(".4f"            print(".6f"            print(".4f"        else:
            print(f"  ❌ {strategy}: 3번 실행 결과 중 {len(run_values)}개만 성공")

    # 종합 재현성 평가
    print("
🏆 종합 재현성 평가"    print("="*80)

    if reproducibility_metrics:
        # 평균 변동계수로 재현성 평가
        avg_cv_sharpe = np.mean([m['sharpe_cv'] for m in reproducibility_metrics.values()])
        avg_cv_cagr = np.mean([m['cagr_cv'] for m in reproducibility_metrics.values()])
        avg_cv_mdd = np.mean([m['mdd_cv'] for m in reproducibility_metrics.values()])
        avg_cv_calmar = np.mean([m['calmar_cv'] for m in reproducibility_metrics.values()])

        print(".4f"        print(".4f"        print(".4f"        print(".4f"
        # 재현성 등급 평가
        def get_reproducibility_grade(cv):
            if cv < 0.01: return "⭐⭐⭐⭐⭐ 완벽"
            elif cv < 0.05: return "⭐⭐⭐⭐ 우수"
            elif cv < 0.10: return "⭐⭐⭐ 양호"
            elif cv < 0.20: return "⭐⭐ 보통"
            else: return "⭐ 개선 필요"

        print("
📋 재현성 등급:"        print(f"  Sharpe: {get_reproducibility_grade(avg_cv_sharpe)}")
        print(f"  CAGR: {get_reproducibility_grade(avg_cv_cagr)}")
        print(f"  MDD: {get_reproducibility_grade(avg_cv_mdd)}")
        print(f"  Calmar: {get_reproducibility_grade(avg_cv_calmar)}")

        # 실행 시간 분석
        print("
⏱️ 실행 시간 분석:"        for run_id in range(1, 4):
            run_key = f'run_{run_id}'
            if run_key in execution_times:
                print(".2f"
        avg_time = np.mean(list(execution_times.values()))
        time_std = np.std(list(execution_times.values()))
        print(".2f"        print(".2f"
        # 최종 결론
        overall_cv = np.mean([avg_cv_sharpe, avg_cv_cagr, avg_cv_mdd, avg_cv_calmar])
        if overall_cv < 0.05:
            conclusion = "✅ 재현성 우수: 3번 재실행 결과가 매우 일관됨"
        elif overall_cv < 0.10:
            conclusion = "🟡 재현성 양호: 약간의 변동성 있지만 안정적"
        elif overall_cv < 0.20:
            conclusion = "🟠 재현성 보통: 추가 검토 필요"
        else:
            conclusion = "❌ 재현성 저조: 시스템 개선 필요"

        print(f"\n🎯 최종 결론: {conclusion}")
        print(".4f"
    else:
        print("❌ 재현성 분석 실패: 결과 데이터 부족")

def restore_backup():
    """백업 파일 복원"""
    print("
🔄 백업 파일 복원 중..."    backup_dir = PROJECT_ROOT / 'data' / 'backup_before_reproducibility_test'
    interim_dir = PROJECT_ROOT / 'data' / 'interim'

    if backup_dir.exists():
        backup_files = list(backup_dir.glob('*.backup'))
        for backup_file in backup_files:
            original_name = backup_file.name.replace('.backup', '')
            dst = interim_dir / original_name
            shutil.copy2(backup_file, dst)
            print(f"  ✅ {original_name} 복원 완료")

        print("🔄 백업 복원 완료")
    else:
        print("⚠️ 백업 디렉토리가 없습니다.")

def main():
    """메인 함수"""
    try:
        # 재현성 테스트 실행
        test_results = run_reproducibility_test()

        if test_results:
            all_results, execution_times = test_results

            # 결과 분석
            analyze_reproducibility_results(all_results, execution_times)

        # 백업 복원
        restore_backup()

        print(f"\n🏆 재현성 검증 테스트 완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    except Exception as e:
        print(f"❌ 테스트 실패: {str(e)}")
        import traceback
        traceback.print_exc()

        # 오류 발생 시 백업 복원
        restore_backup()

if __name__ == "__main__":
    main()