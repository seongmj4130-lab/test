"""
재현성 검증 간단 버전

L6-L7 파이프라인을 3번 재실행하여 백테스트 결과 일관성 검증
"""

import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

def backup_existing_results():
    """기존 결과 백업"""
    print("📋 기존 백테스트 결과 백업 중...")

    interim_dir = PROJECT_ROOT / 'data' / 'interim'
    backup_dir = PROJECT_ROOT / 'data' / 'backup_reproducibility'
    backup_dir.mkdir(parents=True, exist_ok=True)

    # 백업할 파일들
    bt_files = [
        'bt_metrics_bt20_ens.parquet',
        'bt_metrics_bt20_short.parquet',
        'bt_metrics_bt120_ens.parquet',
        'bt_metrics_bt120_long.parquet'
    ]

    for file in bt_files:
        src = interim_dir / file
        if src.exists():
            dst = backup_dir / f"{file}.backup"
            shutil.copy2(src, dst)
            print(f"  ✅ {file} 백업 완료")

    print("📋 백업 완료\n")

def clear_backtest_cache():
    """백테스트 캐시 삭제"""
    print("🗑️ 백테스트 캐시 삭제 중...")

    interim_dir = PROJECT_ROOT / 'data' / 'interim'

    # 삭제할 파일들
    bt_files = [
        'bt_metrics_bt20_ens.parquet',
        'bt_metrics_bt20_short.parquet',
        'bt_metrics_bt120_ens.parquet',
        'bt_metrics_bt120_long.parquet'
    ]

    for file in bt_files:
        file_path = interim_dir / file
        if file_path.exists():
            file_path.unlink()
            print(f"  ✅ {file} 삭제 완료")

    print("🗑️ 캐시 정리 완료\n")

def run_single_backtest(run_id):
    """단일 백테스트 실행"""
    print(f"🚀 백테스트 재실행 #{run_id} 시작")
    print("="*60)

    start_time = time.time()

    try:
        # 백테스트 실행
        cmd = [sys.executable, 'scripts/run_backtest_4models.py']
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)

        if result.returncode != 0:
            print(f"❌ 실행 #{run_id} 실패")
            print(f"stderr: {result.stderr[-500:]}")  # 마지막 500자만 출력
            return None

        # 결과 로드
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
    results_dir = PROJECT_ROOT / 'artifacts' / 'reports' / 'reproducibility_runs'
    results_dir.mkdir(parents=True, exist_ok=True)

    for strategy, df in run_results.items():
        filename = f"{strategy}_run_{run_id}.parquet"
        filepath = results_dir / filename
        df.to_parquet(filepath, index=False)

def run_reproducibility_test():
    """재현성 테스트 실행"""
    print("🔬 재현성 검증 테스트 시작 (간단 버전)")
    print("="*80)
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("📝 테스트 방식: L6-L7 파이프라인 3번 재실행")
    print("📝 주의: L0-L5 데이터는 기존 데이터 사용 (외부 API 의존성으로 인한 제약)")

    # 1. 기존 결과 백업
    backup_existing_results()

    # 2. 3번 재실행
    all_results = {}
    execution_times = {}

    for run_id in range(1, 4):
        print(f"\n{'='*80}")
        print(f"🔄 RUN {run_id}/3")
        print('='*80)

        # 캐시 정리 (완전한 재현성 보장)
        if run_id > 1:
            clear_backtest_cache()

        result = run_single_backtest(run_id)
        if result:
            run_results, exec_time = result
            all_results[f'run_{run_id}'] = run_results
            execution_times[f'run_{run_id}'] = exec_time

            # 결과 저장
            save_run_results(run_results, run_id)

        else:
            print(f"❌ RUN {run_id} 실패로 테스트 중단")
            return None

    return all_results, execution_times

def analyze_reproducibility(all_results, execution_times):
    """재현성 분석"""
    print("
📊 재현성 분석 결과"    print("="*80)

    if not all_results:
        print("❌ 분석할 결과가 없습니다.")
        return

    # 각 전략별로 실행 간 차이 분석
    strategies = ['bt_metrics_bt20_ens', 'bt_metrics_bt20_short', 'bt_metrics_bt120_ens', 'bt_metrics_bt120_long']

        print("\n🔍 각 실행의 Holdout 성과 비교")    print("-" * 80)

    reproducibility_data = []

    for strategy in strategies:
        print(f"\n🎯 {strategy}")
        print("-" * 40)

        run_sharpes = []
        run_cagrs = []
        run_mdds = []
        run_calmars = []

        for run_id in range(1, 4):
            run_key = f'run_{run_id}'
            if run_key in all_results and strategy in all_results[run_key]:
                df = all_results[run_key][strategy]

                # Holdout 결과만 사용
                holdout_data = df[df['phase'] == 'holdout']
                if len(holdout_data) > 0:
                    sharpe = holdout_data['net_sharpe'].iloc[0]
                    cagr = holdout_data['net_cagr'].iloc[0]
                    mdd = holdout_data['net_mdd'].iloc[0]
                    calmar = holdout_data['net_calmar_ratio'].iloc[0]

                    run_sharpes.append(sharpe)
                    run_cagrs.append(cagr)
                    run_mdds.append(mdd)
                    run_calmars.append(calmar)

                    print(f"  RUN {run_id}: Sharpe={sharpe:.4f}, CAGR={cagr:.4f}, MDD={mdd:.4f}, Calmar={calmar:.4f}")

        if len(run_sharpes) == 3:
            # 변동성 계산
            sharpe_std = np.std(run_sharpes)
            sharpe_cv = sharpe_std / abs(np.mean(run_sharpes)) if np.mean(run_sharpes) != 0 else 0

            cagr_std = np.std(run_cagrs)
            cagr_cv = cagr_std / abs(np.mean(run_cagrs)) if np.mean(run_cagrs) != 0 else 0

            mdd_std = np.std(run_mdds)
            mdd_cv = mdd_std / abs(np.mean(run_mdds)) if np.mean(run_mdds) != 0 else 0

            calmar_std = np.std(run_calmars)
            calmar_cv = calmar_std / abs(np.mean(run_calmars)) if np.mean(run_calmars) != 0 else 0

            reproducibility_data.append({
                'strategy': strategy,
                'sharpe_cv': sharpe_cv,
                'cagr_cv': cagr_cv,
                'mdd_cv': mdd_cv,
                'calmar_cv': calmar_cv,
                'sharpe_std': sharpe_std,
                'cagr_std': cagr_std,
                'mdd_std': mdd_std,
                'calmar_std': calmar_std
            })

            print(".4f"            print(".4f"            print(".4f"            print(".4f"    # 종합 분석
    if reproducibility_data:
        print("
🏆 종합 재현성 평가"        print("="*80)

        df_repro = pd.DataFrame(reproducibility_data)

        # 평균 변동계수
        avg_cv_sharpe = df_repro['sharpe_cv'].mean()
        avg_cv_cagr = df_repro['cagr_cv'].mean()
        avg_cv_mdd = df_repro['mdd_cv'].mean()
        avg_cv_calmar = df_repro['calmar_cv'].mean()

        print(".4f"        print(".4f"        print(".4f"        print(".4f"
        # 재현성 등급
        def get_reproducibility_grade(cv):
            if cv < 0.001: return "⭐⭐⭐⭐⭐ 완벽 (완전 일치)"
            elif cv < 0.005: return "⭐⭐⭐⭐⭐ 우수 (극미한 변동)"
            elif cv < 0.01: return "⭐⭐⭐⭐ 우수"
            elif cv < 0.05: return "⭐⭐⭐ 양호"
            elif cv < 0.10: return "⭐⭐ 보통"
            elif cv < 0.20: return "⭐ 개선 필요"
            else: return "❌ 심각한 문제"

        print("
📋 재현성 등급:"        print(f"  Sharpe Ratio: {get_reproducibility_grade(avg_cv_sharpe)}")
        print(f"  CAGR: {get_reproducibility_grade(avg_cv_cagr)}")
        print(f"  MDD: {get_reproducibility_grade(avg_cv_mdd)}")
        print(f"  Calmar Ratio: {get_reproducibility_grade(avg_cv_calmar)}")

        # 실행 시간 분석
        print("
⏱️ 실행 시간 분석:"        for run_id, exec_time in execution_times.items():
            print(".2f"
        avg_time = np.mean(list(execution_times.values()))
        time_std = np.std(list(execution_times.values()))
        print(".2f"        print(".2f"
        # 최고 성과 전략
        print("
🏅 최고 성과 전략 (평균 기준):"        best_sharpe_strategy = df_repro.loc[df_repro['sharpe_std'].idxmin(), 'strategy']
        print(f"  Sharpe 안정성 최고: {best_sharpe_strategy}")

        # 최종 결론
        overall_cv = np.mean([avg_cv_sharpe, avg_cv_cagr, avg_cv_mdd, avg_cv_calmar])

        if overall_cv < 0.005:
            conclusion = "✅ 재현성 완벽: 3번 재실행 결과가 거의 동일함 (시스템 안정성 우수)"
            grade = "A+"
        elif overall_cv < 0.01:
            conclusion = "✅ 재현성 우수: 3번 재실행 결과가 매우 일관됨"
            grade = "A"
        elif overall_cv < 0.05:
            conclusion = "🟡 재현성 양호: 약간의 변동성 있지만 안정적"
            grade = "B"
        elif overall_cv < 0.10:
            conclusion = "🟠 재현성 보통: 추가 모니터링 필요"
            grade = "C"
        else:
            conclusion = "❌ 재현성 저조: 시스템 개선 필요"
            grade = "D"

        print(f"\n🎯 최종 재현성 등급: {grade}")
        print(f"🎯 종합 평가: {conclusion}")
        print(".4f"
        # 상세 결과 저장
        results_summary = {
            'test_timestamp': datetime.now().isoformat(),
            'reproducibility_grade': grade,
            'overall_cv': overall_cv,
            'avg_cv_sharpe': avg_cv_sharpe,
            'avg_cv_cagr': avg_cv_cagr,
            'avg_cv_mdd': avg_cv_mdd,
            'avg_cv_calmar': avg_cv_calmar,
            'avg_execution_time': avg_time,
            'execution_time_std': time_std,
            'strategies_tested': len(strategies),
            'runs_completed': 3,
            'data_used': 'L0-L4: 기존 데이터, L5-L7: 재실행'
        }

        summary_file = PROJECT_ROOT / 'artifacts' / 'reports' / 'reproducibility_test_summary.json'
        import json
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n💾 상세 결과 저장: {summary_file}")

def restore_backup():
    """백업 파일 복원"""
    print("
🔄 백업 파일 복원 중..."    backup_dir = PROJECT_ROOT / 'data' / 'backup_reproducibility'
    interim_dir = PROJECT_ROOT / 'data' / 'interim'

    if backup_dir.exists():
        backup_files = list(backup_dir.glob('*.backup'))
        restored_count = 0
        for backup_file in backup_files:
            original_name = backup_file.name.replace('.backup', '')
            dst = interim_dir / original_name
            shutil.copy2(backup_file, dst)
            restored_count += 1

        print(f"🔄 {restored_count}개 파일 복원 완료")
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
            analyze_reproducibility(all_results, execution_times)

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
