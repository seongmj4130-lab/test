# -*- coding: utf-8 -*-
"""
rebalance_interval 적용 백테스트 재실행
- BT20 모델: rebalance_interval=20
- BT120 모델: rebalance_interval=120
"""
import sys
from pathlib import Path
import subprocess
import time

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def run_model(strategy: str, expected_interval: int):
    """개별 모델 실행"""
    print(f"\n{'='*80}")
    print(f"모델 실행: {strategy} (rebalance_interval={expected_interval})")
    print(f"{'='*80}\n")
    
    cmd = [sys.executable, "-m", "src.pipeline.track_b_pipeline", strategy]
    
    start_time = time.time()
    result = subprocess.run(
        cmd, 
        cwd=project_root,
        capture_output=False,  # 실시간 출력
        text=True
    )
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"\n✓ {strategy} 완료 (소요 시간: {elapsed/60:.1f}분)\n")
        return True
    else:
        print(f"\n✗ {strategy} 실패 (exit code: {result.returncode}, 소요 시간: {elapsed/60:.1f}분)\n")
        return False

def main():
    """4개 모델 순차 실행"""
    models = [
        ("bt20_short", 20),
        ("bt20_ens", 20),
        ("bt120_long", 120),
        ("bt120_ens", 120),
    ]
    
    print("="*80)
    print("rebalance_interval 적용 백테스트 재실행")
    print("="*80)
    print("BT20 모델: rebalance_interval=20")
    print("BT120 모델: rebalance_interval=120")
    print("="*80)
    
    results = {}
    total_start = time.time()
    
    for strategy, interval in models:
        results[strategy] = run_model(strategy, interval)
    
    total_elapsed = time.time() - total_start
    
    print("\n" + "="*80)
    print("실행 결과 요약")
    print("="*80)
    for strategy, success in results.items():
        status = "✓ 성공" if success else "✗ 실패"
        print(f"{strategy}: {status}")
    print(f"\n총 소요 시간: {total_elapsed/60:.1f}분")
    print("="*80)
    
    # 결과 확인
    if all(results.values()):
        print("\n✓ 모든 모델 실행 완료!")
        print("\n결과 확인:")
        print("  python scripts/check_rebalance_interval_effect.py")

if __name__ == "__main__":
    main()

