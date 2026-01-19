# -*- coding: utf-8 -*-
"""
4개 모델을 rebalance_interval 설정으로 백테스트 재실행
- BT20 모델: rebalance_interval=20
- BT120 모델: rebalance_interval=120
"""
import subprocess
import sys
from pathlib import Path

def run_model(strategy: str):
    """개별 모델 실행"""
    print(f"\n{'='*80}")
    print(f"모델 실행: {strategy}")
    print(f"{'='*80}\n")
    
    cmd = [sys.executable, "-m", "src.pipeline.track_b_pipeline", strategy]
    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)
    
    if result.returncode == 0:
        print(f"\n✓ {strategy} 완료\n")
    else:
        print(f"\n✗ {strategy} 실패 (exit code: {result.returncode})\n")
    
    return result.returncode == 0

def main():
    """4개 모델 순차 실행"""
    models = ["bt20_short", "bt20_ens", "bt120_long", "bt120_ens"]
    
    print("="*80)
    print("rebalance_interval 적용 백테스트 재실행")
    print("="*80)
    print("BT20 모델: rebalance_interval=20")
    print("BT120 모델: rebalance_interval=120")
    print("="*80)
    
    results = {}
    for strategy in models:
        results[strategy] = run_model(strategy)
    
    print("\n" + "="*80)
    print("실행 결과 요약")
    print("="*80)
    for strategy, success in results.items():
        status = "✓ 성공" if success else "✗ 실패"
        print(f"{strategy}: {status}")
    print("="*80)

if __name__ == "__main__":
    main()

