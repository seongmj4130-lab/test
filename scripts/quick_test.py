#!/usr/bin/env python3
"""
빠른 테스트: 3개 전략 × 1개 기간씩
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import yaml
from run_dynamic_period_backtest import run_single_backtest


def main():
    # 설정 로드
    with open("configs/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("🚀 빠른 테스트: 개선 파라미터 적용 확인")
    print("=" * 50)

    results = []

    # 단기 20일
    print("1. 단기 전략 (bt20_short) 20일 테스트...")
    result1 = run_single_backtest(cfg, "bt20_short", 20)
    results.append(result1)

    # 통합 60일
    print("2. 통합 전략 (bt20_ens) 60일 테스트...")
    result2 = run_single_backtest(cfg, "bt20_ens", 60)
    results.append(result2)

    # 장기 120일
    print("3. 장기 전략 (bt120_long) 120일 테스트...")
    result3 = run_single_backtest(cfg, "bt120_long", 120)
    results.append(result3)

    # 결과 출력
    print("\n📊 빠른 테스트 결과:")
    print("-" * 30)
    for i, result in enumerate(results, 1):
        if result and "sharpe" in result:
            strategy_name = result["strategy"]
            holding_days = result["holding_days"]
            sharpe = result["sharpe"]
            cagr = result["cagr"]
            print(
                f"{i}. {strategy_name} {holding_days}일: Sharpe {sharpe:.3f}, CAGR {cagr:.2f}%"
            )
        else:
            print(f"{i}. 실패: 결과 없음")

    print("\n✅ 개선 파라미터 적용 확인 완료!")


if __name__ == "__main__":
    main()
