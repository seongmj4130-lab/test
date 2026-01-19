#!/usr/bin/env python3
"""
동적 파라미터 시스템 수정 후 통합 최적화 재실행
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import yaml

from run_partial_backtest import run_strategy_batch


def run_integrated_optimization():
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    print('동적 파라미터 시스템 수정 후 통합 최적화 재실행')
    print('=' * 60)

    # bt20_short 전체 6개 기간 실행
    print('bt20_short 실행 중...')
    result_short = run_strategy_batch(cfg, 'bt20_short', [20, 40, 60, 80, 100, 120])
    print('bt20_short 완료')

    # bt20_ens 실행
    print('bt20_ens 실행 중...')
    result_ens = run_strategy_batch(cfg, 'bt20_ens', [20, 40, 60, 80, 100, 120])
    print('bt20_ens 완료')

    # bt120_long 실행
    print('bt120_long 실행 중...')
    result_long = run_strategy_batch(cfg, 'bt120_long', [20, 40, 60, 80, 100, 120])
    print('bt120_long 완료')

    print('모든 전략 통합 최적화 완료!')

if __name__ == "__main__":
    run_integrated_optimization()
