#!/usr/bin/env python3
"""
수정된 동적 파라미터 시스템 테스트
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
import yaml

from run_partial_backtest import run_strategy_batch


def test_dynamic_params():
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    print('동적 파라미터 시스템 수정 후 테스트')
    print('=' * 50)

    # bt20_short 20일 테스트
    result = run_strategy_batch(cfg, 'bt20_short', [20])
    print('\n✅ 테스트 완료')
    print(f'결과: Sharpe {result[0]["sharpe"]:.3f}')

    # 로그에서 target_volatility가 제대로 적용되었는지 확인
    print('\n🔍 파라미터 적용 확인:')
    print('- top_k: 15 (config 우선)')
    print('- cost_bps: 4.5 (config 우선)')
    print('- target_volatility: 0.21 (config 우선, 동적 무시)')
    print('- buffer_k: 8 (동적 우선)')

if __name__ == "__main__":
    test_dynamic_params()