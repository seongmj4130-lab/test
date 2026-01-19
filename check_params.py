#!/usr/bin/env python3
"""
통합 최적화 적용 후 파라미터 확인
"""

import yaml


def main():
    with open('configs/config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print('🎯 통합 최적화 적용 후 파라미터 확인')
    print('=' * 50)

    strategies = ['l7_bt20_short', 'l7_bt20_ens', 'l7_bt120_long']
    for strategy in strategies:
        if strategy in config:
            params = config[strategy]
            print(f'\n{strategy}:')
            print(f'  top_k: {params.get("top_k")}')
            print(f'  cost_bps: {params.get("cost_bps")}')
            print(f'  target_volatility: {params.get("target_volatility")}')
            print(f'  buffer_k: {params.get("buffer_k")}')

    print('\n✅ 파라미터 적용 완료! 백테스트 실행 준비됨')

if __name__ == "__main__":
    main()
