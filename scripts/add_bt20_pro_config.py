# -*- coding: utf-8 -*-
"""
bt20_pro 설정을 config.yaml에 추가하는 스크립트
"""

import yaml
from pathlib import Path


def add_bt20_pro_config():
    """
    config.yaml에 bt20_pro 설정 추가
    """
    config_path = Path('configs/config.yaml')

    # 현재 config 로드
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # bt20_pro 설정 추가
    bt20_pro_config = {
        'holding_days': 20,
        'top_k': 12,
        'cost_bps': 10.0,
        'slippage_bps': 0.0,
        'buffer_k': 15,
        'weighting': 'equal',
        'score_col': 'score_total_short',  # 단기 랭킹만 사용
        'return_col': 'true_short',
        'rebalance_interval': 1,  # 기본값 (적응형 로직에서 조정)
        'smart_buffer_enabled': True,
        'smart_buffer_stability_threshold': 0.7,
        'volatility_adjustment_enabled': True,
        'volatility_lookback_days': 60,
        'target_volatility': 0.15,
        'volatility_adjustment_max': 1.2,
        'volatility_adjustment_min': 0.7,
        'risk_scaling_enabled': True,
        'risk_scaling_bear_multiplier': 0.8,
        'risk_scaling_neutral_multiplier': 1.0,
        'risk_scaling_bull_multiplier': 1.0,
        'signal_source': 'model',
        'ranking_score_source': 'score_total',
        # [bt20 프로페셔널] 적응형 리밸런싱 설정
        'adaptive_rebalancing_enabled': True,  # 적응형 리밸런싱 활성화
        'signal_strength_thresholds': {
            'strong': 0.8,    # 80점 이상: 15일 리밸런싱
            'medium': 0.6,    # 60-79점: 20일 리밸런싱
            'weak': 0.6       # 60점 미만: 25일 리밸런싱
        },
        'rebalance_intervals': {
            'strong': 15,     # 강한 시그널: 15일
            'medium': 20,     # 중간 시그널: 20일
            'weak': 25        # 약한 시그널: 25일
        },
        'signal_strength_calculation': {
            'method': 'rolling_ic',  # 롤링 IC 기반 계산
            'window_days': 60,       # 60일 롤링 윈도우
            'min_periods': 20        # 최소 기간
        },
        'diversify': {
            'enabled': True,
            'group_col': 'sector_name',
            'max_names_per_group': 4
        },
        'regime': {
            'enabled': True,
            'exposure_bull_weak': 1.2,
            'exposure_bear_strong': 0.6,
            'exposure_bear_weak': 0.8,
            'exposure_neutral': 1.0,
            'top_k_bull': 15,
            'top_k_bear': 30,
            'exposure_bull': 1.0,
            'exposure_bear': 1.0
        }
    }

    # config에 bt20_pro 추가
    config['l7_bt20_pro'] = bt20_pro_config

    # 저장
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print("✅ bt20_pro 설정이 config.yaml에 추가되었습니다!")
    print("설정 키:", list(config.keys())[-1])  # 마지막에 추가된 키
    return True


def verify_bt20_pro_config():
    """
    bt20_pro 설정 검증
    """
    config_path = Path('configs/config.yaml')

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    if 'l7_bt20_pro' in config:
        bt20_pro = config['l7_bt20_pro']
        print("✅ bt20_pro 설정 검증 성공!")
        print(f"  - score_col: {bt20_pro.get('score_col')}")
        print(f"  - adaptive_rebalancing_enabled: {bt20_pro.get('adaptive_rebalancing_enabled')}")
        print(f"  - signal_strength_thresholds: {bt20_pro.get('signal_strength_thresholds', {})}")
        return True
    else:
        print("❌ bt20_pro 설정을 찾을 수 없습니다.")
        return False


if __name__ == "__main__":
    print("🔧 bt20_pro 설정 추가 스크립트")
    print("="*40)

    # 설정 추가
    success = add_bt20_pro_config()

    if success:
        # 검증
        print("\n🔍 설정 검증...")
        verify_bt20_pro_config()

        print("\n🎯 다음 단계:")
        print("1. python -m src.pipeline.track_b_pipeline  # bt20_pro 백테스트 실행")
        print("2. python scripts/show_backtest_metrics.py  # 결과 확인")
    else:
        print("❌ 설정 추가 실패")