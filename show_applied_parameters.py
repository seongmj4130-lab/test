import yaml
import os

def show_applied_parameters():
    """실제 적용된 Track A와 Track B 파라미터들을 보여줌"""

    print("🎯 실제 적용된 Track A & Track B 파라미터 전체 현황")
    print("=" * 80)

    # config.yaml 로드
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("\n📊 Track A: 모델링 파라미터 (L5 + L6)")
    print("-" * 60)

    # L5 파라미터
    l5 = config.get('l5', {})
    print("\n🔹 L5 (모델 학습):")
    l5_params = {
        'model_type': l5.get('model_type', 'N/A'),
        'ridge_alpha': l5.get('ridge_alpha', 'N/A'),
        'target_transform': l5.get('target_transform', 'N/A'),
        'feature_weights_config_short': l5.get('feature_weights_config_short', 'N/A'),
        'feature_weights_config_long': l5.get('feature_weights_config_long', 'N/A'),
        'filter_features_by_ic': l5.get('filter_features_by_ic', 'N/A'),
        'min_feature_ic': l5.get('min_feature_ic', 'N/A'),
        'use_rank_ic': l5.get('use_rank_ic', 'N/A'),
        'tune_metric': l5.get('tune_metric', 'N/A')
    }

    for param, value in l5_params.items():
        print(f"  • {param}: {value}")

    # L6 파라미터
    l6 = config.get('l6', {})
    print("\n🔹 L6 (스코어 결합):")
    l6_params = {
        'weight_long': l6.get('weight_long', 'N/A'),
        'weight_short': l6.get('weight_short', 'N/A'),
        'invert_score_sign': l6.get('invert_score_sign', 'N/A')
    }

    for param, value in l6_params.items():
        print(f"  • {param}: {value}")

    # L4 파라미터 (교차검증)
    l4 = config.get('l4', {})
    print("\n🔹 L4 (교차검증):")
    l4_params = {
        'horizon_short': l4.get('horizon_short', 'N/A'),
        'horizon_long': l4.get('horizon_long', 'N/A'),
        'step_days': l4.get('step_days', 'N/A'),
        'embargo_days': l4.get('embargo_days', 'N/A'),
        'inner_cv_k': l4.get('inner_cv_k', 'N/A'),
        'rolling_train_years_short': l4.get('rolling_train_years_short', 'N/A'),
        'rolling_train_years_long': l4.get('rolling_train_years_long', 'N/A')
    }

    for param, value in l4_params.items():
        print(f"  • {param}: {value}")

    print("\n📊 Track B: 백테스트 파라미터 (L7)")
    print("-" * 60)

    # 기본 L7 파라미터
    l7 = config.get('l7', {})
    print("\n🔹 L7 기본 설정:")

    base_l7_params = {
        'holding_days': l7.get('holding_days', 'N/A'),
        'rebalance_interval': l7.get('rebalance_interval', 'N/A'),
        'cost_bps': l7.get('cost_bps', 'N/A'),
        'slippage_bps': l7.get('slippage_bps', 'N/A'),
        'top_k': l7.get('top_k', 'N/A'),
        'target_volatility': l7.get('target_volatility', 'N/A'),
        'volatility_adjustment_enabled': l7.get('volatility_adjustment_enabled', 'N/A'),
        'volatility_adjustment_min': l7.get('volatility_adjustment_min', 'N/A'),
        'volatility_adjustment_max': l7.get('volatility_adjustment_max', 'N/A'),
        'risk_scaling_enabled': l7.get('risk_scaling_enabled', 'N/A'),
        'smart_buffer_enabled': l7.get('smart_buffer_enabled', 'N/A'),
        'regime.enabled': l7.get('regime', {}).get('enabled', 'N/A'),
        'score_col': l7.get('score_col', 'N/A'),
        'signal_source': l7.get('signal_source', 'N/A'),
        'weighting': l7.get('weighting', 'N/A')
    }

    for param, value in base_l7_params.items():
        print(f"  • {param}: {value}")

    # 각 전략별 특별 설정
    strategies = ['bt20_short', 'bt20_ens', 'bt120_long', 'bt120_ens']

    for strategy in strategies:
        section_name = f'l7_{strategy}'
        if section_name in config:
            strategy_config = config[section_name]
            print(f"\n🔹 {strategy.upper()} 특별 설정:")

            strategy_params = {
                'holding_days': strategy_config.get('holding_days', '기본값 사용'),
                'rebalance_interval': strategy_config.get('rebalance_interval', '기본값 사용'),
                'cost_bps': strategy_config.get('cost_bps', '기본값 사용'),
                'slippage_bps': strategy_config.get('slippage_bps', '기본값 사용'),
                'top_k': strategy_config.get('top_k', '기본값 사용'),
                'target_volatility': strategy_config.get('target_volatility', '기본값 사용'),
                'volatility_adjustment_enabled': strategy_config.get('volatility_adjustment_enabled', '기본값 사용'),
                'volatility_adjustment_min': strategy_config.get('volatility_adjustment_min', '기본값 사용'),
                'volatility_adjustment_max': strategy_config.get('volatility_adjustment_max', '기본값 사용'),
                'risk_scaling_enabled': strategy_config.get('risk_scaling_enabled', '기본값 사용'),
                'smart_buffer_enabled': strategy_config.get('smart_buffer_enabled', '기본값 사용'),
                'score_col': strategy_config.get('score_col', '기본값 사용'),
                'buffer_k': strategy_config.get('buffer_k', '기본값 사용'),
                'regime.enabled': strategy_config.get('regime', {}).get('enabled', '기본값 사용'),
                'overlapping_tranches_enabled': strategy_config.get('overlapping_tranches_enabled', '기본값 사용')
            }

            for param, value in strategy_params.items():
                if value != '기본값 사용':  # 특별히 설정된 값만 표시
                    print(f"  • {param}: {value}")

    print("\n🎯 파라미터 적용 우선순위")
    print("-" * 40)
    print("1. 전략별 특별 설정 (l7_bt20_short 등)")
    print("2. 기본 L7 설정")
    print("3. Track A 설정 (L4, L5, L6)")

    print("\n💡 현재 적용 상태 요약")
    print("-" * 40)
    print("• 보수적 설정 적용됨 (cost_bps=20, slippage_bps=10)")
    print("• 변동성 제어 완화 (min=0.3, max=0.8)")
    print("• 리스크 관리 전략 비활성화")
    print("• Top K = 10 (집중 투자)")
    print("• 앙상블 모델 + Ridge 정규화 적용")

if __name__ == "__main__":
    show_applied_parameters()