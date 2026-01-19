import yaml
import os
from pathlib import Path

def show_strategy_parameters():
    """4개 전략의 백테스트 파라미터값 표시"""

    print("🔧 4개 전략 백테스트 파라미터값 상세 비교")
    print("=" * 70)

    # config 파일 경로
    config_path = Path("configs/config.yaml")

    if not config_path.exists():
        print("❌ config.yaml 파일을 찾을 수 없습니다.")
        return

    # YAML 파일 로드
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 전략 이름 매핑
    strategy_mapping = {
        'l7_bt20_short': 'BT20 단기',
        'l7_bt20_ens': 'BT20 앙상블',
        'l7_bt120_long': 'BT120 장기',
        'l7_bt120_ens': 'BT120 앙상블'
    }

    # 각 전략별 파라미터 추출 및 표시
    for config_key, display_name in strategy_mapping.items():
        if config_key in config:
            params = config[config_key]
            print(f"\n🏆 {display_name} ({config_key})")
            print("-" * 50)

            # 주요 파라미터 그룹화
            core_params = {}
            risk_params = {}
            buffer_params = {}
            tranche_params = {}
            regime_params = {}

            for key, value in params.items():
                if key in ['top_k', 'holding_days', 'rebalance_interval', 'cost_bps', 'slippage_bps', 'score_col', 'return_col', 'weighting']:
                    core_params[key] = value
                elif key in ['volatility_adjustment_enabled', 'target_volatility', 'volatility_adjustment_max', 'volatility_adjustment_min', 'volatility_lookback_days']:
                    risk_params[key] = value
                elif key in ['smart_buffer_enabled', 'smart_buffer_stability_threshold', 'buffer_k']:
                    buffer_params[key] = value
                elif key in ['overlapping_tranches_enabled', 'tranche_holding_days', 'tranche_max_active', 'tranche_allocation_mode']:
                    tranche_params[key] = value
                elif 'regime' in key or 'risk_scaling' in key or 'exposure' in key:
                    if isinstance(value, dict):
                        regime_params[key] = str(value)
                    else:
                        regime_params[key] = value

            # 코어 파라미터
            print("📊 코어 파라미터:")
            for key, value in core_params.items():
                print(f"   • {key}: {value}")

            # 리스크 관리 파라미터
            print("\n🛡️ 리스크 관리:")
            for key, value in risk_params.items():
                print(f"   • {key}: {value}")

            # 버퍼 파라미터
            print("\n🔄 스마트 버퍼:")
            for key, value in buffer_params.items():
                print(f"   • {key}: {value}")

            # 트랜치 파라미터 (BT120만)
            if tranche_params:
                print("\n📈 오버래핑 트랜치:")
                for key, value in tranche_params.items():
                    print(f"   • {key}: {value}")

            # 시장 국면 파라미터
            if regime_params:
                print("\n🌊 시장 국면 조정:")
                for key, value in regime_params.items():
                    print(f"   • {key}: {value}")

        else:
            print(f"\n❌ {display_name} ({config_key}): 설정을 찾을 수 없습니다.")

    print("\n" + "=" * 70)
    print("📋 파라미터 설명:")
    print("- top_k: 선택할 종목 수")
    print("- holding_days: 포지션 유지 기간")
    print("- rebalance_interval: 리밸런싱 주기")
    print("- buffer_k: 버퍼 종목 수 (안정성)")
    print("- overlapping_tranches: 다중 트랜치 모드")
    print("- volatility_adjustment: 변동성 기반 스케일링")
    print("- risk_scaling: 시장 국면별 리스크 조정")
    print("- smart_buffer: 종목 유지율 기반 조정")

if __name__ == "__main__":
    show_strategy_parameters()