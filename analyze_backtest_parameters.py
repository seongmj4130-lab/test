#!/usr/bin/env python3
"""
백테스트 성과지표 산출에 영향을 주는 모든 파라미터 분석
"""

from pathlib import Path

import yaml


def analyze_backtest_parameters():
    """백테스트 성과에 영향을 주는 모든 파라미터 분석"""

    print("🔧 백테스트 성과지표 산출에 영향을 주는 모든 파라미터 분석")
    print("=" * 80)

    # 1. 기본 config.yaml 로드
    config_path = Path('configs/config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 2. 재설계된 파라미터 로드
    redesigned_path = Path('configs/redesigned_backtest_params.yaml')
    if redesigned_path.exists():
        with open(redesigned_path, 'r', encoding='utf-8') as f:
            redesigned = yaml.safe_load(f)
    else:
        redesigned = {}

    print("📊 1. 기본 설정 파라미터 (params 섹션)")
    print("-" * 50)
    params = config.get('params', {})
    for key, value in params.items():
        print(f"   {key}: {value}")

    print("\n📊 2. L4 CV 파라미터")
    print("-" * 50)
    l4 = config.get('l4', {})
    for key, value in l4.items():
        print(f"   {key}: {value}")

    print("\n📊 3. L5 모델 파라미터")
    print("-" * 50)
    l5 = config.get('l5', {})
    for key, value in l5.items():
        print(f"   {key}: {value}")

    print("\n📊 4. L6 스코어링 파라미터")
    print("-" * 50)
    l6 = config.get('l6', {})
    for key, value in l6.items():
        print(f"   {key}: {value}")

    print("\n📊 5. L7 기본 백테스트 파라미터")
    print("-" * 50)
    l7 = config.get('l7', {})
    for key, value in l7.items():
        print(f"   {key}: {value}")

    print("\n📊 6. 전략별 L7 파라미터")
    print("-" * 50)
    strategies = ['l7_bt20_short', 'l7_bt120_long', 'l7_bt20_ens', 'l7_bt120_ens']
    for strategy in strategies:
        if strategy in config:
            print(f"\n   🔹 {strategy}:")
            strat_config = config[strategy]
            for key, value in strat_config.items():
                print(f"      {key}: {value}")

    print("\n📊 7. 동적 기간 파라미터 (holding_days별)")
    print("-" * 50)
    dynamic_params = config.get('holding_days_dynamic_params', {})
    for holding_days, params in dynamic_params.items():
        print(f"\n   🔹 {holding_days}일:")
        for key, value in params.items():
            print(f"      {key}: {value}")

    print("\n📊 8. 재설계된 파라미터 (업계표준 적용)")
    print("-" * 50)
    redesigned_params = redesigned.get('params', {})
    for key, value in redesigned_params.items():
        print(f"   {key}: {value}")

    strategies_redesigned = ['bt20_short', 'bt120_long', 'bt20_ens']
    for strategy in strategies_redesigned:
        if strategy in redesigned:
            print(f"\n   🔹 {strategy}:")
            strat_config = redesigned[strategy]
            for key, value in strat_config.items():
                print(f"      {key}: {value}")

    print("\n📊 9. 현재 백테스트 적용 파라미터 (run_dynamic_period_backtest.py)")
    print("-" * 50)
    print("   🔹 전략별 cost_bps (업계표준 적용):")
    print("      bt20_short: 15 (0.15%)")
    print("      bt120_long: 10 (0.10%)")
    print("      bt20_ens: 12 (0.12%)")
    print("   🔹 slippage_bps: 0 (현재 비활성화)")
    print("   🔹 holding_days: [20, 40, 60, 80, 100, 120]")
    print("   🔹 phase: HOLDOUT (2023-01-31 ~ 2024-11-18)")
    print("   🔹 데이터: rebalance_scores_corrected.parquet")

    print("\n📊 10. 파라미터 영향도 분석")
    print("-" * 50)
    print("   🔴 핵심 영향 파라미터:")
    print("      • cost_bps: 거래비용 (턴오버 기반 적용)")
    print("      • top_k: 선택 종목 수")
    print("      • buffer_k: 안정성 버퍼")
    print("      • rebalance_interval: 리밸런싱 주기")
    print("      • holding_days: 수익률 계산 기간")
    print("      • target_vol: 변동성 목표치")
    print("      • regime_enabled: 국면 기반 조정")
    print("   🟡 중간 영향 파라미터:")
    print("      • ridge_alpha: 모델 정규화 강도")
    print("      • weight_short/weight_long: 스코어 가중치")
    print("      • step_days/embargo_days: CV 파라미터")
    print("      • volatility_adjustment_enabled: 변동성 조정")
    print("   🟢 낮은 영향 파라미터:")
    print("      • slippage_bps: 현재 0")
    print("      • softmax_temperature: weighting='equal' 사용")
    print("      • risk_scaling_multiplier: 보조적 적용")

    print("\n🎯 파라미터 최적화 상태")
    print("-" * 50)
    print("   ✅ 적용 완료:")
    print("      • 업계표준 거래비용 (cost_bps)")
    print("      • 동적 기간 파라미터 (holding_days별)")
    print("      • HOLDOUT 구간 테스트")
    print("      • 전략별 특화 파라미터")
    print("   🔄 조정 가능:")
    print("      • top_k: 성과 vs 리스크 트레이드오프")
    print("      • buffer_k: 안정성 vs 수익성")
    print("      • regime 파라미터: 시장 국면 활용")
    print("   ⚠️ 검토 필요:")
    print("      • slippage_bps: 현실성 향상")
    print("      • market_regime 데이터: regime 기능 활성화")

    print("\n📈 현재 파라미터 조합 결과 요약")
    print("-" * 50)
    print("   • 단기 전략 (bt20_short): 20일 초점, top_k=5, 비용=0.15%")
    print("   • 장기 전략 (bt120_long): 120일 초점, top_k=8, 비용=0.10%")
    print("   • 통합 전략 (bt20_ens): 업계평균 목표, top_k=10, 비용=0.12%")
    print("   • 동적 적용: holding_days별 파라미터 자동 조정")
    print("   • 평가 구간: HOLDOUT (시장 현실성 확보)")

if __name__ == "__main__":
    analyze_backtest_parameters()
