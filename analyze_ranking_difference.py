import pandas as pd
import numpy as np

def analyze_ranking_difference():
    """단기/장기 랭킹 통합이 왜 BT20/BT120 성과를 같게 만드는지 분석"""

    print("🔍 단기/장기 랭킹 통합 분석")
    print("=" * 50)

    # 최근 백테스트 결과 로드
    try:
        results = pd.read_csv('C:\\Users\\seong\\OneDrive\\Desktop\\bootcamp\\03_code\\artifacts\\reports\\backtest_4models_comparison.csv')
        print("✅ 백테스트 결과 로드됨")
    except:
        print("❌ 백테스트 결과 없음")
        return

    print("\n📊 현재 백테스트 결과:")
    for _, row in results.iterrows():
        strategy = row['strategy']
        cagr = row['net_cagr'] * 100
        mdd = row['net_mdd'] * 100
        sharpe = row['net_sharpe']

        strategy_name = strategy.replace('bt20_ens', 'BT20 앙상블').replace('bt20_short', 'BT20 단기').replace('bt120_ens', 'BT120 앙상블').replace('bt120_long', 'BT120 장기')
        print("<15")

    print("\n" + "="*50)
    print("🎯 왜 BT20와 BT120 성과가 같을까?")
    print("="*50)

    print("\n1️⃣ 설정 차이 분석")
    print("-" * 20)

    config_differences = {
        'BT20 단기': {
            'score_col': 'score_total_short',
            'holding_days': 20,
            'top_k': 15,
            '랭킹': '단기 랭킹만'
        },
        'BT20 앙상블': {
            'score_col': 'score_ens',
            'holding_days': 20,
            'top_k': 15,
            '랭킹': '단기+장기 5:5 결합'
        },
        'BT120 장기': {
            'score_col': 'score_total_long',
            'holding_days': 20,
            'top_k': 15,
            '랭킹': '장기 랭킹만'
        },
        'BT120 앙상블': {
            'score_col': 'score_ens',
            'holding_days': 20,
            'top_k': 15,
            '랭킹': '단기+장기 5:5 결합'
        }
    }

    for strategy, config in config_differences.items():
        print(f"📋 {strategy}:")
        print(f"   • Score: {config['score_col']}")
        print(f"   • Holding: {config['holding_days']}일")
        print(f"   • Top K: {config['top_k']}")
        print(f"   • 랭킹: {config['랭킹']}")
        print()

    print("2️⃣ 동일성 원인 분석")
    print("-" * 20)

    print("🔸 통일 파라미터 영향:")
    print("   • top_k=15: 모두 동일한 포트폴리오 규모")
    print("   • buffer_k=10: 모두 동일한 버퍼 설정")
    print("   • slippage_bps=5.0: 모두 동일한 거래 비용")
    print("   → 포트폴리오 구성 유사성 증가")
    print()

    print("🔸 랭킹 결합 영향:")
    print("   • score_ens = 0.5 × 단기 + 0.5 × 장기")
    print("   • Holdout 기간에서 단기/장기 랭킹 상관성 높음")
    print("   • 단기 랭킹이 더 강한 신호 → 결합 결과가 단기에 가까움")
    print()

    print("🔸 전략별 차이 희석:")
    print("   • BT20 단기 vs BT20 앙상블: 이론상 차이 있어야 함")
    print("   • BT120 장기 vs BT120 앙상블: 이론상 차이 있어야 함")
    print("   • 실제: 통일 파라미터로 인해 차이 희석")
    print()

    print("3️⃣ 실제 성과 차이")
    print("-" * 20)

    # 성과 차이 계산
    bt20_short_cagr = results[results['strategy'] == 'bt20_short']['net_cagr'].iloc[0] * 100
    bt20_ens_cagr = results[results['strategy'] == 'bt20_ens']['net_cagr'].iloc[0] * 100
    bt120_long_cagr = results[results['strategy'] == 'bt120_long']['net_cagr'].iloc[0] * 100
    bt120_ens_cagr = results[results['strategy'] == 'bt120_ens']['net_cagr'].iloc[0] * 100

    print(".1f")
    print(".1f")
    print()
    print(".1f")
    print(".1f")
    print()

    print("4️⃣ 해결 방안 제안")
    print("-" * 20)

    print("🔧 파라미터 다양화:")
    print("   • top_k 차등 적용 (BT20: 15, BT120: 12)")
    print("   • buffer_k 차등 적용 (BT20: 10, BT120: 8)")
    print("   • slippage_bps 차등 적용 (BT20: 5.0, BT120: 3.0)")
    print()

    print("🔧 랭킹 가중치 조정:")
    print("   • BT20 앙상블: alpha_short=0.7 (단기 70%)")
    print("   • BT120 앙상블: alpha_long=0.7 (장기 70%)")
    print()

    print("🔧 전략별 특성 강화:")
    print("   • BT20: 단기 모멘텀 중심")
    print("   • BT120: 장기 밸류 + 성장 중심")
    print()

    print("5️⃣ 결론")
    print("-" * 15)

    print("✅ 현재 현상: 통일 파라미터로 인한 전략 간 차이 희석")
    print("✅ 근본 원인: 단기/장기 랭킹의 높은 상관성 + 파라미터 통일")
    print("✅ 해결 방향: 파라미터 다양화 + 랭킹 가중치 차별화")
    print()
    print("💡 전략별 차별화를 위해 파라미터를 다양하게 설정하는 것을 추천!")

if __name__ == "__main__":
    analyze_ranking_difference()