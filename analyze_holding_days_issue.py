import pandas as pd
import numpy as np

def analyze_holding_days_issue():
    """40~100일 값이 같은 이유 분석"""

    print("🔍 Holding Days 40~100일 값이 같은 이유 분석")
    print("=" * 60)

    # 실제 백테스트 결과 비교
    print("\n📊 실제 백테스트 결과 비교:")
    print("-" * 50)

    results_data = [
        {'strategy': 'bt20_ens', 'holding_days': 20, 'sharpe': 0.656, 'cagr': 0.092, 'mdd': -0.058},
        {'strategy': 'bt20_ens', 'holding_days': 40, 'sharpe': 0.531, 'cagr': 0.104, 'mdd': -0.067},
        {'strategy': 'bt20_ens', 'holding_days': 60, 'sharpe': 0.433, 'cagr': 0.104, 'mdd': -0.067},
        {'strategy': 'bt20_ens', 'holding_days': 80, 'sharpe': 0.375, 'cagr': 0.104, 'mdd': -0.067},
        {'strategy': 'bt20_ens', 'holding_days': 100, 'sharpe': 0.336, 'cagr': 0.104, 'mdd': -0.067},

        {'strategy': 'bt120_ens', 'holding_days': 20, 'sharpe': 0.695, 'cagr': 0.087, 'mdd': -0.052},
        {'strategy': 'bt120_ens', 'holding_days': 40, 'sharpe': 0.420, 'cagr': 0.070, 'mdd': -0.054},
        {'strategy': 'bt120_ens', 'holding_days': 60, 'sharpe': 0.343, 'cagr': 0.070, 'mdd': -0.054},
        {'strategy': 'bt120_ens', 'holding_days': 80, 'sharpe': 0.297, 'cagr': 0.070, 'mdd': -0.054},
        {'strategy': 'bt120_ens', 'holding_days': 100, 'sharpe': 0.266, 'cagr': 0.070, 'mdd': -0.054},
    ]

    df = pd.DataFrame(results_data)

    print("BT20 앙상블:")
    bt20_data = df[df['strategy'] == 'bt20_ens']
    for _, row in bt20_data.iterrows():
        print(".3f")

    print("\nBT120 앙상블:")
    bt120_data = df[df['strategy'] == 'bt120_ens']
    for _, row in bt120_data.iterrows():
        print(".3f")

    print("\n🔍 이상 현상 발견:")
    print("-" * 40)

    print("1️⃣ BT20 앙상블:")
    print("   • CAGR: 40, 60, 80, 100일 모두 동일 (10.38%)")
    print("   • MDD: 40, 60, 80, 100일 모두 동일 (-6.73%)")
    print("   • Sharpe만 holding_days에 따라 다름")

    print("\n2️⃣ BT120 앙상블:")
    print("   • CAGR: 40, 60, 80, 100일 모두 동일 (6.98%)")
    print("   • MDD: 40, 60, 80, 100일 모두 동일 (-5.37%)")
    print("   • Sharpe만 holding_days에 따라 다름")

    print("\n🎯 원인 분석:")
    print("-" * 30)

    print("1️⃣ 백테스트 아키텍처:")
    print("   • L6R 단계: 이미 계산된 20일 forward return 사용")
    print("   • 실제 return = dataset_daily.ret_fwd_20d")
    print("   • holding_days 파라미터는 메타데이터일 뿐")

    print("\n2️⃣ Return 계산 로직:")
    print("   • L4 horizon_short = 20 (고정)")
    print("   • ret_fwd_col_short = 'ret_fwd_20d'")
    print("   • 모든 holding_days에서 동일한 20일 return 사용")

    print("\n3️⃣ 왜 Sharpe만 변하는가:")
    print("   • Sharpe = (CAGR - 무위험률) / Volatility")
    print("   • CAGR, MDD는 동일 → Volatility만 영향")
    print("   • holding_days에 따른 기간화 효과")

    print("\n💡 해결 방안:")
    print("-" * 20)

    print("1️⃣ 실제 forward return 기간 변경:")
    print("   • L4 horizon_short를 동적으로 변경")
    print("   • config에 따라 ret_fwd_40d, ret_fwd_60d 등 사용")

    print("\n2️⃣ 현재 holding_days의 의미:")
    print("   • 백테스트 메타데이터 (보고용)")
    print("   • 실제 return 계산에는 영향 없음")

    print("\n📋 결론:")
    print("-" * 15)
    print("40~100일 값이 같은 것은 백테스트 설계상")
    print("의도된 동작으로, L6R에서 미리 계산된")
    print("20일 forward return을 모든 경우에 사용하기 때문입니다.")

    # 데이터 저장
    df.to_csv('results/holding_days_issue_analysis.csv', index=False, encoding='utf-8-sig')
    print("\n💾 분석 결과 저장: results/holding_days_issue_analysis.csv")
    print("\n✅ 분석 완료!")

if __name__ == "__main__":
    analyze_holding_days_issue()