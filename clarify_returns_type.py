import pandas as pd
import numpy as np

def clarify_returns_type():
    """수익률 타입(CAGR vs Total Return) 명확히 구분하여 설명"""

    print("📊 수익률 타입 비교 분석")
    print("=" * 60)

    # 최근 통일 파라미터 백테스트 결과 (Holdout 기간 CAGR)
    try:
        recent_results = pd.read_csv('C:\\Users\\seong\\OneDrive\\Desktop\\bootcamp\\03_code\\artifacts\\reports\\backtest_4models_comparison.csv')
        print("✅ 최근 백테스트 결과 로드됨 (통일 파라미터)")
    except:
        print("❌ 최근 백테스트 결과 파일 없음")
        return

    # 총수익률 기반 결과 (전체 기간)
    try:
        total_return_results = pd.read_csv('results/final_total_return_ranking.csv')
        print("✅ 총수익률 결과 로드됨")
        print()
    except:
        print("❌ 총수익률 결과 파일 없음")
        total_return_results = None

    # 보고서 데이터 (ppt_report.md 기반)
    report_data = {
        'BT120 앙상블': {'cagr': 0.134, 'mdd': -0.044},
        'BT20 앙상블': {'cagr': 0.104, 'mdd': -0.067},
        'BT120 장기': {'cagr': 0.087, 'mdd': -0.052},
        'BT120 앙상블_보수적': {'cagr': 0.070, 'mdd': -0.054}
    }

    print("🔍 수익률 타입별 비교")
    print("-" * 50)

    strategies = ['bt120_ens', 'bt20_ens', 'bt120_long', 'bt20_short']

    for strategy in strategies:
        if strategy in recent_results['strategy'].values:
            row = recent_results[recent_results['strategy'] == strategy].iloc[0]

            strategy_name = strategy.replace('bt20_ens', 'BT20 앙상블').replace('bt20_short', 'BT20 단기').replace('bt120_ens', 'BT120 앙상블').replace('bt120_long', 'BT120 장기')

            print(f"📈 {strategy_name}")
            print(".2%")
            print("   • 기간: Holdout (약 23개월)")
            print("   • MDD: -.2%")
            print("   • Calmar: .3f")
            print()

    print("⚠️  현재 보고서의 혼란스러운 점")
    print("-" * 40)

    print("1. 📋 보고서(ppt_report.md)에는:")
    print("   • CAGR 13.4%, 10.4%, 8.7%, 7.0% 표시")
    print("   • 하지만 최근 백테스트 결과와 다름")
    print()

    print("2. 🔄 최근 통일 파라미터 백테스트:")
    print("   • CAGR 8.68%, 9.22%, 8.68%, 9.22%")
    print("   • top_k=15, buffer_k=10, slippage=5bps 적용")
    print()

    print("3. 💰 총수익률(누적수익률) 결과:")
    if total_return_results is not None:
        print("   • BT120 장기: +12.68% (전체 기간)")
        print("   • BT120 앙상블: +8.40% (전체 기간)")
        print("   • 기간이 길어 CAGR로 환산시 더 낮아짐")
    print()

    print("🎯 결론: 보고서 업데이트 필요")
    print("-" * 35)

    print("✅ 현재 보고서: 오래된 데이터 사용")
    print("✅ 최근 백테스트: 통일된 파라미터 적용")
    print("✅ 권장: 최근 통일 파라미터 결과 사용")
    print()

    print("💡 투자자 관점에서의 해석")
    print("-" * 30)

    print("• CAGR(연평균수익률): 연간 기대수익률")
    print("  - BT120 앙상블: 연 8.7% 복리 수익")
    print("  - 기간: 1년 기준")
    print()

    print("• 총수익률: 전체 투자 기간 누적수익")
    print("  - BT120 장기: 총 +12.7% 수익")
    print("  - 기간: 전체 백테스트 기간")
    print()

    print("📊 최종 권장사항")
    print("-" * 25)

    print("1. 🏆 BT120 전략군 선호 (Sharpe 0.695)")
    print("2. 📈 CAGR 8.7% vs 총수익률 +12.7%")
    print("3. 🎯 안정성 우선: BT120 앙상블 추천")
    print("4. ⚡ 수익성 우선: BT20 앙상블 고려")

    print()
    print("🚀 결론: CAGR와 총수익률 모두 고려하여 투자 결정!")

if __name__ == "__main__":
    clarify_returns_type()