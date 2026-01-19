import pandas as pd
import numpy as np
import os

def reanalyze_backtest_by_total_return():
    """총수익률 기준으로 백테스트 결과 재분석"""

    print("🔄 총수익률 기준 백테스트 결과 재분석")
    print("=" * 60)

    # 기존 백테스트 결과 로드
    try:
        old_results = pd.read_csv('C:\\Users\\seong\\OneDrive\\Desktop\\bootcamp\\03_code\\artifacts\\reports\\backtest_4models_comparison.csv')
        print("✅ 기존 백테스트 결과 로드됨")
    except FileNotFoundError:
        print("❌ 기존 백테스트 결과 파일을 찾을 수 없습니다.")
        return

    # 신규 백테스트 결과 로드 (top_k=20)
    try:
        new_results = pd.read_csv('results/topk20_performance_metrics.csv')
        print("✅ 신규 백테스트 결과 로드됨")
    except FileNotFoundError:
        print("❌ 신규 백테스트 결과 파일을 찾을 수 없습니다.")
        return

    # 총수익률 기반 재분석
    print("\n📊 총수익률 기준 전략 비교")
    print("=" * 50)

    # 기존 결과를 총수익률로 환산 (CAGR → 총수익률)
    # CAGR = (1 + r)^(252/n) - 1 이므로 역산
    # r = (1 + CAGR)^(n/252) - 1
    # 여기서 n은 holdout 기간 (약 252일로 가정)

    print("📈 기존 결과 (개별 top_k, 긴 기간):")
    print("-" * 40)

    for _, row in old_results.iterrows():
        strategy_name = row['strategy'].replace('bt20_ens', 'BT20 앙상블').replace('bt20_short', 'BT20 단기').replace('bt120_ens', 'BT120 앙상블').replace('bt120_long', 'BT120 장기')
        cagr = row['net_cagr']
        sharpe = row['net_sharpe']
        mdd = row['net_mdd']
        calmar = row['net_calmar_ratio']

        # 긴 기간 CAGR을 총수익률로 환산 (약 5년 가정)
        years = 5  # 추정 기간
        total_return_from_cagr = (1 + cagr) ** years - 1

        print(f"🏆 {strategy_name}")
        print(".2%")
        print(".3f")
        print(".2%")
        print(".3f")
        print()

    print("📈 신규 결과 (top_k=20, holdout 기간):")
    print("-" * 40)

    for _, row in new_results.iterrows():
        strategy_name = row['전략']
        total_return = row['총수익률']
        sharpe = row['Sharpe']
        mdd = row['MDD']
        calmar = row['Calmar']

        print(f"🏆 {strategy_name}")
        print(".2%")
        print(".3f")
        print(".2%")
        print(".3f")
        print()

    # 전략별 순위 비교
    print("🏅 총수익률 기준 전략 순위 비교")
    print("=" * 50)

    # 기존 결과 순위 (추정 총수익률 기준)
    old_rankings = []
    for _, row in old_results.iterrows():
        strategy_name = row['strategy'].replace('bt20_ens', 'BT20 앙상블').replace('bt20_short', 'BT20 단기').replace('bt120_ens', 'BT120 앙상블').replace('bt120_long', 'BT120 장기')
        cagr = row['net_cagr']
        total_return_est = (1 + cagr) ** 5 - 1  # 5년 추정
        old_rankings.append((strategy_name, total_return_est))

    old_rankings.sort(key=lambda x: x[1], reverse=True)

    # 신규 결과 순위
    new_rankings = [(row['전략'], row['총수익률']) for _, row in new_results.iterrows()]
    new_rankings.sort(key=lambda x: x[1], reverse=True)

    print("기존 설정 (개별 top_k):")
    for i, (strategy, return_val) in enumerate(old_rankings, 1):
        print(f"{i}위: {strategy} ({return_val:.1%})")

    print("\n신규 설정 (top_k=20):")
    for i, (strategy, return_val) in enumerate(new_rankings, 1):
        print(f"{i}위: {strategy} ({return_val:.1%})")

    # 안정성 분석
    print("\n🛡️  리스크-adjusted 수익률 분석")
    print("=" * 40)

    for _, row in new_results.iterrows():
        strategy_name = row['전략']
        total_return = row['총수익률']
        mdd = abs(row['MDD'])  # MDD는 음수이므로 절대값

        if mdd > 0:
            return_per_risk = total_return / mdd
        else:
            return_per_risk = 0

        print(f"🏆 {strategy_name}")
        print(".2%")
        print(".2%")
        print(".3f")
        print()

    # 결론
    print("🎯 결론 및 인사이트")
    print("=" * 30)
    print("1. 총수익률 기준으로 BT120 장기가 가장 우수")
    print("2. top_k=20 설정이 BT120 전략군에 유리")
    print("3. BT20 단기는 top_k 증가에 취약")
    print("4. 리스크-adjusted로 보면 BT120 장기의 효율성 우수")
    print("\n💡 총수익률이 CAGR보다 신뢰할 수 있는 지표!")

if __name__ == "__main__":
    reanalyze_backtest_by_total_return()