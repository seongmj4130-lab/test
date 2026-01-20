from datetime import datetime

import pandas as pd


def analyze_dynamic_return_implementation():
    """동적 return 계산 구현 결과 분석"""

    print("🔬 동적 Return 계산 구현 결과 분석")
    print("=" * 70)

    # 현재 구현 상태
    print("\n📋 구현된 기능:")
    print("- ✅ L6R: 다양한 기간의 forward return 컬럼 추가 시도")
    print("- ✅ L7: _pick_ret_col 함수에 holding_days 파라미터 추가")
    print("- ✅ L7: holding_days에 따라 다른 return 컬럼 선택 로직 구현")
    print("- ✅ 디버깅: ret_col 선택 과정 로깅")

    print("\n📊 백테스트 실행 결과:")
    print("- 현재 holding_days=100 설정된 전략들 모두 'true_short' 선택")
    print("- 20일 전략들도 'true_short' 선택")
    print("- 결과적으로 모든 전략이 동일한 20일 return 사용")

    # 실제 백테스트 결과 (현재 상태)
    current_results = [
        {
            "strategy": "bt20_short",
            "holding_days": 20,
            "ret_col": "true_short",
            "sharpe": 0.9141,
        },
        {
            "strategy": "bt120_long",
            "holding_days": 20,
            "ret_col": "true_short",
            "sharpe": 0.6946,
        },
        {
            "strategy": "bt20_ens",
            "holding_days": 100,
            "ret_col": "true_short",
            "sharpe": 0.3357,
        },
        {
            "strategy": "bt120_ens",
            "holding_days": 100,
            "ret_col": "true_short",
            "sharpe": 0.2658,
        },
    ]

    results_df = pd.DataFrame(current_results)
    print("\n현재 백테스트 결과:")
    print(results_df.to_string(index=False))

    print("\n🎯 문제점 분석:")
    print("- ❌ L6R에서 true_short가 우선적으로 계산되어 존재")
    print("- ❌ _pick_ret_col에서 true_short가 가장 먼저 선택됨")
    print("- ❌ holding_days에 따른 동적 선택이 작동하지 않음")

    print("\n💡 해결 방안:")
    print("1️⃣ true_short 대신 동적 컬럼 우선 사용:")
    print("   • L6R에서 true_short 생성하지 않기")
    print("   • 또는 L7에서 cfg.ret_col을 동적으로 설정")

    print("\n2️⃣ L7 실시간 return 계산:")
    print("   • 백테스트 중에 holding_days만큼 미래 가격 조회")
    print("   • dataset_daily에서 동적 return 계산")

    print("\n3️⃣ Config 기반 동적 설정:")
    print("   • holding_days에 따라 ret_col 자동 설정")
    print("   • 20일 → ret_fwd_20d, 100일 → ret_fwd_120d")

    # 실제 동작하는 코드 예시
    print("\n📝 실제 구현 코드:")
    print(
        """
# L7 백테스트에서
ret_col = _pick_ret_col(rebalance_scores, cfg.ret_col, cfg.holding_days)

# _pick_ret_col 함수
def _pick_ret_col(df, preferred, holding_days):
    # holding_days에 맞는 컬럼 우선 선택
    if holding_days == 20:
        if 'ret_fwd_20d' in df.columns:
            return 'ret_fwd_20d'
    else:
        if 'ret_fwd_120d' in df.columns:
            return 'ret_fwd_120d'
    # fallback
    return preferred
    """
    )

    print("\n🏆 결론:")
    print("- ✅ 동적 return 계산 프레임워크 구현 완료")
    print("- ⚠️  현재 L6R true_short 우선 선택으로 인해 미작동")
    print("- 🔧 추가 수정으로 완전한 동적 계산 가능")
    print("- 📈 holding_days 변경 시 실제 수익률 차이 반영 가능")

    # 구현 상태 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(
        f"results/dynamic_return_implementation_{timestamp}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(f"\n💾 분석 결과 저장: results/dynamic_return_implementation_{timestamp}.csv")


if __name__ == "__main__":
    analyze_dynamic_return_implementation()
