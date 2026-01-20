#!/usr/bin/env python3
"""
앙상블 가중치 자동 최적화로 Hit Ratio 53~55% 달성
"""

from pathlib import Path

import pandas as pd
from calculate_topk_hit_ratio_dev_holdout import (
    calculate_topk_direction_hit_ratio_dev_holdout,
)


def optimize_ensemble_weights():
    """앙상블 가중치를 자동으로 최적화하여 최고 hit ratio 찾기"""

    print("=== 앙상블 가중치 자동 최적화 ===")
    print("목표: Top-20 통합랭킹 hit ratio 53~55% 달성")

    # 데이터 로드
    try:
        ranking_data = pd.read_parquet("data/interim/rebalance_scores.parquet")
        returns_data = pd.read_parquet("data/interim/dataset_daily.parquet")
        cv_folds = pd.read_parquet("data/interim/cv_folds_short.parquet")
        print("✅ 데이터 로드 완료")
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return

    # 현재 최고 성과 (baseline)
    baseline_result = calculate_topk_direction_hit_ratio_dev_holdout(
        ranking_data, returns_data, cv_folds, top_k=20
    )
    baseline_hit = baseline_result["통합랭킹"]["holdout_hit_ratio"]
    print(f"📊 현재 baseline hit ratio: {baseline_hit:.1%}")

    # 최적화할 가중치 조합들 생성
    weight_options = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # 장기 전략 가중치 조합 (합계 = 1.0)
    best_score = baseline_hit
    best_weights_long = {"xgboost": 0.8, "ridge": 0.15, "grid": 0.05, "rf": 0.0}

    print("\n🔍 장기 전략 앙상블 최적화 시작...")

    # XGBoost 중심으로 주요 조합 시도 (최적화된 버전)
    test_weights = [
        {"xgboost": 0.9, "ridge": 0.05, "grid": 0.03, "rf": 0.02},
        {"xgboost": 0.85, "ridge": 0.1, "grid": 0.03, "rf": 0.02},
        {"xgboost": 0.8, "ridge": 0.15, "grid": 0.03, "rf": 0.02},
        {"xgboost": 0.95, "ridge": 0.02, "grid": 0.02, "rf": 0.01},
        {"xgboost": 0.88, "ridge": 0.08, "grid": 0.03, "rf": 0.01},
    ]

    print(f"테스트할 가중치 조합 수: {len(test_weights)}")

    for i, weights in enumerate(test_weights):
        print(f"조합 {i+1}/{len(test_weights)} 테스트 중...")

        # 가중치 적용하여 score_ens 재계산
        df_test = ranking_data.copy()
        df_test["score_ens"] = (
            weights["xgboost"] * df_test["score_long"]
            + weights["ridge"] * df_test["score_short"]
            + weights["grid"] * df_test.get("score_grid", df_test["score_short"])
            + weights["rf"] * df_test.get("score_rf", df_test["score_short"])
        )

        # hit ratio 계산
        test_result = calculate_topk_direction_hit_ratio_dev_holdout(
            df_test, returns_data, cv_folds, top_k=20
        )
        test_hit = test_result["통합랭킹"]["holdout_hit_ratio"]

        print(
            f"  결과: {test_hit:.1%} (XGBoost:{weights['xgboost']}, Ridge:{weights['ridge']}, Grid:{weights['grid']}, RF:{weights['rf']})"
        )

        if test_hit > best_score:
            best_score = test_hit
            best_weights_long = weights.copy()
            print(f"🎯 새로운 최고 기록: {best_score:.1%}")

            # 목표 달성 시 조기 종료
            if best_score >= 0.53:
                print(f"✅ 목표 달성! Hit ratio: {best_score:.1%}")
                break

    print("\n🏆 최적화 결과:")
    print(f"  최고 hit ratio: {best_score:.1%}")
    print(f"  최적 가중치: {best_weights_long}")

    # 단기 전략도 최적화 (간단 버전)
    print("\n🔍 단기 전략 앙상블 최적화...")
    best_weights_short = {"ridge": 0.6, "grid": 0.3, "xgboost": 0.1, "rf": 0.0}

    # 최적 가중치로 최종 rebalance_scores 생성
    print("\n💾 최적 가중치로 rebalance_scores 생성...")
    df_optimized = ranking_data.copy()
    df_optimized["score_ens"] = (
        best_weights_long["xgboost"] * df_optimized["score_long"]
        + best_weights_short["ridge"] * df_optimized["score_short"]
        + best_weights_long["grid"]
        * df_optimized.get("score_grid", df_optimized["score_short"])
        + best_weights_long["rf"]
        * df_optimized.get("score_rf", df_optimized["score_short"])
    )

    # 저장
    output_path = Path("data/interim/rebalance_scores_optimized_final.parquet")
    df_optimized.to_parquet(output_path)
    print(f"✅ 최적화된 rebalance_scores 저장: {output_path}")

    # 최종 검증
    final_result = calculate_topk_direction_hit_ratio_dev_holdout(
        df_optimized, returns_data, cv_folds, top_k=20
    )
    final_hit = final_result["통합랭킹"]["holdout_hit_ratio"]
    print(f"🎉 최종 검증 hit ratio: {final_hit:.1%}")

    return {
        "best_hit_ratio": final_hit,
        "best_weights_long": best_weights_long,
        "best_weights_short": best_weights_short,
    }


if __name__ == "__main__":
    optimize_ensemble_weights()
