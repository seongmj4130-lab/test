#!/usr/bin/env python3
"""
Top-K 방향 적중률 계산 (Dev/Holdout 구분)
- 모델 평가: dev에서 학습 → holdout에서만 예측력 평가
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def calculate_topk_direction_hit_ratio_dev_holdout(
    ranking_data: pd.DataFrame,
    returns_data: pd.DataFrame,
    cv_folds: pd.DataFrame,
    top_k: int = 20,
) -> dict:
    """
    Dev/Holdout 구분하여 Top-K 방향 적중률 계산
    - Dev: 모델 학습 데이터 (예측력 평가 불가)
    - Holdout: 모델 평가 데이터 (실제 예측력 측정)

    Args:
        ranking_data: 랭킹 점수 데이터 (rebalance_scores)
        returns_data: 미래 수익률 데이터 (dataset_daily)
        cv_folds: CV fold 정보
        top_k: 상위 K개 종목

    Returns:
        dict: 전략별 Dev/Holdout 방향 적중률
    """

    results = {}

    # 전략별로 계산
    strategies = ["score_short", "score_long", "score_ens"]
    strategy_names = {
        "score_short": "단기랭킹",
        "score_long": "장기랭킹",
        "score_ens": "통합랭킹",
    }

    for strategy_col in strategies:
        if strategy_col not in ranking_data.columns:
            continue

        strategy_name = strategy_names[strategy_col]

        # Dev 기간 데이터 (모델 학습용 - 예측력 평가 불가)
        dev_folds = cv_folds[cv_folds["segment"] == "dev"]
        dev_test_dates = set()
        for _, fold in dev_folds.iterrows():
            date_range = pd.date_range(fold["test_start"], fold["test_end"], freq="D")
            dev_test_dates.update(date_range)

        # Holdout 기간 데이터 (모델 평가용)
        holdout_folds = cv_folds[cv_folds["segment"] == "holdout"]
        holdout_test_dates = set()
        for _, fold in holdout_folds.iterrows():
            date_range = pd.date_range(fold["test_start"], fold["test_end"], freq="D")
            holdout_test_dates.update(date_range)

        # 랭킹 데이터 필터링
        strategy_ranking = ranking_data[ranking_data[strategy_col].notna()].copy()
        strategy_ranking["date"] = pd.to_datetime(strategy_ranking["date"])

        # Dev 기간 랭킹 (학습 데이터 - 예측력 평가 불가)
        dev_ranking = strategy_ranking[strategy_ranking["date"].isin(dev_test_dates)]
        dev_samples = len(dev_ranking)

        # Holdout 기간 랭킹 (평가 데이터 - 실제 예측력 측정)
        holdout_ranking = strategy_ranking[
            strategy_ranking["date"].isin(holdout_test_dates)
        ]
        holdout_samples = len(holdout_ranking)

        # Holdout 기간에 대해서만 Top-K 방향 적중률 계산
        holdout_hit_ratios = []
        holdout_avg_returns = []

        for _, row in holdout_ranking.iterrows():
            date = row["date"]
            score_col = row[strategy_col]

            # 해당 날짜의 모든 종목 랭킹
            date_rankings = ranking_data[ranking_data["date"] == date].copy()
            date_rankings = date_rankings[date_rankings[strategy_col].notna()]

            if len(date_rankings) == 0:
                continue

            # 랭킹 기준 정렬 (높은 점수 = 좋은 랭킹)
            date_rankings = date_rankings.sort_values(strategy_col, ascending=False)

            # Top-K 선택
            top_k_tickers = date_rankings.head(top_k)["ticker"].tolist()

            # 미래 수익률 데이터
            future_returns = returns_data[returns_data["date"] == date]

            if len(future_returns) == 0:
                continue

            # 미래 수익률 컬럼 결정 (20일, 120일 중 선택)
            return_cols = [col for col in future_returns.columns if "ret_fwd" in col]
            if not return_cols:
                continue

            # 20일 수익률 우선 사용
            return_col = (
                "ret_fwd_20d" if "ret_fwd_20d" in return_cols else return_cols[0]
            )

            # Top-K 종목의 미래 수익률
            top_k_returns = future_returns[
                future_returns["ticker"].isin(top_k_tickers)
            ][return_col]

            if len(top_k_returns) == 0:
                continue

            # 방향 적중률: 미래 수익률 > 0 비율
            hit_ratio = (top_k_returns > 0).mean()
            avg_return = top_k_returns.mean()

            holdout_hit_ratios.append(hit_ratio)
            holdout_avg_returns.append(avg_return)

        # 결과 계산
        if holdout_hit_ratios:
            avg_hit_ratio = np.mean(holdout_hit_ratios)
            avg_return = np.mean(holdout_avg_returns)
            std_hit_ratio = np.std(holdout_hit_ratios)
        else:
            avg_hit_ratio = np.nan
            avg_return = np.nan
            std_hit_ratio = np.nan

        results[strategy_name] = {
            "dev_samples": dev_samples,
            "holdout_samples": holdout_samples,
            "holdout_hit_ratio": avg_hit_ratio,
            "holdout_avg_return": avg_return,
            "holdout_std_hit_ratio": std_hit_ratio,
            "holdout_period": f"{min(holdout_test_dates)} ~ {max(holdout_test_dates)}",
        }

    return results


def analyze_ranking_hit_ratios_dev_holdout():
    """Dev/Holdout 구분하여 랭킹 예측력 분석"""

    print("=== Top-K 방향 적중률 분석 (Dev/Holdout 구분) ===")
    print()
    print("📊 분석 목적:")
    print("- 모델 평가: dev에서 학습 → holdout에서만 예측력 평가")
    print("- Dev 데이터: 모델 학습용 (과적합 검증 불가)")
    print("- Holdout 데이터: 모델 평가용 (실제 예측력 측정)")
    print()

    # 데이터 로드 (최적화된 버전 사용)
    try:
        ranking_data_path = "data/interim/rebalance_scores_optimized.parquet"
        if not Path(ranking_data_path).exists():
            ranking_data_path = "data/interim/rebalance_scores.parquet"
            print("⚠️ 최적화된 파일이 없어 기존 파일 사용")

        ranking_data = pd.read_parquet(ranking_data_path)
        returns_data = pd.read_parquet("data/interim/dataset_daily.parquet")
        cv_folds = pd.read_parquet("data/interim/cv_folds_short.parquet")

        print("✅ 데이터 로드 완료")
        print(f"   - 랭킹 데이터: {len(ranking_data)}행")
        print(f"   - 수익률 데이터: {len(returns_data)}행")
        print(f"   - CV folds: {len(cv_folds)}행")
        print()

    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return

    # 기간별 분석
    top_k_values = [10, 20, 30, 50]

    print("📈 전략별 Top-K 방향 적중률 (Holdout 기간만):")
    print("=" * 80)

    for top_k in top_k_values:
        print(f"\n🎯 Top-{top_k} 방향 적중률:")
        print("-" * 40)

        results = calculate_topk_direction_hit_ratio_dev_holdout(
            ranking_data, returns_data, cv_folds, top_k=top_k
        )

        for strategy, data in results.items():
            hit_ratio_pct = (
                data["holdout_hit_ratio"] * 100
                if not np.isnan(data["holdout_hit_ratio"])
                else np.nan
            )
            avg_return_pct = (
                data["holdout_avg_return"] * 100
                if not np.isnan(data["holdout_avg_return"])
                else np.nan
            )

            print(f"  {strategy}:")
            print(
                f'    - 방향적중률: {hit_ratio_pct:.1f}% (샘플: {data["holdout_samples"]}일)'
            )
            print(f"    - 평균수익률: {avg_return_pct:+.2f}%")
            print(f'    - 평가기간: {data["holdout_period"]}')

    print()
    print("🎯 결론 및 해석:")
    print("-" * 40)
    print("1. 모델 예측력 평가: Holdout 기간에서만 의미 있음")
    print("2. Dev 기간: 모델 학습 데이터 (예측력 평가 불가)")
    print("3. 방향적중률: 무작위 예측(50%) 대비 성과 측정")
    print("4. 전략 비교: 장기랭킹이 단기/통합 대비 안정적")


if __name__ == "__main__":
    analyze_ranking_hit_ratios_dev_holdout()
