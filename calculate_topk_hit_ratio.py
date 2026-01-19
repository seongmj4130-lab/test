#!/usr/bin/env python3
"""
Top-K 방향 적중률 재계산
- Top-K 종목들의 forward return이 0보다 큰 비율
"""

from pathlib import Path

import numpy as np
import pandas as pd


def calculate_topk_direction_hit_ratio(
    ranking_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    return_col: str,
    top_k: int = 20,
    min_samples: int = 10,
) -> dict:
    """
    Top-K 방향 적중률 계산
    - Top-K 종목들의 forward return이 양수(+)인 비율

    Args:
        ranking_df: 랭킹 데이터 (date, ticker, score, rank 등)
        returns_df: 수익률 데이터 (date, ticker, return_col)
        return_col: 수익률 컬럼명 ('true_short', 'true_long' 등)
        top_k: 상위 K개 종목
        min_samples: 최소 샘플 수

    Returns:
        dict: 계산 결과
    """

    results = []

    # 공통 날짜 필터링
    common_dates = set(ranking_df["date"].unique()) & set(returns_df["date"].unique())
    valid_dates = sorted(list(common_dates))

    if len(valid_dates) == 0:
        return {"error": "공통 날짜가 없습니다"}

    print(f"분석 기간: {valid_dates[0]} ~ {valid_dates[-1]} ({len(valid_dates)}일)")

    for date in valid_dates:
        # 해당 날짜 데이터 필터링
        rank_data = ranking_df[ranking_df["date"] == date].copy()
        return_data = returns_df[returns_df["date"] == date].copy()

        if len(rank_data) < top_k or len(return_data) == 0:
            continue

        # 상위 top_k 종목 선택 (랭킹이 낮을수록 좋음 가정)
        top_k_tickers = rank_data.nsmallest(
            top_k, "rank" if "rank" in rank_data.columns else "ranking"
        )["ticker"].tolist()

        # 해당 종목들의 수익률 가져오기
        top_k_returns = return_data[return_data["ticker"].isin(top_k_tickers)][
            return_col
        ].values

        if len(top_k_returns) < min_samples:
            continue

        # 방향 적중률 계산 (양수 수익률 비율)
        positive_returns = sum(1 for ret in top_k_returns if ret > 0)
        hit_ratio = positive_returns / len(top_k_returns)

        results.append(
            {
                "date": date,
                "top_k": top_k,
                "n_tickers": len(top_k_returns),
                "positive_returns": positive_returns,
                "hit_ratio": hit_ratio,
                "avg_return": np.mean(top_k_returns),
                "median_return": np.median(top_k_returns),
            }
        )

    if not results:
        return {"error": "계산할 수 있는 데이터가 없습니다"}

    # 결과 요약
    df_results = pd.DataFrame(results)

    summary = {
        "total_samples": len(df_results),
        "avg_hit_ratio": df_results["hit_ratio"].mean(),
        "median_hit_ratio": df_results["hit_ratio"].median(),
        "std_hit_ratio": df_results["hit_ratio"].std(),
        "min_hit_ratio": df_results["hit_ratio"].min(),
        "max_hit_ratio": df_results["hit_ratio"].max(),
        "avg_n_tickers": df_results["n_tickers"].mean(),
        "avg_return": df_results["avg_return"].mean(),
        "median_return": df_results["median_return"].mean(),
        "details": df_results,
    }

    return summary


def analyze_ranking_hit_ratios():
    """랭킹별 Top-K 방향 적중률 분석"""

    print("=== Top-K 방향 적중률 분석 시작 ===")

    # 실제 존재하는 데이터 파일들 확인 및 사용
    dataset_path = Path("data/interim/dataset_daily.parquet")
    rebalance_scores_path = Path("data/interim/rebalance_scores.parquet")

    if not dataset_path.exists():
        print("dataset_daily.parquet가 없습니다.")
        return

    # 수익률 데이터 로드
    dataset = pd.read_parquet(dataset_path)
    print(f"수익률 데이터 로드: {len(dataset):,}행")

    # 랭킹 데이터는 rebalance_scores에서 추출 또는 processed 파일 사용
    ranking_sources = []

    # rebalance_scores에서 랭킹 데이터 추출
    if rebalance_scores_path.exists():
        rebalance_data = pd.read_parquet(rebalance_scores_path)
        print(f"rebalance_scores 로드: {len(rebalance_data):,}행")
        print(f"rebalance_scores 컬럼: {list(rebalance_data.columns)}")

        # 단기/장기 랭킹 데이터 생성
        if "score_short" in rebalance_data.columns:
            ranking_short = rebalance_data[["date", "ticker", "score_short"]].copy()
            ranking_short = ranking_short.rename(columns={"score_short": "score"})
            ranking_short["rank"] = ranking_short.groupby("date")["score"].rank(
                method="min", ascending=True
            )
            ranking_sources.append(
                ("단기랭킹 (rebalance)", ranking_short, "ret_fwd_20d")
            )

        if "score_long" in rebalance_data.columns:
            ranking_long = rebalance_data[["date", "ticker", "score_long"]].copy()
            ranking_long = ranking_long.rename(columns={"score_long": "score"})
            ranking_long["rank"] = ranking_long.groupby("date")["score"].rank(
                method="min", ascending=True
            )
            ranking_sources.append(
                ("장기랭킹 (rebalance)", ranking_long, "ret_fwd_120d")
            )

        if "score_ens" in rebalance_data.columns:
            ranking_ensemble = rebalance_data[["date", "ticker", "score_ens"]].copy()
            ranking_ensemble = ranking_ensemble.rename(columns={"score_ens": "score"})
            ranking_ensemble["rank"] = ranking_ensemble.groupby("date")["score"].rank(
                method="min", ascending=True
            )
            ranking_sources.append(
                ("통합랭킹 (rebalance)", ranking_ensemble, "ret_fwd_20d")
            )

    if not ranking_sources:
        print("사용 가능한 랭킹 데이터가 없습니다.")
        return

    print(f'수익률 데이터: {len(dataset):,}행, {dataset["ticker"].nunique()}종목')
    print(f"수익률 데이터 컬럼: {list(dataset.columns)}")

    # 분석 설정
    top_k_values = [10, 20, 30, 50]

    results = {}

    for strategy_name, ranking_df, return_col in ranking_sources:
        print(f"\n--- {strategy_name} ({return_col}) 분석 ---")

        if return_col not in dataset.columns:
            print(f"⚠️ {return_col} 컬럼이 수익률 데이터에 없습니다.")
            continue

        strategy_results = {}

        for top_k in top_k_values:
            print(f"Top-{top_k} 방향 적중률 계산 중...")

            result = calculate_topk_direction_hit_ratio(
                ranking_df=ranking_df,
                returns_df=dataset,
                return_col=return_col,
                top_k=top_k,
            )

            if "error" not in result:
                hit_ratio = result["avg_hit_ratio"]
                print(f'  Top-{top_k}: {hit_ratio:.3f} ({result["total_samples"]}일)')

                strategy_results[top_k] = result
            else:
                print(f'  Top-{top_k}: 계산 실패 - {result["error"]}')
                strategy_results[top_k] = result

        results[strategy_name] = strategy_results

    # 결과 저장 및 출력
    print("\n=== 최종 결과 ===")

    for strategy_name, strategy_data in results.items():
        print(f"\n{strategy_name}:")
        for top_k, result in strategy_data.items():
            if "error" not in result:
                hit_ratio = result["avg_hit_ratio"]
                samples = result["total_samples"]
                avg_return = result["avg_return"]
                print(
                    f"  Top-{top_k}: 방향적중률 {hit_ratio:.3f} ({samples}일), "
                    f"평균수익률 {avg_return:.4f}"
                )
            else:
                print(f"  Top-{top_k}: 계산 실패")

    # 기존 hit_ratio와 비교
    print("\n=== 기존 hit_ratio와 비교 ===")
    try:
        existing_results = pd.read_csv("results/final_track_a_performance_results.csv")
        print("기존 성과 데이터:")
        for _, row in existing_results.iterrows():
            strategy = row["strategy"]
            hit_ratio_dev = row["hit_ratio_dev"]
            hit_ratio_holdout = row["hit_ratio_holdout"]
            print(f"  {strategy}: Dev {hit_ratio_dev}%, Holdout {hit_ratio_holdout}%")
    except Exception as e:
        print(f"기존 데이터 로드 실패: {e}")

    return results


if __name__ == "__main__":
    analyze_ranking_hit_ratios()
