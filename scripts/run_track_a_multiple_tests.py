#!/usr/bin/env python
"""
Track A 다중 실행 테스트 스크립트
Track A 랭킹 생성을 3번 실행해서 결과를 비교
"""

import time

import pandas as pd

from src.tracks.track_a.ranking_service import generate_rankings


def run_track_a_multiple_times(times=3):
    """Track A를 여러 번 실행하고 결과를 수집"""
    results = []

    for i in range(times):
        print(f"=== Track A 실행 {i+1}/{times} ===")
        try:
            rankings = generate_rankings()
            results.append(rankings)
            time.sleep(1)  # 실행 간격
        except Exception as e:
            print(f"실행 {i+1} 실패: {e}")
            results.append(None)

    return results


def analyze_ranking_quality(ranking_df):
    """랭킹 데이터의 품질 분석"""
    if ranking_df is None or len(ranking_df) == 0:
        return None

    # 기본 통계
    stats = {
        "total_records": len(ranking_df),
        "unique_dates": ranking_df["date"].nunique(),
        "unique_tickers": ranking_df["ticker"].nunique(),
        "date_range": f"{ranking_df['date'].min()} ~ {ranking_df['date'].max()}",
        "score_total_mean": ranking_df["score_total"].mean(),
        "score_total_std": ranking_df["score_total"].std(),
        "rank_total_mean": ranking_df["rank_total"].mean(),
        "rank_total_std": ranking_df["rank_total"].std(),
        "top_10_score_avg": ranking_df[ranking_df["rank_total"] <= 10][
            "score_total"
        ].mean(),
        "bottom_10_score_avg": ranking_df[
            ranking_df["rank_total"] > ranking_df["rank_total"].max() - 10
        ]["score_total"].mean(),
    }

    return stats


def compare_track_a_results(results):
    """Track A 결과 비교 및 출력"""
    print("\n=== Track A 3회 실행 결과 비교 ===")

    ranking_types = ["ranking_short_daily", "ranking_long_daily"]

    for ranking_type in ranking_types:
        print(f"\n--- {ranking_type} ---")

        extracted_stats = []
        for i, result in enumerate(results, 1):
            if result and ranking_type in result:
                ranking_df = result[ranking_type]
                stats = analyze_ranking_quality(ranking_df)
                if stats:
                    print(f"실행 {i}:")
                    for key, value in stats.items():
                        if isinstance(value, float):
                            print(f"  {key}: {value:.4f}")
                        else:
                            print(f"  {key}: {value}")
                    extracted_stats.append(stats)
                else:
                    print(f"실행 {i}: 랭킹 데이터 분석 실패")
                    extracted_stats.append(None)
            else:
                print(f"실행 {i}: {ranking_type} 데이터 없음")
                extracted_stats.append(None)

        # 평균 계산 (수치형 데이터만)
        if extracted_stats and all(s is not None for s in extracted_stats):
            print(f"\n{ranking_type} 평균:")
            numeric_keys = [
                "score_total_mean",
                "score_total_std",
                "rank_total_mean",
                "rank_total_std",
                "top_10_score_avg",
                "bottom_10_score_avg",
            ]

            for key in numeric_keys:
                values = [
                    s[key] for s in extracted_stats if key in s and s[key] is not None
                ]
                if values:
                    avg = sum(values) / len(values)
                    std = pd.Series(values).std()
                    print(f"  {key}: {avg:.4f} ± {std:.4f}")


if __name__ == "__main__":
    results = run_track_a_multiple_times(3)
    compare_track_a_results(results)
    print("\n" + "=" * 50 + "\n")
