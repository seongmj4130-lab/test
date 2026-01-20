"""L6 Interval20이 L8 랭킹을 올바르게 포함하는지 확인"""

import pandas as pd

l6_scores_interval20 = pd.read_parquet(
    "data/interim/rebalance_scores_from_ranking_interval_20.parquet"
)
l8_short = pd.read_parquet("data/interim/ranking_short_daily.parquet")
l8_long = pd.read_parquet("data/interim/ranking_long_daily.parquet")

# 공통 날짜 선택
common_dates = set(l6_scores_interval20["date"].unique()) & set(
    l8_short["date"].unique()
)
test_date = sorted(common_dates)[0]

print(f"테스트 날짜: {test_date}")
print("=" * 100)

l6_date = l6_scores_interval20[l6_scores_interval20["date"] == test_date].copy()
l8_short_date = l8_short[l8_short["date"] == test_date].copy()
l8_long_date = l8_long[l8_long["date"] == test_date].copy()

# L6와 L8 Short 병합
merged_short = l6_date.merge(
    l8_short_date[["ticker", "score_total", "rank_total"]],
    on="ticker",
    how="left",
    suffixes=("_l6", "_l8"),
)

# score_total_short 비교 (L6의 score_total_short가 L8 Short와 일치해야 함)
if (
    "score_total_short" in merged_short.columns
    and "score_total_l8" in merged_short.columns
):
    valid = merged_short[
        merged_short["score_total_l8"].notna()
        & merged_short["score_total_short"].notna()
    ]
    if len(valid) > 0:
        corr = valid["score_total_short"].corr(valid["score_total_l8"])
        print("\nL6 score_total_short vs L8 Short score_total:")
        print(f"  상관계수: {corr:.4f}")
        print(f"  일치하는 종목 수: {len(valid)}/{len(merged_short)}")

        # 완전 일치 여부
        exact_match = (valid["score_total_short"] == valid["score_total_l8"]).sum()
        print(f"  완전 일치 종목 수: {exact_match}/{len(valid)}")

        if exact_match == len(valid):
            print("  ✅ 완전 일치")
        else:
            print("  ⚠️ 일부 불일치")
            diff = (valid["score_total_short"] - valid["score_total_l8"]).abs()
            print(f"  최대 차이: {diff.max():.6f}")
            print(f"  평균 차이: {diff.mean():.6f}")

# score_total 비교 (L6의 score_total은 score_ens일 수 있음)
if "score_total" in merged_short.columns and "score_total_l8" in merged_short.columns:
    valid = merged_short[
        merged_short["score_total_l8"].notna() & merged_short["score_total"].notna()
    ]
    if len(valid) > 0:
        corr = valid["score_total"].corr(valid["score_total_l8"])
        print("\nL6 score_total vs L8 Short score_total (참고):")
        print(f"  상관계수: {corr:.4f}")
        print("  참고: L6의 score_total은 score_ens일 수 있음")

# L6와 L8 Long 병합
merged_long = l6_date.merge(
    l8_long_date[["ticker", "score_total", "rank_total"]],
    on="ticker",
    how="left",
    suffixes=("_l6", "_l8"),
)

# score_total_long 비교
if (
    "score_total_long" in merged_long.columns
    and "score_total_l8" in merged_long.columns
):
    valid = merged_long[
        merged_long["score_total_l8"].notna() & merged_long["score_total_long"].notna()
    ]
    if len(valid) > 0:
        corr = valid["score_total_long"].corr(valid["score_total_l8"])
        print("\nL6 score_total_long vs L8 Long score_total:")
        print(f"  상관계수: {corr:.4f}")
        print(f"  일치하는 종목 수: {len(valid)}/{len(merged_long)}")

        # 완전 일치 여부
        exact_match = (valid["score_total_long"] == valid["score_total_l8"]).sum()
        print(f"  완전 일치 종목 수: {exact_match}/{len(valid)}")

        if exact_match == len(valid):
            print("  ✅ 완전 일치")
        else:
            print("  ⚠️ 일부 불일치")
            diff = (valid["score_total_long"] - valid["score_total_l8"]).abs()
            print(f"  최대 차이: {diff.max():.6f}")
            print(f"  평균 차이: {diff.mean():.6f}")
