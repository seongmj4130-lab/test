"""
랭킹 데이터와 rebalance_scores의 날짜 분포 비교
"""

from pathlib import Path

import numpy as np
import pandas as pd

data_dir = Path("data/interim")

print("=" * 80)
print("랭킹 데이터 날짜 분포 확인")
print("=" * 80)

# ranking_short_daily 확인
ranking_short_path = data_dir / "ranking_short_daily.parquet"
if ranking_short_path.exists():
    df_short = pd.read_parquet(ranking_short_path)
    print("\nranking_short_daily:")
    print(f"  총 행 수: {len(df_short):,}")
    print(f"  날짜 수: {df_short['date'].nunique():,}")

    # 날짜 간격 분석
    dates = sorted(df_short["date"].unique())
    if len(dates) > 1:
        date_diffs = [
            (pd.to_datetime(dates[i + 1]) - pd.to_datetime(dates[i])).days
            for i in range(len(dates) - 1)
        ]
        print("  날짜 간격 통계:")
        print(f"    평균: {np.mean(date_diffs):.1f}일")
        print(f"    중앙값: {np.median(date_diffs):.1f}일")
        print(f"    최소: {min(date_diffs)}일")
        print(f"    최대: {max(date_diffs)}일")
        print("  샘플 날짜 (처음 20개):")
        for d in dates[:20]:
            print(f"    {d}")
else:
    print("\nranking_short_daily 파일이 없습니다.")

# ranking_long_daily 확인
ranking_long_path = data_dir / "ranking_long_daily.parquet"
if ranking_long_path.exists():
    df_long = pd.read_parquet(ranking_long_path)
    print("\nranking_long_daily:")
    print(f"  총 행 수: {len(df_long):,}")
    print(f"  날짜 수: {df_long['date'].nunique():,}")

    # 날짜 간격 분석
    dates = sorted(df_long["date"].unique())
    if len(dates) > 1:
        date_diffs = [
            (pd.to_datetime(dates[i + 1]) - pd.to_datetime(dates[i])).days
            for i in range(len(dates) - 1)
        ]
        print("  날짜 간격 통계:")
        print(f"    평균: {np.mean(date_diffs):.1f}일")
        print(f"    중앙값: {np.median(date_diffs):.1f}일")
        print(f"    최소: {min(date_diffs)}일")
        print(f"    최대: {max(date_diffs)}일")
else:
    print("\nranking_long_daily 파일이 없습니다.")

# rebalance_scores 확인
rebalance_scores_path = data_dir / "rebalance_scores_from_ranking.parquet"
if rebalance_scores_path.exists():
    df_scores = pd.read_parquet(rebalance_scores_path)
    print("\nrebalance_scores_from_ranking:")
    print(f"  총 행 수: {len(df_scores):,}")
    print(f"  날짜 수: {df_scores['date'].nunique():,}")

    # 날짜 간격 분석
    dates = sorted(df_scores["date"].unique())
    if len(dates) > 1:
        date_diffs = [
            (pd.to_datetime(dates[i + 1]) - pd.to_datetime(dates[i])).days
            for i in range(len(dates) - 1)
        ]
        print("  날짜 간격 통계:")
        print(f"    평균: {np.mean(date_diffs):.1f}일")
        print(f"    중앙값: {np.median(date_diffs):.1f}일")
        print(f"    최소: {min(date_diffs)}일")
        print(f"    최대: {max(date_diffs)}일")
        print("  샘플 날짜 (처음 20개):")
        for d in dates[:20]:
            print(f"    {d}")

    # ranking과 rebalance_scores의 날짜 비교
    if ranking_short_path.exists():
        ranking_dates = set(df_short["date"].unique())
        rebalance_dates = set(df_scores["date"].unique())
        print("\n날짜 비교:")
        print(f"  ranking_short_daily 날짜 수: {len(ranking_dates):,}")
        print(f"  rebalance_scores 날짜 수: {len(rebalance_dates):,}")
        print(
            f"  rebalance_scores가 ranking의 부분집합인가? {rebalance_dates.issubset(ranking_dates)}"
        )
        print(
            f"  rebalance_scores에만 있는 날짜: {len(rebalance_dates - ranking_dates)}개"
        )
        print(f"  ranking에만 있는 날짜: {len(ranking_dates - rebalance_dates):,}개")
else:
    print("\nrebalance_scores_from_ranking 파일이 없습니다.")

# cv_folds_short 확인
cv_folds_path = data_dir / "cv_folds_short.parquet"
if cv_folds_path.exists():
    df_cv = pd.read_parquet(cv_folds_path)
    print("\ncv_folds_short:")
    print(f"  총 행 수: {len(df_cv):,}")
    if "test_end" in df_cv.columns:
        test_ends = sorted(df_cv["test_end"].unique())
        print(f"  test_end 날짜 수: {len(test_ends):,}")
        if len(test_ends) > 1:
            date_diffs = [
                (pd.to_datetime(test_ends[i + 1]) - pd.to_datetime(test_ends[i])).days
                for i in range(len(test_ends) - 1)
            ]
            print("  test_end 날짜 간격 통계:")
            print(f"    평균: {np.mean(date_diffs):.1f}일")
            print(f"    중앙값: {np.median(date_diffs):.1f}일")
            print(f"    최소: {min(date_diffs)}일")
            print(f"    최대: {max(date_diffs)}일")
            print("  샘플 test_end 날짜 (처음 20개):")
            for d in test_ends[:20]:
                print(f"    {d}")
else:
    print("\ncv_folds_short 파일이 없습니다.")
