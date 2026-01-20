#!/usr/bin/env python3
"""
cv_folds 데이터 구조 및 dev/holdout 구분 확인
"""

from pathlib import Path

import pandas as pd


def analyze_cv_folds():
    """cv_folds 데이터 구조 분석"""

    print("=== CV Folds 데이터 구조 분석 ===")

    # cv_folds 데이터 확인
    cv_path = Path("data/interim/cv_folds_short.parquet")
    if cv_path.exists():
        df = pd.read_parquet(cv_path)
        print("cv_folds_short 데이터 구조:")
        print(f"총 행 수: {len(df)}")
        print(f"컬럼: {list(df.columns)}")
        print()
        print("처음 5행:")
        print(df.head())
        print()
        print("segment별 분포:")
        print(df["segment"].value_counts())
        print()
        print("fold_id별 분포:")
        fold_counts = df["fold_id"].value_counts().head(10)
        print(fold_counts)

        # dev vs holdout 구분
        print()
        print("=== Dev vs Holdout 구분 ===")
        dev_data = df[df["segment"] == "dev"]
        holdout_data = df[df["segment"] == "holdout"]

        print(f"Dev 데이터: {len(dev_data)}행")
        print(f"Holdout 데이터: {len(holdout_data)}행")

        if len(dev_data) > 0:
            print(
                f'Dev test 기간: {dev_data["test_start"].min()} ~ {dev_data["test_end"].max()}'
            )

        if len(holdout_data) > 0:
            print(
                f'Holdout test 기간: {holdout_data["test_start"].min()} ~ {holdout_data["test_end"].max()}'
            )

        # fold_id 패턴 분석
        print()
        print("=== Fold ID 패턴 분석 ===")
        dev_folds = df[df["fold_id"].str.startswith("dev")]["fold_id"].unique()
        holdout_folds = df[df["fold_id"].str.startswith("holdout")]["fold_id"].unique()

        print(f"Dev folds: {sorted(dev_folds)}")
        print(f"Holdout folds: {sorted(holdout_folds)}")

    # rebalance_scores 데이터와의 관계 확인
    print()
    print("=== Rebalance Scores와의 관계 ===")
    rebalance_path = Path("data/interim/rebalance_scores.parquet")
    if rebalance_path.exists():
        df_rebalance = pd.read_parquet(rebalance_path)
        print(f"rebalance_scores 총 행 수: {len(df_rebalance)}")

        # 날짜 범위 비교
        cv_dates = set(df["date"].unique()) if cv_path.exists() else set()
        rebalance_dates = set(df_rebalance["date"].unique())

        common_dates = cv_dates & rebalance_dates
        print(f"CV 데이터 날짜 수: {len(cv_dates)}")
        print(f"Rebalance 데이터 날짜 수: {len(rebalance_dates)}")
        print(f"공통 날짜 수: {len(common_dates)}")

        if len(common_dates) > 0:
            common_sorted = sorted(list(common_dates))
            print(f"공통 날짜 범위: {common_sorted[0]} ~ {common_sorted[-1]}")


if __name__ == "__main__":
    analyze_cv_folds()
