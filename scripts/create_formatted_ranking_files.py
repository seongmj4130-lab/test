#!/usr/bin/env python3
"""
단기/장기 랭킹 데이터를 지정된 양식으로 변환하여 CSV와 Parquet 파일 생성
"""

from pathlib import Path

import pandas as pd


def create_formatted_ranking_files():
    """단기/장기 랭킹 파일 생성"""

    project_root = Path("C:/Users/seong/OneDrive/Desktop/bootcamp/000_code")

    # 장기 파일 처리
    print("=== 장기 랭킹 파일 생성 중 ===")
    long_file = (
        project_root
        / "data"
        / "daily_all_business_days_long_ranking_top20_formatted.csv"
    )
    long_df = pd.read_csv(long_file)

    # 필요한 컬럼들만 선택 및 재구성
    long_result = pd.DataFrame(
        {
            "랭킹": long_df["ranking"],
            "종목명(ticker)": long_df["company_name"].astype(str)
            + "("
            + long_df["ticker_formatted"].astype(str)
            + ")",
            "날짜": long_df["date"],
            "score": long_df["score_ens"],
            "top3 피쳐그룹": long_df["top1_feature_group"].astype(str)
            + ","
            + long_df["top2_feature_group"].astype(str)
            + ","
            + long_df["top3_feature_group"].astype(str),
        }
    )

    # CSV 저장
    long_csv_path = project_root / "data" / "daily_ranking_long_formatted.csv"
    long_result.to_csv(long_csv_path, index=False, encoding="utf-8-sig")
    print(f"장기 CSV 파일 생성 완료: {long_csv_path}")

    # Parquet 저장
    long_parquet_path = project_root / "data" / "daily_ranking_long_formatted.parquet"
    long_result.to_parquet(long_parquet_path, index=False)
    print(f"장기 Parquet 파일 생성 완료: {long_parquet_path}")

    # 단기 파일 처리
    print("\n=== 단기 랭킹 파일 생성 중 ===")
    short_file = (
        project_root
        / "data"
        / "daily_all_business_days_short_ranking_top20_formatted.csv"
    )
    short_df = pd.read_csv(short_file)

    # 필요한 컬럼들만 선택 및 재구성
    short_result = pd.DataFrame(
        {
            "랭킹": short_df["ranking"],
            "종목명(ticker)": short_df["company_name"].astype(str)
            + "("
            + short_df["ticker_formatted"].astype(str)
            + ")",
            "날짜": short_df["date"],
            "score": short_df["score_ens"],
            "top3 피쳐그룹": short_df["top1_feature_group"].astype(str)
            + ","
            + short_df["top2_feature_group"].astype(str)
            + ","
            + short_df["top3_feature_group"].astype(str),
        }
    )

    # CSV 저장
    short_csv_path = project_root / "data" / "daily_ranking_short_formatted.csv"
    short_result.to_csv(short_csv_path, index=False, encoding="utf-8-sig")
    print(f"단기 CSV 파일 생성 완료: {short_csv_path}")

    # Parquet 저장
    short_parquet_path = project_root / "data" / "daily_ranking_short_formatted.parquet"
    short_result.to_parquet(short_parquet_path, index=False)
    print(f"단기 Parquet 파일 생성 완료: {short_parquet_path}")

    # 결과 확인
    print("\n=== 생성된 파일 정보 ===")
    print(f"장기 데이터 행 수: {len(long_result)}")
    print(f"단기 데이터 행 수: {len(short_result)}")

    print("\n=== 샘플 데이터 (장기) ===")
    print(long_result.head(3).to_string())

    print("\n=== 샘플 데이터 (단기) ===")
    print(short_result.head(3).to_string())

    print("\n=== 파일 생성 완료 ===")
    print("장기 CSV: data/daily_ranking_long_formatted.csv")
    print("장기 Parquet: data/daily_ranking_long_formatted.parquet")
    print("단기 CSV: data/daily_ranking_short_formatted.csv")
    print("단기 Parquet: data/daily_ranking_short_formatted.parquet")


if __name__ == "__main__":
    create_formatted_ranking_files()
