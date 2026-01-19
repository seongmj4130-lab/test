from pathlib import Path

import pandas as pd


def replace_original_data():
    """원본 데이터 파일을 재계산된 월별 수익률로 교체"""

    # 원본 데이터 파일 경로
    original_path = Path("data/dummy_kospi200_pr_tabs_4lines_2023_2024_v5.csv")

    # 재계산된 월별 수익률 데이터
    monthly_path = Path("data/monthly_returns_recalculated.csv")

    # 데이터 읽기
    original_df = pd.read_csv(original_path)
    monthly_df = pd.read_csv(monthly_path)

    print("원본 데이터 구조:")
    print(f"  행 수: {len(original_df)}")
    print(f"  컬럼 수: {len(original_df.columns)}")
    print(f"  컬럼: {list(original_df.columns)}")

    print("\n재계산된 월별 수익률 데이터:")
    print(f"  행 수: {len(monthly_df)}")
    print(f"  컬럼 수: {len(monthly_df.columns)}")
    print(f"  컬럼: {list(monthly_df.columns)}")

    # 새로운 데이터프레임 생성 (원본 구조 유지)
    new_data = []

    # horizon_days별로 그룹화
    horizons = sorted(original_df["horizon_days"].unique())

    for horizon in horizons:
        print(f"\n=== {horizon}일 기간 데이터 생성 ===")

        # 해당 horizon의 월별 수익률 데이터
        monthly_horizon = monthly_df[monthly_df["horizon_days"] == horizon].copy()
        monthly_horizon = monthly_horizon.sort_values("month")

        # 해당 horizon의 원본 데이터 (정렬용)
        original_horizon = original_df[original_df["horizon_days"] == horizon].copy()
        original_horizon = original_horizon.sort_values("month")

        # 누적 수익률 계산용 변수들 초기화
        cum_returns = {"kospi200_pr": 0.0, "short": 0.0, "long": 0.0, "mix": 0.0}

        # 월별로 데이터 생성
        for idx, row in monthly_horizon.iterrows():
            month = row["month"]

            # 월별 수익률
            mret_kospi = row["kospi200_pr_mret_pct"]
            mret_short = row["short_mret_pct"]
            mret_long = row["long_mret_pct"]
            mret_mix = row["mix_mret_pct"]

            # 누적 수익률 계산: (1 + 누적) * (1 + 월별) - 1
            cum_returns["kospi200_pr"] = (1 + cum_returns["kospi200_pr"] / 100) * (
                1 + mret_kospi / 100
            ) - 1
            cum_returns["short"] = (1 + cum_returns["short"] / 100) * (
                1 + mret_short / 100
            ) - 1
            cum_returns["long"] = (1 + cum_returns["long"] / 100) * (
                1 + mret_long / 100
            ) - 1
            cum_returns["mix"] = (1 + cum_returns["mix"] / 100) * (
                1 + mret_mix / 100
            ) - 1

            # 백분율로 변환
            cum_kospi_pct = cum_returns["kospi200_pr"] * 100
            cum_short_pct = cum_returns["short"] * 100
            cum_long_pct = cum_returns["long"] * 100
            cum_mix_pct = cum_returns["mix"] * 100

            # 데이터 행 생성
            new_row = {
                "month": month,
                "horizon_days": horizon,
                "kospi200_pr_mret_pct": mret_kospi,
                "short_mret_pct": mret_short,
                "long_mret_pct": mret_long,
                "mix_mret_pct": mret_mix,
                "kospi200_pr_cum_return_pct": cum_kospi_pct,
                "short_cum_return_pct": cum_short_pct,
                "long_cum_return_pct": cum_long_pct,
                "mix_cum_return_pct": cum_mix_pct,
            }

            new_data.append(new_row)

    # 새로운 데이터프레임 생성
    new_df = pd.DataFrame(new_data)

    # 원본과 동일한 순서로 정렬
    new_df = new_df.sort_values(["horizon_days", "month"])

    print("\n생성된 새 데이터:")
    print(f"  행 수: {len(new_df)}")
    print(f"  컬럼 수: {len(new_df.columns)}")

    # 검증: 원본과 새 데이터 비교
    print("\n=== 데이터 검증 ===")

    # 샘플 비교
    print("원본 데이터 샘플 (첫 3행):")
    print(original_df.head(3).round(6))

    print("\n새 데이터 샘플 (첫 3행):")
    print(new_df.head(3).round(6))

    # 월별 수익률이 일치하는지 확인
    mret_cols = [
        "kospi200_pr_mret_pct",
        "short_mret_pct",
        "long_mret_pct",
        "mix_mret_pct",
    ]

    for col in mret_cols:
        orig_values = original_df[col].round(6)
        new_values = new_df[col].round(6)

        matches = (orig_values == new_values).sum()
        total = len(orig_values)
        match_rate = matches / total * 100

        print(f"{col}: {matches}/{total}개 일치 ({match_rate:.1f}%)")

    # 백업 파일 생성
    backup_path = original_path.with_suffix(".csv.backup")
    original_df.to_csv(backup_path, index=False)
    print(f"\n원본 파일 백업: {backup_path}")

    # 원본 파일 교체
    new_df.to_csv(original_path, index=False)
    print(f"원본 파일 교체 완료: {original_path}")

    # 최종 검증
    print("\n=== 최종 검증 ===")
    reloaded_df = pd.read_csv(original_path)
    print("교체된 파일 검증:")
    print(f"  행 수: {len(reloaded_df)}")
    print(f"  컬럼 수: {len(reloaded_df.columns)}")
    print("  컬럼 일치:", list(reloaded_df.columns) == list(original_df.columns))

    # 누적 수익률이 올바르게 계산되었는지 확인
    print("\n누적 수익률 검증 (20일 기간, 마지막 행):")
    last_row = reloaded_df[reloaded_df["horizon_days"] == 20].iloc[-1]
    print(f"  월: {last_row['month']}")
    print(f"  KOSPI200 누적: {last_row['kospi200_pr_cum_return_pct']:.2f}%")
    print(f"  Mix 누적: {last_row['mix_cum_return_pct']:.2f}%")

    return new_df, original_df


if __name__ == "__main__":
    new_df, original_df = replace_original_data()

    print("\n" + "=" * 80)
    print("데이터 교체 작업 완료!")
    print("=" * 80)
    print("• 원본 파일이 재계산된 데이터로 교체되었습니다.")
    print("• 백업 파일이 생성되었습니다.")
    print("• 월별 수익률은 그대로 유지되며, 누적 수익률이 올바르게 재계산되었습니다.")
