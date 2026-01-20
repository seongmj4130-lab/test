"""
ohlcv_daily 데이터의 전체 기간 coverage 확인
"""

from pathlib import Path

import pandas as pd

data_dir = Path("data/interim")

print("=" * 80)
print("ohlcv_daily 데이터 Coverage 확인")
print("=" * 80)

# ohlcv_daily 로드
ohlcv_path = data_dir / "ohlcv_daily"
if (
    not ohlcv_path.with_suffix(".parquet").exists()
    and not ohlcv_path.with_suffix(".csv").exists()
):
    print("❌ ohlcv_daily 파일이 없습니다.")
    exit(1)

try:
    if ohlcv_path.with_suffix(".parquet").exists():
        ohlcv = pd.read_parquet(ohlcv_path.with_suffix(".parquet"))
    else:
        ohlcv = pd.read_csv(ohlcv_path.with_suffix(".csv"))
        if "date" in ohlcv.columns:
            ohlcv["date"] = pd.to_datetime(ohlcv["date"], errors="coerce")

    print(f"\n✓ ohlcv_daily 로드 완료: {len(ohlcv):,}행")

    # 날짜 범위 확인
    ohlcv["date"] = pd.to_datetime(ohlcv["date"], errors="coerce")
    ohlcv = ohlcv.dropna(subset=["date"])

    min_date = ohlcv["date"].min()
    max_date = ohlcv["date"].max()
    date_range = (max_date - min_date).days

    print("\n날짜 범위:")
    print(f"  시작일: {min_date.strftime('%Y-%m-%d')}")
    print(f"  종료일: {max_date.strftime('%Y-%m-%d')}")
    print(f"  기간: {date_range:,}일 ({date_range/365.25:.1f}년)")

    # 날짜별 coverage 확인
    date_counts = ohlcv.groupby("date").size()
    unique_dates = sorted(ohlcv["date"].unique())

    print("\n날짜별 Coverage:")
    print(f"  고유 날짜 수: {len(unique_dates):,}개")
    print(f"  평균 종목 수/일: {date_counts.mean():.0f}개")
    print(f"  최소 종목 수/일: {date_counts.min()}개")
    print(f"  최대 종목 수/일: {date_counts.max()}개")

    # 연속성 확인 (공휴일/주말 제외)
    date_diffs = [
        (pd.to_datetime(unique_dates[i + 1]) - pd.to_datetime(unique_dates[i])).days
        for i in range(len(unique_dates) - 1)
    ]

    print("\n날짜 간격 분석:")
    print(f"  평균 간격: {pd.Series(date_diffs).mean():.1f}일")
    print(f"  중앙값 간격: {pd.Series(date_diffs).median():.1f}일")
    print(f"  최소 간격: {min(date_diffs)}일")
    print(f"  최대 간격: {max(date_diffs)}일")

    # 큰 간격 확인 (데이터 누락 가능성)
    large_gaps = [d for d in date_diffs if d > 5]
    if large_gaps:
        print(f"\n⚠️  큰 간격 (>5일) 발견: {len(large_gaps)}개")
        print(f"  최대 간격: {max(large_gaps)}일")
    else:
        print(f"\n✓ 날짜 간격이 연속적입니다 (최대 간격: {max(date_diffs)}일)")

    # 연도별 coverage
    ohlcv["year"] = ohlcv["date"].dt.year
    year_counts = ohlcv.groupby("year")["date"].nunique()

    print("\n연도별 Coverage:")
    for year in sorted(year_counts.index):
        print(f"  {year}: {year_counts[year]:,}개 거래일")

    # 필수 컬럼 확인
    required_cols = ["date", "ticker", "close"]
    missing_cols = [col for col in required_cols if col not in ohlcv.columns]
    if missing_cols:
        print(f"\n❌ 필수 컬럼 누락: {missing_cols}")
    else:
        print(f"\n✓ 필수 컬럼 모두 존재: {required_cols}")

    # volume 컬럼 확인
    if "volume" in ohlcv.columns:
        volume_coverage = ohlcv["volume"].notna().sum() / len(ohlcv) * 100
        print("\n거래량 데이터 Coverage:")
        print("  volume 컬럼 존재: ✓")
        print(f"  volume 데이터 비율: {volume_coverage:.1f}%")
    else:
        print("\n⚠️  volume 컬럼 없음 (거래량 지표 사용 불가)")

    # rebalance_dates와 비교
    print("\n" + "=" * 80)
    print("rebalance_dates와 비교")
    print("=" * 80)

    # cv_folds_short에서 test_end 확인
    cv_path = data_dir / "cv_folds_short"
    if cv_path.with_suffix(".parquet").exists() or cv_path.with_suffix(".csv").exists():
        if cv_path.with_suffix(".parquet").exists():
            cv = pd.read_parquet(cv_path.with_suffix(".parquet"))
        else:
            cv = pd.read_csv(cv_path.with_suffix(".csv"))
            if "test_end" in cv.columns:
                cv["test_end"] = pd.to_datetime(cv["test_end"], errors="coerce")

        if "test_end" in cv.columns:
            rebalance_dates = pd.to_datetime(cv["test_end"]).dropna().unique()
            rebalance_dates = sorted(rebalance_dates)

            print("\nrebalance_dates 범위:")
            print(f"  시작일: {rebalance_dates[0].strftime('%Y-%m-%d')}")
            print(f"  종료일: {rebalance_dates[-1].strftime('%Y-%m-%d')}")
            print(f"  총 {len(rebalance_dates)}개 리밸런싱 날짜")

            # rebalance_dates가 ohlcv 범위 내에 있는지 확인
            all_in_range = all(
                (date >= min_date) and (date <= max_date) for date in rebalance_dates
            )

            if all_in_range:
                print("\n✓ 모든 rebalance_dates가 ohlcv 범위 내에 있습니다")
            else:
                out_of_range = [
                    d for d in rebalance_dates if (d < min_date) or (d > max_date)
                ]
                print(
                    f"\n❌ {len(out_of_range)}개 rebalance_dates가 ohlcv 범위를 벗어났습니다"
                )
                print(f"  예시: {out_of_range[:5]}")

            # 각 rebalance_date에 대해 lookback_days=60일 데이터가 있는지 확인
            lookback_days = 60
            coverage_issues = []
            for rebal_date in rebalance_dates[:10]:  # 처음 10개만 확인
                lookback_start = rebal_date - pd.Timedelta(days=lookback_days)
                period_data = ohlcv[
                    (ohlcv["date"] >= lookback_start) & (ohlcv["date"] <= rebal_date)
                ]
                if len(period_data) == 0:
                    coverage_issues.append(rebal_date)

            if coverage_issues:
                print(
                    f"\n⚠️  {len(coverage_issues)}개 rebalance_date에서 lookback_days={lookback_days} 데이터 부족"
                )
            else:
                print(
                    f"\n✓ 샘플 rebalance_dates에서 lookback_days={lookback_days} 데이터 충분"
                )

    print("\n" + "=" * 80)
    print("결론")
    print("=" * 80)
    print(
        f"ohlcv_daily 데이터는 {min_date.strftime('%Y-%m-%d')}부터 {max_date.strftime('%Y-%m-%d')}까지"
    )
    print(f"총 {len(unique_dates):,}개 거래일에 대해 데이터가 확보되어 있습니다.")

except Exception as e:
    print(f"❌ 오류 발생: {e}")
    import traceback

    traceback.print_exc()
