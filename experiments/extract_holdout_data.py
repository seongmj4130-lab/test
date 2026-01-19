import pandas as pd

# ui_cumulative_returns.csv 파일 읽기
df = pd.read_csv("data/ui_cumulative_returns.csv")

# 홀드아웃 기간 필터링 (2023년 이후)
df_holdout = df[df["date"] >= "2023-01-31"].copy()

# 필요한 전략만 선택
strategies = ["bt20_short", "bt120_long", "bt20_ens", "bt120_ens", "KOSPI200"]
df_filtered = df_holdout[df_holdout["strategy"].isin(strategies)].copy()

# 피벗 테이블로 변환 (날짜별 전략별 누적 수익률)
df_pivot = df_filtered.pivot(
    index="date", columns="strategy", values="cumulative_return"
).reset_index()

# 날짜를 datetime으로 변환하고 정렬
df_pivot["date"] = pd.to_datetime(df_pivot["date"])
df_pivot = df_pivot.sort_values("date")

# CSV 파일로 저장
output_path_csv = "data/holdout_monthly_cumulative_returns.csv"
df_pivot.to_csv(output_path_csv, index=False)
print(f"CSV 파일 저장 완료: {output_path_csv}")

# Parquet 파일로 저장
output_path_parquet = "data/holdout_monthly_cumulative_returns.parquet"
df_pivot.to_parquet(output_path_parquet, index=False)
print(f"Parquet 파일 저장 완료: {output_path_parquet}")

# 데이터 확인
print("\n데이터 미리보기:")
print(df_pivot.head())
print(f'\n전체 기간: {df_pivot["date"].min()} ~ {df_pivot["date"].max()}')
print(f"총 데이터 포인트: {len(df_pivot)}")
