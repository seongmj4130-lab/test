from pathlib import Path

import pandas as pd

# 실제 데이터의 마지막 날짜 확인
data_paths = [
    "data/interim/dataset_daily.parquet",
    "data/strategies_daily_returns_holdout.csv",
]

for data_path in data_paths:
    if Path(data_path).exists():
        if data_path.endswith(".parquet"):
            df = pd.read_parquet(data_path)
        else:
            df = pd.read_csv(data_path)

        print(f"{data_path}:")
        print(f'  데이터 기간: {df["date"].min()} ~ {df["date"].max()}')

        # Holdout 관련 데이터 확인
        if "holdout" in data_path.lower():
            print(f"  데이터 shape: {df.shape}")
            print("  마지막 5개 날짜:")
            last_dates = df["date"].tail(5).tolist()
            for date in last_dates:
                print(f"    {date}")

        # 전략 수익률 데이터의 경우
        if "strategies" in data_path:
            print(f"  컬럼들: {list(df.columns)}")
            # 0이 아닌 값들의 마지막 날짜 확인
            non_zero_cols = df.columns[1:]  # date 제외
            non_zero_mask = (df[non_zero_cols] != 0).any(axis=1)
            if non_zero_mask.any():
                last_non_zero_date = df.loc[non_zero_mask, "date"].max()
                print(f"  마지막 실제 수익률 날짜: {last_non_zero_date}")

            # 각 전략별 마지막 데이터 확인
            for col in non_zero_cols:
                non_zero_data = df[df[col] != 0]
                if len(non_zero_data) > 0:
                    last_date = non_zero_data["date"].max()
                    print(f"  {col}: 마지막 데이터 {last_date}")

        print()
