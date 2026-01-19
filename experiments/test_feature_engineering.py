# 피쳐 엔지니어링 테스트
import numpy as np
import pandas as pd

# 샘플 데이터 생성
sample_data = pd.DataFrame(
    {
        "date": pd.date_range("2020-01-01", periods=100, freq="D"),
        "ticker": ["AAPL"] * 100,
        "open": np.random.uniform(100, 200, 100),
        "high": np.random.uniform(100, 200, 100),
        "low": np.random.uniform(100, 200, 100),
        "close": np.random.uniform(100, 200, 100),
    }
)

print("샘플 데이터 생성 완료")
print(f"데이터 shape: {sample_data.shape}")

# 간단한 피쳐 생성 테스트
sample_data["close_to_52w_high"] = (
    sample_data["close"] / sample_data["close"].rolling(10, min_periods=5).max()
)

sample_data["intraday_price_position"] = (sample_data["close"] - sample_data["low"]) / (
    sample_data["high"] - sample_data["low"]
).replace(0, np.nan)

print("피쳐 생성 완료")
print(
    "새 컬럼들:",
    [
        col
        for col in sample_data.columns
        if col not in ["date", "ticker", "open", "high", "low", "close"]
    ],
)
print("샘플 값들:")
print(sample_data[["close", "close_to_52w_high", "intraday_price_position"]].head(10))

# NaN 값 확인
print("NaN 값 개수:")
print(sample_data.isnull().sum())
