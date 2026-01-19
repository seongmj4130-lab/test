"""CSV 파일 수정 시간 확인"""
import os
from datetime import datetime

csv_path = "artifacts/reports/backtest_4models_comparison.csv"
if os.path.exists(csv_path):
    mtime = os.path.getmtime(csv_path)
    mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
    print(f"backtest_4models_comparison.csv 수정 시간: {mtime_str}")

    # 파일 내용 확인
    import pandas as pd

    df = pd.read_csv(csv_path)
    print("\nCSV 파일 내용:")
    print(df.to_string(index=False))
else:
    print("CSV 파일이 없습니다.")
