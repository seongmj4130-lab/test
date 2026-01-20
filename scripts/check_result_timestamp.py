"""결과값의 시점과 설정 확인"""

import os
from datetime import datetime

import pandas as pd

strategies = ["bt20_ens", "bt20_short", "bt120_ens", "bt120_long"]
print("=" * 100)
print("백테스트 결과값의 시점 및 설정 확인")
print("=" * 100)

# 문서에 기록된 기대값
expected = {
    "bt20_ens": {"sharpe": 0.6826, "cagr": 0.1498, "mdd": -0.1098, "calmar": 1.3641},
    "bt20_short": {"sharpe": 0.6464, "cagr": 0.1384, "mdd": -0.0909, "calmar": 1.5223},
    "bt120_ens": {"sharpe": 0.6263, "cagr": 0.1166, "mdd": -0.0769, "calmar": 1.5156},
    "bt120_long": {"sharpe": 0.6839, "cagr": 0.1360, "mdd": -0.0866, "calmar": 1.5700},
}

for s in strategies:
    file_path = f"data/interim/bt_metrics_{s}.parquet"
    if os.path.exists(file_path):
        df = pd.read_parquet(file_path)
        holdout = df[df["phase"] == "holdout"]

        if len(holdout) > 0:
            h = holdout.iloc[0]
            mtime = os.path.getmtime(file_path)
            mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")

            print(f"\n[{s}]")
            print(f"  파일 수정 시간: {mtime_str}")
            print(
                f'  저장된 값: Sharpe={h["net_sharpe"]:.4f}, CAGR={h["net_cagr"]:.4%}, MDD={h["net_mdd"]:.4%}'
            )
            print(
                f'  문서 기대값: Sharpe={expected[s]["sharpe"]:.4f}, CAGR={expected[s]["cagr"]:.4%}, MDD={expected[s]["mdd"]:.4%}'
            )

            # 값 비교
            sharpe_match = abs(h["net_sharpe"] - expected[s]["sharpe"]) < 0.0001
            cagr_match = abs(h["net_cagr"] - expected[s]["cagr"]) < 0.0001

            if sharpe_match and cagr_match:
                print("  ✅ 저장된 값과 문서 기대값 일치")
            else:
                print("  ⚠️ 저장된 값과 문서 기대값 불일치")
                print(
                    f'    차이: Sharpe {abs(h["net_sharpe"] - expected[s]["sharpe"]):.4f}, CAGR {abs(h["net_cagr"] - expected[s]["cagr"]):.4%}'
                )

            # 설정값 확인
            print("\n  실제 사용된 설정:")
            print(f'    top_k: {h["top_k"]}')
            print(f'    holding_days: {h["holding_days"]}')
            print(f'    cost_bps: {h["cost_bps"]}')
            print(f'    buffer_k: {h["buffer_k"]}')
            print(f'    weighting: {h["weighting"]}')
            print(f'    date_start: {h["date_start"]}')
            print(f'    date_end: {h["date_end"]}')

print("\n" + "=" * 100)
print("결론: 문서의 기대값은 run_backtest_4models.py 실행 시 출력된 값입니다.")
print(
    "      저장된 파일은 이전 실행 결과일 수 있으므로, 실행 로그의 출력값을 기준으로 합니다."
)
print("=" * 100)
