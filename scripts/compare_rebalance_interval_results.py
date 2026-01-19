# -*- coding: utf-8 -*-
"""
rebalance_interval 적용 전후 결과 비교
"""
import pandas as pd
from pathlib import Path

data_dir = Path("data/interim")

print("=" * 80)
print("rebalance_interval 적용 전후 백테스트 결과 비교")
print("=" * 80)

# 기존 결과 (rebalance_interval=1 또는 월별)
baseline_files = {
    "bt20_short": "bt_metrics_bt20_short.parquet",
    "bt20_ens": "bt_metrics_bt20_ens.parquet",
    "bt120_long": "bt_metrics_bt120_long.parquet",
    "bt120_ens": "bt_metrics_bt120_ens.parquet",
}

# 새로운 결과 (rebalance_interval 적용)
new_files = {
    "bt20_short": "bt_metrics_bt20_short.parquet",  # 같은 파일 (덮어쓰기됨)
    "bt20_ens": "bt_metrics_bt20_ens.parquet",
    "bt120_long": "bt_metrics_bt120_long.parquet",
    "bt120_ens": "bt_metrics_bt120_ens.parquet",
}

# rebalance_scores 날짜 수 확인
print("\n" + "=" * 80)
print("rebalance_scores 날짜 수 확인")
print("=" * 80)

for model_name in ["bt20_short", "bt20_ens", "bt120_long", "bt120_ens"]:
    # rebalance_interval별로 다른 캐시 파일 확인
    for interval in [1, 20, 120]:
        scores_path = data_dir / f"rebalance_scores_from_ranking_interval_{interval}.parquet"
        if scores_path.exists():
            df = pd.read_parquet(scores_path)
            print(f"\n{model_name} (rebalance_interval={interval}):")
            print(f"  rebalance_scores 날짜 수: {df['date'].nunique():,}개")
            print(f"  총 행 수: {len(df):,}행")

# 백테스트 결과 확인
print("\n" + "=" * 80)
print("백테스트 지표 비교")
print("=" * 80)

for model_name in ["bt20_short", "bt20_ens", "bt120_long", "bt120_ens"]:
    metrics_path = data_dir / f"bt_metrics_{model_name}.parquet"
    if metrics_path.exists():
        df = pd.read_parquet(metrics_path)
        print(f"\n{model_name}:")
        cols = ["phase", "net_cagr", "net_sharpe", "net_mdd"]
        if "avg_turnover" in df.columns:
            cols.append("avg_turnover")
        elif "turnover_oneway_mean" in df.columns:
            cols.append("turnover_oneway_mean")
        print(df[cols].to_string())
        
        # 리밸런싱 날짜 수 확인
        returns_path = data_dir / f"bt_returns_{model_name}.parquet"
        if returns_path.exists():
            returns_df = pd.read_parquet(returns_path)
            print(f"\n  리밸런싱 날짜 수: {returns_df['date'].nunique():,}개")
            print(f"  Dev: {returns_df[returns_df['phase']=='dev']['date'].nunique():,}개")
            print(f"  Holdout: {returns_df[returns_df['phase']=='holdout']['date'].nunique():,}개")

print("\n" + "=" * 80)
print("요약")
print("=" * 80)
print("BT20 모델 (rebalance_interval=20):")
print("  - 예상 리밸런싱 날짜: 약 123개 (전체 2,458개 중)")
print("  - 실제 리밸런싱 날짜: 로그에서 확인 필요")
print("\nBT120 모델 (rebalance_interval=120):")
print("  - 예상 리밸런싱 날짜: 약 20개 (전체 2,458개 중)")
print("  - 실제 리밸런싱 날짜: 로그에서 확인 필요")

