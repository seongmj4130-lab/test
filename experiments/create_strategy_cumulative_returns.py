import pandas as pd
import os

# 기존 데이터 로드
df_existing = pd.read_csv('data/holdout_monthly_cumulative_returns.csv')

# 컬럼명 변경 (더 읽기 쉽게)
column_mapping = {
    'bt20_short': 'BT20 단기 (20일)',
    'bt120_long': 'BT120 장기 (120일)',
    'bt20_ens': 'BT20 앙상블 (20일)',
    'bt120_ens': 'BT120 앙상블 (120일)',
    'KOSPI200': 'KOSPI200'
}

df_renamed = df_existing.rename(columns=column_mapping)

# 데이터 검증
print("=== 4가지 전략 + KOSPI200 월별 누적 수익률 데이터 ===")
print(f"기간: {df_renamed['date'].min()} ~ {df_renamed['date'].max()}")
print(f"총 데이터 포인트: {len(df_renamed)}개월")
print(f"전략 수: {len(df_renamed.columns) - 1}개")  # date 제외

print("\n=== 데이터 미리보기 ===")
print(df_renamed.head())

print("\n=== 최종 수익률 (2024-10-31 기준) ===")
final_row = df_renamed.iloc[-1]
for col in df_renamed.columns:
    if col != 'date':
        value = final_row[col]
        pct_value = ".1f"
        print(f"{col}: {pct_value}")

# CSV 파일로 저장 (명확한 이름으로)
output_csv = 'data/strategies_kospi200_monthly_cumulative_returns.csv'
df_renamed.to_csv(output_csv, index=False, encoding='utf-8-sig')
print(f"\n✅ CSV 파일 저장 완료: {output_csv}")

# Parquet 파일로 저장
output_parquet = 'data/strategies_kospi200_monthly_cumulative_returns.parquet'
df_renamed.to_parquet(output_parquet, index=False)
print(f"✅ Parquet 파일 저장 완료: {output_parquet}")

# 월별 수익률 계산 (전월 대비)
print("\n=== 월별 수익률 계산 (전월 대비) ===")
df_monthly_returns = df_renamed.copy()
for col in df_renamed.columns:
    if col != 'date':
        # 전월 수익률 계산
        df_monthly_returns[f'{col}_monthly_return'] = df_monthly_returns[col].pct_change().fillna(0)

# 월별 수익률 CSV 저장
monthly_csv = 'data/strategies_kospi200_monthly_returns.csv'
df_monthly_returns.to_csv(monthly_csv, index=False, encoding='utf-8-sig')
print(f"✅ 월별 수익률 CSV 저장 완료: {monthly_csv}")

# 월별 수익률 Parquet 저장
monthly_parquet = 'data/strategies_kospi200_monthly_returns.parquet'
df_monthly_returns.to_parquet(monthly_parquet, index=False)
print(f"✅ 월별 수익률 Parquet 저장 완료: {monthly_parquet}")

print("\n=== 데이터 구조 설명 ===")
print("📊 누적 수익률 데이터:")
print("  - date: 월말 날짜")
print("  - 각 전략 컬럼: 시작점(2023-01-31)을 기준으로 한 누적 수익률")
print("  - 예: 1.0 = 100% (원금), 1.10 = 110% (10% 수익)")
print()
print("📈 월별 수익률 데이터:")
print("  - date: 월말 날짜")
print("  - 각 전략 컬럼: 해당 월의 수익률 (전월 대비)")
print("  - monthly_return 컬럼: 전월 대비 수익률")
print("  - 예: 0.05 = 5% 수익, -0.03 = 3% 손실")

print("\n=== 전략별 성과 요약 ===")
summary_stats = {}
for col in df_renamed.columns:
    if col != 'date':
        final_return = df_renamed[col].iloc[-1] - 1  # 누적 수익률에서 1을 빼서 순수익 계산
        monthly_returns = df_monthly_returns[f'{col}_monthly_return']
        positive_months = (monthly_returns > 0).sum()
        total_months = len(monthly_returns)

        summary_stats[col] = {
            'final_cumulative_return': ".1f",
            'win_rate': ".1f",
            'best_month': ".1f",
            'worst_month': ".1f"
        }

for strategy, stats in summary_stats.items():
    print(f"\n{strategy}:")
    print(f"  누적 수익률: {stats['final_cumulative_return']}")
    print(f"  월별 승률: {stats['win_rate']}")
    print(f"  최고 월 수익률: {stats['best_month']}")
    print(f"  최악 월 수익률: {stats['worst_month']}")