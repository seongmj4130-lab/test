import pandas as pd
import numpy as np

# 데이터 읽기
df = pd.read_csv('data/ui_strategies_cumulative_comparison.csv')
df['month'] = pd.to_datetime(df['month'])
df = df.set_index('month')

print("📊 KOSPI200 vs 전략 비교 데이터 분석")
print("=" * 60)

# 1. 월별 수익률 계산 (누적수익률 → 월별 수익률)
monthly_returns = df.pct_change().fillna(0) * 100  # 퍼센트로 변환

print("\n1️⃣ 월별 수익률 데이터 (첫 5개월):")
print("-" * 40)
print(monthly_returns.head().round(2))

# 2. 초과수익률 계산 (전략 - KOSPI200)
excess_returns = monthly_returns.subtract(monthly_returns['kospi200'], axis=0)

print("\n2️⃣ 초과수익률 데이터 (전략 - KOSPI200, 첫 5개월):")
print("-" * 40)
excess_cols = [col for col in excess_returns.columns if col != 'kospi200']
print(excess_returns[excess_cols].head().round(2))

# 3. 전략 그룹별 분석
strategies = {
    'BT20 Short': [f'bt20_short_{period}' for period in [20, 40, 60, 80, 100, 120]],
    'BT120 Long': [f'bt120_long_{period}' for period in [20, 40, 60, 80, 100, 120]],
    'BT20 Ensemble': [f'bt20_ens_{period}' for period in [20, 40, 60, 80, 100, 120]]
}

print("\n3️⃣ 전략 그룹별 월별 초과수익률 평균:")
print("-" * 45)
for group_name, cols in strategies.items():
    group_excess = excess_returns[cols]
    avg_excess = group_excess.mean()
    print(f"\n{group_name}:")
    for col, avg in avg_excess.items():
        period = col.split('_')[-1]
        print(".2f")

# 4. 누적 초과수익률
cumulative_excess = excess_returns.cumsum()

print("\n4️⃣ 누적 초과수익률 (전체 기간):")
print("-" * 35)
for group_name, cols in strategies.items():
    print(f"\n{group_name}:")
    final_cumulative = cumulative_excess[cols].iloc[-1]
    for col, cum_return in final_cumulative.items():
        period = col.split('_')[-1]
        print(".1f")

# 5. 승률 계산 (월별 초과수익률 > 0)
print("\n5️⃣ 승률 분석 (월별 초과수익률 > 0):")
print("-" * 30)
for group_name, cols in strategies.items():
    print(f"\n{group_name}:")
    win_rates = (excess_returns[cols] > 0).mean() * 100
    for col, win_rate in win_rates.items():
        period = col.split('_')[-1]
        print(".1f")

# 6. 변동성 비교
print("\n6️⃣ 변동성 비교 (월별 수익률 표준편차):")
print("-" * 35)
print(f"KOSPI200: {monthly_returns['kospi200'].std():.2f}%")
for group_name, cols in strategies.items():
    print(f"\n{group_name}:")
    for col in cols:
        vol = monthly_returns[col].std()
        period = col.split('_')[-1]
        print(".2f")

# 7. 샤프 지수 계산 (연율화)
print("\n7️⃣ 샤프 지수 (연율화, 무위험수익률 2% 가정):")
print("-" * 40)
risk_free_rate = 2.0  # 연간 2%

print(f"KOSPI200: {(monthly_returns['kospi200'].mean() * 12 - risk_free_rate) / (monthly_returns['kospi200'].std() * np.sqrt(12)):.2f}")

for group_name, cols in strategies.items():
    print(f"\n{group_name}:")
    for col in cols:
        mean_return = monthly_returns[col].mean() * 12  # 연율화
        volatility = monthly_returns[col].std() * np.sqrt(12)  # 연율화
        sharpe = (mean_return - risk_free_rate) / volatility
        period = col.split('_')[-1]
        print(".2f")

print("\n" + "=" * 60)
print("💡 비교 방법 요약:")
print("1. 월별 수익률: pct_change()로 계산")
print("2. 초과수익률: 전략_수익률 - KOSPI200_수익률")
print("3. 승률: 초과수익률 > 0 비율")
print("4. 샤프 지수: (평균초과수익률) / 변동성")
print("5. MDD: 누적수익률에서 최고점-최저점")

# 데이터 저장
monthly_returns.to_csv('data/monthly_returns_comparison.csv')
excess_returns.to_csv('data/excess_returns_comparison.csv')
cumulative_excess.to_csv('data/cumulative_excess_returns.csv')

print("\n💾 분석 데이터 저장:")
print("   - 월별 수익률: data/monthly_returns_comparison.csv")
print("   - 초과수익률: data/excess_returns_comparison.csv")
print("   - 누적 초과수익률: data/cumulative_excess_returns.csv")