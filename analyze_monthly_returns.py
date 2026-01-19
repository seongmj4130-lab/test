from pathlib import Path

import numpy as np
import pandas as pd


def analyze_monthly_returns():
    """월별 수익률 분석 및 보고서 생성"""

    # 데이터 파일 읽기
    data_path = Path("data/monthly_returns_recalculated.csv")
    df = pd.read_csv(data_path)

    print("=== 월별 수익률 분석 보고서 ===\n")

    # 기간별 분석
    horizons = sorted(df['horizon_days'].unique())

    # 1. 연간 누적 수익률 계산
    print("1. 연간 누적 수익률 분석:")
    print("-" * 50)

    yearly_results = []

    for horizon in horizons:
        horizon_data = df[df['horizon_days'] == horizon].copy()
        horizon_data['year'] = horizon_data['month'].str[:4]

        for year in ['2023', '2024']:
            year_data = horizon_data[horizon_data['year'] == year]

            if len(year_data) > 0:
                # 연간 누적 수익률 계산: (1 + 월별수익률/100)의 곱 - 1
                kospi_cum = (1 + year_data['kospi200_pr_mret_pct']/100).prod() - 1
                short_cum = (1 + year_data['short_mret_pct']/100).prod() - 1
                long_cum = (1 + year_data['long_mret_pct']/100).prod() - 1
                mix_cum = (1 + year_data['mix_mret_pct']/100).prod() - 1

                yearly_results.append({
                    'horizon_days': horizon,
                    'year': year,
                    'kospi200_cum': kospi_cum * 100,
                    'short_cum': short_cum * 100,
                    'long_cum': long_cum * 100,
                    'mix_cum': mix_cum * 100
                })

    yearly_df = pd.DataFrame(yearly_results)
    print(yearly_df.round(2).to_string(index=False))

    # 2. 기간별 성과 비교
    print("\n\n2. 기간별 평균 성과 비교:")
    print("-" * 50)

    horizon_summary = []

    for horizon in horizons:
        horizon_data = df[df['horizon_days'] == horizon]

        summary = {
            'horizon_days': horizon,
            'kospi200_avg': horizon_data['kospi200_pr_mret_pct'].mean(),
            'kospi200_std': horizon_data['kospi200_pr_mret_pct'].std(),
            'short_avg': horizon_data['short_mret_pct'].mean(),
            'short_std': horizon_data['short_mret_pct'].std(),
            'long_avg': horizon_data['long_mret_pct'].mean(),
            'long_std': horizon_data['long_mret_pct'].std(),
            'mix_avg': horizon_data['mix_mret_pct'].mean(),
            'mix_std': horizon_data['mix_mret_pct'].std(),
            'mix_sharpe': horizon_data['mix_mret_pct'].mean() / horizon_data['mix_mret_pct'].std() if horizon_data['mix_mret_pct'].std() > 0 else 0
        }
        horizon_summary.append(summary)

    summary_df = pd.DataFrame(horizon_summary)
    print(summary_df.round(4).to_string(index=False))

    # 3. 월별 패턴 분석
    print("\n\n3. 월별 패턴 분석 (2023-2024 전체):")
    print("-" * 50)

    # 월별 평균 수익률
    df['month_only'] = df['month'].str[5:7]  # MM 형식
    monthly_pattern = df.groupby('month_only').agg({
        'kospi200_pr_mret_pct': 'mean',
        'short_mret_pct': 'mean',
        'long_mret_pct': 'mean',
        'mix_mret_pct': 'mean'
    }).round(3)

    print("월별 평균 수익률:")
    print(monthly_pattern.to_string())

    # 4. 최고/최저 수익률 분석
    print("\n\n4. 최고/최저 수익률 월 분석:")
    print("-" * 50)

    best_months = {}
    worst_months = {}

    for strategy in ['kospi200_pr_mret_pct', 'short_mret_pct', 'long_mret_pct', 'mix_mret_pct']:
        strategy_name = strategy.replace('_mret_pct', '')

        # 최고 수익률 월
        best_idx = df[strategy].idxmax()
        best_row = df.loc[best_idx]
        best_months[strategy_name] = {
            'month': best_row['month'],
            'return': best_row[strategy],
            'horizon': best_row['horizon_days']
        }

        # 최저 수익률 월
        worst_idx = df[strategy].idxmin()
        worst_row = df.loc[worst_idx]
        worst_months[strategy_name] = {
            'month': worst_row['month'],
            'return': worst_row[strategy],
            'horizon': worst_row['horizon_days']
        }

    print("최고 수익률 월:")
    for strategy, data in best_months.items():
        print(f"  {strategy}: {data['month']} ({data['return']:.2f}%, {data['horizon']}일 기간)")

    print("\n최저 수익률 월:")
    for strategy, data in worst_months.items():
        print(f"  {strategy}: {data['month']} ({data['return']:.2f}%, {data['horizon']}일 기간)")

    # 5. 전략별 상관관계 분석
    print("\n\n5. 전략별 상관관계 분석:")
    print("-" * 50)

    # 20일 기간 기준으로 상관관계 분석
    correlation_data = df[df['horizon_days'] == 20][[
        'kospi200_pr_mret_pct', 'short_mret_pct', 'long_mret_pct', 'mix_mret_pct'
    ]]

    correlation_matrix = correlation_data.corr().round(4)
    print("20일 기간 전략간 상관관계:")
    print(correlation_matrix)

    # 6. 승률 분석
    print("\n\n6. 승률 분석 (플러스 수익률 비율):")
    print("-" * 50)

    win_rates = {}
    for strategy in ['kospi200_pr_mret_pct', 'short_mret_pct', 'long_mret_pct', 'mix_mret_pct']:
        strategy_name = strategy.replace('_mret_pct', '')
        positive_returns = (df[strategy] > 0).sum()
        total_months = len(df[strategy])
        win_rate = positive_returns / total_months * 100
        win_rates[strategy_name] = win_rate

    for strategy, rate in win_rates.items():
        print(".1f")

    # 7. 연간 CAGR 계산
    print("\n\n7. 연간 CAGR 분석:")
    print("-" * 50)

    # 2023-2024 전체 기간 CAGR (24개월)
    total_months = 24

    for horizon in horizons:
        horizon_data = df[df['horizon_days'] == horizon]

        # 전체 기간 누적 수익률
        kospi_cum = (1 + horizon_data['kospi200_pr_mret_pct']/100).prod() - 1
        short_cum = (1 + horizon_data['short_mret_pct']/100).prod() - 1
        long_cum = (1 + horizon_data['long_mret_pct']/100).prod() - 1
        mix_cum = (1 + horizon_data['mix_mret_pct']/100).prod() - 1

        # CAGR 계산: (1 + 누적수익률)^(12/개월수) - 1
        kospi_cagr = (1 + kospi_cum) ** (12/total_months) - 1
        short_cagr = (1 + short_cum) ** (12/total_months) - 1
        long_cagr = (1 + long_cum) ** (12/total_months) - 1
        mix_cagr = (1 + mix_cum) ** (12/total_months) - 1

        print(f"{horizon}일 기간 CAGR (연율화):")
        print(f"  KOSPI200: {kospi_cagr*100:.2f}%")
        print(f"  Short: {short_cagr*100:.2f}%")
        print(f"  Long: {long_cagr*100:.2f}%")
        print(f"  Mix: {mix_cagr*100:.2f}%")
        print()

    # 분석 결과 저장
    output_path = Path("data/monthly_returns_analysis_report.csv")
    summary_df.to_csv(output_path, index=False)
    print(f"분석 결과가 {output_path}에 저장되었습니다.")

    return df, yearly_df, summary_df

if __name__ == "__main__":
    analyze_monthly_returns()
