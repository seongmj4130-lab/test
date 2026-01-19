from pathlib import Path

import numpy as np
import pandas as pd


def recalculate_monthly_returns():
    """월별 수익률을 누적 수익률에서 재계산"""

    # 데이터 파일 경로
    data_path = Path("data/dummy_kospi200_pr_tabs_4lines_2023_2024_v5.csv")

    # 데이터 읽기
    df = pd.read_csv(data_path)

    print("원본 데이터 구조:")
    print(df.head())
    print(f"\n총 행 수: {len(df)}")

    # horizon_days별로 그룹화
    horizons = df['horizon_days'].unique()
    print(f"\nHorizon days: {sorted(horizons)}")

    results = []

    for horizon in sorted(horizons):
        print(f"\n=== {horizon}일 기간 분석 ===")

        # 해당 horizon의 데이터만 선택
        horizon_data = df[df['horizon_days'] == horizon].copy()
        horizon_data = horizon_data.sort_values('month')

        print(f"데이터 행 수: {len(horizon_data)}")

        # 월별 수익률 재계산을 위한 준비
        strategies = ['kospi200_pr', 'short', 'long', 'mix']

        # 이전 달의 누적 수익률을 저장할 변수들
        prev_cum_returns = {}

        for idx, row in horizon_data.iterrows():
            month = row['month']

            for strategy in strategies:
                cum_col = f'{strategy}_cum_return_pct'
                mret_col = f'{strategy}_mret_pct'

                current_cum = row[cum_col]

                if month == '2023-01':
                    # 첫 달은 월별 수익률 = 누적 수익률
                    calculated_mret = current_cum
                else:
                    # 월별 수익률 = (1 + 현재 누적) / (1 + 이전 누적) - 1
                    prev_cum = prev_cum_returns[strategy]
                    calculated_mret = (1 + current_cum/100) / (1 + prev_cum/100) - 1
                    calculated_mret *= 100

                # 결과 저장
                prev_cum_returns[strategy] = current_cum

                # 비교를 위해 원본과 계산값 저장
                results.append({
                    'month': month,
                    'horizon_days': horizon,
                    'strategy': strategy,
                    'original_mret': row[mret_col],
                    'calculated_mret': calculated_mret,
                    'cum_return': current_cum,
                    'difference': row[mret_col] - calculated_mret
                })

    # 결과를 데이터프레임으로 변환
    results_df = pd.DataFrame(results)

    # 차이가 큰 경우 확인
    significant_diff = results_df[abs(results_df['difference']) > 0.01]
    if len(significant_diff) > 0:
        print(f"\n차이가 0.01% 이상인 경우: {len(significant_diff)}개")
        print(significant_diff.head(10))
    else:
        print("\n모든 월별 수익률이 누적 수익률과 일치합니다.")

    # 각 전략별로 월별 수익률 계산
    print("\n=== 월별 수익률 요약 ===")

    for horizon in sorted(horizons):
        print(f"\n{horizon}일 기간:")
        horizon_results = results_df[results_df['horizon_days'] == horizon]

        for strategy in strategies:
            strategy_data = horizon_results[horizon_results['strategy'] == strategy]
            print(f"  {strategy}: 평균 {strategy_data['calculated_mret'].mean():.2f}%, "
                  f"표준편차 {strategy_data['calculated_mret'].std():.2f}%")

    # 월별 수익률만 포함한 새로운 데이터프레임 생성
    monthly_returns_only = []

    for horizon in sorted(horizons):
        horizon_data = df[df['horizon_days'] == horizon].copy()
        horizon_data = horizon_data.sort_values('month')

        for _, row in horizon_data.iterrows():
            monthly_row = {
                'month': row['month'],
                'horizon_days': row['horizon_days'],
                'kospi200_pr_mret_pct': row['kospi200_pr_mret_pct'],
                'short_mret_pct': row['short_mret_pct'],
                'long_mret_pct': row['long_mret_pct'],
                'mix_mret_pct': row['mix_mret_pct']
            }
            monthly_returns_only.append(monthly_row)

    monthly_df = pd.DataFrame(monthly_returns_only)

    # 월별 수익률만 저장
    output_path = Path("data/monthly_returns_recalculated.csv")
    monthly_df.to_csv(output_path, index=False)
    print(f"\n월별 수익률 데이터가 {output_path}에 저장되었습니다.")

    return monthly_df, results_df

if __name__ == "__main__":
    monthly_df, results_df = recalculate_monthly_returns()

    # 각 horizon별로 월별 수익률 표시
    print("\n=== 월별 수익률 표 ===")
    for horizon in sorted(monthly_df['horizon_days'].unique()):
        print(f"\n{horizon}일 기간 월별 수익률:")
        horizon_data = monthly_df[monthly_df['horizon_days'] == horizon].copy()
        horizon_data = horizon_data.round(4)
        print(horizon_data.to_string(index=False))
