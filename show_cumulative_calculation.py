#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
누적 수익률 계산 방법 상세 설명
"""

import pandas as pd
import numpy as np

def demonstrate_cumulative_calculation():
    """누적 수익률 계산 방법 단계별 설명"""

    print("=== 누적 수익률 계산 방법 상세 설명 ===\n")

    # 실제 데이터 로드
    df = pd.read_csv('data/ui_strategies_cumulative_comparison_updated.csv')
    print("1. 원본 데이터 (이미 누적 수익률 형태):")
    sample_data = df.head(5)[['month', 'bt20_short_20', 'bt120_long_120', 'kospi200']]
    print(sample_data.to_string())
    print()

    # 백분율을 소수점으로 변환
    numeric_cols = ['bt20_short_20', 'bt120_long_120', 'kospi200']
    df_calc = df.copy()
    df_calc[numeric_cols] = df_calc[numeric_cols] / 100.0

    print("2. 백분율 → 소수점 변환:")
    print(df_calc.head(3)[['month'] + numeric_cols].to_string())
    print()

    # 월별 수익률 계산
    df_monthly = df_calc.copy()
    df_monthly.iloc[0, 1:] = df_calc.iloc[0, 1:]  # 첫 번째 행은 그대로

    print("3. 월별 수익률 계산:")
    print("   r_t = 누적_t - 누적_(t-1)")
    print()

    for i in range(1, len(df_calc)):
        current_month = df_calc.iloc[i]['month']
        prev_month = df_calc.iloc[i-1]['month']

        print(f"   {current_month} 월별 수익률 계산:")
        for col in numeric_cols:
            current_cum = df_calc.iloc[i][col]
            prev_cum = df_calc.iloc[i-1][col]
            monthly_return = current_cum - prev_cum

            df_monthly.iloc[i, df_monthly.columns.get_loc(col)] = monthly_return

            print(f"     {col}: {current_cum:.6f} - {prev_cum:.6f} = {monthly_return:.6f}")
        print()

    print("4. 월별 수익률 결과:")
    print(df_monthly.head(5)[['month'] + numeric_cols].to_string())
    print()

    # 누적 수익률 재계산
    df_corrected = df_monthly.copy()

    print("5. 누적 수익률 재계산:")
    print("   누적_t = Π(1 + r_1) × (1 + r_2) × ... × (1 + r_t) - 1")
    print()

    for col in numeric_cols:
        monthly_returns = df_monthly[col].values
        cumulative_returns = np.cumprod(1 + monthly_returns) - 1
        df_corrected[col] = cumulative_returns

        print(f"   {col} 누적 계산 과정:")
        for i in range(min(5, len(monthly_returns))):
            if i == 0:
                cum_prod = 1 + monthly_returns[i]
            else:
                cum_prod *= (1 + monthly_returns[i])

            cum_return = cum_prod - 1
            print(f"     기간{i+1}: Π(1+r)={cum_prod:.6f}, 누적수익률={cum_return:.6f}")
        print()

    # 백분율로 변환
    df_corrected[numeric_cols] = df_corrected[numeric_cols] * 100.0

    print("6. 최종 정정된 누적 수익률 (백분율):")
    print(df_corrected.head(5)[['month'] + numeric_cols].to_string())
    print()

    # 수학적 정확성 검증
    print("7. 수학적 정확성 검증:")
    print("   월별 수익률들의 합 = 최종 누적 수익률")

    for col in numeric_cols:
        monthly_sum = (df_monthly[col] * 100).sum()
        final_cumulative = df_corrected[col].iloc[-1]
        print(f"   {col}: 월별합={monthly_sum:.1f}%, 최종누적={final_cumulative:.1f}%")
    print()

    # 저장
    output_file = 'data/ui_strategies_cumulative_comparison_corrected.csv'
    df_corrected.to_csv(output_file, index=False)
    print(f"✅ 정정된 누적 수익률 데이터가 '{output_file}'로 저장되었습니다.")

    return df_corrected

def show_mathematical_basis():
    """수학적 근거 설명"""

    print("\n=== 수학적 근거 ===")
    print()
    print("복리 계산의 기본 원리:")
    print("• 단리: 최종금액 = 원금 × (1 + r × t)")
    print("• 복리: 최종금액 = 원금 × (1 + r)^t")
    print("• 월별 복리: 최종금액 = 원금 × Π(1 + r_i) for i=1 to t")
    print()
    print("누적 수익률 계산:")
    print("• 수익률_t = (최종금액_t / 원금) - 1")
    print("• 누적수익률_t = Π(1 + r_1) × Π(1 + r_2) × ... × Π(1 + r_t) - 1")
    print()
    print("월별 수익률 변환:")
    print("• 관측된_누적_t = Σ(월별수익률_1 to t)")
    print("• 실제 월별수익률_t = 관측된_누적_t - 관측된_누적_(t-1)")
    print("• 정정된_누적_t = Π(1 + 실제월별수익률_1 to t) - 1")

if __name__ == "__main__":
    demonstrate_cumulative_calculation()
    show_mathematical_basis()