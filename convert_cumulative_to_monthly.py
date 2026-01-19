#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
누적 수익률 데이터를 월별 수익률로 변환하고 다시 누적 계산
"""

import numpy as np
import pandas as pd


def convert_cumulative_to_monthly_returns():
    """누적 수익률을 월별 수익률로 변환하고 다시 누적 계산"""

    # 데이터 로드
    df = pd.read_csv('data/ui_strategies_cumulative_comparison_updated.csv')
    print("=== 누적 수익률 → 월별 수익률 변환 ===")
    print(f"원본 데이터: {len(df)}개월, {len(df.columns)-1}개 전략")

    # 백분율을 소수점으로 변환 (현재 데이터가 % 단위임)
    numeric_cols = df.columns[1:]  # month 컬럼 제외
    df[numeric_cols] = df[numeric_cols] / 100.0  # % → 소수점 변환

    print("\n원본 데이터 샘플 (소수점 변환 후):")
    print(df.head(3).to_string())

    # 월별 수익률 계산
    df_monthly = df.copy()
    df_monthly.iloc[0, 1:] = df.iloc[0, 1:]  # 첫 번째 행은 그대로 (월별 = 누적)

    for i in range(1, len(df)):
        # 월별 수익률 = 현재 누적 - 이전 누적
        df_monthly.iloc[i, 1:] = df.iloc[i, 1:] - df.iloc[i-1, 1:]

    print("\n월별 수익률 계산 결과:")
    print(df_monthly.head(5).to_string())

    # 다시 누적 수익률 계산: cumprod(1 + r_t) - 1
    df_corrected_cumulative = df_monthly.copy()

    for col in numeric_cols:
        monthly_returns = df_monthly[col].values
        # cumprod(1 + r) - 1로 누적 계산
        cumulative = np.cumprod(1 + monthly_returns) - 1
        df_corrected_cumulative[col] = cumulative

    print("\n정정된 누적 수익률 (월별 수익률 기반):")
    print(df_corrected_cumulative.head(5).to_string())

    # 백분율로 다시 변환
    df_corrected_cumulative[numeric_cols] = df_corrected_cumulative[numeric_cols] * 100.0

    print("\n최종 결과 (백분율):")
    print(df_corrected_cumulative.head(5).to_string())

    # 파일 저장
    output_file = 'data/ui_strategies_cumulative_comparison_corrected.csv'
    df_corrected_cumulative.to_csv(output_file, index=False)
    print(f"\n✅ 정정된 데이터가 '{output_file}'로 저장되었습니다.")

    # 검증: 원본 vs 정정본 비교
    print("\n=== 검증: 원본 vs 정정본 비교 ===")
    original_last = df.iloc[-1, 1:] * 100  # 마지막 행, 백분율
    corrected_last = df_corrected_cumulative.iloc[-1, 1:]

    comparison = pd.DataFrame({
        '전략': numeric_cols,
        '원본_누적': original_last.values,
        '정정본_누적': corrected_last.values,
        '차이': corrected_last.values - original_last.values
    })

    print("최종 누적 수익률 비교:")
    print(comparison.to_string())

    return df_corrected_cumulative

if __name__ == "__main__":
    convert_cumulative_to_monthly_returns()
