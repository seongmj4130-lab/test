"""
홀드아웃 기간 월별 초과수익률 데이터 산출
트렌치별(holding_days) 단기/장기/통합 전략의 KOSPI200 대비 초과수익률 계산
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd


def load_data():
    """필요한 데이터 파일 로드"""
    base_dir = Path(__file__).parent

    # 월별 누적 수익률 데이터
    monthly_returns_path = base_dir / "results" / "monthly_cumulative_returns_holDOUT.csv"
    monthly_df = pd.read_csv(monthly_returns_path)

    # KOSPI200 월별 누적 수익률 데이터
    kospi_path = base_dir / "data" / "ui_strategies_cumulative_comparison.csv"
    kospi_df = pd.read_csv(kospi_path)

    return monthly_df, kospi_df

def process_monthly_data(monthly_df, kospi_df):
    """월별 데이터를 처리하여 초과수익률 계산"""

    # KOSPI200 데이터를 월별 포맷으로 변환
    kospi_monthly = []
    for _, row in kospi_df.iterrows():
        month = row['month']
        kospi_value = row['kospi200']

        kospi_monthly.append({
            'month': month,
            'kospi200_cumulative_pct': kospi_value
        })

    kospi_df_processed = pd.DataFrame(kospi_monthly)

    # monthly_df의 month를 날짜 형식으로 변환
    def convert_month_to_date(month_num):
        if month_num <= 12:
            year = 2023
            month = month_num
        else:
            year = 2024
            month = month_num - 12
        return f"{year}-{month:02d}"

    monthly_df['month_date'] = monthly_df['month'].apply(convert_month_to_date)

    # 전략 데이터를 재구조화
    strategy_data = []

    # 고유한 전략 조합 추출
    unique_strategies = monthly_df[['strategy', 'holding_days']].drop_duplicates()

    for _, strategy_info in unique_strategies.iterrows():
        strategy_name = strategy_info['strategy']
        holding_days = strategy_info['holding_days']

        # 전략 유형 분류
        if 'short' in strategy_name:
            strategy_type = 'short'
        elif 'long' in strategy_name:
            strategy_type = 'long'
        elif 'ens' in strategy_name:
            strategy_type = 'ens'
        else:
            continue

        # 해당 전략의 월별 데이터 추출
        strategy_monthly = monthly_df[
            (monthly_df['strategy'] == strategy_name) &
            (monthly_df['holding_days'] == holding_days)
        ].copy()

        # KOSPI200 데이터와 병합
        merged_df = pd.merge(
            strategy_monthly,
            kospi_df_processed,
            left_on='month_date',
            right_on='month',
            how='left'
        )

        # 활성 수익률 및 초과수익률 계산
        # cumulative_return_pct는 백분율이므로 100으로 나누어 실제 수익률로 변환
        merged_df['strategy_cumulative_return'] = merged_df['cumulative_return_pct'] / 100
        merged_df['kospi_cumulative_return'] = merged_df['kospi200_cumulative_pct'] / 100

        # 활성 수익률 = 전략 수익률 / KOSPI 수익률
        merged_df['active_equity'] = (1 + merged_df['strategy_cumulative_return']) / (1 + merged_df['kospi_cumulative_return'])

        # 초과수익률 = 활성 수익률 - 1
        merged_df['excess_return'] = merged_df['active_equity'] - 1

        # 초과수익률 백분율
        merged_df['excess_return_pct'] = merged_df['excess_return'] * 100

        # 결과 저장
        for _, row in merged_df.iterrows():
            strategy_data.append({
                'holding_days': holding_days,
                'strategy_type': strategy_type,
                'month': row['month_date'],
                'month_num': row['month_x'],
                'strategy_cumulative_pct': row['cumulative_return_pct'],
                'kospi_cumulative_pct': row['kospi200_cumulative_pct'],
                'excess_return_pct': row['excess_return_pct']
            })

    return pd.DataFrame(strategy_data)

def calculate_kpi_metrics(monthly_df, kospi_df):
    """KPI 지표 계산 (Sharpe, MDD 등)"""

    # 성과 지표 데이터 로드
    perf_path = Path(__file__).parent / "results" / "performance_metrics_basic_holDOUT.csv"
    perf_df = pd.read_csv(perf_path)

    kpi_data = []

    for _, row in perf_df.iterrows():
        strategy_name = row['strategy']
        holding_days = row['holding_days']

        # 전략 유형 분류
        if 'short' in strategy_name:
            strategy_type = 'short'
        elif 'long' in strategy_name:
            strategy_type = 'long'
        elif 'ens' in strategy_name:
            strategy_type = 'ens'
        else:
            continue

        # 월별 초과수익률 데이터에서 최종값 추출
        final_excess = monthly_df[
            (monthly_df['holding_days'] == holding_days) &
            (monthly_df['strategy_type'] == strategy_type)
        ]['excess_return_pct'].iloc[-1] if len(monthly_df[
            (monthly_df['holding_days'] == holding_days) &
            (monthly_df['strategy_type'] == strategy_type)
        ]) > 0 else 0

        kpi_data.append({
            'holding_days': holding_days,
            'strategy_type': strategy_type,
            'final_excess_pct': final_excess,
            'sharpe': row['sharpe'],
            'mdd_pct': row['mdd_pct'] * 100,  # 백분율로 변환
            'total_return_pct': row['total_return_pct'],
            'cagr_pct': row['cagr_pct'],
            'hit_ratio_pct': row['hit_ratio_pct'],
            'profit_factor': row['profit_factor'],
            'avg_turnover': row['avg_turnover']
        })

    return pd.DataFrame(kpi_data)

def create_ui_format_data(monthly_excess_df, kpi_df):
    """UI용 데이터 포맷 생성"""

    ui_data = {}

    # 각 트렌치별로 데이터 구성
    for holding_days in [20, 40, 60, 80, 100, 120]:
        horizon_key = f"horizon_{holding_days}"

        # 월별 시계열 데이터
        series_data = []
        trench_data = monthly_excess_df[monthly_excess_df['holding_days'] == holding_days]

        for _, row in trench_data.iterrows():
            series_data.append({
                "date": row['month'],
                "short_excess_pct": row['excess_return_pct'] if row['strategy_type'] == 'short' else None,
                "long_excess_pct": row['excess_return_pct'] if row['strategy_type'] == 'long' else None,
                "mix_excess_pct": row['excess_return_pct'] if row['strategy_type'] == 'ens' else None
            })

        # 중복 제거 및 병합
        series_df = pd.DataFrame(series_data)
        series_df = series_df.groupby('date').first().reset_index()

        # KPI 데이터
        kpi_data = {}
        trench_kpi = kpi_df[kpi_df['holding_days'] == holding_days]

        for _, row in trench_kpi.iterrows():
            strategy_type = row['strategy_type']
            kpi_data[strategy_type] = {
                "final_excess_pct": round(row['final_excess_pct'], 2),
                "sharpe": round(row['sharpe'], 3),
                "mdd_pct": round(row['mdd_pct'], 2),
                "total_return_pct": round(row['total_return_pct'], 2),
                "cagr_pct": round(row['cagr_pct'], 2),
                "hit_ratio_pct": round(row['hit_ratio_pct'], 1),
                "profit_factor": round(row['profit_factor'], 3),
                "avg_turnover": round(row['avg_turnover'], 3)
            }

        ui_data[horizon_key] = {
            "horizon_days": holding_days,
            "benchmark": "KOSPI200_PR",
            "series": series_df.to_dict('records'),
            "kpi": kpi_data
        }

    return ui_data

def main():
    """메인 실행 함수"""

    print("홀드아웃 기간 월별 초과수익률 데이터 산출을 시작합니다...")

    # 데이터 로드
    monthly_df, kospi_df = load_data()
    print(f"월별 수익률 데이터: {len(monthly_df)} 행")
    print(f"KOSPI200 데이터: {len(kospi_df)} 행")

    # 월별 초과수익률 계산
    excess_df = process_monthly_data(monthly_df, kospi_df)
    print(f"초과수익률 데이터: {len(excess_df)} 행")

    # KPI 지표 계산
    kpi_df = calculate_kpi_metrics(excess_df, kospi_df)
    print(f"KPI 데이터: {len(kpi_df)} 행")

    # UI 포맷 데이터 생성
    ui_data = create_ui_format_data(excess_df, kpi_df)

    # 결과 저장
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    # 월별 초과수익률 CSV 저장
    excess_output_path = output_dir / "monthly_excess_returns_holdout.csv"
    excess_df.to_csv(excess_output_path, index=False)
    print(f"월별 초과수익률 데이터 저장: {excess_output_path}")

    # KPI CSV 저장
    kpi_output_path = output_dir / "kpi_metrics_holdout.csv"
    kpi_df.to_csv(kpi_output_path, index=False)
    print(f"KPI 지표 데이터 저장: {kpi_output_path}")

    # UI JSON 데이터 저장
    import json
    ui_output_path = output_dir / "ui_excess_returns_data.json"
    with open(ui_output_path, 'w', encoding='utf-8') as f:
        json.dump(ui_data, f, indent=2, ensure_ascii=False)
    print(f"UI용 JSON 데이터 저장: {ui_output_path}")

    # 결과 요약 출력
    print("\n=== 홀드아웃 기간 초과수익률 분석 결과 ===")

    for holding_days in [20, 40, 60, 80, 100, 120]:
        print(f"\n트렌치 {holding_days}일:")
        trench_data = kpi_df[kpi_df['holding_days'] == holding_days]

        for _, row in trench_data.iterrows():
            strategy_name = "단기" if row['strategy_type'] == 'short' else "장기" if row['strategy_type'] == 'long' else "통합"
            print(f"  {strategy_name}: 초과수익률 {row['final_excess_pct']:.2f}%, Sharpe {row['sharpe']:.3f}, MDD {row['mdd_pct']:.2f}%")

    print("\n분석 완료!")

if __name__ == "__main__":
    main()
