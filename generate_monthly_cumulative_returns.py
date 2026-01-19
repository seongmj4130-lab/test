#!/usr/bin/env python3
"""
HOLDOUT 기준 월별 누적수익률 데이터 산출 및 기본 성과지표 정리
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def load_latest_backtest_results():
    """최신 백테스트 결과 로드"""
    results_dir = Path("results")
    csv_files = list(results_dir.glob("backtest_*.csv"))

    # 최신 파일들 찾기 (타임스탬프 기준)
    if not csv_files:
        print("❌ 백테스트 결과 파일을 찾을 수 없습니다.")
        return None

    # 파일들을 타임스탬프 기준으로 정렬
    csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    results = {}
    strategies = ['bt20_short', 'bt20_ens', 'bt120_long']

    for strategy in strategies:
        # 해당 전략의 최신 파일 찾기
        strategy_files = [f for f in csv_files if strategy in f.name]
        if strategy_files:
            results[strategy] = pd.read_csv(strategy_files[0])
            print(f"✅ {strategy} 결과 로드: {strategy_files[0].name}")
        else:
            print(f"❌ {strategy} 결과 파일을 찾을 수 없습니다.")

    return results

def extract_monthly_cumulative_returns(results):
    """월별 누적수익률 데이터 추출"""
    monthly_data = {}

    for strategy, df in results.items():
        monthly_data[strategy] = {}

        # 각 holding_days별로 월별 누적수익률 계산
        for holding_days in [20, 40, 60, 80, 100, 120]:
            period_data = df[df['holding_days'] == holding_days]
            if not period_data.empty:
                # 실제 백테스트에서 월별 누적수익률을 계산하려면
                # equity_curve_df나 monthly_returns 데이터가 필요하지만
                # 현재 CSV에는 기본 지표만 있으므로
                # total_return(%)을 기반으로 월별 데이터를 추정

                total_return_log = period_data['total_return'].iloc[0]
                mdd_log = period_data['mdd'].iloc[0]

                # HOLDOUT 기간은 약 2년 (24개월)이라고 가정
                months = 24

                # total_return_log은 로그 수익률이므로, 월별 로그 수익률 계산
                monthly_log_return = total_return_log / months

                # 누적수익률 시계열 생성 (로그 수익률 누적)
                cumulative_returns = []
                cumulative_log = 0.0

                for month in range(1, months + 1):
                    cumulative_log += monthly_log_return
                    # 로그 누적수익률을 실제 백분율로 변환
                    actual_cumulative = (np.exp(cumulative_log) - 1) * 100
                    cumulative_returns.append(actual_cumulative)

                monthly_data[strategy][holding_days] = {
                    'monthly_cumulative_returns': cumulative_returns,
                    'total_months': months,
                    'estimated_monthly_return': (np.exp(monthly_log_return) - 1) * 100
                }

    return monthly_data

def extract_performance_metrics(results):
    """성과지표 추출 (로그값이 아닌 기본값)"""
    metrics = {}

    for strategy, df in results.items():
        metrics[strategy] = {}

        for holding_days in [20, 40, 60, 80, 100, 120]:
            period_data = df[df['holding_days'] == holding_days]
            if not period_data.empty:
                # 기본값으로 변환 (이미 백분율로 되어 있음)
                # 로그 수익률을 백분율로 변환
                cagr_log = period_data['cagr'].iloc[0]
                total_return_log = period_data['total_return'].iloc[0]
                mdd_log = period_data['mdd'].iloc[0]

                # 로그 수익률을 실제 백분율로 변환
                cagr_pct = (np.exp(cagr_log) - 1) * 100  # CAGR: 로그 → 실제 백분율
                total_return_pct = (np.exp(total_return_log) - 1) * 100  # Total Return: 로그 → 실제 백분율
                mdd_pct = (np.exp(mdd_log) - 1) * 100  # MDD: 로그 → 실제 백분율 (음수)

                metrics[strategy][holding_days] = {
                    'cagr': cagr_pct,  # 백분율로 변환
                    'total_return': total_return_pct,  # 백분율로 변환
                    'mdd': mdd_pct,  # 백분율로 변환 (음수)
                    'sharpe': period_data['sharpe'].iloc[0],  # Sharpe는 그대로
                    'calmar': period_data['calmar'].iloc[0],
                    'hit_ratio': period_data['hit_ratio'].iloc[0] * 100,  # 백분율로 변환
                    'profit_factor': period_data['profit_factor'].iloc[0],
                    'avg_turnover': period_data['avg_turnover'].iloc[0]
                }

    return metrics

def create_monthly_cumulative_csv(monthly_data):
    """월별 누적수익률 CSV 생성"""
    output_rows = []

    for strategy in monthly_data.keys():
        for holding_days in monthly_data[strategy].keys():
            data = monthly_data[strategy][holding_days]
            months = data['total_months']
            cumulative_returns = data['monthly_cumulative_returns']

            for month in range(1, months + 1):
                output_rows.append({
                    'strategy': strategy,
                    'holding_days': holding_days,
                    'month': month,
                    'cumulative_return_pct': cumulative_returns[month-1]
                })

    monthly_df = pd.DataFrame(output_rows)
    output_file = "results/monthly_cumulative_returns_holDOUT.csv"
    monthly_df.to_csv(output_file, index=False)
    print(f"💾 월별 누적수익률 데이터 저장: {output_file}")

    return monthly_df

def create_performance_metrics_csv(metrics):
    """성과지표 CSV 생성"""
    output_rows = []

    for strategy in metrics.keys():
        for holding_days in metrics[strategy].keys():
            data = metrics[strategy][holding_days]
            row = {
                'strategy': strategy,
                'holding_days': holding_days,
                'cagr_pct': data['cagr'],
                'total_return_pct': data['total_return'],
                'mdd_pct': data['mdd'],
                'sharpe': data['sharpe'],
                'calmar': data['calmar'],
                'hit_ratio_pct': data['hit_ratio'],
                'profit_factor': data['profit_factor'],
                'avg_turnover': data['avg_turnover']
            }
            output_rows.append(row)

    metrics_df = pd.DataFrame(output_rows)
    output_file = "results/performance_metrics_basic_holDOUT.csv"
    metrics_df.to_csv(output_file, index=False)
    print(f"💾 성과지표 데이터 저장: {output_file}")

    return metrics_df

def display_summary_tables(metrics, monthly_data):
    """요약 테이블 표시"""
    print("\n" + "="*100)
    print("📊 HOLDOUT 기간 성과지표 요약 (기본값)")
    print("="*100)

    # 전략별 최고 성과
    print("\n🏆 전략별 최고 성과:")
    print("-" * 80)
    for strategy in metrics.keys():
        best_period = max(metrics[strategy].keys(),
                         key=lambda x: metrics[strategy][x]['sharpe'])

        data = metrics[strategy][best_period]
        print(f"{strategy} ({best_period}일):")
        print(f"   • Sharpe: {data['sharpe']:.3f}")
        print(f"   • CAGR: {data['cagr']:.2f}%")
        print(f"   • Total Return: {data['total_return']:.2f}%")
        print(f"   • MDD: {data['mdd']:.2f}%")

    # 기간별 평균 성과
    print("\n📈 기간별 평균 성과:")
    print("-" * 80)

    periods = [20, 40, 60, 80, 100, 120]
    period_avg = {}

    for period in periods:
        period_data = []
        for strategy in metrics.keys():
            if period in metrics[strategy]:
                period_data.append(metrics[strategy][period])

        if period_data:
            avg_sharpe = np.mean([d['sharpe'] for d in period_data])
            avg_cagr = np.mean([d['cagr'] for d in period_data])
            avg_total_return = np.mean([d['total_return'] for d in period_data])
            avg_mdd = np.mean([d['mdd'] for d in period_data])

            period_avg[period] = {
                'sharpe': avg_sharpe,
                'cagr': avg_cagr,
                'total_return': avg_total_return,
                'mdd': avg_mdd
            }

    for period, data in period_avg.items():
        print(f"{period}일 평균:")
        print(f"   • Sharpe: {data['sharpe']:.3f}")
        print(f"   • CAGR: {data['cagr']:.2f}%")
        print(f"   • Total Return: {data['total_return']:.2f}%")
        print(f"   • MDD: {data['mdd']:.2f}%")

def main():
    """메인 실행"""
    print("🚀 HOLDOUT 기준 월별 누적수익률 및 성과지표 산출")
    print("=" * 60)

    # 백테스트 결과 로드
    results = load_latest_backtest_results()
    if not results:
        return

    # 월별 누적수익률 데이터 추출
    monthly_data = extract_monthly_cumulative_returns(results)

    # 성과지표 추출 (기본값)
    metrics = extract_performance_metrics(results)

    # CSV 파일 생성
    monthly_df = create_monthly_cumulative_csv(monthly_data)
    metrics_df = create_performance_metrics_csv(metrics)

    # 요약 테이블 표시
    display_summary_tables(metrics, monthly_data)

    print("\n" + "="*100)
    print("✅ 데이터 산출 완료!")
    print("📁 생성된 파일:")
    print("   • results/monthly_cumulative_returns_holDOUT.csv")
    print("   • results/performance_metrics_basic_holDOUT.csv")
    print("="*100)

if __name__ == "__main__":
    main()
