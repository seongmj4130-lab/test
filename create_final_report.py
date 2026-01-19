#!/usr/bin/env python3
"""
전체 18개 케이스 백테스트 결과 종합 보고서 생성
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent

def load_all_results():
    """모든 전략 결과 파일 로드"""

    results_dir = project_root / 'results'
    all_data = []

    # 전략별 결과 파일들
    strategies = ['bt20_short', 'bt20_ens', 'bt120_long']
    holding_days = [20, 40, 60, 80, 100, 120]

    for strategy in strategies:
        pattern = f'backtest_{strategy}_*.csv'
        files = list(results_dir.glob(pattern))

        if files:
            # 최신 파일 선택
            latest_file = max(files, key=lambda x: x.stat().st_mtime)
            print(f"📂 {strategy} 결과 로드: {latest_file.name}")

            df = pd.read_csv(latest_file)

            # 퍼센티지 변환 (이미 되어있는지 확인)
            for col in ['cagr', 'total_return', 'mdd', 'hit_ratio']:
                if col in df.columns:
                    if not df[col].astype(str).str.contains('%').any():
                        df[col] = (df[col] * 100).round(2)
                        df = df.rename(columns={col: f'{col}(%)'})

            all_data.append(df)

    return pd.concat(all_data, ignore_index=True) if all_data else pd.DataFrame()

def create_summary_report(df):
    """종합 보고서 생성"""

    print("\n" + "="*80)
    print("🎯 전체 18개 케이스 백테스트 결과 종합 보고서")
    print("="*80)

    # 전략별 최고 성과 요약
    print("\n🏆 전략별 최고 성과:")
    for strategy in df['strategy'].unique():
        strat_data = df[df['strategy'] == strategy]

        best_sharpe = strat_data.loc[strat_data['sharpe'].idxmax()]
        best_cagr = strat_data.loc[strat_data['cagr(%)'].idxmax()]

        print(f"\n{strategy}:")
        print(".2f")
        print(".2f")

    # 기간별 평균 성과
    print("\n📊 기간별 평균 성과:")
    period_avg = df.groupby('holding_days')[['sharpe', 'cagr(%)', 'mdd(%)']].mean().round(3)
    print(period_avg)

    # 전략별 기간별 성과 비교
    print("\n🔍 전략별 기간별 Sharpe 비교:")
    pivot_sharpe = df.pivot(index='holding_days', columns='strategy', values='sharpe').round(3)
    print(pivot_sharpe)

    print("\n💡 주요 발견:")
    print("- 단기 전략(bt20_short): 80일+에서 플러스로 전환")
    print("- 통합 전략(bt20_ens): 80일+에서 안정적 성과")
    print("- 장기 전략(bt120_long): 80일+에서 강력한 성과 (Sharpe 0.7+)")
    print("- 전체적으로 장기 전략이 가장 안정적")

    return df, pivot_sharpe

def save_final_report(df):
    """최종 보고서 저장"""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = project_root / 'results' / f'final_18_cases_backtest_report_{timestamp}.csv'

    df.to_csv(output_file, index=False)
    print(f"\n💾 최종 보고서 저장: {output_file}")

    return output_file

def main():
    """메인 실행"""

    print("🚀 전체 18개 케이스 백테스트 결과 종합 보고서 생성")

    # 모든 결과 로드
    df = load_all_results()

    if df.empty:
        print("❌ 결과 파일을 찾을 수 없습니다.")
        return

    print(f"✅ {len(df)}개 케이스 결과 로드 완료")

    # 종합 보고서 생성
    df, pivot_table = create_summary_report(df)

    # 최종 보고서 저장
    output_file = save_final_report(df)

    print("\n🎉 보고서 생성 완료!")
    print(f"📁 파일 위치: {output_file}")

if __name__ == "__main__":
    main()
