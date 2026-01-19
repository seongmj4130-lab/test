import numpy as np
import pandas as pd


def create_baseline2_ui_data_final():
    """Baseline2 기준 KOSPI200 TR vs 4전략 비교 데이터 생성 (최종 버전)"""

    print("📊 Baseline2 UI 데이터 생성 (최종 버전)")
    print("=" * 60)

    # 기존 UI 데이터 로드
    try:
        existing_data = pd.read_csv('data/ui_monthly_log_returns_data.csv')
        print("✅ 기존 UI 데이터 로드됨")
        print(f"   데이터 크기: {len(existing_data)}행 × {len(existing_data.columns)}열")
    except Exception as e:
        print(f"❌ 기존 UI 데이터 로드 실패: {e}")
        return

    # 데이터 구조 확인 및 변환
    print("\n🔄 데이터 구조 변환")

    # 로그 수익률을 일반 수익률로 변환
    baseline2_data = existing_data.copy()

    # KOSPI TR 데이터 (이미 TR로 되어 있음)
    baseline2_data['kospi_tr_monthly_return'] = np.exp(baseline2_data['kospi_tr_monthly_log_return']) - 1
    baseline2_data['kospi_tr_cumulative_return'] = np.exp(baseline2_data['kospi_tr_cumulative_log_return']) - 1
    baseline2_data['kospi_tr_log_cumulative_return'] = baseline2_data['kospi_tr_cumulative_log_return']

    # 전략별 데이터 변환
    strategies = ['bt20_단기', 'bt20_앙상블', 'bt120_장기', 'bt120_앙상블']
    for strategy in strategies:
        monthly_log_col = f'{strategy}_monthly_log_return'
        cumulative_log_col = f'{strategy}_cumulative_log_return'

        if monthly_log_col in baseline2_data.columns:
            baseline2_data[f'{strategy}_monthly_return'] = np.exp(baseline2_data[monthly_log_col]) - 1
            baseline2_data[f'{strategy}_cumulative_return'] = np.exp(baseline2_data[cumulative_log_col]) - 1
            baseline2_data[f'{strategy}_log_cumulative_return'] = baseline2_data[cumulative_log_col]

    print("✅ 데이터 변환 완료")

    # 필요한 컬럼만 선택
    required_columns = [
        'year_month', 'date',
        'kospi_tr_monthly_return', 'kospi_tr_cumulative_return', 'kospi_tr_log_cumulative_return'
    ]

    for strategy in strategies:
        required_columns.extend([
            f'{strategy}_monthly_return',
            f'{strategy}_cumulative_return',
            f'{strategy}_log_cumulative_return'
        ])

    baseline2_data = baseline2_data[required_columns]
    print(f"✅ 최종 데이터: {len(baseline2_data)}행 × {len(baseline2_data.columns)}열")

    # 성과 지표 계산
    print("\n📊 성과 지표 계산")

    performance_metrics = {}

    # KOSPI200 TR 성과 계산
    kospi_returns = baseline2_data['kospi_tr_monthly_return'].values
    kospi_total_return = baseline2_data['kospi_tr_cumulative_return'].iloc[-1]
    kospi_cagr = (1 + kospi_total_return) ** (12 / len(baseline2_data)) - 1
    kospi_volatility = np.std(kospi_returns) * np.sqrt(12)
    kospi_sharpe = kospi_cagr / kospi_volatility if kospi_volatility != 0 else 0

    # MDD 계산 (로그 누적 수익률 기준)
    cumulative_returns = baseline2_data['kospi_tr_cumulative_return']
    kospi_mdd = np.min(cumulative_returns - np.maximum.accumulate(cumulative_returns))

    performance_metrics['KOSPI200 TR'] = {
        '총수익률': kospi_total_return,
        '연평균수익률': kospi_cagr,
        'MDD': kospi_mdd,
        'Sharpe': kospi_sharpe,
        'Hit_Ratio': None
    }

    # 전략별 성과 계산
    strategy_names = {
        'bt20_단기': 'BT20 단기',
        'bt20_앙상블': 'BT20 앙상블',
        'bt120_장기': 'BT120 장기',
        'bt120_앙상블': 'BT120 앙상블'
    }

    for strategy_key, strategy_name in strategy_names.items():
        monthly_col = f'{strategy_key}_monthly_return'
        cumulative_col = f'{strategy_key}_cumulative_return'

        if monthly_col in baseline2_data.columns:
            returns = baseline2_data[monthly_col].values
            total_return = baseline2_data[cumulative_col].iloc[-1]
            cagr = (1 + total_return) ** (12 / len(baseline2_data)) - 1
            volatility = np.std(returns) * np.sqrt(12)
            sharpe = cagr / volatility if volatility != 0 else 0

            # MDD 계산
            cumulative_returns = baseline2_data[cumulative_col]
            mdd = np.min(cumulative_returns - np.maximum.accumulate(cumulative_returns))

            # Hit Ratio
            hit_ratio = (returns > 0).mean()

            performance_metrics[strategy_name] = {
                '총수익률': total_return,
                '연평균수익률': cagr,
                'MDD': mdd,
                'Sharpe': sharpe,
                'Hit_Ratio': hit_ratio
            }

    # 데이터 저장
    print("\n💾 데이터 저장")

    # 월별 데이터 CSV
    monthly_csv_path = 'data/ui_baseline2_monthly_log_returns.csv'
    baseline2_data.to_csv(monthly_csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ 월별 데이터: {monthly_csv_path}")

    # 성과 지표 CSV
    metrics_df = pd.DataFrame.from_dict(performance_metrics, orient='index')
    metrics_csv_path = 'data/ui_baseline2_performance_metrics.csv'
    metrics_df.to_csv(metrics_csv_path, encoding='utf-8-sig')
    print(f"✅ 성과 지표: {metrics_csv_path}")

    # 결과 요약
    print("\n📋 결과 요약")
    print("-" * 50)

    print("월별 데이터 컬럼:")
    for col in baseline2_data.columns[:8]:  # 처음 8개만 표시
        print(f"  • {col}")
    if len(baseline2_data.columns) > 8:
        print(f"  • ... (+{len(baseline2_data.columns)-8}개 컬럼)")

    print("\n성과 지표:")
    for name, metrics in performance_metrics.items():
        print(f"  • {name}:")
        print(".2%")
        print(".3f")
        if metrics['Hit_Ratio'] is not None:
            print(".1%")

    print("\n🎯 Baseline2 UI 데이터 생성 완료!")
    print("   - KOSPI200 TR 로그 누적 수익률 그래프 생성 가능")
    print("   - 4개 전략 로그 누적 수익률 비교 그래프 생성 가능")
    print("   - UI 구현을 위한 월별 데이터 제공")

    # 샘플 데이터 표시
    print("\n📊 샘플 데이터 (첫 3개월):")
    print("-" * 40)
    sample_data = baseline2_data.head(3)[['year_month', 'kospi_tr_log_cumulative_return',
                                         'bt20_단기_log_cumulative_return', 'bt120_장기_log_cumulative_return']]
    print(sample_data.to_string(index=False))

if __name__ == "__main__":
    create_baseline2_ui_data_final()
