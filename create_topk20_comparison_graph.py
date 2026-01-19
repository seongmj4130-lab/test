import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 스타일 설정
plt.style.use('default')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.family'] = 'Malgun Gothic' if os.name == 'nt' else 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

def load_backtest_returns():
    """4가지 전략의 백테스트 수익률 데이터 로드"""

    strategies = {
        'bt20_ens': 'BT20 앙상블',
        'bt20_short': 'BT20 단기',
        'bt120_ens': 'BT120 앙상블',
        'bt120_long': 'BT120 장기'
    }

    returns_data = {}

    print("📊 백테스트 수익률 데이터 로드 중...")

    for strategy_code, strategy_name in strategies.items():
        file_path = f'C:\\Users\\seong\\OneDrive\\Desktop\\bootcamp\\03_code\\data\\interim\\bt_returns_{strategy_code}.csv'

        try:
            df = pd.read_csv(file_path)
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')

            # holdout 기간 데이터만 선택
            df_holdout = df[df['phase'] == 'holdout'].copy()

            if len(df_holdout) > 0:
                # 누적 수익률 계산
                df_holdout['cumulative_return'] = (1 + df_holdout['net_return']).cumprod() - 1

                returns_data[strategy_name] = df_holdout[['date', 'net_return', 'cumulative_return']]

                print(f"✅ {strategy_name}: {len(df_holdout)}개 데이터 포인트")

        except FileNotFoundError:
            print(f"❌ {strategy_name}: 파일을 찾을 수 없음 ({file_path})")

    return returns_data

def create_cumulative_returns_comparison(returns_data, output_path):
    """전략별 누적 수익률 비교 그래프 생성 (top_k=20 고정)"""

    plt.figure(figsize=(14, 8))

    # 색상 설정
    colors = {
        'BT20 단기': '#FF6B6B',     # Red
        'BT20 앙상블': '#4ECDC4',   # Teal
        'BT120 장기': '#96CEB4',    # Mint Green
        'BT120 앙상블': '#FECA57'   # Yellow
    }

    # 각 전략의 누적 수익률 그래프
    for strategy_name, df in returns_data.items():
        plt.plot(df['date'], df['cumulative_return'],
                 label=strategy_name, color=colors[strategy_name],
                 linewidth=2.5, alpha=0.9)

    # 0선 추가
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.8, linewidth=1)

    # 그래프 설정
    plt.title('전략별 누적 수익률 비교 (top_k=20 고정)', fontsize=16, fontweight='bold')
    plt.ylabel('누적 수익률', fontsize=12)
    plt.xlabel('기간', fontsize=12)

    # X축 날짜 포맷팅
    plt.xticks(rotation=45, ha='right')
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))

    plt.legend(loc='upper left', fontsize=10, frameon=True, framealpha=0.7)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    # 저장
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ 누적 수익률 비교 그래프 생성: {output_path}")

def calculate_performance_metrics(returns_data):
    """성과 지표 계산"""

    print("\n📊 전략별 성과 지표 (top_k=20 고정)")
    print("=" * 80)

    metrics_data = []

    for strategy_name, df in returns_data.items():
        returns = df['net_return']

        # 기본 지표 계산
        total_return = df['cumulative_return'].iloc[-1]
        annual_return = total_return / (len(df) / 252)  # 연환산

        # Sharpe 비율 (무위험 수익률 0% 가정)
        sharpe = returns.mean() / returns.std() * np.sqrt(252)

        # MDD 계산
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        mdd = drawdown.min()

        # CAGR
        years = len(df) / 252
        cagr = (1 + total_return) ** (1/years) - 1

        # Calmar 비율
        calmar = cagr / abs(mdd) if mdd != 0 else 0

        metrics_data.append({
            '전략': strategy_name,
            '총수익률': total_return,
            '연환산수익률': annual_return,
            'Sharpe': sharpe,
            'CAGR': cagr,
            'MDD': mdd,
            'Calmar': calmar,
            '데이터포인트': len(df)
        })

    # DataFrame 생성 및 출력
    metrics_df = pd.DataFrame(metrics_data)

    for _, row in metrics_df.iterrows():
        print(f"\n🏆 {row['전략']}")
        print(".2%")
        print(".2%")
        print(".3f")
        print(".2%")
        print(".2%")
        print(".3f")
        print(f"   • 데이터 포인트: {int(row['데이터포인트'])}개")

    # CSV로 저장
    metrics_df.to_csv('results/topk20_performance_metrics.csv', index=False, encoding='utf-8-sig')
    print(f"\n✅ 성과 지표 CSV 저장: results/topk20_performance_metrics.csv")

    return metrics_df

def compare_with_previous_results():
    """이전 결과와 비교"""

    print("\n🔄 top_k 변경 전후 비교")
    print("=" * 60)

    try:
        # 기존 결과 로드 (예: bt_metrics 파일에서)
        old_results = pd.read_csv('C:\\Users\\seong\\OneDrive\\Desktop\\bootcamp\\03_code\\artifacts\\reports\\backtest_4models_comparison.csv')

        print("기존 결과 (top_k=20 앙상블, 15 기타):")
        for _, row in old_results.iterrows():
            strategy_name = row['strategy'].replace('bt20_ens', 'BT20 앙상블').replace('bt20_short', 'BT20 단기').replace('bt120_ens', 'BT120 앙상블').replace('bt120_long', 'BT120 장기')
            print(".3f")

        print("\n📋 변경사항:")
        print("• BT20 앙상블: top_k 15 → 20")
        print("• BT20 단기: top_k 12 → 20")
        print("• BT120 장기: top_k 15 → 20")
        print("• BT120 앙상블: top_k 20 (유지)")

    except FileNotFoundError:
        print("기존 결과 파일을 찾을 수 없습니다.")

def main():
    """메인 실행 함수"""

    print("🎯 top_k=20 고정 전략 비교 분석 시작")
    print("=" * 50)

    # 백테스트 수익률 데이터 로드
    returns_data = load_backtest_returns()

    if not returns_data:
        print("❌ 백테스트 데이터를 찾을 수 없습니다.")
        return

    # 성과 지표 계산
    metrics_df = calculate_performance_metrics(returns_data)

    # 누적 수익률 비교 그래프 생성
    output_path = 'results/topk20_cumulative_returns_comparison.png'
    create_cumulative_returns_comparison(returns_data, output_path)

    # 이전 결과와 비교
    compare_with_previous_results()

    print("\n🎉 분석 완료!")
    print(f"   • 메인 그래프: {output_path}")
    print("   • 성과 지표: results/topk20_performance_metrics.csv")

if __name__ == "__main__":
    main()
