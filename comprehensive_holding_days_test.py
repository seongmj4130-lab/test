import yaml
import pandas as pd
from pathlib import Path
import subprocess
import sys
from datetime import datetime

def run_backtest_for_config(config_path, results_list):
    """특정 config로 백테스트 실행하고 결과 수집"""

    try:
        # 백테스트 실행
        result = subprocess.run([
            sys.executable, 'scripts/run_backtest_4models.py'
        ], capture_output=True, text=True, cwd=config_path.parent)

        if result.returncode == 0:
            # 결과 파싱 (마지막 비교 리포트에서 추출)
            lines = result.stdout.split('\n')
            start_parsing = False
            for line in lines:
                if '비교 리포트' in line:
                    start_parsing = True
                    continue
                if start_parsing and line.strip().startswith('strategy'):
                    # 헤더 라인
                    continue
                elif start_parsing and line.strip() and not line.startswith('['):
                    # 데이터 라인 파싱
                    parts = line.split()
                    if len(parts) >= 6:
                        try:
                            strategy = parts[0]
                            holding_days = int(parts[1])
                            sharpe = float(parts[2])
                            cagr = float(parts[3])
                            mdd = float(parts[4])
                            calmar = float(parts[5])

                            results_list.append({
                                'strategy': strategy,
                                'holding_days': holding_days,
                                'sharpe': sharpe,
                                'cagr': cagr,
                                'mdd': mdd,
                                'calmar': calmar,
                                'timestamp': datetime.now().isoformat()
                            })
                        except (ValueError, IndexError):
                            continue
                    break
            return True
        else:
            print(f"백테스트 실패: {result.stderr}")
            return False

    except Exception as e:
        print(f"백테스트 실행 중 오류: {e}")
        return False

def comprehensive_holding_days_test():
    """모든 전략에 대해 holding_days 20,40,60,80,100,120일 테스트"""

    print("🔬 포괄적 Holding Days 테스트 (모든 전략)")
    print("=" * 70)

    project_root = Path(__file__).resolve().parent
    config_path = project_root / 'configs' / 'config.yaml'

    # 테스트할 holding_days 값들
    holding_days_values = [20, 40, 60, 80, 100, 120]

    # 전략 설정 매핑
    strategies = {
        'l7_bt20_short': 'bt20_short',
        'l7_bt20_ens': 'bt20_ens',
        'l7_bt120_long': 'bt120_long',
        'l7_bt120_ens': 'bt120_ens'
    }

    results = []

    total_tests = len(holding_days_values) * len(strategies)
    test_count = 0

    print(f"📋 총 테스트 수: {total_tests}")
    print(f"   • Holding Days: {holding_days_values}")
    print(f"   • 전략 수: {len(strategies)}")
    print("-" * 70)

    # 각 holding_days에 대해 모든 전략 테스트
    for hd in holding_days_values:
        print(f"\n🏃 Holding Days = {hd} 테스트 시작")
        print("-" * 50)

        # config 파일 읽기
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 모든 전략의 holding_days 변경
        for strategy_key in strategies.keys():
            if strategy_key in config:
                config[strategy_key]['holding_days'] = hd
                print(f"   • {strategies[strategy_key]}: holding_days = {hd}")

        # config 파일 저장
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        print("   ✅ Config 업데이트 완료")

        # 백테스트 실행
        print("   🚀 백테스트 실행 중...")
        success = run_backtest_for_config(project_root, results)

        if success:
            print("   ✅ 백테스트 완료")
            test_count += len(strategies)
            print(f"   📊 진행률: {test_count}/{total_tests}")
        else:
            print("   ❌ 백테스트 실패")
            continue

    # 결과 정리
    print("\n📊 테스트 완료! 결과 정리 중...")
    print("=" * 70)

    if results:
        results_df = pd.DataFrame(results)

        # 전략명 한글 변환
        strategy_names = {
            'bt20_short': 'BT20 단기',
            'bt20_ens': 'BT20 앙상블',
            'bt120_long': 'BT120 장기',
            'bt120_ens': 'BT120 앙상블'
        }
        results_df['strategy_name'] = results_df['strategy'].map(strategy_names)

        print(f"✅ 수집된 결과 수: {len(results_df)}")

        # Sharpe Ratio 피벗 테이블
        print("\n📈 Sharpe Ratio 비교표:")
        sharpe_pivot = results_df.pivot_table(
            index='strategy_name',
            columns='holding_days',
            values='sharpe',
            aggfunc='first'
        ).round(3)

        print(sharpe_pivot)

        # CAGR 피벗 테이블
        print("💰 CAGR 비교표:")        cagr_pivot = results_df.pivot_table(
            index='strategy_name',
            columns='holding_days',
            values='cagr',
            aggfunc='first'
        ).round(4)

        print(cagr_pivot)

        # MDD 피벗 테이블
        print("📉 MDD 비교표:")        mdd_pivot = results_df.pivot_table(
            index='strategy_name',
            columns='holding_days',
            values='mdd',
            aggfunc='first'
        ).round(4)

        print(mdd_pivot)

        # 결과 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f'results/comprehensive_holding_days_test_{timestamp}.csv'
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n💾 결과 저장: {output_file}")

        # 분석 요약
        print("
🎯 분석 요약:"        print("-" * 50)

        # 각 전략별 최적 holding_days 찾기
        for strategy in results_df['strategy_name'].unique():
            strategy_data = results_df[results_df['strategy_name'] == strategy]
            best_sharpe = strategy_data.loc[strategy_data['sharpe'].idxmax()]

            print(f"• {strategy}:")
            print(".3f")
            print(".1%")
            print(".1%")

    else:
        print("❌ 수집된 결과가 없습니다.")

    # config를 원래 상태로 복원 (첫 번째 전략만 20일로)
    print("
🔄 Config 복원 중..."    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    config['l7_bt20_short']['holding_days'] = 20
    config['l7_bt20_ens']['holding_days'] = 20
    config['l7_bt120_long']['holding_days'] = 20
    config['l7_bt120_ens']['holding_days'] = 20

    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    print("✅ Config 복원 완료")

if __name__ == "__main__":
    comprehensive_holding_days_test()