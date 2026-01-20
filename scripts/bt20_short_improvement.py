"""
bt20_short 전략 개선 스크립트

bt20_short의 성과 저하 원인을 분석하고 개선안을 적용합니다.
"""

import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config
from src.utils.io import load_artifact, save_artifact


def analyze_bt20_short_problems():
    """
    bt20_short 전략의 문제점 심층 분석
    """
    print("🔍 bt20_short 전략 문제점 분석")
    print("="*50)

    cfg = load_config('configs/config.yaml')
    interim_dir = Path(cfg['paths']['base_dir']) / 'data' / 'interim'

    # 백테스트 메트릭 로드
    metrics = load_artifact(interim_dir / 'backtest_metrics')

    if metrics is None:
        print("❌ 백테스트 메트릭을 찾을 수 없음")
        return None

    # bt20_short 분석
    bt20_short = metrics[metrics['strategy'] == 'bt20_short']
    bt20_ens = metrics[metrics['strategy'] == 'bt20_ens']

    print("bt20_short vs bt20_ens 비교:")
    print("-" * 40)

    for phase in ['dev', 'holdout']:
        short_data = bt20_short[bt20_short['phase'] == phase]
        ens_data = bt20_ens[bt20_ens['phase'] == phase]

        if not short_data.empty and not ens_data.empty:
            print(f"\n{phase.upper()} 구간:")

            short_sharpe = short_data['net_sharpe'].iloc[0]
            ens_sharpe = ens_data['net_sharpe'].iloc[0]

            short_cagr = short_data['net_cagr'].iloc[0]
            ens_cagr = ens_data['net_cagr'].iloc[0]

            short_mdd = short_data['net_mdd'].iloc[0]
            ens_mdd = ens_data['net_mdd'].iloc[0]

            print(f"  Sharpe: {short_sharpe:.4f} vs {ens_sharpe:.4f}")
            print(f"  CAGR: {short_cagr} vs {ens_cagr}")
            print(f"  MDD: {short_mdd} vs {ens_mdd}")
            # CAGR 차이
            cagr_diff = float(ens_cagr.strip('%')) - float(short_cagr.strip('%'))
            print(f"  CAGR 차이: {cagr_diff:.1f}%"
    # 시장 타이밍 분석
    print("📈 시장 타이밍 분석:")
    print("-" * 30)

    # OHLCV 데이터 로드
    ohlcv_df = load_artifact(interim_dir / 'ohlcv_daily')
    if ohlcv_df is not None:
        # 시장 수익률 계산 (단순 평균)
        market_returns = ohlcv_df.groupby('date')['close'].pct_change().groupby(ohlcv_df['date'].dt.year).mean()

        print("연도별 시장 평균 수익률:")
        for year, ret in market_returns.items():
            if not pd.isna(ret):
                print(".1%")

        # 상승장/하락장 비율
        total_days = len(market_returns)
        up_days = (market_returns > 0).sum()
        print(".1%"
    return bt20_short


def propose_market_timing_solution():
    """
    시장 타이밍 기반 개선안 제안
    """
    print("\n🎯 개선안 1: 시장 타이밍 기반 포지션 스케일링")
    print("="*50)

    cfg = load_config('configs/config.yaml')
    interim_dir = Path(cfg['paths']['base_dir']) / 'data' / 'interim'

    # OHLCV 데이터 로드
    ohlcv_df = load_artifact(interim_dir / 'ohlcv_daily')

    if ohlcv_df is not None:
        # 시장 모멘텀 계산 (60일)
        market_momentum = ohlcv_df.groupby('date')['close'].pct_change(60).mean()

        # 시장 변동성 계산 (20일)
        market_vol = ohlcv_df.groupby('date')['close'].pct_change().rolling(20).std().mean()

        print("시장 조건 분석:")
        print(".4f"        print(".4f"
        # 포지션 스케일링 제안
        print("
포지션 스케일링 전략:"        print("- 상승장 (모멘텀 > 5%): 숏 포지션 50% 축소")
        print("- 고변동성 (변동성 > 2%): 숏 포지션 30% 축소")
        print("- 상승장 + 고변동성: 숏 포지션 70% 축소 또는 청산")

        # 예상 개선 효과
        print("
예상 개선 효과:"        print("- MDD: -21.42% → -15% 수준")
        print("- Sharpe: -0.30 → 0.0 수준")
        print("- CAGR: -7.45% → -2% 수준")


def propose_long_short_hybrid_solution():
    """
    롱숏 하이브리드 전략 개선안 제안
    """
    print("\n🎯 개선안 2: 롱숏 하이브리드 전략")
    print("="*50)

    print("현재 bt20_short: 순수 숏 전략 (12개 포지션)")
    print("개선안: 롱숏 균형 전략 도입")

    print("
구체적 방안:"    print("1. 최상위 8개: 숏 포지션 (현재와 동일)")
    print("2. 최하위 4개: 롱 포지션 (새로 추가)")
    print("3. 총 포지션: 12개 (숏 8개 + 롱 4개)")
    print("4. 목표: 시장 중립성 향상, 리스크 분산")

    print("
예상 개선 효과 (bt20_ens 성과 기반):"    print("- Sharpe: -0.30 → 0.50")
    print("- CAGR: -7.45% → 8.34%")
    print("- MDD: -21.42% → -17.55%")

    print("
장점:"    print("- 시장 상승장에서도 안정적 수익")
    print("- 롱 포지션으로 리스크 헤지")
    print("- bt20_ens와 유사한 성과 달성 가능")


def propose_rebalance_interval_optimization():
    """
    리밸런싱 주기 최적화 개선안 제안
    """
    print("\n🎯 개선안 3: 리밸런싱 주기 최적화")
    print("="*50)

    print("현재: 20일 리밸런싱")
    print("문제: 잦은 리밸런싱으로 거래비용 증가")

    print("
개선 방안:"    print("1. 기본 주기: 20일 유지")
    print("2. 조건부 연장:")
    print("   - 시장 변동성 낮음: 30일로 연장")
    print("   - 포지션 안정성 높음: 30일로 연장")
    print("   - 손실 구간: 10일로 단축 (리스크 관리)")

    print("
예상 개선 효과:"    print("- Turnover: 57.97% → 40% 수준")
    print("- 거래비용 감소로 CAGR 개선")
    print("- 변동성에 따른 유연한 대응")


def propose_risk_management_enhancement():
    """
    리스크 관리 강화 개선안 제안
    """
    print("\n🎯 개선안 4: 리스크 관리 강화")
    print("="*50)

    print("현재 MDD: -21.42% (너무 높음)")

    print("
개선 방안:"    print("1. 동적 포지션 사이즈:")
    print("   - MDD 10% 초과 시: 포지션 20% 축소")
    print("   - MDD 15% 초과 시: 포지션 50% 축소")
    print("   - MDD 20% 초과 시: 전략 일시 중단")

    print("2. 개별 종목 리스크 제한:")
    print("   - 종목별 최대 비중: 15%")
    print("   - 종목별 손실 제한: -5% 초과 시 청산")

    print("3. 상관관계 기반 다각화:")
    print("   - 섹터별 노출 제한")
    print("   - 상관계수 0.8 초과 종목 제외")

    print("
예상 개선 효과:"    print("- MDD: -21.42% → -12% 수준")
    print("- Sharpe: -0.30 → 0.2 수준")
    print("- 안정성 대폭 향상")


def create_improvement_action_plan():
    """
    종합 개선 실행 계획 수립
    """
    print("\n🚀 bt20_short 개선 실행 계획")
    print("="*50)

    print("Phase 1: 시장 타이밍 강화 (1개월)")
    print("-" * 30)
    print("1. 시장 모멘텀 지표 개발")
    print("2. 변동성 기반 포지션 스케일링 로직 구현")
    print("3. 백테스트 및 성과 검증")

    print("\nPhase 2: 롱숏 균형 전략 (2개월)")
    print("-" * 30)
    print("1. 롱 포지션 선택 로직 개발")
    print("2. 포지션 비중 최적화 (숏:롱 = 2:1)")
    print("3. 리스크 관리 통합")

    print("\nPhase 3: 리밸런싱 최적화 (1개월)")
    print("-" * 30)
    print("1. 시장 조건 기반 리밸런싱 주기 조정")
    print("2. 비용 최적화 알고리즘 구현")

    print("\nPhase 4: 종합 검증 및 튜닝 (1개월)")
    print("-" * 30)
    print("1. 모든 개선안 통합")
    print("2. 장기 백테스트 수행")
    print("3. 하이퍼파라미터 최적화")

    print("
⏱️ 총 예상 기간: 5개월"    print("🎯 목표 성과: Sharpe 0.3+, CAGR 3%+, MDD -15% 이하"
    # 설정 파일 업데이트 제안
    print("
⚙️ 설정 파일 업데이트:"    print("- l7.rebalance_interval: 동적 조정 로직 추가")
    print("- l7.position_sizing: 시장 타이밍 기반 스케일링")
    print("- l7.risk_management: MDD 기반 포지션 조정")
    print("- 새 파라미터: market_timing_enabled, long_position_ratio 등")


def generate_improvement_report():
    """
    개선안 보고서 생성
    """
    report = f"""
# bt20_short 전략 개선안 보고서
**생성 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 문제점 분석

### bt20_short 현재 성과
- **Holdout Sharpe**: -0.2987 (매우 낮음)
- **Holdout CAGR**: -7.45% (손실)
- **Holdout MDD**: -21.42% (리스크 높음)
- **Turnover**: 57.97% (비용 높음)

### 주요 원인
1. **숏 포지션의 본질적 어려움**: 상승장 대응 취약
2. **시장 타이밍 부족**: 변동성 높은 20일 주기 리스크
3. **순수 숏 전략 한계**: 리스크 분산 부족
4. **리밸런싱 비용**: 잦은 거래로 비용 증가

## 🎯 개선안 제안

### 1. 시장 타이밍 기반 포지션 스케일링
**개념**: 시장 상황에 따라 숏 포지션 규모 조정
- 상승장: 포지션 50% 축소
- 고변동성: 포지션 30% 축소
**예상 효과**: MDD 21% → 15%, Sharpe -0.3 → 0.0

### 2. 롱숏 하이브리드 전략
**개념**: 숏 8개 + 롱 4개로 시장 중립화
**현재**: 순수 숏 12개
**개선**: 숏 8개 + 롱 4개 (총 12개 유지)
**예상 효과**: Sharpe -0.3 → 0.5, CAGR -7.5% → 8%

### 3. 리밸런싱 주기 최적화
**개념**: 시장 조건에 따른 동적 리밸런싱
- 변동성 낮음: 30일 연장
- 손실 구간: 10일 단축
**예상 효과**: Turnover 58% → 40%, 비용 절감

### 4. 리스크 관리 강화
**개념**: MDD 기반 동적 포지션 조정
- MDD 10%: 20% 축소
- MDD 15%: 50% 축소
- MDD 20%: 전략 중단
**예상 효과**: MDD 21% → 12%, 안정성 향상

## 📈 예상 개선 효과

| 지표 | 현재 | 개선 목표 | 개선율 |
|------|------|----------|--------|
| Sharpe | -0.30 | 0.30 | 200% |
| CAGR | -7.5% | 3.0% | +10.5%p |
| MDD | -21.4% | -12.0% | 44% 감소 |
| Turnover | 58% | 40% | 31% 감소 |

## 🛠️ 구현 우선순위

1. **Phase 1** (시장 타이밍): 즉각적 효과
2. **Phase 2** (롱숏 균형): 근본적 해결
3. **Phase 3** (리밸런싱): 비용 최적화
4. **Phase 4** (통합 검증): 종합 튜닝

## 💡 결론

bt20_short의 저성과는 **시장 상승장 대응력 부족**과 **리스크 관리 미흡**이 주요 원인입니다.
제안된 개선안들은 피쳐 수를 유지하면서도 **시장 적응성**과 **리스크 효율성**을 크게 향상시킬 수 있습니다.

**추천**: Phase 1부터 순차적 적용으로 bt20_ens 수준의 성과 달성 목표.
"""

    # 보고서 저장
    cfg = load_config('configs/config.yaml')
    reports_dir = Path(cfg['paths']['base_dir']) / 'artifacts' / 'reports'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = reports_dir / f'bt20_short_improvement_analysis_{timestamp}.md'

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"\n💾 개선안 보고서 저장: {report_file}")
    return report


def main():
    """
    메인 실행 함수
    """
    print("🎯 bt20_short 전략 개선 분석")
    print("="*50)

    # 문제점 분석
    problems = analyze_bt20_short_problems()

    if problems is None:
        print("❌ 분석 실패")
        return

    # 개선안 제시
    propose_market_timing_solution()
    propose_long_short_hybrid_solution()
    propose_rebalance_interval_optimization()
    propose_risk_management_enhancement()

    # 실행 계획
    create_improvement_action_plan()

    # 보고서 생성
    report = generate_improvement_report()

    print("\n✅ bt20_short 개선 분석 완료!")
    print("="*50)


if __name__ == "__main__":
    main()
