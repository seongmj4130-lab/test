#!/usr/bin/env python3
"""
성과 지표 간의 관계 설명
"""

from pathlib import Path

import pandas as pd


def explain_metrics_relationship():
    """성과 지표들이 왜 함께 변하는지 설명"""

    print("🔗 성과 지표 간의 관계 설명")
    print("=" * 50)

    # 최신 결과 파일 로드
    results_dir = Path("results")
    csv_files = list(results_dir.glob("dynamic_period_backtest_clean_*.csv"))
    latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)

    df = pd.read_csv(latest_file)
    print(f"📊 분석 파일: {latest_file.name}")
    print(f"📈 데이터: {len(df)} 행")
    print()

    # 샘플 데이터로 관계 설명
    sample = df[df["strategy"] == "bt20_short"].head(3)
    print("🔍 단기 전략 샘플 데이터:")
    print(
        sample[
            [
                "strategy",
                "holding_days",
                "sharpe",
                "CAGR (%)",
                "Total Return (%)",
                "MDD (%)",
            ]
        ].to_string(index=False, float_format="%.2f")
    )
    print()

    print("📋 성과 지표 계산 관계:")
    print("=" * 40)
    print("1️⃣  CAGR (연간 복리 수익률)")
    print("   - 실제 수익률 곡선에서 계산")
    print("   - true_short/true_long이 직접적인 입력값")
    print()

    print("2️⃣  Total Return (총 수익률)")
    print("   - 기간 전체 누적 수익률")
    print("   - CAGR과 밀접한 관계")
    print()

    print("3️⃣  MDD (Maximum Drawdown)")
    print("   - 수익률 곡선의 최대 하락폭")
    print("   - 실제 수익률이 작아지면 MDD도 작아짐")
    print()

    print("4️⃣  Sharpe 비율")
    print("   - (평균 수익률 - 무위험수익률) / 변동성")
    print("   - 수익률 ↓ → Sharpe ↓ (또는 음수)")
    print()

    print("5️⃣  Hit Ratio (승률)")
    print("   - 수익이 난 거래의 비율")
    print("   - 실제 수익률 계산에 따라 결정")
    print()

    print("6️⃣  Profit Factor")
    print("   - 총 이익 / 총 손실")
    print("   - 실제 수익률 크기에 따라 결정")
    print()

    print("7️⃣  Calmar 비율")
    print("   - Sharpe 비율 / |MDD|")
    print("   - Sharpe와 MDD의 복합 효과")
    print()

    # 수학적 관계 시각화
    print("📊 수학적 관계:")
    print("-" * 30)
    print("수익률 = f(true_short, true_long)")
    print("↓")
    print("Sharpe = (수익률 - rf) / σ(수익률)")
    print("MDD = max(수익률_고점 - 수익률_저점)")
    print("Hit Ratio = count(수익률 > 0) / total_trades")
    print("Profit Factor = sum(수익률 > 0) / |sum(수익률 < 0)|")
    print()

    # 실제 데이터로 증명
    print("💡 실제 데이터 증거:")
    print("-" * 30)

    # 20일 vs 80일 비교
    day20 = df[(df["strategy"] == "bt20_short") & (df["holding_days"] == 20)]
    day80 = df[(df["strategy"] == "bt20_short") & (df["holding_days"] == 80)]

    if len(day20) > 0 and len(day80) > 0:
        print("단기 전략 20일 vs 80일 비교:")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".2f")
    print()
    print("🎯 결론:")
    print("수익률은 모든 성과 지표의 '근본 입력값'입니다.")
    print("수익률을 수정하면 관련된 모든 지표가 함께 변합니다.")
    print("이는 정상적인 현상이며, 데이터의 일관성을 보장합니다.")


if __name__ == "__main__":
    explain_metrics_relationship()
