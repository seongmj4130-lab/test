#!/usr/bin/env python3
"""
실무 관점에서 백테스트 결과 분석
"""

from pathlib import Path

import pandas as pd


def analyze_practical_issues():
    """실무 관점에서 성과 지표 분석 및 문제점 도출"""

    print("🔬 실무 관점 성과 지표 분석")
    print("=" * 60)

    # 최신 결과 파일 로드
    results_dir = Path("results")
    csv_files = list(results_dir.glob("dynamic_period_backtest_clean_*.csv"))
    latest_file = max(csv_files, key=lambda x: x.stat().st_mtime)

    df = pd.read_csv(latest_file)
    print(f"📊 분석 파일: {latest_file.name}")
    print(f"📈 데이터: {len(df)} 행")
    print()

    # 전략별 평균 성과 계산
    strategy_summary = (
        df.groupby("strategy")
        .agg(
            {
                "sharpe": "mean",
                "CAGR (%)": "mean",
                "Total Return (%)": "mean",
                "MDD (%)": "mean",
                "calmar": "mean",
                "Hit Ratio (%)": "mean",
                "avg_turnover": "mean",
                "profit_factor": "mean",
            }
        )
        .round(3)
    )

    print("📊 전략별 평균 성과:")
    print(strategy_summary)
    print()

    # 실무 관점 문제점 분석
    issues = []

    # 1. 수익률 문제
    avg_cagr = df["CAGR (%)"].mean()
    if avg_cagr < 1.0:
        issues.append(
            {
                "문제": "수익률 부진",
                "심각도": "높음",
                "설명": f"평균 CAGR {avg_cagr:.2f}%는 투자 매력도가 낮음",
                "실무적 의미": "연 1% 미만 수익률로는 인플레이션도 커버 못함",
            }
        )

    # 2. Sharpe 비율 문제
    avg_sharpe = df["sharpe"].mean()
    if avg_sharpe < 0.5:
        issues.append(
            {
                "문제": "리스크 대비 수익률 저조",
                "심각도": "높음",
                "설명": f"평균 Sharpe {avg_sharpe:.2f}는 시장 평균(0.5-1.0)보다 낮음",
                "실무적 의미": "리스크를 감수할 만큼의 초과수익률 없음",
            }
        )

    # 3. MDD 문제
    avg_mdd = df["MDD (%)"].mean()
    if abs(avg_mdd) > 10:
        issues.append(
            {
                "문제": "하락 위험 과대",
                "심각도": "중간",
                "설명": f"평균 MDD {avg_mdd:.2f}%는 투자자 심리적 부담 큼",
                "실무적 의미": "10% 이상 하락 시 투자자 이탈 가능성 높음",
            }
        )

    # 4. Hit Ratio 문제
    avg_hit_ratio = df["Hit Ratio (%)"].mean()
    if avg_hit_ratio < 50:
        issues.append(
            {
                "문제": "승률 낮음",
                "심각도": "중간",
                "설명": f"평균 Hit Ratio {avg_hit_ratio:.1f}%는 개선 필요",
                "실무적 의미": "50% 미만 승률은 전략 신뢰성 의문",
            }
        )

    # 5. Turnover 문제
    avg_turnover = df["avg_turnover"].mean()
    if avg_turnover > 0.5:  # 50% 이상
        issues.append(
            {
                "문제": "거래 비용 과다",
                "심각도": "중간",
                "설명": f"평균 Turnover {avg_turnover:.2f}는 거래비용 부담 큼",
                "실무적 의미": "높은 턴오버는 수익률을 잠식",
            }
        )

    # 6. Profit Factor 문제
    avg_pf = df["profit_factor"].mean()
    if avg_pf < 1.2:
        issues.append(
            {
                "문제": "손익 비율 불균형",
                "심각도": "높음",
                "설명": f"평균 Profit Factor {avg_pf:.2f}는 1.2 이상이 바람직",
                "실무적 의미": "이익보다 손실이 더 큼",
            }
        )

    # 7. 전략 간 차별성 부족
    strategy_std = df.groupby("strategy")["Total Return (%)"].std().mean()
    if strategy_std < 2.0:
        issues.append(
            {
                "문제": "전략 차별성 부족",
                "심각도": "중간",
                "설명": f"전략 간 수익률 표준편차 {strategy_std:.2f}%로 차별성 부족",
                "실무적 의미": "다양한 시장 상황 대응력 부족",
            }
        )

    # 8. 기간별 안정성 부족
    period_std = df.groupby("holding_days")["Total Return (%)"].std().mean()
    if period_std > 5.0:
        issues.append(
            {
                "문제": "기간별 안정성 부족",
                "심각도": "중간",
                "설명": f"기간별 수익률 변동성 {period_std:.2f}%로 불안정",
                "실무적 의미": "투자 기간 선택의 어려움",
            }
        )

    # 문제점 출력
    print("🚨 실무 관점 주요 문제점:")
    print("=" * 60)

    severity_order = {"높음": 3, "중간": 2, "낮음": 1}
    issues_sorted = sorted(
        issues, key=lambda x: severity_order[x["심각도"]], reverse=True
    )

    for i, issue in enumerate(issues_sorted, 1):
        print(f"{i}. {issue['문제']} ({issue['심각도']})")
        print(f"   설명: {issue['설명']}")
        print(f"   실무적 의미: {issue['실무적 의미']}")
        print()

    # 종합 평가
    high_count = sum(1 for issue in issues if issue["심각도"] == "높음")
    medium_count = sum(1 for issue in issues if issue["심각도"] == "중간")

    print("📋 종합 평가:")
    print(f"총 문제점: {len(issues)}개")
    print(f"고위험 문제: {high_count}개")
    print(f"中위험 문제: {medium_count}개")

    if high_count >= 2:
        print("⚠️  실전 투자가 어려운 수준 - 전략 전면 재검토 필요")
    elif high_count == 1 and medium_count >= 2:
        print("⚠️  부분적 개선 필요 - 핵심 전략 재설계 고려")
    else:
        print("✅ 기본적 개선 가능 - 파라미터 튜닝으로 해결 가능")

    print("\n💡 Hit Ratio 수정 방안:")
    print("- L6 단계의 피처별 Hit Ratio를 L7 백테스트에 통합")
    print("- 랭킹 산정 단계의 예측력을 정확히 반영")
    print("- IC와 Hit Ratio를 결합한 종합 평가 지표 개발")


if __name__ == "__main__":
    analyze_practical_issues()
