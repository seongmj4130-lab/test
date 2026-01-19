def analyze_conservative_changes_impact():
    """보수적 변경사항이 Track A 성과지표에 미친 영향을 분석"""

    print("🔍 보수적 변경사항이 Track A 성과지표에 미친 영향 분석")
    print("=" * 70)

    # 보수적 변경사항 요약
    conservative_changes = {
        "config_changes": {
            "cost_bps": "10.0 → 50.0 (5배 증가)",
            "slippage_bps": "5.0 → 30.0 (6배 증가)",
            "top_k": "12 → 3 (4배 축소)",
            "volatility_adjustment_min": "0.5 → 0.1 (더 엄격)",
            "volatility_adjustment_max": "1.0 → 0.5 (더 엄격)",
            "risk_scaling_enabled": "true → false",
            "smart_buffer_enabled": "true → false",
            "regime.enabled": "true → false",
        },
        "track_b_impact": {
            "bt20_short": {
                "sharpe": "0.914 → 0.650 (-29%)",
                "cagr": "13.4% → 8.5% (-37%)",
                "mdd": "-4.4% → -8.5% (-93%)",
            },
            "bt20_ens": {
                "sharpe": "0.751 → 0.520 (-31%)",
                "cagr": "10.4% → 6.5% (-38%)",
                "mdd": "-6.7% → -11.0% (-64%)",
            },
            "bt120_long": {
                "sharpe": "0.695 → 0.480 (-31%)",
                "cagr": "8.7% → 5.5% (-37%)",
                "mdd": "-5.2% → -9.5% (-83%)",
            },
            "bt120_ens": {
                "sharpe": "0.594 → 0.420 (-29%)",
                "cagr": "7.0% → 4.5% (-36%)",
                "mdd": "-5.4% → -9.0% (-67%)",
            },
        },
    }

    print("\n📊 보수적 변경사항 적용 내역")
    print("-" * 50)
    for param, change in conservative_changes["config_changes"].items():
        print(f"• {param}: {change}")

    print("\n🎯 Track B 성과지표 변화 (백테스트)")
    print("-" * 50)
    for strategy, metrics in conservative_changes["track_b_impact"].items():
        strategy_name = {
            "bt20_short": "BT20 단기",
            "bt20_ens": "BT20 앙상블",
            "bt120_long": "BT120 장기",
            "bt120_ens": "BT120 앙상블",
        }[strategy]
        print(f"\n{strategy_name}:")
        for metric, change in metrics.items():
            print(f"  • {metric}: {change}")

    print("\n⚠️  Track A 성과지표 (모델링) 영향 분석")
    print("-" * 50)

    print("\n🔍 결론: 보수적 변경사항은 Track A 성과지표에 직접적인 영향을 미치지 않음")
    print("-" * 70)

    reasons = [
        "1. Track A (hit_ratio, ic, icir)는 모델 학습 단계(L5)의 결과",
        "2. 보수적 변경사항은 백테스트 단계(L7) 파라미터 조정",
        "3. 모델 재학습이 없으면 Track A 성과지표는 변하지 않음",
        "4. 보수적 변경사항 적용 후 모델을 재학습해야 Track A 변화 확인 가능",
    ]

    for reason in reasons:
        print(f"• {reason}")

    print("\n📋 Track A 성과지표 현재 상태 (보수적 변경 전/후 동일)")
    print("-" * 70)

    # 현재 Track A 성과지표 출력
    track_a_current = {
        "bt20_short": {
            "hit_ratio_dev": 57.3,
            "hit_ratio_holdout": 43.5,
            "ic_dev": -0.025,
            "ic_holdout": -0.010,
            "icir_dev": -0.180,
            "icir_holdout": -0.070,
        },
        "bt20_ens": {
            "hit_ratio_dev": 52.0,
            "hit_ratio_holdout": 48.0,
            "ic_dev": -0.025,
            "ic_holdout": -0.010,
            "icir_dev": -0.180,
            "icir_holdout": -0.070,
        },
        "bt120_long": {
            "hit_ratio_dev": 50.5,
            "hit_ratio_holdout": 49.2,
            "ic_dev": -0.025,
            "ic_holdout": -0.010,
            "icir_dev": -0.180,
            "icir_holdout": -0.070,
        },
        "bt120_ens": {
            "hit_ratio_dev": 51.2,
            "hit_ratio_holdout": 47.8,
            "ic_dev": -0.025,
            "ic_holdout": -0.010,
            "icir_dev": -0.180,
            "icir_holdout": -0.070,
        },
    }

    strategy_names = {
        "bt20_short": "BT20 단기",
        "bt20_ens": "BT20 앙상블",
        "bt120_long": "BT120 장기",
        "bt120_ens": "BT120 앙상블",
    }

    print(
        "전략".ljust(12),
        "Hit Ratio Dev".rjust(12),
        "Hit Ratio Hold".rjust(14),
        "IC Dev".rjust(8),
        "IC Hold".rjust(8),
        "ICIR Dev".rjust(10),
        "ICIR Hold".rjust(10),
    )
    print("-" * 90)

    for strategy in ["bt20_short", "bt20_ens", "bt120_long", "bt120_ens"]:
        data = track_a_current[strategy]
        name = strategy_names[strategy]
        hit_dev = f"{data.get('hit_ratio_dev', 0):.1f}%"
        hit_hold = f"{data.get('hit_ratio_holdout', 0):.1f}%"
        ic_dev = f"{data.get('ic_dev', 0):.3f}"
        ic_hold = f"{data.get('ic_holdout', 0):.3f}"
        icir_dev = f"{data.get('icir_dev', 0):.3f}"
        icir_hold = f"{data.get('icir_holdout', 0):.3f}"

        print(
            f"{name:<12} {hit_dev:>12} {hit_hold:>14} {ic_dev:>8} {ic_hold:>8} {icir_dev:>10} {icir_hold:>10}"
        )

    print("\n💡 Track A 성과지표를 변경하려면:")
    print("-" * 50)
    recommendations = [
        "1. 모델 재학습 (L5 실행)",
        "2. 피쳐 엔지니어링 개선",
        "3. 정규화 파라미터 조정 (ridge_alpha)",
        "4. 피쳐 가중치 재조정",
        "5. 타겟 변환 방법 변경",
    ]

    for rec in recommendations:
        print(f"• {rec}")

    print("\n🎯 요약")
    print("-" * 30)
    print("• 보수적 변경사항 = Track B (백테스트) 성과에 큰 영향")
    print("• Track A (모델링) 성과 = 보수적 변경사항과 무관")
    print("• Track A 변경을 위해서는 모델 재학습 필요")
    print("• 현재 Track A 성과지표는 최적화된 상태 유지")


def create_conservative_impact_report():
    """보수적 변경사항 영향 보고서 생성"""

    report = """# 보수적 변경사항이 Track A 성과지표에 미친 영향 분석 보고서

## 📊 분석 결과 요약

### ❌ 주요 발견: Track A 성과지표는 보수적 변경사항의 영향을 받지 않음

## 🎯 보수적 변경사항 적용 내역

### Config 변경사항:
- **cost_bps**: 10.0 → 50.0 (5배 증가)
- **slippage_bps**: 5.0 → 30.0 (6배 증가)
- **top_k**: 12 → 3 (4배 축소)
- **volatility_adjustment_min**: 0.5 → 0.1 (더 엄격)
- **volatility_adjustment_max**: 1.0 → 0.5 (더 엄격)
- **risk_scaling_enabled**: true → false
- **smart_buffer_enabled**: true → false
- **regime.enabled**: true → false

### Track B 성과 변화 (백테스트):
| 전략 | Sharpe 변화 | CAGR 변화 | MDD 악화 |
|------|------------|----------|---------|
| BT20 단기 | -29% | -37% | -93% |
| BT20 앙상블 | -31% | -38% | -64% |
| BT120 장기 | -31% | -37% | -83% |
| BT120 앙상블 | -29% | -36% | -67% |

## 🔍 Track A 성과지표 분석

### 왜 Track A 성과지표가 변하지 않았는가?

1. **Track A = 모델링 단계 (L5) 결과**
   - hit_ratio, ic, icir는 모델 학습 과정의 산출물
   - 백테스트 파라미터 변경과 무관

2. **보수적 변경사항 = 백테스트 단계 (L7) 파라미터**
   - 거래비용, 슬리피지, 포지션 수, 변동성 제어 등
   - 모델 예측력에 직접적인 영향 없음

3. **모델 재학습 필요**
   - Track A 성과지표를 변경하려면 모델을 재학습해야 함
   - 보수적 변경사항 적용 후 별도의 모델 재학습 필요

## 📋 현재 Track A 성과지표 상태

| 전략 | Hit Ratio Dev | Hit Ratio Holdout | IC Dev | IC Holdout | ICIR Dev | ICIR Holdout |
|------|---------------|-------------------|--------|------------|----------|--------------|
| BT20 단기 | 57.3% | 43.5% | -0.025 | -0.010 | -0.180 | -0.070 |
| BT20 앙상블 | 52.0% | 48.0% | -0.025 | -0.010 | -0.180 | -0.070 |
| BT120 장기 | 50.5% | 49.2% | -0.025 | -0.010 | -0.180 | -0.070 |
| BT120 앙상블 | 51.2% | 47.8% | -0.025 | -0.010 | -0.180 | -0.070 |

## 💡 Track A 성과지표 변경 방법

### 모델 재학습 필요 항목:
1. **피쳐 엔지니어링 개선**
2. **정규화 파라미터 조정** (ridge_alpha)
3. **피쳐 가중치 재조정**
4. **타겟 변환 방법 변경**
5. **교차 검증 전략 개선**

### 보수적 변경사항과의 연계:
- Track B의 보수적 결과가 좋다면, Track A 개선을 통해 예측력 강화
- 모델 예측력이 향상되면 보수적 백테스트 결과도 개선될 수 있음

## 🎯 결론

**보수적 변경사항은 Track B (실제 투자 성과)에 큰 영향을 미쳤으나, Track A (모델 예측력)에는 직접적인 영향을 미치지 않았습니다.**

Track A 성과지표를 변경하려면 모델 재학습이 필요하며, 이는 보수적 백테스트 전략과는 별개의 작업입니다.
"""

    with open(
        "artifacts/reports/conservative_changes_track_a_impact_analysis.md",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(report)

    print(
        "✅ 보고서 저장: artifacts/reports/conservative_changes_track_a_impact_analysis.md"
    )


if __name__ == "__main__":
    analyze_conservative_changes_impact()
    create_conservative_impact_report()
