"""
rebalance_interval 변경 시 지표가 동일한 이유 분석
"""

print("=" * 80)
print("문제 분석: rebalance_interval 변경 시 지표가 동일한 경우")
print("=" * 80)

print("\n1. Forward Return의 특성:")
print("   - true_short = 리밸런싱 날짜로부터 20일 후의 수익률")
print("   - true_long = 리밸런싱 날짜로부터 120일 후의 수익률")
print(
    "   - 이것은 '미래 수익률'이므로, 리밸런싱 날짜가 달라지면 다른 수익률을 사용합니다"
)

print("\n2. 리밸런싱 빈도가 줄어들면:")
print("   - 리밸런싱 날짜 수가 줄어듭니다 (예: 252일 → 12일)")
print("   - 각 리밸런싱 날짜마다 다른 forward return을 사용합니다")
print("   - 따라서 누적 수익률이 달라져야 합니다")

print("\n3. 지표가 동일한 경우의 가능한 원인:")
print("   ⚠️  원인 1: rebalance_interval 필터링이 제대로 작동하지 않음")
print("      - 코드에서 dphase 필터링이 제대로 되지 않았을 수 있음")
print("      - 또는 입력 데이터 자체가 이미 필터링된 상태일 수 있음")
print("")
print("   ⚠️  원인 2: 입력 데이터의 날짜가 이미 sparse함")
print("      - rebalance_scores 데이터가 이미 월별/분기별로만 존재")
print("      - 따라서 rebalance_interval을 변경해도 실제 리밸런싱 날짜가 동일")
print("")
print("   ⚠️  원인 3: Forward return의 중복 사용")
print("      - 리밸런싱 날짜가 달라도, forward return이 겹치는 구간이 있을 수 있음")
print("      - 예: Day 1의 20일 후 = Day 21, Day 21의 20일 후 = Day 41")
print("      - 하지만 이것도 누적 수익률은 달라야 합니다")

print("\n4. 실제 확인 필요 사항:")
print("   - rebalance_scores 데이터의 날짜 분포 확인")
print("   - rebalance_interval 필터링 전후의 날짜 수 비교")
print("   - 각 리밸런싱 날짜의 forward return 값 확인")

print("\n" + "=" * 80)
print("해결 방법:")
print("=" * 80)
print("1. rebalance_scores 데이터의 날짜 분포를 확인하세요")
print("2. rebalance_interval 필터링 로직이 제대로 작동하는지 확인하세요")
print("3. 실제 리밸런싱 날짜 수가 변경되었는지 확인하세요")
