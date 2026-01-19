"""Hit Ratio 최적화 전후 비교"""

from pathlib import Path

import pandas as pd

base_dir = Path(__file__).parent.parent
backup_dir = base_dir / "artifacts" / "reports" / "backup_before_weight_change"

# 백업 파일 찾기
backup_files = list(backup_dir.glob("*bt20_short*.csv")) if backup_dir.exists() else []

if not backup_files:
    print("백업 파일이 없습니다.")
    exit(1)

backup_file = backup_files[0]
current_file = (
    base_dir / "artifacts" / "reports" / "bt_metrics_bt20_short_optimized.csv"
)

before = pd.read_csv(backup_file)
after = pd.read_csv(current_file)

print("=" * 80)
print("BT20_SHORT Hit Ratio 최적화 전후 비교")
print("=" * 80)

for phase in ["dev", "holdout"]:
    b = before[before["phase"] == phase].iloc[0]
    a = after[after["phase"] == phase].iloc[0]

    print(f"\n[{phase.upper()}]")
    print(
        f"  Hit Ratio: {b['net_hit_ratio']:.4f} → {a['net_hit_ratio']:.4f} ({a['net_hit_ratio'] - b['net_hit_ratio']:+.4f})"
    )
    print(
        f"  Sharpe:    {b['net_sharpe']:.4f} → {a['net_sharpe']:.4f} ({a['net_sharpe'] - b['net_sharpe']:+.4f})"
    )
    print(
        f"  CAGR:      {b['net_cagr']:.4f} → {a['net_cagr']:.4f} ({a['net_cagr'] - b['net_cagr']:+.4f})"
    )
    print(
        f"  MDD:       {b['net_mdd']:.4f} → {a['net_mdd']:.4f} ({a['net_mdd'] - b['net_mdd']:+.4f})"
    )

# 과적합 계산
dev_hr_before = before[before["phase"] == "dev"]["net_hit_ratio"].iloc[0]
holdout_hr_before = before[before["phase"] == "holdout"]["net_hit_ratio"].iloc[0]
overfitting_before = dev_hr_before - holdout_hr_before

dev_hr_after = after[after["phase"] == "dev"]["net_hit_ratio"].iloc[0]
holdout_hr_after = after[after["phase"] == "holdout"]["net_hit_ratio"].iloc[0]
overfitting_after = dev_hr_after - holdout_hr_after

print("\n" + "=" * 80)
print("과적합 분석")
print("=" * 80)
print(
    f"  변경 전: Dev {dev_hr_before:.4f} - Holdout {holdout_hr_before:.4f} = {overfitting_before:.4f}"
)
print(
    f"  변경 후: Dev {dev_hr_after:.4f} - Holdout {holdout_hr_after:.4f} = {overfitting_after:.4f}"
)
print(
    f"  개선: {overfitting_before:.4f} → {overfitting_after:.4f} ({overfitting_after - overfitting_before:+.4f})"
)

print("\n" + "=" * 80)
print("목표 달성 여부")
print("=" * 80)
target_met_before = holdout_hr_before >= 0.50
target_met_after = holdout_hr_after >= 0.50
overfitting_ok_before = overfitting_before <= 0.10
overfitting_ok_after = overfitting_after <= 0.10

print(
    f"  Holdout Hit Ratio ≥ 50%: {'✅' if target_met_before else '❌'} → {'✅' if target_met_after else '❌'}"
)
print(
    f"  과적합 ≤ 10%: {'✅' if overfitting_ok_before else '❌'} → {'✅' if overfitting_ok_after else '❌'}"
)

if target_met_after and overfitting_ok_after:
    print("\n✅ 목표 달성!")
elif target_met_after:
    print("\n⚠️ Hit Ratio는 달성했으나 과적합이 여전히 존재합니다.")
elif overfitting_ok_after:
    print("\n⚠️ 과적합은 개선되었으나 Hit Ratio가 50% 미만입니다.")
else:
    print("\n❌ 추가 최적화가 필요합니다.")
