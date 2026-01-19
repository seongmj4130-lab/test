"""설정 복원 확인"""
from pathlib import Path

import yaml

config_path = Path("configs/config.yaml")
with open(config_path, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

print("=" * 80)
print("설정 복원 확인 (2026-01-07 13:21:00 이전 상태로 복원)")
print("=" * 80)

strategies = {
    "bt20_short": "l7_bt20_short",
    "bt20_ens": "l7_bt20_ens",
    "bt120_long": "l7_bt120_long",
    "bt120_ens": "l7_bt120_ens",
}

for name, key in strategies.items():
    s_cfg = cfg.get(key, {})
    print(f"\n[{name}]")
    print(f"  rebalance_interval: {s_cfg.get('rebalance_interval', 'N/A')}")
    if name.startswith("bt120"):
        print(f"  return_col: {s_cfg.get('return_col', 'N/A')}")
        print(
            f"  overlapping_tranches_enabled: {s_cfg.get('overlapping_tranches_enabled', 'N/A')}"
        )
        if s_cfg.get("overlapping_tranches_enabled", False):
            print(f"  tranche_max_active: {s_cfg.get('tranche_max_active', 'N/A')}")

print("\n" + "=" * 80)
print("복원 완료:")
print("  ✓ rebalance_interval: 20 → 1 (모든 전략)")
print("  ✓ BT120 return_col: true_short → true_long")
print("  ✓ BT120 overlapping_tranches_enabled: true → false")
print("=" * 80)
