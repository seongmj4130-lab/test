"""백테스트 결과에 영향을 주는 모든 config 설정값 표시"""
import sys
from pathlib import Path

import yaml

# BacktestConfig 기본값 가져오기
sys.path.insert(
    0, str(Path("C:/Users/seong/OneDrive/Desktop/bootcamp/03_code").resolve())
)
from src.tracks.track_b.stages.backtest.l7_backtest import BacktestConfig

base_dir = Path("C:/Users/seong/OneDrive/Desktop/bootcamp/03_code")

# config.yaml 로드
config_path = base_dir / "configs" / "config.yaml"
with open(config_path, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# BacktestConfig 기본값
default_cfg = BacktestConfig()

# 전략별 설정
strategies = {
    "bt20_short": "l7_bt20_short",
    "bt20_ens": "l7_bt20_ens",
    "bt120_long": "l7_bt120_long",
    "bt120_ens": "l7_bt120_ens",
}

print("=" * 120)
print("백테스트 결과에 영향을 주는 모든 config 설정값")
print("=" * 120)
print()

# BacktestConfig에서 정의된 모든 속성 가져오기
from dataclasses import fields

bt_config_fields = {
    f.name: getattr(default_cfg, f.name) for f in fields(BacktestConfig)
}

print("[BacktestConfig 기본값 목록]")
print("-" * 120)
print(f"{'설정 키':<45} | {'타입':<15} | {'기본값':<30}")
print("-" * 120)
for field in fields(BacktestConfig):
    field_type = (
        str(field.type).replace("<class '", "").replace("'>", "").replace("typing.", "")
    )
    if "Optional" in field_type:
        field_type = field_type.replace("Optional[", "").replace("]", "?")
    default_val = getattr(default_cfg, field.name)
    if isinstance(default_val, str):
        default_val_str = f'"{default_val}"'
    elif isinstance(default_val, bool):
        default_val_str = str(default_val)
    elif default_val is None:
        default_val_str = "None"
    else:
        default_val_str = str(default_val)
    print(f"{field.name:<45} | {field_type:<15} | {default_val_str:<30}")

print("\n" + "=" * 120)
print("전략별 config.yaml 설정값 (실제 적용되는 값)")
print("=" * 120)

for strategy_name, config_key in strategies.items():
    print(f"\n[{strategy_name}]")
    print("=" * 120)

    l7_cfg = cfg.get(config_key, {})
    l7_regime = (
        (l7_cfg.get("regime", {}) or {})
        if isinstance(l7_cfg.get("regime", {}), dict)
        else {}
    )
    l7_div = (
        (l7_cfg.get("diversify", {}) or {})
        if isinstance(l7_cfg.get("diversify", {}), dict)
        else {}
    )

    # 기본 설정
    print("\n[기본 설정]")
    print("-" * 120)
    basic_configs = [
        ("holding_days", "holding_days"),
        ("top_k", "top_k"),
        ("cost_bps", "cost_bps"),
        ("slippage_bps", "slippage_bps"),
        ("score_col", "score_col"),
        ("return_col", "ret_col"),
        ("weighting", "weighting"),
        ("softmax_temperature", "softmax_temp"),
        ("buffer_k", "buffer_k"),
        ("rebalance_interval", "rebalance_interval"),
    ]

    for config_key_name, attr_name in basic_configs:
        config_val = (
            l7_cfg.get(config_key_name)
            if config_key_name != "softmax_temperature"
            else l7_cfg.get("softmax_temperature", l7_cfg.get("softmax_temp"))
        )
        default_val = getattr(default_cfg, attr_name)
        if config_val is not None:
            mark = " ⚠️ (기본값과 다름)" if config_val != default_val else " ✅"
            if isinstance(config_val, str):
                config_val_str = f'"{config_val}"'
            else:
                config_val_str = str(config_val)
            print(
                f"  {config_key_name:<35} = {config_val_str:<30} (기본값: {default_val}){mark}"
            )
        else:
            default_val_str = (
                f'"{default_val}"' if isinstance(default_val, str) else str(default_val)
            )
            print(f"  {config_key_name:<35} = {default_val_str:<30} (기본값 사용) ✅")

    # 고급 설정
    print("\n[고급 설정 - 스마트 버퍼/변동성 조정]")
    print("-" * 120)
    advanced_configs = [
        ("smart_buffer_enabled", "smart_buffer_enabled"),
        ("smart_buffer_stability_threshold", "smart_buffer_stability_threshold"),
        ("volatility_adjustment_enabled", "volatility_adjustment_enabled"),
        ("volatility_lookback_days", "volatility_lookback_days"),
        ("target_volatility", "target_volatility"),
        ("volatility_adjustment_max", "volatility_adjustment_max"),
        ("volatility_adjustment_min", "volatility_adjustment_min"),
    ]

    for config_key_name, attr_name in advanced_configs:
        config_val = l7_cfg.get(config_key_name)
        default_val = getattr(default_cfg, attr_name)
        if config_val is not None:
            mark = " ⚠️ (기본값과 다름)" if config_val != default_val else " ✅"
            print(
                f"  {config_key_name:<35} = {config_val:<30} (기본값: {default_val}){mark}"
            )
        else:
            print(f"  {config_key_name:<35} = {default_val:<30} (기본값 사용) ✅")

    # 리스크 스케일링
    print("\n[리스크 스케일링]")
    print("-" * 120)
    risk_configs = [
        ("risk_scaling_enabled", "risk_scaling_enabled"),
        ("risk_scaling_bear_multiplier", "risk_scaling_bear_multiplier"),
        ("risk_scaling_neutral_multiplier", "risk_scaling_neutral_multiplier"),
        ("risk_scaling_bull_multiplier", "risk_scaling_bull_multiplier"),
    ]

    for config_key_name, attr_name in risk_configs:
        config_val = l7_cfg.get(config_key_name)
        default_val = getattr(default_cfg, attr_name)
        if config_val is not None:
            mark = " ⚠️ (기본값과 다름)" if config_val != default_val else " ✅"
            print(
                f"  {config_key_name:<35} = {config_val:<30} (기본값: {default_val}){mark}"
            )
        else:
            print(f"  {config_key_name:<35} = {default_val:<30} (기본값 사용) ✅")

    # 오버래핑 트랜치
    print("\n[오버래핑 트랜치]")
    print("-" * 120)
    tranche_configs = [
        ("overlapping_tranches_enabled", "overlapping_tranches_enabled"),
        ("tranche_holding_days", "tranche_holding_days"),
        ("tranche_max_active", "tranche_max_active"),
        ("tranche_allocation_mode", "tranche_allocation_mode"),
    ]

    for config_key_name, attr_name in tranche_configs:
        config_val = l7_cfg.get(config_key_name)
        default_val = getattr(default_cfg, attr_name)
        if config_val is not None:
            mark = " ⚠️ (기본값과 다름)" if config_val != default_val else " ✅"
            config_val_str = (
                f'"{config_val}"' if isinstance(config_val, str) else str(config_val)
            )
            default_val_str = (
                f'"{default_val}"' if isinstance(default_val, str) else str(default_val)
            )
            print(
                f"  {config_key_name:<35} = {config_val_str:<30} (기본값: {default_val_str}){mark}"
            )
        else:
            default_val_str = (
                f'"{default_val}"' if isinstance(default_val, str) else str(default_val)
            )
            print(f"  {config_key_name:<35} = {default_val_str:<30} (기본값 사용) ✅")

    # 업종 분산
    print("\n[업종 분산]")
    print("-" * 120)
    div_configs = [
        ("diversify.enabled", "diversify_enabled"),
        ("diversify.group_col", "group_col"),
        ("diversify.max_names_per_group", "max_names_per_group"),
    ]

    div_map = {
        "diversify.enabled": ("enabled", "diversify_enabled"),
        "diversify.group_col": ("group_col", "group_col"),
        "diversify.max_names_per_group": ("max_names_per_group", "max_names_per_group"),
    }

    for config_key_name, attr_name in div_configs:
        key_path = div_map[config_key_name][0]
        config_val = (
            l7_div.get(key_path)
            if key_path in ["enabled", "group_col", "max_names_per_group"]
            else l7_div.get(key_path)
        )
        default_val = getattr(default_cfg, attr_name)
        if config_val is not None:
            mark = " ⚠️ (기본값과 다름)" if config_val != default_val else " ✅"
            config_val_str = (
                f'"{config_val}"' if isinstance(config_val, str) else str(config_val)
            )
            default_val_str = (
                f'"{default_val}"' if isinstance(default_val, str) else str(default_val)
            )
            print(
                f"  {config_key_name:<35} = {config_val_str:<30} (기본값: {default_val_str}){mark}"
            )
        else:
            default_val_str = (
                f'"{default_val}"' if isinstance(default_val, str) else str(default_val)
            )
            print(f"  {config_key_name:<35} = {default_val_str:<30} (기본값 사용) ✅")

    # 시장 국면 (Regime)
    print("\n[시장 국면 (Regime)]")
    print("-" * 120)

    regime_configs = [
        ("regime.enabled", "regime_enabled"),
        ("regime.lookback_days", None),  # BacktestConfig에 없음
        ("regime.threshold_pct", None),  # BacktestConfig에 없음
        ("regime.neutral_band", None),  # BacktestConfig에 없음
        ("regime.top_k_bull_strong", "regime_top_k_bull_strong"),
        ("regime.top_k_bull_weak", "regime_top_k_bull_weak"),
        ("regime.top_k_bear_strong", "regime_top_k_bear_strong"),
        ("regime.top_k_bear_weak", "regime_top_k_bear_weak"),
        ("regime.top_k_neutral", "regime_top_k_neutral"),
        ("regime.exposure_bull_strong", "regime_exposure_bull_strong"),
        ("regime.exposure_bull_weak", "regime_exposure_bull_weak"),
        ("regime.exposure_bear_strong", "regime_exposure_bear_strong"),
        ("regime.exposure_bear_weak", "regime_exposure_bear_weak"),
        ("regime.exposure_neutral", "regime_exposure_neutral"),
        ("regime.top_k_bull", "regime_top_k_bull"),
        ("regime.top_k_bear", "regime_top_k_bear"),
        ("regime.exposure_bull", "regime_exposure_bull"),
        ("regime.exposure_bear", "regime_exposure_bear"),
    ]

    for config_key_name, attr_name in regime_configs:
        key = config_key_name.replace("regime.", "")
        config_val = l7_regime.get(key)

        if attr_name is None:
            # BacktestConfig에 없는 설정
            if config_val is not None:
                config_val_str = (
                    f'"{config_val}"'
                    if isinstance(config_val, str)
                    else str(config_val)
                )
                print(
                    f"  {config_key_name:<35} = {config_val_str:<30} (BacktestConfig에 없음)"
                )
            else:
                print(f"  {config_key_name:<35} = (설정 안 됨)")
        else:
            default_val = getattr(default_cfg, attr_name)
            if config_val is not None:
                mark = " ⚠️ (기본값과 다름)" if config_val != default_val else " ✅"
                print(
                    f"  {config_key_name:<35} = {config_val:<30} (기본값: {default_val}){mark}"
                )
            else:
                default_val_str = "None" if default_val is None else str(default_val)
                print(
                    f"  {config_key_name:<35} = {default_val_str:<30} (기본값 사용) ✅"
                )

print("\n" + "=" * 120)
print("설명")
print("=" * 120)
print(
    """
✅ = 기본값과 동일하거나 config.yaml에 명시되지 않아 기본값 사용
⚠️ = config.yaml에서 기본값과 다르게 설정됨

주의:
- config.yaml에 명시된 값이 우선 적용됩니다
- config.yaml에 없는 설정은 BacktestConfig 기본값이 사용됩니다
- 모든 설정값은 백테스트 결과에 영향을 줍니다
"""
)
