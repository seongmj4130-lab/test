"""전략별 공통 설정값 분석"""
from collections import Counter
from pathlib import Path

import yaml

base_dir = Path("C:/Users/seong/OneDrive/Desktop/bootcamp/03_code")

# config.yaml 로드
config_path = base_dir / "configs" / "config.yaml"
with open(config_path, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

strategies = {
    "bt20_short": "l7_bt20_short",
    "bt20_ens": "l7_bt20_ens",
    "bt120_long": "l7_bt120_long",
    "bt120_ens": "l7_bt120_ens",
}

# 각 전략의 설정값 수집
all_configs = {}
for strategy_name, config_key in strategies.items():
    l7_cfg = cfg.get(config_key, {})
    all_configs[strategy_name] = l7_cfg

# 기본 설정값 추출
basic_configs = {}
for strategy_name in strategies.keys():
    config = all_configs[strategy_name]
    basic_configs[strategy_name] = {
        "holding_days": config.get("holding_days"),
        "top_k": config.get("top_k"),
        "cost_bps": config.get("cost_bps"),
        "slippage_bps": config.get("slippage_bps"),
        "score_col": config.get("score_col"),
        "return_col": config.get("return_col"),
        "weighting": config.get("weighting"),
        "softmax_temp": config.get("softmax_temperature", config.get("softmax_temp")),
        "buffer_k": config.get("buffer_k"),
        "rebalance_interval": config.get("rebalance_interval"),
        "smart_buffer_enabled": config.get("smart_buffer_enabled"),
        "smart_buffer_stability_threshold": config.get(
            "smart_buffer_stability_threshold"
        ),
        "volatility_adjustment_enabled": config.get("volatility_adjustment_enabled"),
        "volatility_lookback_days": config.get("volatility_lookback_days"),
        "target_volatility": config.get("target_volatility"),
        "volatility_adjustment_max": config.get("volatility_adjustment_max"),
        "volatility_adjustment_min": config.get("volatility_adjustment_min"),
        "risk_scaling_enabled": config.get("risk_scaling_enabled"),
        "risk_scaling_bear_multiplier": config.get("risk_scaling_bear_multiplier"),
        "risk_scaling_neutral_multiplier": config.get(
            "risk_scaling_neutral_multiplier"
        ),
        "risk_scaling_bull_multiplier": config.get("risk_scaling_bull_multiplier"),
        "overlapping_tranches_enabled": config.get("overlapping_tranches_enabled"),
        "tranche_holding_days": config.get("tranche_holding_days"),
        "tranche_max_active": config.get("tranche_max_active"),
        "tranche_allocation_mode": config.get("tranche_allocation_mode"),
    }

# 공통값 찾기
print("=" * 100)
print("전략별 공통 설정값 분석")
print("=" * 100)
print()

common_configs = {}
for key in basic_configs["bt20_short"].keys():
    values = [
        basic_configs[s][key]
        for s in strategies.keys()
        if basic_configs[s][key] is not None
    ]
    if len(values) == len(strategies) and len(set(values)) == 1:
        common_configs[key] = values[0]
        print(f"✅ {key:<40} = {values[0]} (모든 전략 동일)")
    elif len(set(values)) > 1:
        value_counts = Counter(values)
        most_common = value_counts.most_common(1)[0]
        print(
            f"⚠️  {key:<40} = {most_common[0]} (대부분, {most_common[1]}/{len(strategies)} 전략)"
        )

print()
print("=" * 100)
print("권장 BacktestConfig 기본값 (공통값 기반)")
print("=" * 100)

# 현재 BacktestConfig와 비교하여 업데이트 권장값
current_defaults = {
    "holding_days": 20,
    "top_k": 20,
    "cost_bps": 10.0,
    "slippage_bps": 0.0,
    "score_col": "score_ens",
    "ret_col": "true_short",
    "weighting": "equal",
    "softmax_temp": 1.0,
    "buffer_k": 0,
    "rebalance_interval": 1,
    "smart_buffer_enabled": False,
    "smart_buffer_stability_threshold": 0.7,
    "volatility_adjustment_enabled": False,
    "volatility_lookback_days": 60,
    "target_volatility": 0.15,
    "volatility_adjustment_max": 1.2,
    "volatility_adjustment_min": 0.6,
    "risk_scaling_enabled": False,
    "risk_scaling_bear_multiplier": 0.7,
    "risk_scaling_neutral_multiplier": 0.9,
    "risk_scaling_bull_multiplier": 1.0,
    "overlapping_tranches_enabled": False,
    "tranche_holding_days": 120,
    "tranche_max_active": 4,
    "tranche_allocation_mode": "fixed_equal",
}

# 공통값 기반 권장값
recommended = {
    "holding_days": common_configs.get(
        "holding_days", current_defaults["holding_days"]
    ),
    "top_k": 20,  # 가장 자주 사용 (bt120_ens)
    "cost_bps": common_configs.get("cost_bps", current_defaults["cost_bps"]),
    "slippage_bps": common_configs.get(
        "slippage_bps", current_defaults["slippage_bps"]
    ),
    "score_col": "score_ens",  # 기본값 유지
    "ret_col": common_configs.get("return_col", current_defaults["ret_col"]),
    "weighting": "equal",  # 기본값 유지
    "softmax_temp": 1.0,  # 기본값 유지
    "buffer_k": 15,  # 3개 전략이 사용
    "rebalance_interval": 20,  # 모든 전략이 사용
    "smart_buffer_enabled": True,  # 모든 전략이 사용
    "smart_buffer_stability_threshold": common_configs.get(
        "smart_buffer_stability_threshold",
        current_defaults["smart_buffer_stability_threshold"],
    ),
    "volatility_adjustment_enabled": True,  # 모든 전략이 사용
    "volatility_lookback_days": common_configs.get(
        "volatility_lookback_days", current_defaults["volatility_lookback_days"]
    ),
    "target_volatility": common_configs.get(
        "target_volatility", current_defaults["target_volatility"]
    ),
    "volatility_adjustment_max": common_configs.get(
        "volatility_adjustment_max", current_defaults["volatility_adjustment_max"]
    ),
    "volatility_adjustment_min": 0.6,  # bt120는 0.6 사용
    "risk_scaling_enabled": True,  # 모든 전략이 사용
    "risk_scaling_bear_multiplier": 0.7,  # bt120는 0.7 사용
    "risk_scaling_neutral_multiplier": 0.9,  # bt120는 0.9 사용
    "risk_scaling_bull_multiplier": common_configs.get(
        "risk_scaling_bull_multiplier", current_defaults["risk_scaling_bull_multiplier"]
    ),
    "overlapping_tranches_enabled": False,  # bt120만 사용
    "tranche_holding_days": common_configs.get(
        "tranche_holding_days", current_defaults["tranche_holding_days"]
    ),
    "tranche_max_active": common_configs.get(
        "tranche_max_active", current_defaults["tranche_max_active"]
    ),
    "tranche_allocation_mode": common_configs.get(
        "tranche_allocation_mode", current_defaults["tranche_allocation_mode"]
    ),
}

print()
for key, value in recommended.items():
    current = current_defaults.get(key)
    if value != current:
        print(f"  {key:<40} = {value} (현재: {current}) ⚠️ 변경 필요")
    else:
        print(f"  {key:<40} = {value} ✅")
