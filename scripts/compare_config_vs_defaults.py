# -*- coding: utf-8 -*-
"""config.yaml 설정값 vs BacktestConfig 기본값 비교"""
import yaml
from pathlib import Path
import sys

# BacktestConfig 기본값 가져오기
sys.path.insert(0, str(Path('C:/Users/seong/OneDrive/Desktop/bootcamp/03_code').resolve()))
from src.tracks.track_b.stages.backtest.l7_backtest import BacktestConfig

base_dir = Path('C:/Users/seong/OneDrive/Desktop/bootcamp/03_code')

# config.yaml 로드
config_path = base_dir / 'configs' / 'config.yaml'
with open(config_path, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

# BacktestConfig 기본값
default_cfg = BacktestConfig()

# 전략별 설정
strategies = {
    'bt20_short': 'l7_bt20_short',
    'bt20_ens': 'l7_bt20_ens',
    'bt120_long': 'l7_bt120_long',
    'bt120_ens': 'l7_bt120_ens',
}

print("=" * 100)
print("config.yaml 설정값 vs BacktestConfig 기본값 비교")
print("=" * 100)
print()

# BacktestConfig 기본값 출력
print("[BacktestConfig 기본값]")
print("-" * 100)
config_attrs = [
    'holding_days', 'top_k', 'cost_bps', 'slippage_bps', 'score_col', 'ret_col',
    'weighting', 'softmax_temp', 'buffer_k', 'rebalance_interval',
    'smart_buffer_enabled', 'smart_buffer_stability_threshold',
    'volatility_adjustment_enabled', 'volatility_lookback_days', 'target_volatility',
    'volatility_adjustment_max', 'volatility_adjustment_min',
    'risk_scaling_enabled', 'risk_scaling_bear_multiplier',
    'risk_scaling_neutral_multiplier', 'risk_scaling_bull_multiplier',
    'overlapping_tranches_enabled', 'tranche_holding_days', 'tranche_max_active',
    'tranche_allocation_mode', 'diversify_enabled', 'group_col', 'max_names_per_group',
    'regime_enabled',
]

defaults_dict = {}
for attr in config_attrs:
    if hasattr(default_cfg, attr):
        defaults_dict[attr] = getattr(default_cfg, attr)
        print(f"  {attr:35s} = {getattr(default_cfg, attr)}")

print("\n" + "=" * 100)
print("전략별 config.yaml 설정값 vs BacktestConfig 기본값 비교")
print("=" * 100)

for strategy_name, config_key in strategies.items():
    print(f"\n[{strategy_name}]")
    print("-" * 100)
    
    l7_cfg = cfg.get(config_key, {})
    l7_regime = (l7_cfg.get("regime", {}) or {}) if isinstance(l7_cfg.get("regime", {}), dict) else {}
    l7_div = (l7_cfg.get("diversify", {}) or {}) if isinstance(l7_cfg.get("diversify", {}), dict) else {}
    
    differences = []
    
    # 기본 설정 비교
    for attr in config_attrs:
        default_val = defaults_dict.get(attr)
        if default_val is None:
            continue
        
        # config.yaml에서 값 가져오기
        config_val = None
        if attr == 'regime_enabled':
            config_val = l7_regime.get("enabled", None)
        elif attr.startswith('regime_'):
            regime_key = attr.replace('regime_', '')
            config_val = l7_regime.get(regime_key, None)
        elif attr == 'diversify_enabled':
            config_val = l7_div.get("enabled", None)
        elif attr == 'group_col':
            config_val = l7_div.get("group_col", None)
        elif attr == 'max_names_per_group':
            config_val = l7_div.get("max_names_per_group", None)
        elif attr == 'softmax_temp':
            config_val = l7_cfg.get("softmax_temperature", l7_cfg.get("softmax_temp", None))
        else:
            config_val = l7_cfg.get(attr, None)
        
        # 비교
        if config_val is not None and config_val != default_val:
            differences.append({
                'attr': attr,
                'default': default_val,
                'config': config_val,
                'diff': f"{default_val} → {config_val}",
            })
    
    if differences:
        print(f"  ⚠️  {len(differences)}개 설정값이 BacktestConfig 기본값과 다름:")
        for diff in differences:
            print(f"    - {diff['attr']:35s}: 기본값={diff['default']!s:15s} | config.yaml={str(diff['config']):15s}")
    else:
        print(f"  ✅ 모든 설정값이 BacktestConfig 기본값과 동일 (또는 config.yaml에 명시되지 않음)")

print("\n" + "=" * 100)
print("요약")
print("=" * 100)

total_diffs = {}
for strategy_name, config_key in strategies.items():
    l7_cfg = cfg.get(config_key, {})
    l7_regime = (l7_cfg.get("regime", {}) or {}) if isinstance(l7_cfg.get("regime", {}), dict) else {}
    l7_div = (l7_cfg.get("diversify", {}) or {}) if isinstance(l7_cfg.get("diversify", {}), dict) else {}
    
    for attr in config_attrs:
        default_val = defaults_dict.get(attr)
        if default_val is None:
            continue
        
        config_val = None
        if attr == 'regime_enabled':
            config_val = l7_regime.get("enabled", None)
        elif attr.startswith('regime_'):
            regime_key = attr.replace('regime_', '')
            config_val = l7_regime.get(regime_key, None)
        elif attr == 'diversify_enabled':
            config_val = l7_div.get("enabled", None)
        elif attr == 'group_col':
            config_val = l7_div.get("group_col", None)
        elif attr == 'max_names_per_group':
            config_val = l7_div.get("max_names_per_group", None)
        elif attr == 'softmax_temp':
            config_val = l7_cfg.get("softmax_temperature", l7_cfg.get("softmax_temp", None))
        else:
            config_val = l7_cfg.get(attr, None)
        
        if config_val is not None and config_val != default_val:
            if attr not in total_diffs:
                total_diffs[attr] = {}
            total_diffs[attr][strategy_name] = {
                'default': default_val,
                'config': config_val,
            }

if total_diffs:
    print(f"\n총 {len(total_diffs)}개 설정값이 config.yaml에서 BacktestConfig 기본값과 다르게 설정됨:")
    for attr, strategy_diffs in total_diffs.items():
        print(f"\n  [{attr}]")
        print(f"    BacktestConfig 기본값: {defaults_dict[attr]}")
        for strategy_name, diff in strategy_diffs.items():
            print(f"      {strategy_name}: {diff['config']} (기본값 {diff['default']}에서 변경)")
else:
    print("\n✅ 모든 전략에서 config.yaml 설정값이 BacktestConfig 기본값과 동일하거나 명시되지 않음")

