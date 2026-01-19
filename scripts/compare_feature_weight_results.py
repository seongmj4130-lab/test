# -*- coding: utf-8 -*-
"""피쳐 가중치 적용 전후 백테스트 결과 비교"""

import pandas as pd
from pathlib import Path

base_dir = Path(__file__).parent.parent
backup_dir = base_dir / 'artifacts' / 'reports' / 'backup_before_weight_change'

# 백업 파일 찾기
backup_files = list(backup_dir.glob('*.csv')) if backup_dir.exists() else []
if not backup_files:
    print("백업 파일이 없습니다.")
    exit(1)

latest_backup = max(backup_files, key=lambda p: p.stat().st_mtime)
print(f"백업 파일: {latest_backup.name}\n")

models = ['bt20_ens', 'bt20_short', 'bt120_ens', 'bt120_long']

for model in models:
    # 백업 파일 찾기
    backup_file = backup_dir / latest_backup.name.replace('bt_metrics_bt20_ens', f'bt_metrics_{model}')
    if not backup_file.exists():
        pattern_files = list(backup_dir.glob(f'*{model}*.csv'))
        if pattern_files:
            backup_file = pattern_files[0]
        else:
            continue
    
    # 현재 파일
    current_file = base_dir / 'artifacts' / 'reports' / f'bt_metrics_{model}_optimized.csv'
    
    if not current_file.exists():
        continue
    
    before = pd.read_csv(backup_file)
    after = pd.read_csv(current_file)
    
    print(f"={model.upper()}=")
    for phase in ['dev', 'holdout']:
        b = before[before['phase'] == phase]
        a = after[after['phase'] == phase]
        
        if len(b) == 0 or len(a) == 0:
            continue
        
        print(f"\n  {phase.upper()}:")
        for metric in ['net_sharpe', 'net_cagr', 'net_mdd', 'net_calmar_ratio']:
            if metric not in b.columns or metric not in a.columns:
                continue
            
            before_val = b[metric].values[0]
            after_val = a[metric].values[0]
            change = after_val - before_val
            change_pct = (change / abs(before_val) * 100) if before_val != 0 else 0
            
            metric_name = metric.replace('net_', '').replace('_', ' ').title()
            improvement = "✅" if (change > 0 and metric != 'net_mdd') or (change < 0 and metric == 'net_mdd') else "❌"
            
            print(f"    {metric_name:20s}: {before_val:8.4f} -> {after_val:8.4f} ({change:+.4f}, {change_pct:+.2f}%) {improvement}")
    print()

