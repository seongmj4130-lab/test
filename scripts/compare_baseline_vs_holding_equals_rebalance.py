# -*- coding: utf-8 -*-
"""
Baseline vs holding_days=rebalance_interval 비교 리포트 생성
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

sys.stdout.reconfigure(encoding='utf-8')

from src.utils.config import load_config, get_path
from src.utils.io import load_artifact

def main():
    """메인 실행 함수"""
    cfg = load_config("configs/config.yaml")
    interim_dir = Path(get_path(cfg, "data_interim"))
    artifacts_dir = Path(get_path(cfg, "artifacts_reports"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # 현재 결과 (holding_days = rebalance_interval)
    current_results = load_artifact(artifacts_dir / "backtest_4models_comparison.csv")
    
    # Baseline 결과 로드
    baseline_files = {
        'bt20_ens': interim_dir / 'bt_metrics_bt20_ens_optimized.parquet',
        'bt20_short': interim_dir / 'bt_metrics_bt20_short_optimized.parquet',
        'bt120_ens': interim_dir / 'bt_metrics_bt120_ens_optimized.parquet',
        'bt120_long': interim_dir / 'bt_metrics_bt120_long_optimized.parquet',
    }
    
    comparison_rows = []
    
    for strategy_name in ['bt20_ens', 'bt20_short', 'bt120_ens', 'bt120_long']:
        # 현재 결과
        current = current_results[current_results['strategy'] == strategy_name]
        if len(current) == 0:
            continue
        
        # Baseline 결과
        baseline_file = baseline_files.get(strategy_name)
        if baseline_file and baseline_file.exists():
            baseline_metrics = load_artifact(baseline_file)
            baseline_holdout = baseline_metrics[baseline_metrics['phase'] == 'holdout']
            
            if len(baseline_holdout) > 0:
                row = {
                    'strategy': strategy_name,
                    'baseline_sharpe': baseline_holdout['net_sharpe'].iloc[0],
                    'current_sharpe': current['net_sharpe'].iloc[0],
                    'baseline_cagr': baseline_holdout['net_cagr'].iloc[0],
                    'current_cagr': current['net_cagr'].iloc[0],
                    'baseline_mdd': baseline_holdout['net_mdd'].iloc[0],
                    'current_mdd': current['net_mdd'].iloc[0],
                    'baseline_calmar': baseline_holdout['net_calmar_ratio'].iloc[0],
                    'current_calmar': current['net_calmar_ratio'].iloc[0],
                }
                
                # 개선도 계산
                if row['baseline_sharpe'] != 0:
                    row['sharpe_change_pct'] = (row['current_sharpe'] - row['baseline_sharpe']) / abs(row['baseline_sharpe']) * 100
                if row['baseline_cagr'] != 0:
                    row['cagr_change_pct'] = (row['current_cagr'] - row['baseline_cagr']) / abs(row['baseline_cagr']) * 100
                if row['baseline_calmar'] != 0:
                    row['calmar_change_pct'] = (row['current_calmar'] - row['baseline_calmar']) / abs(row['baseline_calmar']) * 100
                
                comparison_rows.append(row)
    
    if comparison_rows:
        comparison_df = pd.DataFrame(comparison_rows)
        comparison_file = artifacts_dir / "baseline_vs_holding_equals_rebalance.csv"
        comparison_df.to_csv(comparison_file, index=False, encoding='utf-8-sig')
        print(f"\n[비교 리포트] {comparison_file}")
        print("\n" + comparison_df.to_string())
        
        # 마크다운 리포트 생성
        report_md = f"""# Baseline vs holding_days=rebalance_interval 비교 리포트

생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 설정 변경
- **BT20 모델**: holding_days=20, rebalance_interval=20 (이전: holding_days=20, rebalance_interval=1)
- **BT120 모델**: holding_days=120, rebalance_interval=120 (이전: holding_days=120, rebalance_interval=6)

## 결과 비교 (Holdout 구간)

"""
        for _, row in comparison_df.iterrows():
            report_md += f"""### {row['strategy']}

| 지표 | Baseline | holding_days=rebalance_interval | 변화율 |
|------|----------|--------------------------------|--------|
| Sharpe | {row['baseline_sharpe']:.4f} | {row['current_sharpe']:.4f} | {row.get('sharpe_change_pct', 0):.2f}% |
| CAGR | {row['baseline_cagr']:.4%} | {row['current_cagr']:.4%} | {row.get('cagr_change_pct', 0):.2f}% |
| MDD | {row['baseline_mdd']:.4%} | {row['current_mdd']:.4%} | - |
| Calmar | {row['baseline_calmar']:.4f} | {row['current_calmar']:.4f} | {row.get('calmar_change_pct', 0):.2f}% |

"""
        
        report_file = artifacts_dir / "baseline_vs_holding_equals_rebalance.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_md)
        print(f"[저장] {report_file}")

if __name__ == "__main__":
    main()

