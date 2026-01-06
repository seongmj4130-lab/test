# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/analyze_stage_evolution.py
"""
Stage0~12 변화값 분석 스크립트
"""
import pandas as pd
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def analyze_stage_evolution():
    """Stage0~12 변화값 분석"""
    # History manifest 로드
    manifest_path = PROJECT_ROOT / "reports" / "history" / "history_manifest.parquet"
    if not manifest_path.exists():
        manifest_path = PROJECT_ROOT / "reports" / "history" / "history_manifest.csv"
    
    if not manifest_path.exists():
        print("ERROR: history_manifest 파일을 찾을 수 없습니다.")
        sys.exit(1)
    
    if manifest_path.suffix == ".parquet":
        df = pd.read_parquet(manifest_path)
    else:
        df = pd.read_csv(manifest_path)
    
    # Stage별로 정렬 (stage_no 기준)
    df = df.sort_values("stage_no").reset_index(drop=True)
    
    # 핵심 지표만 선택
    key_metrics = [
        "stage_no",
        "run_tag",
        "change_title",
        "holdout_sharpe",
        "holdout_mdd",
        "holdout_cagr",
        "holdout_total_return",
        "net_sharpe",
        "net_mdd",
        "net_total_return",
        "information_ratio",
        "tracking_error_ann",
        "avg_turnover_oneway",
    ]
    
    available_cols = [c for c in key_metrics if c in df.columns]
    analysis_df = df[available_cols].copy()
    
    # 숫자 컬럼을 숫자 타입으로 변환
    numeric_cols = [
        "holdout_sharpe", "holdout_mdd", "holdout_cagr", "holdout_total_return",
        "net_sharpe", "net_mdd", "net_total_return",
        "information_ratio", "tracking_error_ann", "avg_turnover_oneway"
    ]
    
    for col in numeric_cols:
        if col in analysis_df.columns:
            analysis_df[col] = pd.to_numeric(analysis_df[col], errors='coerce')
    
    print("="*80)
    print("Stage0~12 변화값 분석")
    print("="*80)
    print(f"\n총 Stage 수: {len(analysis_df)}개")
    print(f"Stage 범위: {analysis_df['stage_no'].min()} ~ {analysis_df['stage_no'].max()}")
    
    # Stage별 주요 지표 출력
    print("\n" + "="*80)
    print("Stage별 주요 지표")
    print("="*80)
    
    for idx, row in analysis_df.iterrows():
        stage_no = row["stage_no"]
        change_title = row.get("change_title", "N/A")
        print(f"\n[Stage {stage_no}] {change_title}")
        print(f"  Run Tag: {row.get('run_tag', 'N/A')}")
        
        if "holdout_sharpe" in row and pd.notna(row["holdout_sharpe"]):
            print(f"  Holdout Sharpe: {float(row['holdout_sharpe']):.4f}")
        if "holdout_mdd" in row and pd.notna(row["holdout_mdd"]):
            print(f"  Holdout MDD: {float(row['holdout_mdd']):.2f}%")
        if "holdout_cagr" in row and pd.notna(row["holdout_cagr"]):
            print(f"  Holdout CAGR: {float(row['holdout_cagr']):.2f}%")
        if "holdout_total_return" in row and pd.notna(row["holdout_total_return"]):
            print(f"  Holdout Total Return: {float(row['holdout_total_return']):.2f}%")
        if "net_sharpe" in row and pd.notna(row["net_sharpe"]):
            print(f"  Net Sharpe: {float(row['net_sharpe']):.4f}")
        if "net_mdd" in row and pd.notna(row["net_mdd"]):
            print(f"  Net MDD: {float(row['net_mdd']):.2f}%")
        if "information_ratio" in row and pd.notna(row["information_ratio"]):
            print(f"  Information Ratio: {float(row['information_ratio']):.4f}")
        if "avg_turnover_oneway" in row and pd.notna(row["avg_turnover_oneway"]):
            print(f"  Avg Turnover: {float(row['avg_turnover_oneway']):.2f}%")
    
    # 변화량 계산 (이전 Stage 대비)
    print("\n" + "="*80)
    print("Stage별 변화량 (이전 Stage 대비)")
    print("="*80)
    
    change_summary = []
    
    for idx in range(1, len(analysis_df)):
        prev_row = analysis_df.iloc[idx - 1]
        curr_row = analysis_df.iloc[idx]
        
        prev_stage = prev_row["stage_no"]
        curr_stage = curr_row["stage_no"]
        
        changes = {
            "from_stage": prev_stage,
            "to_stage": curr_stage,
            "change_title": curr_row.get("change_title", "N/A"),
        }
        
        # 주요 지표 변화량 계산
        metrics_to_diff = [
            "holdout_sharpe",
            "holdout_mdd",
            "holdout_cagr",
            "holdout_total_return",
            "net_sharpe",
            "net_mdd",
            "information_ratio",
        ]
        
        for metric in metrics_to_diff:
            if metric in prev_row and metric in curr_row:
                prev_val = prev_row[metric]
                curr_val = curr_row[metric]
                
                if pd.notna(prev_val) and pd.notna(curr_val):
                    diff = curr_val - prev_val
                    pct_change = (diff / abs(prev_val) * 100) if prev_val != 0 else 0
                    changes[f"{metric}_diff"] = diff
                    changes[f"{metric}_pct"] = pct_change
        
        change_summary.append(changes)
    
    # 변화량 요약 출력
    for change in change_summary:
        print(f"\n[Stage {change['from_stage']} → {change['to_stage']}] {change['change_title']}")
        
        if "holdout_sharpe_diff" in change:
            print(f"  Holdout Sharpe: {change['holdout_sharpe_diff']:+.4f} ({change.get('holdout_sharpe_pct', 0):+.2f}%)")
        if "holdout_mdd_diff" in change:
            print(f"  Holdout MDD: {change['holdout_mdd_diff']:+.2f}%p ({change.get('holdout_mdd_pct', 0):+.2f}%)")
        if "holdout_cagr_diff" in change:
            print(f"  Holdout CAGR: {change['holdout_cagr_diff']:+.2f}%p ({change.get('holdout_cagr_pct', 0):+.2f}%)")
        if "information_ratio_diff" in change:
            print(f"  Information Ratio: {change['information_ratio_diff']:+.4f} ({change.get('information_ratio_pct', 0):+.2f}%)")
    
    # 최종 요약
    print("\n" + "="*80)
    print("전체 기간 변화 요약 (Baseline → Stage12)")
    print("="*80)
    
    baseline_row = analysis_df[analysis_df["stage_no"] == -1].iloc[0] if len(analysis_df[analysis_df["stage_no"] == -1]) > 0 else None
    final_row = analysis_df.iloc[-1]
    
    if baseline_row is not None:
        print(f"\nBaseline (Stage -1) → Final (Stage {final_row['stage_no']})")
        
        for metric in metrics_to_diff:
            if metric in baseline_row and metric in final_row:
                baseline_val = baseline_row[metric]
                final_val = final_row[metric]
                
                if pd.notna(baseline_val) and pd.notna(final_val):
                    diff = final_val - baseline_val
                    pct_change = (diff / abs(baseline_val) * 100) if baseline_val != 0 else 0
                    
                    metric_name = metric.replace("_", " ").title()
                    print(f"  {metric_name}: {baseline_val:.4f} → {final_val:.4f} ({diff:+.4f}, {pct_change:+.2f}%)")
    
    # CSV로 저장
    output_path = PROJECT_ROOT / "reports" / "analysis" / "stage_evolution_analysis.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 분석 결과를 DataFrame으로 변환
    evolution_df = analysis_df.copy()
    evolution_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n[저장] 분석 결과: {output_path}")
    
    # 변화량 요약도 저장
    if change_summary:
        change_df = pd.DataFrame(change_summary)
        change_output_path = PROJECT_ROOT / "reports" / "analysis" / "stage_evolution_changes.csv"
        change_df.to_csv(change_output_path, index=False, encoding='utf-8-sig')
        print(f"[저장] 변화량 요약: {change_output_path}")

if __name__ == "__main__":
    analyze_stage_evolution()
