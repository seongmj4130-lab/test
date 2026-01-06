# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/run_stage_with_reports.py
"""
단계별 파이프라인 실행 + KPI 생성 + Delta 생성 + Manifest/Audit 생성 통합 스크립트
프로젝트 규칙에 따라 필수 절차를 자동으로 실행합니다.
"""
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# 고정 설정
PROJECT_ROOT = Path(r"C:\Users\seong\OneDrive\Desktop\bootcamp\03_code")
BASELINE_TAG = "baseline_prerefresh_20251219_143636"

def run_command(cmd: list, cwd: Path, description: str, log_file: Optional[Path] = None) -> int:
    """명령어 실행 및 로그 저장"""
    print(f"\n{'='*60}")
    print(f"[{description}]")
    print(f"Command: {' '.join(cmd)}")
    print(f"Working Directory: {cwd}")
    print(f"{'='*60}\n")
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, 'w', encoding='utf-8') as f:
            result = subprocess.run(cmd, cwd=str(cwd), stdout=f, stderr=subprocess.STDOUT, text=True)
        # 로그 파일 내용 일부 출력
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print("".join(lines[-50:]))  # 마지막 50줄만 출력
        except:
            pass
    else:
        result = subprocess.run(cmd, cwd=str(cwd))
    
    if result.returncode != 0:
        print(f"\n❌ [{description}] Failed with exit code {result.returncode}")
        if log_file:
            print(f"   Log file: {log_file}")
    else:
        print(f"\n✅ [{description}] Completed")
    
    return result.returncode

def check_file_exists(file_path: Path, description: str) -> bool:
    """파일 존재 여부 확인"""
    if file_path.exists():
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: NOT FOUND - {file_path}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Run pipeline stage with full report generation (KPI + Delta + Manifest/Audit)"
    )
    parser.add_argument("--stage-tag", type=str, required=True,
                       help="Stage tag (e.g., stage1_leakage_universe_fix_20251219_143636)")
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Config file path")
    parser.add_argument("--from-stage", type=str, default="L0",
                       help="Start stage (default: L0)")
    parser.add_argument("--to-stage", type=str, default="L7D",
                       help="End stage (default: L7D)")
    parser.add_argument("--stage", type=str, default=None,
                       help="Single stage to run (overrides --from-stage and --to-stage)")
    parser.add_argument("--force", action="store_true",
                       help="Force re-run even if outputs exist")
    parser.add_argument("--skip-pipeline", action="store_true",
                       help="Skip pipeline execution (only generate reports)")
    parser.add_argument("--skip-manifest", action="store_true",
                       help="Skip manifest/audit generation")
    parser.add_argument("--root", type=str, default=None,
                       help="Project root directory (default: fixed path)")
    args = parser.parse_args()
    
    # 루트 경로 결정
    if args.root:
        root = Path(args.root)
    else:
        root = PROJECT_ROOT
    
    if not root.exists():
        print(f"❌ Project root not found: {root}")
        sys.exit(1)
    
    stage_tag = args.stage_tag
    baseline_tag = BASELINE_TAG
    
    print("\n" + "="*60)
    print("PROJECT EXECUTION RULES - AUTOMATED RUNNER")
    print("="*60)
    print(f"Root: {root}")
    print(f"Stage Tag: {stage_tag}")
    print(f"Baseline Tag: {baseline_tag}")
    print(f"Config: {args.config}")
    print("="*60 + "\n")
    
    # 로그 디렉토리 생성
    logs_dir = root / "reports" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"run__{stage_tag}.log"
    
    # ============================================================
    # 1) 파이프라인 실행
    # ============================================================
    if not args.skip_pipeline:
        pipeline_cmd = [
            sys.executable,
            str(root / "src" / "run_all.py"),
            "--config", args.config,
            "--run-tag", stage_tag,
        ]
        
        if args.stage:
            pipeline_cmd.extend(["--stage", args.stage])
        else:
            pipeline_cmd.extend(["--from", args.from_stage, "--to", args.to_stage])
        
        if args.force:
            pipeline_cmd.append("--force")
        
        exit_code = run_command(
            pipeline_cmd,
            cwd=root,
            description="Pipeline Execution",
            log_file=log_file
        )
        
        if exit_code != 0:
            print(f"\n❌ Pipeline execution failed. Check log: {log_file}")
            sys.exit(exit_code)
    else:
        print("\n⏭️  Skipping pipeline execution (--skip-pipeline)")
    
    # ============================================================
    # 2) KPI 생성
    # ============================================================
    kpi_cmd = [
        sys.executable,
        str(root / "src" / "tools" / "export_kpi_table.py"),
        "--config", args.config,
        "--tag", stage_tag,
    ]
    
    exit_code = run_command(
        kpi_cmd,
        cwd=root,
        description="KPI Generation"
    )
    
    if exit_code != 0:
        print(f"\n❌ KPI generation failed")
        sys.exit(exit_code)
    
    # KPI 파일 존재 확인
    kpi_csv = root / "reports" / "kpi" / f"kpi_table__{stage_tag}.csv"
    kpi_md = root / "reports" / "kpi" / f"kpi_table__{stage_tag}.md"
    
    if not check_file_exists(kpi_csv, "KPI CSV"):
        print("❌ KPI CSV file missing. Exiting.")
        sys.exit(1)
    
    if not check_file_exists(kpi_md, "KPI MD"):
        print("❌ KPI MD file missing. Exiting.")
        sys.exit(1)
    
    # KPI 상위 40줄 출력
    print("\n" + "="*60)
    print("KPI Summary (first 40 lines):")
    print("="*60)
    try:
        with open(kpi_md, 'r', encoding='utf-8') as f:
            lines = f.readlines()[:40]
            print("".join(lines))
    except Exception as e:
        print(f"Error reading KPI MD: {e}")
    
    # ============================================================
    # 3) Baseline KPI 확인 및 생성 (필요시)
    # ============================================================
    baseline_kpi_csv = root / "reports" / "kpi" / f"kpi_table__{baseline_tag}.csv"
    
    if not baseline_kpi_csv.exists():
        print(f"\n⚠️  Baseline KPI not found: {baseline_kpi_csv}")
        print("   Checking if baseline artifacts exist in data/interim/...")
        
        baseline_interim_dir = root / "data" / "interim" / baseline_tag
        if baseline_interim_dir.exists():
            print(f"   ✅ Baseline artifacts found. Generating baseline KPI...")
            baseline_kpi_cmd = [
                sys.executable,
                str(root / "src" / "tools" / "export_kpi_table.py"),
                "--config", args.config,
                "--tag", baseline_tag,
            ]
            exit_code = run_command(
                baseline_kpi_cmd,
                cwd=root,
                description="Baseline KPI Generation"
            )
            if exit_code != 0:
                print("   ⚠️  Baseline KPI generation failed. Continuing with current stage only.")
        else:
            print("   ⚠️  Baseline artifacts not found. Skipping delta generation.")
            print("   💡 This is expected for Stage0 (baseline creation).")
            baseline_tag = None
    
    # ============================================================
    # 4) KPI Delta 생성
    # ============================================================
    if baseline_tag:
        delta_cmd = [
            sys.executable,
            str(root / "src" / "tools" / "build_kpi_delta.py"),
            "--baseline-tag", baseline_tag,
            "--tag", stage_tag,
        ]
        
        exit_code = run_command(
            delta_cmd,
            cwd=root,
            description="Delta Report Generation"
        )
        
        if exit_code != 0:
            print(f"\n❌ Delta generation failed")
            sys.exit(exit_code)
        
        # Delta 파일 존재 확인
        delta_csv = root / "reports" / "delta" / f"delta_kpi__{baseline_tag}__vs__{stage_tag}.csv"
        delta_md = root / "reports" / "delta" / f"delta_report__{baseline_tag}__vs__{stage_tag}.md"
        
        if not check_file_exists(delta_csv, "Delta CSV"):
            print("❌ Delta CSV file missing. Exiting.")
            sys.exit(1)
        
        if not check_file_exists(delta_md, "Delta MD"):
            print("❌ Delta MD file missing. Exiting.")
            sys.exit(1)
        
        # Delta 상위 60줄 출력
        print("\n" + "="*60)
        print("Delta Summary (first 60 lines):")
        print("="*60)
        try:
            with open(delta_md, 'r', encoding='utf-8') as f:
                lines = f.readlines()[:60]
                print("".join(lines))
        except Exception as e:
            print(f"Error reading Delta MD: {e}")
    else:
        print("\n⏭️  Skipping Delta generation (baseline not available)")
    
    # ============================================================
    # 5) Manifest/Audit 생성 (선택)
    # ============================================================
    if not args.skip_manifest:
        # Manifest
        manifest_script = root / "src" / "tools" / "write_manifest.py"
        if manifest_script.exists():
            manifest_cmd = [
                sys.executable,
                str(manifest_script),
                "--config", args.config,
                "--tag", stage_tag,
            ]
            exit_code = run_command(
                manifest_cmd,
                cwd=root,
                description="Manifest Generation"
            )
            if exit_code == 0:
                manifest_json = root / "reports" / "manifests" / f"manifest__{stage_tag}.json"
                check_file_exists(manifest_json, "Manifest JSON")
        
        # Audit
        audit_script = root / "src" / "tools" / "audit_pipeline_features.py"
        if audit_script.exists():
            audit_cmd = [
                sys.executable,
                str(audit_script),
                "--config", args.config,
                "--tag", stage_tag,
            ]
            exit_code = run_command(
                audit_cmd,
                cwd=root,
                description="Audit Generation"
            )
            if exit_code == 0:
                audit_md = root / "reports" / "audit" / f"audit__{stage_tag}.md"
                audit_json = root / "reports" / "audit" / f"audit__{stage_tag}.json"
                check_file_exists(audit_md, "Audit MD")
                if audit_json.exists():
                    check_file_exists(audit_json, "Audit JSON")
    else:
        print("\n⏭️  Skipping Manifest/Audit generation (--skip-manifest)")
    
    # ============================================================
    # 6) 최종 출력 요약
    # ============================================================
    print("\n" + "="*60)
    print("FINAL OUTPUT SUMMARY")
    print("="*60)
    
    outputs = []
    
    # KPI
    if kpi_md.exists():
        outputs.append(("KPI MD", str(kpi_md.absolute())))
    if kpi_csv.exists():
        outputs.append(("KPI CSV", str(kpi_csv.absolute())))
    
    # Delta
    if baseline_tag:
        delta_csv_path = root / "reports" / "delta" / f"delta_kpi__{baseline_tag}__vs__{stage_tag}.csv"
        delta_md_path = root / "reports" / "delta" / f"delta_report__{baseline_tag}__vs__{stage_tag}.md"
        if delta_csv_path.exists():
            outputs.append(("Delta CSV", str(delta_csv_path.absolute())))
        if delta_md_path.exists():
            outputs.append(("Delta MD", str(delta_md_path.absolute())))
    
    # Manifest/Audit
    manifest_json_path = root / "reports" / "manifests" / f"manifest__{stage_tag}.json"
    audit_md_path = root / "reports" / "audit" / f"audit__{stage_tag}.md"
    audit_json_path = root / "reports" / "audit" / f"audit__{stage_tag}.json"
    
    if manifest_json_path.exists():
        outputs.append(("Manifest JSON", str(manifest_json_path.absolute())))
    if audit_md_path.exists():
        outputs.append(("Audit MD", str(audit_md_path.absolute())))
    if audit_json_path.exists():
        outputs.append(("Audit JSON", str(audit_json_path.absolute())))
    
    print("\n생성된 파일 목록 (절대경로):")
    for i, (desc, path) in enumerate(outputs, 1):
        print(f"{i}) {desc}:")
        print(f"   {path}")
    
    # 핵심 KPI 요약 (KPI CSV에서 추출)
    print("\n" + "="*60)
    print("핵심 KPI 요약 (상위 10개)")
    print("="*60)
    
    try:
        import pandas as pd
        kpi_df = pd.read_csv(kpi_csv, encoding='utf-8-sig')
        
        # 핵심 KPI 목록
        core_metrics = [
            "net_total_return", "net_sharpe", "net_mdd",
            "information_ratio", "tracking_error_ann", "avg_turnover_oneway",
            "ic_rank_mean", "cost_bps_used", "cost_bps_mismatch_flag",
            "gross_total_return"
        ]
        
        core_rows = kpi_df[kpi_df["metric"].isin(core_metrics)].head(10)
        
        if not core_rows.empty:
            print("\n| Metric | Dev Value | Holdout Value | Unit |")
            print("|---|---|---|---|")
            for _, row in core_rows.iterrows():
                metric = row["metric"]
                dev_val = row.get("dev_value", "N/A")
                holdout_val = row.get("holdout_value", "N/A")
                unit = row.get("unit", "")
                
                # 값 포맷팅
                if pd.notna(dev_val) and isinstance(dev_val, (int, float)):
                    dev_str = f"{dev_val:.4f}" if unit == "ratio" else f"{dev_val:.2f}"
                else:
                    dev_str = str(dev_val) if dev_val is not None else "N/A"
                
                if pd.notna(holdout_val) and isinstance(holdout_val, (int, float)):
                    holdout_str = f"{holdout_val:.4f}" if unit == "ratio" else f"{holdout_val:.2f}"
                else:
                    holdout_str = str(holdout_val) if holdout_val is not None else "N/A"
                
                print(f"| {metric} | {dev_str} | {holdout_str} | {unit} |")
        else:
            print("⚠️  Core KPIs not found in CSV")
    except Exception as e:
        print(f"⚠️  Error reading KPI CSV: {e}")
    
    print("\n" + "="*60)
    print("✅ ALL STEPS COMPLETED")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
