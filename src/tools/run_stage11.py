# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/run_stage11.py
"""
[코드 매니저] Stage 11 실행 엔트리포인트
UI Payload Builder + Demo Performance

Stage 실행 → KPI 생성 → Δ 생성 → 체크리포트 → History Manifest 업데이트
"""
import argparse
import sys
import yaml
from pathlib import Path
from datetime import datetime

from stage_runner_common import (
    verify_l2_reuse,
    get_baseline_tag,
    verify_base_dir,
    run_command,
    generate_run_tag,
    get_stage_track,
    print_success_summary,
    get_file_hash,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STAGE_NUM = 11
STAGE_NAME = "L11"  # L11 = UI Payload Builder
STAGE_TRACK = "ranking"  # Stage11은 ranking 트랙

def main():
    parser = argparse.ArgumentParser(description=f"[코드 매니저] Stage {STAGE_NUM} 실행 (UI Payload Builder)")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config 파일 경로")
    parser.add_argument("--run-tag", type=str, default=None, help="Run tag (없으면 자동 생성)")
    parser.add_argument("--baseline-tag", type=str, default=None, help="Baseline tag (Stage10 run_tag, 없으면 자동 감지)")
    parser.add_argument("--change-title", type=str, default="UI payload + 단순 성과곡선 생성", help="Change title (History Manifest용)")
    parser.add_argument("--change-summary", type=str, nargs="*", default=[
        "Top/Bottom daily payload",
        "Top20 시뮬 곡선 vs 벤치마크",
        "UI snapshot/metrics 저장"
    ], help="Change summary (최대 3개)")
    parser.add_argument("--modified-files", type=str, default="src/stages/ui_payload_builder.py;src/stages/ranking_demo_performance.py;src/tools/run_stage11.py", help="Modified files")
    parser.add_argument("--modified-functions", type=str, default="build_ui_payload;build_demo_equity_curves", help="Modified functions")
    args = parser.parse_args()
    
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    
    # base_dir 검증
    base_dir_valid, base_dir_msg = verify_base_dir(PROJECT_ROOT)
    if not base_dir_valid:
        print(f"ERROR: {base_dir_msg}", file=sys.stderr)
        sys.exit(1)
    
    # Run tag 생성
    if args.run_tag:
        run_tag = args.run_tag
    else:
        run_tag = generate_run_tag("stage11_ui_payload")
    
    # Baseline 태그 결정 (Stage10 run_tag 또는 ranking_baseline_tag)
    if args.baseline_tag:
        baseline_tag_used = args.baseline_tag
    else:
        # 최신 Stage10 run_tag 자동 감지 시도
        import glob
        stage10_dirs = glob.glob(str(PROJECT_ROOT / "data" / "interim" / "stage10_*"))
        if stage10_dirs:
            stage10_tags = [Path(d).name for d in stage10_dirs]
            stage10_tags.sort(reverse=True)
            baseline_tag_used = stage10_tags[0]
            print(f"[INFO] 최신 Stage10 run_tag 자동 감지: {baseline_tag_used}")
        else:
            # config에서 ranking_baseline_tag 사용
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f) or {}
            baseline_cfg = cfg.get("baseline", {})
            baseline_tag_used = baseline_cfg.get("ranking_baseline_tag")
            if not baseline_tag_used:
                # 최신 ranking_daily가 있는 run_tag 찾기
                ranking_candidates = list((PROJECT_ROOT / "data" / "interim").glob("*/ranking_daily.parquet"))
                if ranking_candidates:
                    ranking_candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    baseline_tag_used = ranking_candidates[0].parent.name
                    print(f"[INFO] 최신 ranking_daily가 있는 run_tag 자동 감지: {baseline_tag_used}")
                else:
                    print("ERROR: Stage10 run_tag를 찾을 수 없고 ranking_baseline_tag도 설정되지 않았습니다. --baseline-tag로 지정하세요.", file=sys.stderr)
                    sys.exit(1)
            else:
                print(f"[INFO] ranking_baseline_tag 사용: {baseline_tag_used}")
    
    print("\n" + "="*60)
    print(f"[코드 매니저] Stage {STAGE_NUM} 실행 (UI Payload Builder)")
    print("="*60)
    print(f"프로젝트 루트: {PROJECT_ROOT}")
    print(f"Run Tag: {run_tag}")
    print(f"Baseline Tag Used (Stage10): {baseline_tag_used}")
    print("="*60 + "\n")
    
    # 로그 파일 경로
    log_file = PROJECT_ROOT / "reports" / "logs" / f"run__stage{STAGE_NUM}__{run_tag}.txt"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 0) L2 재사용 검증 (사전)
    l2_file = PROJECT_ROOT / "data" / "interim" / "fundamentals_annual.parquet"
    if not l2_file.exists():
        print("ERROR: L2 파일이 없습니다. fundamentals_annual.parquet를 먼저 준비하세요.", file=sys.stderr)
        sys.exit(1)
    
    l2_hash_before = None
    try:
        l2_hash_full = get_file_hash(l2_file)
        l2_hash_before = l2_hash_full[:16] + "..."
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"[L2 검증] 실행 전 해시: {l2_hash_before}\n")
    except Exception as e:
        print(f"WARNING: L2 해시 계산 실패: {e}", file=sys.stderr)
    
    # ============================================================
    # 1) Stage 실행 (L11 UI Payload Builder)
    # ============================================================
    print("\n[1/6] Stage 실행 (L11 UI Payload Builder)...")
    
    stage_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "run_all.py"),
        "--config", args.config,
        "--run-tag", run_tag,
        "--stage", STAGE_NAME,  # L11
        "--force-rebuild",  # skip_if_exists 무시
        "--skip-l2",  # L2 재사용 강제
        "--baseline-tag", baseline_tag_used,  # ranking_daily 로드용
    ]
    
    if run_command(stage_cmd, PROJECT_ROOT, f"Stage {STAGE_NUM} 실행 (L11)", log_file) != 0:
        sys.exit(1)
    
    # L2 재사용 검증 (사후)
    l2_valid, l2_msg, _, l2_hash_after = verify_l2_reuse(PROJECT_ROOT, log_file)
    if not l2_valid:
        print(f"ERROR: {l2_msg}", file=sys.stderr)
        sys.exit(1)
    
    # 산출물 확인
    ui_top_bottom_path = PROJECT_ROOT / "data" / "interim" / run_tag / "ui_top_bottom_daily.parquet"
    ui_equity_path = PROJECT_ROOT / "data" / "interim" / run_tag / "ui_equity_curves.parquet"
    ui_snapshot_path_interim = PROJECT_ROOT / "data" / "interim" / run_tag / "ui_snapshot.parquet"
    ui_metrics_path_interim = PROJECT_ROOT / "data" / "interim" / run_tag / "ui_metrics.parquet"
    
    if not ui_top_bottom_path.exists():
        print(f"ERROR: ui_top_bottom_daily.parquet가 생성되지 않았습니다: {ui_top_bottom_path}", file=sys.stderr)
        sys.exit(1)
    
    if not ui_equity_path.exists():
        print(f"ERROR: ui_equity_curves.parquet가 생성되지 않았습니다: {ui_equity_path}", file=sys.stderr)
        sys.exit(1)
    
    import pandas as pd
    ui_top_bottom = pd.read_parquet(ui_top_bottom_path)
    ui_equity = pd.read_parquet(ui_equity_path)
    
    print(f"[OK] ui_top_bottom_daily.parquet 생성 완료: {len(ui_top_bottom):,} rows")
    print(f"[OK] ui_equity_curves.parquet 생성 완료: {len(ui_equity):,} rows")
    
    if ui_snapshot_path_interim.exists():
        print(f"[OK] ui_snapshot.parquet 생성 완료: {len(pd.read_parquet(ui_snapshot_path_interim)):,} rows")
    if ui_metrics_path_interim.exists():
        print(f"[OK] ui_metrics.parquet 생성 완료: {len(pd.read_parquet(ui_metrics_path_interim)):,} rows")
    
    # ============================================================
    # 2) UI 리포트 생성 (snapshot, metrics)
    # ============================================================
    print("\n[2/6] UI 리포트 생성 중...")
    
    ui_reports_dir = PROJECT_ROOT / "reports" / "ui"
    ui_reports_dir.mkdir(parents=True, exist_ok=True)
    
    # UI snapshot (최신일 기준 Top10/Bottom10)
    ui_snapshot_path = ui_reports_dir / f"ui_snapshot__{run_tag}.csv"
    ui_snapshot_path_interim = PROJECT_ROOT / "data" / "interim" / run_tag / "ui_snapshot.parquet"
    if ui_snapshot_path_interim.exists():
        ui_snapshot = pd.read_parquet(ui_snapshot_path_interim)
        ui_snapshot.to_csv(ui_snapshot_path, index=False, encoding='utf-8-sig')
        print(f"[OK] ui_snapshot.csv 생성 완료: {ui_snapshot_path} ({len(ui_snapshot)} rows)")
    else:
        print(f"WARNING: ui_snapshot.parquet가 없습니다. 건너뜁니다.", file=sys.stderr)
        ui_snapshot_path = None
    
    # UI metrics
    ui_metrics_path = ui_reports_dir / f"ui_metrics__{run_tag}.csv"
    ui_metrics_path_interim = PROJECT_ROOT / "data" / "interim" / run_tag / "ui_metrics.parquet"
    if ui_metrics_path_interim.exists():
        ui_metrics = pd.read_parquet(ui_metrics_path_interim)
        ui_metrics.to_csv(ui_metrics_path, index=False, encoding='utf-8-sig')
        print(f"[OK] ui_metrics.csv 생성 완료: {ui_metrics_path} ({len(ui_metrics)} rows)")
    else:
        print(f"WARNING: ui_metrics.parquet가 없습니다. 건너뜁니다.", file=sys.stderr)
        ui_metrics_path = None
    
    # ============================================================
    # 3) KPI 생성
    # ============================================================
    print("\n[3/6] KPI 테이블 생성 중...")
    
    kpi_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "tools" / "export_kpi_table.py"),
        "--config", args.config,
        "--tag", run_tag,
    ]
    
    if run_command(kpi_cmd, PROJECT_ROOT, "KPI 테이블 생성", log_file) != 0:
        sys.exit(1)
    
    kpi_csv = PROJECT_ROOT / "reports" / "kpi" / f"kpi_table__{run_tag}.csv"
    if not kpi_csv.exists():
        print(f"ERROR: KPI CSV가 생성되지 않았습니다: {kpi_csv}", file=sys.stderr)
        sys.exit(1)
    print(f"[OK] KPI CSV 생성 완료: {kpi_csv}")
    
    # ============================================================
    # 4) Δ 생성
    # ============================================================
    print("\n[4/6] Δ 리포트 생성 중...")
    
    delta_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "tools" / "export_delta_report.py"),
        "--config", args.config,
        "--baseline-tag", baseline_tag_used,
        "--run-tag", run_tag,
    ]
    
    if run_command(delta_cmd, PROJECT_ROOT, "Δ 리포트 생성", log_file) != 0:
        print("WARNING: Delta 리포트 생성 실패 (계속 진행)", file=sys.stderr)
    else:
        delta_csv = PROJECT_ROOT / "reports" / "delta" / f"delta_kpi__{baseline_tag_used}__vs__{run_tag}.csv"
        if delta_csv.exists():
            print(f"[OK] Delta CSV 생성 완료: {delta_csv}")
    
    # ============================================================
    # 5) 체크리포트 생성
    # ============================================================
    print("\n[5/6] Stage 체크리포트 생성 중...")
    
    check_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "tools" / "check_stage_completion.py"),
        "--config", args.config,
        "--stage", str(STAGE_NUM),
        "--run-tag", run_tag,
        "--baseline-tag", baseline_tag_used,
    ]
    
    if run_command(check_cmd, PROJECT_ROOT, "체크리포트 생성", log_file) != 0:
        print("WARNING: 체크리포트 생성 실패 (계속 진행)", file=sys.stderr)
    else:
        check_report = PROJECT_ROOT / "reports" / "stages" / f"check__stage{STAGE_NUM}__{run_tag}.md"
        if check_report.exists():
            print(f"[OK] 체크리포트 생성 완료: {check_report}")
    
    # ============================================================
    # 6) History Manifest 업데이트
    # ============================================================
    print("\n[6/6] History Manifest 업데이트 중...")
    
    history_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "tools" / "update_history_manifest.py"),
        "--config", args.config,
        "--stage", str(STAGE_NUM),
        "--track", STAGE_TRACK,
        "--run-tag", run_tag,
        "--baseline-tag", baseline_tag_used,
        "--change-title", args.change_title,
    ]
    
    if args.change_summary:
        history_cmd.extend(["--change-summary"] + args.change_summary)
    if args.modified_files:
        history_cmd.extend(["--modified-files", args.modified_files])
    if args.modified_functions:
        history_cmd.extend(["--modified-functions", args.modified_functions])
    
    if run_command(history_cmd, PROJECT_ROOT, "History Manifest 업데이트", log_file) != 0:
        print("WARNING: History Manifest 업데이트 실패 (계속 진행)", file=sys.stderr)
    else:
        history_manifest = PROJECT_ROOT / "reports" / "history" / "history_manifest.parquet"
        if history_manifest.exists():
            print(f"[OK] History Manifest 업데이트 완료: {history_manifest}")
    
    # ============================================================
    # 최종 요약
    # ============================================================
    output_files = [
        ("산출물1", PROJECT_ROOT / "data" / "interim" / run_tag / "ui_top_bottom_daily.parquet"),
        ("산출물2", PROJECT_ROOT / "data" / "interim" / run_tag / "ui_equity_curves.parquet"),
        ("UI Snapshot", ui_snapshot_path if ui_snapshot_path.exists() else None),
        ("UI Metrics", ui_metrics_path if ui_metrics_path.exists() else None),
        ("KPI CSV", PROJECT_ROOT / "reports" / "kpi" / f"kpi_table__{run_tag}.csv"),
        ("KPI MD", PROJECT_ROOT / "reports" / "kpi" / f"kpi_table__{run_tag}.md"),
        ("Delta CSV", PROJECT_ROOT / "reports" / "delta" / f"delta_kpi__{baseline_tag_used}__vs__{run_tag}.csv"),
        ("Delta MD", PROJECT_ROOT / "reports" / "delta" / f"delta_report__{baseline_tag_used}__vs__{run_tag}.md"),
        ("체크리포트", PROJECT_ROOT / "reports" / "stages" / f"check__stage{STAGE_NUM}__{run_tag}.md"),
        ("History Manifest", PROJECT_ROOT / "reports" / "history" / "history_manifest.parquet"),
        ("로그", log_file),
    ]
    
    # None 제거
    output_files = [(desc, path) for desc, path in output_files if path is not None]
    
    print_success_summary(run_tag, baseline_tag_used, None, output_files)

if __name__ == "__main__":
    main()
