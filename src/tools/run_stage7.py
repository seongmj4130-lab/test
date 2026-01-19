# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/run_stage7.py
"""
[코드 매니저] Stage 7 실행 엔트리포인트 (랭킹 baseline 생성)
Stage 실행 → 산출물 생성 확인 → KPI 생성 → Δ 생성 → 체크리포트 → baselines.yaml 업데이트 → History Manifest 업데이트
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import yaml
from stage_runner_common import (
    generate_run_tag,
    get_baseline_tag,
    get_file_hash,
    get_stage_track,
    print_success_summary,
    run_command,
    verify_base_dir,
    verify_l2_reuse,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STAGE_NUM = 7
STAGE_NAME = "L8"  # L8 = Ranking 엔진
STAGE_TRACK = "ranking"  # Stage7은 ranking 트랙

def update_baselines_yaml(base_dir: Path, pipeline_baseline: str, ranking_baseline: str) -> None:
    """baselines.yaml 생성/업데이트"""
    baselines_path = base_dir / "reports" / "history" / "baselines.yaml"
    baselines_path.parent.mkdir(parents=True, exist_ok=True)

    baselines = {
        "pipeline_baseline_tag": pipeline_baseline,
        "ranking_baseline_tag": ranking_baseline,
        "updated_at": datetime.now().isoformat(),
    }

    with open(baselines_path, 'w', encoding='utf-8') as f:
        yaml.dump(baselines, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"[Baselines] 업데이트 완료: {baselines_path}")
    print(f"  pipeline_baseline_tag: {pipeline_baseline}")
    print(f"  ranking_baseline_tag: {ranking_baseline}")

def main():
    parser = argparse.ArgumentParser(description=f"[코드 매니저] Stage {STAGE_NUM} 실행 (랭킹 baseline 생성)")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config 파일 경로")
    parser.add_argument("--run-tag", type=str, default=None, help="Run tag (없으면 자동 생성)")
    parser.add_argument("--change-title", type=str, default="Ranking baseline 생성", help="Change title (History Manifest용)")
    parser.add_argument("--change-summary", type=str, nargs="*", default=[
        "ranking_daily 산출",
        "UI Top/Bottom snapshot 생성",
        "baseline_tag 분리 운영 시작"
    ], help="Change summary (최대 3개)")
    parser.add_argument("--modified-files", type=str, default="src/tools/run_stage7.py;src/tools/update_history_manifest.py", help="Modified files")
    parser.add_argument("--modified-functions", type=str, default="run_stage7;build_ranking_daily;update_history_manifest", help="Modified functions")
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
        run_tag = generate_run_tag("stage7_ranking_baseline")

    # Baseline 태그 결정
    baseline_tag_used = get_baseline_tag(config_path, STAGE_NUM)  # pipeline_baseline_tag
    pipeline_baseline_tag = baseline_tag_used

    # 로그 파일 설정
    logs_dir = PROJECT_ROOT / "reports" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"run__stage{STAGE_NUM}__{run_tag}.txt"
    log_file.write_text(f"Stage {STAGE_NUM} 실행 로그\n{'='*60}\n", encoding='utf-8')

    print("\n" + "="*60)
    print(f"[코드 매니저] Stage {STAGE_NUM} 실행 (랭킹 baseline 생성)")
    print("="*60)
    print(f"프로젝트 루트: {PROJECT_ROOT}")
    print(f"Run Tag: {run_tag}")
    print(f"Pipeline Baseline Tag: {pipeline_baseline_tag}")
    print(f"Baseline Tag Used (Δ 비교): {baseline_tag_used}")
    print(f"Log File: {log_file}")
    print("="*60 + "\n")

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
    # 1) Stage 실행 (L8 Ranking 엔진)
    # ============================================================
    print("\n[1/7] Stage 실행 (L8 Ranking 엔진)...")

    stage_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "run_all.py"),
        "--config", args.config,
        "--run-tag", run_tag,
        "--stage", STAGE_NAME,  # L8
        "--force-rebuild",  # skip_if_exists 무시
    ]

    if run_command(stage_cmd, PROJECT_ROOT, f"Stage {STAGE_NUM} 실행 (L8)", log_file) != 0:
        sys.exit(1)

    # L2 재사용 검증 (사후)
    l2_valid, l2_msg, _, l2_hash_after = verify_l2_reuse(PROJECT_ROOT, log_file)
    if not l2_valid:
        print(f"ERROR: {l2_msg}", file=sys.stderr)
        sys.exit(1)

    # 산출물 확인
    ranking_daily_path = PROJECT_ROOT / "data" / "interim" / run_tag / "ranking_daily.parquet"
    if not ranking_daily_path.exists():
        print(f"ERROR: ranking_daily.parquet가 생성되지 않았습니다: {ranking_daily_path}", file=sys.stderr)
        sys.exit(1)

    import pandas as pd
    ranking_daily = pd.read_parquet(ranking_daily_path)
    if len(ranking_daily) == 0:
        print(f"ERROR: ranking_daily.parquet가 비어있습니다.", file=sys.stderr)
        sys.exit(1)

    print(f"[OK] ranking_daily.parquet 생성 완료: {len(ranking_daily):,} rows")

    # ============================================================
    # 2) ranking_snapshot 생성 (최신일 기준 Top10/Bottom10)
    # ============================================================
    print("\n[2/7] ranking_snapshot 생성 중...")

    ranking_snapshot_dir = PROJECT_ROOT / "reports" / "ranking"
    ranking_snapshot_dir.mkdir(parents=True, exist_ok=True)

    # 최신일 추출
    latest_date = ranking_daily["date"].max()
    snapshot_df = ranking_daily[ranking_daily["date"] == latest_date].copy()

    # in_universe=True만 필터링
    if "in_universe" in snapshot_df.columns:
        snapshot_df = snapshot_df[snapshot_df["in_universe"]].copy()

    # rank_total 기준 정렬
    snapshot_df = snapshot_df[snapshot_df["rank_total"].notna()].copy()

    # Top10 / Bottom10
    top10 = snapshot_df.nsmallest(10, "rank_total")[["ticker", "score_total", "rank_total"]].copy()
    top10["snapshot_type"] = "top10"
    top10["snapshot_date"] = latest_date

    bottom10 = snapshot_df.nlargest(10, "rank_total")[["ticker", "score_total", "rank_total"]].copy()
    bottom10["snapshot_type"] = "bottom10"
    bottom10["snapshot_date"] = latest_date

    snapshot_result = pd.concat([top10, bottom10], ignore_index=True)
    snapshot_result = snapshot_result[["snapshot_date", "snapshot_type", "ticker", "score_total", "rank_total"]].copy()

    snapshot_path = ranking_snapshot_dir / f"ranking_snapshot__{run_tag}.csv"
    snapshot_result.to_csv(snapshot_path, index=False, encoding='utf-8-sig')
    print(f"[OK] ranking_snapshot 생성 완료: {snapshot_path}")
    print(f"   Snapshot Date: {latest_date.strftime('%Y-%m-%d')}")
    print(f"   Top10: {len(top10)} rows, Bottom10: {len(bottom10)} rows")

    # ============================================================
    # 3) KPI 생성
    # ============================================================
    print("\n[3/7] KPI 테이블 생성 중...")

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
    # 4) Δ 생성 (선택)
    # ============================================================
    print("\n[4/7] Δ 리포트 생성 중 (선택)...")

    delta_csv = PROJECT_ROOT / "reports" / "delta" / f"delta_kpi__{baseline_tag_used}__vs__{run_tag}.csv"

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
        if delta_csv.exists():
            print(f"[OK] Delta CSV 생성 완료: {delta_csv}")

    # ============================================================
    # 5) 체크리포트 생성
    # ============================================================
    print("\n[5/7] Stage 체크리포트 생성 중...")

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
    # 6) baselines.yaml 업데이트
    # ============================================================
    print("\n[6/7] baselines.yaml 업데이트 중...")

    update_baselines_yaml(PROJECT_ROOT, pipeline_baseline_tag, run_tag)

    # ============================================================
    # 7) History Manifest 업데이트
    # ============================================================
    print("\n[7/7] History Manifest 업데이트 중...")

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
        ("산출물", PROJECT_ROOT / "data" / "interim" / run_tag / "ranking_daily.parquet"),
        ("Snapshot", PROJECT_ROOT / "reports" / "ranking" / f"ranking_snapshot__{run_tag}.csv"),
        ("KPI CSV", PROJECT_ROOT / "reports" / "kpi" / f"kpi_table__{run_tag}.csv"),
        ("KPI MD", PROJECT_ROOT / "reports" / "kpi" / f"kpi_table__{run_tag}.md"),
        ("Delta CSV", PROJECT_ROOT / "reports" / "delta" / f"delta_kpi__{baseline_tag_used}__vs__{run_tag}.csv"),
        ("Delta MD", PROJECT_ROOT / "reports" / "delta" / f"delta_report__{baseline_tag_used}__vs__{run_tag}.md"),
        ("체크리포트", PROJECT_ROOT / "reports" / "stages" / f"check__stage{STAGE_NUM}__{run_tag}.md"),
        ("Baselines", PROJECT_ROOT / "reports" / "history" / "baselines.yaml"),
        ("History Manifest", PROJECT_ROOT / "reports" / "history" / "history_manifest.parquet"),
        ("로그", log_file),
    ]

    print_success_summary(run_tag, baseline_tag_used, run_tag, output_files)

if __name__ == "__main__":
    main()
