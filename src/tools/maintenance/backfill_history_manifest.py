# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/maintenance/backfill_history_manifest.py
"""
History Manifest 백필 스크립트
이미 생성된 baseline/stage0~6(+stage8)의 KPI/Δ/랭킹 산출물을 읽어서
history_manifest를 백필(backfill)합니다.

중요 규칙:
- 산출물 재생성 금지 (리포트 집계만 수행)
- L2 재무는 재사용 고정: data/interim/fundamentals_annual.parquet 해시를 매번 기록
- 누락 파일은 FAIL이 아니라 NA로 채우고 gate_notes에 MISSING 경로 기록
"""
import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 백필할 태그 목록 (하드코딩 금지: 이번 백필용 목록)
BACKFILL_TAGS = [
    {
        "run_tag": "baseline_prerefresh_20251219_143636",
        "stage_no": -1,  # baseline은 -1로 표시
        "track": "pipeline",
        "baseline_tag": "baseline_prerefresh_20251219_143636",
        "change_title": "Baseline (pre-refresh)",
        "change_summary": ["초기 baseline 설정"],
    },
    {
        "run_tag": "stage0_rebuild_tagged_20251219_220938",
        "stage_no": 0,
        "track": "pipeline",
        "baseline_tag": "baseline_prerefresh_20251219_143636",
        "change_title": "Stage0: Rebuild Tagged",
        "change_summary": ["태그 기반 산출물 관리 도입"],
    },
    {
        "run_tag": "stage1_cost_model_fix_20251219_221942",
        "stage_no": 1,
        "track": "pipeline",
        "baseline_tag": "baseline_prerefresh_20251219_143636",
        "change_title": "Stage1: Cost Model Fix",
        "change_summary": ["거래비용 모델 수정"],
    },
    {
        "run_tag": "stage2_explainability_20251219_224241",
        "stage_no": 2,
        "track": "pipeline",
        "baseline_tag": "baseline_prerefresh_20251219_143636",
        "change_title": "Stage2: Explainability",
        "change_summary": ["피처 중요도 분석 추가"],
    },
    {
        "run_tag": "stage3_alpha_tuning_20251220_182453",
        "stage_no": 3,
        "track": "pipeline",
        "baseline_tag": "baseline_prerefresh_20251219_143636",
        "change_title": "Stage3: Alpha Tuning",
        "change_summary": ["Ridge alpha 튜닝 추가"],
    },
    {
        "run_tag": "stage4_sector_diversify_20251220_184214",
        "stage_no": 4,
        "track": "pipeline",
        "baseline_tag": "baseline_prerefresh_20251219_143636",
        "change_title": "Stage4: Sector Diversification",
        "change_summary": ["업종 분산 제약 추가"],
    },
    {
        "run_tag": "stage5_regime_switch_20251220_193618",
        "stage_no": 5,
        "track": "pipeline",
        "baseline_tag": "baseline_prerefresh_20251219_143636",
        "change_title": "Stage5: Regime Switch",
        "change_summary": ["시장 국면 기반 전략 추가"],
    },
    {
        "run_tag": "stage6_sector_relative_feature_balance_20251220_194928",
        "stage_no": 6,
        "track": "pipeline",
        "baseline_tag": "baseline_prerefresh_20251219_143636",
        "change_title": "Stage6: Sector Relative Feature Balance",
        "change_summary": ["섹터 상대 피처 밸런스 조정"],
    },
    {
        "run_tag": "stage8_sector_relative_20251220_212625",
        "stage_no": 8,
        "track": "ranking",
        "baseline_tag": "stage6_sector_relative_feature_balance_20251220_194928",  # 임시(비교기준 변경 예정)
        "change_title": "Stage8: Sector Relative Ranking",
        "change_summary": ["섹터 상대 랭킹 엔진"],
    },
]


def check_file_exists(filepath: Path) -> tuple[bool, str]:
    """파일 존재 여부 확인 및 경로 반환"""
    exists = filepath.exists()
    return exists, str(filepath) if exists else None


def collect_missing_files(
    base_dir: Path, run_tag: str, stage_no: int, track: str, baseline_tag: str
) -> list[str]:
    """누락된 파일 목록 수집"""
    missing = []

    # KPI CSV
    kpi_csv = base_dir / "reports" / "kpi" / f"kpi_table__{run_tag}.csv"
    if not kpi_csv.exists():
        missing.append(f"MISSING: {kpi_csv}")

    # Delta CSV (baseline은 자기 자신과 비교하지 않음)
    if stage_no >= 0 and baseline_tag:
        delta_csv = (
            base_dir
            / "reports"
            / "delta"
            / f"delta_kpi__{baseline_tag}__vs__{run_tag}.csv"
        )
        if not delta_csv.exists():
            missing.append(f"MISSING: {delta_csv}")

    # 랭킹 파일 (Stage7+)
    if stage_no >= 7:
        ranking_parquet = (
            base_dir / "data" / "interim" / run_tag / "ranking_daily.parquet"
        )
        if not ranking_parquet.exists():
            missing.append(f"MISSING: {ranking_parquet}")

        sector_csv = (
            base_dir / "reports" / "ranking" / f"sector_concentration__{run_tag}.csv"
        )
        if not sector_csv.exists():
            missing.append(f"MISSING: {sector_csv}")

    return missing


def run_update_history_manifest(
    base_dir: Path,
    config_path: Path,
    run_tag: str,
    stage_no: int,
    track: str,
    baseline_tag: str,
    change_title: str = None,
    change_summary: list[str] = None,
    modified_files: str = None,
    modified_functions: str = None,
) -> tuple[bool, list[str]]:
    """update_history_manifest.py 실행"""
    cmd = [
        sys.executable,
        str(base_dir / "src" / "tools" / "update_history_manifest.py"),
        "--config",
        str(config_path.relative_to(base_dir)),
        "--stage",
        str(stage_no),
        "--track",
        track,
        "--run-tag",
        run_tag,
    ]

    if baseline_tag:
        cmd.extend(["--baseline-tag", baseline_tag])

    if change_title:
        cmd.extend(["--change-title", change_title])

    if change_summary:
        cmd.extend(["--change-summary"] + change_summary)

    if modified_files:
        cmd.extend(["--modified-files", modified_files])

    if modified_functions:
        cmd.extend(["--modified-functions", modified_functions])

    print(f"\n[백필] {run_tag} 처리 중...")
    print(f"  Stage: {stage_no}, Track: {track}, Baseline: {baseline_tag}")

    # 누락 파일 확인
    missing = collect_missing_files(base_dir, run_tag, stage_no, track, baseline_tag)
    if missing:
        print(f"  [WARN] 누락 파일: {len(missing)}개")
        for m in missing:
            print(f"     - {m}")
    else:
        print("  [OK] 모든 필수 파일 존재")

    # 실행
    result = subprocess.run(
        cmd,
        cwd=str(base_dir),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    if result.returncode != 0:
        print(f"  [FAIL] 실패 (exit code: {result.returncode})")
        if result.stderr:
            try:
                stderr_text = (
                    result.stderr[:500]
                    .encode("utf-8", errors="replace")
                    .decode("utf-8", errors="replace")
                )
                print(f"  stderr: {stderr_text}")
            except:
                print("  stderr: (인코딩 오류로 표시 불가)")
        return False, missing

    print("  [OK] 완료")
    return True, missing


def main():
    parser = argparse.ArgumentParser(description="History Manifest 백필 스크립트")
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Config 파일 경로"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="실제 실행 없이 누락 파일만 확인"
    )
    args = parser.parse_args()

    base_dir = PROJECT_ROOT
    config_path = base_dir / args.config

    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    # base_dir 확인
    import yaml

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    config_base_dir = Path(cfg.get("paths", {}).get("base_dir", ""))
    expected_base_dir = Path("C:/Users/seong/OneDrive/Desktop/bootcamp/03_code")

    if str(config_base_dir).replace("\\", "/") != str(expected_base_dir).replace(
        "\\", "/"
    ):
        print("WARNING: config.yaml의 base_dir이 예상과 다릅니다:")
        print(f"  예상: {expected_base_dir}")
        print(f"  실제: {config_base_dir}")
        print("  계속 진행합니다...")

    # reports/history/ 폴더 생성
    history_dir = base_dir / "reports" / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    print(f"[백필] History 디렉토리: {history_dir}")

    # L2 파일 확인
    l2_file = base_dir / "data" / "interim" / "fundamentals_annual.parquet"
    if not l2_file.exists():
        print(f"WARNING: L2 파일이 없습니다: {l2_file}")
        print("  L2 해시는 기록되지 않습니다.")
    else:
        print(f"[백필] L2 파일 확인: {l2_file}")

    print("\n" + "=" * 60)
    print(f"[백필] 총 {len(BACKFILL_TAGS)}개 태그 처리 시작")
    print("=" * 60)

    success_count = 0
    fail_count = 0
    all_missing = {}

    for i, tag_info in enumerate(BACKFILL_TAGS, 1):
        run_tag = tag_info["run_tag"]
        stage_no = tag_info["stage_no"]
        track = tag_info["track"]
        baseline_tag = tag_info.get("baseline_tag")
        change_title = tag_info.get("change_title")
        change_summary = tag_info.get("change_summary", [])

        print(f"\n[{i}/{len(BACKFILL_TAGS)}] {run_tag}")

        if args.dry_run:
            # Dry-run: 누락 파일만 확인
            missing = collect_missing_files(
                base_dir, run_tag, stage_no, track, baseline_tag
            )
            if missing:
                all_missing[run_tag] = missing
                print(f"  [WARN] 누락 파일: {len(missing)}개")
            else:
                print("  [OK] 모든 필수 파일 존재")
        else:
            # 실제 실행
            success, missing = run_update_history_manifest(
                base_dir=base_dir,
                config_path=config_path,
                run_tag=run_tag,
                stage_no=stage_no,
                track=track,
                baseline_tag=baseline_tag,
                change_title=change_title,
                change_summary=change_summary,
            )

            if success:
                success_count += 1
            else:
                fail_count += 1

            if missing:
                all_missing[run_tag] = missing

    # 최종 요약
    print("\n" + "=" * 60)
    print("[백필] 완료 요약")
    print("=" * 60)

    if args.dry_run:
        print(f"처리한 run_tag 개수: {len(BACKFILL_TAGS)}")
        print(f"누락 파일이 있는 태그: {len(all_missing)}개")
        if all_missing:
            print("\n누락 파일 요약:")
            for run_tag, missing_list in all_missing.items():
                print(f"  {run_tag}: {len(missing_list)}개")
                for m in missing_list[:3]:  # 최대 3개만 표시
                    print(f"    - {m}")
                if len(missing_list) > 3:
                    print(f"    ... 외 {len(missing_list) - 3}개")
    else:
        print(f"처리한 run_tag 개수: {len(BACKFILL_TAGS)}")
        print(f"성공: {success_count}개")
        print(f"실패: {fail_count}개")
        print(f"누락 파일이 있는 태그: {len(all_missing)}개")

        if all_missing:
            print("\n누락 파일 요약:")
            for run_tag, missing_list in all_missing.items():
                print(f"  {run_tag}: {len(missing_list)}개")
                for m in missing_list[:3]:  # 최대 3개만 표시
                    print(f"    - {m}")
                if len(missing_list) > 3:
                    print(f"    ... 외 {len(missing_list) - 3}개")

        # History Manifest 파일 확인
        manifest_parquet = history_dir / "history_manifest.parquet"
        manifest_csv = history_dir / "history_manifest.csv"
        manifest_md = history_dir / "history_manifest.md"
        timeline_csv = history_dir / "history_timeline_ppt.csv"

        print("\n생성된 파일:")
        for fpath, desc in [
            (manifest_parquet, "history_manifest.parquet"),
            (manifest_csv, "history_manifest.csv"),
            (manifest_md, "history_manifest.md"),
            (timeline_csv, "history_timeline_ppt.csv"),
        ]:
            if fpath.exists():
                size = fpath.stat().st_size
                print(f"  [OK] {desc} ({size:,} bytes)")
            else:
                print(f"  [MISSING] {desc} (생성되지 않음)")

        # 최종 row 수 확인
        if manifest_csv.exists():
            import pandas as pd

            try:
                df = pd.read_csv(manifest_csv)
                print(f"\n최종 row 수: {len(df)}")
            except Exception as e:
                print(f"\n[WARN] CSV 읽기 실패: {e}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
