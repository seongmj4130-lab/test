# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/pipeline/stage_manager.py
"""
[코드 매니저] Stage 실행 통합 스크립트
- Stage 실행 → 산출물 생성 확인 → KPI 생성 → Δ 리포트 생성 → 체크리스트 리포트 생성
- 모든 산출물은 run_tag 기준으로 새로 생성 (L2 제외)
- skip_if_exists 무시 (L2 제외)
"""
import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# 고정 설정
PROJECT_ROOT = Path(r"C:\Users\seong\OneDrive\Desktop\bootcamp\03_code")
BASELINE_TAG = (
    "stage6_sector_relative_feature_balance_20251220_194928"  # 2차 사이클 기준
)
HISTORICAL_BASELINE_TAG = "baseline_prerefresh_20251219_143636"  # 참조용


def generate_run_tag(stage_name: str) -> str:
    """
    run_tag 생성: stage명 + 타임스탬프
    예: stage7_rank_engine_20251220_194928
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{stage_name}_{timestamp}"


def run_command(
    cmd: list[str], cwd: Path, description: str, check_returncode: bool = True
) -> tuple[int, str, str]:
    """
    명령어 실행 및 결과 반환

    Returns:
        (returncode, stdout, stderr)
    """
    print(f"\n{'='*60}")
    print(f"[{description}]")
    print(f"Command: {' '.join(cmd)}")
    print(f"Working Directory: {cwd}")
    print(f"{'='*60}\n")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)

        if check_returncode and result.returncode != 0:
            print(f"\n[FAIL] [{description}] Failed with exit code {result.returncode}")
        else:
            print(f"\n[OK] [{description}] Completed")

        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        print(f"\n[ERROR] [{description}] Exception: {e}")
        return 1, "", str(e)


def verify_artifact_exists(file_path: Path, description: str) -> bool:
    """산출물 파일 존재 확인"""
    if file_path.exists():
        file_size = file_path.stat().st_size
        print(f"[OK] {description}: {file_path} ({file_size:,} bytes)")
        return True
    else:
        print(f"[MISSING] {description}: NOT FOUND - {file_path}")
        return False


def verify_interim_artifacts(
    run_tag: str, stage: Optional[int], base_interim_dir: Path
) -> tuple[bool, list[str]]:
    """
    Stage별 필수 산출물 존재 확인

    Returns:
        (all_exist, missing_list)
    """
    interim_dir = base_interim_dir / run_tag

    # Stage별 필수 산출물 정의
    stage_artifacts = {
        0: ["universe_k200_membership_monthly"],
        1: ["ohlcv_daily"],
        2: ["fundamentals_annual"],  # base_interim_dir에 있음
        3: ["panel_merged_daily"],
        4: ["dataset_daily", "cv_folds_short", "cv_folds_long"],
        5: ["pred_short_oos", "pred_long_oos", "model_metrics"],
        6: ["rebalance_scores", "rebalance_scores_summary"],
        7: ["bt_positions", "bt_returns", "bt_equity_curve", "bt_metrics"],
    }

    if stage is None:
        # 모든 Stage 산출물 확인
        required = []
        for artifacts in stage_artifacts.values():
            required.extend(artifacts)
    else:
        required = stage_artifacts.get(stage, [])

    missing = []
    for artifact_name in required:
        if artifact_name == "fundamentals_annual":
            # L2는 base_interim_dir에 있음
            artifact_path = base_interim_dir / f"{artifact_name}.parquet"
        else:
            artifact_path = interim_dir / f"{artifact_name}.parquet"

        if not artifact_path.exists():
            missing.append(str(artifact_path))
            print(f"❌ Missing: {artifact_name} - {artifact_path}")
        else:
            file_size = artifact_path.stat().st_size
            print(f"✅ Found: {artifact_name} - {artifact_path} ({file_size:,} bytes)")

    return len(missing) == 0, missing


def main():
    parser = argparse.ArgumentParser(
        description="[코드 매니저] Stage 실행 통합 스크립트"
    )
    parser.add_argument(
        "--stage-name",
        type=str,
        required=True,
        help="Stage 이름 (예: stage7_rank_engine, stage0_rebuild_tagged)",
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Config 파일 경로"
    )
    parser.add_argument(
        "--from-stage", type=str, default="L0", help="시작 Stage (기본: L0)"
    )
    parser.add_argument(
        "--to-stage", type=str, default="L7D", help="종료 Stage (기본: L7D)"
    )
    parser.add_argument(
        "--stage",
        type=str,
        default=None,
        help="단일 Stage 실행 (--from-stage, --to-stage 무시)",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        default=True,
        help="강제 Rebuild: 기존 산출물 무시하고 새로 생성 (L2 제외, 기본: True)",
    )
    parser.add_argument(
        "--baseline-tag",
        type=str,
        default=BASELINE_TAG,
        help=f"Baseline 태그 (기본: {BASELINE_TAG})",
    )
    parser.add_argument(
        "--skip-pipeline",
        action="store_true",
        help="파이프라인 실행 건너뛰기 (리포트만 생성)",
    )
    parser.add_argument("--skip-kpi", action="store_true", help="KPI 생성 건너뛰기")
    parser.add_argument(
        "--skip-delta", action="store_true", help="Delta 리포트 생성 건너뛰기"
    )
    parser.add_argument(
        "--skip-check", action="store_true", help="체크리스트 리포트 생성 건너뛰기"
    )
    args = parser.parse_args()

    # run_tag 생성
    run_tag = generate_run_tag(args.stage_name)
    baseline_tag = args.baseline_tag

    print("\n" + "=" * 60)
    print("[코드 매니저] Stage 실행 통합 스크립트")
    print("=" * 60)
    print(f"프로젝트 루트: {PROJECT_ROOT}")
    print(f"Run Tag: {run_tag}")
    print(f"Baseline Tag: {baseline_tag}")
    print(f"Config: {args.config}")
    print(f"Force Rebuild: {args.force_rebuild}")
    print("=" * 60 + "\n")

    if not PROJECT_ROOT.exists():
        print(f"❌ 프로젝트 루트가 존재하지 않습니다: {PROJECT_ROOT}")
        sys.exit(1)

    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"❌ Config 파일이 존재하지 않습니다: {config_path}")
        sys.exit(1)

    base_interim_dir = PROJECT_ROOT / "data" / "interim"
    base_interim_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 1) Stage 실행 (파이프라인)
    # ============================================================
    if not args.skip_pipeline:
        print("\n[1/5] Stage 파이프라인 실행 중...")

        pipeline_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "src" / "run_all.py"),
            "--config",
            args.config,
            "--run-tag",
            run_tag,
            "--baseline-tag",
            baseline_tag,
            "--force-rebuild",  # 항상 force-rebuild (skip_if_exists 무시)
        ]

        if args.stage:
            pipeline_cmd.extend(["--stage", args.stage.upper()])
        else:
            pipeline_cmd.extend(
                ["--from", args.from_stage.upper(), "--to", args.to_stage.upper()]
            )

        returncode, stdout, stderr = run_command(
            pipeline_cmd,
            cwd=PROJECT_ROOT,
            description="Stage 파이프라인 실행",
            check_returncode=True,
        )

        if returncode != 0:
            print(f"\n❌ 파이프라인 실행 실패 (exit code: {returncode})")
            print(f"stdout:\n{stdout}")
            print(f"stderr:\n{stderr}")
            sys.exit(returncode)

        # 산출물 존재 확인
        stage_num = None
        if args.stage:
            # Stage 번호 추출 (L0 -> 0, L7 -> 7)
            try:
                stage_num = int(args.stage.upper().replace("L", ""))
            except:
                pass

        print("\n[1/5] 산출물 존재 확인 중...")
        all_exist, missing = verify_interim_artifacts(
            run_tag, stage_num, base_interim_dir
        )

        if not all_exist:
            print("\n⚠️  일부 산출물이 누락되었습니다:")
            for m in missing:
                print(f"   - {m}")
            print("\n⚠️  계속 진행하지만, KPI/Delta 리포트 생성이 실패할 수 있습니다.")
        else:
            print("\n✅ 모든 필수 산출물이 생성되었습니다.")
    else:
        print("\n⏭️  파이프라인 실행 건너뛰기 (--skip-pipeline)")

    # ============================================================
    # 2) KPI 테이블 생성
    # ============================================================
    if not args.skip_kpi:
        print("\n[2/5] KPI 테이블 생성 중...")

        kpi_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "src" / "tools" / "export_kpi_table.py"),
            "--config",
            args.config,
            "--tag",
            run_tag,
        ]

        returncode, stdout, stderr = run_command(
            kpi_cmd,
            cwd=PROJECT_ROOT,
            description="KPI 테이블 생성",
            check_returncode=True,
        )

        if returncode != 0:
            print(f"\n❌ KPI 테이블 생성 실패 (exit code: {returncode})")
            sys.exit(returncode)

        # KPI 파일 존재 확인
        kpi_csv = PROJECT_ROOT / "reports" / "kpi" / f"kpi_table__{run_tag}.csv"
        kpi_md = PROJECT_ROOT / "reports" / "kpi" / f"kpi_table__{run_tag}.md"

        if not verify_artifact_exists(kpi_csv, "KPI CSV"):
            print("❌ KPI CSV 파일이 생성되지 않았습니다.")
            sys.exit(1)

        if not verify_artifact_exists(kpi_md, "KPI MD"):
            print("❌ KPI MD 파일이 생성되지 않았습니다.")
            sys.exit(1)
    else:
        print("\n⏭️  KPI 테이블 생성 건너뛰기 (--skip-kpi)")

    # ============================================================
    # 3) Δ 리포트 생성
    # ============================================================
    if not args.skip_delta:
        print("\n[3/5] Δ 리포트 생성 중...")

        # Baseline KPI 확인
        baseline_kpi_csv = (
            PROJECT_ROOT / "reports" / "kpi" / f"kpi_table__{baseline_tag}.csv"
        )

        if not baseline_kpi_csv.exists():
            print(f"\n⚠️  Baseline KPI가 없습니다: {baseline_kpi_csv}")
            print("   Baseline KPI를 먼저 생성합니다...")

            baseline_kpi_cmd = [
                sys.executable,
                str(PROJECT_ROOT / "src" / "tools" / "export_kpi_table.py"),
                "--config",
                args.config,
                "--tag",
                baseline_tag,
            ]

            returncode, _, _ = run_command(
                baseline_kpi_cmd,
                cwd=PROJECT_ROOT,
                description="Baseline KPI 생성",
                check_returncode=False,  # 실패해도 계속 진행
            )

            if returncode != 0:
                print("⚠️  Baseline KPI 생성 실패. Delta 리포트 생성을 건너뜁니다.")
                baseline_tag = None

        if baseline_tag:
            delta_cmd = [
                sys.executable,
                str(PROJECT_ROOT / "src" / "tools" / "export_delta_report.py"),
                "--baseline-tag",
                baseline_tag,
                "--run-tag",
                run_tag,
            ]

            returncode, stdout, stderr = run_command(
                delta_cmd,
                cwd=PROJECT_ROOT,
                description="Δ 리포트 생성",
                check_returncode=True,
            )

            if returncode != 0:
                print(f"\n❌ Δ 리포트 생성 실패 (exit code: {returncode})")
                sys.exit(returncode)

            # Delta 파일 존재 확인
            delta_csv = (
                PROJECT_ROOT
                / "reports"
                / "delta"
                / f"delta_kpi__{baseline_tag}__vs__{run_tag}.csv"
            )
            delta_md = (
                PROJECT_ROOT
                / "reports"
                / "delta"
                / f"delta_report__{baseline_tag}__vs__{run_tag}.md"
            )

            if not verify_artifact_exists(delta_csv, "Delta CSV"):
                print("❌ Delta CSV 파일이 생성되지 않았습니다.")
                sys.exit(1)

            if not verify_artifact_exists(delta_md, "Delta MD"):
                print("❌ Delta MD 파일이 생성되지 않았습니다.")
                sys.exit(1)
    else:
        print("\n⏭️  Δ 리포트 생성 건너뛰기 (--skip-delta)")

    # ============================================================
    # 4) [Stage8] 섹터 농도 계산 (L8 실행 시)
    # ============================================================
    stage_num = None
    if args.stage:
        try:
            stage_num = int(args.stage.upper().replace("L", ""))
        except:
            pass
    elif args.from_stage:
        try:
            stage_num = int(args.from_stage.upper().replace("L", ""))
        except:
            pass

    if stage_num == 8 or (args.from_stage and "L8" in args.from_stage.upper()):
        print("\n[4/6] [Stage8] 섹터 농도 계산 중...")

        sector_concentration_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "src" / "tools" / "calculate_sector_concentration.py"),
            "--config",
            args.config,
            "--run-tag",
            run_tag,
            "--baseline-tag",
            baseline_tag,
            "--top-k",
            "20",
        ]

        returncode, stdout, stderr = run_command(
            sector_concentration_cmd,
            cwd=PROJECT_ROOT,
            description="[Stage8] 섹터 농도 계산",
            check_returncode=False,  # 실패해도 계속 진행
        )

        if returncode == 0:
            sector_concentration_csv = (
                PROJECT_ROOT
                / "reports"
                / "ranking"
                / f"sector_concentration__{run_tag}.csv"
            )
            verify_artifact_exists(sector_concentration_csv, "[Stage8] 섹터 농도 CSV")
        else:
            print("⚠️  섹터 농도 계산 실패 (계속 진행)")

    # ============================================================
    # 5) Stage 체크리스트 리포트 생성
    # ============================================================
    if not args.skip_check:
        print("\n[5/6] Stage 체크리스트 리포트 생성 중...")

        if stage_num is not None:
            check_cmd = [
                sys.executable,
                str(PROJECT_ROOT / "src" / "tools" / "check_stage_completion.py"),
                "--config",
                args.config,
                "--run-tag",
                run_tag,
                "--stage",
                str(stage_num),
                "--baseline-tag",
                baseline_tag,
            ]

            returncode, stdout, stderr = run_command(
                check_cmd,
                cwd=PROJECT_ROOT,
                description="Stage 체크리스트 리포트 생성",
                check_returncode=False,  # 실패해도 계속 진행
            )

            if returncode == 0:
                check_report = (
                    PROJECT_ROOT
                    / "reports"
                    / "stages"
                    / f"check__stage{stage_num}__{run_tag}.md"
                )
                verify_artifact_exists(check_report, "체크리스트 리포트")
            else:
                print("⚠️  체크리스트 리포트 생성 실패 (계속 진행)")
        else:
            print("⚠️  Stage 번호를 추출할 수 없어 체크리스트 리포트 생성을 건너뜁니다.")
    else:
        print("\n⏭️  체크리스트 리포트 생성 건너뛰기 (--skip-check)")

    # ============================================================
    # 6) 최종 요약 출력
    # ============================================================
    print("\n" + "=" * 60)
    print("[6/6] 최종 요약")
    print("=" * 60)

    outputs = []

    # 산출물 경로
    interim_dir = base_interim_dir / run_tag
    if interim_dir.exists():
        parquet_files = list(interim_dir.glob("*.parquet"))
        csv_files = list(interim_dir.glob("*.csv"))
        outputs.append(
            (
                "산출물 (interim)",
                f"{interim_dir} ({len(parquet_files)} parquet, {len(csv_files)} csv)",
            )
        )

    # KPI 리포트
    kpi_csv = PROJECT_ROOT / "reports" / "kpi" / f"kpi_table__{run_tag}.csv"
    kpi_md = PROJECT_ROOT / "reports" / "kpi" / f"kpi_table__{run_tag}.md"
    if kpi_csv.exists():
        outputs.append(("KPI CSV", str(kpi_csv)))
    if kpi_md.exists():
        outputs.append(("KPI MD", str(kpi_md)))

    # Delta 리포트
    delta_csv = (
        PROJECT_ROOT
        / "reports"
        / "delta"
        / f"delta_kpi__{baseline_tag}__vs__{run_tag}.csv"
    )
    delta_md = (
        PROJECT_ROOT
        / "reports"
        / "delta"
        / f"delta_report__{baseline_tag}__vs__{run_tag}.md"
    )
    if delta_csv.exists():
        outputs.append(("Delta CSV", str(delta_csv)))
    if delta_md.exists():
        outputs.append(("Delta MD", str(delta_md)))

    # 체크리스트 리포트
    if stage_num is not None:
        check_report = (
            PROJECT_ROOT
            / "reports"
            / "stages"
            / f"check__stage{stage_num}__{run_tag}.md"
        )
        if check_report.exists():
            outputs.append(("체크리스트 리포트", str(check_report)))

    # [Stage8] 섹터 농도 리포트
    if stage_num == 8:
        sector_concentration_csv = (
            PROJECT_ROOT
            / "reports"
            / "ranking"
            / f"sector_concentration__{run_tag}.csv"
        )
        if sector_concentration_csv.exists():
            outputs.append(("[Stage8] 섹터 농도 CSV", str(sector_concentration_csv)))

    # [Stage8] 섹터 농도 리포트
    if stage_num == 8:
        sector_concentration_csv = (
            PROJECT_ROOT
            / "reports"
            / "ranking"
            / f"sector_concentration__{run_tag}.csv"
        )
        if sector_concentration_csv.exists():
            outputs.append(("[Stage8] 섹터 농도 CSV", str(sector_concentration_csv)))

    print("\n생성된 파일 목록:")
    for i, (desc, path) in enumerate(outputs, 1):
        print(f"{i}) {desc}:")
        print(f"   {path}")

    print("\n" + "=" * 60)
    print("✅ 모든 단계 완료")
    print("=" * 60)
    print(f"\nRun Tag: {run_tag}")
    print(f"Baseline Tag: {baseline_tag}")
    print(f"\n생성된 파일 수: {len(outputs)}개")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
