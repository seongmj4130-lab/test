# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/validation/check_stage0_stage1_completion.py
"""
Stage0/Stage1 완료 점검 스크립트
"""
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

# 프로젝트 루트
BASE_DIR = Path(r"C:\Users\seong\OneDrive\Desktop\bootcamp\03_code")
BASELINE_TAG = "baseline_prerefresh_20251219_143636"

# 필수 산출물 파일 목록
REQUIRED_ARTIFACTS = [
    "universe_k200_membership_monthly.parquet",
    "ohlcv_daily.parquet",
    "panel_merged_daily.parquet",
    "dataset_daily.parquet",
    "model_metrics.parquet",
    "rebalance_scores_summary.parquet",
    "bt_metrics.parquet",
    "bt_returns.parquet",
    "bt_benchmark_compare.parquet",
    "bt_yearly_metrics.parquet",
    "bt_rolling_sharpe.parquet",
    "bt_drawdown_events.parquet",
]

# 필수 리포트 파일 목록
REQUIRED_REPORTS = [
    ("kpi", "kpi_table__{tag}.csv"),
    ("kpi", "kpi_table__{tag}.md"),
    ("delta", "delta_kpi__{baseline_tag}__vs__{tag}.csv"),
    ("delta", "delta_report__{baseline_tag}__vs__{tag}.md"),
]


def find_stage_tags() -> tuple[Optional[str], Optional[str]]:
    """Stage0/Stage1 태그 자동 탐지"""
    interim_dir = BASE_DIR / "data" / "interim"

    if not interim_dir.exists():
        return None, None

    stage0_tags = []
    stage1_tags = []

    # 디렉토리명 정규식 매칭
    pattern0 = re.compile(r"^stage0_.*")
    pattern1 = re.compile(r"^stage1_.*")

    for item in interim_dir.iterdir():
        if item.is_dir():
            name = item.name
            if pattern0.match(name):
                mtime = item.stat().st_mtime
                stage0_tags.append((name, mtime))
            elif pattern1.match(name):
                mtime = item.stat().st_mtime
                stage1_tags.append((name, mtime))

    # 최신 순으로 정렬
    stage0_tags.sort(key=lambda x: x[1], reverse=True)
    stage1_tags.sort(key=lambda x: x[1], reverse=True)

    stage0_tag = stage0_tags[0][0] if stage0_tags else None
    stage1_tag = stage1_tags[0][0] if stage1_tags else None

    return stage0_tag, stage1_tag


def check_file_exists(filepath: Path) -> bool:
    """파일 존재 여부 확인"""
    return filepath.exists() and filepath.is_file()


def check_reports(tag: str, baseline_tag: str) -> dict[str, bool]:
    """필수 리포트 파일 존재 체크"""
    results = {}

    reports_dir = BASE_DIR / "reports"

    for subdir, filename_template in REQUIRED_REPORTS:
        filename = filename_template.format(tag=tag, baseline_tag=baseline_tag)
        filepath = reports_dir / subdir / filename
        results[filename] = check_file_exists(filepath)

    return results


def check_artifacts(tag: str) -> dict[str, bool]:
    """필수 산출물(parquet) 존재 체크"""
    results = {}
    interim_dir = BASE_DIR / "data" / "interim" / tag

    for filename in REQUIRED_ARTIFACTS:
        filepath = interim_dir / filename
        results[filename] = check_file_exists(filepath)

    return results


def check_l2_reuse() -> dict[str, any]:
    """L2 재무 재사용 규칙 체크"""
    results = {
        "root_file_exists": False,
        "root_file_path": None,
    }

    root_file = BASE_DIR / "data" / "interim" / "fundamentals_annual.parquet"

    if check_file_exists(root_file):
        results["root_file_exists"] = True
        results["root_file_path"] = str(root_file)

    return results


def check_legacy_root_save() -> dict[str, any]:
    """legacy-root-save 감지"""
    results = {
        "suspected": False,
        "root_artifacts_count": 0,
        "missing_tags": [],
    }

    interim_root = BASE_DIR / "data" / "interim"

    # 루트에 parquet 파일 개수 확인
    root_parquets = list(interim_root.glob("*.parquet"))
    results["root_artifacts_count"] = len(root_parquets)

    # 필수 파일들이 루트에 있는지 확인
    required_in_root = 0
    for filename in REQUIRED_ARTIFACTS:
        if check_file_exists(interim_root / filename):
            required_in_root += 1

    # stage0/stage1 태그가 없고 루트에 파일이 많으면 의심
    stage0_tag, stage1_tag = find_stage_tags()

    if (stage0_tag is None or stage1_tag is None) and required_in_root >= 5:
        results["suspected"] = True
        if stage0_tag is None:
            results["missing_tags"].append("stage0")
        if stage1_tag is None:
            results["missing_tags"].append("stage1")

    return results


def format_markdown_report(
    stage0_tag: Optional[str],
    stage1_tag: Optional[str],
    stage0_reports: dict[str, bool],
    stage1_reports: dict[str, bool],
    stage0_artifacts: dict[str, bool],
    stage1_artifacts: dict[str, bool],
    l2_check: dict[str, any],
    legacy_check: dict[str, any],
) -> str:
    """Markdown 형식 보고서 생성"""
    lines = []

    # 헤더
    lines.append("# Stage0/Stage1 완료 점검 결과")
    lines.append("")
    lines.append(f"**점검 일시**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Base Directory**: `{BASE_DIR}`")
    lines.append(f"**Baseline Tag**: `{BASELINE_TAG}`")
    lines.append("")
    lines.append("---")
    lines.append("")

    # [1] 태그 탐지 결과
    lines.append("## [1] 태그 자동 탐지")
    lines.append("")
    lines.append(f"- **Stage0 태그**: `{stage0_tag if stage0_tag else 'NONE'}`")
    lines.append(f"- **Stage1 태그**: `{stage1_tag if stage1_tag else 'NONE'}`")
    lines.append("")

    # [2] 리포트 파일 체크
    lines.append("## [2] 필수 리포트 파일 존재 체크")
    lines.append("")

    if stage0_tag:
        lines.append(f"### Stage0 (`{stage0_tag}`)")
        lines.append("")
        lines.append("| 파일명 | 존재 여부 |")
        lines.append("|--------|----------|")
        for filename, exists in stage0_reports.items():
            status = "[OK]" if exists else "[MISSING]"
            lines.append(f"| `{filename}` | {status} |")
        lines.append("")
    else:
        lines.append("### Stage0")
        lines.append("")
        lines.append("[FAIL] **태그 없음** - 리포트 파일 체크 불가")
        lines.append("")

    if stage1_tag:
        lines.append(f"### Stage1 (`{stage1_tag}`)")
        lines.append("")
        lines.append("| 파일명 | 존재 여부 |")
        lines.append("|--------|----------|")
        for filename, exists in stage1_reports.items():
            status = "[OK]" if exists else "[MISSING]"
            lines.append(f"| `{filename}` | {status} |")
        lines.append("")
    else:
        lines.append("### Stage1")
        lines.append("")
        lines.append("[FAIL] **태그 없음** - 리포트 파일 체크 불가")
        lines.append("")

    # [3] 산출물 체크
    lines.append("## [3] 필수 산출물(parquet) 존재 체크")
    lines.append("")

    if stage0_tag:
        lines.append(f"### Stage0 (`{stage0_tag}`)")
        lines.append("")
        lines.append("| 파일명 | 존재 여부 |")
        lines.append("|--------|----------|")
        missing_stage0 = []
        for filename, exists in stage0_artifacts.items():
            status = "[OK]" if exists else "[MISSING]"
            lines.append(f"| `{filename}` | {status} |")
            if not exists:
                missing_stage0.append(filename)
        lines.append("")
        if missing_stage0:
            lines.append(f"**누락 파일**: {', '.join(missing_stage0)}")
            lines.append("")
    else:
        lines.append("### Stage0")
        lines.append("")
        # 태그가 없어도 루트에서 체크 시도
        interim_root = BASE_DIR / "data" / "interim"
        root_artifacts = {}
        for filename in REQUIRED_ARTIFACTS:
            root_artifacts[filename] = check_file_exists(interim_root / filename)

        if any(root_artifacts.values()):
            lines.append("[WARN] **태그 없음** - 루트 디렉토리에서 산출물 확인 시도")
            lines.append("")
            lines.append("| 파일명 | 존재 여부 (루트) |")
            lines.append("|--------|----------------|")
            missing_stage0 = []
            for filename, exists in root_artifacts.items():
                status = "[OK]" if exists else "[MISSING]"
                lines.append(f"| `{filename}` | {status} |")
                if not exists:
                    missing_stage0.append(filename)
            lines.append("")
            lines.append("**참고**: 루트에 파일이 있지만 태그 디렉토리가 없습니다.")
            lines.append(
                "`--legacy-root-save` 모드로 실행되었거나 `--run-tag` 옵션이 사용되지 않은 것으로 보입니다."
            )
            lines.append("")
        else:
            lines.append("[FAIL] **태그 없음** - 산출물 체크 불가")
            lines.append("")
        missing_stage0 = []

    if stage1_tag:
        lines.append(f"### Stage1 (`{stage1_tag}`)")
        lines.append("")
        lines.append("| 파일명 | 존재 여부 |")
        lines.append("|--------|----------|")
        missing_stage1 = []
        for filename, exists in stage1_artifacts.items():
            status = "[OK]" if exists else "[MISSING]"
            lines.append(f"| `{filename}` | {status} |")
            if not exists:
                missing_stage1.append(filename)
        lines.append("")
        if missing_stage1:
            lines.append(f"**누락 파일**: {', '.join(missing_stage1)}")
            lines.append("")
    else:
        lines.append("### Stage1")
        lines.append("")
        # 태그가 없어도 루트에서 체크 시도
        interim_root = BASE_DIR / "data" / "interim"
        root_artifacts = {}
        for filename in REQUIRED_ARTIFACTS:
            root_artifacts[filename] = check_file_exists(interim_root / filename)

        if any(root_artifacts.values()):
            lines.append("[WARN] **태그 없음** - 루트 디렉토리에서 산출물 확인 시도")
            lines.append("")
            lines.append("| 파일명 | 존재 여부 (루트) |")
            lines.append("|--------|----------------|")
            missing_stage1 = []
            for filename, exists in root_artifacts.items():
                status = "[OK]" if exists else "[MISSING]"
                lines.append(f"| `{filename}` | {status} |")
                if not exists:
                    missing_stage1.append(filename)
            lines.append("")
            lines.append("**참고**: 루트에 파일이 있지만 태그 디렉토리가 없습니다.")
            lines.append(
                "`--legacy-root-save` 모드로 실행되었거나 `--run-tag` 옵션이 사용되지 않은 것으로 보입니다."
            )
            lines.append("")
        else:
            lines.append("[FAIL] **태그 없음** - 산출물 체크 불가")
            lines.append("")
        missing_stage1 = []

    # [4] L2 재사용 규칙 체크
    lines.append("## [4] L2 재무 재사용 규칙 체크")
    lines.append("")

    if l2_check["root_file_exists"]:
        lines.append("[OK] **PASS**: 루트 파일 존재")
        lines.append(f"   - 경로: `{l2_check['root_file_path']}`")
        lines.append("")
        lines.append("**참고**: L2 재사용 규칙에 따라 `fundamentals_annual.parquet`는")
        lines.append("태그 폴더에 없어도 루트에만 있으면 정상입니다.")
        lines.append("")
    else:
        lines.append("[FAIL] **FAIL**: 루트 파일 없음")
        lines.append(
            f"   - 예상 경로: `{BASE_DIR / 'data' / 'interim' / 'fundamentals_annual.parquet'}`"
        )
        lines.append("")

    # [5] Legacy Root Save 감지
    lines.append("## [5] Legacy Root Save 감지")
    lines.append("")

    if legacy_check["suspected"]:
        lines.append("[WARN] **LEGACY_ROOT_SAVE_SUSPECTED**")
        lines.append("")
        lines.append(
            f"- 루트 parquet 파일 개수: {legacy_check['root_artifacts_count']}"
        )
        lines.append(f"- 누락 태그: {', '.join(legacy_check['missing_tags'])}")
        lines.append("")
        lines.append("**의미**: `--legacy-root-save` 모드로 실행되었거나,")
        lines.append("run-tag 기반 저장 구조가 적용되지 않은 것으로 보입니다.")
        lines.append("")
    else:
        lines.append("[OK] **정상**: 태그별 디렉토리 구조 사용 중")
        lines.append("")

    # [6] 최종 판정
    lines.append("## [6] 최종 판정")
    lines.append("")

    # Stage0 판정
    stage0_completed = (
        stage0_tag is not None
        and all(stage0_reports.values())
        and all(stage0_artifacts.values())
    )

    lines.append(
        f"### Stage0: {'[OK] **COMPLETED**' if stage0_completed else '[FAIL] **NOT COMPLETED**'}"
    )
    lines.append("")

    if stage0_completed:
        lines.append("**PASS 항목**:")
        lines.append("- 모든 필수 리포트 파일 존재")
        lines.append("- 모든 필수 산출물 존재")
        lines.append("")
    else:
        lines.append("**FAIL 항목**:")
        if stage0_tag is None:
            lines.append("- Stage0 태그 없음")
        if stage0_tag and not all(stage0_reports.values()):
            missing_reports = [f for f, exists in stage0_reports.items() if not exists]
            lines.append(f"- 리포트 파일 누락: {', '.join(missing_reports)}")
        if stage0_tag and not all(stage0_artifacts.values()):
            missing_artifacts = [
                f for f, exists in stage0_artifacts.items() if not exists
            ]
            lines.append(f"- 산출물 누락: {', '.join(missing_artifacts)}")
        lines.append("")

    # Stage1 판정
    stage1_completed = (
        stage1_tag is not None
        and all(stage1_reports.values())
        and all(stage1_artifacts.values())
    )

    lines.append(
        f"### Stage1: {'[OK] **COMPLETED**' if stage1_completed else '[FAIL] **NOT COMPLETED**'}"
    )
    lines.append("")

    if stage1_completed:
        lines.append("**PASS 항목**:")
        lines.append("- 모든 필수 리포트 파일 존재")
        lines.append("- 모든 필수 산출물 존재")
        lines.append("")
    else:
        lines.append("**FAIL 항목**:")
        if stage1_tag is None:
            lines.append("- Stage1 태그 없음")
        if stage1_tag and not all(stage1_reports.values()):
            missing_reports = [f for f, exists in stage1_reports.items() if not exists]
            lines.append(f"- 리포트 파일 누락: {', '.join(missing_reports)}")
        if stage1_tag and not all(stage1_artifacts.values()):
            missing_artifacts = [
                f for f, exists in stage1_artifacts.items() if not exists
            ]
            lines.append(f"- 산출물 누락: {', '.join(missing_artifacts)}")
        lines.append("")

    # L2 재사용 규칙 판정
    l2_pass = l2_check["root_file_exists"]
    lines.append(
        f"### L2 재무 재사용 규칙 준수: {'[OK] **PASS**' if l2_pass else '[FAIL] **FAIL**'}"
    )
    lines.append("")

    if l2_pass:
        lines.append("**근거**: 루트 파일 존재 확인됨")
        lines.append("")
    else:
        lines.append("**근거**: 루트 파일 없음 (즉시 중단 필요)")
        lines.append("")

    return "\n".join(lines)


def main():
    """메인 함수"""
    print("=" * 80)
    print("Stage0/Stage1 완료 점검")
    print("=" * 80)
    print()
    print(f"점검 일시: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base Directory: {BASE_DIR}")
    print(f"Baseline Tag: {BASELINE_TAG}")
    print()

    # 태그 탐지
    stage0_tag, stage1_tag = find_stage_tags()

    # 리포트 체크
    stage0_reports = check_reports(stage0_tag, BASELINE_TAG) if stage0_tag else {}
    stage1_reports = check_reports(stage1_tag, BASELINE_TAG) if stage1_tag else {}

    # 산출물 체크
    stage0_artifacts = check_artifacts(stage0_tag) if stage0_tag else {}
    stage1_artifacts = check_artifacts(stage1_tag) if stage1_tag else {}

    # L2 재사용 규칙 체크
    l2_check = check_l2_reuse()

    # Legacy Root Save 감지
    legacy_check = check_legacy_root_save()

    # 보고서 생성 및 출력
    report = format_markdown_report(
        stage0_tag,
        stage1_tag,
        stage0_reports,
        stage1_reports,
        stage0_artifacts,
        stage1_artifacts,
        l2_check,
        legacy_check,
    )

    print(report)

    # 판정 요약 출력
    print("=" * 80)
    print("판정 요약")
    print("=" * 80)
    print()

    stage0_completed = (
        stage0_tag is not None
        and all(stage0_reports.values())
        and all(stage0_artifacts.values())
    )

    stage1_completed = (
        stage1_tag is not None
        and all(stage1_reports.values())
        and all(stage1_artifacts.values())
    )

    l2_pass = l2_check["root_file_exists"]

    print(f"Stage0: {'COMPLETED' if stage0_completed else 'NOT COMPLETED'}")
    print(f"Stage1: {'COMPLETED' if stage1_completed else 'NOT COMPLETED'}")
    print(f"L2 재사용 규칙: {'PASS' if l2_pass else 'FAIL'}")
    print(f"Legacy Root Save 의심: {'YES' if legacy_check['suspected'] else 'NO'}")
    print()

    # 다음 조치
    print("=" * 80)
    print("다음 조치")
    print("=" * 80)
    print()

    if not stage0_tag or not stage1_tag:
        print("1. run-tag 기반 저장 구조 적용 필요: YES")
        print("   - 파이프라인 실행 시 --run-tag 옵션 사용")
        print("   - --legacy-root-save 옵션 사용 금지")
    else:
        print("1. run-tag 기반 저장 구조 적용 필요: NO (이미 적용됨)")

    if not stage0_completed or not stage1_completed:
        print("2. KPI/Delta 생성 스크립트 실행 필요: YES")
        if stage0_tag and not all(stage0_reports.values()):
            print(
                f"   - Stage0: python src\\tools\\export_kpi_table.py --config configs\\config.yaml --tag {stage0_tag}"
            )
            print(
                f"   - Stage0: python src\\tools\\build_kpi_delta.py --baseline-tag {BASELINE_TAG} --tag {stage0_tag}"
            )
        if stage1_tag and not all(stage1_reports.values()):
            print(
                f"   - Stage1: python src\\tools\\export_kpi_table.py --config configs\\config.yaml --tag {stage1_tag}"
            )
            print(
                f"   - Stage1: python src\\tools\\build_kpi_delta.py --baseline-tag {BASELINE_TAG} --tag {stage1_tag}"
            )
    else:
        print("2. KPI/Delta 생성 스크립트 실행 필요: NO (이미 완료됨)")

    if not l2_pass:
        print("3. L2 재사용 규칙 위반 여부: YES - 즉시 중단 필요")
        print("   - data\\interim\\fundamentals_annual.parquet 파일이 없습니다")
        print("   - L2 재사용 규칙에 따라 이 파일은 필수입니다")
    else:
        print("3. L2 재사용 규칙 위반 여부: NO (정상)")

    print()


if __name__ == "__main__":
    main()
