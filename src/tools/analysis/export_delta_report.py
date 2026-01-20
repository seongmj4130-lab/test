# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/analysis/export_delta_report.py
"""
Delta 리포트 추출 스크립트 (Stage0용)
generate_delta_report.py를 래핑하여 --run-tag 파라미터 지원
"""
import argparse

# generate_delta_report.py를 import하여 재사용
import sys
from pathlib import Path

# 같은 디렉토리의 generate_delta_report.py를 import
tools_dir = Path(__file__).parent
sys.path.insert(0, str(tools_dir))

from generate_delta_report import (
    calculate_delta,
    generate_markdown_report,
    load_kpi_csv,
)


def main():
    parser = argparse.ArgumentParser(
        description="Export Delta Report comparing baseline vs run_tag (Stage0 wrapper)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Config file path (not used, kept for compatibility)",
    )
    parser.add_argument(
        "--baseline-tag",
        type=str,
        required=True,
        help="Baseline tag (e.g., baseline_prerefresh_20251219_143636)",
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        required=True,
        help="Current run tag (e.g., stage0_rebuild_tagged_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument("--root", type=str, default=None, help="Project root directory")
    parser.add_argument(
        "--kpi-dir", type=str, default="reports/kpi", help="KPI directory"
    )
    parser.add_argument(
        "--delta-dir",
        type=str,
        default="reports/delta",
        help="Delta report output directory",
    )
    parser.add_argument(
        "--no-md",
        action="store_true",
        default=False,
        help="[TASK A-1] MD 렌더링 비활성화 (CSV만 생성)",
    )
    args = parser.parse_args()

    # 루트 경로 결정
    if args.root:
        root = Path(args.root)
    else:
        root = Path(__file__).resolve().parents[2]

    kpi_dir = root / args.kpi_dir
    delta_dir = root / args.delta_dir
    delta_dir.mkdir(parents=True, exist_ok=True)

    # KPI CSV 파일 경로
    baseline_kpi_csv = kpi_dir / f"kpi_table__{args.baseline_tag}.csv"
    current_kpi_csv = kpi_dir / f"kpi_table__{args.run_tag}.csv"

    print(f"[Delta Report] Baseline: {baseline_kpi_csv}")
    print(f"[Delta Report] Current (run-tag): {current_kpi_csv}")

    # KPI 로드
    try:
        baseline_df = load_kpi_csv(baseline_kpi_csv)
        current_df = load_kpi_csv(current_kpi_csv)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        print(
            "KPI CSV 파일이 없습니다. 먼저 export_kpi_table.py를 실행하세요.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[Delta Report] Baseline KPIs: {len(baseline_df)}")
    print(f"[Delta Report] Current KPIs: {len(current_df)}")

    # Delta 계산
    delta_df = calculate_delta(baseline_df, current_df)

    print(f"[Delta Report] Delta KPIs: {len(delta_df)}")

    # CSV 저장
    csv_path = delta_dir / f"delta_kpi__{args.baseline_tag}__vs__{args.run_tag}.csv"
    delta_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[Delta Report] CSV saved: {csv_path}")

    # Markdown 저장 ([TASK A-1] --no-md 옵션으로 비활성화 가능)
    if not args.no_md:
        md_path = (
            delta_dir / f"delta_report__{args.baseline_tag}__vs__{args.run_tag}.md"
        )
        md_content = generate_markdown_report(delta_df, args.baseline_tag, args.run_tag)
        md_path.write_text(md_content, encoding="utf-8")
        print(f"[Delta Report] Markdown saved: {md_path}")
    else:
        print("[Delta Report] --no-md 활성화: Markdown 생성 건너뛰기")

    print("\n[Delta Report] Completed.")


if __name__ == "__main__":
    main()
