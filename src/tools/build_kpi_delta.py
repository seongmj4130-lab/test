# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/build_kpi_delta.py
"""
KPI Delta 보고서 생성 래퍼 스크립트
generate_delta_report.py를 호출하여 baseline과 stage KPI 비교 delta csv/md 생성
"""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Build KPI Delta Report")
    parser.add_argument("--baseline-tag", type=str, required=True, help="Baseline tag")
    parser.add_argument("--tag", type=str, required=True, help="Current stage tag")
    parser.add_argument("--root", type=str, default=None, help="Project root directory")
    args = parser.parse_args()

    # 루트 경로 결정
    if args.root:
        root = Path(args.root)
    else:
        root = Path(__file__).resolve().parents[2]

    # generate_delta_report.py 호출
    delta_script = root / "src" / "tools" / "generate_delta_report.py"

    if not delta_script.exists():
        print(f"ERROR: Delta report script not found: {delta_script}", file=sys.stderr)
        sys.exit(1)

    cmd = [
        sys.executable,
        str(delta_script),
        "--baseline-tag", args.baseline_tag,
        "--current-tag", args.tag,
    ]

    result = subprocess.run(cmd, cwd=str(root))

    if result.returncode != 0:
        print(f"ERROR: Delta report generation failed with exit code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    print(f"[Build KPI Delta] Completed: {args.baseline_tag} vs {args.tag}")

if __name__ == "__main__":
    main()
