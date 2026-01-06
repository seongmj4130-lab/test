# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/run_stage13.py
"""
[Stage13] K_eff 복원 실행 스크립트
- top_k=20인데 실제 avg_n_tickers(K_eff)가 8~13개로 떨어지는 근원 원인을 데이터 기반으로 로깅
- 가능한 범위에서 K_eff를 20에 가깝게 복원 (fallback 적용)
"""
import argparse
import sys
import subprocess
from pathlib import Path
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(
        description="[Stage13] K_eff 복원 실행 스크립트"
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml",
                       help="Config 파일 경로")
    parser.add_argument("--run-tag", type=str, default=None,
                       help="Run tag (없으면 자동 생성)")
    parser.add_argument("--baseline-tag", type=str, default=None,
                       help="Baseline tag (직전 Stage, 없으면 자동 탐지)")
    parser.add_argument("--global-baseline-tag", type=str, default=None,
                       help="Global baseline tag (Stage12 최신, 없으면 자동 탐지)")
    args = parser.parse_args()
    
    # Run tag 생성
    if args.run_tag:
        run_tag = args.run_tag
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_tag = f"stage13_keff_restore_{timestamp}"
    
    # run_stage_pipeline.py 호출
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    pipeline_script = script_dir / "run_stage_pipeline.py"
    
    cmd = [
        sys.executable,
        str(pipeline_script),
        "--stage", "13",
        "--run-tag", run_tag,
        "--config", args.config,
    ]
    
    if args.baseline_tag:
        cmd.extend(["--baseline-tag", args.baseline_tag])
    
    if args.global_baseline_tag:
        cmd.extend(["--global-baseline-tag", args.global_baseline_tag])
    
    # 실행
    result = subprocess.run(cmd, cwd=str(project_root))
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()
