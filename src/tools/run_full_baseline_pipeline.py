# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/run_full_baseline_pipeline.py
"""
베이스라인 풀런 오케스트레이터
- L0~L7D까지 순차 실행
- 진행도 대시보드 자동 업데이트 (RUN_STATUS.md)
- 단계별 검증 자동 수행
- 최종 리포트 생성
"""
import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

# 프로젝트 루트를 sys.path에 추가
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from src.utils.config import get_path, load_config
from src.utils.io import artifact_exists, load_artifact

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# =========================
# 상수 정의
# =========================
STAGES = ["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L7B", "L7C", "L7D"]

STAGE_OUTPUTS = {
    "L0": ["universe_k200_membership_monthly"],
    "L1": ["ohlcv_daily"],
    "L2": ["fundamentals_annual"],
    "L3": ["panel_merged_daily"],
    "L4": ["dataset_daily", "cv_folds_short", "cv_folds_long"],
    "L5": ["pred_short_oos", "pred_long_oos", "model_metrics"],
    "L6": ["rebalance_scores", "rebalance_scores_summary"],
    "L7": ["bt_positions", "bt_returns", "bt_equity_curve", "bt_metrics"],
    "L7B": ["bt_sensitivity_metrics"],
    "L7C": ["bt_vs_benchmark", "bt_benchmark_compare", "bt_benchmark_returns"],
    "L7D": ["bt_yearly_metrics", "bt_rolling_sharpe", "bt_drawdown_events"],
}

REQUIRED_COLS_BY_OUTPUT = {
    "universe_k200_membership_monthly": ["date", "ticker"],
    "ohlcv_daily": ["date", "ticker"],
    "fundamentals_annual": ["date", "ticker"],
    "panel_merged_daily": ["date", "ticker"],
    "dataset_daily": ["date", "ticker"],
    "cv_folds_short": ["fold_id", "segment", "train_start", "train_end", "test_start", "test_end"],
    "cv_folds_long": ["fold_id", "segment", "train_start", "train_end", "test_start", "test_end"],
    "pred_short_oos": ["date", "ticker", "y_true", "y_pred", "fold_id", "phase", "horizon"],
    "pred_long_oos": ["date", "ticker", "y_true", "y_pred", "fold_id", "phase", "horizon"],
    "model_metrics": ["horizon", "phase", "rmse"],
    "rebalance_scores": ["date", "ticker", "phase"],
    "rebalance_scores_summary": ["date", "phase", "n_tickers"],
    "bt_positions": ["date", "phase", "ticker"],
    "bt_returns": ["date", "phase"],
    "bt_equity_curve": ["date", "phase"],
    "bt_metrics": ["phase", "net_total_return", "net_sharpe", "net_mdd"],
    "bt_sensitivity_metrics": ["phase"],
    "bt_vs_benchmark": ["date", "phase", "bench_return", "excess_return"],
    "bt_benchmark_compare": ["phase", "tracking_error_ann", "information_ratio"],
    "bt_benchmark_returns": ["date", "phase", "bench_return"],
    "bt_yearly_metrics": ["phase", "year"],
    "bt_rolling_sharpe": ["date", "phase"],
    "bt_drawdown_events": ["phase", "peak_date", "trough_date", "drawdown"],
}

# =========================
# 진행도 관리
# =========================
class RunStatus:
    def __init__(self, run_tag: str, root: Path, config_path: Path, baseline_source: str):
        self.run_tag = run_tag
        self.root = root
        self.config_path = config_path
        self.baseline_source = baseline_source
        self.started_at = datetime.now().isoformat()
        self.current_stage = None
        self.stage_status: Dict[str, Dict[str, Any]] = {}
        self.last_update = None
        self.last_cmd = None
        self.last_exit_code = None
        self.last_log_file = None
        self.validation_summary: List[Dict[str, Any]] = []
        self.file_pointers: List[str] = []

        self.logs_dir = root / "logs" / run_tag
        self.reports_dir = root / "reports" / run_tag
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def update_stage(self, stage: str, status: str, start_time: Optional[datetime] = None,
                     end_time: Optional[datetime] = None, exit_code: Optional[int] = None,
                     log_file: Optional[str] = None, cmd: Optional[str] = None):
        """단계 상태 업데이트"""
        if stage not in self.stage_status:
            self.stage_status[stage] = {}

        self.stage_status[stage].update({
            "status": status,  # "PENDING", "RUNNING", "DONE", "FAIL"
            "start_time": start_time.isoformat() if start_time else None,
            "end_time": end_time.isoformat() if end_time else None,
            "exit_code": exit_code,
            "log_file": log_file,
            "cmd": cmd,
        })

        self.current_stage = stage
        self.last_update = datetime.now().isoformat()
        if exit_code is not None:
            self.last_exit_code = exit_code
        if log_file:
            self.last_log_file = log_file
        if cmd:
            self.last_cmd = cmd

    def add_validation(self, stage: str, output_name: str, status: str, checks: str, notes: str = ""):
        """검증 결과 추가"""
        self.validation_summary.append({
            "stage": stage,
            "output": output_name,
            "status": status,  # "PASS", "WARN", "FAIL"
            "checks": checks,
            "notes": notes,
        })

    def add_file_pointer(self, path: str):
        """파일 포인터 추가"""
        if path not in self.file_pointers:
            self.file_pointers.append(path)

    def save_markdown(self, path: Path):
        """RUN_STATUS.md 저장"""
        lines = [
            "# BASELINE FULL RUN DASHBOARD",
            f"- run_tag: {self.run_tag}",
            f"- started_at: {self.started_at}",
            f"- root: {self.root}",
            f"- config: {self.config_path}",
            f"- baseline_source_backup: {self.baseline_source}",
            f"- logs_dir: {self.logs_dir}",
            f"- reports_dir: {self.reports_dir}",
            "",
            "## Stage Progress",
        ]

        for stage in STAGES:
            status_info = self.stage_status.get(stage, {})
            status = status_info.get("status", "PENDING")
            checkbox = "[x]" if status == "DONE" else "[ ]" if status == "FAIL" else "[ ]"
            lines.append(f"- {checkbox} {stage} ... ({status})")
            if status_info.get("start_time"):
                lines.append(f"  - 시작: {status_info['start_time']}")
            if status_info.get("end_time"):
                lines.append(f"  - 종료: {status_info['end_time']}")
            if status_info.get("exit_code") is not None:
                lines.append(f"  - exit_code: {status_info['exit_code']}")
            if status_info.get("log_file"):
                lines.append(f"  - 로그: {status_info['log_file']}")

        lines.extend([
            "",
            "## Latest Status (auto-updated)",
            f"- current_stage: {self.current_stage or 'N/A'}",
            f"- last_update: {self.last_update or 'N/A'}",
            f"- last_cmd: {self.last_cmd or 'N/A'}",
            f"- last_exit_code: {self.last_exit_code if self.last_exit_code is not None else 'N/A'}",
            f"- last_log_file: {self.last_log_file or 'N/A'}",
            "",
            "## Validation Summary",
            "| stage | output | status | key_checks | notes |",
            "|---|---|---|---|---|",
        ])

        for v in self.validation_summary:
            lines.append(f"| {v['stage']} | {v['output']} | {v['status']} | {v['checks']} | {v['notes']} |")

        lines.extend([
            "",
            "## File Pointers (found)",
        ])

        for fp in self.file_pointers:
            lines.append(f"- {fp} (exists)")

        path.write_text("\n".join(lines), encoding="utf-8")

    def save_json(self, path: Path):
        """run_status.json 저장"""
        data = {
            "run_tag": self.run_tag,
            "started_at": self.started_at,
            "root": str(self.root),
            "config": str(self.config_path),
            "baseline_source": self.baseline_source,
            "logs_dir": str(self.logs_dir),
            "reports_dir": str(self.reports_dir),
            "current_stage": self.current_stage,
            "last_update": self.last_update,
            "last_cmd": self.last_cmd,
            "last_exit_code": self.last_exit_code,
            "last_log_file": self.last_log_file,
            "stage_status": self.stage_status,
            "validation_summary": self.validation_summary,
            "file_pointers": self.file_pointers,
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

# =========================
# 검증 함수
# =========================
def validate_stage_output(stage: str, output_name: str, interim_dir: Path,
                         status: RunStatus) -> Tuple[str, str, str]:
    """
    단계별 산출물 검증
    Returns: (status, checks, notes)
    """
    output_base = interim_dir / output_name

    # 파일 존재 여부 확인
    if not artifact_exists(output_base):
        status.add_file_pointer(str(output_base) + " (NOT FOUND)")
        return "FAIL", "파일 없음", f"{output_name} 산출물이 없습니다"

    status.add_file_pointer(str(output_base))

    try:
        df = load_artifact(output_base)

        # 기본 체크
        checks = []
        notes = []

        # 행/열 수 확인
        n_rows = len(df)
        n_cols = len(df.columns)
        checks.append(f"rows={n_rows}, cols={n_cols}")

        if n_rows == 0:
            return "FAIL", ", ".join(checks), "행이 0개입니다"

        # 필수 컬럼 확인
        required_cols = REQUIRED_COLS_BY_OUTPUT.get(output_name, [])
        if required_cols:
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                notes.append(f"필수 컬럼 누락: {missing_cols}")
                return "WARN", ", ".join(checks), "; ".join(notes)

        # L7 특수 검증
        if stage == "L7" and output_name == "bt_returns":
            if "net_return" in df.columns:
                min_ret = df["net_return"].min()
                if min_ret <= -1.0:
                    notes.append(f"최소 수익률이 -100% 이하: {min_ret:.2%}")
                    return "FAIL", ", ".join(checks), "; ".join(notes)

                # 비용 적용 여부 확인
                if "gross_return" in df.columns and "net_return" in df.columns:
                    if (df["gross_return"] == df["net_return"]).all():
                        notes.append("gross==net (비용 미적용 가능성)")
                        return "WARN", ", ".join(checks), "; ".join(notes)

        return "PASS", ", ".join(checks), "; ".join(notes) if notes else "OK"

    except Exception as e:
        return "FAIL", "로드 실패", str(e)

# =========================
# 단계 실행
# =========================
def run_stage(stage: str, config_path: Path, root: Path, status: RunStatus,
              force: bool = False) -> Tuple[bool, Optional[str]]:
    """
    단계 실행
    Returns: (success, log_file_path)
    """
    logger.info(f"{'='*20} START {stage} {'='*20}")

    start_time = datetime.now()
    status.update_stage(stage, "RUNNING", start_time=start_time)

    log_file = status.logs_dir / f"{stage}.log"

    # run_all.py 실행 명령 구성 (단일 스테이지 실행)
    cmd_parts = [
        sys.executable,
        str(root / "src" / "run_all.py"),
        "--config", str(config_path),
        "--stage", stage,
        "--run-tag", status.run_tag,
    ]

    if force:
        cmd_parts.append("--force")

    # strict-params는 L7에서만 적용
    if stage == "L7" and hasattr(status, 'strict_params') and status.strict_params:
        cmd_parts.append("--strict-params")

    cmd = " ".join(cmd_parts)
    status.update_stage(stage, "RUNNING", start_time=start_time, cmd=cmd, log_file=str(log_file))
    status.save_markdown(root / "RUN_STATUS.md")
    status.save_json(status.logs_dir / "run_status.json")

    # 실행
    try:
        with log_file.open("w", encoding="utf-8") as f:
            result = subprocess.run(
                cmd_parts,
                cwd=str(root),
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
            )

        end_time = datetime.now()
        exit_code = result.returncode

        if exit_code == 0:
            status.update_stage(stage, "DONE", start_time=start_time, end_time=end_time,
                              exit_code=exit_code, log_file=str(log_file), cmd=cmd)
            logger.info(f"{'='*20} END {stage} (OK) {'='*20}")
            return True, str(log_file)
        else:
            status.update_stage(stage, "FAIL", start_time=start_time, end_time=end_time,
                              exit_code=exit_code, log_file=str(log_file), cmd=cmd)
            logger.error(f"{'='*20} END {stage} (FAIL, exit_code={exit_code}) {'='*20}")
            return False, str(log_file)

    except Exception as e:
        end_time = datetime.now()
        status.update_stage(stage, "FAIL", start_time=start_time, end_time=end_time,
                          exit_code=-1, log_file=str(log_file), cmd=cmd)
        logger.error(f"{'='*20} END {stage} (EXCEPTION: {e}) {'='*20}")
        return False, str(log_file)

# =========================
# 메인 실행
# =========================
def main():
    parser = argparse.ArgumentParser(description="Baseline Full Pipeline Runner")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--run-tag", type=str, default=None)
    parser.add_argument("--baseline-tag", type=str, default="baseline_prerefresh_20251219_143636",
                       help="Baseline tag for Delta report comparison")
    parser.add_argument("--baseline-source", type=str,
                       default="data/snapshots/baseline_after_L7BCD")
    parser.add_argument("--from", dest="from_stage", type=str, default=None)
    parser.add_argument("--to", dest="to_stage", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--strict-params", action="store_true",
                       help="Fail pipeline if config parameters don't match actual usage")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[2]
    config_path = root / args.config

    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    # run_tag 생성
    if args.run_tag:
        run_tag = args.run_tag
    else:
        run_tag = f"baseline_prerefresh_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # RunStatus 초기화
    status = RunStatus(
        run_tag=run_tag,
        root=root,
        config_path=config_path,
        baseline_source=args.baseline_source,
    )
    status.strict_params = args.strict_params  # strict-params 플래그 저장

    # 초기 RUN_STATUS.md 생성
    status.save_markdown(root / "RUN_STATUS.md")
    status.save_json(status.logs_dir / "run_status.json")

    logger.info("=== BASELINE FULL RUN ORCHESTRATOR ===")
    logger.info(f"RUN_TAG: {run_tag}")
    logger.info(f"ROOT: {root}")
    logger.info(f"CONFIG: {config_path}")
    logger.info(f"BASELINE_SOURCE: {args.baseline_source}")
    logger.info(f"STAGES: {STAGES}")

    if args.dry_run:
        logger.info("[DRY-RUN] Skipping actual execution.")
        return

    # 설정 로드
    cfg = load_config(str(config_path))
    interim_dir = get_path(cfg, "data_interim")

    # 단계별 실행
    target_stages = STAGES
    if args.from_stage and args.to_stage:
        start_idx = STAGES.index(args.from_stage.upper())
        end_idx = STAGES.index(args.to_stage.upper())
        target_stages = STAGES[start_idx:end_idx + 1]

    for stage in target_stages:
        # 단계 실행
        success, log_file = run_stage(
            stage=stage,
            config_path=config_path,
            root=root,
            status=status,
            force=args.force,
        )

        # 검증 수행
        if success:
            outputs = STAGE_OUTPUTS.get(stage, [])
            for output_name in outputs:
                v_status, v_checks, v_notes = validate_stage_output(
                    stage=stage,
                    output_name=output_name,
                    interim_dir=interim_dir,
                    status=status,
                )
                status.add_validation(stage, output_name, v_status, v_checks, v_notes)

        # 진행도 저장
        status.save_markdown(root / "RUN_STATUS.md")
        status.save_json(status.logs_dir / "run_status.json")

        # 실패 시 중단
        if not success:
            logger.error(f"Pipeline stopped at {stage}. Check logs: {log_file}")
            sys.exit(1)

    # 최종 리포트 생성
    logger.info("Generating final reports...")
    generate_summary_report(status, root, interim_dir)
    generate_validation_table(status, root)

    # Manifest 생성
    logger.info("Generating manifest...")
    generate_manifest(status, root, args.config, interim_dir)

    # KPI 테이블 생성
    logger.info("Generating KPI table...")
    generate_kpi_table(status, root, args.config)

    # Delta 보고서 생성 (베이스라인과 비교)
    logger.info("Generating Delta report...")
    generate_delta_report(status, root, args.baseline_tag)

    # Audit 리포트 생성
    logger.info("Generating audit report...")
    generate_audit_report(status, root)

    # 최종 보고서 경로 출력
    logger.info("\n=== FINAL REPORTS ===")
    report_paths = {
        "manifest": f"reports/manifests/manifest__{status.run_tag}.json",
        "kpi_csv": f"reports/kpi/kpi_table__{status.run_tag}.csv",
        "kpi_md": f"reports/kpi/kpi_table__{status.run_tag}.md",
        "delta_csv": f"reports/delta/delta_kpi__{args.baseline_tag}__vs__{status.run_tag}.csv",
        "delta_md": f"reports/delta/delta_report__{args.baseline_tag}__vs__{status.run_tag}.md",
        "audit_json": f"reports/audit/audit__{status.run_tag}.json",
        "audit_md": f"reports/audit/audit__{status.run_tag}.md",
    }

    for name, path in report_paths.items():
        full_path = root / path
        if full_path.exists():
            logger.info(f"✅ {name}: {path}")
        else:
            logger.warning(f"⚠️ {name}: {path} (not found)")

    logger.info("=== PIPELINE COMPLETED SUCCESSFULLY ===")

def generate_summary_report(status: RunStatus, root: Path, interim_dir: Path):
    """최종 요약 리포트 생성"""
    report_path = status.reports_dir / "SUMMARY.md"

    lines = [
        "# Baseline Full Run Summary",
        f"- Run Tag: {status.run_tag}",
        f"- Started: {status.started_at}",
        f"- Baseline Source: {status.baseline_source}",
        "",
        "## Execution Summary",
        "",
        "| Stage | Status | Start Time | End Time | Exit Code |",
        "|---|---|---|---|---|",
    ]

    for stage in STAGES:
        info = status.stage_status.get(stage, {})
        lines.append(f"| {stage} | {info.get('status', 'N/A')} | "
                    f"{info.get('start_time', 'N/A')} | "
                    f"{info.get('end_time', 'N/A')} | "
                    f"{info.get('exit_code', 'N/A')} |")

    lines.extend([
        "",
        "## Key Artifacts",
    ])

    for fp in status.file_pointers:
        if "NOT FOUND" not in fp:
            lines.append(f"- `{fp}`")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Summary report saved: {report_path}")

def generate_validation_table(status: RunStatus, root: Path):
    """검증 테이블 리포트 생성"""
    report_path = status.reports_dir / "VALIDATION_TABLE.md"

    lines = [
        "# Validation Table",
        "",
        "| Stage | Output | Status | Key Checks | Notes |",
        "|---|---|---|---|---|",
    ]

    for v in status.validation_summary:
        lines.append(f"| {v['stage']} | {v['output']} | {v['status']} | "
                    f"{v['checks']} | {v['notes']} |")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Validation table saved: {report_path}")

def generate_kpi_table(status: RunStatus, root: Path, config_path: str):
    """KPI 테이블 생성"""
    import subprocess

    kpi_script = root / "src" / "tools" / "export_kpi_table.py"
    if not kpi_script.exists():
        logger.warning(f"KPI export script not found: {kpi_script}")
        return

    cmd = [
        sys.executable,
        str(kpi_script),
        "--tag", status.run_tag,
        "--config", config_path,
        "--format", "both",
    ]

    try:
        result = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"KPI table generated for tag: {status.run_tag}")
            if result.stdout:
                logger.debug(result.stdout)
        else:
            logger.error(f"KPI table generation failed: {result.stderr}")
            if result.stdout:
                logger.error(result.stdout)
    except Exception as e:
        logger.error(f"KPI table generation error: {e}")

def generate_manifest(status: RunStatus, root: Path, config_path: str, interim_dir: Path):
    """Manifest 생성"""
    import subprocess

    manifest_script = root / "src" / "tools" / "write_manifest.py"
    if not manifest_script.exists():
        logger.warning(f"Manifest script not found: {manifest_script}")
        return

    cmd = [
        sys.executable,
        str(manifest_script),
        "--run-tag", status.run_tag,
        "--config", config_path,
        "--interim-dir", str(interim_dir),
    ]

    try:
        result = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"Manifest generated for tag: {status.run_tag}")
        else:
            logger.error(f"Manifest generation failed: {result.stderr}")
    except Exception as e:
        logger.error(f"Manifest generation error: {e}")

def generate_delta_report(status: RunStatus, root: Path, baseline_tag: str):
    """Delta 보고서 생성 (베이스라인과 비교)"""
    import subprocess

    delta_script = root / "src" / "tools" / "generate_delta_report.py"
    if not delta_script.exists():
        logger.warning(f"Delta report script not found: {delta_script}")
        return

    cmd = [
        sys.executable,
        str(delta_script),
        "--baseline-tag", baseline_tag,
        "--current-tag", status.run_tag,
    ]

    try:
        result = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"Delta report generated: {baseline_tag} vs {status.run_tag}")
        else:
            logger.error(f"Delta report generation failed: {result.stderr}")
    except Exception as e:
        logger.error(f"Delta report generation error: {e}")

def generate_audit_report(status: RunStatus, root: Path):
    """Audit 리포트 생성"""
    import subprocess

    audit_script = root / "src" / "tools" / "audit_pipeline_features.py"
    if not audit_script.exists():
        logger.warning(f"Audit script not found: {audit_script}")
        return

    cmd = [
        sys.executable,
        str(audit_script),
        "--run-tag", status.run_tag,
    ]

    try:
        result = subprocess.run(cmd, cwd=str(root), capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"Audit report generated for tag: {status.run_tag}")
        else:
            logger.error(f"Audit report generation failed: {result.stderr}")
    except Exception as e:
        logger.error(f"Audit report generation error: {e}")

if __name__ == "__main__":
    main()
