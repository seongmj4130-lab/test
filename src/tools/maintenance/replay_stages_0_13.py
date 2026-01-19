# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/maintenance/replay_stages_0_13.py
"""
[TASK B-1] Stage0~13 리플레이 자동화 스크립트

기능:
1. plan.yaml을 읽어 Stage를 순서대로 실행
2. 각 실행은 기본 옵션을 강제 (--skip-l2 --force-rebuild --no-scan --profile)
3. 실행 완료 후 산출물 존재 체크
4. Stage별 KPI 수집 및 CSV 생성
5. Stage별 변화량 테이블 생성 (vs-prev, vs-baseline)
6. 최종 요약 MD 생성
"""
import argparse
import logging
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# 프로젝트 루트 경로 고정
PROJECT_ROOT = Path("C:/Users/seong/OneDrive/Desktop/bootcamp/03_code")

def load_plan(plan_path: Path) -> Dict[str, Any]:
    """plan.yaml 로드"""
    if not plan_path.exists():
        raise FileNotFoundError(f"Plan 파일을 찾을 수 없습니다: {plan_path}")

    with open(plan_path, "r", encoding="utf-8") as f:
        plan = yaml.safe_load(f)

    return plan

def resolve_baseline_tag(
    baseline_for_delta: str,
    base_interim_dir: Path,
    current_stage_no: int,
    track: str,
) -> str:
    """
    baseline_for_delta를 실제 태그로 변환

    Args:
        baseline_for_delta: plan.yaml의 baseline_for_delta 값 (예: "stage6_", "baseline_prerefresh_...")
        base_interim_dir: interim 디렉토리
        current_stage_no: 현재 Stage 번호
        track: "pipeline" 또는 "ranking"

    Returns:
        실제 baseline 태그
    """
    # 이미 전체 태그인 경우 (baseline_prerefresh_...)
    if baseline_for_delta.startswith("baseline_"):
        return baseline_for_delta

    # stage{N}_ 패턴인 경우 최신 태그 찾기
    if baseline_for_delta.startswith("stage") and baseline_for_delta.endswith("_"):
        stage_prefix = baseline_for_delta
        candidates = []
        for folder in base_interim_dir.iterdir():
            if folder.is_dir() and folder.name.startswith(stage_prefix):
                # Track에 맞는 산출물 확인
                if track == "pipeline":
                    if (folder / "rebalance_scores.parquet").exists() or (folder / "bt_returns.parquet").exists():
                        candidates.append((folder.name, folder.stat().st_mtime))
                elif track == "ranking":
                    if (folder / "ranking_daily.parquet").exists():
                        candidates.append((folder.name, folder.stat().st_mtime))

        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]

    # 기본값 반환
    return baseline_for_delta

def generate_run_tag(prefix: str) -> str:
    """Stage 실행용 run_tag 생성"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}{timestamp}"

def run_stage(
    stage_def: Dict[str, Any],
    config_path: Path,
    base_dir: Path,
    plan: Dict[str, Any],
) -> Tuple[bool, str, str, float]:
    """
    단일 Stage 실행 (실시간 스트리밍)

    Returns:
        (성공 여부, run_tag, baseline_tag, 실행 시간)
    """
    run_tag = generate_run_tag(stage_def["run_tag_prefix"])

    # baseline_for_delta를 실제 태그로 변환
    base_interim_dir = base_dir / "data" / "interim"
    baseline_tag = resolve_baseline_tag(
        stage_def["baseline_for_delta"],
        base_interim_dir,
        stage_def["stage_no"],
        stage_def["track"],
    )

    # [TASK B-stable] input_tag 결정: stage별 override 또는 전역 기본값
    input_tag = stage_def.get("input_tag")
    if input_tag is None:
        # Pipeline track (L7 포함)은 전역 input_tag 기본값 사용
        if stage_def["track"] == "pipeline":
            input_tag = plan.get("default_input_tag", "stage6_sector_relative_feature_balance_20251220_194928")
        else:
            input_tag = None  # Ranking track은 input_tag 불필요

    logger.info(f"\n{'='*80}")
    logger.info(f"Stage {stage_def['stage_no']} 실행 시작")
    logger.info(f"Run Tag: {run_tag}")
    logger.info(f"Baseline Tag: {baseline_tag} (비교 기준)")
    logger.info(f"Input Tag: {input_tag} (입력 산출물 소스)")
    logger.info(f"From: {stage_def['from']}, To: {stage_def['to']}")
    logger.info(f"Track: {stage_def['track']}")
    logger.info(f"Notes: {stage_def['notes']}")
    logger.info(f"{'='*80}\n")

    # run_all.py 실행 명령 구성 (기본 옵션 강제)
    cmd = [
        sys.executable,
        "-u",  # unbuffered 출력
        str(base_dir / "src" / "run_all.py"),
        "--from", stage_def["from"],
        "--to", stage_def["to"],
        "--run-tag", run_tag,
        "--baseline-tag", baseline_tag,
        "--config", str(config_path),
        "--skip-l2",  # 기본 옵션 강제
        "--force-rebuild",  # 기본 옵션 강제
        "--no-scan",  # 가능한 경우 활성화
        "--profile",  # 프로파일링 활성화
    ]

    # [TASK B-stable] input_tag 추가
    if input_tag:
        cmd.extend(["--input-tag", input_tag])

    # Stage13만 max_rebalances 적용 (스모크 테스트)
    if stage_def["stage_no"] == 13:
        cmd.extend(["--max-rebalances", "10"])

    # [TASK B-stable] 실시간 스트리밍 실행
    logger.info(f"[실행] 명령: {' '.join(cmd)}")
    logger.info(f"[실행] 실시간 로그 스트리밍 시작...\n")
    start_time = time.time()

    try:
        # Popen으로 실시간 스트리밍
        process = subprocess.Popen(
            cmd,
            cwd=str(base_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # stderr를 stdout에 병합
            text=True,
            bufsize=1,  # line buffered
            universal_newlines=True,
        )

        # 실시간 출력
        stdout_lines = []
        for line in process.stdout:
            line = line.rstrip()
            print(line, flush=True)  # 실시간 출력
            stdout_lines.append(line)

        # 프로세스 종료 대기
        return_code = process.wait()
        elapsed = time.time() - start_time

        stdout_text = "\n".join(stdout_lines)

        if return_code == 0:
            logger.info(f"\n[성공] Stage {stage_def['stage_no']} 실행 완료: {elapsed:.1f}초")
            return True, run_tag, baseline_tag, elapsed
        else:
            logger.error(f"\n[실패] Stage {stage_def['stage_no']} 실행 실패: {elapsed:.1f}초")
            logger.error(f"[실패] Return code: {return_code}")
            logger.error(f"[실패] 출력:\n{stdout_text}")
            return False, run_tag, baseline_tag, elapsed

    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[실패] Stage {stage_def['stage_no']} 실행 중 예외 발생: {elapsed:.1f}초")
        logger.error(f"[실패] 예외: {e}")
        import traceback
        logger.error(f"[실패] 트레이스백:\n{traceback.format_exc()}")
        return False, run_tag, baseline_tag, elapsed

def check_artifacts(
    run_tag: str,
    base_dir: Path,
    track: str,
) -> Dict[str, bool]:
    """
    산출물 존재 체크

    Returns:
        {artifact_name: exists} 딕셔너리
    """
    base_interim_dir = base_dir / "data" / "interim"
    run_dir = base_interim_dir / run_tag

    checks = {}

    if track == "pipeline":
        # Pipeline track: 백테스트 산출물 확인
        checks["bt_returns"] = (run_dir / "bt_returns.parquet").exists()
        checks["bt_positions"] = (run_dir / "bt_positions.parquet").exists()
        checks["bt_metrics"] = (run_dir / "bt_metrics.parquet").exists()
        checks["rebalance_scores"] = (run_dir / "rebalance_scores.parquet").exists()
    elif track == "ranking":
        # Ranking track: 랭킹 산출물 확인
        checks["ranking_daily"] = (run_dir / "ranking_daily.parquet").exists()
        checks["ranking_snapshot"] = (run_dir / "ranking_snapshot.parquet").exists()
        sector_csv = base_dir / "reports" / "ranking" / f"sector_concentration__{run_tag}.csv"
        checks["sector_concentration"] = sector_csv.exists()

    # 공통 리포트 확인
    kpi_csv = base_dir / "reports" / "kpi" / f"kpi_table__{run_tag}.csv"
    checks["kpi_table"] = kpi_csv.exists()

    return checks

def extract_kpi_from_bt_metrics(
    run_tag: str,
    base_dir: Path,
) -> Dict[str, Any]:
    """
    bt_metrics.parquet에서 KPI 추출

    Returns:
        KPI 딕셔너리
    """
    kpis = {}

    bt_metrics_path = base_dir / "data" / "interim" / run_tag / "bt_metrics.parquet"
    if not bt_metrics_path.exists():
        return kpis

    try:
        df = pd.read_parquet(bt_metrics_path)

        # holdout 우선, 없으면 dev
        if (df["phase"].astype(str) == "holdout").any():
            row = df[df["phase"].astype(str) == "holdout"].iloc[0]
        else:
            row = df.iloc[0]

        kpis["holdout_sharpe"] = float(row.get("net_sharpe", None)) if pd.notna(row.get("net_sharpe")) else None
        kpis["holdout_mdd"] = float(row.get("net_mdd", None)) if pd.notna(row.get("net_mdd")) else None
        kpis["holdout_cagr"] = float(row.get("net_cagr", None)) if pd.notna(row.get("net_cagr")) else None
        kpis["holdout_total_return"] = float(row.get("net_total_return", None)) if pd.notna(row.get("net_total_return")) else None
        kpis["avg_turnover"] = float(row.get("avg_turnover_oneway", None)) if pd.notna(row.get("avg_turnover_oneway")) else None
        kpis["avg_n_tickers"] = float(row.get("avg_n_tickers", None)) if pd.notna(row.get("avg_n_tickers")) else None
        kpis["cost_bps_used"] = float(row.get("cost_bps", None)) if pd.notna(row.get("cost_bps")) else None
        kpis["n_rebalances"] = int(row.get("n_rebalances", None)) if pd.notna(row.get("n_rebalances")) else None

        kpis["metric_source"] = "bt_metrics.parquet"
    except Exception as e:
        logger.warning(f"[KPI 추출] bt_metrics 읽기 실패: {e}")
        kpis["metric_source"] = f"ERROR: {e}"

    return kpis

def extract_kpi_from_kpi_table(
    run_tag: str,
    base_dir: Path,
) -> Dict[str, Any]:
    """
    kpi_table__{run_tag}.csv에서 KPI 추출

    Returns:
        KPI 딕셔너리
    """
    kpis = {}

    kpi_csv = base_dir / "reports" / "kpi" / f"kpi_table__{run_tag}.csv"
    if not kpi_csv.exists():
        return kpis

    try:
        df = pd.read_csv(kpi_csv)

        # holdout_value 추출
        def get_value(section: str, metric: str) -> Optional[float]:
            mask = (df["section"] == section) & (df["metric"] == metric)
            matches = df[mask]
            if len(matches) == 0:
                return None
            val = matches.iloc[0].get("holdout_value")
            return float(val) if pd.notna(val) else None

        kpis["holdout_sharpe"] = get_value("BACKTEST", "net_sharpe")
        kpis["holdout_mdd"] = get_value("BACKTEST", "net_mdd")
        kpis["holdout_cagr"] = get_value("BACKTEST", "net_cagr")
        kpis["avg_turnover"] = get_value("BACKTEST", "avg_turnover_oneway")
        kpis["cost_bps_used"] = get_value("BACKTEST", "cost_bps_used")

        kpis["metric_source"] = "kpi_table__{run_tag}.csv"
    except Exception as e:
        logger.warning(f"[KPI 추출] kpi_table 읽기 실패: {e}")
        kpis["metric_source"] = f"ERROR: {e}"

    return kpis

def extract_ranking_kpi(
    run_tag: str,
    base_dir: Path,
) -> Dict[str, Any]:
    """
    ranking KPI 추출 (sector_concentration CSV에서)

    Returns:
        Ranking KPI 딕셔너리
    """
    kpis = {}

    sector_csv = base_dir / "reports" / "ranking" / f"sector_concentration__{run_tag}.csv"
    if not sector_csv.exists():
        return kpis

    try:
        df = pd.read_csv(sector_csv)

        if "hhi" in df.columns:
            kpis["mean_hhi"] = float(df["hhi"].mean())
        if "max_sector_share" in df.columns:
            kpis["mean_max_sector_share"] = float(df["max_sector_share"].mean())
        if "n_sectors" in df.columns:
            kpis["mean_n_sectors"] = float(df["n_sectors"].mean())
    except Exception as e:
        logger.warning(f"[KPI 추출] sector_concentration 읽기 실패: {e}")

    return kpis

def collect_stage_kpis(
    stage_no: int,
    run_tag: str,
    base_dir: Path,
    track: str,
) -> Dict[str, Any]:
    """
    Stage별 KPI 수집

    Returns:
        KPI 딕셔너리
    """
    kpis = {
        "stage_no": stage_no,
        "run_tag": run_tag,
        "track": track,
    }

    # Pipeline track: 백테스트 KPI 추출
    if track == "pipeline":
        # bt_metrics 우선 시도
        bt_kpis = extract_kpi_from_bt_metrics(run_tag, base_dir)
        if bt_kpis.get("metric_source") == "bt_metrics.parquet":
            kpis.update(bt_kpis)
        else:
            # kpi_table 시도
            kpi_table_kpis = extract_kpi_from_kpi_table(run_tag, base_dir)
            kpis.update(kpi_table_kpis)
            if "metric_source" not in kpis or not kpis["metric_source"].startswith("kpi_table"):
                kpis["metric_source"] = "NA(no backtest)"

    # Ranking track: ranking KPI 추출
    elif track == "ranking":
        ranking_kpis = extract_ranking_kpi(run_tag, base_dir)
        kpis.update(ranking_kpis)
        kpis["metric_source"] = "NA(no backtest)"  # 백테스트 없음

    return kpis

def build_evolution_csv(
    stage_results: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """
    Stage별 KPI 진화 CSV 생성

    Args:
        stage_results: Stage 실행 결과 리스트
        output_path: 출력 파일 경로
    """
    rows = []

    for result in stage_results:
        if not result["success"]:
            continue

        kpis = result.get("kpis", {})
        rows.append(kpis)

    if not rows:
        logger.warning("[진화 CSV] 데이터가 없어 CSV를 생성하지 않습니다.")
        return

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"[진화 CSV] 저장 완료: {output_path}")

def build_changes_csv(
    stage_results: List[Dict[str, Any]],
    output_path: Path,
    compare_type: str,  # "prev" or "baseline"
) -> None:
    """
    Stage별 변화량 CSV 생성

    Args:
        stage_results: Stage 실행 결과 리스트
        output_path: 출력 파일 경로
        compare_type: "prev" (이전 Stage) 또는 "baseline" (기준 baseline)
    """
    rows = []

    baseline_kpis = None

    for i, result in enumerate(stage_results):
        if not result["success"]:
            continue

        kpis = result.get("kpis", {})

        if compare_type == "baseline":
            # 첫 번째 pipeline track Stage를 baseline으로 사용
            if baseline_kpis is None and kpis.get("track") == "pipeline":
                baseline_kpis = kpis.copy()
                continue

            if baseline_kpis is None:
                continue

        elif compare_type == "prev":
            # 이전 Stage의 KPI 사용
            if i == 0:
                baseline_kpis = kpis.copy()
                continue

        if baseline_kpis is None:
            continue

        # 변화량 계산
        change_row = {
            "stage_no": kpis.get("stage_no"),
            "run_tag": kpis.get("run_tag"),
            "baseline_stage_no": baseline_kpis.get("stage_no"),
            "baseline_run_tag": baseline_kpis.get("run_tag"),
        }

        # 백테스트 KPI 변화량
        for metric in ["holdout_sharpe", "holdout_mdd", "holdout_cagr", "avg_turnover", "cost_bps_used"]:
            current_val = kpis.get(metric)
            baseline_val = baseline_kpis.get(metric)

            if current_val is not None and baseline_val is not None:
                change_row[f"{metric}_current"] = current_val
                change_row[f"{metric}_baseline"] = baseline_val
                change_row[f"{metric}_delta"] = current_val - baseline_val
                if baseline_val != 0:
                    change_row[f"{metric}_pct_change"] = (current_val - baseline_val) / abs(baseline_val) * 100

        # Ranking KPI 변화량
        for metric in ["mean_hhi", "mean_max_sector_share", "mean_n_sectors"]:
            current_val = kpis.get(metric)
            baseline_val = baseline_kpis.get(metric)

            if current_val is not None and baseline_val is not None:
                change_row[f"{metric}_current"] = current_val
                change_row[f"{metric}_baseline"] = baseline_val
                change_row[f"{metric}_delta"] = current_val - baseline_val

        rows.append(change_row)

        # 다음 비교를 위해 업데이트
        if compare_type == "prev":
            baseline_kpis = kpis.copy()

    if not rows:
        logger.warning(f"[변화량 CSV] 데이터가 없어 CSV를 생성하지 않습니다. (type: {compare_type})")
        return

    df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"[변화량 CSV] 저장 완료: {output_path}")

def build_summary_md(
    stage_results: List[Dict[str, Any]],
    plan: Dict[str, Any],
    output_path: Path,
    base_dir: Path,
) -> None:
    """
    최종 요약 MD 생성

    Args:
        stage_results: Stage 실행 결과 리스트
        plan: plan.yaml 내용
        output_path: 출력 파일 경로
        base_dir: 프로젝트 루트 디렉토리
    """
    lines = []
    lines.append("# Stage0~13 리플레이 실행 요약 리포트\n")
    lines.append(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("\n---\n")

    # 전체 실행 통계
    total_stages = len(stage_results)
    success_stages = sum(1 for r in stage_results if r["success"])
    failed_stages = total_stages - success_stages

    lines.append("## 전체 실행 통계\n")
    lines.append(f"- 총 Stage 수: {total_stages}")
    lines.append(f"- 성공: {success_stages}")
    lines.append(f"- 실패: {failed_stages}")
    lines.append("\n")

    # Stage별 개선점 및 수치표
    lines.append("## Stage별 개선점 및 정확한 수치표\n")
    lines.append("\n")

    for result in stage_results:
        if not result["success"]:
            continue

        stage_no = result["stage_no"]
        run_tag = result["run_tag"]
        baseline_tag = result["baseline_tag"]
        kpis = result.get("kpis", {})
        track = kpis.get("track", "unknown")

        # Stage 정의 찾기
        stage_def = None
        for s in plan["stages"]:
            if s["stage_no"] == stage_no:
                stage_def = s
                break

        if stage_def is None:
            continue

        lines.append(f"### Stage {stage_no}: {stage_def['notes']}\n")
        lines.append(f"**Run Tag**: `{run_tag}`\n")
        lines.append(f"**Baseline Tag**: `{baseline_tag}`\n")
        lines.append(f"**Track**: {track}\n")
        lines.append("\n")

        # 개선점 (notes)
        lines.append(f"**개선점**: {stage_def['notes']}\n")
        lines.append("\n")

        # 정확한 수치표
        lines.append("**정확한 수치표**:\n")
        lines.append("\n")

        if track == "pipeline":
            # 백테스트 KPI
            lines.append("| 지표 | 값 | 출처 |\n")
            lines.append("|------|-----|------|\n")

            metric_source = kpis.get("metric_source", "N/A")

            for metric, label in [
                ("holdout_sharpe", "Holdout Sharpe"),
                ("holdout_mdd", "Holdout MDD"),
                ("holdout_cagr", "Holdout CAGR"),
                ("avg_turnover", "평균 Turnover"),
                ("cost_bps_used", "사용된 Cost (bps)"),
                ("n_rebalances", "리밸런싱 횟수"),
            ]:
                val = kpis.get(metric)
                if val is not None:
                    lines.append(f"| {label} | {val:.4f} | {metric_source} |\n")
                else:
                    lines.append(f"| {label} | N/A | {metric_source} |\n")

        elif track == "ranking":
            # Ranking KPI
            lines.append("| 지표 | 값 | 출처 |\n")
            lines.append("|------|-----|------|\n")

            for metric, label in [
                ("mean_hhi", "평균 HHI"),
                ("mean_max_sector_share", "평균 최대 섹터 비중"),
                ("mean_n_sectors", "평균 섹터 수"),
            ]:
                val = kpis.get(metric)
                if val is not None:
                    lines.append(f"| {label} | {val:.4f} | sector_concentration CSV |\n")
                else:
                    lines.append(f"| {label} | N/A | sector_concentration CSV |\n")

            lines.append(f"| 백테스트 KPI | N/A | NA(no backtest) |\n")

        lines.append("\n")

    # 리포트 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"[요약 MD] 저장 완료: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="[TASK B-1] Stage0~13 리플레이 자동화 스크립트"
    )
    parser.add_argument(
        "--plan",
        type=str,
        default="configs/stage_replay_plan.yaml",
        help="리플레이 계획 파일 경로",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Config 파일 경로",
    )
    args = parser.parse_args()

    # 경로 확인
    base_dir = PROJECT_ROOT
    plan_path = base_dir / args.plan
    config_path = base_dir / args.config

    if not plan_path.exists():
        logger.error(f"Plan 파일을 찾을 수 없습니다: {plan_path}")
        sys.exit(1)

    if not config_path.exists():
        logger.error(f"Config 파일을 찾을 수 없습니다: {config_path}")
        sys.exit(1)

    # Plan 로드
    plan = load_plan(plan_path)
    logger.info(f"Plan 로드 완료: {plan_path}")
    logger.info(f"Baseline Tag: {plan.get('baseline_tag', 'N/A')} (비교 기준)")
    logger.info(f"Input Tag (기본값): {plan.get('default_input_tag', 'N/A')} (입력 산출물 소스)")
    logger.info(f"총 {len(plan['stages'])}개 Stage")

    # Stage별 실행
    stage_results = []
    total_start_time = time.time()

    for stage_def in plan["stages"]:
        stage_start_time = time.time()

        success, run_tag, baseline_tag, elapsed = run_stage(
            stage_def=stage_def,
            config_path=config_path,
            base_dir=base_dir,
            plan=plan,
        )

        # 산출물 존재 체크
        artifact_checks = {}
        if success:
            artifact_checks = check_artifacts(run_tag, base_dir, stage_def["track"])
            logger.info(f"[산출물 체크] {artifact_checks}")

        # KPI 수집
        kpis = {}
        if success:
            kpis = collect_stage_kpis(
                stage_def["stage_no"],
                run_tag,
                base_dir,
                stage_def["track"],
            )
            logger.info(f"[KPI 수집] 완료: {kpis.get('metric_source', 'N/A')}")

        stage_results.append({
            "stage_no": stage_def["stage_no"],
            "run_tag": run_tag,
            "baseline_tag": baseline_tag,
            "success": success,
            "elapsed_time": elapsed,
            "artifact_checks": artifact_checks,
            "kpis": kpis,
        })

        if not success:
            logger.error(f"[중단] Stage {stage_def['stage_no']} 실행 실패로 인해 리플레이를 중단합니다.")
            break

    total_elapsed = time.time() - total_start_time

    # CSV 생성
    analysis_dir = base_dir / "reports" / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    # Stage 진화 CSV
    evolution_csv = analysis_dir / "stage_evolution_0_13.csv"
    build_evolution_csv(stage_results, evolution_csv)

    # 변화량 CSV (vs-prev)
    changes_prev_csv = analysis_dir / "stage_changes_vs_prev_0_13.csv"
    build_changes_csv(stage_results, changes_prev_csv, "prev")

    # 변화량 CSV (vs-baseline)
    changes_baseline_csv = analysis_dir / "stage_changes_vs_baseline_0_13.csv"
    build_changes_csv(stage_results, changes_baseline_csv, "baseline")

    # 최종 요약 MD
    summary_md = analysis_dir / "stage_replay_summary_0_13.md"
    build_summary_md(stage_results, plan, summary_md, base_dir)

    logger.info(f"\n{'='*80}")
    logger.info(f"리플레이 완료: 총 {total_elapsed:.1f}초")
    logger.info(f"진화 CSV: {evolution_csv}")
    logger.info(f"변화량 CSV (vs-prev): {changes_prev_csv}")
    logger.info(f"변화량 CSV (vs-baseline): {changes_baseline_csv}")
    logger.info(f"요약 MD: {summary_md}")
    logger.info(f"{'='*80}\n")

if __name__ == "__main__":
    main()
