# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/maintenance/replay_stages.py
"""
Stage0~13 리플레이 자동화 스크립트

[기능]
1. Stage0~13을 순차적으로 실행
2. 각 Stage 실행 후 KPI, Delta, Check 리포트 자동 생성
3. Stage별 변화 요약 리포트 생성 (개선점 + 수치 변화)

[규칙]
- L2는 재생성하지 않음 (--skip-l2 기본)
- 그 외는 --force-rebuild 기본
- baseline_tag는 Stage 성격에 맞게 자동 선택
- OneDrive 환경 고려하여 불필요한 스캔 최소화
"""
import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# 프로젝트 루트 경로 고정
PROJECT_ROOT = Path("C:/Users/seong/OneDrive/Desktop/bootcamp/03_code")

# Stage0~13 정의 (Stage 번호 -> 실행할 L Stage들)
STAGE_DEFINITIONS = {
    "stage0": {
        "name": "Stage 0: 태그 기반 저장 구조 재구축",
        "from_stage": "L0",
        "to_stage": "L0",
        "baseline_tag": "baseline_prerefresh_20251219_143636",
        "description": "L0 유니버스 재구축 및 태그 기반 저장 구조 적용",
    },
    "stage1": {
        "name": "Stage 1: 거래비용 모델 수정",
        "from_stage": "L7",
        "to_stage": "L7",
        "baseline_tag": "stage0_rebuild_tagged_20251219_220938",
        "description": "L7 백테스트에서 거래비용 모델 수정 (cost_bps 반영)",
    },
    "stage2": {
        "name": "Stage 2: 모델 설명가능성 추가",
        "from_stage": "L5",
        "to_stage": "L5",
        "baseline_tag": "stage1_cost_model_fix_20251219_221942",
        "description": "L5 모델 학습에 피처 중요도 리포트 추가",
    },
    "stage3": {
        "name": "Stage 3: Alpha 튜닝 추가",
        "from_stage": "L4",
        "to_stage": "L5",
        "baseline_tag": "stage2_explainability_20251219_224241",
        "description": "L4 내부 CV folds 생성 및 L5 Alpha 튜닝 추가",
    },
    "stage4": {
        "name": "Stage 4: 업종 분산 제약 추가",
        "from_stage": "L1B",
        "to_stage": "L7",
        "baseline_tag": "stage3_alpha_tuning_20251220_182453",
        "description": "L1B 섹터 매핑 추가 및 L7 업종 분산 제약 적용",
    },
    "stage5": {
        "name": "Stage 5: 시장 국면 기반 전략",
        "from_stage": "L1D",
        "to_stage": "L7",
        "baseline_tag": "stage4_sector_diversify_20251220_184214",
        "description": "L1D 시장 국면 계산 및 L7 시장 국면 기반 전략 적용",
    },
    "stage6": {
        "name": "Stage 6: 섹터 상대 피처 균형",
        "from_stage": "L3",
        "to_stage": "L7",
        "baseline_tag": "stage5_regime_switch_20251220_193618",
        "description": "L3 섹터 상대 피처 추가 및 L7까지 재실행",
    },
    "stage7": {
        "name": "Stage 7: Ranking 엔진 추가",
        "from_stage": "L8",
        "to_stage": "L8",
        "baseline_tag": "stage6_sector_relative_feature_balance_20251220_194928",
        "description": "L8 Ranking 엔진 추가 (백테스트 없음, 랭킹 KPI만)",
    },
    "stage8": {
        "name": "Stage 8: 섹터 상대 랭킹",
        "from_stage": "L8",
        "to_stage": "L8",
        "baseline_tag": "stage7_ranking_baseline_20251220_235149",
        "description": "L8 섹터 상대 랭킹 추가",
    },
    "stage9": {
        "name": "Stage 9: 랭킹 설명가능성",
        "from_stage": "L8",
        "to_stage": "L8",
        "baseline_tag": "stage8_sector_relative_20251221_000821",
        "description": "L8 랭킹 설명가능성 추가",
    },
    "stage10": {
        "name": "Stage 10: 시장 국면 랭킹",
        "from_stage": "L1D",
        "to_stage": "L8",
        "baseline_tag": "stage9_ranking_explainability_20251221_001912",
        "description": "L1D 시장 국면 계산 및 L8 시장 국면 랭킹 추가",
    },
    "stage11": {
        "name": "Stage 11: UI Payload Builder",
        "from_stage": "L11",
        "to_stage": "L11",
        "baseline_tag": "stage10_market_regime_20251221_004433",
        "description": "L11 UI Payload Builder 추가 (백테스트 없음)",
    },
    "stage12": {
        "name": "Stage 12: 최종 Export",
        "from_stage": "L0",
        "to_stage": "L7D",
        "baseline_tag": "stage11_ui_payload_20251221_012244",
        "description": "전체 파이프라인 최종 실행 및 Export",
    },
    "stage13": {
        "name": "Stage 13: 스모크 테스트 및 런타임 프로파일",
        "from_stage": "L7",
        "to_stage": "L7",
        "baseline_tag": "stage12_final_export_20251221_013411",
        "description": "L7 스모크 테스트 및 런타임 프로파일 추가",
    },
}

# Pipeline track (백테스트 생성): baseline은 pipeline 최신 Stage 사용
PIPELINE_TRACK_STAGES = ["stage0", "stage1", "stage2", "stage3", "stage4", "stage5", "stage6", "stage12", "stage13"]

# Ranking/UI track (백테스트 없음): baseline은 직전 Ranking/UI Stage 사용
RANKING_TRACK_STAGES = ["stage7", "stage8", "stage9", "stage10", "stage11"]

def get_latest_pipeline_baseline(base_interim_dir: Path, current_stage_num: int) -> Optional[str]:
    """
    Pipeline track의 경우 최신 pipeline Stage를 baseline으로 사용
    
    Args:
        base_interim_dir: interim 디렉토리
        current_stage_num: 현재 Stage 번호 (이전 Stage만 고려)
    
    Returns:
        최신 pipeline baseline 태그 또는 None
    """
    # 현재 Stage 이전의 pipeline track Stage들만 고려
    pipeline_stages = [s for s in PIPELINE_TRACK_STAGES if int(s.replace("stage", "")) < current_stage_num]
    
    if not pipeline_stages:
        return None
    
    # 각 pipeline Stage의 run_tag 패턴으로 검색
    candidates = []
    for stage_name in pipeline_stages:
        stage_num = stage_name.replace("stage", "")
        # stage{N}_로 시작하는 폴더 찾기
        pattern = f"stage{stage_num}_*"
        for folder in base_interim_dir.glob(pattern):
            if folder.is_dir():
                # rebalance_scores 또는 bt_returns가 있으면 pipeline track
                if (folder / "rebalance_scores.parquet").exists() or (folder / "bt_returns.parquet").exists():
                    candidates.append((folder.name, folder.stat().st_mtime))
    
    if not candidates:
        return None
    
    # 최신 순으로 정렬
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]

def get_latest_ranking_baseline(base_interim_dir: Path, current_stage_num: int) -> Optional[str]:
    """
    Ranking/UI track의 경우 최신 Ranking/UI Stage를 baseline으로 사용
    
    Args:
        base_interim_dir: interim 디렉토리
        current_stage_num: 현재 Stage 번호 (이전 Stage만 고려)
    
    Returns:
        최신 ranking baseline 태그 또는 None
    """
    # 현재 Stage 이전의 ranking track Stage들만 고려
    ranking_stages = [s for s in RANKING_TRACK_STAGES if int(s.replace("stage", "")) < current_stage_num]
    
    if not ranking_stages:
        return None
    
    # 각 ranking Stage의 run_tag 패턴으로 검색
    candidates = []
    for stage_name in ranking_stages:
        stage_num = stage_name.replace("stage", "")
        # stage{N}_로 시작하는 폴더 찾기
        pattern = f"stage{stage_num}_*"
        for folder in base_interim_dir.glob(pattern):
            if folder.is_dir():
                # ranking_daily가 있으면 ranking track
                if (folder / "ranking_daily.parquet").exists():
                    candidates.append((folder.name, folder.stat().st_mtime))
    
    if not candidates:
        return None
    
    # 최신 순으로 정렬
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]

def resolve_baseline_tag(
    stage_name: str,
    base_interim_dir: Path,
    default_baseline: str,
    skip_scan: bool = False,
) -> str:
    """
    Stage 성격에 맞게 baseline_tag 결정
    
    Args:
        stage_name: Stage 이름 (예: "stage0", "stage7")
        base_interim_dir: interim 디렉토리
        default_baseline: 기본 baseline 태그 (STAGE_DEFINITIONS에서 가져온 값)
        skip_scan: True면 스캔 건너뛰고 default_baseline 사용 (OneDrive 최적화)
    
    Returns:
        결정된 baseline 태그
    """
    if skip_scan:
        logger.info(f"[Baseline] 스캔 건너뛰기: {default_baseline} 사용")
        return default_baseline
    
    stage_num = int(stage_name.replace("stage", ""))
    
    # Pipeline track: 최신 pipeline Stage 사용
    if stage_name in PIPELINE_TRACK_STAGES:
        baseline = get_latest_pipeline_baseline(base_interim_dir, stage_num)
        if baseline:
            logger.info(f"[Baseline] Pipeline track: {baseline} 사용")
            return baseline
        else:
            logger.warning(f"[Baseline] Pipeline baseline을 찾을 수 없어 기본값 사용: {default_baseline}")
            return default_baseline
    
    # Ranking/UI track: 최신 ranking Stage 사용
    elif stage_name in RANKING_TRACK_STAGES:
        baseline = get_latest_ranking_baseline(base_interim_dir, stage_num)
        if baseline:
            logger.info(f"[Baseline] Ranking track: {baseline} 사용")
            return baseline
        else:
            logger.warning(f"[Baseline] Ranking baseline을 찾을 수 없어 기본값 사용: {default_baseline}")
            return default_baseline
    
    # 기본값 사용
    logger.info(f"[Baseline] 기본값 사용: {default_baseline}")
    return default_baseline

def generate_run_tag(stage_name: str) -> str:
    """Stage 실행용 run_tag 생성"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stage_num = stage_name.replace("stage", "")
    return f"{stage_name}_{timestamp}"

def run_stage(
    stage_name: str,
    stage_def: Dict,
    config_path: Path,
    base_dir: Path,
    skip_l2: bool = True,
    force_rebuild: bool = True,
    skip_scan: bool = False,
    max_rebalances: Optional[int] = None,
) -> Tuple[bool, str, Optional[str]]:
    """
    단일 Stage 실행
    
    Returns:
        (성공 여부, run_tag, baseline_tag)
    """
    run_tag = generate_run_tag(stage_name)
    baseline_tag = resolve_baseline_tag(
        stage_name,
        base_dir / "data" / "interim",
        stage_def["baseline_tag"],
        skip_scan=skip_scan,
    )
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Stage 실행 시작: {stage_def['name']}")
    logger.info(f"Run Tag: {run_tag}")
    logger.info(f"Baseline Tag: {baseline_tag}")
    logger.info(f"From: {stage_def['from_stage']}, To: {stage_def['to_stage']}")
    logger.info(f"{'='*80}\n")
    
    # run_all.py 실행 명령 구성
    cmd = [
        sys.executable,
        str(base_dir / "src" / "run_all.py"),
        "--from", stage_def["from_stage"],
        "--to", stage_def["to_stage"],
        "--run-tag", run_tag,
        "--baseline-tag", baseline_tag,
        "--config", str(config_path),
    ]
    
    if skip_l2:
        cmd.append("--skip-l2")
    
    if force_rebuild:
        cmd.append("--force-rebuild")
    
    if max_rebalances:
        cmd.extend(["--max-rebalances", str(max_rebalances)])
    
    # 실행
    logger.info(f"[실행] 명령: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=str(base_dir),
            capture_output=True,
            text=True,
            check=True,
        )
        elapsed = time.time() - start_time
        logger.info(f"[성공] Stage 실행 완료: {elapsed:.1f}초")
        logger.info(f"[성공] Run Tag: {run_tag}")
        return True, run_tag, baseline_tag
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        logger.error(f"[실패] Stage 실행 실패: {elapsed:.1f}초")
        logger.error(f"[실패] Return code: {e.returncode}")
        logger.error(f"[실패] stdout:\n{e.stdout}")
        logger.error(f"[실패] stderr:\n{e.stderr}")
        return False, run_tag, baseline_tag

def generate_stage_summary_report(
    stage_results: List[Dict],
    output_path: Path,
    base_dir: Path,
) -> None:
    """
    Stage별 변화 요약 리포트 생성
    
    Args:
        stage_results: Stage 실행 결과 리스트
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
    
    # Stage별 상세 결과
    lines.append("## Stage별 실행 결과\n")
    lines.append("| Stage | 이름 | Run Tag | Baseline Tag | 성공 여부 | 실행 시간 (초) |")
    lines.append("|-------|------|---------|--------------|-----------|----------------|")
    
    for result in stage_results:
        stage_name = result["stage_name"]
        stage_def = STAGE_DEFINITIONS[stage_name]
        success = "✓" if result["success"] else "✗"
        elapsed = result.get("elapsed_time", 0)
        lines.append(
            f"| {stage_name} | {stage_def['name']} | {result['run_tag']} | "
            f"{result['baseline_tag']} | {success} | {elapsed:.1f} |"
        )
    
    lines.append("\n")
    
    # Stage별 개선점 및 수치 변화 (KPI 리포트에서 추출)
    lines.append("## Stage별 개선점 및 수치 변화\n")
    lines.append("> 주의: 수치 변화는 KPI 리포트와 Delta 리포트를 참조하세요.\n")
    lines.append("> 자동 추출 기능은 향후 구현 예정입니다.\n")
    lines.append("\n")
    
    for result in stage_results:
        if not result["success"]:
            continue
        
        stage_name = result["stage_name"]
        stage_def = STAGE_DEFINITIONS[stage_name]
        run_tag = result["run_tag"]
        baseline_tag = result["baseline_tag"]
        
        lines.append(f"### {stage_name}: {stage_def['name']}\n")
        lines.append(f"**Run Tag**: `{run_tag}`\n")
        lines.append(f"**Baseline Tag**: `{baseline_tag}`\n")
        lines.append(f"**설명**: {stage_def['description']}\n")
        lines.append("\n")
        
        # KPI 리포트 링크
        kpi_csv = base_dir / "reports" / "kpi" / f"kpi_table__{run_tag}.csv"
        kpi_md = base_dir / "reports" / "kpi" / f"kpi_table__{run_tag}.md"
        delta_csv = base_dir / "reports" / "delta" / f"delta_kpi__{baseline_tag}__vs__{run_tag}.csv"
        delta_md = base_dir / "reports" / "delta" / f"delta_report__{baseline_tag}__vs__{run_tag}.md"
        check_md = base_dir / "reports" / "stages" / f"check__{stage_name}__{run_tag}.md"
        
        lines.append("**리포트 파일**:\n")
        if kpi_csv.exists():
            lines.append(f"- KPI CSV: `{kpi_csv.relative_to(base_dir)}`\n")
        if kpi_md.exists():
            lines.append(f"- KPI MD: `{kpi_md.relative_to(base_dir)}`\n")
        if delta_csv.exists():
            lines.append(f"- Delta CSV: `{delta_csv.relative_to(base_dir)}`\n")
        if delta_md.exists():
            lines.append(f"- Delta MD: `{delta_md.relative_to(base_dir)}`\n")
        if check_md.exists():
            lines.append(f"- Check MD: `{check_md.relative_to(base_dir)}`\n")
        
        lines.append("\n")
    
    # 리포트 저장
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"[리포트] Stage 요약 리포트 저장: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Stage0~13 리플레이 자동화 스크립트"
    )
    parser.add_argument(
        "--from-stage",
        type=str,
        default="stage0",
        help="시작 Stage (기본: stage0)",
    )
    parser.add_argument(
        "--to-stage",
        type=str,
        default="stage13",
        help="종료 Stage (기본: stage13)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Config 파일 경로",
    )
    parser.add_argument(
        "--skip-l2",
        action="store_true",
        default=True,
        help="L2 Stage 스킵 (기본: True)",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        default=True,
        help="강제 Rebuild (기본: True)",
    )
    parser.add_argument(
        "--skip-scan",
        action="store_true",
        default=False,
        help="Baseline 스캔 건너뛰기 (OneDrive 최적화, 기본: False)",
    )
    parser.add_argument(
        "--max-rebalances",
        type=int,
        default=None,
        help="스모크 테스트용: 최근 N개 리밸런싱만 실행 (Stage13만 적용)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="요약 리포트 출력 경로 (기본: reports/analysis/stage_replay_summary__{timestamp}.md)",
    )
    args = parser.parse_args()
    
    # 경로 확인
    base_dir = PROJECT_ROOT
    config_path = base_dir / args.config
    
    if not config_path.exists():
        logger.error(f"Config 파일을 찾을 수 없습니다: {config_path}")
        sys.exit(1)
    
    # Stage 범위 확인
    stage_names = sorted(STAGE_DEFINITIONS.keys(), key=lambda x: int(x.replace("stage", "")))
    from_idx = stage_names.index(args.from_stage) if args.from_stage in stage_names else 0
    to_idx = stage_names.index(args.to_stage) if args.to_stage in stage_names else len(stage_names) - 1
    
    if from_idx > to_idx:
        logger.error(f"잘못된 Stage 범위: {args.from_stage} > {args.to_stage}")
        sys.exit(1)
    
    target_stages = stage_names[from_idx : to_idx + 1]
    
    logger.info(f"리플레이 대상 Stage: {target_stages}")
    logger.info(f"Config: {config_path}")
    logger.info(f"Skip L2: {args.skip_l2}")
    logger.info(f"Force Rebuild: {args.force_rebuild}")
    logger.info(f"Skip Scan: {args.skip_scan}")
    
    # Stage별 실행
    stage_results = []
    total_start_time = time.time()
    
    for stage_name in target_stages:
        stage_def = STAGE_DEFINITIONS[stage_name]
        stage_start_time = time.time()
        
        # Stage13만 max_rebalances 적용
        max_rebalances = args.max_rebalances if stage_name == "stage13" else None
        
        success, run_tag, baseline_tag = run_stage(
            stage_name=stage_name,
            stage_def=stage_def,
            config_path=config_path,
            base_dir=base_dir,
            skip_l2=args.skip_l2,
            force_rebuild=args.force_rebuild,
            skip_scan=args.skip_scan,
            max_rebalances=max_rebalances,
        )
        
        stage_elapsed = time.time() - stage_start_time
        
        stage_results.append({
            "stage_name": stage_name,
            "run_tag": run_tag,
            "baseline_tag": baseline_tag,
            "success": success,
            "elapsed_time": stage_elapsed,
        })
        
        if not success:
            logger.error(f"[중단] {stage_name} 실행 실패로 인해 리플레이를 중단합니다.")
            break
    
    total_elapsed = time.time() - total_start_time
    
    # 요약 리포트 생성
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = base_dir / "reports" / "analysis" / f"stage_replay_summary__{timestamp}.md"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generate_stage_summary_report(stage_results, output_path, base_dir)
    
    logger.info(f"\n{'='*80}")
    logger.info(f"리플레이 완료: 총 {total_elapsed:.1f}초")
    logger.info(f"요약 리포트: {output_path}")
    logger.info(f"{'='*80}\n")

if __name__ == "__main__":
    main()
