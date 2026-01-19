# -*- coding: utf-8 -*-
"""
Alpha 최적화 스크립트 (Grid Search)

BT20: Total Return 중심 최적화 (수익률 위주)
BT120: Sharpe 지수 중심 최적화 (안정성 추구)
"""
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

from src.pipeline.track_b_pipeline import run_track_b_pipeline
from src.utils.config import get_path, load_config
from src.utils.io import artifact_exists, load_artifact, save_artifact

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def run_backtest_with_alpha(
    strategy: str,
    alpha_short: float,
    config_path: str = "configs/config.yaml",
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    특정 alpha 값으로 백테스트를 실행하고 성과 지표를 반환

    Args:
        strategy: 전략 이름 ("bt20_short", "bt20_ens", "bt120_long", "bt120_ens")
        alpha_short: 단기 랭킹 가중치 (0.0~1.0)
        config_path: 설정 파일 경로

    Returns:
        (total_return_dev, total_return_holdout, cagr_dev, cagr_holdout, sharpe_dev, sharpe_holdout)
    """
    import os
    import tempfile

    try:
        # 설정 로드 및 alpha 수정
        cfg = load_config(config_path)

        # base_dir을 절대 경로로 보장
        original_base_dir = get_path(cfg, "base_dir")
        if isinstance(original_base_dir, str):
            base_dir_abs = Path(original_base_dir).resolve()
        else:
            base_dir_abs = original_base_dir.resolve() if hasattr(original_base_dir, 'resolve') else Path(original_base_dir)

        # l6r 설정에 alpha_short 추가
        if "l6r" not in cfg:
            cfg["l6r"] = {}
        cfg["l6r"]["alpha_short"] = alpha_short
        cfg["l6r"]["alpha_long"] = None  # 자동 계산

        # paths.base_dir을 절대 경로로 설정
        if "paths" not in cfg:
            cfg["paths"] = {}
        cfg["paths"]["base_dir"] = str(base_dir_abs)

        # 임시 config 파일 생성
        # Path 객체를 문자열로 변환
        def convert_paths(obj):
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            elif isinstance(obj, Path):
                return str(obj)
            else:
                return obj

        cfg_serializable = convert_paths(deepcopy(cfg))

        # 임시 config 파일을 프로젝트 디렉토리에 생성 (경로 문제 방지)
        temp_config_path = base_dir_abs / f"configs/config_temp_alpha_{alpha_short:.3f}.yaml"
        temp_config_path.parent.mkdir(parents=True, exist_ok=True)

        with open(temp_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(cfg_serializable, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        try:
            # 백테스트 실행
            try:
                result = run_track_b_pipeline(
                    config_path=str(temp_config_path),  # 절대 경로로 변환
                    strategy=strategy,
                    force_rebuild=False,  # alpha 변경은 L6R에서만 재계산되므로 False
                )
            except Exception as e:
                logger.error(f"[{strategy}] alpha={alpha_short:.3f} 백테스트 실행 중 오류: {type(e).__name__}: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                return (None, None, None, None, None, None)

            if result is None:
                logger.warning(f"[{strategy}] alpha={alpha_short:.3f}: 결과가 None")
                return (None, None, None, None, None, None)

            # 성과 지표 추출
            bt_metrics = result.get("bt_metrics")
            if bt_metrics is None or bt_metrics.empty:
                logger.warning(f"[{strategy}] alpha={alpha_short:.3f}: 메트릭 없음 (bt_metrics: {bt_metrics})")
                return (None, None, None, None, None, None)

            # Dev/Holdout 구분
            dev_metrics = bt_metrics[bt_metrics["phase"] == "dev"]
            holdout_metrics = bt_metrics[bt_metrics["phase"] == "holdout"]

            if dev_metrics.empty or holdout_metrics.empty:
                logger.warning(f"[{strategy}] alpha={alpha_short:.3f}: Dev/Holdout 구분 실패")
                return (None, None, None, None, None, None)

            # Total Return 추출
            total_return_dev = float(dev_metrics.iloc[0]["net_total_return"]) if "net_total_return" in dev_metrics.columns else None
            total_return_holdout = float(holdout_metrics.iloc[0]["net_total_return"]) if "net_total_return" in holdout_metrics.columns else None

            # CAGR 추출
            cagr_dev = float(dev_metrics.iloc[0]["net_cagr"]) if "net_cagr" in dev_metrics.columns else None
            cagr_holdout = float(holdout_metrics.iloc[0]["net_cagr"]) if "net_cagr" in holdout_metrics.columns else None

            # Sharpe 추출
            sharpe_dev = float(dev_metrics.iloc[0]["net_sharpe"]) if "net_sharpe" in dev_metrics.columns else None
            sharpe_holdout = float(holdout_metrics.iloc[0]["net_sharpe"]) if "net_sharpe" in holdout_metrics.columns else None

            return (total_return_dev, total_return_holdout, cagr_dev, cagr_holdout, sharpe_dev, sharpe_holdout)

        finally:
            # 임시 파일 삭제
            try:
                if temp_config_path.exists():
                    temp_config_path.unlink()
            except Exception as e:
                logger.warning(f"임시 config 파일 삭제 실패: {e}")

    except Exception as e:
        logger.error(f"[{strategy}] alpha={alpha_short:.3f} 실행 실패: {e}")
        return (None, None, None, None, None, None)


def optimize_alpha_grid_search(
    strategy: str,
    optimization_target: str,  # "total_return", "cagr", or "sharpe"
    alpha_grid: List[float],
    config_path: str = "configs/config.yaml",
    n_jobs: int = -1,
) -> Dict:
    """
    Grid Search로 alpha 최적화

    Args:
        strategy: 전략 이름
        optimization_target: 최적화 목표 ("total_return", "cagr", or "sharpe")
        alpha_grid: 테스트할 alpha 값 리스트
        config_path: 설정 파일 경로
        n_jobs: 병렬 작업 수 (-1이면 CPU 코어 수)

    Returns:
        {
            "best_alpha": float,
            "best_score": float,
            "all_results": pd.DataFrame,
        }
    """
    logger.info(f"[{strategy}] Alpha 최적화 시작 (목표: {optimization_target})")
    logger.info(f"  Alpha 그리드: {alpha_grid}")

    # 병렬 실행
    results = []
    if n_jobs == -1:
        n_jobs = None  # ProcessPoolExecutor는 None이면 CPU 코어 수 사용

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {
            executor.submit(run_backtest_with_alpha, strategy, alpha, config_path): alpha
            for alpha in alpha_grid
        }

        for future in as_completed(futures):
            alpha = futures[future]
            try:
                total_return_dev, total_return_holdout, cagr_dev, cagr_holdout, sharpe_dev, sharpe_holdout = future.result()
                results.append({
                    "alpha": alpha,
                    "total_return_dev": total_return_dev,
                    "total_return_holdout": total_return_holdout,
                    "cagr_dev": cagr_dev,
                    "cagr_holdout": cagr_holdout,
                    "sharpe_dev": sharpe_dev,
                    "sharpe_holdout": sharpe_holdout,
                })
                # None 값 포맷팅 방지
                def safe_format(val, fmt=".4f"):
                    if val is None:
                        return "N/A"
                    try:
                        return f"{val:{fmt}}"
                    except:
                        return "N/A"

                logger.info(f"  alpha={alpha:.3f}: Dev Total Return={safe_format(total_return_dev)}, "
                          f"Holdout Total Return={safe_format(total_return_holdout)}, "
                          f"Dev CAGR={safe_format(cagr_dev)}, "
                          f"Holdout CAGR={safe_format(cagr_holdout)}, "
                          f"Dev Sharpe={safe_format(sharpe_dev)}, "
                          f"Holdout Sharpe={safe_format(sharpe_holdout)}")
            except Exception as e:
                logger.error(f"  alpha={alpha:.3f} 실행 중 오류: {e}")

    # 결과 정리
    df_results = pd.DataFrame(results)

    # 최적 alpha 선택 (Holdout 기준)
    if optimization_target == "total_return":
        target_col = "total_return_holdout"
    elif optimization_target == "cagr":
        target_col = "cagr_holdout"
    elif optimization_target == "sharpe":
        target_col = "sharpe_holdout"
    else:
        raise ValueError(f"Unknown optimization_target: {optimization_target}. Must be 'total_return', 'cagr', or 'sharpe'")

    # 유효한 결과만 필터링
    df_valid = df_results[df_results[target_col].notna()].copy()
    if df_valid.empty:
        logger.warning(f"[{strategy}] 유효한 결과가 없습니다.")
        return {
            "best_alpha": None,
            "best_score": None,
            "all_results": df_results,
        }

    # 최적 alpha 선택
    best_idx = df_valid[target_col].idxmax()
    best_alpha = float(df_valid.loc[best_idx, "alpha"])
    best_score = float(df_valid.loc[best_idx, target_col])

    logger.info(f"[{strategy}] 최적 alpha: {best_alpha:.3f} (Holdout {optimization_target.upper()}: {best_score:.4f})")

    return {
        "best_alpha": best_alpha,
        "best_score": best_score,
        "all_results": df_results,
    }


def check_prerequisites(config_path: str = "configs/config.yaml") -> bool:
    """
    최적화 실행 전 필수 데이터 확인

    Returns:
        bool: 모든 필수 데이터가 준비되어 있으면 True
    """
    cfg = load_config(config_path)
    interim_dir = Path(get_path(cfg, "data_interim"))

    # 필수 아티팩트 확인
    required_artifacts = [
        "universe_k200_membership_monthly.parquet",
        "ranking_short_daily.parquet",  # Track A 산출물
        "ranking_long_daily.parquet",   # Track A 산출물
    ]

    missing = []
    for artifact in required_artifacts:
        artifact_path = interim_dir / artifact
        if not artifact_path.exists():
            missing.append(artifact)

    if missing:
        logger.error("=" * 80)
        logger.error("필수 데이터가 준비되지 않았습니다!")
        logger.error("=" * 80)
        logger.error("누락된 파일:")
        for m in missing:
            logger.error(f"  - {m}")
        logger.error("")
        logger.error("먼저 다음을 실행하세요:")
        logger.error("  1. 공통 데이터 준비: python scripts/run_pipeline_l0_l7.py")
        logger.error("  2. Track A 실행: python -m src.pipeline.track_a_pipeline")
        logger.error("=" * 80)
        return False

    logger.info("필수 데이터 확인 완료")
    return True


def main():
    """메인 실행 함수"""
    config_path = "configs/config.yaml"

    # 필수 데이터 확인
    if not check_prerequisites(config_path):
        logger.error("최적화를 중단합니다.")
        return

    # Alpha 그리드 정의
    alpha_grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # 최적화 설정
    # BT20: 수익률 위주 (Total Return 중심)
    # BT120: 안정성 추구 (Sharpe 지수 중심)
    optimizations = [
        {
            "strategy": "bt20_short",
            "optimization_target": "total_return",
            "description": "BT20 단기 랭킹 - Total Return 최적화 (수익률 위주)",
        },
        {
            "strategy": "bt20_ens",
            "optimization_target": "total_return",
            "description": "BT20 통합 랭킹 - Total Return 최적화 (수익률 위주)",
        },
        {
            "strategy": "bt120_long",
            "optimization_target": "sharpe",
            "description": "BT120 장기 랭킹 - Sharpe 최적화 (안정성 추구)",
        },
        {
            "strategy": "bt120_ens",
            "optimization_target": "sharpe",
            "description": "BT120 통합 랭킹 - Sharpe 최적화 (안정성 추구)",
        },
    ]

    # 결과 저장
    all_results = {}
    summary = []

    for opt in optimizations:
        strategy = opt["strategy"]
        target = opt["optimization_target"]
        desc = opt["description"]

        logger.info("=" * 80)
        logger.info(f"{desc}")
        logger.info("=" * 80)

        result = optimize_alpha_grid_search(
            strategy=strategy,
            optimization_target=target,
            alpha_grid=alpha_grid,
            config_path=config_path,
            n_jobs=-1,  # 병렬 실행
        )

        all_results[strategy] = result

        summary.append({
            "strategy": strategy,
            "optimization_target": target,
            "best_alpha": result["best_alpha"],
            "best_score": result["best_score"],
        })

        # 결과 저장
        cfg = load_config(config_path)
        reports_dir = Path(get_path(cfg, "artifacts_reports"))
        reports_dir.mkdir(parents=True, exist_ok=True)

        # 상세 결과 저장
        result_path = reports_dir / f"alpha_optimization_{strategy}.csv"
        result["all_results"].to_csv(result_path, index=False, encoding="utf-8-sig")
        logger.info(f"  결과 저장: {result_path}")

    # 요약 저장
    df_summary = pd.DataFrame(summary)
    summary_path = Path(get_path(load_config(config_path), "artifacts_reports")) / "alpha_optimization_summary.csv"
    df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    logger.info(f"\n요약 저장: {summary_path}")

    # 요약 출력
    logger.info("\n" + "=" * 80)
    logger.info("최적화 결과 요약")
    logger.info("=" * 80)
    print(df_summary.to_string(index=False))

    # 최적 alpha를 config.yaml에 반영할지 물어보기
    logger.info("\n최적 alpha 값을 config.yaml에 반영하려면 수동으로 업데이트하세요:")
    for _, row in df_summary.iterrows():
        if row["best_alpha"] is not None:
            logger.info(f"  {row['strategy']}: alpha_short = {row['best_alpha']:.3f}")

    # 마크다운 리포트 생성
    generate_optimization_report(all_results, df_summary, config_path)


def generate_optimization_report(
    all_results: Dict,
    df_summary: pd.DataFrame,
    config_path: str = "configs/config.yaml",
):
    """
    최적화 결과를 마크다운 리포트로 생성

    Args:
        all_results: 모든 전략의 최적화 결과
        df_summary: 요약 데이터프레임
        config_path: 설정 파일 경로
    """
    cfg = load_config(config_path)
    reports_dir = Path(get_path(cfg, "artifacts_reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 리포트 생성
    report_lines = []
    report_lines.append("# Grid Search Alpha 최적화 결과")
    report_lines.append("")
    report_lines.append(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("")
    report_lines.append("## 최적화 전략")
    report_lines.append("- **BT20 (단기)**: Total Return 중심 최적화 (수익률 위주)")
    report_lines.append("- **BT120 (장기)**: Sharpe 지수 중심 최적화 (안정성 추구)")
    report_lines.append("")
    report_lines.append("## 최적화 결과")
    report_lines.append("")

    # 각 전략별 상세 결과
    for _, row in df_summary.iterrows():
        strategy = row["strategy"]
        target = row["optimization_target"]
        best_alpha = row["best_alpha"]
        best_score = row["best_score"]

        if strategy not in all_results:
            continue

        result = all_results[strategy]
        all_results_df = result["all_results"]

        # 최적 alpha의 전체 메트릭 조회
        best_row = None
        if best_alpha is not None and not all_results_df.empty:
            best_rows = all_results_df[all_results_df["alpha"] == best_alpha]
            if not best_rows.empty:
                best_row = best_rows.iloc[0]

        report_lines.append(f"### {strategy}")
        report_lines.append("")
        if best_alpha is not None:
            report_lines.append(f"- **최적 Alpha**: {best_alpha:.2f}")
            report_lines.append(f"- **최적화 점수**: {best_score:.4f}")

            if best_row is not None:
                # 추가 메트릭 표시
                if "net_sharpe" in best_row and pd.notna(best_row["net_sharpe"]):
                    report_lines.append(f"- **Net Sharpe**: {best_row['net_sharpe']:.4f}")
                if "net_cagr" in best_row and pd.notna(best_row["net_cagr"]):
                    report_lines.append(f"- **Net CAGR**: {best_row['net_cagr']:.4f}%")
                if "net_total_return" in best_row and pd.notna(best_row["net_total_return"]):
                    report_lines.append(f"- **Net Total Return**: {best_row['net_total_return']:.4f}")
                if "net_mdd" in best_row and pd.notna(best_row["net_mdd"]):
                    report_lines.append(f"- **Net MDD**: {best_row['net_mdd']:.4f}%")
                if "net_calmar_ratio" in best_row and pd.notna(best_row["net_calmar_ratio"]):
                    report_lines.append(f"- **Net Calmar Ratio**: {best_row['net_calmar_ratio']:.4f}")
        else:
            report_lines.append("- 최적 alpha를 찾을 수 없습니다.")

        report_lines.append("")

    # 리포트 저장
    report_path = reports_dir / "optimization_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logger.info(f"\n최적화 리포트 저장: {report_path}")


if __name__ == "__main__":
    main()
