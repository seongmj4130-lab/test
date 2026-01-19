"""
가중치 방식(equal vs softmax) 비교 최적화 스크립트

4가지 전략 모두에 대해 equal과 softmax 두 가지 가중치 방식을 비교합니다.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import yaml
from copy import deepcopy
import os

from src.utils.config import load_config, get_path
from src.utils.io import load_artifact, artifact_exists, save_artifact
from src.pipeline.track_b_pipeline import run_track_b_pipeline

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

def safe_format(value, fmt_str):
    return fmt_str.format(value) if pd.notna(value) else "N/A"

def convert_paths(obj):
    """Path 객체를 문자열로 변환"""
    if isinstance(obj, dict):
        return {k: convert_paths(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_paths(item) for item in obj]
    elif isinstance(obj, Path):
        return str(obj)
    else:
        return obj

def run_backtest_with_weighting(
    strategy: str,
    weighting: str,
    softmax_temperature: float = 0.5,
    config_path: str = "configs/config.yaml",
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    특정 weighting 방식으로 백테스트 실행
    
    Args:
        strategy: 전략명 (bt20_short, bt20_ens, bt120_long, bt120_ens)
        weighting: 가중치 방식 ('equal' 또는 'softmax')
        softmax_temperature: softmax 온도 파라미터 (softmax일 때만 사용)
        config_path: 설정 파일 경로
    """
    try:
        cfg = load_config(config_path)
        
        # 전략별 설정 섹션 찾기
        strategy_config_key = f"l7_{strategy}"
        if strategy_config_key not in cfg:
            logger.error(f"[{strategy}] 설정 섹션을 찾을 수 없습니다: {strategy_config_key}")
            return (None, None, None, None, None, None)
        
        # weighting 설정 수정
        cfg[strategy_config_key]["weighting"] = weighting
        if weighting == "softmax":
            cfg[strategy_config_key]["softmax_temperature"] = softmax_temperature
        
        # Config를 YAML로 저장하기 위해 Path 객체 변환
        cfg_serializable = convert_paths(deepcopy(cfg))
        
        # 임시 config 파일 생성
        base_dir = Path(get_path(cfg, "base_dir"))
        temp_config_path = base_dir / f"configs/config_temp_weighting_{strategy}_{weighting}.yaml"
        temp_config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(temp_config_path, 'w', encoding='utf-8') as f:
            yaml.dump(cfg_serializable, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        try:
            result = run_track_b_pipeline(
                config_path=str(temp_config_path),
                strategy=strategy,
                force_rebuild=False,  # L6R, L7만 재실행
            )
            
            if result is None:
                logger.warning(f"[{strategy}] weighting={weighting}: 결과가 None")
                return (None, None, None, None, None, None)
            
            bt_metrics = result.get("bt_metrics")
            if bt_metrics is None or bt_metrics.empty:
                logger.warning(f"[{strategy}] weighting={weighting}: 메트릭 없음")
                return (None, None, None, None, None, None)
            
            dev_metrics = bt_metrics[bt_metrics["phase"] == "dev"]
            holdout_metrics = bt_metrics[bt_metrics["phase"] == "holdout"]
            
            if dev_metrics.empty or holdout_metrics.empty:
                logger.warning(f"[{strategy}] weighting={weighting}: Dev/Holdout 구분 실패")
                return (None, None, None, None, None, None)
            
            total_return_dev = float(dev_metrics.iloc[0]["net_total_return"]) if "net_total_return" in dev_metrics.columns else None
            total_return_holdout = float(holdout_metrics.iloc[0]["net_total_return"]) if "net_total_return" in holdout_metrics.columns else None
            cagr_dev = float(dev_metrics.iloc[0]["net_cagr"]) if "net_cagr" in dev_metrics.columns else None
            cagr_holdout = float(holdout_metrics.iloc[0]["net_cagr"]) if "net_cagr" in holdout_metrics.columns else None
            sharpe_dev = float(dev_metrics.iloc[0]["net_sharpe"]) if "net_sharpe" in dev_metrics.columns else None
            sharpe_holdout = float(holdout_metrics.iloc[0]["net_sharpe"]) if "net_sharpe" in holdout_metrics.columns else None
            
            return (total_return_dev, total_return_holdout, cagr_dev, cagr_holdout, sharpe_dev, sharpe_holdout)
            
        finally:
            if os.path.exists(temp_config_path):
                os.unlink(temp_config_path)
                
    except Exception as e:
        logger.error(f"[{strategy}] weighting={weighting} 실행 실패: {type(e).__name__}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return (None, None, None, None, None, None)

def compare_weighting_methods(
    strategies: List[str],
    weighting_methods: List[str] = ["equal", "softmax"],
    softmax_temperature: float = 0.5,
    config_path: str = "configs/config.yaml",
    n_jobs: int = -1,
) -> Dict:
    """
    여러 전략에 대해 가중치 방식 비교
    
    Args:
        strategies: 전략 리스트
        weighting_methods: 비교할 가중치 방식 리스트
        softmax_temperature: softmax 온도 파라미터
    """
    logger.info("=" * 80)
    logger.info("가중치 방식(equal vs softmax) 비교 최적화 시작")
    logger.info("=" * 80)
    logger.info(f"전략: {strategies}")
    logger.info(f"가중치 방식: {weighting_methods}")
    if "softmax" in weighting_methods:
        logger.info(f"softmax_temperature: {softmax_temperature}")
    
    all_results = []
    if n_jobs == -1:
        n_jobs = None
    
    # 모든 조합 실행
    futures = {}
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        for strategy in strategies:
            for weighting in weighting_methods:
                future = executor.submit(
                    run_backtest_with_weighting,
                    strategy,
                    weighting,
                    softmax_temperature,
                    config_path
                )
                futures[future] = (strategy, weighting)
        
        for future in as_completed(futures):
            strategy, weighting = futures[future]
            try:
                total_return_dev, total_return_holdout, cagr_dev, cagr_holdout, sharpe_dev, sharpe_holdout = future.result()
                all_results.append({
                    "strategy": strategy,
                    "weighting": weighting,
                    "softmax_temperature": softmax_temperature if weighting == "softmax" else None,
                    "total_return_dev": total_return_dev,
                    "total_return_holdout": total_return_holdout,
                    "cagr_dev": cagr_dev,
                    "cagr_holdout": cagr_holdout,
                    "sharpe_dev": sharpe_dev,
                    "sharpe_holdout": sharpe_holdout,
                })
                logger.info(f"  [{strategy}] {weighting}: Dev TR={safe_format(total_return_dev, '{:.4f}')}, "
                          f"Holdout TR={safe_format(total_return_holdout, '{:.4f}')}, "
                          f"Dev Sharpe={safe_format(sharpe_dev, '{:.4f}')}, "
                          f"Holdout Sharpe={safe_format(sharpe_holdout, '{:.4f}')}")
            except Exception as e:
                logger.error(f"  [{strategy}] {weighting} 실행 중 오류: {e}")
    
    df_results = pd.DataFrame(all_results)
    
    # 전략별로 최적 weighting 방식 찾기 (Holdout Total Return 기준)
    best_weighting = {}
    for strategy in strategies:
        df_strategy = df_results[df_results["strategy"] == strategy].copy()
        df_valid = df_strategy[df_strategy["total_return_holdout"].notna()].copy()
        
        if df_valid.empty:
            logger.warning(f"[{strategy}] 유효한 결과가 없습니다.")
            best_weighting[strategy] = None
            continue
        
        best_idx = df_valid["total_return_holdout"].idxmax()
        best_w = df_valid.loc[best_idx, "weighting"]
        best_score = float(df_valid.loc[best_idx, "total_return_holdout"])
        best_weighting[strategy] = {
            "weighting": best_w,
            "score": best_score,
        }
        logger.info(f"[{strategy}] 최적 weighting: {best_w} (Holdout Total Return: {best_score:.4f})")
    
    return {
        "best_weighting": best_weighting,
        "all_results": df_results,
    }

def generate_comparison_report(result: Dict, config_path: str):
    """비교 리포트 생성"""
    cfg = load_config(config_path)
    reports_dir = Path(get_path(cfg, "artifacts_reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = reports_dir / "weighting_comparison_optimization_report.md"
    
    df_results = result["all_results"]
    best_weighting = result["best_weighting"]
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# 가중치 방식(equal vs softmax) 비교 최적화 결과\n\n")
        f.write(f"생성 시간: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## 실험 조건\n\n")
        f.write(f"- **전략**: bt20_short, bt20_ens, bt120_long, bt120_ens (4가지)\n")
        f.write(f"- **가중치 방식**: equal, softmax (2가지)\n")
        f.write(f"- **softmax_temperature**: 0.5\n")
        f.write(f"- **비교 기준**: Holdout 구간 Total Return\n\n")
        f.write(f"## 최적화 결과\n\n")
        
        f.write(f"### 전략별 최적 가중치 방식\n\n")
        f.write(f"| 전략 | 최적 Weighting | Holdout Total Return | Holdout Sharpe | Holdout CAGR |\n")
        f.write(f"|------|---------------|---------------------|----------------|--------------|\n")
        
        for strategy in ["bt20_short", "bt20_ens", "bt120_long", "bt120_ens"]:
            if strategy in best_weighting and best_weighting[strategy] is not None:
                best_w = best_weighting[strategy]["weighting"]
                best_row = df_results[(df_results["strategy"] == strategy) & 
                                    (df_results["weighting"] == best_w)].iloc[0]
                f.write(f"| {strategy} | **{best_w}** | "
                       f"{safe_format(best_row['total_return_holdout'], '{:.4f}')} | "
                       f"{safe_format(best_row['sharpe_holdout'], '{:.4f}')} | "
                       f"{safe_format(best_row['cagr_holdout'], '{:.4%}')} |\n")
            else:
                f.write(f"| {strategy} | N/A | N/A | N/A | N/A |\n")
        f.write("\n")
        
        f.write(f"## 전체 결과 비교\n\n")
        for strategy in ["bt20_short", "bt20_ens", "bt120_long", "bt120_ens"]:
            f.write(f"### {strategy}\n\n")
            df_strategy = df_results[df_results["strategy"] == strategy].copy()
            
            f.write(f"| Weighting | Net Sharpe (Holdout) | Net CAGR (Holdout) | Net Total Return (Holdout) | "
                   f"Net Sharpe (Dev) | Net CAGR (Dev) | Net Total Return (Dev) |\n")
            f.write(f"|-----------|---------------------|-------------------|---------------------------|"
                   f"----------------|----------------|----------------------|\n")
            
            for _, row in df_strategy.iterrows():
                f.write(f"| {row['weighting']} | "
                       f"{safe_format(row['sharpe_holdout'], '{:.4f}')} | "
                       f"{safe_format(row['cagr_holdout'], '{:.4%}')} | "
                       f"{safe_format(row['total_return_holdout'], '{:.4f}')} | "
                       f"{safe_format(row['sharpe_dev'], '{:.4f}')} | "
                       f"{safe_format(row['cagr_dev'], '{:.4%}')} | "
                       f"{safe_format(row['total_return_dev'], '{:.4f}')} |\n")
            f.write("\n")
        
        f.write(f"## 결론\n\n")
        f.write(f"### 주요 발견사항\n\n")
        
        # 전략별 비교 요약
        for strategy in ["bt20_short", "bt20_ens", "bt120_long", "bt120_ens"]:
            df_strategy = df_results[df_results["strategy"] == strategy].copy()
            if len(df_strategy) >= 2:
                equal_row = df_strategy[df_strategy["weighting"] == "equal"]
                softmax_row = df_strategy[df_strategy["weighting"] == "softmax"]
                
                if not equal_row.empty and not softmax_row.empty:
                    equal_tr = equal_row.iloc[0]["total_return_holdout"]
                    softmax_tr = softmax_row.iloc[0]["total_return_holdout"]
                    
                    if pd.notna(equal_tr) and pd.notna(softmax_tr):
                        diff = softmax_tr - equal_tr
                        diff_pct = (diff / abs(equal_tr)) * 100 if equal_tr != 0 else 0
                        better = "softmax" if diff > 0 else "equal"
                        f.write(f"- **{strategy}**: {better}가 더 우수 (차이: {diff:.4f}, {diff_pct:+.2f}%)\n")
        
        f.write("\n")
        f.write(f"### 권장사항\n\n")
        f.write(f"- 각 전략별로 최적 가중치 방식을 선택하여 config.yaml에 반영 권장\n")
        f.write(f"- Holdout 구간 성과를 기준으로 최적화 완료\n")
    
    logger.info(f"비교 리포트 생성 완료: {report_path}")

def main():
    config_path = "configs/config.yaml"
    
    # 실험 조건
    strategies = ["bt20_short", "bt20_ens", "bt120_long", "bt120_ens"]
    weighting_methods = ["equal", "softmax"]
    softmax_temperature = 0.5
    
    logger.info("=" * 80)
    logger.info("가중치 방식(equal vs softmax) 비교 최적화 시작")
    logger.info("=" * 80)
    logger.info(f"전략: {strategies}")
    logger.info(f"가중치 방식: {weighting_methods}")
    logger.info(f"softmax_temperature: {softmax_temperature}")
    
    result = compare_weighting_methods(
        strategies=strategies,
        weighting_methods=weighting_methods,
        softmax_temperature=softmax_temperature,
        config_path=config_path,
        n_jobs=-1,
    )
    
    # 결과 저장
    cfg = load_config(config_path)
    reports_dir = Path(get_path(cfg, "artifacts_reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    result_path = reports_dir / "weighting_comparison_optimization_results.csv"
    result["all_results"].to_csv(result_path, index=False, encoding="utf-8-sig")
    logger.info(f"\n결과 저장: {result_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("최적화 결과 요약")
    logger.info("=" * 80)
    
    for strategy in strategies:
        if strategy in result["best_weighting"] and result["best_weighting"][strategy] is not None:
            best_w = result["best_weighting"][strategy]["weighting"]
            best_score = result["best_weighting"][strategy]["score"]
            logger.info(f"[{strategy}] 최적 weighting: {best_w} (Holdout Total Return: {best_score:.4f})")
    
    print("\n전체 결과:")
    print(result["all_results"].to_string(index=False))
    
    # 리포트 생성
    generate_comparison_report(result, config_path)

if __name__ == "__main__":
    main()

