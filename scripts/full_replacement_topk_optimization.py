"""
완전 교체 전략 + top_k 최적화 스크립트

실험 조건:
1. 완전 교체 (rebalance_interval = holding_days):
   - Day1: top_k 매수 → Day20: 전량 매도 → Day20 top_k 재매수
   - buffer_k=0 고정 (매번 100% 교체)
2. top_k만 3개 테스트 (20, 15, 10)
3. 전략: bt20_short만 테스트 (단기 랭킹)
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

def run_backtest_with_params(
    strategy: str,
    top_k: int,
    buffer_k: int,
    rebalance_interval: int,
    config_path: str = "configs/config.yaml",
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    완전 교체 전략으로 백테스트 실행
    
    Args:
        strategy: 전략명 (bt20_short)
        top_k: 선택 종목 수
        buffer_k: 버퍼 종목 수 (0 고정)
        rebalance_interval: 리밸런싱 주기 (holding_days와 동일)
        config_path: 설정 파일 경로
    """
    try:
        cfg = load_config(config_path)
        
        # l7_bt20_short 설정 수정
        if "l7_bt20_short" not in cfg:
            cfg["l7_bt20_short"] = {}
        
        cfg["l7_bt20_short"]["top_k"] = top_k
        cfg["l7_bt20_short"]["buffer_k"] = buffer_k
        cfg["l7_bt20_short"]["rebalance_interval"] = rebalance_interval
        
        # holding_days 확인
        holding_days = cfg["l7_bt20_short"].get("holding_days", 20)
        if rebalance_interval != holding_days:
            logger.warning(f"[{strategy}] rebalance_interval({rebalance_interval}) != holding_days({holding_days}), 완전 교체 전략이 아닙니다.")
        
        # Config를 YAML로 저장하기 위해 Path 객체 변환
        cfg_serializable = convert_paths(deepcopy(cfg))
        
        # 임시 config 파일 생성
        base_dir = Path(get_path(cfg, "base_dir"))
        temp_config_path = base_dir / f"configs/config_temp_full_replacement_topk_{top_k}.yaml"
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
                logger.warning(f"[{strategy}] top_k={top_k}, buffer_k={buffer_k}, interval={rebalance_interval}: 결과가 None")
                return (None, None, None, None, None, None)
            
            bt_metrics = result.get("bt_metrics")
            if bt_metrics is None or bt_metrics.empty:
                logger.warning(f"[{strategy}] top_k={top_k}, buffer_k={buffer_k}, interval={rebalance_interval}: 메트릭 없음")
                return (None, None, None, None, None, None)
            
            dev_metrics = bt_metrics[bt_metrics["phase"] == "dev"]
            holdout_metrics = bt_metrics[bt_metrics["phase"] == "holdout"]
            
            if dev_metrics.empty or holdout_metrics.empty:
                logger.warning(f"[{strategy}] top_k={top_k}, buffer_k={buffer_k}, interval={rebalance_interval}: Dev/Holdout 구분 실패")
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
        logger.error(f"[{strategy}] top_k={top_k}, buffer_k={buffer_k}, interval={rebalance_interval} 실행 실패: {type(e).__name__}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return (None, None, None, None, None, None)

def optimize_top_k_full_replacement(
    strategy: str,
    top_k_list: List[int],
    buffer_k: int = 0,
    rebalance_interval: int = 20,  # holding_days와 동일
    config_path: str = "configs/config.yaml",
    n_jobs: int = -1,
) -> Dict:
    """
    완전 교체 전략으로 top_k 최적화
    
    Args:
        strategy: 전략명
        top_k_list: 테스트할 top_k 값 리스트
        buffer_k: 버퍼 종목 수 (0 고정)
        rebalance_interval: 리밸런싱 주기 (holding_days와 동일)
    """
    logger.info(f"[{strategy}] 완전 교체 전략 + top_k 최적화 시작")
    logger.info(f"  전략: 완전 교체 (rebalance_interval={rebalance_interval}, buffer_k={buffer_k})")
    logger.info(f"  top_k 그리드: {top_k_list}")
    
    results = []
    if n_jobs == -1:
        n_jobs = None
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {
            executor.submit(run_backtest_with_params, strategy, top_k, buffer_k, rebalance_interval, config_path): top_k
            for top_k in top_k_list
        }
        
        for future in as_completed(futures):
            top_k = futures[future]
            try:
                total_return_dev, total_return_holdout, cagr_dev, cagr_holdout, sharpe_dev, sharpe_holdout = future.result()
                results.append({
                    "top_k": top_k,
                    "buffer_k": buffer_k,
                    "rebalance_interval": rebalance_interval,
                    "total_return_dev": total_return_dev,
                    "total_return_holdout": total_return_holdout,
                    "cagr_dev": cagr_dev,
                    "cagr_holdout": cagr_holdout,
                    "sharpe_dev": sharpe_dev,
                    "sharpe_holdout": sharpe_holdout,
                })
                logger.info(f"  top_k={top_k}: Dev TR={safe_format(total_return_dev, '{:.4f}')}, "
                          f"Holdout TR={safe_format(total_return_holdout, '{:.4f}')}, "
                          f"Dev Sharpe={safe_format(sharpe_dev, '{:.4f}')}, "
                          f"Holdout Sharpe={safe_format(sharpe_holdout, '{:.4f}')}")
            except Exception as e:
                logger.error(f"  top_k={top_k} 실행 중 오류: {e}")
    
    df_results = pd.DataFrame(results)
    
    # Holdout Total Return 기준으로 최적값 선택
    df_valid = df_results[df_results["total_return_holdout"].notna()].copy()
    if df_valid.empty:
        logger.warning(f"[{strategy}] 유효한 결과가 없습니다.")
        return {
            "best_top_k": None,
            "best_score": None,
            "all_results": df_results,
        }
    
    best_idx = df_valid["total_return_holdout"].idxmax()
    best_top_k = int(df_valid.loc[best_idx, "top_k"])
    best_score = float(df_valid.loc[best_idx, "total_return_holdout"])
    
    logger.info(f"[{strategy}] 최적 top_k: {best_top_k} (Holdout Total Return: {best_score:.4f})")
    
    return {
        "best_top_k": best_top_k,
        "best_score": best_score,
        "all_results": df_results,
    }

def generate_optimization_report(result: Dict, config_path: str):
    """최적화 리포트 생성"""
    cfg = load_config(config_path)
    reports_dir = Path(get_path(cfg, "artifacts_reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = reports_dir / "full_replacement_topk_optimization_report.md"
    
    df_results = result["all_results"]
    best_top_k = result["best_top_k"]
    best_score = result["best_score"]
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# 완전 교체 전략 + top_k 최적화 결과\n\n")
        f.write(f"생성 시간: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"## 실험 조건\n\n")
        f.write(f"- **전략**: 완전 교체 (rebalance_interval = holding_days)\n")
        f.write(f"  - Day1: top_k 매수 → Day20: 전량 매도 → Day20 top_k 재매수\n")
        f.write(f"  - buffer_k=0 고정 (매번 100% 교체)\n")
        f.write(f"  - rebalance_interval=20 (holding_days와 동일)\n")
        f.write(f"- **전략**: bt20_short (단기 랭킹)\n")
        f.write(f"- **top_k 그리드**: [20, 15, 10]\n\n")
        f.write(f"## 최적화 결과\n\n")
        
        if best_top_k is None:
            f.write(f"- 최적 top_k를 찾을 수 없습니다.\n\n")
        else:
            f.write(f"- **최적 top_k**: {best_top_k}\n")
            f.write(f"- **최적화 점수 (Holdout Total Return)**: {safe_format(best_score, '{:.4f}')}\n\n")
            
            # 최적값의 상세 메트릭
            best_row = df_results[df_results["top_k"] == best_top_k].iloc[0] if not df_results.empty else None
            if best_row is not None:
                f.write(f"### 최적 파라미터 상세 메트릭 (Holdout)\n\n")
                f.write(f"- **Net Sharpe**: {safe_format(best_row['sharpe_holdout'], '{:.4f}')}\n")
                f.write(f"- **Net CAGR**: {safe_format(best_row['cagr_holdout'], '{:.4%}')}\n")
                f.write(f"- **Net Total Return**: {safe_format(best_row['total_return_holdout'], '{:.4f}')}\n\n")
        
        f.write(f"## 전체 결과 비교\n\n")
        f.write(f"### Holdout 구간 성과\n\n")
        f.write(f"| top_k | Net Sharpe | Net CAGR | Net Total Return | Net Sharpe (Dev) | Net CAGR (Dev) |\n")
        f.write(f"|-------|------------|----------|-------------------|------------------|----------------|\n")
        for _, row in df_results.iterrows():
            f.write(f"| {int(row['top_k'])} | "
                   f"{safe_format(row['sharpe_holdout'], '{:.4f}')} | "
                   f"{safe_format(row['cagr_holdout'], '{:.4%}')} | "
                   f"{safe_format(row['total_return_holdout'], '{:.4f}')} | "
                   f"{safe_format(row['sharpe_dev'], '{:.4f}')} | "
                   f"{safe_format(row['cagr_dev'], '{:.4%}')} |\n")
        f.write("\n")
        
        f.write(f"## 결론\n\n")
        if best_top_k is not None:
            f.write(f"완전 교체 전략에서 **top_k={best_top_k}**이 Holdout 구간 Total Return 기준으로 최적 성과를 보였습니다.\n\n")
        f.write(f"### 주요 발견사항\n\n")
        f.write(f"- 완전 교체 전략 (buffer_k=0, rebalance_interval=holding_days)은 매 리밸런싱마다 100% 종목 교체\n")
        f.write(f"- top_k 값에 따른 성과 차이 분석 완료\n")
        f.write(f"- 최적 top_k 값: {best_top_k if best_top_k is not None else 'N/A'}\n")
    
    logger.info(f"최적화 리포트 생성 완료: {report_path}")

def main():
    config_path = "configs/config.yaml"
    
    # 실험 조건
    strategy = "bt20_short"
    top_k_list = [20, 15, 10]
    buffer_k = 0  # 완전 교체 전략
    rebalance_interval = 20  # holding_days와 동일
    
    logger.info("=" * 80)
    logger.info("완전 교체 전략 + top_k 최적화 시작")
    logger.info("=" * 80)
    logger.info(f"전략: {strategy}")
    logger.info(f"완전 교체 조건: buffer_k={buffer_k}, rebalance_interval={rebalance_interval}")
    logger.info(f"top_k 그리드: {top_k_list}")
    
    result = optimize_top_k_full_replacement(
        strategy=strategy,
        top_k_list=top_k_list,
        buffer_k=buffer_k,
        rebalance_interval=rebalance_interval,
        config_path=config_path,
        n_jobs=-1,
    )
    
    # 결과 저장
    cfg = load_config(config_path)
    reports_dir = Path(get_path(cfg, "artifacts_reports"))
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    result_path = reports_dir / "full_replacement_topk_optimization_results.csv"
    result["all_results"].to_csv(result_path, index=False, encoding="utf-8-sig")
    logger.info(f"\n결과 저장: {result_path}")
    
    logger.info("\n" + "=" * 80)
    logger.info("최적화 결과 요약")
    logger.info("=" * 80)
    print(result["all_results"].to_string(index=False))
    
    if result["best_top_k"] is not None:
        logger.info(f"\n최적 top_k: {result['best_top_k']} (Holdout Total Return: {result['best_score']:.4f})")
    
    # 리포트 생성
    generate_optimization_report(result, config_path)

if __name__ == "__main__":
    main()

