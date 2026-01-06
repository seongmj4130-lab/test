# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/verify_additional_root_cause.py
"""
추가 확정 검증 3개

1. avg_n_tickers를 Phase(dev/holdout)로 분리해서 다시 계산
2. Stage0~6의 bt_positions에서 '리밸런싱당 보유 종목 수' 분포를 출력
3. Stage7 stage_evolution 값의 "출처 컬럼(metric_source)"를 강제 기록
"""
import pandas as pd
import sys
import yaml
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_config(config_path: Path) -> dict:
    """Config 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}

def verify_avg_n_tickers_by_phase(run_tags: List[str], base_interim_dir: Path) -> Dict:
    """
    검증 1: avg_n_tickers를 Phase(dev/holdout)로 분리해서 다시 계산
    """
    print("\n" + "="*80)
    print("추가 검증 1: avg_n_tickers를 Phase별로 분리 계산")
    print("="*80)
    
    results = {}
    
    for run_tag in run_tags:
        bt_metrics_path = base_interim_dir / run_tag / "bt_metrics.parquet"
        bt_positions_path = base_interim_dir / run_tag / "bt_positions.parquet"
        
        if not bt_metrics_path.exists() or not bt_positions_path.exists():
            results[run_tag] = {"error": "파일 없음"}
            continue
        
        try:
            metrics_df = pd.read_parquet(bt_metrics_path)
            positions_df = pd.read_parquet(bt_positions_path)
            
            # Phase별 avg_n_tickers 계산
            phase_results = {}
            
            if "phase" in positions_df.columns:
                for phase in ["dev", "holdout"]:
                    phase_positions = positions_df[positions_df["phase"] == phase]
                    
                    if len(phase_positions) > 0:
                        # 날짜별 종목 수 계산
                        date_counts = phase_positions.groupby("date")["ticker"].count()
                        avg_n_tickers = date_counts.mean()
                        
                        phase_results[phase] = {
                            "avg_n_tickers": avg_n_tickers,
                            "min": date_counts.min(),
                            "max": date_counts.max(),
                            "median": date_counts.median(),
                            "n_rebalances": len(date_counts),
                        }
                
                # bt_metrics에서도 확인
                metrics_by_phase = {}
                if "phase" in metrics_df.columns:
                    for phase in ["dev", "holdout"]:
                        phase_metrics = metrics_df[metrics_df["phase"] == phase]
                        if len(phase_metrics) > 0:
                            if "avg_n_tickers" in phase_metrics.columns:
                                metrics_by_phase[phase] = {
                                    "avg_n_tickers_from_metrics": phase_metrics["avg_n_tickers"].iloc[0],
                                }
                
                results[run_tag] = {
                    "from_positions": phase_results,
                    "from_metrics": metrics_by_phase,
                }
                
                print(f"\n[{run_tag}]")
                for phase in ["dev", "holdout"]:
                    if phase in phase_results:
                        pos_data = phase_results[phase]
                        print(f"  [{phase.upper()}]")
                        print(f"    avg_n_tickers (positions): {pos_data['avg_n_tickers']:.2f}")
                        print(f"    min: {pos_data['min']}, max: {pos_data['max']}, median: {pos_data['median']:.2f}")
                        print(f"    n_rebalances: {pos_data['n_rebalances']}")
                        
                        if phase in metrics_by_phase:
                            print(f"    avg_n_tickers (metrics): {metrics_by_phase[phase].get('avg_n_tickers_from_metrics', 'N/A')}")
            else:
                # phase 컬럼이 없으면 전체 평균만
                date_counts = positions_df.groupby("date")["ticker"].count()
                avg_n_tickers = date_counts.mean()
                results[run_tag] = {
                    "avg_n_tickers": avg_n_tickers,
                    "note": "phase 컬럼 없음"
                }
                print(f"\n[{run_tag}] phase 컬럼 없음")
                print(f"  avg_n_tickers: {avg_n_tickers:.2f}")
            
        except Exception as e:
            results[run_tag] = {"error": str(e)}
            print(f"\n[{run_tag}] ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    return results

def verify_positions_distribution(run_tags: List[str], base_interim_dir: Path) -> Dict:
    """
    검증 2: Stage0~6의 bt_positions에서 '리밸런싱당 보유 종목 수' 분포를 출력
    """
    print("\n" + "="*80)
    print("추가 검증 2: 리밸런싱당 보유 종목 수 분포")
    print("="*80)
    
    results = {}
    
    for run_tag in run_tags:
        bt_positions_path = base_interim_dir / run_tag / "bt_positions.parquet"
        
        if not bt_positions_path.exists():
            results[run_tag] = {"error": "bt_positions.parquet 없음"}
            continue
        
        try:
            df = pd.read_parquet(bt_positions_path)
            
            # 날짜별 종목 수 계산
            if "date" in df.columns:
                date_counts = df.groupby("date")["ticker"].count()
                
                # 분포 통계
                stats = {
                    "mean": date_counts.mean(),
                    "median": date_counts.median(),
                    "std": date_counts.std(),
                    "min": date_counts.min(),
                    "max": date_counts.max(),
                    "p5": date_counts.quantile(0.05),
                    "p25": date_counts.quantile(0.25),
                    "p75": date_counts.quantile(0.75),
                    "p95": date_counts.quantile(0.95),
                    "n_rebalances": len(date_counts),
                }
                
                # 0종목/5종목 등 특이 케이스 확인
                zero_count = (date_counts == 0).sum()
                low_count = (date_counts < 5).sum()
                high_count = (date_counts >= 15).sum()
                
                stats["zero_count"] = zero_count
                stats["low_count"] = low_count
                stats["high_count"] = high_count
                
                results[run_tag] = stats
                
                print(f"\n[{run_tag}]")
                print(f"  평균: {stats['mean']:.2f}")
                print(f"  중앙값: {stats['median']:.2f}")
                print(f"  표준편차: {stats['std']:.2f}")
                print(f"  범위: {stats['min']} ~ {stats['max']}")
                print(f"  5%: {stats['p5']:.2f}, 25%: {stats['p25']:.2f}, 75%: {stats['p75']:.2f}, 95%: {stats['p95']:.2f}")
                print(f"  리밸런싱 수: {stats['n_rebalances']}")
                print(f"  0종목: {zero_count}일, <5종목: {low_count}일, >=15종목: {high_count}일")
                
                # 분포 히스토그램
                print(f"\n  종목 수 분포:")
                dist = date_counts.value_counts().sort_index()
                for count, freq in dist.items():
                    pct = freq / len(date_counts) * 100
                    print(f"    {count:2d}개: {freq:3d}일 ({pct:5.1f}%)")
            
        except Exception as e:
            results[run_tag] = {"error": str(e)}
            print(f"\n[{run_tag}] ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    return results

def verify_stage7_metric_source(
    stage7_tag: str,
    base_interim_dir: Path,
    manifest_df: pd.DataFrame
) -> Dict:
    """
    검증 3: Stage7 stage_evolution 값의 "출처 컬럼(metric_source)"를 강제 기록
    """
    print("\n" + "="*80)
    print("추가 검증 3: Stage7 metric_source 확인")
    print("="*80)
    
    results = {
        "stage7_tag": stage7_tag,
        "has_bt_metrics": False,
        "has_bt_returns": False,
        "has_bt_positions": False,
        "metric_source": None,
        "carried_from": None,
    }
    
    # Stage7에 bt_metrics가 있는지 확인
    stage7_bt_metrics = base_interim_dir / stage7_tag / "bt_metrics.parquet"
    stage7_bt_returns = base_interim_dir / stage7_tag / "bt_returns.parquet"
    stage7_bt_positions = base_interim_dir / stage7_tag / "bt_positions.parquet"
    
    results["has_bt_metrics"] = stage7_bt_metrics.exists()
    results["has_bt_returns"] = stage7_bt_returns.exists()
    results["has_bt_positions"] = stage7_bt_positions.exists()
    
    print(f"\n[Stage7: {stage7_tag}]")
    print(f"  bt_metrics 존재: {results['has_bt_metrics']}")
    print(f"  bt_returns 존재: {results['has_bt_returns']}")
    print(f"  bt_positions 존재: {results['has_bt_positions']}")
    
    if results["has_bt_metrics"]:
        results["metric_source"] = "bt_metrics.parquet"
        print(f"  [PASS] Stage7에 백테스트 산출물 존재")
    else:
        # Stage7에 백테스트 산출물이 없으면, 어디서 가져왔는지 확인
        results["metric_source"] = "NA(no backtest)"
        
        # Stage6 run_tag 찾기
        stage6_rows = manifest_df[
            (manifest_df["stage_no"] == 6) & 
            (manifest_df["track"] == "pipeline")
        ]
        
        if len(stage6_rows) > 0:
            stage6_tag = stage6_rows.iloc[-1]["run_tag"]  # 최신
            stage6_bt_metrics = base_interim_dir / stage6_tag / "bt_metrics.parquet"
            
            if stage6_bt_metrics.exists():
                # Stage6의 지표와 Stage7의 지표 비교
                try:
                    stage6_metrics = pd.read_parquet(stage6_bt_metrics)
                    stage7_row = manifest_df[manifest_df["run_tag"] == stage7_tag].iloc[0]
                    
                    # 주요 지표 비교
                    if "holdout_sharpe" in stage7_row and pd.notna(stage7_row["holdout_sharpe"]):
                        stage6_holdout = stage6_metrics[stage6_metrics["phase"] == "holdout"]
                        if len(stage6_holdout) > 0:
                            stage6_sharpe = stage6_holdout["net_sharpe"].iloc[0]
                            stage7_sharpe = stage7_row["holdout_sharpe"]
                            
                            if abs(stage6_sharpe - stage7_sharpe) < 0.0001:
                                results["carried_from"] = stage6_tag
                                results["metric_source"] = f"carried_from:{stage6_tag}"
                                print(f"  [확인] Stage7 지표가 Stage6와 일치")
                                print(f"    Stage6 Sharpe: {stage6_sharpe:.4f}")
                                print(f"    Stage7 Sharpe: {stage7_sharpe:.4f}")
                                print(f"    metric_source: {results['metric_source']}")
                            else:
                                print(f"  [WARNING] Stage7 지표가 Stage6와 다름")
                                print(f"    Stage6 Sharpe: {stage6_sharpe:.4f}")
                                print(f"    Stage7 Sharpe: {stage7_sharpe:.4f}")
                except Exception as e:
                    print(f"  [ERROR] 비교 실패: {e}")
        
        print(f"  [결론] Stage7에 백테스트 산출물 없음")
        print(f"    metric_source: {results['metric_source']}")
        if results["carried_from"]:
            print(f"    carried_from: {results['carried_from']}")
    
    return results

def generate_stage_positions_summary(run_tags: List[str], base_interim_dir: Path) -> pd.DataFrame:
    """
    Stage0~6 각각의 bt_positions에서 date별 보유 종목 수(n_names) 요약표 생성
    """
    print("\n" + "="*80)
    print("추가 검증 4: Stage별 date별 보유 종목 수 요약표")
    print("="*80)
    
    summary_rows = []
    
    for run_tag in run_tags:
        bt_positions_path = base_interim_dir / run_tag / "bt_positions.parquet"
        
        if not bt_positions_path.exists():
            continue
        
        try:
            df = pd.read_parquet(bt_positions_path)
            
            if "date" in df.columns:
                date_counts = df.groupby("date")["ticker"].count()
                
                # Phase별로도 분리
                if "phase" in df.columns:
                    for phase in ["dev", "holdout"]:
                        phase_df = df[df["phase"] == phase]
                        if len(phase_df) > 0:
                            phase_date_counts = phase_df.groupby("date")["ticker"].count()
                            
                            summary_rows.append({
                                "run_tag": run_tag,
                                "phase": phase,
                                "n_rebalances": len(phase_date_counts),
                                "mean": phase_date_counts.mean(),
                                "median": phase_date_counts.median(),
                                "min": phase_date_counts.min(),
                                "max": phase_date_counts.max(),
                                "p5": phase_date_counts.quantile(0.05),
                                "p95": phase_date_counts.quantile(0.95),
                                "zero_count": (phase_date_counts == 0).sum(),
                                "low_count": (phase_date_counts < 5).sum(),
                            })
                else:
                    # phase 없으면 전체
                    summary_rows.append({
                        "run_tag": run_tag,
                        "phase": "all",
                        "n_rebalances": len(date_counts),
                        "mean": date_counts.mean(),
                        "median": date_counts.median(),
                        "min": date_counts.min(),
                        "max": date_counts.max(),
                        "p5": date_counts.quantile(0.05),
                        "p95": date_counts.quantile(0.95),
                        "zero_count": (date_counts == 0).sum(),
                        "low_count": (date_counts < 5).sum(),
                    })
        
        except Exception as e:
            print(f"[{run_tag}] ERROR: {e}")
    
    summary_df = pd.DataFrame(summary_rows)
    
    if len(summary_df) > 0:
        print("\n[요약표]")
        print(summary_df.to_string(index=False))
    
    return summary_df

def main():
    config_path = PROJECT_ROOT / "configs" / "config.yaml"
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    
    cfg = load_config(config_path)
    
    # base_interim_dir 가져오기
    paths = cfg.get("paths", {})
    base_dir = Path(paths.get("base_dir", PROJECT_ROOT))
    base_interim_dir = base_dir / "data" / "interim"
    
    # history_manifest 로드
    manifest_path = PROJECT_ROOT / "reports" / "history" / "history_manifest.parquet"
    if not manifest_path.exists():
        manifest_path = PROJECT_ROOT / "reports" / "history" / "history_manifest.csv"
    
    if not manifest_path.exists():
        print("ERROR: history_manifest 파일 없음", file=sys.stderr)
        sys.exit(1)
    
    if manifest_path.suffix == ".parquet":
        manifest_df = pd.read_parquet(manifest_path)
    else:
        manifest_df = pd.read_csv(manifest_path)
    
    # Pipeline 트랙의 run_tag만 필터링 (백테스트가 있는 Stage)
    pipeline_tags = manifest_df[
        (manifest_df["track"] == "pipeline") & 
        (manifest_df["stage_no"] >= 0) & 
        (manifest_df["stage_no"] <= 6)
    ]["run_tag"].tolist()
    
    # Ranking 트랙의 Stage7도 포함
    ranking_stage7 = manifest_df[
        (manifest_df["track"] == "ranking") & 
        (manifest_df["stage_no"] == 7)
    ]["run_tag"].tolist()
    
    all_tags = sorted(set(pipeline_tags + ranking_stage7))
    stage7_tag = ranking_stage7[-1] if ranking_stage7 else None
    
    print("="*80)
    print("추가 확정 검증 3개")
    print("="*80)
    print(f"\n분석 대상 run_tag ({len(all_tags)}개):")
    for tag in all_tags:
        print(f"  - {tag}")
    
    # 검증 1: Phase별 avg_n_tickers
    phase_results = verify_avg_n_tickers_by_phase(pipeline_tags, base_interim_dir)
    
    # 검증 2: 리밸런싱당 보유 종목 수 분포
    distribution_results = verify_positions_distribution(pipeline_tags, base_interim_dir)
    
    # 검증 3: Stage7 metric_source
    if stage7_tag:
        stage7_results = verify_stage7_metric_source(stage7_tag, base_interim_dir, manifest_df)
    else:
        print("\n[SKIP] Stage7 run_tag 없음")
        stage7_results = {}
    
    # 검증 4: Stage별 요약표 생성
    summary_df = generate_stage_positions_summary(pipeline_tags, base_interim_dir)
    
    # 결과 저장
    output_dir = PROJECT_ROOT / "reports" / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Phase별 결과 저장
    phase_summary = []
    for run_tag, data in phase_results.items():
        if "from_positions" in data:
            for phase, phase_data in data["from_positions"].items():
                phase_summary.append({
                    "run_tag": run_tag,
                    "phase": phase,
                    **phase_data
                })
    
    if phase_summary:
        phase_df = pd.DataFrame(phase_summary)
        phase_output = output_dir / "avg_n_tickers_by_phase.csv"
        phase_df.to_csv(phase_output, index=False, encoding='utf-8-sig')
        print(f"\n[저장] Phase별 avg_n_tickers: {phase_output}")
    
    # 분포 결과 저장
    if distribution_results:
        dist_df = pd.DataFrame([
            {"run_tag": k, **v} for k, v in distribution_results.items() 
            if "error" not in v
        ])
        dist_output = output_dir / "positions_distribution.csv"
        dist_df.to_csv(dist_output, index=False, encoding='utf-8-sig')
        print(f"[저장] 종목 수 분포: {dist_output}")
    
    # 요약표 저장
    if len(summary_df) > 0:
        summary_output = output_dir / "stage_positions_summary.csv"
        summary_df.to_csv(summary_output, index=False, encoding='utf-8-sig')
        print(f"[저장] Stage별 요약표: {summary_output}")
    
    # Stage7 metric_source 저장
    if stage7_results:
        stage7_df = pd.DataFrame([stage7_results])
        stage7_output = output_dir / "stage7_metric_source.csv"
        stage7_df.to_csv(stage7_output, index=False, encoding='utf-8-sig')
        print(f"[저장] Stage7 metric_source: {stage7_output}")
    
    print("\n" + "="*80)
    print("추가 검증 완료")
    print("="*80)

if __name__ == "__main__":
    main()







