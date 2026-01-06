# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/verify_stage_root_cause.py
"""
Stage0~12 변화값의 "근원 원인" 확정을 위한 필수 검증 5개

각 run_tag별로 bt_returns / bt_positions / bt_metrics를 확인하여:
1. bt_returns의 (date_start, date_end, n_rebalances) 완전 일치 여부
2. avg_n_tickers가 top_k와 일치하는지
3. cost_bps_used vs config(l7_cost_bps) 불일치 여부
4. Stage6→Stage7에서 bt_positions가 동일한지(diff)
5. manifest/링크가 가리키는 산출물 경로가 진짜 그 run_tag 폴더인지
"""
import pandas as pd
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_config(config_path: Path) -> dict:
    """Config 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}

def verify_bt_returns_consistency(run_tags: List[str], base_interim_dir: Path) -> Dict:
    """
    검증 1: bt_returns의 (date_start, date_end, n_rebalances) 완전 일치 여부
    """
    print("\n" + "="*80)
    print("검증 1: bt_returns 일관성 확인")
    print("="*80)
    
    results = {}
    
    for run_tag in run_tags:
        bt_returns_path = base_interim_dir / run_tag / "bt_returns.parquet"
        
        if not bt_returns_path.exists():
            results[run_tag] = {
                "exists": False,
                "error": "bt_returns.parquet 없음"
            }
            continue
        
        try:
            df = pd.read_parquet(bt_returns_path)
            
            # date_start, date_end, n_rebalances 계산
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                date_start = df["date"].min()
                date_end = df["date"].max()
                n_dates = len(df["date"].unique())
            else:
                date_start = None
                date_end = None
                n_dates = 0
            
            # n_rebalances 계산 (phase별로)
            if "phase" in df.columns:
                n_rebalances = len(df.groupby(["phase", "date"]).groups)
            else:
                n_rebalances = len(df["date"].unique()) if "date" in df.columns else 0
            
            results[run_tag] = {
                "exists": True,
                "date_start": date_start,
                "date_end": date_end,
                "n_dates": n_dates,
                "n_rebalances": n_rebalances,
            }
            
            print(f"\n[{run_tag}]")
            print(f"  date_start: {date_start}")
            print(f"  date_end: {date_end}")
            print(f"  n_dates: {n_dates}")
            print(f"  n_rebalances: {n_rebalances}")
            
        except Exception as e:
            results[run_tag] = {
                "exists": False,
                "error": str(e)
            }
    
    # 일치 여부 확인
    print("\n[일치 여부 확인]")
    valid_results = {k: v for k, v in results.items() if v.get("exists", False)}
    
    if len(valid_results) < 2:
        print("  경고: 비교 가능한 run_tag가 2개 미만입니다.")
        return results
    
    # 첫 번째 run_tag를 기준으로 비교
    baseline_tag = list(valid_results.keys())[0]
    baseline = valid_results[baseline_tag]
    
    all_match = True
    for run_tag, data in valid_results.items():
        if run_tag == baseline_tag:
            continue
        
        date_start_match = data["date_start"] == baseline["date_start"]
        date_end_match = data["date_end"] == baseline["date_end"]
        n_rebalances_match = data["n_rebalances"] == baseline["n_rebalances"]
        
        if not (date_start_match and date_end_match and n_rebalances_match):
            all_match = False
            print(f"\n  [FAIL] {run_tag} vs {baseline_tag}:")
            if not date_start_match:
                print(f"    date_start 불일치: {data['date_start']} vs {baseline['date_start']}")
            if not date_end_match:
                print(f"    date_end 불일치: {data['date_end']} vs {baseline['date_end']}")
            if not n_rebalances_match:
                print(f"    n_rebalances 불일치: {data['n_rebalances']} vs {baseline['n_rebalances']}")
    
    if all_match:
        print(f"\n  [PASS] 모든 run_tag의 bt_returns 일관성 확인됨")
    
    return results

def verify_avg_n_tickers_vs_top_k(run_tags: List[str], base_interim_dir: Path, cfg: dict) -> Dict:
    """
    검증 2: avg_n_tickers가 top_k와 일치하는지
    """
    print("\n" + "="*80)
    print("검증 2: avg_n_tickers vs top_k 일치 여부")
    print("="*80)
    
    # config에서 top_k 가져오기
    l7_config = cfg.get("l7", {}) or {}
    config_top_k = l7_config.get("top_k", 20)
    
    print(f"\nConfig top_k: {config_top_k}")
    
    results = {}
    
    for run_tag in run_tags:
        bt_metrics_path = base_interim_dir / run_tag / "bt_metrics.parquet"
        
        if not bt_metrics_path.exists():
            results[run_tag] = {
                "exists": False,
                "error": "bt_metrics.parquet 없음"
            }
            continue
        
        try:
            df = pd.read_parquet(bt_metrics_path)
            
            # avg_n_tickers 찾기
            if "avg_n_tickers" in df.columns:
                avg_n_tickers = df["avg_n_tickers"].iloc[0] if len(df) > 0 else None
            elif "n_tickers_mean" in df.columns:
                avg_n_tickers = df["n_tickers_mean"].iloc[0] if len(df) > 0 else None
            else:
                avg_n_tickers = None
            
            # top_k 찾기
            if "top_k" in df.columns:
                top_k_used = df["top_k"].iloc[0] if len(df) > 0 else None
            else:
                top_k_used = config_top_k
            
            match = (avg_n_tickers is not None and top_k_used is not None and 
                    abs(avg_n_tickers - top_k_used) < 0.5)  # 반올림 허용
            
            results[run_tag] = {
                "exists": True,
                "avg_n_tickers": avg_n_tickers,
                "top_k_used": top_k_used,
                "config_top_k": config_top_k,
                "match": match,
            }
            
            status = "[PASS]" if match else "[FAIL]"
            print(f"\n[{run_tag}] {status}")
            print(f"  avg_n_tickers: {avg_n_tickers}")
            print(f"  top_k_used: {top_k_used}")
            print(f"  config_top_k: {config_top_k}")
            
            if not match:
                print(f"  ⚠️ 불일치! 종목 수가 예상과 다릅니다.")
                print(f"    → 결측/필터, 선택 로직 버그, 섹터 제약 가능성")
            
        except Exception as e:
            results[run_tag] = {
                "exists": False,
                "error": str(e)
            }
    
    return results

def verify_cost_bps_consistency(run_tags: List[str], base_interim_dir: Path, cfg: dict) -> Dict:
    """
    검증 3: cost_bps_used vs config(l7_cost_bps) 불일치 여부
    """
    print("\n" + "="*80)
    print("검증 3: cost_bps_used vs config(l7_cost_bps) 일치 여부")
    print("="*80)
    
    # config에서 cost_bps 가져오기
    l7_config = cfg.get("l7", {}) or {}
    config_cost_bps = l7_config.get("cost_bps", 0.0)
    
    print(f"\nConfig cost_bps: {config_cost_bps}")
    
    results = {}
    
    for run_tag in run_tags:
        bt_metrics_path = base_interim_dir / run_tag / "bt_metrics.parquet"
        
        if not bt_metrics_path.exists():
            results[run_tag] = {
                "exists": False,
                "error": "bt_metrics.parquet 없음"
            }
            continue
        
        try:
            df = pd.read_parquet(bt_metrics_path)
            
            # cost_bps_used 찾기
            if "cost_bps_used" in df.columns:
                cost_bps_used = df["cost_bps_used"].iloc[0] if len(df) > 0 else None
            elif "cost_bps" in df.columns:
                cost_bps_used = df["cost_bps"].iloc[0] if len(df) > 0 else None
            else:
                cost_bps_used = None
            
            match = (cost_bps_used is not None and 
                    abs(cost_bps_used - config_cost_bps) < 0.01)  # 0.01bps 허용
            
            results[run_tag] = {
                "exists": True,
                "cost_bps_used": cost_bps_used,
                "config_cost_bps": config_cost_bps,
                "match": match,
            }
            
            status = "[PASS]" if match else "[FAIL]"
            print(f"\n[{run_tag}] {status}")
            print(f"  cost_bps_used: {cost_bps_used}")
            print(f"  config_cost_bps: {config_cost_bps}")
            
            if not match:
                print(f"  ⚠️ 불일치! 비용 설정이 다릅니다.")
                print(f"    → net/gross 관계가 바뀔 수 있습니다.")
            
        except Exception as e:
            results[run_tag] = {
                "exists": False,
                "error": str(e)
            }
    
    return results

def verify_stage6_to_stage7_positions_diff(
    stage6_tag: str,
    stage7_tag: str,
    base_interim_dir: Path
) -> Dict:
    """
    검증 4: Stage6→Stage7에서 bt_positions가 동일한지(diff)
    """
    print("\n" + "="*80)
    print("검증 4: Stage6→Stage7 bt_positions 동일 여부")
    print("="*80)
    
    stage6_path = base_interim_dir / stage6_tag / "bt_positions.parquet"
    stage7_path = base_interim_dir / stage7_tag / "bt_positions.parquet"
    
    results = {
        "stage6_exists": stage6_path.exists(),
        "stage7_exists": stage7_path.exists(),
    }
    
    if not stage6_path.exists():
        print(f"[FAIL] Stage6 bt_positions 없음: {stage6_path}")
        results["error"] = "Stage6 bt_positions 없음"
        return results
    
    if not stage7_path.exists():
        print(f"[FAIL] Stage7 bt_positions 없음: {stage7_path}")
        results["error"] = "Stage7 bt_positions 없음"
        return results
    
    try:
        df6 = pd.read_parquet(stage6_path)
        df7 = pd.read_parquet(stage7_path)
        
        print(f"\nStage6 shape: {df6.shape}")
        print(f"Stage7 shape: {df7.shape}")
        
        # 컬럼 확인
        cols6 = set(df6.columns)
        cols7 = set(df7.columns)
        
        print(f"\nStage6 컬럼: {sorted(cols6)}")
        print(f"Stage7 컬럼: {sorted(cols7)}")
        
        # 공통 컬럼으로 비교
        common_cols = cols6 & cols7
        key_cols = ["date", "ticker", "position"] if "position" in common_cols else ["date", "ticker"]
        available_key_cols = [c for c in key_cols if c in common_cols]
        
        if not available_key_cols:
            results["error"] = "비교 가능한 키 컬럼 없음"
            return results
        
        # 정렬 후 비교
        df6_sorted = df6[available_key_cols + list(common_cols - set(available_key_cols))].sort_values(available_key_cols).reset_index(drop=True)
        df7_sorted = df7[available_key_cols + list(common_cols - set(available_key_cols))].sort_values(available_key_cols).reset_index(drop=True)
        
        # 행 수 비교
        if len(df6_sorted) != len(df7_sorted):
            results["identical"] = False
            results["diff_rows"] = abs(len(df6_sorted) - len(df7_sorted))
            print(f"\n[FAIL] 행 수 불일치: Stage6={len(df6_sorted)}, Stage7={len(df7_sorted)}")
            return results
        
        # 값 비교 (공통 컬럼만)
        numeric_cols = [c for c in common_cols if df6_sorted[c].dtype in ['float64', 'int64'] and c not in available_key_cols]
        
        if numeric_cols:
            # 숫자 컬럼 비교
            diff_summary = {}
            for col in numeric_cols[:5]:  # 최대 5개만
                if df6_sorted[col].equals(df7_sorted[col]):
                    diff_summary[col] = "identical"
                else:
                    diff = (df6_sorted[col] - df7_sorted[col]).abs().sum()
                    diff_summary[col] = f"diff_sum={diff:.6f}"
            
            results["diff_summary"] = diff_summary
            
            # 완전 일치 여부
            all_identical = all(v == "identical" for v in diff_summary.values())
            
            if all_identical:
                print(f"\n[PASS] bt_positions 완전 일치")
                results["identical"] = True
            else:
                print(f"\n[FAIL] bt_positions 차이 발견")
                for col, status in diff_summary.items():
                    print(f"  {col}: {status}")
                results["identical"] = False
        else:
            # 키 컬럼만 비교
            if df6_sorted[available_key_cols].equals(df7_sorted[available_key_cols]):
                print(f"\n[PASS] 키 컬럼 일치 (값 비교 불가)")
                results["identical"] = True
            else:
                print(f"\n[FAIL] 키 컬럼 불일치")
                results["identical"] = False
        
    except Exception as e:
        results["error"] = str(e)
        print(f"\n[ERROR] 비교 실패: {e}")
        import traceback
        traceback.print_exc()
    
    return results

def verify_artifact_paths_in_manifest(run_tags: List[str], base_interim_dir: Path) -> Dict:
    """
    검증 5: manifest/링크가 가리키는 산출물 경로가 진짜 그 run_tag 폴더인지
    """
    print("\n" + "="*80)
    print("검증 5: 산출물 경로 일치 여부")
    print("="*80)
    
    results = {}
    
    # history_manifest 확인
    manifest_path = PROJECT_ROOT / "reports" / "history" / "history_manifest.parquet"
    if not manifest_path.exists():
        manifest_path = PROJECT_ROOT / "reports" / "history" / "history_manifest.csv"
    
    if not manifest_path.exists():
        print("[WARNING] history_manifest 파일 없음")
        return results
    
    try:
        if manifest_path.suffix == ".parquet":
            manifest_df = pd.read_parquet(manifest_path)
        else:
            manifest_df = pd.read_csv(manifest_path)
        
        # 각 run_tag별로 확인
        for run_tag in run_tags:
            tag_results = {
                "manifest_exists": False,
                "expected_paths": {},
                "actual_paths": {},
                "matches": {},
            }
            
            # manifest에서 해당 run_tag 찾기
            tag_rows = manifest_df[manifest_df["run_tag"] == run_tag]
            
            if len(tag_rows) == 0:
                tag_results["error"] = "manifest에 run_tag 없음"
                results[run_tag] = tag_results
                continue
            
            tag_results["manifest_exists"] = True
            
            # 예상 경로
            expected_base = base_interim_dir / run_tag
            expected_paths = {
                "bt_returns": expected_base / "bt_returns.parquet",
                "bt_positions": expected_base / "bt_positions.parquet",
                "bt_metrics": expected_base / "bt_metrics.parquet",
            }
            
            tag_results["expected_paths"] = {k: str(v) for k, v in expected_paths.items()}
            
            # 실제 파일 존재 확인
            for name, path in expected_paths.items():
                exists = path.exists()
                tag_results["actual_paths"][name] = str(path) if exists else "NOT_FOUND"
                tag_results["matches"][name] = exists
            
            # 결과 출력
            print(f"\n[{run_tag}]")
            all_match = all(tag_results["matches"].values())
            status = "[PASS]" if all_match else "[FAIL]"
            print(f"  {status}")
            
            for name, match in tag_results["matches"].items():
                match_str = "[OK]" if match else "[MISSING]"
                print(f"    {name}: {match_str} {tag_results['actual_paths'][name]}")
            
            if not all_match:
                print(f"  [WARNING] 일부 산출물이 예상 경로에 없습니다.")
                print(f"    -> 다른 run의 산출물을 참조했을 가능성")
            
            results[run_tag] = tag_results
            
    except Exception as e:
        print(f"[ERROR] manifest 확인 실패: {e}")
        import traceback
        traceback.print_exc()
    
    return results

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
    
    # history_manifest에서 모든 run_tag 가져오기
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
    
    print("="*80)
    print("근원 원인 확정을 위한 필수 검증 5개")
    print("="*80)
    print(f"\n분석 대상 run_tag ({len(all_tags)}개):")
    for tag in all_tags:
        print(f"  - {tag}")
    
    # 검증 1: bt_returns 일관성
    verify_bt_returns_consistency(all_tags, base_interim_dir)
    
    # 검증 2: avg_n_tickers vs top_k
    verify_avg_n_tickers_vs_top_k(all_tags, base_interim_dir, cfg)
    
    # 검증 3: cost_bps 일치 여부
    verify_cost_bps_consistency(all_tags, base_interim_dir, cfg)
    
    # 검증 4: Stage6→Stage7 positions diff
    stage6_tags = [t for t in all_tags if "stage6" in t]
    stage7_tags = [t for t in all_tags if "stage7" in t or ("ranking" in str(manifest_df[manifest_df["run_tag"] == t]["track"].iloc[0]) if len(manifest_df[manifest_df["run_tag"] == t]) > 0 else False)]
    
    if stage6_tags and stage7_tags:
        stage6_tag = stage6_tags[-1]  # 최신
        stage7_tag = stage7_tags[-1]  # 최신
        verify_stage6_to_stage7_positions_diff(stage6_tag, stage7_tag, base_interim_dir)
    else:
        print("\n[SKIP] Stage6 또는 Stage7 run_tag 없음")
    
    # 검증 5: 산출물 경로 일치 여부
    verify_artifact_paths_in_manifest(all_tags, base_interim_dir)
    
    print("\n" + "="*80)
    print("검증 완료")
    print("="*80)

if __name__ == "__main__":
    main()
