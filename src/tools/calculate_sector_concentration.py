# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/calculate_sector_concentration.py
"""
[Stage8] Top20 섹터 농도 계산 스크립트
- HHI (Herfindahl-Hirschman Index)
- Max Sector Share
- Stage7 대비 비교
"""
import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Tuple
from datetime import datetime

import pandas as pd
import numpy as np

def calculate_hhi(weights: pd.Series) -> float:
    """
    HHI (Herfindahl-Hirschman Index) 계산
    
    Args:
        weights: 섹터별 가중치 Series
    
    Returns:
        HHI 값 (0~1, 1에 가까울수록 집중도 높음)
    """
    if len(weights) == 0:
        return 0.0
    
    # 가중치 정규화 (합 = 1)
    weights_norm = weights / weights.sum() if weights.sum() > 0 else weights
    
    # HHI = sum(weight^2)
    hhi = (weights_norm ** 2).sum()
    
    return float(hhi)

def calculate_max_sector_share(weights: pd.Series) -> float:
    """
    최대 섹터 비중 계산
    
    Args:
        weights: 섹터별 가중치 Series
    
    Returns:
        최대 섹터 비중 (0~1)
    """
    if len(weights) == 0:
        return 0.0
    
    # 가중치 정규화 (합 = 1)
    weights_norm = weights / weights.sum() if weights.sum() > 0 else weights
    
    return float(weights_norm.max())

def calculate_sector_concentration(
    ranking_daily: pd.DataFrame,
    top_k: int = 20,
    date_col: str = "date",
    ticker_col: str = "ticker",
    rank_col: str = "rank_total",
    sector_col: str = "sector_name",
) -> pd.DataFrame:
    """
    Top K 섹터 농도 계산
    
    Args:
        ranking_daily: ranking_daily DataFrame (date, ticker, rank_total, sector_name 포함)
        top_k: 상위 K개 종목 (기본: 20)
        date_col: 날짜 컬럼명
        ticker_col: 티커 컬럼명
        rank_col: 랭킹 컬럼명
        sector_col: 섹터 컬럼명
    
    Returns:
        DataFrame with columns: [date, top_k, n_sectors, hhi, max_sector_share, sector_distribution]
    """
    results = []
    
    for date, group in ranking_daily.groupby(date_col, sort=False):
        # Top K 선택 (rank_total이 낮을수록 상위)
        top_k_df = group.nsmallest(top_k, rank_col).copy()
        
        if len(top_k_df) == 0:
            continue
        
        # 섹터별 가중치 계산 (균등 가중치)
        sector_weights = top_k_df[sector_col].value_counts() if sector_col in top_k_df.columns else pd.Series()
        
        if len(sector_weights) == 0:
            # 섹터 정보가 없으면 스킵
            continue
        
        # HHI 계산
        hhi = calculate_hhi(sector_weights)
        
        # Max Sector Share 계산
        max_sector_share = calculate_max_sector_share(sector_weights)
        
        # 섹터 분포 (문자열)
        sector_dist = ", ".join([f"{sector}:{count}" for sector, count in sector_weights.items()])
        
        results.append({
            date_col: date,
            "top_k": top_k,
            "n_sectors": len(sector_weights),
            "hhi": hhi,
            "max_sector_share": max_sector_share,
            "sector_distribution": sector_dist,
        })
    
    if not results:
        return pd.DataFrame(columns=[date_col, "top_k", "n_sectors", "hhi", "max_sector_share", "sector_distribution"])
    
    df = pd.DataFrame(results)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    
    return df

def compare_with_baseline(
    current_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    date_col: str = "date",
) -> Dict[str, float]:
    """
    Baseline 대비 비교
    
    Args:
        current_df: 현재 실행 결과
        baseline_df: Baseline 실행 결과
        date_col: 날짜 컬럼명
    
    Returns:
        비교 결과 딕셔너리
    """
    results = {}
    
    # 평균 HHI 비교
    if "hhi" in current_df.columns and "hhi" in baseline_df.columns:
        current_hhi_mean = current_df["hhi"].mean()
        baseline_hhi_mean = baseline_df["hhi"].mean()
        results["hhi_mean_current"] = current_hhi_mean
        results["hhi_mean_baseline"] = baseline_hhi_mean
        results["hhi_mean_delta"] = current_hhi_mean - baseline_hhi_mean
        results["hhi_improved"] = current_hhi_mean <= baseline_hhi_mean  # 낮을수록 좋음
    
    # 평균 Max Sector Share 비교
    if "max_sector_share" in current_df.columns and "max_sector_share" in baseline_df.columns:
        current_max_mean = current_df["max_sector_share"].mean()
        baseline_max_mean = baseline_df["max_sector_share"].mean()
        results["max_sector_share_mean_current"] = current_max_mean
        results["max_sector_share_mean_baseline"] = baseline_max_mean
        results["max_sector_share_mean_delta"] = current_max_mean - baseline_max_mean
        results["max_sector_share_improved"] = current_max_mean <= baseline_max_mean  # 낮을수록 좋음
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description="[Stage8] Top20 섹터 농도 계산"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Config 파일 경로"
    )
    parser.add_argument(
        "--run-tag",
        type=str,
        required=True,
        help="현재 실행 태그"
    )
    parser.add_argument(
        "--baseline-tag",
        type=str,
        default="stage6_sector_relative_feature_balance_20251220_194928",
        help="Baseline 태그 (비교 대상)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="상위 K개 종목 (기본: 20)"
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="프로젝트 루트 디렉토리"
    )
    args = parser.parse_args()
    
    # 루트 경로 결정
    if args.root:
        root = Path(args.root)
    else:
        root = Path(__file__).resolve().parents[2]
    
    if not root.exists():
        print(f"[ERROR] 프로젝트 루트가 존재하지 않습니다: {root}")
        sys.exit(1)
    
    base_interim_dir = root / "data" / "interim"
    reports_ranking_dir = root / "reports" / "ranking"
    reports_ranking_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[Stage8] 섹터 농도 계산")
    print(f"Run Tag: {args.run_tag}")
    print(f"Baseline Tag: {args.baseline_tag}")
    print(f"Top K: {args.top_k}")
    
    # ranking_daily 로드
    ranking_daily_path = base_interim_dir / args.run_tag / "ranking_daily.parquet"
    
    if not ranking_daily_path.exists():
        print(f"[ERROR] ranking_daily 파일이 없습니다: {ranking_daily_path}")
        sys.exit(1)
    
    print(f"\n[1/3] ranking_daily 로드: {ranking_daily_path}")
    ranking_daily = pd.read_parquet(ranking_daily_path)
    
    # 필수 컬럼 확인
    required_cols = ["date", "ticker", "rank_total"]
    missing_cols = [c for c in required_cols if c not in ranking_daily.columns]
    if missing_cols:
        print(f"[ERROR] 필수 컬럼이 없습니다: {missing_cols}")
        sys.exit(1)
    
    # sector_name 확인
    sector_col = "sector_name"
    if sector_col not in ranking_daily.columns:
        print(f"[WARN] {sector_col} 컬럼이 없습니다. dataset_daily에서 가져오기 시도...")
        
        # dataset_daily 또는 panel_merged_daily에서 sector_name 가져오기
        # 여러 경로 시도
        candidate_paths = [
            base_interim_dir / args.run_tag / "panel_merged_daily.parquet",
            base_interim_dir / args.run_tag / "dataset_daily.parquet",
            base_interim_dir / "panel_merged_daily.parquet",
            base_interim_dir / "dataset_daily.parquet",
        ]
        
        # baseline 태그 폴더에서도 찾기
        baseline_interim_dir = base_interim_dir / args.baseline_tag
        if baseline_interim_dir.exists():
            candidate_paths.extend([
                baseline_interim_dir / "panel_merged_daily.parquet",
                baseline_interim_dir / "dataset_daily.parquet",
            ])
        
        # 재귀적으로 찾기
        all_panel_files = list(base_interim_dir.rglob("panel_merged_daily.parquet"))
        if all_panel_files:
            candidate_paths.append(max(all_panel_files, key=lambda p: p.stat().st_mtime))
        
        source_df = None
        source_path = None
        for path in candidate_paths:
            if path.exists():
                try:
                    df_test = pd.read_parquet(path)
                    if sector_col in df_test.columns:
                        source_df = df_test
                        source_path = path
                        break
                except:
                    continue
        
        if source_df is not None and source_path:
            sector_info = source_df[["date", "ticker", sector_col]].drop_duplicates(["date", "ticker"])
            ranking_daily = ranking_daily.merge(sector_info, on=["date", "ticker"], how="left")
            print(f"[OK] {sector_col} 병합 완료 (source: {source_path})")
        else:
            print(f"[ERROR] {sector_col}을 찾을 수 없습니다.")
            print(f"   시도한 경로: {candidate_paths[:5]}...")
            sys.exit(1)
    
    # 섹터 농도 계산
    print(f"\n[2/3] 섹터 농도 계산 중...")
    concentration_df = calculate_sector_concentration(
        ranking_daily,
        top_k=args.top_k,
        date_col="date",
        ticker_col="ticker",
        rank_col="rank_total",
        sector_col=sector_col,
    )
    
    if len(concentration_df) == 0:
        print("[WARN] 섹터 농도 계산 결과가 비어있습니다.")
        sys.exit(1)
    
    # 저장
    output_path = reports_ranking_dir / f"sector_concentration__{args.run_tag}.csv"
    concentration_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[OK] 섹터 농도 저장: {output_path}")
    
    # Baseline 비교
    print(f"\n[3/3] Baseline 대비 비교 중...")
    baseline_ranking_path = base_interim_dir / args.baseline_tag / "ranking_daily.parquet"
    
    comparison_results = {}
    if baseline_ranking_path.exists():
        baseline_ranking = pd.read_parquet(baseline_ranking_path)
        
        # Baseline에도 sector_name이 있는지 확인
        if sector_col in baseline_ranking.columns:
            baseline_concentration = calculate_sector_concentration(
                baseline_ranking,
                top_k=args.top_k,
                date_col="date",
                ticker_col="ticker",
                rank_col="rank_total",
                sector_col=sector_col,
            )
            
            comparison_results = compare_with_baseline(
                concentration_df,
                baseline_concentration,
                date_col="date",
            )
            
            print("\n=== Baseline 대비 비교 결과 ===")
            print(f"HHI 평균:")
            print(f"  Current: {comparison_results.get('hhi_mean_current', 'N/A'):.4f}")
            print(f"  Baseline: {comparison_results.get('hhi_mean_baseline', 'N/A'):.4f}")
            print(f"  Delta: {comparison_results.get('hhi_mean_delta', 'N/A'):.4f}")
            print(f"  개선 여부: {'[개선]' if comparison_results.get('hhi_improved', False) else '[악화]'}")
            
            print(f"\nMax Sector Share 평균:")
            print(f"  Current: {comparison_results.get('max_sector_share_mean_current', 'N/A'):.4f}")
            print(f"  Baseline: {comparison_results.get('max_sector_share_mean_baseline', 'N/A'):.4f}")
            print(f"  Delta: {comparison_results.get('max_sector_share_mean_delta', 'N/A'):.4f}")
            print(f"  개선 여부: {'[개선]' if comparison_results.get('max_sector_share_improved', False) else '[악화]'}")
        else:
            print("[WARN] Baseline ranking_daily에 sector_name이 없어 비교를 건너뜁니다.")
    else:
        print(f"[WARN] Baseline ranking_daily 파일이 없습니다: {baseline_ranking_path}")
    
    # 요약 통계 출력
    print("\n=== 현재 실행 요약 통계 ===")
    print(f"기간: {concentration_df['date'].min()} ~ {concentration_df['date'].max()}")
    print(f"평균 HHI: {concentration_df['hhi'].mean():.4f}")
    print(f"평균 Max Sector Share: {concentration_df['max_sector_share'].mean():.4f}")
    print(f"평균 섹터 수: {concentration_df['n_sectors'].mean():.1f}")
    
    print(f"\n[OK] 섹터 농도 계산 완료")
    print(f"출력 파일: {output_path}")

if __name__ == "__main__":
    main()
