# -*- coding: utf-8 -*-
"""
최적 가중치 적용 및 검증 스크립트

1. 최적 가중치를 적용한 L8 랭킹 실행
2. Dev/Holdout 구간별 성과 비교
3. 결과 분석 및 보고서 생성
"""
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

# 프로젝트 루트 경로 추가
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.tracks.track_a.stages.ranking.l8_rank_engine import run_L8_rank_engine
from src.tracks.track_a.stages.ranking.ranking_metrics import (
    calculate_ranking_metrics_with_lagged_returns,
)


def load_optimal_weights(horizon: str):
    """최적 가중치 파일 로드"""
    configs_dir = project_root / "configs"

    if horizon == "short":
        weights_file = configs_dir / "feature_groups_short_optimized_grid_20260108_135117.yaml"
    else:
        weights_file = configs_dir / "feature_groups_long_optimized_grid_20260108_145118.yaml"

    if not weights_file.exists():
        raise FileNotFoundError(f"최적 가중치 파일을 찾을 수 없습니다: {weights_file}")

    with open(weights_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config, weights_file

def evaluate_on_folds(ranking_daily: pd.DataFrame, forward_returns: pd.DataFrame,
                      cv_folds: pd.DataFrame, horizon: str):
    """Dev/Holdout 구간별 성과 평가"""
    ret_col = "ret_fwd_20d" if horizon == "short" else "ret_fwd_120d"

    results = {}

    # Dev 구간
    dev_folds = cv_folds[cv_folds["fold_id"] != "holdout"]
    if len(dev_folds) > 0:
        dev_dates = pd.to_datetime(dev_folds["test_end"].unique(), errors="coerce")
        dev_dates = dev_dates[dev_dates.notna()]

        if len(dev_dates) > 0:
            dev_ranking = ranking_daily[ranking_daily["date"].isin(dev_dates)]
            dev_returns = forward_returns[
                (forward_returns["date"].isin(dev_dates)) &
                (forward_returns[ret_col].notna())
            ]

            if len(dev_ranking) > 0 and len(dev_returns) > 0:
                dev_metrics = calculate_ranking_metrics_with_lagged_returns(
                    dev_ranking,
                    dev_returns,
                    ret_col=ret_col,
                    lag_days=0,
                )
                results["dev"] = dev_metrics

    # Holdout 구간
    holdout_folds = cv_folds[cv_folds["fold_id"] == "holdout"]
    if len(holdout_folds) > 0:
        holdout_dates = pd.to_datetime(holdout_folds["test_end"].unique(), errors="coerce")
        holdout_dates = holdout_dates[holdout_dates.notna()]

        if len(holdout_dates) > 0:
            holdout_ranking = ranking_daily[ranking_daily["date"].isin(holdout_dates)]
            holdout_returns = forward_returns[
                (forward_returns["date"].isin(holdout_dates)) &
                (forward_returns[ret_col].notna())
            ]

            if len(holdout_ranking) > 0 and len(holdout_returns) > 0:
                holdout_metrics = calculate_ranking_metrics_with_lagged_returns(
                    holdout_ranking,
                    holdout_returns,
                    ret_col=ret_col,
                    lag_days=0,
                )
                results["holdout"] = holdout_metrics

    return results

def main():
    """메인 함수"""
    print("=" * 80)
    print("최적 가중치 적용 및 검증")
    print("=" * 80)

    data_dir = project_root / "data" / "interim"
    configs_dir = project_root / "configs"
    output_dir = project_root / "artifacts" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_summary = {}

    for horizon in ["short", "long"]:
        print(f"\n[{horizon.upper()} 랭킹]")
        print("-" * 80)

        # 최적 가중치 로드
        optimal_config, weights_file = load_optimal_weights(horizon)
        print(f"최적 가중치 파일: {weights_file.name}")

        # 데이터 로드
        panel_file = data_dir / "panel_merged_daily.parquet"
        if not panel_file.exists():
            print(f"⚠️  데이터 파일을 찾을 수 없습니다: {panel_file}")
            continue

        panel_df = pd.read_parquet(panel_file)

        # Forward Returns 확인
        ret_col = "ret_fwd_20d" if horizon == "short" else "ret_fwd_120d"
        if ret_col not in panel_df.columns:
            dataset_file = data_dir / "dataset_daily.parquet"
            if dataset_file.exists():
                dataset_df = pd.read_parquet(dataset_file)
                panel_df = panel_df.merge(
                    dataset_df[["date", "ticker", ret_col]],
                    on=["date", "ticker"],
                    how="left"
                )

        # CV folds 로드
        cv_folds_file = data_dir / f"cv_folds_{horizon}.parquet"
        if not cv_folds_file.exists():
            print(f"⚠️  CV folds 파일을 찾을 수 없습니다: {cv_folds_file}")
            continue

        cv_folds = pd.read_parquet(cv_folds_file)

        # L8 랭킹 실행 (최적 가중치 적용)
        print("L8 랭킹 실행 중...")
        try:
            ranking_result, warns = run_L8_rank_engine(
                panel_df,
                horizon=horizon,
                feature_groups_config=str(weights_file),
            )

            ranking_daily = ranking_result.get("ranking_daily", pd.DataFrame())

            if len(ranking_daily) == 0:
                print("⚠️  랭킹 결과가 비어있습니다.")
                continue

            print(f"✅ 랭킹 생성 완료: {len(ranking_daily)}행")

            # Forward Returns 준비
            forward_returns = panel_df[
                ["date", "ticker", ret_col]
            ].dropna(subset=[ret_col])

            # Dev/Holdout 구간별 평가
            print("Dev/Holdout 구간별 성과 평가 중...")
            fold_results = evaluate_on_folds(
                ranking_daily,
                forward_returns,
                cv_folds,
                horizon
            )

            results_summary[horizon] = {
                "optimal_weights_file": str(weights_file),
                "optimal_config": optimal_config,
                "ranking_count": len(ranking_daily),
                "fold_results": fold_results,
            }

            # 결과 출력
            if "dev" in fold_results:
                dev = fold_results["dev"]
                print(f"\n[Dev 구간]")
                print(f"  Hit Ratio: {dev.get('hit_ratio', 0)*100:.2f}%")
                print(f"  IC Mean: {dev.get('ic_mean', 0):.4f}")
                print(f"  ICIR: {dev.get('icir', 0):.4f}")

            if "holdout" in fold_results:
                holdout = fold_results["holdout"]
                print(f"\n[Holdout 구간]")
                print(f"  Hit Ratio: {holdout.get('hit_ratio', 0)*100:.2f}%")
                print(f"  IC Mean: {holdout.get('ic_mean', 0):.4f}")
                print(f"  ICIR: {holdout.get('icir', 0):.4f}")

        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()

    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = output_dir / f"optimal_weights_validation_{timestamp}.md"

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("# 최적 가중치 적용 및 검증 결과\n\n")
        f.write(f"**생성일**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("=" * 80 + "\n\n")

        for horizon, result in results_summary.items():
            f.write(f"## {horizon.upper()} 랭킹\n\n")
            f.write(f"**최적 가중치 파일**: {result['optimal_weights_file']}\n\n")
            f.write(f"**랭킹 생성**: {result['ranking_count']}행\n\n")

            fold_results = result['fold_results']

            if "dev" in fold_results:
                dev = fold_results["dev"]
                f.write("### Dev 구간 성과\n\n")
                f.write(f"- Hit Ratio: {dev.get('hit_ratio', 0)*100:.2f}%\n")
                f.write(f"- IC Mean: {dev.get('ic_mean', 0):.4f}\n")
                f.write(f"- ICIR: {dev.get('icir', 0):.4f}\n\n")

            if "holdout" in fold_results:
                holdout = fold_results["holdout"]
                f.write("### Holdout 구간 성과\n\n")
                f.write(f"- Hit Ratio: {holdout.get('hit_ratio', 0)*100:.2f}%\n")
                f.write(f"- IC Mean: {holdout.get('ic_mean', 0):.4f}\n")
                f.write(f"- ICIR: {holdout.get('icir', 0):.4f}\n\n")

            if "dev" in fold_results and "holdout" in fold_results:
                dev = fold_results["dev"]
                holdout = fold_results["holdout"]
                f.write("### Dev vs Holdout 비교\n\n")
                f.write(f"- Hit Ratio 차이: {holdout.get('hit_ratio', 0) - dev.get('hit_ratio', 0):.4f}\n")
                f.write(f"- IC Mean 차이: {holdout.get('ic_mean', 0) - dev.get('ic_mean', 0):.4f}\n")
                f.write(f"- ICIR 차이: {holdout.get('icir', 0) - dev.get('icir', 0):.4f}\n\n")

    print(f"\n✅ 결과 저장: {summary_file}")

    return results_summary

if __name__ == "__main__":
    main()
