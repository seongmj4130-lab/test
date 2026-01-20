"""
Holdout 구간 성과 평가 및 Dev/Holdout 구간 성과 비교

1. 최적 가중치 적용한 L8 랭킹 실행 (이미 config.yaml에 적용됨)
2. Holdout 구간에서 성과 평가
3. Dev 구간 성과와 비교 (Grid Search 결과 활용)
4. 과적합 여부 최종 확인
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# 프로젝트 루트 경로 추가
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.components.ranking.score_engine import build_ranking_daily
from src.tracks.track_b.stages.backtest.l7_backtest import _rank_ic, _safe_corr


def calculate_ic(scores: pd.Series, returns: pd.Series) -> float:
    """IC (Pearson correlation)"""
    return _safe_corr(scores, returns)


def calculate_rank_ic(scores: pd.Series, returns: pd.Series) -> float:
    """Rank IC (Spearman correlation)"""
    return _rank_ic(scores, returns)


def calculate_hit_ratio(returns: np.ndarray) -> float:
    """Hit Ratio: percentage of positive returns"""
    if len(returns) == 0:
        return 0.0
    return float((returns > 0).mean())


def calculate_icir(ic_values: np.ndarray) -> float:
    """ICIR = mean(IC) / std(IC)"""
    if len(ic_values) == 0:
        return 0.0
    ic_mean = np.nanmean(ic_values)
    ic_std = np.nanstd(ic_values)
    if ic_std < 1e-10:
        return 0.0
    return float(ic_mean / ic_std)


def evaluate_on_fold(
    ranking_daily: pd.DataFrame,
    forward_returns: pd.DataFrame,
    cv_folds: pd.DataFrame,
    fold_type: str,
    horizon: str,
) -> dict:
    """특정 fold 구간에서 성과 평가"""
    ret_col = "ret_fwd_20d" if horizon == "short" else "ret_fwd_120d"

    # Fold 구간 필터링
    if fold_type == "dev":
        folds = cv_folds[~cv_folds["fold_id"].str.startswith("holdout", na=False)]
    else:  # holdout
        folds = cv_folds[cv_folds["fold_id"].str.startswith("holdout", na=False)]

    print(f"    {fold_type} 구간: fold_id 필터링 후 {len(folds)}개 fold")
    if len(folds) > 0:
        print(f"      fold_id 목록: {folds['fold_id'].unique()}")

    if len(folds) == 0:
        print(f"    ⚠️  {fold_type} 구간: fold 없음")
        return None

    # 날짜 추출
    dates = pd.to_datetime(folds["test_end"].unique(), errors="coerce")
    dates = (
        pd.Series(dates).dropna().unique()
    )  # DatetimeArray를 Series로 변환 후 dropna

    if len(dates) == 0:
        print(f"    ⚠️  {fold_type} 구간: test_end 날짜 없음")
        return None

    print(f"    {fold_type} 구간: test_end 날짜 {len(dates)}개")
    print(f"      날짜 범위: {dates.min()} ~ {dates.max()}")

    # ranking_daily와 forward_returns의 날짜 확인
    ranking_dates = pd.to_datetime(ranking_daily["date"].unique(), errors="coerce")
    ranking_dates = pd.Series(ranking_dates).dropna().unique()
    returns_dates = pd.to_datetime(forward_returns["date"].unique(), errors="coerce")
    returns_dates = pd.Series(returns_dates).dropna().unique()

    print(
        f"      랭킹 날짜: {len(ranking_dates)}개 ({ranking_dates.min()} ~ {ranking_dates.max()})"
    )
    print(
        f"      Returns 날짜: {len(returns_dates)}개 ({returns_dates.min()} ~ {returns_dates.max()})"
    )

    # 공통 날짜 확인
    common_dates = pd.Series(list(set(dates) & set(ranking_dates) & set(returns_dates)))
    print(f"      공통 날짜: {len(common_dates)}개")

    if len(common_dates) == 0:
        print(f"    ⚠️  {fold_type} 구간: 공통 날짜 없음")
        return None

    # 해당 구간의 랭킹과 수익률 필터링 (공통 날짜 사용)
    ranking_filtered = ranking_daily[ranking_daily["date"].isin(common_dates)].copy()
    returns_filtered = forward_returns[
        (forward_returns["date"].isin(common_dates))
        & (forward_returns[ret_col].notna())
    ].copy()

    if len(ranking_filtered) == 0:
        print(
            f"    ⚠️  {fold_type} 구간: 랭킹 데이터 없음 (필터링 후 {len(ranking_filtered)}행)"
        )
        return None

    if len(returns_filtered) == 0:
        print(
            f"    ⚠️  {fold_type} 구간: Forward Returns 데이터 없음 (필터링 후 {len(returns_filtered)}행)"
        )
        return None

    print(
        f"    {fold_type} 구간: 랭킹 {len(ranking_filtered)}행, Forward Returns {len(returns_filtered)}행"
    )

    # 날짜/티커 타입 정규화
    ranking_filtered["date"] = pd.to_datetime(ranking_filtered["date"], errors="coerce")
    ranking_filtered["ticker"] = ranking_filtered["ticker"].astype(str).str.zfill(6)
    returns_filtered["date"] = pd.to_datetime(returns_filtered["date"], errors="coerce")
    returns_filtered["ticker"] = returns_filtered["ticker"].astype(str).str.zfill(6)

    # Merge
    merged = ranking_filtered.merge(
        returns_filtered[["date", "ticker", ret_col]],
        on=["date", "ticker"],
        how="inner",
    )

    if len(merged) == 0:
        print(f"    ⚠️  {fold_type} 구간: Merge 실패 (공통 날짜/티커 없음)")
        print(
            f"      랭킹 날짜 범위: {ranking_filtered['date'].min()} ~ {ranking_filtered['date'].max()}"
        )
        print(
            f"      Returns 날짜 범위: {returns_filtered['date'].min()} ~ {returns_filtered['date'].max()}"
        )
        return None

    print(f"    {fold_type} 구간: Merge 성공 ({len(merged)}행)")

    # 날짜별로 그룹화하여 평가
    results_by_date = []

    valid_dates = 0
    skipped_dates = 0

    for date in merged["date"].unique():
        date_data = merged[merged["date"] == date]

        if len(date_data) < 2:
            skipped_dates += 1
            continue

        scores = pd.Series(date_data["score_total"].values)
        returns = pd.Series(date_data[ret_col].values)

        # 지표 계산
        hit_ratio = calculate_hit_ratio(returns.values)
        ic = calculate_ic(scores, returns)
        rank_ic = calculate_rank_ic(scores, returns)

        # NaN 체크
        if pd.isna(ic) and pd.isna(rank_ic):
            skipped_dates += 1
            continue

        results_by_date.append(
            {
                "date": date,
                "hit_ratio": hit_ratio,
                "ic": ic,
                "rank_ic": rank_ic,
                "n_observations": len(date_data),
            }
        )
        valid_dates += 1

    print(
        f"    {fold_type} 구간: 유효 날짜 {valid_dates}개, 스킵 {skipped_dates}개 (merged {len(merged)}행, 고유 날짜 {len(merged['date'].unique())}개)"
    )

    if len(results_by_date) == 0:
        print(f"    ⚠️  {fold_type} 구간: 날짜별 결과 없음")
        return None

    print(f"    {fold_type} 구간: 날짜별 결과 {len(results_by_date)}개")

    results_df = pd.DataFrame(results_by_date)

    # 전체 지표 계산
    hit_ratio_mean = results_df["hit_ratio"].mean()
    ic_mean = results_df["ic"].mean()
    rank_ic_mean = results_df["rank_ic"].mean()
    icir = calculate_icir(results_df["ic"].values)
    rank_icir = calculate_icir(results_df["rank_ic"].values)

    return {
        "fold_type": fold_type,
        "n_dates": len(results_by_date),
        "n_observations": len(merged),
        "hit_ratio": hit_ratio_mean,
        "ic_mean": ic_mean,
        "rank_ic_mean": rank_ic_mean,
        "icir": icir,
        "rank_icir": rank_icir,
        "ic_std": results_df["ic"].std(),
        "rank_ic_std": results_df["rank_ic"].std(),
        "results_by_date": results_df,
    }


def load_grid_search_results(horizon: str) -> dict:
    """Grid Search 결과 로드 (Dev 구간 성과)"""
    results_dir = project_root / "artifacts" / "reports"

    if horizon == "short":
        file = results_dir / "track_a_group_weights_grid_search_20260108_135117.csv"
    else:
        file = results_dir / "track_a_group_weights_grid_search_20260108_145118.csv"

    if not file.exists():
        return None

    df = pd.read_csv(file)
    best = df.loc[df["objective_score"].idxmax()]

    return {
        "objective_score": float(best["objective_score"]),
        "hit_ratio": float(best["hit_ratio"]),
        "ic_mean": float(best["ic_mean"]),
        "icir": float(best["icir"]),
        "rank_ic_mean": float(best.get("rank_ic_mean", 0)),
        "rank_icir": float(best.get("rank_icir", 0)),
    }


def compare_dev_holdout(
    dev_results: dict, holdout_results: dict, grid_dev: dict
) -> dict:
    """Dev/Holdout 구간 성과 비교"""
    comparison = {
        "metrics_comparison": {},
        "overfitting_analysis": {},
        "recommendations": [],
    }

    # Grid Search Dev 결과와 실제 Holdout 결과 비교
    if grid_dev and holdout_results:
        metrics = ["hit_ratio", "ic_mean", "icir", "rank_ic_mean", "rank_icir"]

        for metric in metrics:
            grid_val = grid_dev.get(metric, 0)
            holdout_val = holdout_results.get(metric, 0)

            if metric == "hit_ratio":
                diff = holdout_val - grid_val
                pct_diff = (diff / grid_val * 100) if grid_val > 0 else 0
            else:
                diff = holdout_val - grid_val
                pct_diff = (diff / abs(grid_val) * 100) if abs(grid_val) > 1e-10 else 0

            comparison["metrics_comparison"][metric] = {
                "grid_dev": grid_val,
                "holdout": holdout_val,
                "difference": diff,
                "pct_difference": pct_diff,
            }

            # 과적합 평가
            if abs(pct_diff) > 30:  # 30% 이상 차이
                risk_level = "high"
            elif abs(pct_diff) > 15:  # 15% 이상 차이
                risk_level = "medium"
            else:
                risk_level = "low"

            comparison["overfitting_analysis"][metric] = {
                "risk_level": risk_level,
                "pct_difference": abs(pct_diff),
            }

        # 종합 과적합 위험 평가
        high_risk_count = sum(
            1
            for v in comparison["overfitting_analysis"].values()
            if v["risk_level"] == "high"
        )
        medium_risk_count = sum(
            1
            for v in comparison["overfitting_analysis"].values()
            if v["risk_level"] == "medium"
        )

        if high_risk_count >= 2:
            overall_risk = "high"
            comparison["recommendations"].append(
                "⚠️ 높은 과적합 위험: Holdout 구간 성과가 Dev 구간 대비 크게 저하됨"
            )
            comparison["recommendations"].append(
                "  - 정규화 강화 (Ridge alpha 증가) 필수"
            )
            comparison["recommendations"].append("  - 피처 수 감소 또는 피처 선택 강화")
            comparison["recommendations"].append("  - 모델 단순화 검토")
        elif medium_risk_count >= 2 or high_risk_count >= 1:
            overall_risk = "medium"
            comparison["recommendations"].append(
                "⚠️ 중간 과적합 위험: 일부 지표에서 성과 차이 관찰됨"
            )
            comparison["recommendations"].append("  - 정규화 조정 검토")
            comparison["recommendations"].append("  - 추가 검증 데이터로 재평가")
        else:
            overall_risk = "low"
            comparison["recommendations"].append(
                "✅ 낮은 과적합 위험: Dev/Holdout 구간 성과가 유사함"
            )
            comparison["recommendations"].append("  - 현재 모델 설정 유지 가능")

        comparison["overall_risk"] = overall_risk

    # 실제 Dev와 Holdout 비교 (선택적)
    if dev_results and holdout_results:
        comparison["actual_dev_holdout"] = {}
        metrics = ["hit_ratio", "ic_mean", "icir", "rank_ic_mean", "rank_icir"]

        for metric in metrics:
            dev_val = dev_results.get(metric, 0)
            holdout_val = holdout_results.get(metric, 0)
            diff = holdout_val - dev_val

            comparison["actual_dev_holdout"][metric] = {
                "dev": dev_val,
                "holdout": holdout_val,
                "difference": diff,
            }

    return comparison


def main():
    """메인 함수"""
    print("=" * 80)
    print("Holdout 구간 성과 평가 및 Dev/Holdout 구간 성과 비교")
    print("=" * 80)

    data_dir = project_root / "data" / "interim"
    configs_dir = project_root / "configs"
    output_dir = project_root / "artifacts" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    # config.yaml 로드
    with open(configs_dir / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    all_results = {}

    for horizon in ["short", "long"]:
        print(f"\n[{horizon.upper()} 랭킹]")
        print("-" * 80)

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
                    how="left",
                )

        # CV folds 로드
        cv_folds_file = data_dir / f"cv_folds_{horizon}.parquet"
        if not cv_folds_file.exists():
            print(f"⚠️  CV folds 파일을 찾을 수 없습니다: {cv_folds_file}")
            continue

        cv_folds = pd.read_parquet(cv_folds_file)

        # 최적 가중치 파일 경로
        l8_config = cfg.get(f"l8_{horizon}", {})
        feature_groups_config = l8_config.get("feature_groups_config")

        if not feature_groups_config:
            print("⚠️  feature_groups_config가 설정되지 않았습니다.")
            continue

        print(f"최적 가중치 파일: {feature_groups_config}")

        # Feature groups 경로 확인
        base_dir = Path(cfg.get("paths", {}).get("base_dir", project_root))
        feature_groups_path = base_dir / feature_groups_config

        if not feature_groups_path.exists():
            print(f"⚠️  Feature groups 파일을 찾을 수 없습니다: {feature_groups_path}")
            continue

        # L8 랭킹 생성 (build_ranking_daily 직접 호출)
        print("L8 랭킹 생성 중...")
        try:
            # 필수 컬럼 확인
            required_cols = ["date", "ticker"]
            missing_cols = [c for c in required_cols if c not in panel_df.columns]
            if missing_cols:
                raise KeyError(
                    f"Input DataFrame missing required columns: {missing_cols}"
                )

            # 날짜/티커 정규화
            panel_df["date"] = pd.to_datetime(panel_df["date"], errors="raise")
            panel_df["ticker"] = panel_df["ticker"].astype(str).str.zfill(6)

            # in_universe 확인
            if "in_universe" not in panel_df.columns:
                panel_df["in_universe"] = True
            else:
                panel_df["in_universe"] = (
                    panel_df["in_universe"].fillna(False).astype(bool)
                )

            # 랭킹 생성 (feature_groups_config는 경로 전달)
            ranking_daily = build_ranking_daily(
                panel_df,
                normalization_method=l8_config.get("normalization_method", "zscore"),
                feature_groups_config=feature_groups_path,  # Path 객체 전달
                use_sector_relative=l8_config.get("use_sector_relative", True),
                sector_col=l8_config.get("sector_col", "sector_name"),
            )

            if len(ranking_daily) == 0:
                print("⚠️  랭킹 결과가 비어있습니다.")
                continue

            print(f"✅ 랭킹 생성 완료: {len(ranking_daily)}행")

            # Forward Returns 준비
            forward_returns = panel_df[["date", "ticker", ret_col]].dropna(
                subset=[ret_col]
            )

            # Dev/Holdout 구간별 평가
            print("Dev 구간 평가 중...")
            dev_results = evaluate_on_fold(
                ranking_daily, forward_returns, cv_folds, "dev", horizon
            )

            print("Holdout 구간 평가 중...")
            holdout_results = evaluate_on_fold(
                ranking_daily, forward_returns, cv_folds, "holdout", horizon
            )

            if dev_results is None or holdout_results is None:
                print("⚠️  Dev 또는 Holdout 구간 평가 실패")
                continue

            # Grid Search Dev 결과 로드
            grid_dev = load_grid_search_results(horizon)

            # 결과 출력
            print("\n[Dev 구간] (실제 평가)")
            print(f"  Hit Ratio: {dev_results['hit_ratio']*100:.2f}%")
            print(f"  IC Mean: {dev_results['ic_mean']:.4f}")
            print(f"  Rank IC Mean: {dev_results['rank_ic_mean']:.4f}")
            print(f"  ICIR: {dev_results['icir']:.4f}")
            print(f"  Rank ICIR: {dev_results['rank_icir']:.4f}")

            if grid_dev:
                print("\n[Dev 구간] (Grid Search 결과)")
                print(f"  Hit Ratio: {grid_dev['hit_ratio']*100:.2f}%")
                print(f"  IC Mean: {grid_dev['ic_mean']:.4f}")
                print(f"  ICIR: {grid_dev['icir']:.4f}")

            print("\n[Holdout 구간]")
            print(f"  Hit Ratio: {holdout_results['hit_ratio']*100:.2f}%")
            print(f"  IC Mean: {holdout_results['ic_mean']:.4f}")
            print(f"  Rank IC Mean: {holdout_results['rank_ic_mean']:.4f}")
            print(f"  ICIR: {holdout_results['icir']:.4f}")
            print(f"  Rank ICIR: {holdout_results['rank_icir']:.4f}")

            # Dev/Holdout 비교
            print("\nDev/Holdout 구간 성과 비교 중...")
            comparison = compare_dev_holdout(dev_results, holdout_results, grid_dev)

            if comparison:
                print("\n[과적합 위험 평가]")
                print(
                    f"  종합 위험도: {comparison.get('overall_risk', 'unknown').upper()}"
                )

                print("\n[지표별 차이] (Grid Dev vs Holdout)")
                for metric, comp in comparison.get("metrics_comparison", {}).items():
                    print(f"  {metric}:")
                    print(f"    Grid Dev: {comp['grid_dev']:.4f}")
                    print(f"    Holdout: {comp['holdout']:.4f}")
                    print(
                        f"    차이: {comp['difference']:.4f} ({comp['pct_difference']:.1f}%)"
                    )
                    risk = comparison["overfitting_analysis"].get(metric, {})
                    print(f"    위험도: {risk.get('risk_level', 'unknown')}")

                print("\n[권장사항]")
                for rec in comparison.get("recommendations", []):
                    print(f"  {rec}")

            all_results[horizon] = {
                "dev": dev_results,
                "holdout": holdout_results,
                "grid_dev": grid_dev,
                "comparison": comparison,
            }

        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            import traceback

            traceback.print_exc()

    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"dev_holdout_final_comparison_{timestamp}.md"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Dev/Holdout 구간 성과 비교 (최종 확인)\n\n")
        f.write(f"**생성일**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("=" * 80 + "\n\n")

        for horizon, results in all_results.items():
            f.write(f"## {horizon.upper()} 랭킹\n\n")

            dev = results["dev"]
            holdout = results["holdout"]
            grid_dev = results.get("grid_dev")
            comparison = results.get("comparison", {})

            # 성과 비교 테이블
            f.write("### 성과 비교\n\n")
            f.write(
                "| 지표 | Grid Dev | 실제 Dev | Holdout | Holdout-Dev 차이 | 위험도 |\n"
            )
            f.write(
                "|------|----------|----------|---------|------------------|--------|\n"
            )

            metrics = ["hit_ratio", "ic_mean", "icir", "rank_ic_mean", "rank_icir"]
            for metric in metrics:
                grid_val = grid_dev.get(metric, 0) if grid_dev else 0
                dev_val = dev.get(metric, 0)
                holdout_val = holdout.get(metric, 0)
                diff = holdout_val - dev_val

                risk_info = comparison.get("overfitting_analysis", {}).get(metric, {})
                risk_level = risk_info.get("risk_level", "unknown")

                f.write(
                    f"| {metric} | {grid_val:.4f} | {dev_val:.4f} | {holdout_val:.4f} | "
                    f"{diff:.4f} | {risk_level} |\n"
                )

            f.write("\n### 과적합 위험 평가\n\n")
            overall_risk = comparison.get("overall_risk", "unknown")
            f.write(f"**종합 위험도**: {overall_risk.upper()}\n\n")

            f.write("### 권장사항\n\n")
            for rec in comparison.get("recommendations", []):
                f.write(f"{rec}\n")

            f.write("\n" + "-" * 80 + "\n\n")

    print(f"\n✅ 결과 저장: {report_file}")

    return all_results


if __name__ == "__main__":
    main()
