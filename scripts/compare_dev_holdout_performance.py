"""
Dev/Holdout 구간 성과 비교 및 과적합 분석

1. 최적 가중치 적용한 L8 랭킹 실행
2. Dev/Holdout 구간별 성과 평가
3. 과적합 분석
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

from src.tracks.track_a.stages.ranking.l8_dual_horizon import (
    run_L8_long_rank_engine,
    run_L8_short_rank_engine,
)
from src.tracks.track_b.stages.backtest.l7_backtest import _rank_ic, _safe_corr


def calculate_ic(scores: pd.Series, returns: pd.Series) -> float:
    """IC (Pearson correlation)"""
    return _safe_corr(scores, returns)


def calculate_rank_ic(scores: pd.Series, returns: pd.Series) -> float:
    """Rank IC (Spearman correlation)"""
    return _rank_ic(scores, returns)


def calculate_hit_ratio(scores: np.ndarray, returns: np.ndarray) -> float:
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


def load_cv_folds(data_dir: Path, horizon: str) -> pd.DataFrame:
    """CV folds 로드"""
    cv_folds_file = data_dir / f"cv_folds_{horizon}.parquet"
    if not cv_folds_file.exists():
        raise FileNotFoundError(f"CV folds 파일을 찾을 수 없습니다: {cv_folds_file}")
    return pd.read_parquet(cv_folds_file)


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
        folds = cv_folds[cv_folds["fold_id"] != "holdout"]
    else:  # holdout
        folds = cv_folds[cv_folds["fold_id"] == "holdout"]

    if len(folds) == 0:
        return None

    # 날짜 추출
    dates = pd.to_datetime(folds["test_end"].unique(), errors="coerce")
    dates = dates[dates.notna()]

    if len(dates) == 0:
        return None

    # 해당 구간의 랭킹과 수익률 필터링
    ranking_filtered = ranking_daily[ranking_daily["date"].isin(dates)].copy()
    returns_filtered = forward_returns[
        (forward_returns["date"].isin(dates)) & (forward_returns[ret_col].notna())
    ].copy()

    if len(ranking_filtered) == 0 or len(returns_filtered) == 0:
        return None

    # Merge
    merged = ranking_filtered.merge(
        returns_filtered[["date", "ticker", ret_col]],
        on=["date", "ticker"],
        how="inner",
    )

    if len(merged) == 0:
        return None

    # 날짜별로 그룹화하여 평가
    results_by_date = []

    for date in merged["date"].unique():
        date_data = merged[merged["date"] == date]

        if len(date_data) < 2:
            continue

        scores = pd.Series(date_data["score_total"].values)
        returns = pd.Series(date_data[ret_col].values)

        # 지표 계산
        hit_ratio = calculate_hit_ratio(
            returns.values, returns.values
        )  # returns > 0 비율
        ic = calculate_ic(scores, returns)
        rank_ic = calculate_rank_ic(scores, returns)

        results_by_date.append(
            {
                "date": date,
                "hit_ratio": hit_ratio,
                "ic": ic,
                "rank_ic": rank_ic,
                "n_observations": len(date_data),
            }
        )

    if len(results_by_date) == 0:
        return None

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


def analyze_overfitting(dev_results: dict, holdout_results: dict) -> dict:
    """과적합 분석"""
    if dev_results is None or holdout_results is None:
        return None

    analysis = {
        "metrics_comparison": {},
        "overfitting_risk": {},
        "recommendations": [],
    }

    # 지표별 비교
    metrics = ["hit_ratio", "ic_mean", "rank_ic_mean", "icir", "rank_icir"]

    for metric in metrics:
        dev_val = dev_results.get(metric, 0)
        holdout_val = holdout_results.get(metric, 0)
        diff = holdout_val - dev_val
        pct_diff = (diff / abs(dev_val) * 100) if dev_val != 0 else 0

        analysis["metrics_comparison"][metric] = {
            "dev": dev_val,
            "holdout": holdout_val,
            "difference": diff,
            "pct_difference": pct_diff,
        }

        # 과적합 위험 평가
        if abs(pct_diff) > 20:  # 20% 이상 차이
            risk_level = "high"
        elif abs(pct_diff) > 10:  # 10% 이상 차이
            risk_level = "medium"
        else:
            risk_level = "low"

        analysis["overfitting_risk"][metric] = {
            "risk_level": risk_level,
            "pct_difference": abs(pct_diff),
        }

    # 종합 과적합 위험 평가
    high_risk_count = sum(
        1 for v in analysis["overfitting_risk"].values() if v["risk_level"] == "high"
    )
    medium_risk_count = sum(
        1 for v in analysis["overfitting_risk"].values() if v["risk_level"] == "medium"
    )

    if high_risk_count >= 2:
        overall_risk = "high"
        analysis["recommendations"].append(
            "⚠️ 높은 과적합 위험: Holdout 구간 성과가 Dev 구간 대비 크게 저하됨"
        )
        analysis["recommendations"].append("  - 정규화 강화 (Ridge alpha 증가)")
        analysis["recommendations"].append("  - 피처 수 감소 또는 피처 선택 강화")
    elif medium_risk_count >= 2 or high_risk_count >= 1:
        overall_risk = "medium"
        analysis["recommendations"].append(
            "⚠️ 중간 과적합 위험: 일부 지표에서 성과 차이 관찰됨"
        )
        analysis["recommendations"].append("  - 정규화 조정 검토")
        analysis["recommendations"].append("  - 추가 검증 데이터로 재평가")
    else:
        overall_risk = "low"
        analysis["recommendations"].append(
            "✅ 낮은 과적합 위험: Dev/Holdout 구간 성과가 유사함"
        )
        analysis["recommendations"].append("  - 현재 모델 설정 유지 가능")

    analysis["overall_risk"] = overall_risk

    return analysis


def main():
    """메인 함수"""
    print("=" * 80)
    print("Dev/Holdout 구간 성과 비교 및 과적합 분석")
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
        try:
            cv_folds = load_cv_folds(data_dir, horizon)
        except FileNotFoundError as e:
            print(f"⚠️  {e}")
            continue

        # 최적 가중치 파일 경로
        l8_config = cfg.get(f"l8_{horizon}", {})
        feature_groups_config = l8_config.get("feature_groups_config")

        if not feature_groups_config:
            print("⚠️  feature_groups_config가 설정되지 않았습니다.")
            continue

        print(f"최적 가중치 파일: {feature_groups_config}")

        # L8 랭킹 실행
        print("L8 랭킹 실행 중...")
        try:
            # cfg와 artifacts 구성
            artifacts = {
                "panel_merged_daily": panel_df,
                "dataset_daily": panel_df,
            }

            # horizon에 따라 적절한 함수 호출
            if horizon == "short":
                ranking_result, warns = run_L8_short_rank_engine(
                    cfg,
                    artifacts,
                    force=True,
                )
                ranking_daily = ranking_result.get(
                    "ranking_short_daily", pd.DataFrame()
                )
            else:  # long
                ranking_result, warns = run_L8_long_rank_engine(
                    cfg,
                    artifacts,
                    force=True,
                )
                ranking_daily = ranking_result.get("ranking_long_daily", pd.DataFrame())

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

            # 결과 출력
            print("\n[Dev 구간]")
            print(f"  Hit Ratio: {dev_results['hit_ratio']*100:.2f}%")
            print(f"  IC Mean: {dev_results['ic_mean']:.4f}")
            print(f"  Rank IC Mean: {dev_results['rank_ic_mean']:.4f}")
            print(f"  ICIR: {dev_results['icir']:.4f}")
            print(f"  Rank ICIR: {dev_results['rank_icir']:.4f}")

            print("\n[Holdout 구간]")
            print(f"  Hit Ratio: {holdout_results['hit_ratio']*100:.2f}%")
            print(f"  IC Mean: {holdout_results['ic_mean']:.4f}")
            print(f"  Rank IC Mean: {holdout_results['rank_ic_mean']:.4f}")
            print(f"  ICIR: {holdout_results['icir']:.4f}")
            print(f"  Rank ICIR: {holdout_results['rank_icir']:.4f}")

            # 과적합 분석
            print("\n과적합 분석 중...")
            overfitting_analysis = analyze_overfitting(dev_results, holdout_results)

            if overfitting_analysis:
                print("\n[과적합 위험 평가]")
                print(f"  종합 위험도: {overfitting_analysis['overall_risk'].upper()}")

                print("\n[지표별 차이]")
                for metric, comp in overfitting_analysis["metrics_comparison"].items():
                    print(f"  {metric}:")
                    print(f"    Dev: {comp['dev']:.4f}")
                    print(f"    Holdout: {comp['holdout']:.4f}")
                    print(
                        f"    차이: {comp['difference']:.4f} ({comp['pct_difference']:.1f}%)"
                    )
                    risk = overfitting_analysis["overfitting_risk"][metric]
                    print(f"    위험도: {risk['risk_level']}")

                print("\n[권장사항]")
                for rec in overfitting_analysis["recommendations"]:
                    print(f"  {rec}")

            all_results[horizon] = {
                "dev": dev_results,
                "holdout": holdout_results,
                "overfitting_analysis": overfitting_analysis,
            }

        except Exception as e:
            print(f"❌ 오류 발생: {e}")
            import traceback

            traceback.print_exc()

    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"dev_holdout_overfitting_analysis_{timestamp}.md"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Dev/Holdout 구간 성과 비교 및 과적합 분석\n\n")
        f.write(f"**생성일**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("=" * 80 + "\n\n")

        for horizon, results in all_results.items():
            f.write(f"## {horizon.upper()} 랭킹\n\n")

            dev = results["dev"]
            holdout = results["holdout"]
            analysis = results["overfitting_analysis"]

            # 성과 비교 테이블
            f.write("### 성과 비교\n\n")
            f.write("| 지표 | Dev 구간 | Holdout 구간 | 차이 | 차이(%) | 위험도 |\n")
            f.write("|------|----------|--------------|------|---------|--------|\n")

            for metric, comp in analysis["metrics_comparison"].items():
                risk = analysis["overfitting_risk"][metric]
                f.write(
                    f"| {metric} | {comp['dev']:.4f} | {comp['holdout']:.4f} | "
                    f"{comp['difference']:.4f} | {comp['pct_difference']:.1f}% | "
                    f"{risk['risk_level']} |\n"
                )

            f.write("\n### 과적합 위험 평가\n\n")
            f.write(f"**종합 위험도**: {analysis['overall_risk'].upper()}\n\n")

            f.write("### 권장사항\n\n")
            for rec in analysis["recommendations"]:
                f.write(f"{rec}\n")

            f.write("\n" + "-" * 80 + "\n\n")

    print(f"\n✅ 결과 저장: {report_file}")

    return all_results


if __name__ == "__main__":
    main()
