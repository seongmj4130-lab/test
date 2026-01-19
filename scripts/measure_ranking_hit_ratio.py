"""
랭킹산정모델 Hit Ratio 측정 및 과적합 판단 스크립트

목적:
1. 랭킹 모델만 실행 (L6R)
2. Hit Ratio 측정 (Dev/Holdout 구분)
3. 과적합 여부 판단 (Dev vs Holdout 비교)

파라미터 최적화:
- ridge_alpha: 0.8 → 1.0~3.0 (과적합 방지)
- min_feature_ic: 0.01 → 0.005 (피처 확대)
- alpha_short: 0.5 → 0.6 (bull), 0.4 (bear) (국면 적응)
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.stages.modeling.l6r_ranking_scoring import run_L6R_ranking_scoring
from src.utils.config import load_config
from src.utils.io import artifact_exists, load_artifact


def calculate_hit_ratio(
    scores: pd.DataFrame,
    return_col: str = "true_short",
    score_col: str = "score_ens",
    phase_col: str = "phase",
) -> dict:
    """
    Hit Ratio 계산: 예측 방향과 실제 수익률 방향이 일치하는 비율

    Args:
        scores: rebalance_scores DataFrame (date, ticker, score_ens, true_short, phase 포함)
        return_col: 실제 수익률 컬럼명
        score_col: 예측 스코어 컬럼명
        phase_col: Phase 컬럼명 (dev/holdout 구분)

    Returns:
        dict: phase별 Hit Ratio 및 전체 Hit Ratio
    """
    df = scores.copy()

    # 필수 컬럼 체크
    required_cols = [return_col, score_col, phase_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")

    # NaN 제거
    df = df.dropna(subset=[return_col, score_col])

    if len(df) == 0:
        return {"overall": np.nan, "dev": np.nan, "holdout": np.nan, "n_samples": 0}

    # 방향 일치 여부 계산 (양수/음수 방향 일치)
    df["pred_direction"] = np.sign(df[score_col])
    df["actual_direction"] = np.sign(df[return_col])
    df["hit"] = (df["pred_direction"] == df["actual_direction"]).astype(int)

    # 전체 Hit Ratio
    overall_hr = float(df["hit"].mean())

    # Phase별 Hit Ratio
    phase_hr = {}
    for phase in df[phase_col].unique():
        phase_df = df[df[phase_col] == phase]
        if len(phase_df) > 0:
            phase_hr[phase] = float(phase_df["hit"].mean())

    result = {"overall": overall_hr, "n_samples": int(len(df)), **phase_hr}

    return result


def calculate_individual_hit_ratios(
    scores: pd.DataFrame, phase_col: str = "phase"
) -> dict:
    """
    단기/장기 랭킹의 개별 Hit Ratio 계산

    Args:
        scores: rebalance_scores DataFrame
        phase_col: Phase 컬럼명

    Returns:
        dict: 단기/장기/통합 Hit Ratio 결과
    """
    results = {}

    # 1. 단기 랭킹 Hit Ratio (score_total_short vs true_short)
    if "score_total_short" in scores.columns and "true_short" in scores.columns:
        short_hr = calculate_hit_ratio(
            scores,
            return_col="true_short",
            score_col="score_total_short",
            phase_col=phase_col,
        )
        results["short"] = short_hr
        results["short"][
            "description"
        ] = "단기 랭킹 (score_total_short vs true_short, 20일 수익률)"
    else:
        results["short"] = {"overall": np.nan, "description": "단기 랭킹 데이터 없음"}

    # 2. 장기 랭킹 Hit Ratio (score_total_long vs true_long)
    if "score_total_long" in scores.columns and "true_long" in scores.columns:
        long_hr = calculate_hit_ratio(
            scores,
            return_col="true_long",
            score_col="score_total_long",
            phase_col=phase_col,
        )
        results["long"] = long_hr
        results["long"][
            "description"
        ] = "장기 랭킹 (score_total_long vs true_long, 120일 수익률)"
    else:
        results["long"] = {"overall": np.nan, "description": "장기 랭킹 데이터 없음"}

    # 3. 통합 랭킹 Hit Ratio (score_ens vs true_short)
    if "score_ens" in scores.columns and "true_short" in scores.columns:
        ens_hr = calculate_hit_ratio(
            scores, return_col="true_short", score_col="score_ens", phase_col=phase_col
        )
        results["ensemble"] = ens_hr
        results["ensemble"][
            "description"
        ] = "통합 랭킹 (score_ens vs true_short, 단기/장기 결합)"
    else:
        results["ensemble"] = {
            "overall": np.nan,
            "description": "통합 랭킹 데이터 없음",
        }

    return results


def detect_overfitting(
    dev_hr: float, holdout_hr: float, threshold: float = 0.10
) -> dict:
    """
    과적합 여부 판단: Dev와 Holdout Hit Ratio 차이 분석

    Args:
        dev_hr: Dev 구간 Hit Ratio
        holdout_hr: Holdout 구간 Hit Ratio
        threshold: 과적합 판단 임계값 (기본 10%p)

    Returns:
        dict: 과적합 판단 결과
    """
    if pd.isna(dev_hr) or pd.isna(holdout_hr):
        return {
            "is_overfitting": None,
            "gap": np.nan,
            "severity": "unknown",
            "message": "Dev 또는 Holdout Hit Ratio가 NaN입니다.",
        }

    gap = dev_hr - holdout_hr

    if gap > threshold:
        severity = "high" if gap > 0.15 else "medium"
        is_overfitting = True
        message = f"과적합 의심: Dev({dev_hr:.2%}) - Holdout({holdout_hr:.2%}) = {gap:.2%}p (임계값 {threshold:.0%}p 초과)"
    elif gap < -threshold:
        severity = "negative"
        is_overfitting = False
        message = f"Holdout 성과가 더 좋음: Dev({dev_hr:.2%}) < Holdout({holdout_hr:.2%}) (차이: {gap:.2%}p)"
    else:
        severity = "low"
        is_overfitting = False
        message = f"정상 범위: Dev({dev_hr:.2%}) - Holdout({holdout_hr:.2%}) = {gap:.2%}p (임계값 {threshold:.0%}p 이내)"

    return {
        "is_overfitting": is_overfitting,
        "gap": float(gap),
        "severity": severity,
        "message": message,
    }


def run_ranking_model_only(
    cfg: dict, artifacts: dict, force: bool = False
) -> pd.DataFrame:
    """
    랭킹 모델만 실행 (L6R)

    Args:
        cfg: config.yaml dict
        artifacts: 필요한 artifacts (dataset_daily, cv_folds_short, universe_k200_membership_monthly 등)
        force: 강제 재실행 여부

    Returns:
        rebalance_scores DataFrame
    """
    print("[랭킹 모델 실행] L6R 랭킹 스코어링 시작...")

    # L6R 실행
    outputs, warns = run_L6R_ranking_scoring(
        cfg=cfg,
        artifacts=artifacts,
        force=force,
    )

    if warns:
        print(f"[경고] {len(warns)}개 경고 발생:")
        for w in warns[:10]:  # 최대 10개만 출력
            print(f"  - {w}")
        if len(warns) > 10:
            print(f"  ... 외 {len(warns) - 10}개 경고")

    rebalance_scores = outputs.get("rebalance_scores")
    if rebalance_scores is None or len(rebalance_scores) == 0:
        raise ValueError("rebalance_scores가 생성되지 않았습니다.")

    print(f"[랭킹 모델 실행] 완료: {len(rebalance_scores):,}행")
    return rebalance_scores


def main():
    parser = argparse.ArgumentParser(
        description="랭킹산정모델 Hit Ratio 측정 및 과적합 판단"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Config 파일 경로"
    )
    parser.add_argument("--force", action="store_true", help="강제 재실행")
    parser.add_argument("--output", type=str, default=None, help="결과 저장 경로 (CSV)")
    parser.add_argument(
        "--generate-l8",
        action="store_true",
        help="L8 단기/장기 랭킹 자동 생성 (없을 경우)",
    )
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"ERROR: Config 파일을 찾을 수 없습니다: {config_path}", file=sys.stderr)
        sys.exit(1)

    # Config 로드
    print(f"[Config 로드] {config_path}")
    cfg = load_config(str(config_path))

    # 필요한 artifacts 로드
    interim_dir = PROJECT_ROOT / "data" / "interim"

    print("\n[Artifacts 로드]")
    artifacts = {}

    required_artifacts = {
        "dataset_daily": "dataset_daily",
        "cv_folds_short": "cv_folds_short",
        "universe_k200_membership_monthly": "universe_k200_membership_monthly",
    }

    for key, filename in required_artifacts.items():
        artifact_path = interim_dir / filename
        if not artifact_exists(artifact_path):
            print(f"ERROR: {key}를 찾을 수 없습니다: {artifact_path}", file=sys.stderr)
            sys.exit(1)
        artifacts[key] = load_artifact(artifact_path)
        print(f"  ✓ {key}: {len(artifacts[key]):,}행")

    # [Dual Horizon] 단기/장기 랭킹 로드 (필수)
    ranking_short_path = interim_dir / "ranking_short_daily"
    ranking_long_path = interim_dir / "ranking_long_daily"

    if artifact_exists(ranking_short_path) and artifact_exists(ranking_long_path):
        artifacts["ranking_short_daily"] = load_artifact(ranking_short_path)
        artifacts["ranking_long_daily"] = load_artifact(ranking_long_path)
        print(f"  ✓ ranking_short_daily: {len(artifacts['ranking_short_daily']):,}행")
        print(f"  ✓ ranking_long_daily: {len(artifacts['ranking_long_daily']):,}행")
        print("  → Dual Horizon 모드 활성화 (단기/장기 랭킹 결합, α=0.5:0.5)")
    else:
        print("\n[경고] ranking_short_daily 또는 ranking_long_daily가 없습니다.")
        if not artifact_exists(ranking_short_path):
            print(f"  ✗ {ranking_short_path} 없음")
        if not artifact_exists(ranking_long_path):
            print(f"  ✗ {ranking_long_path} 없음")
        print("\n[해결 방법] L8 단기/장기 랭킹을 먼저 생성해야 합니다:")
        print("  python -m src.pipeline.track_a_pipeline")
        print("\n또는 --generate-l8 플래그를 사용하여 자동 생성할 수 있습니다.")
        print("  (단일 랭킹 모드로 실행하려면 Enter를 누르세요)")

        # 사용자에게 선택권 제공 (자동 생성 옵션)
        # force 플래그가 있으면 무조건 재생성
        if args.force or args.generate_l8:
            print("\n[자동 생성] L8 단기/장기 랭킹 생성 중...")
            try:
                from src.pipeline.track_a_pipeline import run_track_a_pipeline

                track_a_outputs = run_track_a_pipeline(
                    config_path=str(config_path), force_rebuild=args.force
                )
                artifacts["ranking_short_daily"] = track_a_outputs.get(
                    "ranking_short_daily"
                )
                artifacts["ranking_long_daily"] = track_a_outputs.get(
                    "ranking_long_daily"
                )
                if (
                    artifacts["ranking_short_daily"] is not None
                    and artifacts["ranking_long_daily"] is not None
                ):
                    print("  ✓ L8 랭킹 생성 완료")
                    print(
                        f"    - ranking_short_daily: {len(artifacts['ranking_short_daily']):,}행"
                    )
                    print(
                        f"    - ranking_long_daily: {len(artifacts['ranking_long_daily']):,}행"
                    )
                else:
                    raise ValueError("L8 랭킹 생성 실패")
            except Exception as e:
                print(f"  ✗ L8 랭킹 생성 실패: {e}", file=sys.stderr)
                import traceback

                traceback.print_exc()
                print("  → 단일 랭킹 모드로 계속 진행합니다.")
        else:
            print("  → 단일 랭킹 모드로 실행합니다.")

    # ohlcv_daily (선택적, 시장 국면용)
    ohlcv_path = interim_dir / "ohlcv_daily"
    if artifact_exists(ohlcv_path):
        artifacts["ohlcv_daily"] = load_artifact(ohlcv_path)
        print(f"  ✓ ohlcv_daily: {len(artifacts['ohlcv_daily']):,}행")

    # 랭킹 모델 실행
    print("\n" + "=" * 60)
    print("[1단계] 랭킹 모델 실행 (L6R)")
    print("=" * 60)

    rebalance_scores = run_ranking_model_only(cfg, artifacts, force=args.force)

    # Hit Ratio 계산 (개별 + 통합)
    print("\n" + "=" * 60)
    print("[2단계] Hit Ratio 계산 (단기/장기/통합)")
    print("=" * 60)

    # 개별 Hit Ratio 계산
    individual_hr = calculate_individual_hit_ratios(rebalance_scores, phase_col="phase")

    # 단기 랭킹 Hit Ratio
    print("\n[단기 랭킹 Hit Ratio]")
    print(f"  기준: {individual_hr['short']['description']}")
    if not pd.isna(individual_hr["short"].get("overall", np.nan)):
        print(
            f"  전체: {individual_hr['short']['overall']:.2%} (n={individual_hr['short']['n_samples']:,})"
        )
        if "dev" in individual_hr["short"]:
            print(f"  Dev: {individual_hr['short']['dev']:.2%}")
        if "holdout" in individual_hr["short"]:
            print(f"  Holdout: {individual_hr['short']['holdout']:.2%}")
    else:
        print("  데이터 없음")

    # 장기 랭킹 Hit Ratio
    print("\n[장기 랭킹 Hit Ratio]")
    print(f"  기준: {individual_hr['long']['description']}")
    if not pd.isna(individual_hr["long"].get("overall", np.nan)):
        print(
            f"  전체: {individual_hr['long']['overall']:.2%} (n={individual_hr['long']['n_samples']:,})"
        )
        if "dev" in individual_hr["long"]:
            print(f"  Dev: {individual_hr['long']['dev']:.2%}")
        if "holdout" in individual_hr["long"]:
            print(f"  Holdout: {individual_hr['long']['holdout']:.2%}")
    else:
        print("  데이터 없음")

    # 통합 랭킹 Hit Ratio
    print("\n[통합 랭킹 Hit Ratio]")
    print(f"  기준: {individual_hr['ensemble']['description']}")
    if not pd.isna(individual_hr["ensemble"].get("overall", np.nan)):
        print(
            f"  전체: {individual_hr['ensemble']['overall']:.2%} (n={individual_hr['ensemble']['n_samples']:,})"
        )
        if "dev" in individual_hr["ensemble"]:
            print(f"  Dev: {individual_hr['ensemble']['dev']:.2%}")
        if "holdout" in individual_hr["ensemble"]:
            print(f"  Holdout: {individual_hr['ensemble']['holdout']:.2%}")
    else:
        print("  데이터 없음")

    # 통합 결과를 기존 변수에 저장 (하위 호환성)
    hit_ratio_results = individual_hr["ensemble"]

    # 과적합 판단
    print("\n" + "=" * 60)
    print("[3단계] 과적합 여부 판단")
    print("=" * 60)

    dev_hr = hit_ratio_results.get("dev", np.nan)
    holdout_hr = hit_ratio_results.get("holdout", np.nan)

    overfitting_result = detect_overfitting(dev_hr, holdout_hr, threshold=0.10)

    print("\n[과적합 판단 결과]")
    print(f"  {overfitting_result['message']}")
    print(f"  심각도: {overfitting_result['severity']}")
    print(f"  Gap: {overfitting_result['gap']:.2%}p")

    # Config 파라미터 정보 출력
    print("\n" + "=" * 60)
    print("[현재 설정 파라미터]")
    print("=" * 60)

    l5 = cfg.get("l5", {}) or {}
    l6r = cfg.get("l6r", {}) or {}

    print("\n[L5 모델 학습]")
    print(f"  ridge_alpha: {l5.get('ridge_alpha', 'N/A')}")
    print(f"  min_feature_ic: {l5.get('min_feature_ic', 'N/A')}")
    print(f"  filter_features_by_ic: {l5.get('filter_features_by_ic', 'N/A')}")
    print(f"  use_rank_ic: {l5.get('use_rank_ic', 'N/A')}")

    print("\n[L6R 랭킹 스코어링]")
    print(f"  alpha_short: {l6r.get('alpha_short', 'N/A')}")
    print(f"  alpha_long: {l6r.get('alpha_long', 'N/A')}")
    regime_alpha = l6r.get("regime_alpha", {})
    if regime_alpha:
        print("  regime_alpha:")
        for regime, alpha in regime_alpha.items():
            print(f"    {regime}: {alpha}")

    # 결과 요약
    print("\n" + "=" * 60)
    print("[결과 요약]")
    print("=" * 60)

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hit_ratio": {
            "overall": hit_ratio_results["overall"],
            "dev": hit_ratio_results.get("dev", np.nan),
            "holdout": hit_ratio_results.get("holdout", np.nan),
            "n_samples": hit_ratio_results["n_samples"],
        },
        "hit_ratio_individual": {
            "short": {
                "overall": individual_hr["short"].get("overall", np.nan),
                "dev": individual_hr["short"].get("dev", np.nan),
                "holdout": individual_hr["short"].get("holdout", np.nan),
                "n_samples": individual_hr["short"].get("n_samples", 0),
                "description": individual_hr["short"].get("description", ""),
            },
            "long": {
                "overall": individual_hr["long"].get("overall", np.nan),
                "dev": individual_hr["long"].get("dev", np.nan),
                "holdout": individual_hr["long"].get("holdout", np.nan),
                "n_samples": individual_hr["long"].get("n_samples", 0),
                "description": individual_hr["long"].get("description", ""),
            },
            "ensemble": {
                "overall": individual_hr["ensemble"].get("overall", np.nan),
                "dev": individual_hr["ensemble"].get("dev", np.nan),
                "holdout": individual_hr["ensemble"].get("holdout", np.nan),
                "n_samples": individual_hr["ensemble"].get("n_samples", 0),
                "description": individual_hr["ensemble"].get("description", ""),
            },
        },
        "overfitting": overfitting_result,
        "parameters": {
            "ridge_alpha": l5.get("ridge_alpha"),
            "min_feature_ic": l5.get("min_feature_ic"),
            "alpha_short": l6r.get("alpha_short"),
            "regime_alpha": regime_alpha,
        },
    }

    print(f"\n✓ Hit Ratio: 전체 {hit_ratio_results['overall']:.2%}")
    if not pd.isna(dev_hr):
        print(f"  - Dev: {dev_hr:.2%}")
    if not pd.isna(holdout_hr):
        print(f"  - Holdout: {holdout_hr:.2%}")

    print(f"\n✓ 과적합 판단: {overfitting_result['severity']}")
    print(f"  - {overfitting_result['message']}")

    # 결과 저장
    if args.output:
        output_path = PROJECT_ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # CSV로 저장 (개별 Hit Ratio 포함)
        summary_df = pd.DataFrame(
            [
                {
                    "timestamp": summary["timestamp"],
                    # 통합 랭킹
                    "hit_ratio_ensemble_overall": summary["hit_ratio"]["overall"],
                    "hit_ratio_ensemble_dev": summary["hit_ratio"]["dev"],
                    "hit_ratio_ensemble_holdout": summary["hit_ratio"]["holdout"],
                    "hit_ratio_ensemble_n_samples": summary["hit_ratio"]["n_samples"],
                    # 단기 랭킹
                    "hit_ratio_short_overall": summary["hit_ratio_individual"]["short"][
                        "overall"
                    ],
                    "hit_ratio_short_dev": summary["hit_ratio_individual"]["short"][
                        "dev"
                    ],
                    "hit_ratio_short_holdout": summary["hit_ratio_individual"]["short"][
                        "holdout"
                    ],
                    "hit_ratio_short_n_samples": summary["hit_ratio_individual"][
                        "short"
                    ]["n_samples"],
                    # 장기 랭킹
                    "hit_ratio_long_overall": summary["hit_ratio_individual"]["long"][
                        "overall"
                    ],
                    "hit_ratio_long_dev": summary["hit_ratio_individual"]["long"][
                        "dev"
                    ],
                    "hit_ratio_long_holdout": summary["hit_ratio_individual"]["long"][
                        "holdout"
                    ],
                    "hit_ratio_long_n_samples": summary["hit_ratio_individual"]["long"][
                        "n_samples"
                    ],
                    # 과적합 판단
                    "overfitting_gap": summary["overfitting"]["gap"],
                    "overfitting_severity": summary["overfitting"]["severity"],
                    # 파라미터
                    "ridge_alpha": summary["parameters"]["ridge_alpha"],
                    "min_feature_ic": summary["parameters"]["min_feature_ic"],
                    "alpha_short": summary["parameters"]["alpha_short"],
                }
            ]
        )
        summary_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\n[결과 저장] {output_path}")

        # 개별 Hit Ratio 비교 요약 출력
        print("\n[개별 Hit Ratio 비교 요약]")
        print(
            f"  단기: {summary['hit_ratio_individual']['short']['overall']:.2%} "
            f"(Dev: {summary['hit_ratio_individual']['short']['dev']:.2%}, "
            f"Holdout: {summary['hit_ratio_individual']['short']['holdout']:.2%})"
        )
        print(
            f"  장기: {summary['hit_ratio_individual']['long']['overall']:.2%} "
            f"(Dev: {summary['hit_ratio_individual']['long']['dev']:.2%}, "
            f"Holdout: {summary['hit_ratio_individual']['long']['holdout']:.2%})"
        )
        print(
            f"  통합: {summary['hit_ratio_individual']['ensemble']['overall']:.2%} "
            f"(Dev: {summary['hit_ratio_individual']['ensemble']['dev']:.2%}, "
            f"Holdout: {summary['hit_ratio_individual']['ensemble']['holdout']:.2%})"
        )

    # 목표 달성 여부 확인
    print("\n" + "=" * 60)
    print("[목표 달성 여부]")
    print("=" * 60)

    target_hr = 0.50  # 목표 Hit Ratio 50%
    overall_hr = hit_ratio_results["overall"]

    if overall_hr >= target_hr:
        print(f"✓ 목표 달성: Hit Ratio {overall_hr:.2%} ≥ {target_hr:.0%}")
    else:
        gap = target_hr - overall_hr
        print(
            f"✗ 목표 미달: Hit Ratio {overall_hr:.2%} < {target_hr:.0%} (차이: {gap:.2%}p)"
        )

    if not pd.isna(holdout_hr):
        if holdout_hr >= target_hr:
            print(f"✓ Holdout 목표 달성: {holdout_hr:.2%} ≥ {target_hr:.0%}")
        else:
            gap = target_hr - holdout_hr
            print(
                f"✗ Holdout 목표 미달: {holdout_hr:.2%} < {target_hr:.0%} (차이: {gap:.2%}p)"
            )


if __name__ == "__main__":
    main()
