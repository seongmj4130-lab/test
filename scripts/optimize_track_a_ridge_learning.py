"""
[Phase 1.3] Ridge 학습 모델 (ML 랭킹) - 병렬 독립 실행

목적:
- 개별 피처 가중치를 Ridge 학습으로 최적화
- 모든 피처를 개별적으로 학습
- Grid Search 모델과 완전히 독립적으로 실행 가능

핵심 개념:
- Grid Search 모델과 완전히 독립적으로 실행 가능
- Raw 데이터(L0~L4) 준비 후 즉시 실행 가능
- Grid Search 모델의 실행 여부와 무관하게 독립 실행

평가 지표:
- Hit Ratio 40% + IC Mean 30% + ICIR 30% (단기 랭킹)
- IC Mean 50% + ICIR 30% + Hit Ratio 20% (장기 랭킹)

사용법:
    python scripts/optimize_track_a_ridge_learning.py --horizon short
    python scripts/optimize_track_a_ridge_learning.py --horizon long
"""
from __future__ import annotations

import argparse
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.components.ranking.score_engine import _pick_feature_cols
from src.utils.config import load_config
from src.utils.io import load_artifact


# [Phase 3] 평가 지표 계산 함수 임포트
# 임시로 여기에 구현 (ranking_metrics.py가 비어있을 경우 대비)
def calculate_hit_ratio(
    scores: pd.Series, returns: pd.Series, top_k: int = 20
) -> float:
    """Hit Ratio: 상위 top_k개 종목의 승률"""
    if len(scores) == 0 or len(returns) == 0:
        return np.nan

    # 상위 top_k개 선택
    top_k_idx = scores.nlargest(top_k).index
    top_k_returns = returns.loc[top_k_idx]

    # 수익률 > 0인 비율
    hit_ratio = (top_k_returns > 0).mean()
    return float(hit_ratio) if not np.isnan(hit_ratio) else np.nan


def calculate_ic(scores: pd.Series, returns: pd.Series) -> float:
    """IC (Information Coefficient): Pearson 상관계수"""
    if len(scores) == 0 or len(returns) == 0:
        return np.nan

    # 결측치 제거
    valid_idx = scores.notna() & returns.notna()
    if valid_idx.sum() < 2:
        return np.nan

    s = pd.to_numeric(scores[valid_idx], errors="coerce")
    r = pd.to_numeric(returns[valid_idx], errors="coerce")

    # 추가 유효성 체크
    final_valid = s.notna() & r.notna()
    if final_valid.sum() < 2:
        return np.nan

    s = s[final_valid]
    r = r[final_valid]

    # 분산이 0인 경우 NaN 반환 (상관계수 계산 불가)
    if s.std() == 0 or r.std() == 0:
        return np.nan

    # 상관계수 계산
    corr = s.corr(r)
    return float(corr) if not np.isnan(corr) else np.nan


def calculate_rank_ic(scores: pd.Series, returns: pd.Series) -> float:
    """Rank IC: Spearman 상관계수"""
    if len(scores) == 0 or len(returns) == 0:
        return np.nan

    # 결측치 제거
    valid_idx = scores.notna() & returns.notna()
    if valid_idx.sum() < 2:
        return np.nan

    s = pd.to_numeric(scores[valid_idx], errors="coerce")
    r = pd.to_numeric(returns[valid_idx], errors="coerce")

    # 추가 유효성 체크
    final_valid = s.notna() & r.notna()
    if final_valid.sum() < 2:
        return np.nan

    s = s[final_valid]
    r = r[final_valid]

    # Rank 계산
    s_rank = s.rank(method="average")
    r_rank = r.rank(method="average")

    # 분산이 0인 경우 NaN 반환
    if s_rank.std() == 0 or r_rank.std() == 0:
        return np.nan

    # 상관계수 계산
    corr = s_rank.corr(r_rank)
    return float(corr) if not np.isnan(corr) else np.nan


def calculate_icir(ic_series: pd.Series) -> float:
    """ICIR: IC의 안정성 (mean / std)"""
    if len(ic_series) == 0:
        return np.nan

    ic_valid = ic_series.dropna()
    if len(ic_valid) == 0:
        return np.nan

    ic_mean = ic_valid.mean()
    ic_std = ic_valid.std()

    if ic_std == 0 or np.isnan(ic_std) or np.isnan(ic_mean):
        return np.nan

    icir = ic_mean / ic_std
    return float(icir) if not np.isnan(icir) else np.nan


def calculate_objective_score(
    hit_ratio: float, ic_mean: float, icir: float, horizon: str = "short"
) -> float:
    """목적함수: Hit Ratio + IC + ICIR 조합"""
    if horizon == "short":
        # 단기: Hit Ratio 중심
        weights = {"hit_ratio": 0.4, "ic_mean": 0.3, "icir": 0.3}
    else:
        # 장기: IC 중심
        weights = {"hit_ratio": 0.2, "ic_mean": 0.5, "icir": 0.3}

    # 정규화 (0~1 범위로 가정)
    hit_ratio_norm = max(0, min(1, hit_ratio)) if not np.isnan(hit_ratio) else 0
    ic_mean_norm = max(-1, min(1, ic_mean)) if not np.isnan(ic_mean) else 0
    icir_norm = (
        max(-5, min(5, icir)) / 5 if not np.isnan(icir) else 0
    )  # -5~5 범위를 -1~1로 정규화

    score = (
        weights["hit_ratio"] * hit_ratio_norm
        + weights["ic_mean"] * (ic_mean_norm + 1) / 2
        + weights["icir"]  # -1~1을 0~1로 변환
        * (icir_norm + 1)
        / 2  # -1~1을 0~1로 변환
    )

    return float(score) if not np.isnan(score) else 0.0


def train_ridge_for_features(
    features: pd.DataFrame,
    target: pd.Series,
    alpha: float = 1.0,
    feature_names: Optional[list[str]] = None,
) -> dict[str, float]:
    """
    Ridge 회귀로 개별 피처 가중치 학습

    Args:
        features: 정규화된 피처 데이터 (각 행이 샘플, 각 열이 피처)
        target: 타겟 변수 (Forward Returns)
        alpha: Ridge 정규화 파라미터
        feature_names: 피처 이름 리스트

    Returns:
        피처별 가중치 딕셔너리
    """
    if feature_names is None:
        feature_names = list(features.columns)

    # 결측치 제거
    valid_idx = target.notna() & features.notna().all(axis=1)
    X = features[valid_idx].values
    y = target[valid_idx].values

    if len(X) == 0 or len(y) == 0:
        return {feat: 0.0 for feat in feature_names}

    # 추가 NaN 체크 및 제거
    nan_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[nan_mask]
    y = y[nan_mask]

    if len(X) == 0 or len(y) == 0:
        return {feat: 0.0 for feat in feature_names}

    # Ridge 학습
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(X, y)

    # 가중치 추출
    weights = ridge.coef_

    # 딕셔너리로 변환
    feature_weights = {}
    for i, feat in enumerate(feature_names):
        if i < len(weights):
            feature_weights[feat] = float(weights[i])
        else:
            feature_weights[feat] = 0.0

    # 절댓값 합 정규화 (음수 가중치 지원)
    abs_sum = sum(abs(w) for w in feature_weights.values())
    if abs_sum > 1e-8:
        feature_weights = {k: v / abs_sum for k, v in feature_weights.items()}
    else:
        # 가중치가 모두 0이면 균등 가중치 적용
        n_features = len(feature_names)
        if n_features > 0:
            feature_weights = {feat: 1.0 / n_features for feat in feature_names}

    return feature_weights


def evaluate_feature_weights(
    panel_data: pd.DataFrame,
    feature_weights: dict[str, float],
    horizon: str = "short",
    cv_folds: Optional[pd.DataFrame] = None,
) -> dict[str, float]:
    """
    피처 가중치 평가 (Hit Ratio, IC, ICIR)

    Args:
        panel_data: 패널 데이터 (date, ticker, 피처들, ret_fwd_20d/120d 포함)
        feature_weights: 피처별 가중치
        horizon: 'short' 또는 'long'
        cv_folds: CV folds 데이터 (Dev 구간 필터링용)

    Returns:
        평가 지표 딕셔너리
    """
    # Forward Returns 컬럼 선택
    target_col = "ret_fwd_20d" if horizon == "short" else "ret_fwd_120d"

    if target_col not in panel_data.columns:
        raise ValueError(f"Target column not found: {target_col}")

    # Dev 구간 필터링
    if cv_folds is not None:
        dev_dates = cv_folds[cv_folds["fold_id"] != "holdout"]["test_end"].unique()
        panel_data = panel_data[panel_data["date"].isin(dev_dates)]

    if len(panel_data) == 0:
        return {
            "hit_ratio": np.nan,
            "ic_mean": np.nan,
            "icir": np.nan,
            "objective_score": 0.0,
        }

    # 피처 컬럼 선택
    feature_cols = list(feature_weights.keys())
    available_features = [f for f in feature_cols if f in panel_data.columns]

    if len(available_features) == 0:
        return {
            "hit_ratio": np.nan,
            "ic_mean": np.nan,
            "icir": np.nan,
            "objective_score": 0.0,
        }

    # 날짜별 평가
    results = []
    dates_processed = 0
    dates_skipped_insufficient = 0
    dates_skipped_no_target = 0
    dates_skipped_insufficient_valid = 0
    dates_no_result = 0

    for date, group in panel_data.groupby("date"):
        dates_processed += 1

        # 최소 종목 수 확인
        if len(group) < 20:  # 최소 20개 종목 필요
            dates_skipped_insufficient += 1
            continue

        # Forward Returns 확인
        if target_col not in group.columns:
            dates_skipped_no_target += 1
            continue

        # 랭킹 점수 계산
        group_features = group[available_features].copy()

        # 정규화된 피처 데이터 생성 (날짜별 cross-sectional 정규화)
        # 학습 시와 동일한 방식으로 정규화 (percentile rank)
        normalized_features = {}
        for feat in available_features:
            if feat not in group_features.columns:
                continue
            if feat not in feature_weights:
                continue
            values = group_features[feat].values
            valid_values = values[~np.isnan(values)]
            if len(valid_values) < 2:
                # 유효 값이 부족하면 0.5로 채움 (중간값) - 학습 시와 동일
                normalized_features[feat] = pd.Series(
                    np.full(len(values), 0.5), index=group.index
                )
            else:
                # Percentile rank - 학습 시와 동일한 방식
                # method='first'는 동일 값도 순서대로 순위 부여 (변동성 보장)
                ranks = pd.Series(values).rank(
                    pct=True, method="first", na_option="keep"
                )
                ranks = ranks.fillna(0.5)  # NaN은 0.5로 채움 - 학습 시와 동일
                # 모든 값이 동일한 경우 변동성 부여 - 학습 시와 동일
                if ranks.std() < 1e-10:
                    n = len(ranks)
                    normalized_features[feat] = pd.Series(
                        np.linspace(0.0, 1.0, n), index=group.index
                    )
                else:
                    normalized_features[feat] = ranks

        # 가중치 합산
        scores = pd.Series(0.0, index=group.index)
        weight_sum = 0.0
        for feat in available_features:
            if feat in normalized_features and feat in feature_weights:
                weight = feature_weights[feat]
                if abs(weight) > 1e-10:  # 매우 작은 가중치는 무시
                    feat_normalized = normalized_features[feat]
                    scores += feat_normalized * weight
                    weight_sum += abs(weight)

        # 가중치 합이 0이면 균등 가중치로 재계산
        if abs(weight_sum) < 1e-10:
            # 모든 피처에 균등 가중치 적용
            n_features = len(
                [f for f in available_features if f in normalized_features]
            )
            if n_features > 0:
                scores = pd.Series(0.0, index=group.index)  # 초기화
                for feat in available_features:
                    if feat in normalized_features:
                        scores += normalized_features[feat] / n_features

        # NaN 체크 및 수정
        if scores.isna().any():
            # NaN이 있으면 해당 인덱스의 다른 피처들로 평균값 계산
            nan_idx = scores.isna()
            if nan_idx.sum() > 0:
                # NaN 인덱스에 대해 평균값으로 대체
                scores[nan_idx] = (
                    scores[~nan_idx].mean() if (~nan_idx).sum() > 0 else 0.5
                )

        # 디버깅: 첫 날짜 가중치 합산 확인
        if dates_processed == 1:
            print("        - 가중치 합산 확인:")
            print(f"          - weight_sum (절댓값 합): {weight_sum:.6f}")
            print("          - scores 계산 후 상세:")
            print(
                f"            - scores: [{scores.min():.6f}, {scores.max():.6f}], std={scores.std():.6f}"
            )
            if scores.std() < 1e-10:
                print("            - ⚠️ scores의 표준편차가 0입니다! (모든 값이 동일)")
                print(f"            - 샘플 scores: {scores.head(5).values}")
                print("            - 원인 분석:")
                # 모든 피처의 percentile rank 합이 상수인지 확인
                sum_of_all_features = pd.Series(0.0, index=group.index)
                for feat in available_features:
                    if feat in normalized_features:
                        sum_of_all_features += normalized_features[feat]
                print(
                    f"              - 모든 피처 percentile rank 합: std={sum_of_all_features.std():.6f}, range=[{sum_of_all_features.min():.6f}, {sum_of_all_features.max():.6f}]"
                )
                if sum_of_all_features.std() < 1e-10 or np.isnan(
                    sum_of_all_features.std()
                ):
                    print(
                        "              - ⚠️ 모든 피처의 percentile rank 합이 상수입니다!"
                    )
                    print(
                        f"              - 샘플 종목별 합: {sum_of_all_features.head(5).values}"
                    )
                    # 각 피처의 percentile rank 확인
                    print("              - 개별 피처 percentile rank 확인 (상위 5개):")
                    for feat in list(available_features)[:5]:
                        if feat in normalized_features:
                            feat_vals = normalized_features[feat]
                            print(
                                f"                {feat}: std={feat_vals.std():.6f}, range=[{feat_vals.min():.6f}, {feat_vals.max():.6f}], 샘플={feat_vals.head(5).values}"
                            )
                    # 피처 간 상관관계 확인
                    if len(normalized_features) >= 2:
                        feat_list = list(normalized_features.keys())[:5]
                        feat_df = pd.DataFrame(
                            {feat: normalized_features[feat] for feat in feat_list}
                        )
                        corr_matrix = feat_df.corr()
                        print(
                            f"              - 상위 5개 피처 간 상관계수 평균: {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean():.4f}"
                        )
                # 상위 5개 피처의 가중 합만 계산
                top_5_features = sorted(
                    [
                        (f, abs(feature_weights.get(f, 0)))
                        for f in available_features
                        if f in feature_weights
                    ],
                    key=lambda x: x[1],
                    reverse=True,
                )[:5]
                print("              - 상위 5개 피처의 가중 합:")
                top_5_sum = pd.Series(0.0, index=group.index)
                for feat, _ in top_5_features:
                    if feat in normalized_features:
                        weight = feature_weights.get(feat, 0)
                        weighted = normalized_features[feat] * weight
                        top_5_sum += weighted
                        print(
                            f"                {feat} (weight={weight:.4f}): std={weighted.std():.6f}"
                        )
                print(
                    f"              - 상위 5개 합계: std={top_5_sum.std():.6f}, range=[{top_5_sum.min():.6f}, {top_5_sum.max():.6f}]"
                )

        # Forward Returns
        returns = group[target_col]

        # 공통 유효 인덱스
        valid_idx = scores.notna() & returns.notna()

        # 디버깅: 첫 번째 날짜 상세 정보
        if dates_processed == 1:
            print(f"      [DEBUG 첫 날짜] 날짜: {date}")
            print(f"        - 총 종목 수: {len(group)}")
            print(
                f"        - scores notna: {scores.notna().sum()}, returns notna: {returns.notna().sum()}"
            )
            print(f"        - valid_idx (교집합): {valid_idx.sum()}")
            print(
                f"        - scores 범위: [{scores.min():.6f}, {scores.max():.6f}], std: {scores.std():.6f}"
            )
            print(
                f"        - returns 범위: [{returns.min():.4f}, {returns.max():.4f}], std: {returns.std():.4f}"
            )

        if valid_idx.sum() < 20:  # 최소 20개 필요
            dates_skipped_insufficient_valid += 1
            if dates_processed == 1:
                print(f"        - ⚠️ 유효 종목 수 부족: {valid_idx.sum()} < 20")
            continue

        scores_valid = scores[valid_idx]
        returns_valid = returns[valid_idx]

        # 디버깅: 샘플 날짜 하나만 상세 로그
        if dates_processed == 1:
            print(f"      [DEBUG 샘플] 날짜: {date}, 유효 종목 수: {valid_idx.sum()}")
            print(
                f"        - scores 범위: [{scores_valid.min():.4f}, {scores_valid.max():.4f}], std: {scores_valid.std():.4f}"
            )
            print(
                f"        - returns 범위: [{returns_valid.min():.4f}, {returns_valid.max():.4f}], std: {returns_valid.std():.4f}"
            )
            # 가중치 요약
            non_zero_weights = {
                k: v for k, v in feature_weights.items() if abs(v) > 1e-6
            }
            print(
                f"        - 가중치 사용: {len(non_zero_weights)}개 피처 (전체 {len(feature_weights)}개)"
            )
            if len(non_zero_weights) > 0:
                top_weights = sorted(
                    non_zero_weights.items(), key=lambda x: abs(x[1]), reverse=True
                )[:5]
                print(
                    f"        - 상위 가중치: {', '.join([f'{k}={v:.4f}' for k, v in top_weights])}"
                )

        # 평가 지표 계산
        hit_ratio = calculate_hit_ratio(scores_valid, returns_valid, top_k=20)
        ic = calculate_ic(scores_valid, returns_valid)
        rank_ic = calculate_rank_ic(scores_valid, returns_valid)

        # 디버깅: 샘플 날짜 결과
        if dates_processed == 1 and len(results) == 0:
            print(
                f"        - Hit Ratio: {hit_ratio:.4f}"
                if not np.isnan(hit_ratio)
                else "        - Hit Ratio: NaN"
            )
            print(
                f"        - IC: {ic:.4f}" if not np.isnan(ic) else "        - IC: NaN"
            )
            print(
                f"        - Rank IC: {rank_ic:.4f}"
                if not np.isnan(rank_ic)
                else "        - Rank IC: NaN"
            )

        # IC가 NaN이 아닌 경우만 추가
        if not np.isnan(ic):
            results.append(
                {
                    "date": date,
                    "hit_ratio": hit_ratio if not np.isnan(hit_ratio) else 0.0,
                    "ic": ic,
                    "rank_ic": rank_ic if not np.isnan(rank_ic) else ic,
                }
            )
        else:
            dates_no_result += 1

    if len(results) == 0:
        # 디버깅: 왜 결과가 없는지 확인
        print("    [DEBUG] 평가 결과 없음:")
        print(f"      - 처리한 날짜 수: {dates_processed}")
        print(f"      - 스킵된 날짜 수 (종목 수 < 20): {dates_skipped_insufficient}")
        print(f"      - 스킵된 날짜 수 (target_col 없음): {dates_skipped_no_target}")
        print(
            f"      - 스킵된 날짜 수 (유효 종목 < 20): {dates_skipped_insufficient_valid}"
        )
        print(f"      - IC가 NaN인 날짜 수: {dates_no_result}")
        print(
            f"      - panel_data 총 행: {len(panel_data):,}, 날짜: {len(panel_data['date'].unique())}개"
        )
        return {
            "hit_ratio": np.nan,
            "ic_mean": np.nan,
            "icir": np.nan,
            "objective_score": 0.0,
        }

    results_df = pd.DataFrame(results)
    print(f"    [DEBUG] 평가 완료: {len(results_df)}개 날짜에서 평가 지표 계산 성공")

    # 집계
    hit_ratio_mean = results_df["hit_ratio"].mean()
    ic_mean = results_df["ic"].mean()
    icir = calculate_icir(results_df["ic"])

    # 목적함수
    objective_score = calculate_objective_score(hit_ratio_mean, ic_mean, icir, horizon)

    return {
        "hit_ratio": hit_ratio_mean,
        "ic_mean": ic_mean,
        "icir": icir,
        "objective_score": objective_score,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Ridge 학습으로 개별 피처 가중치 최적화"
    )
    parser.add_argument(
        "--horizon",
        type=str,
        choices=["short", "long"],
        default="short",
        help="단기 또는 장기 랭킹 (default: short)",
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="Ridge 정규화 파라미터 (default: 1.0)"
    )
    parser.add_argument(
        "--grid-search-alpha",
        action="store_true",
        help="Alpha 값 그리드 서치 (기본값: False)",
    )
    parser.add_argument(
        "--alpha-range",
        type=str,
        default="0.1,1.0,10.0",
        help="Alpha 그리드 범위 (쉼표 구분, default: 0.1,1.0,10.0)",
    )
    parser.add_argument(
        "--grid-search-config",
        type=str,
        default=None,
        help="Grid Search 결과 파일 경로 (초기 가중치 사용)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print(f"[Phase 3] Ridge 학습으로 개별 피처 가중치 최적화 ({args.horizon} 랭킹)")
    print("=" * 80)

    # Config 로드
    cfg = load_config("configs/config.yaml")
    base_dir = Path(cfg["paths"]["base_dir"])
    interim_dir = base_dir / "data" / "interim"

    # 데이터 로드
    print("\n[1/5] 데이터 로드 중...")
    panel_data = load_artifact(interim_dir / "panel_merged_daily")
    dataset_daily = load_artifact(interim_dir / "dataset_daily")
    print(f"  - 패널 데이터: {len(panel_data):,} 행")
    print(f"  - 데이터셋 일일: {len(dataset_daily):,} 행")

    # Forward Returns 병합 (panel_merged_daily에 ret_fwd_20d/120d 추가)
    forward_return_cols = ["ret_fwd_20d", "ret_fwd_120d"]
    merge_cols = ["date", "ticker"]
    if all(col in dataset_daily.columns for col in forward_return_cols + merge_cols):
        # 병합 전 상태 확인
        print(
            f"  - 병합 전: panel_data {len(panel_data):,}행, dataset_daily {len(dataset_daily):,}행"
        )
        panel_data = panel_data.merge(
            dataset_daily[merge_cols + forward_return_cols],
            on=merge_cols,
            how="left",
            suffixes=("", "_from_dataset"),
        )
        # 중복 컬럼 처리
        for col in forward_return_cols:
            if f"{col}_from_dataset" in panel_data.columns:
                panel_data[col] = panel_data[f"{col}_from_dataset"].fillna(
                    panel_data.get(col, np.nan)
                )
                panel_data = panel_data.drop(columns=[f"{col}_from_dataset"])

        print("  - Forward Returns 병합 완료")
        print(
            f"  - 병합 후: panel_data {len(panel_data):,}행, ret_fwd_20d 존재: {'ret_fwd_20d' in panel_data.columns}, non-null: {panel_data['ret_fwd_20d'].notna().sum() if 'ret_fwd_20d' in panel_data.columns else 0}"
        )

    # CV folds 로드
    cv_folds_file = f"cv_folds_{args.horizon}.parquet"
    cv_folds = load_artifact(interim_dir / cv_folds_file)
    print(f"  - CV folds: {len(cv_folds)} folds")

    # Forward Returns 컬럼 선택
    target_col = "ret_fwd_20d" if args.horizon == "short" else "ret_fwd_120d"
    print(f"  - 타겟 변수: {target_col}")

    # Dev 구간 필터링
    dev_folds = cv_folds[cv_folds["fold_id"] != "holdout"]
    dev_dates = dev_folds["test_end"].unique()
    panel_dev = panel_data[panel_data["date"].isin(dev_dates)].copy()
    print(f"  - Dev 구간 데이터: {len(panel_dev):,} 행, {len(dev_dates)} 날짜")

    # 피처 컬럼 선택
    print("\n[2/5] 피처 선택 중...")
    feature_cols = _pick_feature_cols(panel_dev)
    print(f"  - 사용 가능한 피처 수: {len(feature_cols)}")
    print(
        f"  - 피처 목록: {', '.join(feature_cols[:10])}..."
        if len(feature_cols) > 10
        else f"  - 피처 목록: {', '.join(feature_cols)}"
    )

    # Grid Search 결과 로드 (초기 가중치, 선택적)
    initial_group_weights = None
    if args.grid_search_config:
        print(f"\n[초기 가중치] Grid Search 결과 로드: {args.grid_search_config}")
        with open(args.grid_search_config, encoding="utf-8") as f:
            grid_config = yaml.safe_load(f)
        initial_group_weights = grid_config.get("feature_groups", {})
        print(f"  - 그룹 수: {len(initial_group_weights)}")

    # Alpha 그리드 서치
    if args.grid_search_alpha:
        alpha_values = [float(x.strip()) for x in args.alpha_range.split(",")]
        print(f"\n[3/5] Alpha 그리드 서치: {alpha_values}")
    else:
        alpha_values = [args.alpha]
        print(f"\n[3/5] Ridge 학습 (alpha={args.alpha})")

    # 날짜별 데이터 준비
    print("\n[4/5] 데이터 준비 중...")
    all_features_normalized = []
    all_targets = []
    all_dates = []

    for date, group in panel_dev.groupby("date"):
        if len(group) < 20:  # 최소 20개 종목 필요
            continue

        # 피처 정규화 (날짜별 cross-sectional percentile rank)
        # 평가 시와 동일한 방식으로 정규화
        group_features = group[feature_cols].copy()
        normalized_group = pd.DataFrame(index=group.index)

        for feat in feature_cols:
            values = group_features[feat].values
            valid_values = values[~np.isnan(values)]
            if len(valid_values) < 2:
                # 유효 값이 부족하면 0.5로 채움 (중간값) - 평가 시와 동일
                normalized_group[feat] = 0.5
            else:
                # Percentile rank - 평가 시와 동일한 방식
                ranks = pd.Series(values).rank(
                    pct=True, method="first", na_option="keep"
                )
                ranks = ranks.fillna(0.5)  # NaN은 0.5로 채움 - 평가 시와 동일
                # 모든 값이 동일한 경우 변동성 부여 - 평가 시와 동일
                if ranks.std() < 1e-10:
                    n = len(ranks)
                    normalized_group[feat] = np.linspace(0.0, 1.0, n)
                else:
                    normalized_group[feat] = ranks.values

        # 타겟 변수
        targets = group[target_col].values

        # 결측치 제거
        valid_idx = ~np.isnan(targets)
        if valid_idx.sum() < 10:  # 최소 10개 샘플 필요
            continue

        all_features_normalized.append(normalized_group[valid_idx])
        all_targets.append(
            pd.Series(targets[valid_idx], index=normalized_group.index[valid_idx])
        )
        all_dates.append(date)

    if len(all_features_normalized) == 0:
        raise ValueError("No valid data for training")

    # 전체 데이터 결합
    features_concat = pd.concat(all_features_normalized, axis=0)
    targets_concat = pd.concat(all_targets, axis=0)
    print(f"  - 학습 데이터: {len(features_concat):,} 행, {len(feature_cols)} 피처")

    # Alpha별 최적화
    best_result = None
    best_alpha = None
    best_weights = None

    results_summary = []

    for alpha in alpha_values:
        print(f"\n  [Alpha={alpha}] Ridge 학습 중...")

        # Ridge 학습
        feature_weights = train_ridge_for_features(
            features_concat, targets_concat, alpha=alpha, feature_names=feature_cols
        )

        print(f"    - 학습된 피처 가중치: {len(feature_weights)}개")
        non_zero_weights = {k: v for k, v in feature_weights.items() if abs(v) > 1e-6}
        print(f"    - 0이 아닌 가중치: {len(non_zero_weights)}개")

        # 평가
        print("    - 평가 중...")
        metrics = evaluate_feature_weights(
            panel_dev, feature_weights, horizon=args.horizon, cv_folds=cv_folds
        )

        print(f"    - Objective Score: {metrics['objective_score']:.4f}")
        print(
            f"    - Hit Ratio: {metrics['hit_ratio']:.4f}"
            if not np.isnan(metrics["hit_ratio"])
            else "    - Hit Ratio: NaN"
        )
        print(
            f"    - IC Mean: {metrics['ic_mean']:.4f}"
            if not np.isnan(metrics["ic_mean"])
            else "    - IC Mean: NaN"
        )
        print(
            f"    - ICIR: {metrics['icir']:.4f}"
            if not np.isnan(metrics["icir"])
            else "    - ICIR: NaN"
        )

        results_summary.append(
            {"alpha": alpha, **metrics, "non_zero_features": len(non_zero_weights)}
        )

        # 최적 결과 업데이트
        if (
            best_result is None
            or metrics["objective_score"] > best_result["objective_score"]
        ):
            best_result = metrics
            best_alpha = alpha
            best_weights = feature_weights.copy()

    # 결과 저장
    print("\n[5/5] 결과 저장 중...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 최적 가중치 YAML 저장
    output_file = (
        base_dir / "configs" / f"feature_weights_{args.horizon}_ridge_{timestamp}.yaml"
    )
    with open(output_file, "w", encoding="utf-8") as f:
        yaml.dump(
            {
                "description": f'[Phase 3 Ridge 학습] 개별 피처 가중치 (Alpha={best_alpha}, Objective Score={best_result["objective_score"]:.4f})',
                "horizon": args.horizon,
                "alpha": best_alpha,
                "metadata": {
                    "objective_score": best_result["objective_score"],
                    "hit_ratio": (
                        float(best_result["hit_ratio"])
                        if not np.isnan(best_result["hit_ratio"])
                        else None
                    ),
                    "ic_mean": (
                        float(best_result["ic_mean"])
                        if not np.isnan(best_result["ic_mean"])
                        else None
                    ),
                    "icir": (
                        float(best_result["icir"])
                        if not np.isnan(best_result["icir"])
                        else None
                    ),
                    "optimization_date": timestamp,
                    "feature_count": len(best_weights),
                    "non_zero_feature_count": len(
                        {k: v for k, v in best_weights.items() if abs(v) > 1e-6}
                    ),
                },
                "feature_weights": best_weights,
            },
            f,
            allow_unicode=True,
            default_flow_style=False,
            sort_keys=False,
        )

    print(f"  - 최적 가중치 파일: {output_file}")

    # 결과 요약 CSV 저장
    results_df = pd.DataFrame(results_summary)
    results_csv = (
        base_dir
        / "artifacts"
        / "reports"
        / f"track_a_ridge_learning_{args.horizon}_{timestamp}.csv"
    )
    results_df.to_csv(results_csv, index=False)
    print(f"  - 결과 요약: {results_csv}")

    # 최종 결과 출력
    print("\n" + "=" * 80)
    print(f"[최적 결과] Alpha={best_alpha}")
    print("=" * 80)
    print(f"Objective Score: {best_result['objective_score']:.4f}")
    print(
        f"Hit Ratio: {best_result['hit_ratio']:.4f}"
        if not np.isnan(best_result["hit_ratio"])
        else "Hit Ratio: NaN"
    )
    print(
        f"IC Mean: {best_result['ic_mean']:.4f}"
        if not np.isnan(best_result["ic_mean"])
        else "IC Mean: NaN"
    )
    print(
        f"ICIR: {best_result['icir']:.4f}"
        if not np.isnan(best_result["icir"])
        else "ICIR: NaN"
    )
    print(f"\n최적 가중치 파일: {output_file}")


if __name__ == "__main__":
    main()
