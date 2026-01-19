"""
[Phase 1.4] XGBoost 학습 모델 (ML 랭킹) - 병렬 독립 실행

목적:
- XGBoost 모델로 개별 피처 가중치 학습
- 모든 피처를 개별적으로 학습
- Grid Search 모델, Ridge 모델과 완전히 독립적으로 실행 가능

핵심 개념:
- Grid Search 모델, Ridge 모델과 완전히 독립적으로 실행 가능
- Raw 데이터(L0~L4) 준비 후 즉시 실행 가능
- 다른 모델의 실행 여부와 무관하게 독립 실행

평가 지표:
- Hit Ratio 40% + IC Mean 30% + ICIR 30% (단기 랭킹)
- IC Mean 50% + ICIR 30% + Hit Ratio 20% (장기 랭킹)

사용법:
    python scripts/optimize_track_a_xgboost_learning.py --horizon short
    python scripts/optimize_track_a_xgboost_learning.py --horizon long
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

try:
    import xgboost as xgb
except ImportError:
    raise ImportError(
        "XGBoost가 설치되지 않았습니다. 'pip install xgboost'로 설치하세요."
    )

from sklearn.model_selection import ParameterGrid

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.components.ranking.score_engine import _pick_feature_cols
from src.utils.config import load_config
from src.utils.io import load_artifact


# [Phase 1.4] 평가 지표 계산 함수 (Ridge 스크립트와 동일)
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


def train_xgboost_for_features(
    features: pd.DataFrame,
    target: pd.Series,
    feature_names: Optional[list[str]] = None,
    n_estimators: int = 100,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    random_state: int = 42,
) -> tuple[xgb.XGBRegressor, dict[str, float]]:
    """
    XGBoost 회귀로 개별 피처 가중치 학습

    Args:
        features: 정규화된 피처 데이터 (각 행이 샘플, 각 열이 피처)
        target: 타겟 변수 (Forward Returns)
        feature_names: 피처 이름 리스트
        n_estimators: 트리 개수
        max_depth: 최대 깊이
        learning_rate: 학습률
        subsample: 샘플링 비율
        colsample_bytree: 피처 샘플링 비율
        reg_alpha: L1 정규화
        reg_lambda: L2 정규화
        random_state: 랜덤 시드

    Returns:
        (학습된 모델, 피처별 가중치 딕셔너리)
    """
    if feature_names is None:
        feature_names = list(features.columns)

    # 결측치 제거
    valid_idx = target.notna() & features.notna().all(axis=1)
    X = features[valid_idx].values
    y = target[valid_idx].values

    if len(X) == 0 or len(y) == 0:
        return None, {feat: 0.0 for feat in feature_names}

    # 추가 NaN 체크 및 제거
    nan_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[nan_mask]
    y = y[nan_mask]

    if len(X) == 0 or len(y) == 0:
        return None, {feat: 0.0 for feat in feature_names}

    # XGBoost 학습
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X, y)

    # 피처 중요도 추출 (gain 사용)
    feature_importance = model.feature_importances_

    # 딕셔너리로 변환
    feature_weights = {}
    for i, feat in enumerate(feature_names):
        if i < len(feature_importance):
            feature_weights[feat] = float(feature_importance[i])
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

    return model, feature_weights


def evaluate_xgboost_model(
    panel_data: pd.DataFrame,
    model: xgb.XGBRegressor,
    feature_cols: list[str],
    horizon: str = "short",
    cv_folds: Optional[pd.DataFrame] = None,
) -> dict[str, float]:
    """
    XGBoost 모델 평가 (Hit Ratio, IC, ICIR)

    Args:
        panel_data: 패널 데이터 (date, ticker, 피처들, ret_fwd_20d/120d 포함)
        model: 학습된 XGBoost 모델
        feature_cols: 피처 컬럼 리스트
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
            dev_dates = cv_folds[~cv_folds["fold_id"].str.startswith("holdout")][
                "test_end"
            ].unique()
            panel_data = panel_data[panel_data["date"].isin(dev_dates)]

    if len(panel_data) == 0:
        return {
            "hit_ratio": np.nan,
            "ic_mean": np.nan,
            "icir": np.nan,
            "objective_score": 0.0,
        }

    # 사용 가능한 피처 확인
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
    dates_skipped = 0

    for date, group in panel_data.groupby("date"):
        dates_processed += 1

        # 최소 종목 수 확인
        if len(group) < 20:  # 최소 20개 종목 필요
            if dates_processed <= 3:
                print(
                    f"      [DEBUG] 날짜 {date} 스킵: 종목 수 부족 ({len(group)} < 20)"
                )
            dates_skipped += 1
            continue

        # Forward Returns 확인
        if target_col not in group.columns:
            if dates_processed <= 3:
                print(f"      [DEBUG] 날짜 {date} 스킵: {target_col} 컬럼 없음")
            dates_skipped += 1
            continue

        # 피처 정규화 (날짜별 cross-sectional percentile rank)
        group_features = group[available_features].copy()
        normalized_features = pd.DataFrame(index=group.index)

        for feat in available_features:
            if feat not in group_features.columns:
                # 피처가 없으면 0.5로 채움
                normalized_features[feat] = pd.Series(
                    np.full(len(group.index), 0.5), index=group.index
                )
                continue
            values = group_features[feat].values
            valid_values = values[~np.isnan(values)]
            if len(valid_values) < 2:
                # 유효 값이 부족하면 0.5로 채움 (중간값)
                normalized_features[feat] = pd.Series(
                    np.full(len(values), 0.5), index=group.index
                )
            else:
                # Percentile rank (인덱스 보존)
                ranks = pd.Series(values, index=group.index).rank(
                    pct=True, method="first", na_option="keep"
                )
                ranks = ranks.fillna(0.5)  # NaN은 0.5로 채움
                # 모든 값이 동일한 경우 변동성 부여
                if ranks.std() < 1e-10:
                    n = len(ranks)
                    normalized_features[feat] = pd.Series(
                        np.linspace(0.0, 1.0, n), index=group.index
                    )
                else:
                    normalized_features[feat] = ranks

        # 모든 피처가 normalized_features에 있는지 확인
        missing_features = [
            f for f in available_features if f not in normalized_features.columns
        ]
        if missing_features:
            if dates_processed <= 3:
                print(
                    f"      [DEBUG] 날짜 {date} 경고: 누락된 피처 {len(missing_features)}개"
                )
            # 누락된 피처는 0.5로 채움
            for feat in missing_features:
                normalized_features[feat] = pd.Series(
                    np.full(len(group.index), 0.5), index=group.index
                )

        # 모델 예측 (피처 순서 보장)
        # available_features 순서대로 정렬
        X_pred = normalized_features[available_features].values

        # 디버깅: 첫 날짜만 상세 로그
        if dates_processed == 1:
            print(f"      [DEBUG 첫 날짜] 날짜: {date}")
            print(f"        - normalized_features shape: {normalized_features.shape}")
            print(f"        - available_features 수: {len(available_features)}")
            print(f"        - X_pred shape: {X_pred.shape}")
            print(f"        - X_pred NaN 수 (행별): {np.isnan(X_pred).sum(axis=1)[:5]}")
            print(f"        - X_pred NaN 수 (열별): {np.isnan(X_pred).sum(axis=0)[:5]}")
            print(
                f"        - X_pred 전체 NaN 비율: {np.isnan(X_pred).sum() / X_pred.size:.2%}"
            )

        # NaN 체크 및 제거
        nan_mask = ~np.isnan(X_pred).any(axis=1)
        if nan_mask.sum() < 20:  # 최소 20개 필요
            if dates_processed <= 3:
                print(
                    f"      [DEBUG] 날짜 {date} 스킵: 유효 피처 데이터 부족 ({nan_mask.sum()} < 20)"
                )
                print(f"        - X_pred shape: {X_pred.shape}")
                print(f"        - nan_mask.sum(): {nan_mask.sum()}")
                print(
                    f"        - NaN이 있는 행 수: {(np.isnan(X_pred).any(axis=1)).sum()}"
                )
            dates_skipped += 1
            continue

        X_pred_valid = X_pred[nan_mask]

        try:
            scores_raw = model.predict(X_pred_valid)
            scores = pd.Series(scores_raw, index=normalized_features.index[nan_mask])
        except Exception as e:
            print(f"      [WARNING] 날짜 {date} 예측 실패: {e}")
            dates_skipped += 1
            continue

        # Forward Returns
        returns = group[target_col]

        # 공통 유효 인덱스
        valid_idx = scores.notna() & returns.notna()

        if valid_idx.sum() < 20:  # 최소 20개 필요
            if dates_processed <= 3:
                print(
                    f"      [DEBUG] 날짜 {date} 스킵: 유효 종목 수 부족 ({valid_idx.sum()} < 20)"
                )
                print(
                    f"        - scores.notna(): {scores.notna().sum()}, returns.notna(): {returns.notna().sum()}"
                )
            dates_skipped += 1
            continue

        scores_valid = scores[valid_idx]
        returns_valid = returns[valid_idx]

        # 디버깅: 첫 날짜만 상세 로그
        if dates_processed == 1:
            print(f"      [DEBUG 첫 날짜] 날짜: {date}")
            print(f"        - 총 종목 수: {len(group)}")
            print(f"        - 예측 후 유효 종목 수: {valid_idx.sum()}")
            print(
                f"        - scores 범위: [{scores_valid.min():.6f}, {scores_valid.max():.6f}], std: {scores_valid.std():.6f}"
            )
            print(
                f"        - returns 범위: [{returns_valid.min():.4f}, {returns_valid.max():.4f}], std: {returns_valid.std():.4f}"
            )

        # 평가 지표 계산
        hit_ratio = calculate_hit_ratio(scores_valid, returns_valid, top_k=20)
        ic = calculate_ic(scores_valid, returns_valid)
        rank_ic = calculate_rank_ic(scores_valid, returns_valid)

        # 디버깅: 첫 날짜만 상세 로그
        if dates_processed == 1:
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
            dates_skipped += 1
            # 디버깅: IC가 NaN인 이유 확인
            if dates_processed <= 3:  # 처음 3개 날짜만
                print(f"      [DEBUG] 날짜 {date} IC가 NaN인 이유:")
                print(f"        - scores_valid.std(): {scores_valid.std():.6f}")
                print(f"        - returns_valid.std(): {returns_valid.std():.6f}")
                print(f"        - scores_valid 유니크 값 수: {scores_valid.nunique()}")
                print(
                    f"        - returns_valid 유니크 값 수: {returns_valid.nunique()}"
                )

    print(
        f"    - 평가 완료: {len(results)}개 날짜에서 평가 지표 계산 성공 (처리: {dates_processed}, 스킵: {dates_skipped})"
    )

    if len(results) == 0:
        return {
            "hit_ratio": np.nan,
            "ic_mean": np.nan,
            "icir": np.nan,
            "objective_score": 0.0,
        }

    results_df = pd.DataFrame(results)

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
        description="XGBoost 학습으로 개별 피처 가중치 최적화"
    )
    parser.add_argument(
        "--horizon",
        type=str,
        choices=["short", "long"],
        default="short",
        help="단기 또는 장기 랭킹 (default: short)",
    )
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="하이퍼파라미터 그리드 서치 (정규화 강화 버전, 기본값: False)",
    )
    parser.add_argument(
        "--optimize-by-holdout",
        action="store_true",
        help="Holdout IC Mean 기준으로 최적화 (기본값: False, Dev 기준)",
    )
    parser.add_argument(
        "--n-estimators", type=int, default=100, help="트리 개수 (default: 100)"
    )
    parser.add_argument(
        "--max-depth", type=int, default=6, help="최대 깊이 (default: 6)"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=0.1, help="학습률 (default: 0.1)"
    )
    parser.add_argument(
        "--subsample", type=float, default=0.8, help="샘플링 비율 (default: 0.8)"
    )
    parser.add_argument(
        "--colsample-bytree",
        type=float,
        default=0.8,
        help="피처 샘플링 비율 (default: 0.8)",
    )
    parser.add_argument(
        "--reg-alpha", type=float, default=0.0, help="L1 정규화 (default: 0.0)"
    )
    parser.add_argument(
        "--reg-lambda", type=float, default=1.0, help="L2 정규화 (default: 1.0)"
    )
    parser.add_argument(
        "--random-state", type=int, default=42, help="랜덤 시드 (default: 42)"
    )
    args = parser.parse_args()

    print("=" * 80)
    print(f"[Phase 1.4] XGBoost 학습으로 개별 피처 가중치 최적화 ({args.horizon} 랭킹)")
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

    # Forward Returns 병합
    forward_return_cols = ["ret_fwd_20d", "ret_fwd_120d"]
    merge_cols = ["date", "ticker"]
    if all(col in dataset_daily.columns for col in forward_return_cols + merge_cols):
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

    # CV folds 로드
    cv_folds_file = f"cv_folds_{args.horizon}.parquet"
    cv_folds = load_artifact(interim_dir / cv_folds_file)
    print(f"  - CV folds: {len(cv_folds)} folds")

    # Forward Returns 컬럼 선택
    target_col = "ret_fwd_20d" if args.horizon == "short" else "ret_fwd_120d"
    print(f"  - 타겟 변수: {target_col}")

    # Dev 구간 필터링 (holdout이 아닌 것들)
    dev_folds = cv_folds[~cv_folds["fold_id"].str.startswith("holdout")]
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

    # 하이퍼파라미터 그리드 정의
    if args.grid_search:
        # 정규화 강화를 위한 그리드 (과적합 방지)
        param_grid = {
            "n_estimators": [50, 100],  # 트리 개수 제한
            "max_depth": [3, 4, 5],  # 트리 깊이 제한 (단순화)
            "learning_rate": [0.05, 0.1],  # 보수적인 학습률
            "subsample": [0.8, 0.9],  # 샘플링 비율
            "colsample_bytree": [0.8, 0.9],  # 피처 샘플링
            "reg_alpha": [0.0, 0.5, 1.0],  # L1 정규화 강화
            "reg_lambda": [1.0, 5.0, 10.0],  # L2 정규화 강화
        }
        print(
            f"\n[3/5] 하이퍼파라미터 그리드 서치 (정규화 강화): {len(list(ParameterGrid(param_grid)))}개 조합"
        )
    else:
        param_grid = {
            "n_estimators": [args.n_estimators],
            "max_depth": [args.max_depth],
            "learning_rate": [args.learning_rate],
            "subsample": [args.subsample],
            "colsample_bytree": [args.colsample_bytree],
            "reg_alpha": [args.reg_alpha],
            "reg_lambda": [args.reg_lambda],
        }
        print("\n[3/5] XGBoost 학습 (단일 파라미터)")

    # 날짜별 데이터 준비
    print("\n[4/5] 데이터 준비 중...")
    all_features_normalized = []
    all_targets = []
    all_dates = []

    for date, group in panel_dev.groupby("date"):
        if len(group) < 20:  # 최소 20개 종목 필요
            continue

        # 피처 정규화 (날짜별 cross-sectional percentile rank)
        group_features = group[feature_cols].copy()
        normalized_group = pd.DataFrame(index=group.index)

        for feat in feature_cols:
            values = group_features[feat].values
            valid_values = values[~np.isnan(values)]
            if len(valid_values) < 2:
                # 유효 값이 부족하면 0.5로 채움 (중간값)
                normalized_group[feat] = 0.5
            else:
                # Percentile rank
                ranks = pd.Series(values).rank(
                    pct=True, method="first", na_option="keep"
                )
                ranks = ranks.fillna(0.5)  # NaN은 0.5로 채움
                # 모든 값이 동일한 경우 변동성 부여
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

    # 하이퍼파라미터별 최적화
    best_result = None
    best_params = None
    best_model = None
    best_weights = None

    results_summary = []

    for params in ParameterGrid(param_grid):
        print(f"\n  [파라미터={params}] XGBoost 학습 중...")

        # XGBoost 학습
        model, feature_weights = train_xgboost_for_features(
            features_concat,
            targets_concat,
            feature_names=feature_cols,
            **params,
            random_state=args.random_state,
        )

        if model is None:
            print("    - ⚠️ 모델 학습 실패 (데이터 부족)")
            continue

        print(f"    - 학습된 피처 가중치: {len(feature_weights)}개")
        non_zero_weights = {k: v for k, v in feature_weights.items() if abs(v) > 1e-6}
        print(f"    - 0이 아닌 가중치: {len(non_zero_weights)}개")

        # 평가 (Dev 구간)
        print("    - Dev 구간 평가 중...")
        metrics_dev = evaluate_xgboost_model(
            panel_dev, model, feature_cols, horizon=args.horizon, cv_folds=cv_folds
        )

        print(f"      - Dev Objective Score: {metrics_dev['objective_score']:.4f}")
        print(
            f"      - Dev Hit Ratio: {metrics_dev['hit_ratio']:.4f}"
            if not np.isnan(metrics_dev["hit_ratio"])
            else "      - Dev Hit Ratio: NaN"
        )
        print(
            f"      - Dev IC Mean: {metrics_dev['ic_mean']:.4f}"
            if not np.isnan(metrics_dev["ic_mean"])
            else "      - Dev IC Mean: NaN"
        )
        print(
            f"      - Dev ICIR: {metrics_dev['icir']:.4f}"
            if not np.isnan(metrics_dev["icir"])
            else "      - Dev ICIR: NaN"
        )

        # Holdout 구간 평가
        holdout_folds = cv_folds[cv_folds["fold_id"].str.startswith("holdout")]
        if len(holdout_folds) > 0:
            holdout_dates = holdout_folds["test_end"].unique()
            panel_holdout = panel_data[panel_data["date"].isin(holdout_dates)].copy()
            print(f"    - Holdout 구간 평가 중... (날짜: {len(holdout_dates)}개)")
            metrics_holdout = evaluate_xgboost_model(
                panel_holdout,
                model,
                feature_cols,
                horizon=args.horizon,
                cv_folds=None,  # Holdout은 필터링 불필요
            )

            print(
                f"      - Holdout Objective Score: {metrics_holdout['objective_score']:.4f}"
            )
            print(
                f"      - Holdout Hit Ratio: {metrics_holdout['hit_ratio']:.4f}"
                if not np.isnan(metrics_holdout["hit_ratio"])
                else "      - Holdout Hit Ratio: NaN"
            )
            print(
                f"      - Holdout IC Mean: {metrics_holdout['ic_mean']:.4f}"
                if not np.isnan(metrics_holdout["ic_mean"])
                else "      - Holdout IC Mean: NaN"
            )
            print(
                f"      - Holdout ICIR: {metrics_holdout['icir']:.4f}"
                if not np.isnan(metrics_holdout["icir"])
                else "      - Holdout ICIR: NaN"
            )
        else:
            metrics_holdout = {
                "hit_ratio": np.nan,
                "ic_mean": np.nan,
                "icir": np.nan,
                "objective_score": 0.0,
            }
            print("    - Holdout 구간 없음 (스킵)")

        # 결과 요약 (Dev 기준으로 최적화, Holdout은 참고용)
        results_summary.append(
            {
                **params,
                "dev_objective_score": metrics_dev["objective_score"],
                "dev_hit_ratio": metrics_dev["hit_ratio"],
                "dev_ic_mean": metrics_dev["ic_mean"],
                "dev_icir": metrics_dev["icir"],
                "holdout_objective_score": metrics_holdout["objective_score"],
                "holdout_hit_ratio": metrics_holdout["hit_ratio"],
                "holdout_ic_mean": metrics_holdout["ic_mean"],
                "holdout_icir": metrics_holdout["icir"],
                "non_zero_features": len(non_zero_weights),
            }
        )

        # 최적 결과 업데이트
        # --optimize-by-holdout 옵션이 있으면 Holdout IC Mean 기준, 없으면 Dev Objective Score 기준
        if "best_result_holdout" not in locals():
            best_result_holdout = metrics_holdout

        if args.optimize_by_holdout:
            # Holdout IC Mean 기준으로 최적화 (과적합 방지)
            if not np.isnan(metrics_holdout.get("ic_mean", np.nan)):
                if best_result is None or (
                    np.isnan(best_result_holdout.get("ic_mean", np.nan))
                    or metrics_holdout["ic_mean"]
                    > best_result_holdout.get("ic_mean", -np.inf)
                ):
                    best_result = metrics_dev
                    best_result_holdout = metrics_holdout
                    best_params = params
                    best_model = model
                    best_weights = feature_weights.copy()
            else:
                # Holdout이 유효하지 않은 경우 Dev 기준
                if (
                    best_result is None
                    or metrics_dev["objective_score"] > best_result["objective_score"]
                ):
                    best_result = metrics_dev
                    best_result_holdout = metrics_holdout
                    best_params = params
                    best_model = model
                    best_weights = feature_weights.copy()
        else:
            # Dev Objective Score 기준으로 최적화 (기본)
            if (
                best_result is None
                or metrics_dev["objective_score"] > best_result["objective_score"]
            ):
                best_result = metrics_dev
                best_result_holdout = metrics_holdout
                best_params = params
                best_model = model
                best_weights = feature_weights.copy()

    if best_model is None:
        raise ValueError("모든 파라미터 조합에서 모델 학습 실패")

    # 결과 저장
    print("\n[5/5] 결과 저장 중...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 최적 가중치 YAML 저장
    output_file = (
        base_dir
        / "configs"
        / f"feature_weights_{args.horizon}_xgboost_{timestamp}.yaml"
    )
    with open(output_file, "w", encoding="utf-8") as f:
        yaml.dump(
            {
                "description": f'[Phase 1.4 XGBoost 학습] 개별 피처 가중치 (Objective Score={best_result["objective_score"]:.4f})',
                "horizon": args.horizon,
                "hyperparameters": best_params,
                "metadata": {
                    "dev_objective_score": best_result["objective_score"],
                    "dev_hit_ratio": (
                        float(best_result["hit_ratio"])
                        if not np.isnan(best_result["hit_ratio"])
                        else None
                    ),
                    "dev_ic_mean": (
                        float(best_result["ic_mean"])
                        if not np.isnan(best_result["ic_mean"])
                        else None
                    ),
                    "dev_icir": (
                        float(best_result["icir"])
                        if not np.isnan(best_result["icir"])
                        else None
                    ),
                    "holdout_objective_score": (
                        float(best_result_holdout["objective_score"])
                        if "best_result_holdout" in locals()
                        and not np.isnan(
                            best_result_holdout.get("objective_score", np.nan)
                        )
                        else None
                    ),
                    "holdout_hit_ratio": (
                        float(best_result_holdout["hit_ratio"])
                        if "best_result_holdout" in locals()
                        and not np.isnan(best_result_holdout.get("hit_ratio", np.nan))
                        else None
                    ),
                    "holdout_ic_mean": (
                        float(best_result_holdout["ic_mean"])
                        if "best_result_holdout" in locals()
                        and not np.isnan(best_result_holdout.get("ic_mean", np.nan))
                        else None
                    ),
                    "holdout_icir": (
                        float(best_result_holdout["icir"])
                        if "best_result_holdout" in locals()
                        and not np.isnan(best_result_holdout.get("icir", np.nan))
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

    # 모델 저장 (선택적)
    model_dir = base_dir / "artifacts" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / f"xgboost_{args.horizon}_{timestamp}.pkl"
    import pickle

    with open(model_file, "wb") as f:
        pickle.dump(best_model, f)
    print(f"  - 모델 파일: {model_file}")

    # 결과 요약 CSV 저장
    results_df = pd.DataFrame(results_summary)
    results_csv = (
        base_dir
        / "artifacts"
        / "reports"
        / f"track_a_xgboost_learning_{args.horizon}_{timestamp}.csv"
    )
    results_df.to_csv(results_csv, index=False)
    print(f"  - 결과 요약: {results_csv}")

    # 최종 결과 출력
    print("\n" + "=" * 80)
    print(f"[최적 결과] 파라미터={best_params}")
    print("=" * 80)
    print("\n[Dev 구간]")
    print(f"  Objective Score: {best_result['objective_score']:.4f}")
    print(
        f"  Hit Ratio: {best_result['hit_ratio']:.4f}"
        if not np.isnan(best_result["hit_ratio"])
        else "  Hit Ratio: NaN"
    )
    print(
        f"  IC Mean: {best_result['ic_mean']:.4f}"
        if not np.isnan(best_result["ic_mean"])
        else "  IC Mean: NaN"
    )
    print(
        f"  ICIR: {best_result['icir']:.4f}"
        if not np.isnan(best_result["icir"])
        else "  ICIR: NaN"
    )

    if "best_result_holdout" in locals() and not np.isnan(
        best_result_holdout.get("ic_mean", np.nan)
    ):
        print("\n[Holdout 구간]")
        print(f"  Objective Score: {best_result_holdout['objective_score']:.4f}")
        print(
            f"  Hit Ratio: {best_result_holdout['hit_ratio']:.4f}"
            if not np.isnan(best_result_holdout["hit_ratio"])
            else "  Hit Ratio: NaN"
        )
        print(
            f"  IC Mean: {best_result_holdout['ic_mean']:.4f}"
            if not np.isnan(best_result_holdout["ic_mean"])
            else "  IC Mean: NaN"
        )
        print(
            f"  ICIR: {best_result_holdout['icir']:.4f}"
            if not np.isnan(best_result_holdout["icir"])
            else "  ICIR: NaN"
        )

    print(f"\n최적 가중치 파일: {output_file}")


if __name__ == "__main__":
    main()
