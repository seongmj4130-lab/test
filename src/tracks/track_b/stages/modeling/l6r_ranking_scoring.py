# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/stages/modeling/l6r_ranking_scoring.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.components.ranking.score_engine import build_ranking_daily


@dataclass(frozen=True)
class RankingRebalanceConfig:
    """
    [개선안 13번] 랭킹 기반 리밸런싱 스코어 생성 설정

    - L7은 rebalance_scores(score_ens)를 입력으로 사용하므로,
      ranking_daily(score_total/rank_total)를 rebalance_scores 형태로 변환한다.

    [Dual Horizon 전략] ranking_short_daily와 ranking_long_daily를 α로 결합:
      score_ens = α * rank_short + (1-α) * rank_long
    """

    # 어떤 랭킹 신호를 score_ens로 쓸지
    score_source: str = "score_total"  # "score_total" | "rank_total"

    # 반환(백테스트) 컬럼: dataset_daily의 ret_fwd_*d를 사용
    return_col: str = "true_short"

    # [Dual Horizon] 단기/장기 랭킹 결합 가중치 (α)
    # None이면 단일 랭킹 모드 (기존 동작)
    alpha_short: Optional[float] = None  # 단기 랭킹 가중치 (0.0~1.0)
    alpha_long: Optional[
        float
    ] = None  # 장기 랭킹 가중치 (0.0~1.0, alpha_short가 있으면 1-alpha_short로 자동 계산)


def _ensure_datetime(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        raise KeyError(f"missing column: {col}")
    df[col] = pd.to_datetime(df[col], errors="raise")


def _ensure_ticker(df: pd.DataFrame, col: str = "ticker") -> None:
    if col not in df.columns:
        raise KeyError(f"missing column: {col}")
    df[col] = df[col].astype(str).str.zfill(6)


def _segment_to_phase(seg: str) -> str:
    s = (seg or "").strip().lower()
    if s in ("holdout", "test"):
        return "holdout"
    return "dev"


def _map_date_to_phase(date: pd.Timestamp, cv_folds: pd.DataFrame) -> str:
    """
    [rebalance_interval 개선] 날짜를 가장 가까운 cv_folds.test_end의 phase로 매핑

    Args:
        date: 매핑할 날짜
        cv_folds: cv_folds_short DataFrame (test_end, segment 컬럼 포함)

    Returns:
        phase: "dev" 또는 "holdout"
    """
    if cv_folds.empty or "test_end" not in cv_folds.columns:
        return "dev"  # 기본값

    cv_folds = cv_folds.copy()
    cv_folds["test_end"] = pd.to_datetime(cv_folds["test_end"], errors="coerce")
    cv_folds = cv_folds.dropna(subset=["test_end"])

    if len(cv_folds) == 0:
        return "dev"

    # 가장 가까운 test_end 찾기
    date_ts = pd.to_datetime(date)
    cv_folds["date_diff"] = (cv_folds["test_end"] - date_ts).abs()
    closest = cv_folds.loc[cv_folds["date_diff"].idxmin()]

    # segment를 phase로 변환
    return _segment_to_phase(str(closest.get("segment", "dev")))


def build_rebalance_scores_from_ranking(
    *,
    dataset_daily: pd.DataFrame,
    cv_folds_short: pd.DataFrame,
    universe_k200_membership_monthly: pd.DataFrame,
    cfg: dict,
    cfg_rank: RankingRebalanceConfig,
    ranking_short_daily: Optional[pd.DataFrame] = None,  # [Dual Horizon] 단기 랭킹
    ranking_long_daily: Optional[pd.DataFrame] = None,  # [Dual Horizon] 장기 랭킹
    ohlcv_daily: Optional[pd.DataFrame] = None,  # 시장 국면 분류용 OHLCV 데이터
    rebalance_interval: int = 1,  # [rebalance_interval 개선] 리밸런싱 주기 (1=월별, >1=일별 필터링)
) -> tuple[pd.DataFrame, pd.DataFrame, dict, list[str]]:
    """
    [개선안 13번] 랭킹 기반 전략용 rebalance_scores 생성

    Args:
        dataset_daily: L4 산출물 (date,ticker,ret_fwd_*, features..., in_universe 포함 권장)
        cv_folds_short: L4 산출물 (fold_id, segment, test_end 등) - 리밸런싱 날짜/phase 결정용
        universe_k200_membership_monthly: L0 산출물 - in_universe 강제 적용용
        cfg: config.yaml dict
        cfg_rank: 랭킹→리밸런싱 변환 설정
        ranking_short_daily: [Dual Horizon] 단기 랭킹 (L8_short 출력)
        ranking_long_daily: [Dual Horizon] 장기 랭킹 (L8_long 출력)

    Returns:
        (rebalance_scores, rebalance_scores_summary, quality, warns)

    Notes:
        - L7의 기본 score_col은 score_ens 이므로, ranking 신호를 score_ens로 만들어준다.
        - L7의 기본 ret_col은 true_short 이므로, dataset_daily의 ret_fwd_{horizon_short}d를 true_short로 복제한다.
        - [Dual Horizon] ranking_short_daily와 ranking_long_daily가 모두 제공되면 α로 결합:
          score_ens = α * rank_short + (1-α) * rank_long
    """
    warns: list[str] = []

    if (
        dataset_daily is None
        or not isinstance(dataset_daily, pd.DataFrame)
        or dataset_daily.empty
    ):
        raise ValueError("dataset_daily is empty")
    if (
        cv_folds_short is None
        or not isinstance(cv_folds_short, pd.DataFrame)
        or cv_folds_short.empty
    ):
        raise ValueError("cv_folds_short is empty")
    if (
        universe_k200_membership_monthly is None
        or universe_k200_membership_monthly.empty
    ):
        raise ValueError("universe_k200_membership_monthly is empty")

    ds = dataset_daily.copy()
    _ensure_datetime(ds, "date")
    _ensure_ticker(ds, "ticker")

    # [Dual Horizon] 단기/장기 랭킹 결합 모드 확인
    use_dual_horizon = (
        ranking_short_daily is not None
        and not ranking_short_daily.empty
        and ranking_long_daily is not None
        and not ranking_long_daily.empty
        and cfg_rank.alpha_short is not None
    )

    # [Dual Horizon] 시장 국면별 α 조정 설정 확인
    regime_alpha_config = None
    market_regime_df_for_alpha = None
    if use_dual_horizon:
        l6r = (cfg.get("l6r", {}) if isinstance(cfg, dict) else {}) or {}
        regime_alpha_config = l6r.get("regime_alpha", None)
        regime_enabled = cfg.get("l7", {}).get("regime", {}).get("enabled", False)

        if regime_alpha_config and regime_enabled:
            # 시장 국면 데이터 생성 (α 조정용) - 외부 API 없이 ohlcv_daily 사용
            from src.tracks.shared.stages.regime.l1d_market_regime import (
                build_market_regime,
            )

            dates = ds["date"].unique()

            if ohlcv_daily is None or ohlcv_daily.empty:
                warns.append(
                    "[L6R Dual Horizon] ohlcv_daily가 없어 시장 국면별 α 조정을 건너뜁니다."
                )
                market_regime_df_for_alpha = None
            else:
                try:
                    regime_cfg = cfg.get("l7", {}).get("regime", {})
                    market_regime_df_for_alpha = build_market_regime(
                        rebalance_dates=dates,
                        ohlcv_daily=ohlcv_daily,
                        lookback_days=regime_cfg.get("lookback_days", 60),
                        neutral_band=regime_cfg.get("neutral_band", 0.05),
                        use_volume=regime_cfg.get("use_volume", True),
                        use_volatility=regime_cfg.get("use_volatility", True),
                    )
                    warns.append(
                        f"[L6R Dual Horizon] 시장 국면별 α 조정 활성화: {len(market_regime_df_for_alpha)}개 날짜"
                    )
                except Exception as e:
                    warns.append(
                        f"[L6R Dual Horizon] 시장 국면 데이터 생성 실패: {e}. 기본 α 사용."
                    )
                    market_regime_df_for_alpha = None

    if use_dual_horizon:
        warns.append(
            f"[L6R Dual Horizon] 단기/장기 랭킹 결합 모드: α_short={cfg_rank.alpha_short:.2f}"
        )
        # ranking_short_daily와 ranking_long_daily를 사용 (별도 계산 불필요)
        ranking_daily = None  # 단일 랭킹 모드가 아님 (rebalance_interval > 1일 때 ranking_short_daily 사용)
    else:
        # 기존 단일 랭킹 모드
        warns.append("[L6R] 단일 랭킹 모드 (기존 동작)")

        # 1) ranking config (reuse L8 config)
        l8 = (
            (cfg.get("l8", {}) if isinstance(cfg, dict) else {})
            or (cfg.get("params", {}).get("l8", {}) if isinstance(cfg, dict) else {})
            or {}
        )
        normalization_method = str(l8.get("normalization_method", "percentile"))
        feature_groups_config = l8.get(
            "feature_groups_config", "configs/feature_groups.yaml"
        )
        feature_weights_config = l8.get(
            "feature_weights_config", None
        )  # [IC 최적화] 최적 가중치 파일 경로
        regime_aware_weights_config = l8.get(
            "regime_aware_weights_config", None
        )  # [국면별 전략] 국면별 가중치 파일 경로
        use_sector_relative = bool(l8.get("use_sector_relative", True))
        sector_col = str(l8.get("sector_col", "sector_name"))

        base_dir = (
            Path((cfg.get("paths", {}) or {}).get("base_dir", Path.cwd()))
            if isinstance(cfg, dict)
            else Path.cwd()
        )
        feature_groups_path = (
            (base_dir / feature_groups_config) if feature_groups_config else None
        )

        # [IC 최적화] 최적 가중치 파일 로드
        optimal_feature_weights = None
        if feature_weights_config:
            weights_path = base_dir / feature_weights_config
            if weights_path.exists():
                try:
                    import yaml

                    with open(weights_path, encoding="utf-8") as f:
                        weights_data = yaml.safe_load(f)
                    optimal_feature_weights = weights_data.get("feature_weights", {})
                    warns.append(
                        f"[L6R IC 최적화] 최적 가중치 로드 완료: {len(optimal_feature_weights)}개 피처"
                    )
                except Exception as e:
                    warns.append(
                        f"[L6R IC 최적화] 최적 가중치 로드 실패: {e}. feature_groups 사용."
                    )
            else:
                warns.append(
                    f"[L6R IC 최적화] 최적 가중치 파일이 없습니다: {weights_path}. feature_groups 사용."
                )

        # [국면별 전략] 국면별 가중치 로드
        regime_weights_config = None
        market_regime_df = None
        regime_enabled = cfg.get("l7", {}).get("regime", {}).get("enabled", False)

        if regime_aware_weights_config and regime_enabled:
            from src.components.ranking.regime_strategy import load_regime_weights

            regime_weights_config = load_regime_weights(
                config_path=regime_aware_weights_config,
                base_dir=base_dir,
            )
            if len(regime_weights_config) > 0:
                warns.append(
                    f"[L6R 국면별 전략] 국면별 가중치 로드 완료: {list(regime_weights_config.keys())}"
                )

                # 시장 국면 데이터 생성 - 외부 API 없이 ohlcv_daily 사용
                from src.tracks.shared.stages.regime.l1d_market_regime import (
                    build_market_regime,
                )

                dates = ds["date"].unique()

                if ohlcv_daily is None or ohlcv_daily.empty:
                    warns.append(
                        "[L6R 국면별 전략] ohlcv_daily가 없어 시장 국면 기능을 건너뜁니다."
                    )
                    regime_weights_config = None
                    market_regime_df = None
                else:
                    try:
                        regime_cfg = cfg.get("l7", {}).get("regime", {})
                        market_regime_df = build_market_regime(
                            rebalance_dates=dates,
                            ohlcv_daily=ohlcv_daily,
                            lookback_days=regime_cfg.get("lookback_days", 60),
                            neutral_band=regime_cfg.get("neutral_band", 0.05),
                            use_volume=regime_cfg.get("use_volume", True),
                            use_volatility=regime_cfg.get("use_volatility", True),
                        )
                        warns.append(
                            f"[L6R 국면별 전략] 시장 국면 데이터 생성 완료: {len(market_regime_df)}개 날짜"
                        )
                    except Exception as e:
                        warns.append(
                            f"[L6R 국면별 전략] 시장 국면 데이터 생성 실패: {e}. 국면별 가중치 비활성화."
                        )
                        regime_weights_config = None
                        market_regime_df = None

        # 최적 가중치가 있으면 우선 사용, 없으면 feature_groups 사용 (국면별 가중치가 없을 때만)
        use_feature_groups = (
            feature_groups_path
            if optimal_feature_weights is None
            and regime_weights_config is None
            and feature_groups_path
            and feature_groups_path.exists()
            else None
        )

        # 2) ranking_daily 계산 (score_total, rank_total)
        ranking_daily = build_ranking_daily(
            ds,
            feature_cols=None,
            feature_weights=optimal_feature_weights,  # [IC 최적화] 최적 가중치 또는 None (국면별 가중치가 없을 때만 사용)
            feature_groups_config=use_feature_groups,  # 최적 가중치가 없을 때만 사용
            normalization_method=normalization_method,  # percentile|zscore
            date_col="date",
            universe_col="in_universe",
            sector_col=sector_col if sector_col in ds.columns else None,
            use_sector_relative=use_sector_relative,
            market_regime_df=market_regime_df,  # [국면별 전략]
            regime_weights_config=regime_weights_config,  # [국면별 전략]
        )

    # 3) 리밸런싱 날짜/phase 결정
    # [bt20 프로페셔널] 적응형 리밸런싱 적용을 위해 bt20_pro는 L6R에서 필터링하지 않음
    # cfg에서 전략 확인
    strategy_name = cfg.get("l7", {}).get("score_col", "")
    if "bt20_pro" in str(strategy_name) or "bt20_pro" in str(cfg.get("l7", {})):
        logger.info(
            "[bt20 프로페셔널] L6R에서 rebalance_interval=1로 설정하여 L7에서 적응형 리밸런싱 적용"
        )
        rebalance_interval = 1
    else:
        rebalance_interval = (
            int(rebalance_interval) if rebalance_interval is not None else 1
        )

    if rebalance_interval == 1:
        # 기존 로직: cv_folds_short.test_end 사용 (월별)
        req_cols = {"test_end", "segment"}
        if not req_cols.issubset(set(cv_folds_short.columns)):
            raise KeyError(
                f"cv_folds_short missing required columns: {sorted(list(req_cols - set(cv_folds_short.columns)))}"
            )

        folds = cv_folds_short.copy()
        folds["test_end"] = pd.to_datetime(folds["test_end"], errors="raise")
        folds["phase"] = folds["segment"].astype(str).map(_segment_to_phase)

        # 날짜가 중복되는 경우(이상 케이스) 보수적으로 holdout 우선
        dup_dates = folds.duplicated(["test_end"]).any()
        if dup_dates:
            warns.append(
                "[L6R] Duplicate rebalance dates in cv_folds_short.test_end -> resolving by phase priority(holdout>dev)"
            )
            pr = folds["phase"].map({"dev": 0, "holdout": 1}).fillna(0).astype(int)
            folds = (
                folds.assign(_prio=pr)
                .sort_values(["test_end", "_prio"])
                .drop_duplicates(["test_end"], keep="last")
            )
            folds = folds.drop(columns=["_prio"])

        rebal_map = (
            folds[["test_end", "phase"]].rename(columns={"test_end": "date"}).copy()
        )
        warns.append(
            f"[L6R rebalance_interval] 월별 리밸런싱 사용 (cv_folds_short.test_end): {len(rebal_map)}개 날짜"
        )
    else:
        # [rebalance_interval 개선] 일별 랭킹 데이터에서 rebalance_interval에 맞게 필터링
        if use_dual_horizon:
            # Dual Horizon 모드: ranking_short_daily 사용
            if ranking_short_daily is None or ranking_short_daily.empty:
                raise ValueError(
                    "[L6R rebalance_interval] rebalance_interval > 1 requires ranking_short_daily"
                )
            all_dates = sorted(pd.to_datetime(ranking_short_daily["date"].unique()))
        else:
            # 단일 랭킹 모드: ranking_daily 사용
            if ranking_daily is None or ranking_daily.empty:
                raise ValueError(
                    "[L6R rebalance_interval] rebalance_interval > 1 requires ranking_daily"
                )
            all_dates = sorted(pd.to_datetime(ranking_daily["date"].unique()))

        # rebalance_interval에 맞게 필터링
        rebalance_dates = [
            all_dates[i] for i in range(0, len(all_dates), rebalance_interval)
        ]

        # Phase는 cv_folds_short에서 매핑
        rebal_map = pd.DataFrame({"date": rebalance_dates})
        rebal_map["phase"] = rebal_map["date"].apply(
            lambda d: _map_date_to_phase(d, cv_folds_short)
        )
        warns.append(
            f"[L6R rebalance_interval] 일별 리밸런싱 사용 (interval={rebalance_interval}): {len(rebal_map)}개 날짜 (전체 {len(all_dates)}개 중)"
        )

    # 4) [Dual Horizon] 단기/장기 랭킹 결합 또는 단일 랭킹 필터링
    if use_dual_horizon:
        # 단기/장기 랭킹 결합
        r_short = ranking_short_daily.copy()
        r_long = ranking_long_daily.copy()
        _ensure_datetime(r_short, "date")
        _ensure_datetime(r_long, "date")
        _ensure_ticker(r_short, "ticker")
        _ensure_ticker(r_long, "ticker")

        # 리밸런싱 날짜로 필터링 + phase 부착
        r_short = r_short.merge(
            rebal_map, on="date", how="inner", validate="many_to_one"
        )
        r_long = r_long.merge(rebal_map, on="date", how="inner", validate="many_to_one")

        # 단기/장기 랭킹 병합
        key = ["date", "ticker", "phase"]
        r = r_short[["date", "ticker", "phase", "score_total", "rank_total"]].merge(
            r_long[["date", "ticker", "phase", "score_total", "rank_total"]],
            on=key,
            how="outer",
            suffixes=("_short", "_long"),
            validate="one_to_one",
        )

        # [Dual Horizon] 시장 국면별 α 조정
        score_source = str(cfg_rank.score_source).strip().lower()
        if score_source == "rank_total":
            # rank_total을 사용: 낮은 rank가 좋으므로 -rank로 변환
            r["score_short_norm"] = -pd.to_numeric(
                r["rank_total_short"], errors="coerce"
            )
            r["score_long_norm"] = -pd.to_numeric(r["rank_total_long"], errors="coerce")
        else:
            # score_total을 사용: 높은 score가 좋음
            r["score_short_norm"] = pd.to_numeric(
                r["score_total_short"], errors="coerce"
            )
            r["score_long_norm"] = pd.to_numeric(r["score_total_long"], errors="coerce")

        # 기본 α
        base_alpha_short = float(cfg_rank.alpha_short)
        base_alpha_long = (
            float(cfg_rank.alpha_long)
            if cfg_rank.alpha_long is not None
            else (1.0 - base_alpha_short)
        )

        # 시장 국면별 α 적용
        if regime_alpha_config and market_regime_df_for_alpha is not None:
            # 시장 국면 데이터와 병합
            r = r.merge(
                market_regime_df_for_alpha[["date", "regime"]],
                on="date",
                how="left",
                validate="many_to_one",
            )

            # 국면별 α 매핑
            def get_alpha_for_regime(regime) -> float:
                """
                [개선안 22번] 3단계/5단계 국면 모두 지원
                - config의 regime_alpha가 bull/neutral/bear 3키만 있어도 동작
                - 기존 bull_strong/bull_weak/... 도 그대로 지원
                """
                # [수정] NaN 처리: pd.isna()로 NaN 체크 후 문자열 변환
                if pd.isna(regime) or regime is None:
                    regime_lower = ""
                else:
                    regime_lower = str(regime).strip().lower()

                if regime_lower in regime_alpha_config:
                    return float(
                        regime_alpha_config.get(regime_lower, base_alpha_short)
                    )
                # 5단계 -> 3단계 매핑
                try:
                    from src.stages.backtest.regime_utils import map_regime_to_3level

                    r3 = map_regime_to_3level(regime_lower)
                    return float(regime_alpha_config.get(r3, base_alpha_short))
                except Exception:
                    return float(base_alpha_short)

            r["alpha_short"] = r["regime"].apply(get_alpha_for_regime)
            r["alpha_long"] = 1.0 - r["alpha_short"]

            warns.append(
                f"[L6R Dual Horizon] 시장 국면별 α 적용: {r['regime'].value_counts().to_dict()}"
            )
        else:
            # 기본 α 사용
            r["alpha_short"] = base_alpha_short
            r["alpha_long"] = base_alpha_long

        # 결합: score_ens = α * score_short + (1-α) * score_long
        r["score_ens"] = r["alpha_short"] * r["score_short_norm"].fillna(0) + r[
            "alpha_long"
        ] * r["score_long_norm"].fillna(0)

        # score_total, rank_total은 결합된 score_ens 기반으로 재계산
        r["score_total"] = r["score_ens"]
        r["rank_total"] = r.groupby(["date", "phase"], sort=False)["score_ens"].rank(
            method="first", ascending=False
        )

        # [단기/장기 랭킹 분리 분석] score_total_short와 score_total_long을 최종 출력에 포함
        # score_short_norm과 score_long_norm을 score_total_short, score_total_long으로 변환
        r["score_total_short"] = r["score_short_norm"]
        r["score_total_long"] = r["score_long_norm"]

        # 임시 컬럼 제거 (score_short_norm, score_long_norm은 제거하되 score_total_short/long은 유지)
        drop_cols = ["score_short_norm", "score_long_norm", "alpha_short", "alpha_long"]
        if "regime" in r.columns:
            drop_cols.append("regime")
        r = r.drop(columns=[c for c in drop_cols if c in r.columns])

        warns.append("[L6R Dual Horizon] 단기/장기 랭킹 결합 완료")
    else:
        # 단일 랭킹 모드 (기존 동작)
        r = ranking_daily.copy()
        _ensure_datetime(r, "date")
        _ensure_ticker(r, "ticker")
        r = r.merge(rebal_map, on="date", how="inner", validate="many_to_one")

    # 5) 월별 유니버스 강제 적용 (L6와 동일한 정책으로 in_universe 필터)
    un = universe_k200_membership_monthly.copy()
    if "ym" not in un.columns:
        if "date" not in un.columns:
            raise KeyError("universe_k200_membership_monthly must have 'date' or 'ym'")
        un["date"] = pd.to_datetime(un["date"], errors="raise")
        un["ym"] = un["date"].dt.to_period("M").astype(str)
    else:
        un["ym"] = un["ym"].astype(str)
    _ensure_ticker(un, "ticker")
    un_key = un[["ym", "ticker"]].drop_duplicates()

    r["ym"] = r["date"].dt.to_period("M").astype(str)
    r = r.merge(un_key, on=["ym", "ticker"], how="left", indicator=True)
    r["in_universe"] = r["_merge"].eq("both")
    r = r.drop(columns=["_merge", "ym"])
    r = r.loc[r["in_universe"]].copy()

    # 6) 수익률(타깃) 컬럼 부착: dataset_daily의 ret_fwd_{horizon_short}d -> true_short, ret_fwd_{horizon_long}d -> true_long
    l4 = (
        (cfg.get("l4", {}) if isinstance(cfg, dict) else {})
        or (cfg.get("params", {}).get("l4", {}) if isinstance(cfg, dict) else {})
        or {}
    )
    horizon_short = int(l4.get("horizon_short", 20))
    horizon_long = int(l4.get("horizon_long", 120))

    # true_short 부착 (기본)
    ret_fwd_col_short = f"ret_fwd_{horizon_short}d"
    if ret_fwd_col_short not in ds.columns:
        raise KeyError(
            f"[L6R] dataset_daily missing required return column: {ret_fwd_col_short}"
        )

    ret_info_short = ds[["date", "ticker", ret_fwd_col_short]].copy()
    ret_info_short = ret_info_short.drop_duplicates(["date", "ticker"])
    r = r.merge(
        ret_info_short, on=["date", "ticker"], how="left", validate="one_to_one"
    )
    # [North Star] BT20 지원: true_short 컬럼 생성 (cfg_rank.return_col이 이미 true_short인 경우 중복 방지)
    if cfg_rank.return_col != "true_short":
        r[cfg_rank.return_col] = r[ret_fwd_col_short].astype(float)
    # [동적 Return] true_short는 holding_days에 따라 동적으로 선택하므로 여기서는 생성하지 않음
    # r["true_short"] = r[ret_fwd_col_short].astype(float)  # [North Star] BT20 지원 - 주석처리

    # true_long 부착 (North Star: BT120 지원)
    ret_fwd_col_long = f"ret_fwd_{horizon_long}d"
    if ret_fwd_col_long in ds.columns:
        ret_info_long = ds[["date", "ticker", ret_fwd_col_long]].copy()
        ret_info_long = ret_info_long.drop_duplicates(["date", "ticker"])
        r = r.merge(
            ret_info_long, on=["date", "ticker"], how="left", validate="one_to_one"
        )
        r["true_long"] = r[ret_fwd_col_long].astype(float)

    # [동적 Return 계산] 다양한 holding_days에 대한 forward return 추가
    dynamic_holding_days = [20, 40, 60, 80, 100, 120]
    for hd in dynamic_holding_days:
        ret_fwd_col = f"ret_fwd_{hd}d"
        if ret_fwd_col in ds.columns:
            ret_info = ds[["date", "ticker", ret_fwd_col]].copy()
            ret_info = ret_info.drop_duplicates(["date", "ticker"])
            r = r.merge(
                ret_info, on=["date", "ticker"], how="left", validate="one_to_one"
            )
            r[f"true_{hd}d"] = r[ret_fwd_col].astype(float)  # [North Star] BT120 지원
        warns.append(
            f"[L6R North Star] true_long 부착 완료 (BT120 지원): {ret_fwd_col_long}"
        )
    else:
        warns.append(
            f"[L6R North Star] true_long 부착 스킵: {ret_fwd_col_long} 컬럼 없음 (BT120 비활성화)"
        )
        r["true_long"] = np.nan

    # 7) score_ens 생성(= L7 입력 점수 컬럼)
    # [Dual Horizon] 이미 score_ens가 계산되어 있으면 스킵
    if "score_ens" not in r.columns:
        score_source = str(cfg_rank.score_source).strip().lower()
        if score_source == "rank_total":
            if "rank_total" not in r.columns:
                raise KeyError("[L6R] rank_total not found in ranking_daily output")
            r["score_ens"] = (-pd.to_numeric(r["rank_total"], errors="coerce")).astype(
                float
            )
        else:
            if "score_total" not in r.columns:
                raise KeyError("[L6R] score_total not found in ranking_daily output")
            r["score_ens"] = pd.to_numeric(r["score_total"], errors="coerce").astype(
                float
            )

    # 8) L7이 기대하는 최소 스키마로 정리 (North Star: true_short/true_long 모두 포함)
    # [단기/장기 랭킹 분리 분석] score_total_short와 score_total_long 포함
    out_cols = [
        "date",
        "ticker",
        "phase",
        "score_ens",
        cfg_rank.return_col,
        "true_long",
        "score_total",
        "rank_total",
        "in_universe",
    ]

    # [동적 Return 계산] 다양한 holding_days에 대한 return 컬럼 추가
    dynamic_holding_days = [20, 40, 60, 80, 100, 120]
    for hd in dynamic_holding_days:
        dynamic_col = f"true_{hd}d"
        if dynamic_col in r.columns:
            out_cols.append(dynamic_col)
    # score_total_short와 score_total_long이 있으면 추가
    if "score_total_short" in r.columns:
        out_cols.append("score_total_short")
    if "score_total_long" in r.columns:
        out_cols.append("score_total_long")
    # 중복 제거 (cfg_rank.return_col이 "true_short"인 경우)
    out_cols = list(dict.fromkeys(out_cols))  # 순서 유지하면서 중복 제거
    out_cols = [c for c in out_cols if c in r.columns]
    out = r[out_cols].copy()

    # key uniqueness (L7는 date/phase 단위로 묶기 때문에 (date,ticker,phase) 유일해야 함)
    dup = int(out.duplicated(["date", "ticker", "phase"]).sum())
    if dup > 0:
        raise ValueError(f"[L6R] duplicate keys(date,ticker,phase)={dup}")

    # 9) summary (L6 스키마 호환)
    # REQUIRED_COLS_BY_OUTPUT["rebalance_scores_summary"] expects:
    #   date, phase, n_tickers, coverage_vs_universe_pct, score_short_missing, score_long_missing, score_ens_missing
    summary = (
        out.groupby(["date", "phase"], sort=False)
        .agg(
            n_tickers=("ticker", "nunique"),
            score_ens_missing=("score_ens", lambda s: float(s.isna().mean())),
        )
        .reset_index()
    )

    # universe coverage (월별 멤버십 대비)
    un_counts = universe_k200_membership_monthly.copy()
    if "ym" not in un_counts.columns:
        if "date" not in un_counts.columns:
            raise KeyError("universe_k200_membership_monthly must have 'date' or 'ym'")
        un_counts["date"] = pd.to_datetime(un_counts["date"], errors="raise")
        un_counts["ym"] = un_counts["date"].dt.to_period("M").astype(str)
    else:
        un_counts["ym"] = un_counts["ym"].astype(str)
    un_counts["ticker"] = un_counts["ticker"].astype(str).str.zfill(6)
    un_counts = (
        un_counts.groupby("ym", sort=False)["ticker"]
        .nunique()
        .rename("universe_n_tickers")
        .reset_index()
    )

    summary["ym"] = (
        pd.to_datetime(summary["date"], errors="coerce").dt.to_period("M").astype(str)
    )
    summary = summary.merge(un_counts, on="ym", how="left")
    summary["coverage_vs_universe_pct"] = (
        summary["n_tickers"] / summary["universe_n_tickers"].replace(0, np.nan) * 100.0
    )
    summary = summary.drop(columns=["ym"])

    # ranking 모드에서는 short/long 모델 스코어가 존재하지 않으므로 "100% missing"으로 표기
    # (컬럼 존재/스키마 호환 목적 + 의미적으로도 '해당 신호 없음'을 명시)
    summary["score_short_missing"] = 1.0
    summary["score_long_missing"] = 1.0

    # 참고용 (L6에는 mean/std가 있지만 REQUIRED에는 없으므로 넣지 않음)

    quality = {
        "scoring_ranking": {
            "mode": "ranking",
            "score_source": score_source,
            "ret_fwd_col_used": ret_fwd_col_short,  # [North Star] 변수명 수정
            "return_col_used": cfg_rank.return_col,
            "true_long_attached": bool(
                "true_long" in out.columns and out["true_long"].notna().any()
            ),  # [North Star] BT120 지원 여부 (JSON 직렬화를 위해 bool 변환)
            "rows": int(len(out)),
            "unique_dates": int(out["date"].nunique()),
            "unique_tickers": int(out["ticker"].nunique()),
            "phases": sorted(out["phase"].dropna().unique().tolist()),
        }
    }

    return (
        out.sort_values(
            ["phase", "date", "score_ens", "ticker"],
            ascending=[True, True, False, True],
        ).reset_index(drop=True),
        summary,
        quality,
        warns,
    )


def run_L6R_ranking_scoring(
    cfg: dict,
    artifacts: dict,
    *,
    force: bool = False,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """
    [개선안 13번] L6R: ranking 기반 rebalance_scores 생성 (L7 입력용)

    outputs:
      - rebalance_scores
      - rebalance_scores_summary

    [Dual Horizon] ranking_short_daily와 ranking_long_daily가 있으면 α로 결합
    """
    warns: list[str] = []  # 경고 메시지 리스트 초기화

    l7 = (cfg.get("l7", {}) if isinstance(cfg, dict) else {}) or {}
    l6r = (cfg.get("l6r", {}) if isinstance(cfg, dict) else {}) or {}

    score_source = str(l7.get("ranking_score_source", "score_total"))

    # [앙상블 가중치 적용] Track A 앙상블 가중치 로드
    ensemble_weights = cfg.get("l6r", {}).get("ensemble_weights", {})

    # 앙상블 가중치 로드 확인 로그
    if ensemble_weights:
        warns.append(
            f"[L6R] 앙상블 가중치 로드 완료: 단기 {len(ensemble_weights.get('short', {}))}개, 장기 {len(ensemble_weights.get('long', {}))}개 모델"
        )
        for horizon, weights in ensemble_weights.items():
            if isinstance(weights, dict):
                warns.append(f"[L6R] {horizon} 앙상블: {weights}")
    else:
        warns.append("[L6R] 앙상블 가중치 설정을 찾을 수 없음")

    # [Dual Horizon] α 설정 (기존 로직 유지)
    alpha_short = l6r.get("alpha_short", None)
    alpha_long = l6r.get("alpha_long", None)

    cfg_rank = RankingRebalanceConfig(
        score_source=score_source,
        return_col="true_short",
        alpha_short=alpha_short,
        alpha_long=alpha_long,
    )

    # [Dual Horizon] ranking_short_daily와 ranking_long_daily 확인
    ranking_short_daily = artifacts.get("ranking_short_daily")
    ranking_long_daily = artifacts.get("ranking_long_daily")

    # 시장 국면 분류용 ohlcv_daily 확인
    ohlcv_daily = artifacts.get("ohlcv_daily")

    # [rebalance_interval 개선] rebalance_interval 설정 읽기
    # 우선순위: l7 설정 > l6r 설정 > 기본값(1)
    l7_rebalance_interval = l7.get("rebalance_interval", None)
    l6r_rebalance_interval = l6r.get("rebalance_interval", None)
    rebalance_interval = (
        l7_rebalance_interval
        if l7_rebalance_interval is not None
        else (l6r_rebalance_interval if l6r_rebalance_interval is not None else 1)
    )
    if rebalance_interval is None:
        rebalance_interval = 1

    out, summary, quality, warns = build_rebalance_scores_from_ranking(
        dataset_daily=artifacts["dataset_daily"],
        cv_folds_short=artifacts["cv_folds_short"],
        universe_k200_membership_monthly=artifacts["universe_k200_membership_monthly"],
        cfg=cfg,
        cfg_rank=cfg_rank,
        ranking_short_daily=ranking_short_daily,
        ranking_long_daily=ranking_long_daily,
        ohlcv_daily=ohlcv_daily,
        rebalance_interval=rebalance_interval,  # [rebalance_interval 개선]
    )

    # run_all.py가 meta에 넣을 quality hook
    artifacts["_l6_quality"] = quality

    return {"rebalance_scores": out, "rebalance_scores_summary": summary}, warns
