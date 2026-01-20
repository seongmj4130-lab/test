# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/stages/data/l4_walkforward_split.py
from __future__ import annotations

import pandas as pd


def _sanitize_panel(panel: pd.DataFrame, price_col: str | None = None):
    df = panel.copy()

    # 중복 컬럼 제거
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()

    if "date" not in df.columns or "ticker" not in df.columns:
        raise KeyError("panel_merged_daily must have columns: date, ticker")

    # date/ticker 타입 강제
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = (
        df["ticker"]
        .astype(str)
        .str.extract(r"(\d{6})", expand=False)
        .fillna(df["ticker"].astype(str))
    )
    df["ticker"] = df["ticker"].astype(str).str.zfill(6)

    # 가격 컬럼 결정
    if price_col and price_col in df.columns:
        px = price_col
    elif "adj_close" in df.columns:
        px = "adj_close"
    elif "close" in df.columns:
        px = "close"
    else:
        raise KeyError("Need price column: adj_close or close (or set l4.price_col)")

    df[px] = pd.to_numeric(df[px], errors="coerce")

    # 정렬(shift 안정성)
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    # 필수 결측 체크(너무 심하면 중단)
    if df["date"].isna().mean() > 0.01:
        raise RuntimeError(
            "Too many NaT in date after parsing. Check panel_merged_daily['date']."
        )

    return df, px


def build_inner_cv_folds(
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    k: int,
    embargo_days: int,
    horizon_days: int,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    [Stage 3] 내부 time-series CV folds 생성

    train_start와 train_end 사이를 k개 fold로 나누되, embargo와 horizon을 고려하여
    각 fold의 validation 구간이 겹치지 않도록 구성.

    Args:
        train_start: 학습 구간 시작일
        train_end: 학습 구간 종료일
        k: 내부 CV fold 개수
        embargo_days: embargo 일수
        horizon_days: horizon 일수
        dates: 전체 거래일 인덱스

    Returns:
        DataFrame with columns: inner_fold_id, inner_train_start, inner_train_end,
                               inner_val_start, inner_val_end
    """
    # train 구간의 날짜 인덱스 찾기
    train_start_pos = int(dates.searchsorted(train_start, side="left"))
    train_end_pos = int(dates.searchsorted(train_end, side="right")) - 1

    if train_end_pos <= train_start_pos:
        return pd.DataFrame(
            columns=[
                "inner_fold_id",
                "inner_train_start",
                "inner_train_end",
                "inner_val_start",
                "inner_val_end",
            ]
        )

    # 사용 가능한 날짜 범위 계산 (embargo + horizon 고려)
    # validation 구간이 train_end 이후에 있어야 하므로, 마지막 validation 시작일은
    # train_end - embargo_days - horizon_days 이전이어야 함
    available_end_pos = train_end_pos - embargo_days - horizon_days

    if available_end_pos <= train_start_pos:
        # 사용 가능한 구간이 없으면 빈 DataFrame 반환
        return pd.DataFrame(
            columns=[
                "inner_fold_id",
                "inner_train_start",
                "inner_train_end",
                "inner_val_start",
                "inner_val_end",
            ]
        )

    # k개 fold로 나누기 (time-series이므로 순차적으로)
    # 각 fold의 validation 구간이 겹치지 않도록 구성
    total_available_days = available_end_pos - train_start_pos + 1

    # 각 fold의 validation 구간 크기 (최소 1일)
    val_window_days = max(1, total_available_days // (k + 1))

    inner_folds = []

    for i in range(k):
        # validation 구간: 뒤에서부터 할당 (최신 데이터를 validation으로)
        val_start_pos = available_end_pos - (k - i) * val_window_days + 1
        val_end_pos = min(val_start_pos + val_window_days - 1, available_end_pos)

        if val_start_pos < train_start_pos or val_end_pos < val_start_pos:
            continue

        # train 구간: train_start부터 validation 시작 전까지 (embargo 고려)
        inner_train_end_pos = val_start_pos - embargo_days - horizon_days - 1

        if inner_train_end_pos < train_start_pos:
            # train 구간이 너무 작으면 스킵
            continue

        inner_train_start = dates[train_start_pos]
        inner_train_end = dates[inner_train_end_pos]
        inner_val_start = dates[val_start_pos]
        inner_val_end = dates[val_end_pos]

        inner_folds.append(
            {
                "inner_fold_id": f"inner_{i+1:02d}",
                "inner_train_start": inner_train_start,
                "inner_train_end": inner_train_end,
                "inner_val_start": inner_val_start,
                "inner_val_end": inner_val_end,
            }
        )

    return pd.DataFrame(inner_folds)


def build_targets_and_folds(
    panel_merged_daily: pd.DataFrame,
    *,
    holdout_years: int = 2,
    step_days: int = 20,
    test_window_days: int = 20,
    embargo_days: int = 20,
    horizon_short: int = 20,
    horizon_long: int = 120,
    rolling_train_years_short: int = 3,
    rolling_train_years_long: int = 5,
    price_col: str | None = None,
    # [Stage 1] 추가 파라미터
    drop_non_universe_before_save: bool = False,
):
    warnings: list[str] = []

    df, px = _sanitize_panel(panel_merged_daily, price_col)

    # forward return 계산 (수정 핵심: 분모는 df[px]!)
    cur = df[px]
    g = df.groupby("ticker", sort=False)[px]

    fwd_s = g.shift(-horizon_short)
    fwd_l = g.shift(-horizon_long)

    # 0/NaN 분모 방어
    cur_safe = cur.where(cur != 0)

    df[f"ret_fwd_{horizon_short}d"] = fwd_s / cur_safe - 1.0
    df[f"ret_fwd_{horizon_long}d"] = fwd_l / cur_safe - 1.0

    # [Phase 5 개선안] Market-Neutral Target 설정: 시장 수익률 계산 및 초과 수익률 생성
    # 시장 수익률 계산 (유니버스 평균, in_universe가 있으면 그것만 사용)
    if "in_universe" in df.columns:
        # 유니버스 멤버만 사용하여 시장 수익률 계산
        universe_mask = df["in_universe"] == True
        market_ret_short = (
            df.loc[universe_mask].groupby("date")[f"ret_fwd_{horizon_short}d"].mean()
        )
        market_ret_long = (
            df.loc[universe_mask].groupby("date")[f"ret_fwd_{horizon_long}d"].mean()
        )
        warnings.append(
            "[Phase 5] Market-Neutral: 시장 수익률 계산 시 in_universe=True 종목만 사용"
        )
    else:
        # in_universe가 없으면 전체 종목 평균 사용
        market_ret_short = df.groupby("date")[f"ret_fwd_{horizon_short}d"].mean()
        market_ret_long = df.groupby("date")[f"ret_fwd_{horizon_long}d"].mean()
        warnings.append(
            "[Phase 5] Market-Neutral: 시장 수익률 계산 시 전체 종목 사용 (in_universe 없음)"
        )

    # 초과 수익률 계산 (타겟 변수로 사용)
    df[f"ret_fwd_{horizon_short}d_excess"] = df[f"ret_fwd_{horizon_short}d"] - df[
        "date"
    ].map(market_ret_short)
    df[f"ret_fwd_{horizon_long}d_excess"] = df[f"ret_fwd_{horizon_long}d"] - df[
        "date"
    ].map(market_ret_long)

    # 시장 수익률 컬럼도 저장 (참고용)
    df["market_ret_20d"] = df["date"].map(market_ret_short)
    df["market_ret_120d"] = df["date"].map(market_ret_long)

    # 초과 수익률 결측률 확인
    miss_s_excess = df[f"ret_fwd_{horizon_short}d_excess"].isna().mean()
    miss_l_excess = df[f"ret_fwd_{horizon_long}d_excess"].isna().mean()
    warnings.append(
        f"[Phase 5] Market-Neutral: 초과 수익률 결측률 ret_fwd_{horizon_short}d_excess={miss_s_excess:.2%}, ret_fwd_{horizon_long}d_excess={miss_l_excess:.2%}"
    )

    # 타깃 결측(마지막 horizon 구간)은 정상이나, 비율 로그용으로 경고만 남김
    miss_s = df[f"ret_fwd_{horizon_short}d"].isna().mean()
    miss_l = df[f"ret_fwd_{horizon_long}d"].isna().mean()
    warnings.append(
        f"[L4] target_missing ret_fwd_{horizon_short}d={miss_s:.2%}, ret_fwd_{horizon_long}d={miss_l:.2%}"
    )

    # trading dates
    dates = pd.DatetimeIndex(pd.unique(df["date"].dropna())).sort_values()
    if len(dates) < 300:
        raise RuntimeError("Too few trading dates after sanitize. Check date parsing.")

    overall_end = dates[-1]
    holdout_threshold = overall_end - pd.DateOffset(years=holdout_years)
    holdout_start = dates[dates.searchsorted(holdout_threshold, side="left")]

    def _build_folds(
        train_years: int,
        horizon_days: int,
        segment: str,
        seg_start: pd.Timestamp,
        seg_end: pd.Timestamp,
    ):
        seg_start = dates[dates.searchsorted(seg_start, side="left")]
        seg_end = dates[dates.searchsorted(seg_end, side="right") - 1]

        start_pos = int(dates.searchsorted(seg_start, side="left"))
        end_pos = int(dates.searchsorted(seg_end, side="right")) - 1

        folds = []
        fold_i = 0
        max_test_start = end_pos - horizon_days - (test_window_days - 1)

        pos = start_pos
        while pos <= max_test_start:
            test_start_pos = pos
            test_end_pos = pos + (test_window_days - 1)

            train_end_pos = test_start_pos - embargo_days - horizon_days - 1
            if train_end_pos < 0:
                pos += step_days
                continue

            train_end = dates[train_end_pos]
            train_start_threshold = train_end - pd.DateOffset(years=train_years)
            train_start_pos = int(
                dates.searchsorted(train_start_threshold, side="left")
            )
            train_start = dates[train_start_pos]

            fold_i += 1
            folds.append(
                {
                    "fold_id": f"{segment}_{fold_i:04d}",
                    "segment": segment,
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": dates[test_start_pos],
                    "test_end": dates[test_end_pos],
                    "train_years": train_years,
                    "horizon_days": horizon_days,
                    "embargo_days": embargo_days,
                    "step_days": step_days,
                    "test_window_days": test_window_days,
                }
            )
            pos += step_days

        return pd.DataFrame(folds)

    dev_start = dates[0]
    dev_end = holdout_start - pd.Timedelta(days=1)

    cv_short = pd.concat(
        [
            _build_folds(
                rolling_train_years_short, horizon_short, "dev", dev_start, dev_end
            ),
            _build_folds(
                rolling_train_years_short,
                horizon_short,
                "holdout",
                holdout_start,
                overall_end,
            ),
        ],
        ignore_index=True,
    )

    cv_long = pd.concat(
        [
            _build_folds(
                rolling_train_years_long, horizon_long, "dev", dev_start, dev_end
            ),
            _build_folds(
                rolling_train_years_long,
                horizon_long,
                "holdout",
                holdout_start,
                overall_end,
            ),
        ],
        ignore_index=True,
    )

    # fold 개수 로그용 경고(메타에 남기기 좋음)
    warnings.append(
        f"[L4] folds_short={len(cv_short):,}, folds_long={len(cv_long):,}, holdout_start={holdout_start.date()}"
    )

    # [Stage 1] drop_non_universe_before_save 옵션
    if drop_non_universe_before_save:
        if "in_universe" in df.columns:
            before = len(df)
            df = df[df["in_universe"] == True].copy()
            after = len(df)
            warnings.append(
                f"[L4] drop_non_universe_before_save=True: {before} -> {after} rows (dropped {before - after})"
            )
        else:
            warnings.append(
                "[L4] drop_non_universe_before_save=True but 'in_universe' column not found -> skipped"
            )

    return df, cv_short, cv_long, warnings
