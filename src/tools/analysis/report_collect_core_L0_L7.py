# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/analysis/report_collect_core_L0_L7.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

# =========================
# 0) 경로/옵션
# =========================
BASE = Path(r"C:/Users/seong/OneDrive/바탕 화면/bootcamp/03_code")
INTERIM = BASE / "data" / "interim"

# 결과 저장 폴더 (보고서용 core 데이터)
OUT_DIR = BASE / "data" / "report_core_L0_L7"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 콘솔 출력 옵션
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)


# =========================
# 1) 유틸
# =========================
def _artifact_path(name: str) -> Path:
    return INTERIM / f"{name}.parquet"


def load_parquet_if_exists(name: str) -> Optional[pd.DataFrame]:
    p = _artifact_path(name)
    if not p.exists():
        print(f"[MISS] {name} (not found) -> {p}")
        return None
    df = pd.read_parquet(p)
    print(f"[OK] {name} -> {p} shape={df.shape}")
    return df


def pick_first_existing_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def select_cols(
    df: pd.DataFrame, required: list[str], optional: list[str] = None
) -> pd.DataFrame:
    optional = optional or []
    cols = []
    missing_required = []
    for c in required:
        if c in df.columns:
            cols.append(c)
        else:
            missing_required.append(c)

    # required가 일부 없어도 "최대한" 뽑되, 누락은 경고만
    if missing_required:
        print(f"[WARN] missing required cols: {missing_required}")

    for c in optional:
        if c in df.columns and c not in cols:
            cols.append(c)

    return df.loc[:, cols].copy()


def save_both(df: pd.DataFrame, out_base: Path) -> tuple[Path, Path]:
    p_parquet = out_base.with_suffix(".parquet")
    p_csv = out_base.with_suffix(".csv")

    df.to_parquet(p_parquet, index=False)
    df.to_csv(p_csv, index=False, encoding="utf-8-sig")

    return p_parquet, p_csv


def print_small_df(df: pd.DataFrame, name: str, max_rows: int = 50) -> None:
    if df is None:
        return
    print("\n" + "=" * 80)
    print(f"[PRINT] {name} shape={df.shape}")
    print("=" * 80)
    if len(df) <= max_rows:
        print(df.to_string(index=False))
    else:
        # 큰 데이터는 콘솔 폭발 방지: head/tail만 보여주고, 전체는 파일로 저장
        print(df.head(10).to_string(index=False))
        print("...")
        print(df.tail(10).to_string(index=False))
        print(f"[INFO] full rows={len(df)} (saved to file, not fully printed)")


# =========================
# 2) CORE 피쳐 스펙 (A~G)
#    - 실제 컬럼명이 조금 달라도 후보군으로 흡수
# =========================
def build_core_extracts() -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}

    # ---- L0 (선택: 보고서 재현성/유니버스 근거) ----
    uni = load_parquet_if_exists("universe_k200_membership_monthly")
    if uni is not None:
        outputs["L0_universe_core"] = select_cols(
            uni, required=["date", "ticker"], optional=[]
        )

    # ---- L1 (선택: 데이터 소스 근거) ----
    ohlcv = load_parquet_if_exists("ohlcv_daily")
    if ohlcv is not None:
        outputs["L1_ohlcv_core"] = select_cols(
            ohlcv,
            required=["date", "ticker"],
            optional=["open", "high", "low", "close", "volume"],
        )

    # ---- L2 (선택: 펀더멘털 근거) ----
    fund = load_parquet_if_exists("fundamentals_annual")
    if fund is not None:
        outputs["L2_fundamentals_core"] = select_cols(
            fund,
            required=["date", "ticker"],
            optional=["net_income", "total_liabilities", "equity", "debt_ratio", "roe"],
        )

    # ---- L3 (선택: 머지 결과 근거 / 너무 크면 보고서엔 일부만 써도 됨) ----
    panel = load_parquet_if_exists("panel_merged_daily")
    if panel is not None:
        outputs["L3_panel_merged_core"] = select_cols(
            panel,
            required=["date", "ticker"],
            optional=[
                "close",
                "volume",
                "net_income",
                "total_liabilities",
                "equity",
                "debt_ratio",
                "roe",
                "ym",
                "in_universe",
            ],
        )

    # ---- L4 (B: 타깃 정의) ----
    ds = load_parquet_if_exists("dataset_daily")
    if ds is not None:
        # dataset_daily에는 phase가 없을 가능성이 높음(현재 로그 기준 없음)
        outputs["L4_dataset_targets_core"] = select_cols(
            ds, required=["date", "ticker"], optional=["ret_fwd_20d", "ret_fwd_120d"]
        )

    # ---- L4 (G: CV folds) ----
    cv_s = load_parquet_if_exists("cv_folds_short")
    if cv_s is not None:
        outputs["L4_cv_folds_short_core"] = select_cols(
            cv_s,
            required=[
                "fold_id",
                "segment",
                "train_start",
                "train_end",
                "test_start",
                "test_end",
            ],
            optional=[],
        )

    cv_l = load_parquet_if_exists("cv_folds_long")
    if cv_l is not None:
        outputs["L4_cv_folds_long_core"] = select_cols(
            cv_l,
            required=[
                "fold_id",
                "segment",
                "train_start",
                "train_end",
                "test_start",
                "test_end",
            ],
            optional=[],
        )

    # ---- L5 (C: 모델 OOS 성능) ----
    mm = load_parquet_if_exists("model_metrics")
    if mm is not None:
        outputs["L5_model_metrics_core"] = select_cols(
            mm,
            required=["horizon", "phase"],
            optional=[
                "ic_rank",
                "rmse",
                "mae",
                "hit_ratio",
                "n_features",
                "n_train",
                "n_test",
            ],
        )

    # ---- L6 (D: 스코어 산출/커버리지 품질) ----
    rss = load_parquet_if_exists("rebalance_scores_summary")
    if rss is not None:
        coverage_col = pick_first_existing_col(
            rss,
            [
                "coverage_vs_universe_pct",
                "coverage_ticker_pct",
                "coverage_vs_universe",
                "coverage_vs_universe_percent",
            ],
        )
        if coverage_col is None:
            print(
                "[WARN] rebalance_scores_summary: coverage column not found (expected coverage_vs_universe_pct or coverage_ticker_pct)"
            )

        base_cols = ["date", "phase", "n_tickers"]
        opt_cols = []
        if coverage_col:
            opt_cols.append(coverage_col)

        opt_cols += ["score_short_missing", "score_long_missing", "score_ens_missing"]

        outputs["L6_rebalance_scores_summary_core"] = select_cols(
            rss, required=base_cols, optional=opt_cols
        )

    # ---- L7 (E: 백테스트 요약 지표) ----
    bm = load_parquet_if_exists("bt_metrics")
    if bm is not None:
        outputs["L7_bt_metrics_core"] = select_cols(
            bm,
            required=["phase"],
            optional=[
                "top_k",
                "holding_days",
                "cost_bps",
                "buffer_k",
                "weighting",
                "net_total_return",
                "net_cagr",
                "net_sharpe",
                "net_mdd",
                "avg_turnover_oneway",
                "date_start",
                "date_end",
            ],
        )

    # ---- L7 (F-1: 리밸런싱별 수익률 시계열) ----
    br = load_parquet_if_exists("bt_returns")
    if br is not None:
        net_ret_col = pick_first_existing_col(
            br, ["net_return", "net_ret", "net_period_return"]
        )
        if net_ret_col is None:
            raise KeyError(
                "[F] bt_returns: net return column not found (net_return / net_ret / net_period_return)"
            )

        # 표준화된 이름으로 맞추고 저장(보고서/그림에서 쓰기 편하게)
        tmp = br.copy()
        if net_ret_col != "net_return":
            tmp["net_return"] = tmp[net_ret_col]
        outputs["L7_bt_returns_core"] = select_cols(
            tmp,
            required=["date", "phase", "net_return"],
            optional=["turnover_oneway", "n_tickers"],
        )

    # ---- L7 (F-2: equity curve + drawdown) ----
    be = load_parquet_if_exists("bt_equity_curve")
    if be is not None:
        outputs["L7_bt_equity_curve_core"] = select_cols(
            be, required=["date", "phase"], optional=["equity", "drawdown"]
        )

    return outputs


def main():
    outputs = build_core_extracts()

    # 1행 n열 목록(컬럼 목록) + 저장 로그(manifest)
    manifest_rows = []
    list_row = {}

    for name, df in outputs.items():
        out_base = OUT_DIR / name
        p_parquet, p_csv = save_both(df, out_base)

        cols_str = ", ".join(df.columns.tolist())
        list_row[name] = cols_str

        manifest_rows.append(
            {
                "artifact_core_name": name,
                "rows": int(df.shape[0]),
                "cols": int(df.shape[1]),
                "columns": cols_str,
                "saved_parquet": str(p_parquet),
                "saved_csv": str(p_csv),
            }
        )

        # 작은 결과만 콘솔에 “생략 없이” 출력 (큰 건 파일 저장이 목적)
        print_small_df(df, name=name, max_rows=50)

    # (요구) 1행 n열 목록
    one_row = pd.DataFrame([list_row])
    save_both(one_row, OUT_DIR / "_CORE_COLUMN_LIST_1row")
    print("\n" + "#" * 80)
    print("[DONE] 1-row N-columns CORE column list saved:")
    print(f" - {OUT_DIR / '_CORE_COLUMN_LIST_1row.parquet'}")
    print(f" - {OUT_DIR / '_CORE_COLUMN_LIST_1row.csv'}")
    print("#" * 80)

    # manifest 저장
    manifest = (
        pd.DataFrame(manifest_rows)
        .sort_values("artifact_core_name")
        .reset_index(drop=True)
    )
    save_both(manifest, OUT_DIR / "_MANIFEST_core_extracts")
    print("\n" + "#" * 80)
    print("[DONE] manifest saved:")
    print(f" - {OUT_DIR / '_MANIFEST_core_extracts.parquet'}")
    print(f" - {OUT_DIR / '_MANIFEST_core_extracts.csv'}")
    print("#" * 80)

    print("\n[OUT_DIR]", OUT_DIR)


if __name__ == "__main__":
    main()
