"""
백테스트 지표 표시 스크립트
"""
from pathlib import Path

import pandas as pd


def format_pct(value):
    """퍼센트 포맷팅"""
    return f"{value*100:.2f}%" if pd.notna(value) else "N/A"


def format_number(value, decimals=4):
    """숫자 포맷팅"""
    return f"{value:.{decimals}f}" if pd.notna(value) else "N/A"


def show_backtest_metrics():
    """백테스트 지표 표시"""
    strategies = ["bt20_short", "bt20_pro", "bt20_ens", "bt120_long", "bt120_ens"]
    interim_dir = Path("data/interim")

    all_metrics = []
    for strategy in strategies:
        metrics_path = interim_dir / f"bt_metrics_{strategy}.parquet"
        if metrics_path.exists():
            df = pd.read_parquet(metrics_path)
            df["strategy"] = strategy
            all_metrics.append(df)

    if not all_metrics:
        print("백테스트 지표 파일을 찾을 수 없습니다.")
        return

    df = pd.concat(all_metrics, ignore_index=True)

    print("\n" + "=" * 100)
    print("백테스트 지표 요약")
    print("=" * 100)

    # 핵심 성과 지표
    print("\n[1. 핵심 성과 지표 (Headline Metrics)]")
    print("-" * 100)

    core_cols = [
        "strategy",
        "phase",
        "net_sharpe",
        "net_total_return",
        "net_cagr",
        "net_mdd",
        "net_calmar_ratio",
    ]
    core_df = df[core_cols].copy()

    # 포맷팅
    display_df = core_df.copy()
    display_df["net_sharpe"] = display_df["net_sharpe"].apply(
        lambda x: format_number(x, 4)
    )
    display_df["net_total_return"] = display_df["net_total_return"].apply(
        lambda x: format_pct(x)
    )
    display_df["net_cagr"] = display_df["net_cagr"].apply(lambda x: format_pct(x))
    display_df["net_mdd"] = display_df["net_mdd"].apply(lambda x: format_pct(x))
    display_df["net_calmar_ratio"] = display_df["net_calmar_ratio"].apply(
        lambda x: format_number(x, 4)
    )

    print(display_df.to_string(index=False))

    # 운용 안정성 지표
    print("\n[2. 운용 안정성 지표 (Operational Viability)]")
    print("-" * 100)

    ops_cols = [
        "strategy",
        "phase",
        "avg_turnover_oneway",
        "net_hit_ratio",
        "net_profit_factor",
        "avg_trade_duration",
        "avg_n_tickers",
        "n_rebalances",
    ]
    ops_df = df[ops_cols].copy()

    display_ops = ops_df.copy()
    display_ops["avg_turnover_oneway"] = display_ops["avg_turnover_oneway"].apply(
        lambda x: format_pct(x)
    )
    display_ops["net_hit_ratio"] = display_ops["net_hit_ratio"].apply(
        lambda x: format_pct(x)
    )
    display_ops["net_profit_factor"] = display_ops["net_profit_factor"].apply(
        lambda x: format_number(x, 4)
    )
    display_ops["avg_trade_duration"] = display_ops["avg_trade_duration"].apply(
        lambda x: format_number(x, 2)
    )
    display_ops["avg_n_tickers"] = display_ops["avg_n_tickers"].apply(
        lambda x: format_number(x, 1)
    )

    print(display_ops.to_string(index=False))

    # 전략별 비교 (Holdout 구간)
    print("\n[3. 전략별 비교 (Holdout 구간)]")
    print("-" * 100)

    holdout_df = df[df["phase"] == "holdout"].copy()
    if len(holdout_df) > 0:
        comparison_cols = [
            "strategy",
            "net_sharpe",
            "net_cagr",
            "net_mdd",
            "net_calmar_ratio",
            "net_hit_ratio",
            "net_profit_factor",
        ]
        comp_df = holdout_df[comparison_cols].copy()

        display_comp = comp_df.copy()
        display_comp["net_sharpe"] = display_comp["net_sharpe"].apply(
            lambda x: format_number(x, 4)
        )
        display_comp["net_cagr"] = display_comp["net_cagr"].apply(
            lambda x: format_pct(x)
        )
        display_comp["net_mdd"] = display_comp["net_mdd"].apply(lambda x: format_pct(x))
        display_comp["net_calmar_ratio"] = display_comp["net_calmar_ratio"].apply(
            lambda x: format_number(x, 4)
        )
        display_comp["net_hit_ratio"] = display_comp["net_hit_ratio"].apply(
            lambda x: format_pct(x)
        )
        display_comp["net_profit_factor"] = display_comp["net_profit_factor"].apply(
            lambda x: format_number(x, 4)
        )

        print(display_comp.to_string(index=False))

    print("\n" + "=" * 100)
    print("완료")
    print("=" * 100)


if __name__ == "__main__":
    show_backtest_metrics()
