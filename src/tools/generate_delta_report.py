# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/generate_delta_report.py
"""
Delta 보고서 생성 스크립트
베이스라인과 현재 stage_tag의 KPI를 비교하여 Delta 보고서 생성
"""
import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# 핵심 KPI 목록 (요약 섹션 상단 고정)
CORE_KPIS = [
    "net_total_return",
    "net_sharpe",
    "net_mdd",
    "information_ratio",
    "tracking_error_ann",
    "avg_turnover_oneway",
    "ic_rank_mean",
]


def load_kpi_csv(csv_path: Path) -> pd.DataFrame:
    """KPI CSV 파일 로드"""
    if not csv_path.exists():
        raise FileNotFoundError(f"KPI CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig")

    # 필수 컬럼 확인
    required_cols = ["section", "metric", "dev_value", "holdout_value", "unit"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in KPI CSV: {missing_cols}")

    return df


def calculate_delta(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    베이스라인과 현재 KPI 비교하여 Delta 계산

    Returns:
        DataFrame with columns: section, metric, unit,
        baseline_dev_value, baseline_holdout_value,
        current_dev_value, current_holdout_value,
        dev_abs_diff, dev_pct_diff, dev_direction,
        holdout_abs_diff, holdout_pct_diff, holdout_direction
    """
    # metric 기준 outer join
    merged = pd.merge(
        baseline_df[["section", "metric", "dev_value", "holdout_value", "unit"]],
        current_df[["section", "metric", "dev_value", "holdout_value", "unit"]],
        on=["section", "metric"],
        how="outer",
        suffixes=("_baseline", "_current"),
    )

    # unit 통합 (baseline 우선, 없으면 current)
    merged["unit"] = merged["unit_baseline"].fillna(merged["unit_current"])

    # 값 추출
    baseline_dev = merged["dev_value_baseline"]
    baseline_holdout = merged["holdout_value_baseline"]
    current_dev = merged["dev_value_current"]
    current_holdout = merged["holdout_value_current"]

    # 숫자 값만 추출 (문자열/날짜는 제외)
    def _to_numeric_safe(series: pd.Series) -> pd.Series:
        """문자열/날짜를 제외하고 숫자만 변환"""
        result = pd.Series(index=series.index, dtype=float)
        for idx in series.index:
            val = series.loc[idx]
            if pd.isna(val):
                result.loc[idx] = np.nan
            elif isinstance(val, (int, float)):
                result.loc[idx] = float(val)
            elif isinstance(val, str):
                # 날짜 형식인지 확인
                try:
                    pd.to_datetime(val)
                    result.loc[idx] = np.nan  # 날짜는 제외
                except:
                    # 숫자 문자열인지 확인
                    try:
                        result.loc[idx] = float(val)
                    except:
                        result.loc[idx] = np.nan  # 변환 불가
            else:
                result.loc[idx] = np.nan
        return result

    baseline_dev_num = _to_numeric_safe(baseline_dev)
    baseline_holdout_num = _to_numeric_safe(baseline_holdout)
    current_dev_num = _to_numeric_safe(current_dev)
    current_holdout_num = _to_numeric_safe(current_holdout)

    # Dev Delta 계산 (숫자 값만)
    dev_abs_diff = current_dev_num - baseline_dev_num

    # pct_diff 계산 (baseline이 0이 아니고 숫자인 경우만)
    dev_pct_diff = pd.Series(index=merged.index, dtype=float)
    dev_pct_diff[:] = np.nan

    if baseline_dev_num.notna().any() and current_dev_num.notna().any():
        baseline_nonzero = baseline_dev_num != 0
        baseline_abs = baseline_dev_num.abs()
        # baseline의 절댓값이 충분히 큰 경우만 pct_diff 계산
        threshold = 1e-6
        valid_mask = (
            baseline_nonzero
            & (baseline_abs > threshold)
            & baseline_dev_num.notna()
            & current_dev_num.notna()
        )
        dev_pct_diff[valid_mask] = (
            dev_abs_diff[valid_mask] / baseline_abs[valid_mask]
        ) * 100

    # 방향 계산 (UP/DOWN/UNCHANGED) - 숫자 값만
    dev_direction = pd.Series(index=merged.index, dtype=str)
    dev_direction[:] = "UNCHANGED"

    valid_mask = baseline_dev_num.notna() & current_dev_num.notna()
    dev_direction[valid_mask & (dev_abs_diff > 1e-6)] = "UP"
    dev_direction[valid_mask & (dev_abs_diff < -1e-6)] = "DOWN"

    # Holdout Delta 계산 (숫자 값만)
    holdout_abs_diff = current_holdout_num - baseline_holdout_num

    holdout_pct_diff = pd.Series(index=merged.index, dtype=float)
    holdout_pct_diff[:] = np.nan

    if baseline_holdout_num.notna().any() and current_holdout_num.notna().any():
        baseline_nonzero = baseline_holdout_num != 0
        baseline_abs = baseline_holdout_num.abs()
        threshold = 1e-6
        valid_mask = (
            baseline_nonzero
            & (baseline_abs > threshold)
            & baseline_holdout_num.notna()
            & current_holdout_num.notna()
        )
        holdout_pct_diff[valid_mask] = (
            holdout_abs_diff[valid_mask] / baseline_abs[valid_mask]
        ) * 100

    holdout_direction = pd.Series(index=merged.index, dtype=str)
    holdout_direction[:] = "UNCHANGED"

    valid_mask = baseline_holdout_num.notna() & current_holdout_num.notna()
    holdout_direction[valid_mask & (holdout_abs_diff > 1e-6)] = "UP"
    holdout_direction[valid_mask & (holdout_abs_diff < -1e-6)] = "DOWN"

    # 결과 DataFrame 구성
    result = pd.DataFrame(
        {
            "section": merged["section"],
            "metric": merged["metric"],
            "unit": merged["unit"],
            "baseline_dev_value": baseline_dev,
            "baseline_holdout_value": baseline_holdout,
            "current_dev_value": current_dev,
            "current_holdout_value": current_holdout,
            "dev_abs_diff": dev_abs_diff,
            "dev_pct_diff": dev_pct_diff,
            "dev_direction": dev_direction,
            "holdout_abs_diff": holdout_abs_diff,
            "holdout_pct_diff": holdout_pct_diff,
            "holdout_direction": holdout_direction,
        }
    )

    return result


def format_delta_value(val: Any, unit: str, is_diff: bool = False) -> str:
    """Delta 값 포맷팅"""
    if pd.isna(val) or val is None:
        return "N/A"

    # 문자열인 경우 그대로 반환 (날짜 등)
    if isinstance(val, str):
        return val

    # 숫자로 변환 시도
    try:
        num_val = float(val)
    except (ValueError, TypeError):
        return str(val)

    if unit == "%":
        return f"{num_val:.2f}"
    elif unit == "bps":
        return f"{num_val:.1f}"
    elif unit == "ratio":
        return f"{num_val:.4f}"
    elif unit == "count":
        if num_val == int(num_val):
            return f"{int(num_val)}"
        return f"{num_val:.1f}"
    else:
        return f"{num_val:.4f}"


def generate_markdown_report(
    delta_df: pd.DataFrame,
    baseline_tag: str,
    current_tag: str,
) -> str:
    """Markdown 형식의 Delta 보고서 생성"""
    lines = [
        f"# Delta Report: {baseline_tag} vs {current_tag}",
        "",
        f"**베이스라인**: `{baseline_tag}`",
        f"**현재 단계**: `{current_tag}`",
        "",
        "---",
        "",
    ]

    # 핵심 KPI 요약 섹션
    lines.append("## 핵심 KPI 요약")
    lines.append("")

    core_rows = []
    for metric in CORE_KPIS:
        row = delta_df[delta_df["metric"] == metric]
        if not row.empty:
            core_rows.append(row.iloc[0])

    if core_rows:
        lines.append(
            "| Metric | Dev (Baseline → Current) | Holdout (Baseline → Current) | Unit |"
        )
        lines.append("|---|---|---|---|")

        for row in core_rows:
            metric = row["metric"]
            unit = row["unit"]

            # Dev
            baseline_dev = format_delta_value(row["baseline_dev_value"], unit)
            current_dev = format_delta_value(row["current_dev_value"], unit)
            dev_diff = format_delta_value(row["dev_abs_diff"], unit, is_diff=True)
            dev_dir = row["dev_direction"]
            dev_pct = (
                format_delta_value(row["dev_pct_diff"], "%")
                if pd.notna(row["dev_pct_diff"])
                else ""
            )

            dev_str = f"{baseline_dev} → {current_dev}"
            if dev_dir != "UNCHANGED":
                dev_str += f" ({dev_dir} {dev_diff}"
                if dev_pct:
                    dev_str += f", {dev_pct}%"
                dev_str += ")"

            # Holdout
            baseline_holdout = format_delta_value(row["baseline_holdout_value"], unit)
            current_holdout = format_delta_value(row["current_holdout_value"], unit)
            holdout_diff = format_delta_value(
                row["holdout_abs_diff"], unit, is_diff=True
            )
            holdout_dir = row["holdout_direction"]
            holdout_pct = (
                format_delta_value(row["holdout_pct_diff"], "%")
                if pd.notna(row["holdout_pct_diff"])
                else ""
            )

            holdout_str = f"{baseline_holdout} → {current_holdout}"
            if holdout_dir != "UNCHANGED":
                holdout_str += f" ({holdout_dir} {holdout_diff}"
                if holdout_pct:
                    holdout_str += f", {holdout_pct}%"
                holdout_str += ")"

            lines.append(f"| {metric} | {dev_str} | {holdout_str} | {unit} |")

        lines.append("")

    # 섹션별 상세 비교
    sections = ["DATA", "MODEL", "BACKTEST", "BENCHMARK", "STABILITY", "SETTINGS"]

    for section in sections:
        section_df = delta_df[delta_df["section"] == section].copy()

        if section_df.empty:
            continue

        # 핵심 KPI는 이미 요약에 포함했으므로 제외
        section_df = section_df[~section_df["metric"].isin(CORE_KPIS)]

        if section_df.empty:
            continue

        lines.append(f"## {section}")
        lines.append("")

        lines.append(
            "| Metric | Dev (Baseline → Current) | Holdout (Baseline → Current) | Unit |"
        )
        lines.append("|---|---|---|---|")

        for _, row in section_df.iterrows():
            metric = row["metric"]
            unit = row["unit"]

            # Dev
            baseline_dev = format_delta_value(row["baseline_dev_value"], unit)
            current_dev = format_delta_value(row["current_dev_value"], unit)
            dev_diff = format_delta_value(row["dev_abs_diff"], unit, is_diff=True)
            dev_dir = row["dev_direction"]
            dev_pct = (
                format_delta_value(row["dev_pct_diff"], "%")
                if pd.notna(row["dev_pct_diff"])
                else ""
            )

            dev_str = f"{baseline_dev} → {current_dev}"
            if dev_dir != "UNCHANGED":
                dev_str += f" ({dev_dir} {dev_diff}"
                if dev_pct:
                    dev_str += f", {dev_pct}%"
                dev_str += ")"

            # Holdout
            baseline_holdout = format_delta_value(row["baseline_holdout_value"], unit)
            current_holdout = format_delta_value(row["current_holdout_value"], unit)
            holdout_diff = format_delta_value(
                row["holdout_abs_diff"], unit, is_diff=True
            )
            holdout_dir = row["holdout_direction"]
            holdout_pct = (
                format_delta_value(row["holdout_pct_diff"], "%")
                if pd.notna(row["holdout_pct_diff"])
                else ""
            )

            holdout_str = f"{baseline_holdout} → {current_holdout}"
            if holdout_dir != "UNCHANGED":
                holdout_str += f" ({holdout_dir} {holdout_diff}"
                if holdout_pct:
                    holdout_str += f", {holdout_pct}%"
                holdout_str += ")"

            lines.append(f"| {metric} | {dev_str} | {holdout_str} | {unit} |")

        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Delta Report comparing baseline vs current stage"
    )
    parser.add_argument(
        "--baseline-tag",
        type=str,
        required=True,
        help="Baseline tag (e.g., baseline_prerefresh_20251219_143636)",
    )
    parser.add_argument(
        "--current-tag",
        type=str,
        required=True,
        help="Current stage tag (e.g., stage0_repro_fix_20251219_143636)",
    )
    parser.add_argument("--root", type=str, default=None, help="Project root directory")
    parser.add_argument(
        "--kpi-dir", type=str, default="reports/kpi", help="KPI directory"
    )
    parser.add_argument(
        "--delta-dir",
        type=str,
        default="reports/delta",
        help="Delta report output directory",
    )
    args = parser.parse_args()

    # 루트 경로 결정
    if args.root:
        root = Path(args.root)
    else:
        root = Path(__file__).resolve().parents[2]

    kpi_dir = root / args.kpi_dir
    delta_dir = root / args.delta_dir
    delta_dir.mkdir(parents=True, exist_ok=True)

    # KPI CSV 파일 경로
    baseline_kpi_csv = kpi_dir / f"kpi_table__{args.baseline_tag}.csv"
    current_kpi_csv = kpi_dir / f"kpi_table__{args.current_tag}.csv"

    print(f"[Delta Report] Baseline: {baseline_kpi_csv}")
    print(f"[Delta Report] Current: {current_kpi_csv}")

    # KPI 로드
    baseline_df = load_kpi_csv(baseline_kpi_csv)
    current_df = load_kpi_csv(current_kpi_csv)

    print(f"[Delta Report] Baseline KPIs: {len(baseline_df)}")
    print(f"[Delta Report] Current KPIs: {len(current_df)}")

    # Delta 계산
    delta_df = calculate_delta(baseline_df, current_df)

    print(f"[Delta Report] Delta KPIs: {len(delta_df)}")

    # CSV 저장
    csv_path = delta_dir / f"delta_kpi__{args.baseline_tag}__vs__{args.current_tag}.csv"
    delta_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[Delta Report] CSV saved: {csv_path}")

    # Markdown 저장
    md_path = (
        delta_dir / f"delta_report__{args.baseline_tag}__vs__{args.current_tag}.md"
    )
    md_content = generate_markdown_report(delta_df, args.baseline_tag, args.current_tag)
    md_path.write_text(md_content, encoding="utf-8")
    print(f"[Delta Report] Markdown saved: {md_path}")

    # 핵심 KPI 요약 출력
    print("\n=== 핵심 KPI 변화 요약 ===")
    for metric in CORE_KPIS:
        row = delta_df[delta_df["metric"] == metric]
        if not row.empty:
            r = row.iloc[0]
            print(f"{metric}:")
            if pd.notna(r["baseline_dev_value"]) and pd.notna(r["current_dev_value"]):
                baseline_str = format_delta_value(
                    r["baseline_dev_value"], r.get("unit", "")
                )
                current_str = format_delta_value(
                    r["current_dev_value"], r.get("unit", "")
                )
                print(f"  Dev: {baseline_str} -> {current_str} ({r['dev_direction']})")
            if pd.notna(r["baseline_holdout_value"]) and pd.notna(
                r["current_holdout_value"]
            ):
                baseline_str = format_delta_value(
                    r["baseline_holdout_value"], r.get("unit", "")
                )
                current_str = format_delta_value(
                    r["current_holdout_value"], r.get("unit", "")
                )
                print(
                    f"  Holdout: {baseline_str} -> {current_str} ({r['holdout_direction']})"
                )

    print("\n[Delta Report] Completed.")


if __name__ == "__main__":
    main()
