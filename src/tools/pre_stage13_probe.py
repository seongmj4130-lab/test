# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/pre_stage13_probe.py
"""
[Stage13 사전 분석] L7 백테스트 실행 전 병목 분석

목표:
- Stage13(L7) 실행이 오래 걸리는 근본 병목을 찾기 위해,
  Stage13 실행 전에 "필수 사전 정보"를 빠르게 산출한다.
- 전체 백테스트를 돌리지 않고도, L7이 어떤 입력을 쓰고 무엇을 다시 계산하는지,
  그리고 어디서 시간이 많이 걸릴지(데이터 크기/조인/검증/루프)를 정량으로 확인한다.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

# 프로젝트 루트 경로 고정
BASE_DIR = Path(r"C:\Users\seong\OneDrive\Desktop\bootcamp\03_code")
sys.path.insert(0, str(BASE_DIR / "src"))


def analyze_l7_code_scan(base_dir: Path) -> dict:
    """
    [1] L7가 "정말로" 무엇을 수행하는지 확인 (코드 스캔)
    """
    print("[1] L7 코드 스캔 중...")

    l7_backtest_file = base_dir / "src" / "stages" / "l7_backtest.py"
    run_all_file = base_dir / "src" / "run_all.py"

    results = {
        "entry_function": "run_backtest (src/stages/l7_backtest.py)",
        "call_path": "run_all.py::run_L7_backtest() → stages.l7_backtest.run_backtest()",
        "inputs": {
            "required": [
                "rebalance_scores (pd.DataFrame)",
                "cfg (BacktestConfig)",
            ],
            "optional": [
                "market_regime (pd.DataFrame, optional)",
                "config_cost_bps (float, optional)",
            ],
        },
        "outputs": [
            "bt_positions (pd.DataFrame)",
            "bt_returns_core (pd.DataFrame)",
            "bt_equity_curve (pd.DataFrame)",
            "bt_metrics (pd.DataFrame)",
            "quality (dict)",
            "warns (List[str])",
            "selection_diagnostics (pd.DataFrame)",
            "bt_returns_diagnostics (pd.DataFrame)",
        ],
        "main_loops": [
            "for phase, dphase in df_sorted.groupby(phase_col)",
            "for dt, g in dphase.groupby(date_col)",
        ],
        "validation_points": [
            "validate_df() with max_missing_pct=95.0 (run_all.py:1168)",
            "Required cols: ['date', 'phase'] for bt_returns",
        ],
    }

    # 코드에서 실제 루프 확인
    if l7_backtest_file.exists():
        content = l7_backtest_file.read_text(encoding="utf-8")
        if "for phase, dphase in df_sorted.groupby" in content:
            results["main_loops"].append("Phase별 그룹화 루프")
        if "for dt, g in dphase.groupby(date_col)" in content:
            results["main_loops"].append("날짜별 그룹화 루프")

    return results


def measure_input_files(
    base_dir: Path, pipeline_tag: str, global_tag: str
) -> pd.DataFrame:
    """
    [2] Stage13 실행에 필요한 "입력 파일 존재/크기/행수"만 빠르게 측정
    """
    print("[2] 입력 파일 존재/크기/행수 측정 중...")

    base_interim_dir = base_dir / "data" / "interim"
    pipeline_dir = base_interim_dir / pipeline_tag
    global_dir = base_interim_dir / global_tag

    files_to_check = [
        # Pipeline baseline (stage6)
        (pipeline_dir / "rebalance_scores.parquet", "rebalance_scores", "pipeline"),
        (
            pipeline_dir / "rebalance_scores_summary.parquet",
            "rebalance_scores_summary",
            "pipeline",
        ),
        (pipeline_dir / "ohlcv_daily.parquet", "ohlcv_daily", "pipeline"),
        (pipeline_dir / "panel_merged_daily.parquet", "panel_merged_daily", "pipeline"),
        # L2 재사용
        (base_interim_dir / "fundamentals_annual.parquet", "fundamentals_annual", "L2"),
        # Global baseline (stage12) - market_regime
        (global_dir / "market_regime.parquet", "market_regime", "global"),
        (global_dir / "market_regime.csv", "market_regime", "global"),
    ]

    rows = []
    for file_path, name, source in files_to_check:
        row = {
            "file_name": name,
            "source": source,
            "path": str(file_path),
            "exists": False,
            "file_size_bytes": None,
            "file_size_mb": None,
            "rows": None,
            "cols": None,
            "date_min": None,
            "date_max": None,
        }

        if file_path.exists():
            row["exists"] = True
            row["file_size_bytes"] = file_path.stat().st_size
            row["file_size_mb"] = round(row["file_size_bytes"] / (1024 * 1024), 2)

            try:
                # Parquet 메타데이터로 빠르게 행수 추정
                if file_path.suffix == ".parquet":
                    parquet_file = pq.ParquetFile(file_path)
                    row["rows"] = parquet_file.metadata.num_rows
                    row["cols"] = len(parquet_file.schema)

                    # 날짜 범위 확인 (date 컬럼이 있으면)
                    try:
                        df_sample = pd.read_parquet(
                            file_path,
                            columns=(
                                ["date"] if "date" in parquet_file.schema.names else []
                            ),
                        )
                        if "date" in df_sample.columns:
                            df_sample["date"] = pd.to_datetime(df_sample["date"])
                            row["date_min"] = str(df_sample["date"].min())
                            row["date_max"] = str(df_sample["date"].max())
                    except Exception:
                        pass
                elif file_path.suffix == ".csv":
                    # CSV는 헤더만 읽어서 컬럼 수 확인
                    df_head = pd.read_csv(file_path, nrows=0)
                    row["cols"] = len(df_head.columns)
                    # 행수는 대략 추정 (파일 크기 기반)
                    # 정확하지 않지만 빠른 추정
                    try:
                        df_sample = pd.read_csv(file_path, nrows=1000)
                        if "date" in df_sample.columns:
                            df_sample["date"] = pd.to_datetime(df_sample["date"])
                            row["date_min"] = str(df_sample["date"].min())
                            row["date_max"] = str(df_sample["date"].max())
                    except Exception:
                        pass
            except Exception as e:
                print(f"  [WARN] {name} 읽기 실패: {e}")

        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def profile_rebalance_scores(base_dir: Path, pipeline_tag: str) -> dict:
    """
    [3] rebalance_scores 품질 점검 (L7 실행시간 결정 1순위)
    """
    print("[3] rebalance_scores 품질 점검 중...")

    rebalance_scores_path = (
        base_dir / "data" / "interim" / pipeline_tag / "rebalance_scores.parquet"
    )

    if not rebalance_scores_path.exists():
        return {
            "error": f"rebalance_scores not found: {rebalance_scores_path}",
            "rows": None,
            "cols": None,
        }

    # 최소한의 컬럼만 로드하여 빠르게 분석
    try:
        # 먼저 스키마 확인
        parquet_file = pq.ParquetFile(rebalance_scores_path)
        all_cols = [field.name for field in parquet_file.schema]

        # 필수 컬럼만 로드
        essential_cols = ["date", "ticker", "phase"]
        if "score_ens" in all_cols:
            essential_cols.append("score_ens")
        if "true_short" in all_cols:
            essential_cols.append("true_short")
        if "sector_name" in all_cols:
            essential_cols.append("sector_name")

        df = pd.read_parquet(rebalance_scores_path, columns=essential_cols)

        # 날짜 변환
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        # 리밸런싱 날짜 수
        n_rebalances = df["date"].nunique() if "date" in df.columns else None

        # 각 리밸런싱당 종목 수 분포
        if "date" in df.columns:
            tickers_per_rebalance = df.groupby("date")["ticker"].nunique()
            ticker_stats = {
                "mean": float(tickers_per_rebalance.mean()),
                "median": float(tickers_per_rebalance.median()),
                "min": int(tickers_per_rebalance.min()),
                "max": int(tickers_per_rebalance.max()),
                "p95": float(tickers_per_rebalance.quantile(0.95)),
            }
        else:
            ticker_stats = None

        # Null 비율 (상위 컬럼만)
        null_pct = {}
        for col in essential_cols:
            if col in df.columns:
                null_pct[col] = float(df[col].isna().mean() * 100)

        return {
            "rows": len(df),
            "cols": len(df.columns),
            "all_columns": all_cols,
            "n_rebalances": int(n_rebalances) if n_rebalances else None,
            "tickers_per_rebalance": ticker_stats,
            "null_pct": null_pct,
            "date_range": {
                "min": str(df["date"].min()) if "date" in df.columns else None,
                "max": str(df["date"].max()) if "date" in df.columns else None,
            },
        }
    except Exception as e:
        return {
            "error": str(e),
            "rows": None,
            "cols": None,
        }


def check_regime_coverage(base_dir: Path, pipeline_tag: str, global_tag: str) -> dict:
    """
    [4] "regime 컬럼 결측 95% 문제" 재발 가능성 사전 판정
    """
    print("[4] regime 컬럼 결측 문제 재발 가능성 판정 중...")

    base_interim_dir = base_dir / "data" / "interim"
    pipeline_dir = base_interim_dir / pipeline_tag
    global_dir = base_interim_dir / global_tag

    # rebalance_scores의 날짜 범위 확인
    rebalance_scores_path = pipeline_dir / "rebalance_scores.parquet"
    if not rebalance_scores_path.exists():
        return {
            "error": "rebalance_scores not found",
            "coverage_pct": None,
        }

    try:
        # rebalance_scores의 날짜만 로드
        df_rebalance = pd.read_parquet(rebalance_scores_path, columns=["date", "phase"])
        df_rebalance["date"] = pd.to_datetime(df_rebalance["date"])
        rebalance_dates = set(df_rebalance["date"].unique())
        rebalance_date_min = df_rebalance["date"].min()
        rebalance_date_max = df_rebalance["date"].max()
    except Exception as e:
        return {
            "error": f"Failed to load rebalance_scores: {e}",
            "coverage_pct": None,
        }

    # market_regime 찾기
    regime_paths = [
        global_dir / "market_regime.parquet",
        global_dir / "market_regime.csv",
        base_interim_dir / "market_regime.parquet",
        base_interim_dir / "market_regime.csv",
    ]

    regime_path = None
    for path in regime_paths:
        if path.exists():
            regime_path = path
            break

    if not regime_path:
        return {
            "error": "market_regime not found in any expected location",
            "coverage_pct": 0.0,
            "rebalance_date_range": {
                "min": str(rebalance_date_min),
                "max": str(rebalance_date_max),
            },
        }

    try:
        # market_regime 로드
        if regime_path.suffix == ".parquet":
            df_regime = pd.read_parquet(regime_path, columns=["date", "regime"])
        else:
            df_regime = pd.read_csv(regime_path, usecols=["date", "regime"])

        df_regime["date"] = pd.to_datetime(df_regime["date"])
        regime_dates = set(df_regime["date"].unique())
        regime_date_min = df_regime["date"].min()
        regime_date_max = df_regime["date"].max()

        # 겹침 계산
        overlap_dates = rebalance_dates & regime_dates
        coverage_pct = (
            (len(overlap_dates) / len(rebalance_dates) * 100)
            if rebalance_dates
            else 0.0
        )

        return {
            "regime_source_path": str(regime_path),
            "rebalance_date_range": {
                "min": str(rebalance_date_min),
                "max": str(rebalance_date_max),
                "count": len(rebalance_dates),
            },
            "regime_date_range": {
                "min": str(regime_date_min),
                "max": str(regime_date_max),
                "count": len(regime_dates),
            },
            "overlap_dates_count": len(overlap_dates),
            "coverage_pct": round(coverage_pct, 2),
            "risk_assessment": (
                "HIGH"
                if coverage_pct < 50
                else "LOW"
                if coverage_pct >= 90
                else "MEDIUM"
            ),
        }
    except Exception as e:
        return {
            "error": f"Failed to load market_regime: {e}",
            "coverage_pct": None,
        }


def analyze_min_run_plan(base_dir: Path) -> dict:
    """
    [5] Stage13 시간을 줄일 수 있는 "최소 실행 플랜" 후보 도출
    """
    print("[5] 최소 실행 플랜 분석 중...")

    l7_backtest_file = base_dir / "src" / "stages" / "l7_backtest.py"
    run_all_file = base_dir / "src" / "run_all.py"

    results = {
        "A_bt_returns_column_cleanup_only": {
            "possible": False,
            "reason": "bt_returns는 run_backtest()에서 전체 재계산되며, 컬럼 정리만 따로 할 수 없음",
            "code_location": "l7_backtest.py:490-514",
        },
        "B_date_range_limit": {
            "possible": False,
            "reason": "run_backtest()에 날짜 범위 제한 옵션이 없음. rebalance_scores를 필터링해야 함",
            "code_location": "l7_backtest.py:243",
        },
        "C_rebalance_subset": {
            "possible": True,
            "reason": "rebalance_scores를 필터링하여 특정 날짜만 포함하면 가능",
            "code_location": "run_all.py:510 (rebalance_scores 전달 전 필터링 가능)",
        },
        "D_validate_only": {
            "possible": False,
            "reason": "validate는 run_all.py에서 출력 저장 후 수행되므로, 백테스트 실행 없이는 불가능",
            "code_location": "run_all.py:1164-1174",
        },
    }

    # 코드 확인
    if l7_backtest_file.exists():
        content = l7_backtest_file.read_text(encoding="utf-8")
        if "def run_backtest" in content:
            # 날짜 필터링 옵션 확인
            if (
                "date_range" not in content
                and "start_date" not in content
                and "end_date" not in content
            ):
                results["B_date_range_limit"]["possible"] = False
            else:
                results["B_date_range_limit"]["possible"] = True

    return results


def save_scan_report(base_dir: Path, scan_results: dict):
    """[1] L7 코드 스캔 결과 저장"""
    output_path = base_dir / "reports" / "analysis" / "pre_stage13_scan.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    md_content = f"""# Stage13 사전 분석: L7 코드 스캔

## 1. L7 실행 함수명 / 호출 경로

- **엔트리 함수**: {scan_results['entry_function']}
- **호출 경로**: {scan_results['call_path']}

## 2. L7 입력 아티팩트 목록

### 필수 입력
{chr(10).join(f"- {inp}" for inp in scan_results['inputs']['required'])}

### 옵션 입력
{chr(10).join(f"- {inp}" for inp in scan_results['inputs']['optional'])}

## 3. L7 출력 아티팩트 목록

{chr(10).join(f"- {out}" for out in scan_results['outputs'])}

## 4. L7에서 "기간 전체 루프"가 도는 지점

{chr(10).join(f"- {loop}" for loop in scan_results['main_loops'])}

## 5. Validation 지점

{chr(10).join(f"- {val}" for val in scan_results['validation_points'])}
"""
    output_path.write_text(md_content, encoding="utf-8")
    print(f"  [OK] 저장 완료: {output_path}")


def save_inputs_report(base_dir: Path, df_inputs: pd.DataFrame):
    """[2] 입력 파일 존재/크기/행수 저장"""
    output_path = base_dir / "reports" / "analysis" / "pre_stage13_inputs.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_inputs.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"  [OK] 저장 완료: {output_path}")


def save_rebalance_profile(base_dir: Path, profile: dict):
    """[3] rebalance_scores 품질 점검 결과 저장"""
    output_path = (
        base_dir / "reports" / "analysis" / "pre_stage13_rebalance_scores_profile.md"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if "error" in profile:
        md_content = f"# Stage13 사전 분석: rebalance_scores 품질 점검\n\n## 오류\n\n{profile['error']}\n"
    else:
        ticker_stats_str = ""
        if profile.get("tickers_per_rebalance"):
            stats = profile["tickers_per_rebalance"]
            ticker_stats_str = f"""
- 평균: {stats['mean']:.1f}개
- 중앙값: {stats['median']:.1f}개
- 최소: {stats['min']}개
- 최대: {stats['max']}개
- P95: {stats['p95']:.1f}개
"""
        null_pct_str = "\n".join(
            f"- {col}: {pct:.2f}%" for col, pct in profile.get("null_pct", {}).items()
        )

        md_content = f"""# Stage13 사전 분석: rebalance_scores 품질 점검

## 기본 정보

- **행 수**: {profile.get('rows', 'N/A'):,}
- **컬럼 수**: {profile.get('cols', 'N/A')}
- **리밸런싱 날짜 수**: {profile.get('n_rebalances', 'N/A'):,}

## 날짜 범위

- **최소**: {profile.get('date_range', {}).get('min', 'N/A')}
- **최대**: {profile.get('date_range', {}).get('max', 'N/A')}

## 각 리밸런싱당 종목 수 분포

{ticker_stats_str}

## 상위 컬럼 Null 비율

{null_pct_str}

## 전체 컬럼 목록

{', '.join(profile.get('all_columns', []))}
"""
    output_path.write_text(md_content, encoding="utf-8")
    print(f"  [OK] 저장 완료: {output_path}")


def save_regime_coverage(base_dir: Path, coverage: dict):
    """[4] regime 컬럼 결측 문제 재발 가능성 판정 결과 저장"""
    output_path = base_dir / "reports" / "analysis" / "pre_stage13_regime_coverage.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if "error" in coverage:
        md_content = f"# Stage13 사전 분석: regime 컬럼 결측 문제 재발 가능성 판정\n\n## 오류\n\n{coverage['error']}\n"
    else:
        risk_msg = ""
        if coverage.get("coverage_pct", 0) < 50:
            risk_msg = "\n\n**⚠️ 경고**: Coverage가 50% 미만입니다. Stage13에서 bt_returns에 regime 포함은 원천적으로 위험합니다."

        md_content = f"""# Stage13 사전 분석: regime 컬럼 결측 문제 재발 가능성 판정

## Regime 소스

- **경로**: {coverage.get('regime_source_path', 'N/A')}

## Rebalance 날짜 범위

- **최소**: {coverage.get('rebalance_date_range', {}).get('min', 'N/A')}
- **최대**: {coverage.get('rebalance_date_range', {}).get('max', 'N/A')}
- **날짜 수**: {coverage.get('rebalance_date_range', {}).get('count', 'N/A'):,}

## Regime 날짜 범위

- **최소**: {coverage.get('regime_date_range', {}).get('min', 'N/A')}
- **최대**: {coverage.get('regime_date_range', {}).get('max', 'N/A')}
- **날짜 수**: {coverage.get('regime_date_range', {}).get('count', 'N/A'):,}

## Coverage 분석

- **겹치는 날짜 수**: {coverage.get('overlap_dates_count', 'N/A'):,}
- **Coverage 비율**: {coverage.get('coverage_pct', 'N/A')}%
- **위험도 평가**: {coverage.get('risk_assessment', 'N/A')}
{risk_msg}
"""
    output_path.write_text(md_content, encoding="utf-8")
    print(f"  [OK] 저장 완료: {output_path}")


def save_min_run_plan(base_dir: Path, plan: dict):
    """[5] 최소 실행 플랜 후보 저장"""
    output_path = base_dir / "reports" / "analysis" / "pre_stage13_min_run_plan.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plan_items = []
    for key, value in plan.items():
        status = "✅ 가능" if value["possible"] else "❌ 불가능"
        plan_items.append(
            f"""
## {key}

**가능 여부**: {status}

**이유**: {value['reason']}

**코드 위치**: {value['code_location']}
"""
        )

    md_content = f"""# Stage13 사전 분석: 최소 실행 플랜 후보

{''.join(plan_items)}
"""
    output_path.write_text(md_content, encoding="utf-8")
    print(f"  [OK] 저장 완료: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Stage13 사전 분석 스크립트")
    parser.add_argument(
        "--pipeline-tag",
        type=str,
        required=True,
        help="Pipeline baseline 태그 (예: stage6_sector_relative_feature_balance_20251220_194928)",
    )
    parser.add_argument(
        "--global-tag",
        type=str,
        required=True,
        help="Global baseline 태그 (예: stage12_final_export_20251221_013411)",
    )
    args = parser.parse_args()

    base_dir = BASE_DIR

    print("=" * 60)
    print("Stage13 사전 분석 시작")
    print("=" * 60)
    print(f"Pipeline baseline: {args.pipeline_tag}")
    print(f"Global baseline: {args.global_tag}")
    print()

    # [1] L7 코드 스캔
    scan_results = analyze_l7_code_scan(base_dir)
    save_scan_report(base_dir, scan_results)

    # [2] 입력 파일 측정
    df_inputs = measure_input_files(base_dir, args.pipeline_tag, args.global_tag)
    save_inputs_report(base_dir, df_inputs)

    # [3] rebalance_scores 품질 점검
    profile = profile_rebalance_scores(base_dir, args.pipeline_tag)
    save_rebalance_profile(base_dir, profile)

    # [4] regime coverage 확인
    coverage = check_regime_coverage(base_dir, args.pipeline_tag, args.global_tag)
    save_regime_coverage(base_dir, coverage)

    # [5] 최소 실행 플랜 분석
    plan = analyze_min_run_plan(base_dir)
    save_min_run_plan(base_dir, plan)

    print()
    print("=" * 60)
    print("Stage13 사전 분석 완료")
    print("=" * 60)
    print("\n생성된 리포트:")
    print("  - reports/analysis/pre_stage13_scan.md")
    print("  - reports/analysis/pre_stage13_inputs.csv")
    print("  - reports/analysis/pre_stage13_rebalance_scores_profile.md")
    print("  - reports/analysis/pre_stage13_regime_coverage.md")
    print("  - reports/analysis/pre_stage13_min_run_plan.md")


if __name__ == "__main__":
    main()
