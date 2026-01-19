# src/tools/update_history_manifest.py
"""
History Manifest 업데이트 스크립트 (PPT급 컬럼 포함)
Stage 완료(PASS) 때마다 실행하여 history_manifest.parquet/.csv/.md를 업데이트

입력:
  - reports/kpi/kpi_table__{run_tag}.csv
  - (있으면) reports/delta/delta_kpi__{baseline_tag_used}__vs__{run_tag}.csv
  - (랭킹이면) data/interim/{run_tag}/ranking_daily.parquet
  - (랭킹이면) reports/ranking/sector_concentration__{run_tag}.csv
  - configs/config.yaml (해시)
  - data/interim/fundamentals_annual.parquet (해시만)

출력:
  - reports/history/history_manifest.parquet
  - reports/history/history_manifest.csv
  - reports/history/history_manifest.md
  - reports/history/history_timeline_ppt.csv
"""

import argparse
import hashlib
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

warnings.filterwarnings("ignore")


def get_file_hash(filepath: Path) -> str:
    """파일의 SHA256 해시 계산"""
    if not filepath.exists():
        return None
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def get_git_commit(repo_dir: Path) -> Optional[str]:
    """Git 커밋 해시 반환"""
    try:
        import subprocess

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]  # 짧은 버전
    except Exception:
        pass
    return None


def get_python_version() -> str:
    """Python 버전 반환"""
    import sys

    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def load_kpi_table(kpi_csv: Path) -> Optional[pd.DataFrame]:
    """KPI 테이블 로드"""
    if not kpi_csv.exists():
        return None
    try:
        return pd.read_csv(kpi_csv)
    except Exception as e:
        print(f"WARNING: KPI 테이블 로드 실패: {e}", file=sys.stderr)
        return None


def load_delta_table(delta_csv: Path) -> Optional[pd.DataFrame]:
    """Delta 테이블 로드"""
    if not delta_csv.exists():
        return None
    try:
        return pd.read_csv(delta_csv)
    except Exception as e:
        print(f"WARNING: Delta 테이블 로드 실패: {e}", file=sys.stderr)
        return None


def extract_kpi_value(
    df: pd.DataFrame, section: str, metric: str, phase: str = "dev"
) -> Optional[Any]:
    """KPI 테이블에서 특정 값 추출"""
    if df is None or df.empty:
        return None

    col = f"{phase}_value" if phase == "dev" else "holdout_value"
    mask = (df["section"] == section) & (df["metric"] == metric)
    matches = df[mask]

    if len(matches) == 0:
        return None

    val = matches.iloc[0][col]
    return val if pd.notna(val) else None


def extract_delta_value(
    df: pd.DataFrame, section: str, metric: str, phase: str = "dev"
) -> Optional[Any]:
    """Delta 테이블에서 특정 값 추출"""
    if df is None or df.empty:
        return None

    col = f"{phase}_delta" if phase == "dev" else "holdout_delta"
    mask = (df["section"] == section) & (df["metric"] == metric)
    matches = df[mask]

    if len(matches) == 0:
        return None

    val = matches.iloc[0][col]
    return val if pd.notna(val) else None


def get_data_range_from_kpi(kpi_df: pd.DataFrame) -> Optional[str]:
    """KPI에서 데이터 범위 추출"""
    if kpi_df is None or kpi_df.empty:
        return None

    # ohlcv_n_dates 또는 date_range 추출 시도
    ohlcv_dates = extract_kpi_value(kpi_df, "DATA", "ohlcv_n_dates")
    if ohlcv_dates:
        return f"{int(ohlcv_dates)} days"

    return None


def get_ranking_kpis(run_tag: str, base_dir: Path) -> dict[str, Any]:
    """랭킹 KPI 추출 (Stage7+)"""
    kpis = {}

    # ranking_daily.parquet 확인
    ranking_path = base_dir / "data" / "interim" / run_tag / "ranking_daily.parquet"
    if ranking_path.exists():
        try:
            df = pd.read_parquet(ranking_path)
            kpis["score_missing"] = (
                df["score"].isna().sum() if "score" in df.columns else None
            )
            kpis["rank_duplicates"] = (
                df.duplicated(["date", "rank"]).sum() if "rank" in df.columns else None
            )

            # top20 HHI 계산
            if "rank" in df.columns:
                top20 = df[df["rank"] <= 20] if "rank" in df.columns else pd.DataFrame()
                if not top20.empty and "ticker" in top20.columns:
                    ticker_counts = top20["ticker"].value_counts()
                    hhi = (ticker_counts / ticker_counts.sum() ** 2).sum() * 10000
                    kpis["top20_hhi"] = hhi
        except Exception as e:
            print(f"WARNING: ranking_daily.parquet 읽기 실패: {e}", file=sys.stderr)

    # sector_concentration.csv 확인
    sector_path = (
        base_dir / "reports" / "ranking" / f"sector_concentration__{run_tag}.csv"
    )
    if sector_path.exists():
        try:
            df = pd.read_csv(sector_path)
            if "max_sector_share" in df.columns:
                kpis["max_sector_share"] = df["max_sector_share"].max()
            if "n_sectors" in df.columns:
                kpis["sector_count"] = df["n_sectors"].max()
            if "hhi" in df.columns:
                kpis["sector_hhi_mean"] = df["hhi"].mean()
        except Exception as e:
            print(f"WARNING: sector_concentration.csv 읽기 실패: {e}", file=sys.stderr)

    return kpis


def get_input_artifact_info(
    artifact_name: str,
    run_tag: str,
    base_dir: Path,
    baseline_tag: Optional[str] = None,
) -> dict[str, Any]:
    """
    입력 아티팩트의 해시/크기/mtime 정보 추출

    Args:
        artifact_name: 아티팩트 이름
        run_tag: 현재 run_tag
        base_dir: 프로젝트 루트 디렉토리
        baseline_tag: baseline 태그 (있으면 우선 사용)

    Returns:
        {hash, size_bytes, mtime} 딕셔너리
    """
    info = {"hash": None, "size_bytes": None, "mtime": None}

    # 경로 후보 (baseline_tag 우선)
    candidates = []
    if baseline_tag:
        candidates.append(
            base_dir / "data" / "interim" / baseline_tag / f"{artifact_name}.parquet"
        )
        candidates.append(
            base_dir / "data" / "interim" / baseline_tag / f"{artifact_name}.csv"
        )

    candidates.append(
        base_dir / "data" / "interim" / run_tag / f"{artifact_name}.parquet"
    )
    candidates.append(base_dir / "data" / "interim" / run_tag / f"{artifact_name}.csv")

    # L2 예외 처리
    if artifact_name == "fundamentals_annual":
        candidates.insert(0, base_dir / "data" / "interim" / f"{artifact_name}.parquet")

    for path in candidates:
        if path.exists():
            try:
                info["hash"] = get_file_hash(path)[:16] if get_file_hash(path) else None
                stat = path.stat()
                info["size_bytes"] = stat.st_size
                info["mtime"] = datetime.fromtimestamp(stat.st_mtime).isoformat()
                break
            except Exception as e:
                print(f"WARNING: {artifact_name} 정보 추출 실패: {e}", file=sys.stderr)

    return info


def get_k_eff_summary(run_tag: str, base_dir: Path) -> dict[str, Any]:
    """
    K_eff 요약 추출 (selection_diagnostics에서)

    Args:
        run_tag: run_tag
        base_dir: 프로젝트 루트 디렉토리

    Returns:
        {k_eff_mean, k_eff_min, k_eff_max, eligible_mean} 딕셔너리
    """
    summary = {
        "k_eff_mean": None,
        "k_eff_min": None,
        "k_eff_max": None,
        "eligible_mean": None,
    }

    diag_path = (
        base_dir / "data" / "interim" / run_tag / "selection_diagnostics.parquet"
    )
    if not diag_path.exists():
        return summary

    try:
        df = pd.read_parquet(diag_path)
        if "selected_count" in df.columns and "top_k" in df.columns:
            # K_eff = selected_count / top_k
            df["k_eff"] = df["selected_count"] / df["top_k"]
            summary["k_eff_mean"] = float(df["k_eff"].mean())
            summary["k_eff_min"] = float(df["k_eff"].min())
            summary["k_eff_max"] = float(df["k_eff"].max())

        if "eligible_count" in df.columns:
            summary["eligible_mean"] = float(df["eligible_count"].mean())
    except Exception as e:
        print(f"WARNING: selection_diagnostics 읽기 실패: {e}", file=sys.stderr)

    return summary


def get_missing_rate_summary(run_tag: str, base_dir: Path) -> dict[str, Any]:
    """
    결측률 요약 추출 (rebalance_scores_summary에서)

    Args:
        run_tag: run_tag
        base_dir: 프로젝트 루트 디렉토리

    Returns:
        {score_short_missing_pct, score_long_missing_pct, score_ens_missing_pct} 딕셔너리
    """
    summary = {
        "score_short_missing_pct": None,
        "score_long_missing_pct": None,
        "score_ens_missing_pct": None,
    }

    summary_path = (
        base_dir / "data" / "interim" / run_tag / "rebalance_scores_summary.parquet"
    )
    if not summary_path.exists():
        return summary

    try:
        df = pd.read_parquet(summary_path)
        if "score_short_missing" in df.columns and "n_tickers" in df.columns:
            df["score_short_missing_pct"] = (
                df["score_short_missing"] / df["n_tickers"] * 100
            )
            summary["score_short_missing_pct"] = float(
                df["score_short_missing_pct"].mean()
            )

        if "score_long_missing" in df.columns and "n_tickers" in df.columns:
            df["score_long_missing_pct"] = (
                df["score_long_missing"] / df["n_tickers"] * 100
            )
            summary["score_long_missing_pct"] = float(
                df["score_long_missing_pct"].mean()
            )

        if "score_ens_missing" in df.columns and "n_tickers" in df.columns:
            df["score_ens_missing_pct"] = (
                df["score_ens_missing"] / df["n_tickers"] * 100
            )
            summary["score_ens_missing_pct"] = float(df["score_ens_missing_pct"].mean())
    except Exception as e:
        print(f"WARNING: rebalance_scores_summary 읽기 실패: {e}", file=sys.stderr)

    return summary


def get_sector_concentration_summary(run_tag: str, base_dir: Path) -> dict[str, Any]:
    """
    섹터농도 요약 추출 (sector_concentration CSV에서)

    Args:
        run_tag: run_tag
        base_dir: 프로젝트 루트 디렉토리

    Returns:
        {sector_hhi_mean, sector_max_share_mean, sector_count_mean} 딕셔너리
    """
    summary = {
        "sector_hhi_mean": None,
        "sector_max_share_mean": None,
        "sector_count_mean": None,
    }

    sector_path = (
        base_dir / "reports" / "ranking" / f"sector_concentration__{run_tag}.csv"
    )
    if not sector_path.exists():
        return summary

    try:
        df = pd.read_csv(sector_path)
        if "hhi" in df.columns:
            summary["sector_hhi_mean"] = float(df["hhi"].mean())
        if "max_sector_share" in df.columns:
            summary["sector_max_share_mean"] = float(df["max_sector_share"].mean())
        if "n_sectors" in df.columns:
            summary["sector_count_mean"] = float(df["n_sectors"].mean())
    except Exception as e:
        print(f"WARNING: sector_concentration 읽기 실패: {e}", file=sys.stderr)

    return summary


def build_history_record(
    stage_no: int,
    track: str,
    run_tag: str,
    baseline_tag_used: Optional[str],
    base_dir: Path,
    config_path: Path,
    change_title: Optional[str] = None,
    change_summary: Optional[list[str]] = None,
    modified_files: Optional[str] = None,
    modified_functions: Optional[str] = None,
    baseline_global_tag: Optional[str] = None,
    baseline_pipeline_tag: Optional[str] = None,
) -> dict[str, Any]:
    """
    History Manifest 레코드 생성 (PPT급 컬럼 포함)

    Args:
        baseline_global_tag: UI/랭킹 최종 비교 기준 (Ranking Track용)
        baseline_pipeline_tag: 백테스트 입력을 보장하는 최신 파이프라인 (Pipeline Track용)
    """
    kpi_csv = base_dir / "reports" / "kpi" / f"kpi_table__{run_tag}.csv"
    delta_csv = None
    if baseline_tag_used:
        delta_csv = (
            base_dir
            / "reports"
            / "delta"
            / f"delta_kpi__{baseline_tag_used}__vs__{run_tag}.csv"
        )

    kpi_df = load_kpi_table(kpi_csv)
    delta_df = load_delta_table(delta_csv) if delta_csv else None

    # 재현성 정보
    config_hash = get_file_hash(config_path) if config_path.exists() else None
    git_commit = get_git_commit(base_dir)
    python_version = get_python_version()

    # L2 해시
    l2_file = base_dir / "data" / "interim" / "fundamentals_annual.parquet"
    l2_hash = get_file_hash(l2_file) if l2_file.exists() else None
    l2_reuse_flag = l2_file.exists()
    l2_size_bytes = l2_file.stat().st_size if l2_file.exists() else None
    l2_mtime = (
        datetime.fromtimestamp(l2_file.stat().st_mtime).isoformat()
        if l2_file.exists()
        else None
    )

    # 데이터 범위
    data_range = get_data_range_from_kpi(kpi_df)

    # 입력 아티팩트 정보 추출 (주요 입력만)
    input_artifacts = []
    if track == "pipeline":
        # Pipeline Track: 주요 입력 추출
        if stage_no >= 6:
            input_artifacts.append("rebalance_scores")
        if stage_no >= 5:
            input_artifacts.extend(["pred_short_oos", "pred_long_oos"])
        if stage_no >= 4:
            input_artifacts.append("dataset_daily")
        if stage_no >= 3:
            input_artifacts.append("panel_merged_daily")
        if stage_no >= 2:
            input_artifacts.append("ohlcv_daily")
    elif track == "ranking":
        # Ranking Track: 주요 입력 추출
        if stage_no >= 11:
            input_artifacts.extend(["ranking_daily", "ohlcv_daily"])
        if stage_no >= 8:
            input_artifacts.append("dataset_daily")

    input_info_summary = {}
    for artifact_name in input_artifacts[:5]:  # 최대 5개만
        info = get_input_artifact_info(
            artifact_name, run_tag, base_dir, baseline_tag_used
        )
        input_info_summary[f"input_{artifact_name}_hash"] = info.get("hash")
        input_info_summary[f"input_{artifact_name}_size_bytes"] = info.get("size_bytes")
        input_info_summary[f"input_{artifact_name}_mtime"] = info.get("mtime")

    # 파이프라인 KPI
    pipeline_kpis = {}
    if kpi_df is not None:
        pipeline_kpis["coverage"] = extract_kpi_value(
            kpi_df, "DATA", "ohlcv_value_nonnull_pct"
        )
        pipeline_kpis["ic_rank"] = extract_kpi_value(
            kpi_df, "MODEL", "ic_rank_mean__20d"
        )
        pipeline_kpis["hit_ratio"] = extract_kpi_value(
            kpi_df, "MODEL", "hit_ratio_mean__20d"
        )

    # 백테스트 KPI (NO-CARRY 규칙 적용)
    # 백테스트 산출물 존재 여부 확인
    bt_metrics_path = base_dir / "data" / "interim" / run_tag / "bt_metrics.parquet"
    if not bt_metrics_path.exists():
        # tag 폴더에 없으면 기본 경로 확인
        bt_metrics_path = base_dir / "data" / "interim" / "bt_metrics.parquet"

    bt_returns_path = base_dir / "data" / "interim" / run_tag / "bt_returns.parquet"
    if not bt_returns_path.exists():
        bt_returns_path = base_dir / "data" / "interim" / "bt_returns.parquet"

    has_backtest = bt_metrics_path.exists() and bt_returns_path.exists()

    backtest_kpis = {}
    backtest_metric_source = None

    if has_backtest:
        # 백테스트 산출물이 있으면 정상 추출
        if kpi_df is not None:
            backtest_kpis["holdout_sharpe"] = extract_kpi_value(
                kpi_df, "BACKTEST", "net_sharpe", "holdout"
            )
            backtest_kpis["holdout_mdd"] = extract_kpi_value(
                kpi_df, "BACKTEST", "net_mdd", "holdout"
            )
            backtest_kpis["holdout_cagr"] = extract_kpi_value(
                kpi_df, "BACKTEST", "net_cagr", "holdout"
            )
            backtest_kpis["turnover"] = extract_kpi_value(
                kpi_df, "BACKTEST", "avg_turnover_oneway", "holdout"
            )
            backtest_kpis["cost"] = extract_kpi_value(
                kpi_df, "BACKTEST", "cost_bps_used", "holdout"
            )
        backtest_metric_source = "bt_metrics.parquet"
    else:
        # [NO-CARRY 규칙] 백테스트 산출물이 없으면 NA로 설정
        backtest_kpis["holdout_sharpe"] = None
        backtest_kpis["holdout_mdd"] = None
        backtest_kpis["holdout_cagr"] = None
        backtest_kpis["turnover"] = None
        backtest_kpis["cost"] = None
        backtest_metric_source = "NA(no backtest)"

    # 랭킹 KPI (Stage7+)
    ranking_kpis = {}
    if stage_no >= 7:
        ranking_kpis = get_ranking_kpis(run_tag, base_dir)

    # K_eff 요약 (L7+)
    k_eff_summary = {}
    if stage_no >= 7:
        k_eff_summary = get_k_eff_summary(run_tag, base_dir)

    # 결측률 요약 (L6+)
    missing_rate_summary = {}
    if stage_no >= 6:
        missing_rate_summary = get_missing_rate_summary(run_tag, base_dir)

    # 섹터농도 요약 (L8+)
    sector_concentration_summary = {}
    if stage_no >= 8:
        sector_concentration_summary = get_sector_concentration_summary(
            run_tag, base_dir
        )

    # 변경 요약
    change_summary_1 = (
        change_summary[0] if change_summary and len(change_summary) > 0 else None
    )
    change_summary_2 = (
        change_summary[1] if change_summary and len(change_summary) > 1 else None
    )
    change_summary_3 = (
        change_summary[2] if change_summary and len(change_summary) > 2 else None
    )

    # 파라미터 변경 추출 (delta에서)
    params_changed = None
    if delta_df is not None and not delta_df.empty:
        settings_delta = (
            delta_df[delta_df["section"] == "SETTINGS"]
            if "section" in delta_df.columns
            else pd.DataFrame()
        )
        if not settings_delta.empty:
            # delta 컬럼명 확인 (dev_delta 또는 dev_value_delta 등)
            dev_col = None
            holdout_col = None
            for col in settings_delta.columns:
                if "dev" in col.lower() and "delta" in col.lower():
                    dev_col = col
                if "holdout" in col.lower() and "delta" in col.lower():
                    holdout_col = col

            # dev_abs_diff 또는 dev_pct_diff가 있는 행 찾기
            if "dev_abs_diff" in settings_delta.columns:
                changed_params = settings_delta[settings_delta["dev_abs_diff"].notna()]
            elif "dev_pct_diff" in settings_delta.columns:
                changed_params = settings_delta[settings_delta["dev_pct_diff"].notna()]
            elif dev_col and holdout_col:
                changed_params = settings_delta[
                    settings_delta[dev_col].notna()
                    | settings_delta[holdout_col].notna()
                ]
            elif dev_col:
                changed_params = settings_delta[settings_delta[dev_col].notna()]
            elif holdout_col:
                changed_params = settings_delta[settings_delta[holdout_col].notna()]
            else:
                changed_params = pd.DataFrame()

            if not changed_params.empty and "metric" in changed_params.columns:
                params_changed = ", ".join(
                    changed_params["metric"].tolist()[:5]
                )  # 최대 5개

    # 게이트 정보 (간단 버전)
    gate_notes = []
    if not kpi_csv.exists():
        gate_notes.append("MISSING: kpi_table.csv")
    if baseline_tag_used and not delta_csv.exists():
        gate_notes.append("MISSING: delta_report.csv")
    if stage_no >= 7:
        ranking_path = base_dir / "data" / "interim" / run_tag / "ranking_daily.parquet"
        if not ranking_path.exists():
            gate_notes.append("MISSING: ranking_daily.parquet")

    gate_notes_str = "; ".join(gate_notes) if gate_notes else None

    record = {
        # 재현성
        "stage_no": stage_no,
        "track": track,
        "run_tag": run_tag,
        "baseline_tag_used": baseline_tag_used,
        "baseline_global_tag": baseline_global_tag,  # [TASK A-2] UI/랭킹 최종
        "baseline_pipeline_tag": baseline_pipeline_tag,  # [TASK A-2] 백테스트 입력 보장
        "created_at": datetime.now().isoformat(),
        "base_dir": str(base_dir),
        "config_hash": config_hash[:16] if config_hash else None,
        "git_commit": git_commit,
        "python_version": python_version,
        "data_range": data_range,
        "l2_reuse_flag": l2_reuse_flag,
        "l2_hash": l2_hash[:16] if l2_hash else None,
        "l2_size_bytes": l2_size_bytes,
        "l2_mtime": l2_mtime,
        # 변경 요약 (PPT용)
        "change_title": change_title,
        "change_summary_1": change_summary_1,
        "change_summary_2": change_summary_2,
        "change_summary_3": change_summary_3,
        "modified_files": modified_files,
        "modified_functions": modified_functions,
        "params_changed": params_changed,
        # 파이프라인 KPI
        "coverage": pipeline_kpis.get("coverage"),
        "ic_rank": pipeline_kpis.get("ic_rank"),
        "hit_ratio": pipeline_kpis.get("hit_ratio"),
        # 백테스트 KPI
        "holdout_sharpe": backtest_kpis.get("holdout_sharpe"),
        "holdout_mdd": backtest_kpis.get("holdout_mdd"),
        "holdout_cagr": backtest_kpis.get("holdout_cagr"),
        "turnover": backtest_kpis.get("turnover"),
        "cost": backtest_kpis.get("cost"),
        # [NO-CARRY 규칙] 백테스트 KPI 출처 명시
        "backtest_metric_source": backtest_metric_source,
        # 랭킹 KPI
        "score_missing": ranking_kpis.get("score_missing"),
        "rank_duplicates": ranking_kpis.get("rank_duplicates"),
        "top20_hhi": ranking_kpis.get("top20_hhi"),
        "max_sector_share": ranking_kpis.get("max_sector_share"),
        "sector_count": ranking_kpis.get("sector_count"),
        # [TASK A-2] 입력 아티팩트 정보
        **input_info_summary,
        # [TASK A-2] K_eff 요약
        "k_eff_mean": k_eff_summary.get("k_eff_mean"),
        "k_eff_min": k_eff_summary.get("k_eff_min"),
        "k_eff_max": k_eff_summary.get("k_eff_max"),
        "eligible_mean": k_eff_summary.get("eligible_mean"),
        # [TASK A-2] 결측률 요약
        "score_short_missing_pct": missing_rate_summary.get("score_short_missing_pct"),
        "score_long_missing_pct": missing_rate_summary.get("score_long_missing_pct"),
        "score_ens_missing_pct": missing_rate_summary.get("score_ens_missing_pct"),
        # [TASK A-2] 섹터농도 요약
        "sector_hhi_mean": sector_concentration_summary.get("sector_hhi_mean"),
        "sector_max_share_mean": sector_concentration_summary.get(
            "sector_max_share_mean"
        ),
        "sector_count_mean": sector_concentration_summary.get("sector_count_mean"),
        # 게이트
        "stage_status": "PASS",  # 기본값 (실패 시 수동 수정)
        "gate_notes": gate_notes_str,
        "ppt_one_liner": change_title or f"Stage {stage_no} {track}",
        "ppt_key_chart_1": None,  # 나중에 추가 가능
        "ppt_key_chart_2": None,
        "ppt_key_chart_3": None,
    }

    return record


def upsert_history_manifest(record: dict[str, Any], history_dir: Path) -> None:
    """
    History Manifest에 레코드 업서트 (upsert)
    """
    history_dir.mkdir(parents=True, exist_ok=True)

    manifest_parquet = history_dir / "history_manifest.parquet"
    manifest_csv = history_dir / "history_manifest.csv"
    manifest_md = history_dir / "history_manifest.md"

    # 기존 manifest 로드
    if manifest_parquet.exists():
        df = pd.read_parquet(manifest_parquet)
    elif manifest_csv.exists():
        df = pd.read_csv(manifest_csv)
    else:
        df = pd.DataFrame()

    # 레코드를 DataFrame으로 변환
    new_row = pd.DataFrame([record])

    # run_tag 기준으로 업서트 (같은 run_tag가 있으면 교체)
    if not df.empty and "run_tag" in df.columns:
        df = df[df["run_tag"] != record["run_tag"]]

    # 새 레코드 추가
    df = pd.concat([df, new_row], ignore_index=True)

    # created_at 기준 정렬 (최신이 위로)
    if "created_at" in df.columns:
        df = df.sort_values("created_at", ascending=False).reset_index(drop=True)

    # 저장
    df.to_parquet(manifest_parquet, index=False)
    df.to_csv(manifest_csv, index=False, encoding="utf-8-sig")

    # Markdown 생성 (간단 버전)
    md_lines = ["# History Manifest", ""]
    header = "| Stage | Track | Run Tag | Created At | Change Title | Holdout Sharpe | Holdout MDD |"
    md_lines.append(header)
    md_lines.append("|---|---|---|---|---|---|---|")

    for _, row in df.head(50).iterrows():  # 최근 50개만
        stage = row.get("stage_no", "N/A")
        track = row.get("track", "N/A")
        run_tag = row.get("run_tag", "N/A")
        created_at = (
            row.get("created_at", "N/A")[:10]
            if pd.notna(row.get("created_at"))
            else "N/A"
        )
        change_title = row.get("change_title", "") or ""
        sharpe = row.get("holdout_sharpe", "N/A")
        mdd = row.get("holdout_mdd", "N/A")

        md_lines.append(
            f"| {stage} | {track} | `{run_tag}` | {created_at} | {change_title} | {sharpe} | {mdd} |"
        )

    md_lines.append("")
    md_lines.append(f"*Total records: {len(df)}*")

    manifest_md.write_text("\n".join(md_lines), encoding="utf-8")

    # Timeline CSV (PPT용)
    timeline_csv = history_dir / "history_timeline_ppt.csv"
    timeline_cols = [
        "created_at",
        "stage_no",
        "track",
        "run_tag",
        "change_title",
        "holdout_sharpe",
        "holdout_mdd",
        "holdout_cagr",
        "ppt_one_liner",
    ]
    # 존재하는 컬럼만 선택
    available_cols = [c for c in timeline_cols if c in df.columns]
    if available_cols:
        timeline_df = df[available_cols].copy()
    else:
        timeline_df = df.copy()
    timeline_df.to_csv(timeline_csv, index=False, encoding="utf-8-sig")

    print(f"[History Manifest] 업데이트 완료: {len(df)} records")
    print(f"[History Manifest] Parquet: {manifest_parquet}")
    print(f"[History Manifest] CSV: {manifest_csv}")
    print(f"[History Manifest] Markdown: {manifest_md}")
    print(f"[History Manifest] Timeline CSV: {timeline_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Update History Manifest (PPT급 컬럼 포함)"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml", help="Config file path"
    )
    parser.add_argument("--stage", type=int, required=True, help="Stage number (0-8)")
    parser.add_argument(
        "--track",
        type=str,
        required=True,
        choices=["pipeline", "ranking"],
        help="Track: pipeline or ranking",
    )
    parser.add_argument("--run-tag", type=str, required=True, help="Run tag")
    parser.add_argument(
        "--baseline-tag",
        type=str,
        default=None,
        help="Baseline tag used for comparison",
    )
    parser.add_argument(
        "--baseline-global-tag",
        type=str,
        default=None,
        help="[TASK A-2] UI/랭킹 최종 비교 기준 (Ranking Track용)",
    )
    parser.add_argument(
        "--baseline-pipeline-tag",
        type=str,
        default=None,
        help="[TASK A-2] 백테스트 입력을 보장하는 최신 파이프라인 (Pipeline Track용)",
    )
    parser.add_argument(
        "--change-title", type=str, default=None, help="Change title (PPT용)"
    )
    parser.add_argument(
        "--change-summary",
        type=str,
        nargs="*",
        default=[],
        help="Change summary (최대 3개)",
    )
    parser.add_argument(
        "--modified-files",
        type=str,
        default=None,
        help="Modified files (comma-separated)",
    )
    parser.add_argument(
        "--modified-functions",
        type=str,
        default=None,
        help="Modified functions (comma-separated)",
    )
    parser.add_argument("--root", type=str, default=None, help="Project root directory")
    args = parser.parse_args()

    # 루트 경로 결정
    if args.root:
        base_dir = Path(args.root)
    else:
        base_dir = Path(__file__).resolve().parents[2]

    config_path = base_dir / args.config
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    history_dir = base_dir / "reports" / "history"

    # 레코드 생성
    record = build_history_record(
        stage_no=args.stage,
        track=args.track,
        run_tag=args.run_tag,
        baseline_tag_used=args.baseline_tag,
        base_dir=base_dir,
        config_path=config_path,
        change_title=args.change_title,
        change_summary=args.change_summary,
        modified_files=args.modified_files,
        modified_functions=args.modified_functions,
        baseline_global_tag=args.baseline_global_tag,
        baseline_pipeline_tag=args.baseline_pipeline_tag,
    )

    # 업서트
    upsert_history_manifest(record, history_dir)

    print("\n[History Manifest] [OK] 완료")
    print(f"Run Tag: {args.run_tag}")
    print(f"Baseline Tag Used: {args.baseline_tag or 'N/A'}")


if __name__ == "__main__":
    main()
