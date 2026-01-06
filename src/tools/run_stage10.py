# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/run_stage10.py
"""
[코드 매니저] Stage 10 실행 엔트리포인트
시장 국면(Regime) 지표 계산 및 ranking_daily에 조인

Stage 실행 → KPI 생성 → Δ 생성 → 체크리포트 → History Manifest 업데이트
"""
import argparse
import sys
import yaml
from pathlib import Path
import glob

from stage_runner_common import (
    verify_l2_reuse,
    verify_base_dir,
    run_command,
    generate_run_tag,
    get_stage_track,
    print_success_summary,
    get_file_hash,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STAGE_NUM = 10
STAGE_TRACK = "ranking"

def main():
    parser = argparse.ArgumentParser(description=f"[코드 매니저] Stage {STAGE_NUM} 실행 (시장 국면 지표 추가)")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config 파일 경로")
    parser.add_argument("--run-tag", type=str, default=None, help="Run tag (없으면 자동 생성)")
    parser.add_argument("--baseline-tag", type=str, default=None, help="Baseline tag (Stage9 run_tag, 없으면 자동 감지)")
    parser.add_argument("--change-title", type=str, default="시장 국면 지표 추가", help="Change title (History Manifest용)")
    parser.add_argument("--change-summary", type=str, nargs="*", default=[
        "실데이터 기반 regime_score 산출",
        "ranking에 regime 조인",
        "UI용 요약 리포트 추가"
    ], help="Change summary (최대 3개)")
    parser.add_argument("--modified-files", type=str, default="src/stages/market_regime.py;src/tools/run_stage10.py", help="Modified files")
    parser.add_argument("--modified-functions", type=str, default="build_market_regime;merge_regime_to_ranking", help="Modified functions")
    args = parser.parse_args()
    
    config_path = PROJECT_ROOT / args.config
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}", file=sys.stderr)
        sys.exit(1)
    
    # base_dir 검증
    base_dir_valid, base_dir_msg = verify_base_dir(PROJECT_ROOT)
    if not base_dir_valid:
        print(f"ERROR: {base_dir_msg}", file=sys.stderr)
        sys.exit(1)
    
    # run_tag 생성
    if args.run_tag:
        run_tag = args.run_tag
    else:
        run_tag = generate_run_tag("stage10_market_regime")
    
    # baseline_tag_used 결정 (지정되지 않으면 최신 Stage9 자동 감지)
    if args.baseline_tag:
        baseline_tag_used = args.baseline_tag
    else:
        # 최신 Stage9 run_tag 자동 감지
        stage9_dirs = glob.glob(str(PROJECT_ROOT / "data" / "interim" / "stage9_ranking_explainability_*"))
        if stage9_dirs:
            stage9_tags = [Path(d).name for d in stage9_dirs]
            stage9_tags.sort(reverse=True)
            baseline_tag_used = stage9_tags[0]
            print(f"[INFO] 최신 Stage9 run_tag 자동 감지: {baseline_tag_used}")
        else:
            print("ERROR: Stage9 run_tag를 찾을 수 없습니다. --baseline-tag로 지정하세요.", file=sys.stderr)
            sys.exit(1)
    
    print("\n" + "="*60)
    print(f"[코드 매니저] Stage {STAGE_NUM} 실행 (시장 국면 지표 추가)")
    print("="*60)
    print(f"프로젝트 루트: {PROJECT_ROOT}")
    print(f"Run Tag: {run_tag}")
    print(f"Baseline Tag Used (Stage9): {baseline_tag_used}")
    print("="*60 + "\n")
    
    # 로그 파일 경로
    log_file = PROJECT_ROOT / "reports" / "logs" / f"run__stage{STAGE_NUM}__{run_tag}.txt"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 0) L2 재사용 검증 (사전)
    l2_file = PROJECT_ROOT / "data" / "interim" / "fundamentals_annual.parquet"
    if not l2_file.exists():
        print("ERROR: L2 파일이 없습니다. fundamentals_annual.parquet를 먼저 준비하세요.", file=sys.stderr)
        sys.exit(1)
    
    l2_hash_before = None
    try:
        l2_hash_full = get_file_hash(l2_file)
        l2_hash_before = l2_hash_full[:16] + "..."
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"[L2 검증] 실행 전 해시: {l2_hash_before}\n")
    except Exception as e:
        print(f"WARNING: L2 해시 계산 실패: {e}", file=sys.stderr)
    
    # Baseline ranking_daily 확인
    baseline_ranking_path = PROJECT_ROOT / "data" / "interim" / baseline_tag_used / "ranking_daily.parquet"
    if not baseline_ranking_path.exists():
        print(f"ERROR: Baseline ranking_daily가 없습니다: {baseline_ranking_path}", file=sys.stderr)
        sys.exit(1)
    
    # Python 모듈 import
    src_dir = PROJECT_ROOT / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    
    import pandas as pd
    from src.stages.export.market_regime import build_market_regime_daily
    from src.stages.ranking.ranking_merge_regime import merge_regime_to_ranking
    from src.utils.config import load_config, get_path
    
    # 설정 로드
    cfg = load_config(str(config_path))
    l8 = cfg.get("l8", {})
    params = cfg.get("params", {})
    start_date = params.get("start_date", "2015-01-02")
    end_date = params.get("end_date", "2024-12-31")
    lookback_days = l8.get("regime", {}).get("lookback_days", 60) if l8.get("regime") else 60
    
    # 1) Stage10 실행 (시장 국면 지표 계산)
    print("\n[1/7] Stage10 실행 (시장 국면 지표 계산)...")
    
    # Baseline ranking_daily 로드
    baseline_ranking = pd.read_parquet(baseline_ranking_path)
    print(f"[OK] Baseline ranking_daily 로드: {len(baseline_ranking):,} rows")
    
    # 날짜 범위 확인
    date_range = baseline_ranking["date"].min(), baseline_ranking["date"].max()
    print(f"[OK] 날짜 범위: {date_range[0]} ~ {date_range[1]}")
    
    # OHLCV 및 universe 데이터 로드 (pykrx 실패 시 사용)
    base_interim_dir = get_path(cfg, "data_interim")
    ohlcv_daily = None
    universe_daily = None
    
    candidate_paths = [
        base_interim_dir / baseline_tag_used / "ohlcv_daily.parquet",
        base_interim_dir / "ohlcv_daily.parquet",
    ]
    
    for path in candidate_paths:
        if path.exists():
            try:
                ohlcv_daily = pd.read_parquet(path)
                print(f"[OK] OHLCV 데이터 로드 (fallback용): {path} ({len(ohlcv_daily):,} rows)")
                break
            except Exception as e:
                print(f"WARNING: {path} 로드 실패: {e}", file=sys.stderr)
    
    # universe 데이터 로드
    universe_paths = [
        base_interim_dir / baseline_tag_used / "dataset_daily.parquet",
        base_interim_dir / "dataset_daily.parquet",
    ]
    
    for path in universe_paths:
        if path.exists():
            try:
                universe_daily = pd.read_parquet(path)
                if "in_universe" in universe_daily.columns:
                    print(f"[OK] Universe 데이터 로드 (fallback용): {path} ({len(universe_daily):,} rows)")
                    break
            except Exception as e:
                print(f"WARNING: {path} 로드 실패: {e}", file=sys.stderr)
    
    # 캐시 디렉토리 설정
    cache_dir = PROJECT_ROOT / "data" / "external" / "cache"
    
    # 시장 국면 지표 계산
    try:
        market_regime_daily = build_market_regime_daily(
            start_date=str(date_range[0].date()),
            end_date=str(date_range[1].date()),
            ohlcv_daily=ohlcv_daily,
            universe_daily=universe_daily,
            use_pykrx=True,
            cache_dir=cache_dir,
            lookback_days=lookback_days,
        )
        print(f"[OK] 시장 국면 지표 계산 완료: {len(market_regime_daily):,} rows")
        print(f"  컬럼: {list(market_regime_daily.columns)}")
    except Exception as e:
        print(f"ERROR: 시장 국면 지표 계산 실패: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 저장
    interim_dir = base_interim_dir / run_tag
    interim_dir.mkdir(parents=True, exist_ok=True)
    
    regime_output_path = interim_dir / "market_regime_daily.parquet"
    market_regime_daily.to_parquet(regime_output_path, index=False)
    print(f"[OK] market_regime_daily 저장: {regime_output_path}")
    
    # ranking_daily에 regime 조인
    try:
        ranking_with_regime = merge_regime_to_ranking(baseline_ranking, market_regime_daily)
        print(f"[OK] ranking_daily에 regime 조인 완료: {len(ranking_with_regime):,} rows")
        print(f"  추가된 컬럼: {[c for c in ranking_with_regime.columns if 'regime' in c.lower()]}")
    except Exception as e:
        print(f"ERROR: regime 조인 실패: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # ranking_daily 저장
    ranking_output_path = interim_dir / "ranking_daily.parquet"
    ranking_with_regime.to_parquet(ranking_output_path, index=False)
    print(f"[OK] ranking_daily 저장: {ranking_output_path}")
    
    # L2 재사용 검증 (사후)
    l2_valid, l2_msg, _, l2_hash_after = verify_l2_reuse(PROJECT_ROOT, log_file)
    if not l2_valid:
        print(f"ERROR: {l2_msg}", file=sys.stderr)
        sys.exit(1)
    
    # 2) regime_summary 리포트 생성
    print("\n[2/7] regime_summary 리포트 생성 중...")
    
    try:
        # 연도별 bull/bear 비중 계산
        market_regime_daily["year"] = pd.to_datetime(market_regime_daily["date"]).dt.year
        summary_rows = []
        
        for year in sorted(market_regime_daily["year"].unique()):
            year_data = market_regime_daily[market_regime_daily["year"] == year]
            
            bull_count = (year_data["regime_label"] == "BULL").sum()
            bear_count = (year_data["regime_label"] == "BEAR").sum()
            neutral_count = (year_data["regime_label"] == "NEUTRAL").sum()
            total_count = len(year_data)
            
            summary_rows.append({
                "year": year,
                "total_days": total_count,
                "bull_days": bull_count,
                "bear_days": bear_count,
                "neutral_days": neutral_count,
                "bull_pct": bull_count / total_count * 100 if total_count > 0 else 0,
                "bear_pct": bear_count / total_count * 100 if total_count > 0 else 0,
                "neutral_pct": neutral_count / total_count * 100 if total_count > 0 else 0,
                "avg_regime_score": year_data["regime_score"].mean(),
                "min_regime_score": year_data["regime_score"].min(),
                "max_regime_score": year_data["regime_score"].max(),
            })
        
        summary_df = pd.DataFrame(summary_rows)
        
        summary_dir = PROJECT_ROOT / "reports" / "ranking"
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_path = summary_dir / f"regime_summary__{run_tag}.csv"
        summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
        print(f"[OK] regime_summary 저장: {summary_path} ({len(summary_df)} rows)")
    except Exception as e:
        print(f"WARNING: regime_summary 생성 실패: {e}", file=sys.stderr)
    
    # 3) KPI 생성 (필수)
    print("\n[3/7] KPI 생성 중...")
    
    kpi_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "tools" / "export_kpi_table.py"),
        "--config", args.config,
        "--tag", run_tag,
    ]
    
    if run_command(kpi_cmd, PROJECT_ROOT, "KPI 생성", log_file) != 0:
        sys.exit(1)
    
    # 4) Δ 생성 (필수) - Stage9 vs Stage10
    print("\n[4/7] Δ 리포트 생성 중...")
    
    delta_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "tools" / "export_delta_report.py"),
        "--baseline-tag", baseline_tag_used,
        "--run-tag", run_tag,
    ]
    
    if run_command(delta_cmd, PROJECT_ROOT, "Δ 리포트 생성 (Stage9 vs Stage10)", log_file) != 0:
        print("ERROR: Delta 리포트 생성 실패", file=sys.stderr)
        sys.exit(1)
    
    delta_csv = PROJECT_ROOT / "reports" / "delta" / f"delta_kpi__{baseline_tag_used}__vs__{run_tag}.csv"
    delta_md = PROJECT_ROOT / "reports" / "delta" / f"delta_report__{baseline_tag_used}__vs__{run_tag}.md"
    
    if not delta_csv.exists() or not delta_md.exists():
        print(f"ERROR: Delta 리포트 파일이 생성되지 않았습니다.", file=sys.stderr)
        sys.exit(1)
    
    print(f"[OK] Delta 리포트 생성 완료: {delta_csv}")
    
    # 5) Stage10 체크리포트 생성 (필수)
    print("\n[5/7] 체크리포트 생성 중...")
    
    check_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "tools" / "check_stage_completion.py"),
        "--config", args.config,
        "--stage", str(STAGE_NUM),
        "--run-tag", run_tag,
        "--baseline-tag", baseline_tag_used,
    ]
    
    if run_command(check_cmd, PROJECT_ROOT, "체크리포트 생성", log_file) != 0:
        print("WARNING: 체크리포트 생성 실패 (계속 진행)", file=sys.stderr)
    
    # 6) History Manifest 업데이트 (필수)
    print("\n[6/7] History Manifest 업데이트 중...")
    
    track = get_stage_track(STAGE_NUM)
    history_cmd = [
        sys.executable,
        str(PROJECT_ROOT / "src" / "tools" / "update_history_manifest.py"),
        "--config", args.config,
        "--stage", str(STAGE_NUM),
        "--track", track,
        "--run-tag", run_tag,
        "--baseline-tag", baseline_tag_used,
        "--change-title", args.change_title,
    ]
    
    if args.change_summary:
        history_cmd.extend(["--change-summary"] + args.change_summary)
    if args.modified_files:
        history_cmd.extend(["--modified-files", args.modified_files])
    if args.modified_functions:
        history_cmd.extend(["--modified-functions", args.modified_functions])
    
    if run_command(history_cmd, PROJECT_ROOT, "History Manifest 업데이트", log_file) != 0:
        print("WARNING: History Manifest 업데이트 실패 (계속 진행)", file=sys.stderr)
    
    # 7) 최종 출력
    output_files = [
        ("산출물 (regime)", regime_output_path),
        ("산출물 (ranking)", ranking_output_path),
        ("Summary", summary_path if 'summary_path' in locals() else None),
        ("KPI CSV", PROJECT_ROOT / "reports" / "kpi" / f"kpi_table__{run_tag}.csv"),
        ("KPI MD", PROJECT_ROOT / "reports" / "kpi" / f"kpi_table__{run_tag}.md"),
        ("Delta CSV", delta_csv),
        ("Delta MD", delta_md),
        ("체크리포트", PROJECT_ROOT / "reports" / "stages" / f"check__stage{STAGE_NUM}__{run_tag}.md"),
        ("History Manifest", PROJECT_ROOT / "reports" / "history" / "history_manifest.parquet"),
        ("로그", log_file),
    ]
    
    # None 제거
    output_files = [(desc, path) for desc, path in output_files if path is not None]
    
    print_success_summary(run_tag, baseline_tag_used, None, output_files)
    
    # 완료 기준 검증
    print("\n[완료 기준 검증]")
    checks = {
        "market_regime_daily.parquet 존재": regime_output_path.exists() and len(market_regime_daily) > 0,
        "ranking_daily.parquet 존재": ranking_output_path.exists() and len(ranking_with_regime) > 0,
        "regime_score 컬럼 존재": "regime_score" in ranking_with_regime.columns,
        "regime_label 컬럼 존재": "regime_label" in ranking_with_regime.columns,
        "KPI CSV 존재": (PROJECT_ROOT / "reports" / "kpi" / f"kpi_table__{run_tag}.csv").exists(),
        "Delta CSV 존재": delta_csv.exists(),
        "체크리포트 존재": (PROJECT_ROOT / "reports" / "stages" / f"check__stage{STAGE_NUM}__{run_tag}.md").exists(),
        "History Manifest 존재": (PROJECT_ROOT / "reports" / "history" / "history_manifest.parquet").exists(),
    }
    
    all_pass = all(checks.values())
    for check_name, check_result in checks.items():
        status = "[PASS]" if check_result else "[FAIL]"
        print(f"  {status} {check_name}")
    
    if all_pass:
        print("\n[PASS] 모든 완료 기준 충족")
    else:
        print("\n[FAIL] 일부 완료 기준 미충족")
        sys.exit(1)

if __name__ == "__main__":
    main()







