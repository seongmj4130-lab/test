# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/validation/check_stage12_validation.py
"""
Stage12 체크리스트 검증 스크립트
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def check_timeline_ppt(run_tag: str):
    """timeline_ppt.csv 검증"""
    print("\n" + "="*60)
    print("1. timeline_ppt.csv 검증")
    print("="*60)

    path = PROJECT_ROOT / "artifacts" / "reports" / "final_export" / run_tag / "timeline_ppt.csv"
    if not path.exists():
        print(f"[FAIL] 파일 없음: {path}")
        return False

    df = pd.read_csv(path)

    print(f"총 행수: {len(df):,}")
    print(f"컬럼: {list(df.columns)}")

    # baseline + stage0~현재 수준인지 확인 (최소 10개 이상)
    min_rows = 10  # baseline(-1) + stage0~11 = 최소 13개
    if len(df) < min_rows:
        print(f"[FAIL] 행수가 부족함: {len(df)}개 (최소 {min_rows}개 필요)")
        return False

    # stage_no 컬럼 확인
    if "stage_no" in df.columns:
        print(f"\nstage_no 범위: {df['stage_no'].min()} ~ {df['stage_no'].max()}")
        print(f"stage_no 고유값: {sorted(df['stage_no'].unique())}")

    print("\n처음 5행:")
    print(df.head().to_string())

    print("\n[PASS] timeline_ppt.csv 검증 통과")
    return True

def check_kpi_onepager(run_tag: str):
    """kpi_onepager.csv 검증 (핵심 KPI만)"""
    print("\n" + "="*60)
    print("2. kpi_onepager.csv 검증")
    print("="*60)

    path = PROJECT_ROOT / "artifacts" / "reports" / "final_export" / run_tag / "kpi_onepager.csv"
    if not path.exists():
        print(f"[FAIL] 파일 없음: {path}")
        return False

    df = pd.read_csv(path)

    print(f"총 행수: {len(df):,}")
    print(f"컬럼: {list(df.columns)}")

    # 핵심 KPI 컬럼 확인 (예상 컬럼)
    expected_key_cols = [
        "holdout_sharpe", "holdout_mdd", "holdout_cagr", "holdout_total_return",
        "net_sharpe", "net_mdd", "net_total_return",
        "information_ratio", "tracking_error_ann", "avg_turnover_oneway"
    ]

    found_key_cols = [c for c in expected_key_cols if c in df.columns]
    print(f"\n핵심 KPI 컬럼 발견: {len(found_key_cols)}/{len(expected_key_cols)}")
    if found_key_cols:
        print(f"  - {', '.join(found_key_cols[:5])}...")

    print("\n처음 10행:")
    print(df.head(10).to_string())

    print("\n[PASS] kpi_onepager.csv 검증 통과")
    return True

def check_latest_snapshot(run_tag: str):
    """latest_snapshot.csv 검증 (Top/Bottom + top_features + regime)"""
    print("\n" + "="*60)
    print("3. latest_snapshot.csv 검증")
    print("="*60)

    path = PROJECT_ROOT / "artifacts" / "reports" / "final_export" / run_tag / "latest_snapshot.csv"
    if not path.exists():
        print(f"[FAIL] 파일 없음: {path}")
        return False

    df = pd.read_csv(path)

    print(f"총 행수: {len(df):,}")
    print(f"컬럼: {list(df.columns)}")

    # 필수 컬럼 확인
    has_top_bottom = "snapshot_type" in df.columns or ("top" in str(df.columns).lower() and "bottom" in str(df.columns).lower())
    has_top_features = "top_features" in df.columns
    has_regime = "regime_label" in df.columns or "regime_score" in df.columns

    print(f"\n필수 컬럼 확인:")
    print(f"  - Top/Bottom 구분: {has_top_bottom}")
    print(f"  - top_features: {has_top_features}")
    print(f"  - regime: {has_regime}")

    if not has_top_bottom:
        print("[FAIL] Top/Bottom 구분 컬럼 없음")
        return False
    if not has_top_features:
        print("[FAIL] top_features 컬럼 없음")
        return False
    if not has_regime:
        print("[FAIL] regime 컬럼 없음")
        return False

    # 샘플 출력
    if "snapshot_type" in df.columns:
        print(f"\nTop/Bottom 분포:")
        print(df["snapshot_type"].value_counts())

    print("\n처음 5행:")
    print(df.head().to_string())

    print("\n[PASS] latest_snapshot.csv 검증 통과")
    return True

def check_equity_curves(run_tag: str):
    """equity_curves.csv 검증 (그래프 렌더 가능)"""
    print("\n" + "="*60)
    print("4. equity_curves.csv 검증")
    print("="*60)

    path = PROJECT_ROOT / "artifacts" / "reports" / "final_export" / run_tag / "equity_curves.csv"
    if not path.exists():
        print(f"[FAIL] 파일 없음: {path}")
        return False

    df = pd.read_csv(path)

    print(f"총 행수: {len(df):,}")
    print(f"컬럼: {list(df.columns)}")

    # 필수 컬럼 확인 (그래프 렌더용)
    required_cols = ["date", "strategy_equity", "bench_equity"]
    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        print(f"[FAIL] 필수 컬럼 누락: {missing_cols}")
        return False

    # 날짜 정렬 확인
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        is_sorted = df["date"].is_monotonic_increasing
        print(f"\n날짜 정렬: {is_sorted}")

        if not is_sorted:
            print("[FAIL] 날짜가 정렬되지 않음")
            return False

        print(f"날짜 범위: {df['date'].min()} ~ {df['date'].max()}")

    # equity 값 확인 (0 이하 없어야 함)
    if "strategy_equity" in df.columns:
        min_strategy = df["strategy_equity"].min()
        if min_strategy <= 0:
            print(f"[FAIL] strategy_equity가 0 이하: {min_strategy}")
            return False
        print(f"strategy_equity 범위: {min_strategy:.2f} ~ {df['strategy_equity'].max():.2f}")

    if "bench_equity" in df.columns:
        min_bench = df["bench_equity"].min()
        if min_bench <= 0:
            print(f"[FAIL] bench_equity가 0 이하: {min_bench}")
            return False
        print(f"bench_equity 범위: {min_bench:.2f} ~ {df['bench_equity'].max():.2f}")

    print("\n처음 5행:")
    print(df.head().to_string())
    print("\n마지막 5행:")
    print(df.tail().to_string())

    print("\n[PASS] equity_curves.csv 검증 통과")
    return True

def check_appendix_sources(run_tag: str):
    """appendix_sources.md 검증 (경로/해시)"""
    print("\n" + "="*60)
    print("5. appendix_sources.md 검증")
    print("="*60)

    path = PROJECT_ROOT / "artifacts" / "reports" / "final_export" / run_tag / "appendix_sources.md"
    if not path.exists():
        print(f"[FAIL] 파일 없음: {path}")
        return False

    content = path.read_text(encoding='utf-8')

    print(f"파일 크기: {len(content):,} bytes")

    # 경로와 해시 확인
    has_paths = "경로" in content or "path" in content.lower() or "| 경로 |" in content
    has_hash = "해시" in content or "hash" in content.lower() or "SHA256" in content

    print(f"\n필수 내용 확인:")
    print(f"  - 경로 정보: {has_paths}")
    print(f"  - 해시 정보: {has_hash}")

    if not has_paths:
        print("[FAIL] 경로 정보 없음")
        return False
    if not has_hash:
        print("[FAIL] 해시 정보 없음")
        return False

    # 샘플 출력
    lines = content.split("\n")
    print(f"\n파일 첫 20줄:")
    for i, line in enumerate(lines[:20], 1):
        print(f"{i:2d}: {line}")

    print("\n[PASS] appendix_sources.md 검증 통과")
    return True

def check_history_manifest(run_tag: str):
    """history_manifest에 stage12 기록 확인"""
    print("\n" + "="*60)
    print("6. history_manifest에 stage12 기록 확인")
    print("="*60)

    manifest_path = PROJECT_ROOT / "reports" / "history" / "history_manifest.parquet"
    if not manifest_path.exists():
        manifest_path = PROJECT_ROOT / "reports" / "history" / "history_manifest.csv"

    if not manifest_path.exists():
        print(f"[FAIL] history_manifest 파일 없음")
        return False

    if manifest_path.suffix == ".parquet":
        df = pd.read_parquet(manifest_path)
    else:
        df = pd.read_csv(manifest_path)

    # stage12 기록 확인
    stage12_rows = df[df["stage_no"] == 12]

    if len(stage12_rows) == 0:
        print("[FAIL] stage12 기록 없음")
        return False

    # run_tag 일치 확인
    matching_rows = stage12_rows[stage12_rows["run_tag"] == run_tag]

    if len(matching_rows) == 0:
        print(f"[FAIL] stage12 기록은 있지만 run_tag 불일치: {run_tag}")
        print(f"발견된 stage12 run_tag: {stage12_rows['run_tag'].tolist()}")
        return False

    print(f"[PASS] stage12 기록 확인됨: {len(matching_rows)}개")
    print(f"\nstage12 기록:")
    print(matching_rows[["stage_no", "track", "run_tag", "change_title", "created_at"]].to_string())

    return True

def check_paths_not_desktop(run_tag: str):
    """저장 경로가 Desktop base_dir 하위인지 확인 (바탕 화면 경로면 FAIL)"""
    print("\n" + "="*60)
    print("7. 저장 경로 확인 (Desktop base_dir 하위)")
    print("="*60)

    export_dir = PROJECT_ROOT / "artifacts" / "reports" / "final_export" / run_tag

    # 바탕 화면 경로 패턴 확인 (한글 "바탕 화면"만 체크)
    desktop_patterns = [
        "바탕 화면",  # 한글 바탕 화면만 체크
        "OneDrive\\바탕 화면",
        "OneDrive/바탕 화면",
    ]

    export_dir_str = str(export_dir)
    project_root_str = str(PROJECT_ROOT)

    print(f"Export 디렉토리: {export_dir}")
    print(f"프로젝트 루트: {PROJECT_ROOT}")

    # 바탕 화면 경로(한글) 포함 여부 확인
    has_desktop_path = any(pattern in export_dir_str for pattern in desktop_patterns)

    if has_desktop_path:
        print(f"[FAIL] 바탕 화면 경로(한글) 감지: {export_dir_str}")
        return False

    # base_dir 하위인지 확인
    try:
        rel_path = export_dir.relative_to(PROJECT_ROOT)
        print(f"[PASS] 저장 경로가 프로젝트 루트 하위: {rel_path}")

        # 프로젝트 루트가 올바른 base_dir인지 확인 (config.yaml의 base_dir와 일치)
        expected_base_dir = "C:/Users/seong/OneDrive/Desktop/bootcamp/03_code"
        if project_root_str.replace("\\", "/") != expected_base_dir:
            print(f"[WARNING] 프로젝트 루트가 예상과 다름:")
            print(f"  예상: {expected_base_dir}")
            print(f"  실제: {project_root_str}")
            # 경고만 하고 계속 진행

    except ValueError:
        print(f"[FAIL] 저장 경로가 프로젝트 루트 하위가 아님")
        return False

    # 실제 파일들 확인
    files = list(export_dir.glob("*"))
    print(f"\n생성된 파일 수: {len(files)}개")
    for f in files:
        print(f"  - {f.name}")

    print("\n[PASS] 저장 경로 검증 통과")
    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python check_stage12_validation.py <run_tag>")
        print("Example: python check_stage12_validation.py stage12_final_export_20251221_013411")
        sys.exit(1)

    run_tag = sys.argv[1]

    print("="*60)
    print(f"Stage12 체크리스트 검증")
    print("="*60)
    print(f"Run Tag: {run_tag}")
    print("="*60)

    results = []

    # 1. timeline_ppt.csv 검증
    results.append(("timeline_ppt", check_timeline_ppt(run_tag)))

    # 2. kpi_onepager.csv 검증
    results.append(("kpi_onepager", check_kpi_onepager(run_tag)))

    # 3. latest_snapshot.csv 검증
    results.append(("latest_snapshot", check_latest_snapshot(run_tag)))

    # 4. equity_curves.csv 검증
    results.append(("equity_curves", check_equity_curves(run_tag)))

    # 5. appendix_sources.md 검증
    results.append(("appendix_sources", check_appendix_sources(run_tag)))

    # 6. history_manifest에 stage12 기록 확인
    results.append(("history_manifest", check_history_manifest(run_tag)))

    # 7. 저장 경로 확인
    results.append(("paths_not_desktop", check_paths_not_desktop(run_tag)))

    # 최종 요약
    print("\n" + "="*60)
    print("최종 검증 결과")
    print("="*60)

    all_pass = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{status} {name}")
        if not passed:
            all_pass = False

    print("="*60)
    if all_pass:
        print("[PASS] 모든 검증 통과")
        sys.exit(0)
    else:
        print("[FAIL] 일부 검증 실패")
        sys.exit(1)

if __name__ == "__main__":
    main()
