# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/validation/check_stage11_validation.py
"""
Stage11 체크리스트 검증 스크립트
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def check_ui_equity_curves(run_tag: str):
    """ui_equity_curves 검증"""
    print("\n" + "="*60)
    print("1. ui_equity_curves 검증")
    print("="*60)

    path = PROJECT_ROOT / "data" / "interim" / run_tag / "ui_equity_curves.parquet"
    if not path.exists():
        print(f"[FAIL] 파일 없음: {path}")
        return False

    df = pd.read_parquet(path)

    # 날짜 중복 체크
    date_duplicates = df["date"].duplicated().sum()
    print(f"총 행수: {len(df):,}")
    print(f"날짜 중복: {date_duplicates}개")

    # 날짜 정렬 체크
    is_sorted = df["date"].is_monotonic_increasing
    print(f"날짜 정렬 확인: {is_sorted}")

    # equity 값 체크
    strategy_min = df["strategy_equity"].min()
    bench_min = df["bench_equity"].min()
    strategy_below_zero = (df["strategy_equity"] <= 0).sum()
    bench_below_zero = (df["bench_equity"] <= 0).sum()

    print(f"\nstrategy_equity 최소값: {strategy_min:.6f}")
    print(f"bench_equity 최소값: {bench_min:.6f}")
    print(f"strategy_equity 0 이하: {strategy_below_zero}개")
    print(f"bench_equity 0 이하: {bench_below_zero}개")

    # 검증 결과
    pass_check = True
    if date_duplicates > 0:
        print(f"[FAIL] 날짜 중복 발견: {date_duplicates}개")
        pass_check = False
    if not is_sorted:
        print("[FAIL] 날짜가 정렬되지 않음")
        pass_check = False
    if strategy_below_zero > 0:
        print(f"[FAIL] strategy_equity가 0 이하인 행: {strategy_below_zero}개")
        pass_check = False
    if bench_below_zero > 0:
        print(f"[FAIL] bench_equity가 0 이하인 행: {bench_below_zero}개")
        pass_check = False

    if pass_check:
        print("\n[PASS] ui_equity_curves 검증 통과")
    else:
        print("\n[FAIL] ui_equity_curves 검증 실패")

    # 샘플 출력
    print("\n처음 5행:")
    print(df.head().to_string())
    print("\n마지막 5행:")
    print(df.tail().to_string())

    return pass_check

def check_ui_top_bottom_daily(run_tag: str, expected_top_k: int = 10, expected_bottom_k: int = 10):
    """ui_top_bottom_daily 검증"""
    print("\n" + "="*60)
    print("2. ui_top_bottom_daily 검증")
    print("="*60)

    path = PROJECT_ROOT / "data" / "interim" / run_tag / "ui_top_bottom_daily.parquet"
    if not path.exists():
        print(f"[FAIL] 파일 없음: {path}")
        return False

    df = pd.read_parquet(path)

    # 날짜 중복 체크
    date_duplicates = df["date"].duplicated().sum()
    print(f"총 행수: {len(df):,}")
    print(f"날짜 중복: {date_duplicates}개")

    # Top/Bottom 리스트 길이 체크
    df["top_count"] = df["top_list"].str.split(",").str.len()
    df["bottom_count"] = df["bottom_list"].str.split(",").str.len()

    print(f"\nTop 리스트 평균 길이: {df['top_count'].mean():.1f}")
    print(f"Bottom 리스트 평균 길이: {df['bottom_count'].mean():.1f}")
    print(f"Top 리스트 길이 범위: {df['top_count'].min()} ~ {df['top_count'].max()}")
    print(f"Bottom 리스트 길이 범위: {df['bottom_count'].min()} ~ {df['bottom_count'].max()}")

    # 기대값과 일치하는지 체크
    top_mismatch = (df["top_count"] != expected_top_k).sum()
    bottom_mismatch = (df["bottom_count"] != expected_bottom_k).sum()

    print(f"\n기대 Top K: {expected_top_k}, 불일치: {top_mismatch}개")
    print(f"기대 Bottom K: {expected_bottom_k}, 불일치: {bottom_mismatch}개")

    # 동일 ticker가 top/bottom 동시 등장 체크
    df["top_set"] = df["top_list"].str.split(",").apply(lambda x: set([t.strip() for t in x if t.strip()]))
    df["bottom_set"] = df["bottom_list"].str.split(",").apply(lambda x: set([t.strip() for t in x if t.strip()]))
    df["overlap"] = df.apply(lambda row: len(row["top_set"] & row["bottom_set"]), axis=1)

    overlap_count = (df["overlap"] > 0).sum()
    print(f"\nTop/Bottom 겹치는 날짜 수: {overlap_count}개")

    # 검증 결과
    pass_check = True
    if date_duplicates > 0:
        print(f"[FAIL] 날짜 중복 발견: {date_duplicates}개")
        pass_check = False
    if top_mismatch > 0:
        print(f"[FAIL] Top 리스트 길이 불일치: {top_mismatch}개 (기대: {expected_top_k})")
        pass_check = False
    if bottom_mismatch > 0:
        print(f"[FAIL] Bottom 리스트 길이 불일치: {bottom_mismatch}개 (기대: {expected_bottom_k})")
        pass_check = False
    if overlap_count > 0:
        print(f"[FAIL] 동일 ticker가 top/bottom 동시 등장: {overlap_count}개")
        print("\n겹치는 날짜 샘플:")
        overlap_df = df[df["overlap"] > 0][["date", "top_list", "bottom_list", "overlap"]].head(10)
        print(overlap_df.to_string())
        pass_check = False

    if pass_check:
        print("\n[PASS] ui_top_bottom_daily 검증 통과")
    else:
        print("\n[FAIL] ui_top_bottom_daily 검증 실패")

    # 샘플 출력
    print("\n처음 3행:")
    print(df[["date", "top_count", "bottom_count", "top_list", "bottom_list"]].head(3).to_string())

    return pass_check

def check_ui_snapshot(run_tag: str):
    """UI Snapshot 검증"""
    print("\n" + "="*60)
    print("3. UI Snapshot 검증")
    print("="*60)

    path = PROJECT_ROOT / "reports" / "ui" / f"ui_snapshot__{run_tag}.csv"
    if not path.exists():
        print(f"[FAIL] 파일 없음: {path}")
        return False

    df = pd.read_csv(path)

    print(f"총 행수: {len(df):,}")
    print(f"컬럼: {list(df.columns)}")

    # 필수 컬럼 체크
    has_regime_label = "regime_label" in df.columns
    has_regime_score = "regime_score" in df.columns
    has_top_features = "top_features" in df.columns

    print(f"\nregime_label 존재: {has_regime_label}")
    print(f"regime_score 존재: {has_regime_score}")
    print(f"top_features 존재: {has_top_features}")

    # 값 분포 체크
    if has_regime_label:
        print(f"\nregime_label 값 분포:")
        print(df["regime_label"].value_counts())

    if has_regime_score:
        print(f"\nregime_score 통계:")
        print(df["regime_score"].describe())

    if has_top_features:
        missing_pct = df["top_features"].isna().sum() / len(df) * 100
        print(f"\ntop_features 결측률: {missing_pct:.1f}%")

    # 검증 결과
    pass_check = True
    if not has_regime_label:
        print("[FAIL] regime_label 컬럼 없음")
        pass_check = False
    if not has_regime_score:
        print("[FAIL] regime_score 컬럼 없음")
        pass_check = False
    if not has_top_features:
        print("[FAIL] top_features 컬럼 없음")
        pass_check = False

    if pass_check:
        print("\n[PASS] UI Snapshot 검증 통과")
    else:
        print("\n[FAIL] UI Snapshot 검증 실패")

    # 샘플 출력
    if has_regime_label:
        print("\nTop10 샘플:")
        top_df = df[df["snapshot_type"] == "top"][["snapshot_date", "snapshot_type", "snapshot_rank", "ticker", "regime_label", "top_features"]].head(5)
        print(top_df.to_string())

    return pass_check

def check_delta_baseline(run_tag: str, expected_baseline: str):
    """Δ 리포트 baseline_tag 확인"""
    print("\n" + "="*60)
    print("4. Δ 리포트 baseline_tag 확인")
    print("="*60)

    # Delta 리포트 파일 찾기
    delta_dir = PROJECT_ROOT / "reports" / "delta"
    delta_files = list(delta_dir.glob(f"delta_report__*__vs__{run_tag}.md"))

    if not delta_files:
        print(f"[FAIL] Delta 리포트 파일 없음 (run_tag: {run_tag})")
        return False

    # 가장 최신 파일 사용
    delta_file = delta_files[0]
    print(f"Delta 리포트 파일: {delta_file.name}")

    # 파일 읽기
    content = delta_file.read_text(encoding="utf-8")

    # baseline 태그 확인
    if expected_baseline in content:
        print(f"[PASS] baseline_tag '{expected_baseline}' 확인됨")
        # 파일 첫 부분 출력
        lines = content.split("\n")[:10]
        print("\nDelta 리포트 첫 10줄:")
        for line in lines:
            print(line)
        return True
    else:
        print(f"[FAIL] baseline_tag '{expected_baseline}'를 찾을 수 없음")
        print(f"파일 내용 일부:")
        print(content[:500])
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python check_stage11_validation.py <run_tag> [baseline_tag]")
        print("Example: python check_stage11_validation.py stage11_ui_payload_20251221_012244 stage10_market_regime_20251221_004433")
        sys.exit(1)

    run_tag = sys.argv[1]
    baseline_tag = sys.argv[2] if len(sys.argv) > 2 else None

    print("="*60)
    print(f"Stage11 체크리스트 검증")
    print("="*60)
    print(f"Run Tag: {run_tag}")
    if baseline_tag:
        print(f"Expected Baseline Tag: {baseline_tag}")
    print("="*60)

    results = []

    # 1. ui_equity_curves 검증
    results.append(("ui_equity_curves", check_ui_equity_curves(run_tag)))

    # 2. ui_top_bottom_daily 검증
    results.append(("ui_top_bottom_daily", check_ui_top_bottom_daily(run_tag, expected_top_k=10, expected_bottom_k=10)))

    # 3. UI Snapshot 검증
    results.append(("ui_snapshot", check_ui_snapshot(run_tag)))

    # 4. Δ 리포트 baseline_tag 확인
    if baseline_tag:
        results.append(("delta_baseline", check_delta_baseline(run_tag, baseline_tag)))

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
