import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(".")
print("🔬 재현성 테스트 시작 (초간단 버전)")

# 백업
interim_dir = PROJECT_ROOT / "data" / "interim"
backup_dir = PROJECT_ROOT / "data" / "backup_repro_test"
backup_dir.mkdir(exist_ok=True)

bt_files = [
    "bt_metrics_bt20_ens.parquet",
    "bt_metrics_bt20_short.parquet",
    "bt_metrics_bt120_ens.parquet",
    "bt_metrics_bt120_long.parquet",
]

print("📋 기존 결과 백업 중...")
for file in bt_files:
    src = interim_dir / file
    if src.exists():
        shutil.copy2(src, backup_dir / f"{file}.backup")
print("✅ 백업 완료")

# 3번 재실행 및 결과 수집
results = {}
for run_id in range(1, 4):
    print(f"\n🚀 RUN {run_id}/3 시작")

    # 캐시 삭제
    for file in bt_files:
        file_path = interim_dir / file
        if file_path.exists():
            file_path.unlink()

    # 백테스트 실행
    result = subprocess.run(
        [sys.executable, "scripts/run_backtest_4models.py"],
        capture_output=True,
        text=True,
        cwd=PROJECT_ROOT,
    )

    if result.returncode == 0:
        print(f"✅ RUN {run_id} 완료")

        # 결과 저장
        run_results = {}
        for file in bt_files:
            try:
                df = pd.read_parquet(interim_dir / file)
                holdout = df[df["phase"] == "holdout"]
                if len(holdout) > 0:
                    run_results[file.replace(".parquet", "")] = {
                        "sharpe": holdout["net_sharpe"].iloc[0],
                        "cagr": holdout["net_cagr"].iloc[0],
                        "mdd": holdout["net_mdd"].iloc[0],
                        "calmar": holdout["net_calmar_ratio"].iloc[0],
                    }
            except:
                print(f"⚠️ {file} 읽기 실패")
        results[f"run_{run_id}"] = run_results
    else:
        print(f"❌ RUN {run_id} 실패")
        break

# 결과 분석
print("\n📊 재현성 분석")
for strategy in [
    "bt_metrics_bt20_ens",
    "bt_metrics_bt20_short",
    "bt_metrics_bt120_ens",
    "bt_metrics_bt120_long",
]:
    print(f"\n🎯 {strategy}")
    sharpes = []
    for run_id in range(1, 4):
        run_key = f"run_{run_id}"
        if run_key in results and strategy in results[run_key]:
            sharpe = results[run_key][strategy]["sharpe"]
            sharpes.append(sharpe)
            print(f"  RUN {run_id}: Sharpe = {sharpe:.6f}")

    if len(sharpes) == 3:
        std = pd.Series(sharpes).std()
        cv = (
            std / abs(pd.Series(sharpes).mean())
            if pd.Series(sharpes).mean() != 0
            else 0
        )
        print(f"  표준편차: {std:.6f}")
        print(f"  변동계수: {cv:.6f}")

        if cv < 0.001:
            grade = "⭐⭐⭐⭐⭐ 완벽"
        elif cv < 0.01:
            grade = "⭐⭐⭐⭐ 우수"
        elif cv < 0.05:
            grade = "⭐⭐⭐ 양호"
        else:
            grade = "⭐⭐ 보통"

        print(f"  재현성: {grade}")

# 백업 복원
print("\n🔄 백업 복원 중...")
for file in bt_files:
    backup_file = backup_dir / f"{file}.backup"
    if backup_file.exists():
        shutil.copy2(backup_file, interim_dir / file)
print("✅ 복원 완료")

print("\n🏆 재현성 테스트 완료")
