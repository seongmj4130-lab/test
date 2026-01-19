# -*- coding: utf-8 -*-
# C:/Users/seong/OneDrive/Desktop/bootcamp/03_code/src/tools/check_replay_progress.py
"""
리플레이 진행 상황 실시간 확인 스크립트
"""
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path("C:/Users/seong/OneDrive/Desktop/bootcamp/03_code")

def check_progress():
    """리플레이 진행 상황 확인"""
    base_dir = PROJECT_ROOT
    base_interim_dir = base_dir / "data" / "interim"
    analysis_dir = base_dir / "reports" / "analysis"

    print("=" * 80)
    print("리플레이 진행 상황 실시간 확인")
    print(f"확인 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()

    # 1. 최신 run_tag 확인
    print("## 1. 최신 실행 태그")
    stage_dirs = sorted(
        [d for d in base_interim_dir.iterdir() if d.is_dir() and d.name.startswith("stage")],
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

    if stage_dirs:
        print(f"최신 실행 태그: {stage_dirs[0].name}")
        print(f"생성 시간: {datetime.fromtimestamp(stage_dirs[0].stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
        print()

        # 최신 태그의 산출물 확인
        latest_dir = stage_dirs[0]
        artifacts = list(latest_dir.glob("*.parquet"))
        print(f"생성된 산출물 수: {len(artifacts)}")
        if artifacts:
            print("주요 산출물:")
            for art in sorted(artifacts, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
                print(f"  - {art.name} ({datetime.fromtimestamp(art.stat().st_mtime).strftime('%H:%M:%S')})")
    else:
        print("실행 중인 Stage가 없습니다.")
    print()

    # 2. Summary 리포트 확인
    print("## 2. 요약 리포트 상태")
    summary_md = analysis_dir / "stage_replay_summary_0_13.md"
    if summary_md.exists():
        print(f"요약 리포트: {summary_md}")
        print(f"수정 시간: {datetime.fromtimestamp(summary_md.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")

        # 내용 일부 읽기
        try:
            with open(summary_md, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines[:15]:
                    print(f"  {line.rstrip()}")
        except Exception as e:
            print(f"  읽기 실패: {e}")
    else:
        print("요약 리포트가 아직 생성되지 않았습니다.")
    print()

    # 3. CSV 파일 확인
    print("## 3. 분석 CSV 파일 상태")
    csv_files = {
        "진화 CSV": analysis_dir / "stage_evolution_0_13.csv",
        "변화량 CSV (vs-prev)": analysis_dir / "stage_changes_vs_prev_0_13.csv",
        "변화량 CSV (vs-baseline)": analysis_dir / "stage_changes_vs_baseline_0_13.csv",
    }

    for name, path in csv_files.items():
        if path.exists():
            mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            try:
                df = pd.read_csv(path)
                print(f"{name}: [OK] 생성됨 ({mtime}, {len(df)}행)")
            except Exception as e:
                print(f"{name}: [OK] 생성됨 ({mtime}, 읽기 실패: {e})")
        else:
            print(f"{name}: [X] 미생성")
    print()

    # 4. Stage별 진행 상황 추정
    print("## 4. Stage별 진행 상황 추정")
    stage_numbers = []
    for stage_dir in stage_dirs[:14]:  # 최신 14개만 확인
        name = stage_dir.name
        if name.startswith("stage") and "_" in name:
            try:
                stage_num = int(name.split("_")[0].replace("stage", ""))
                if 0 <= stage_num <= 13:
                    stage_numbers.append((stage_num, name, stage_dir.stat().st_mtime))
            except:
                pass

    if stage_numbers:
        stage_numbers.sort(key=lambda x: x[0])  # stage_no 순으로 정렬
        print("완료된 Stage (추정):")
        for stage_no, name, mtime in stage_numbers:
            time_str = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            # 산출물 확인
            has_bt = (base_interim_dir / name / "bt_metrics.parquet").exists()
            has_ranking = (base_interim_dir / name / "ranking_daily.parquet").exists()
            artifact_status = ""
            if has_bt:
                artifact_status = " [백테스트 산출물 있음]"
            elif has_ranking:
                artifact_status = " [랭킹 산출물 있음]"
            print(f"  Stage {stage_no}: {name} ({time_str}){artifact_status}")
    else:
        print("완료된 Stage가 없습니다.")
    print()

    print("=" * 80)

if __name__ == "__main__":
    check_progress()
