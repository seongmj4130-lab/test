import os
import shutil
from pathlib import Path

def create_new_structure():
    """
    프로젝트를 모듈화하여 구조화합니다.
    baseline과 핵심 파일들은 절대 수정하지 않습니다.
    """
    base_dir = Path("c:/Users/seong/OneDrive/Desktop/bootcamp/000_code")

    print("🔄 프로젝트 구조화 시작...")

    # 새로운 폴더 구조 생성
    new_folders = [
        'scripts',      # 실행 스크립트들
        'experiments',  # 실험/분석 스크립트들
        'results',      # 결과 파일들 (PNG, TXT 등)
        'docs',         # 문서 파일들 (중복 정리)
    ]

    for folder in new_folders:
        (base_dir / folder).mkdir(exist_ok=True)
        print(f"📁 폴더 생성: {folder}/")

    # 이동할 파일들 정의 (분류별)
    moves = {
        # scripts 폴더로 이동 (프로젝트 실행용)
        'scripts': [
            'run_multiple_tests.py',
            'run_track_a_multiple_tests.py',
        ],

        # experiments 폴더로 이동 (분석/실험용)
        'experiments': [
            'analyze_track_a_performance.py',
            'calculate_combined_performance.py',
            'calculate_correct_ic_metrics.py',
            'calculate_track_a_ic_metrics.py',
            'create_baseline_backup.py',
            'create_strategy_cumulative_returns.py',
            'extract_holdout_data.py',
            'extract_performance_metrics.py',
            'temp_analysis.py',
            'test_feature_engineering.py',
            'enable_all_features.py',
        ],

        # results 폴더로 이동 (결과물)
        'results': [
            'backtest_strategy_comparison.png',
            'grid_output.txt',
            'test_final.txt',
            'test_output.txt',
            'test_result.txt',
        ],

        # docs 폴더로 이동 (문서 정리)
        'docs': [
            'ppt_report.md',
            'CLEANUP_SUMMARY.md',
        ],
    }

    # 파일 이동 실행
    for target_folder, files in moves.items():
        for file in files:
            src_path = base_dir / file
            dst_path = base_dir / target_folder / file

            if src_path.exists():
                shutil.move(str(src_path), str(dst_path))
                print(f"📄 이동: {file} → {target_folder}/")

    # 중복 파일 정리 (유사한 final_*.md 파일들)
    final_files = [
        'final_backtest_report.md',
        'final_easy_report.md',
        'final_ranking_report.md',
        'final_report.md'
    ]

    print("\n📋 중복 문서 파일 정리:")
    for file in final_files:
        src_path = base_dir / file
        if src_path.exists():
            # docs 폴더로 이동
            dst_path = base_dir / 'docs' / file
            shutil.move(str(src_path), str(dst_path))
            print(f"📄 이동: {file} → docs/")

    # backup_final_state.py는 experiments로 이동
    backup_file = base_dir / 'backup_final_state.py'
    if backup_file.exists():
        shutil.move(str(backup_file), str(base_dir / 'experiments' / 'backup_final_state.py'))
        print("📄 이동: backup_final_state.py → experiments/")

    # 빈 폴더 정리 (logs, reports 폴더가 비어있으면 삭제)
    empty_folders = ['logs', 'reports']
    for folder in empty_folders:
        folder_path = base_dir / folder
        if folder_path.exists() and not any(folder_path.iterdir()):
            folder_path.rmdir()
            print(f"🗑️ 빈 폴더 삭제: {folder}/")

    print("\n✅ 구조화 완료!")
    print("\n📂 새로운 폴더 구조:")
    print("├── configs/           # 설정 파일들 (유지)")
    print("├── src/              # 핵심 코드 (유지)")
    print("├── data/             # 데이터 파일들 (유지)")
    print("├── artifacts/        # 산출물 (유지)")
    print("├── baseline_*/       # Baseline 백업 (유지)")
    print("├── scripts/          # 실행 스크립트들")
    print("├── experiments/      # 분석/실험 스크립트들")
    print("├── results/          # 결과 파일들")
    print("├── docs/             # 문서 파일들")
    print("└── README.md         # 메인 문서")

    # README.md 업데이트
    update_readme(base_dir)

def update_readme(base_dir):
    """README.md에 새로운 구조 정보를 추가"""
    readme_path = base_dir / 'README.md'

    if readme_path.exists():
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 새로운 구조 정보를 추가
        structure_info = """

## 📂 프로젝트 폴더 구조 (모듈화 완료)

```
000_code/
├── configs/              # ⚙️ 설정 파일들
├── src/                  # 💻 핵심 소스 코드
├── data/                 # 📊 데이터 파일들
├── artifacts/            # 🏆 모델 및 산출물
├── baseline_*/           # 📦 Baseline 백업
├── scripts/              # 🚀 실행 스크립트들
│   ├── run_multiple_tests.py
│   └── run_track_a_multiple_tests.py
├── experiments/          # 🔬 분석/실험 스크립트들
│   ├── analyze_*.py
│   ├── calculate_*.py
│   ├── extract_*.py
│   └── test_*.py
├── results/              # 📈 결과 파일들
│   ├── *.png
│   └── *.txt
├── docs/                 # 📚 문서 파일들
│   ├── ppt_report.md
│   ├── final_*.md
│   └── *.md
└── README.md
```

### 📋 폴더 설명

- **configs/**: 모든 YAML 설정 파일들
- **src/**: Track A/B 구현, 데이터 파이프라인, 유틸리티
- **data/**: 원시/중간/최종 데이터 파일들
- **artifacts/**: 학습된 모델과 분석 리포트
- **baseline_*/**: 프로젝트 완료 시점의 완전 백업
- **scripts/**: 프로젝트 실행을 위한 메인 스크립트들
- **experiments/**: 분석, 실험, 테스트용 스크립트들
- **results/**: 차트, 로그, 출력 파일들
- **docs/**: 모든 문서 파일들 (PPT, 보고서 등)
"""

        # 기존 내용에 추가
        if "## 📂 프로젝트 폴더 구조" not in content:
            # 적절한 위치에 삽입 (프로젝트 개요 후)
            insert_pos = content.find("## 프로젝트 개요")
            if insert_pos != -1:
                content = content[:insert_pos] + structure_info + "\n" + content[insert_pos:]

                with open(readme_path, 'w', encoding='utf-8') as f:
                    f.write(content)

                print("📝 README.md 업데이트 완료")

if __name__ == "__main__":
    create_new_structure()