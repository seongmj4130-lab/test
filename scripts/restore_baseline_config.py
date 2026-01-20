"""
Baseline 설정 복원 스크립트

백업된 baseline 설정으로 복원합니다.
"""

import shutil
from pathlib import Path


def restore_baseline_config():
    """
    Baseline 설정으로 복원
    """
    configs_dir = Path("configs")

    print("🔄 Baseline 설정 복원 시작...")
    print("=" * 50)

    # 복원할 파일들
    restore_files = {
        "config.yaml": "config_baseline_backup.yaml",
        "features_short_v1.yaml": "features_short_v1_baseline_backup.yaml",
        "features_long_v1.yaml": "features_long_v1_baseline_backup.yaml",
    }

    restored_count = 0

    for target_file, backup_file in restore_files.items():
        target_path = configs_dir / target_file
        backup_path = configs_dir / backup_file

        if backup_path.exists():
            print(f"📋 {target_file} 복원 중...")
            shutil.copy2(backup_path, target_path)
            print(f"✅ {target_file} 복원 완료")
            restored_count += 1
        else:
            print(f"⚠️ {backup_file} 백업 파일이 존재하지 않음")

    print(f"\n📊 복원 결과: {restored_count}/{len(restore_files)}개 파일 복원 완료")
    print("=" * 50)
    print("🎯 Baseline 설정으로 복원되었습니다.")
    print("🚀 이제 Track A/B를 재실행하여 baseline 성과를 확인할 수 있습니다.")


def show_backup_status():
    """
    백업 상태 확인
    """
    configs_dir = Path("configs")

    print("📦 백업 파일 상태")
    print("=" * 30)

    backup_files = [
        "config_baseline_backup.yaml",
        "features_short_v1_baseline_backup.yaml",
        "features_long_v1_baseline_backup.yaml",
    ]

    for backup_file in backup_files:
        backup_path = configs_dir / backup_file
        if backup_path.exists():
            size = backup_path.stat().st_size
            mtime = backup_path.stat().st_mtime
            print(f"✅ {backup_file}: {size} bytes")
        else:
            print(f"❌ {backup_file}: 없음")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--status":
        show_backup_status()
    else:
        restore_baseline_config()
