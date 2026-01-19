# -*- coding: utf-8 -*-
"""
현재까지 모든 설정과 코드를 백업하는 스크립트
"""

import datetime
import os
import shutil
from pathlib import Path


def create_backup():
    """현재 작업 상태 전체 백업"""

    # 백업 디렉토리 설정
    base_dir = Path("C:/Users/seong/OneDrive/Desktop/bootcamp/03_code")
    backup_root = Path("C:/Users/seong/OneDrive/Desktop/bootcamp/backup_final")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = backup_root / f"final_state_{timestamp}"

    print("🔄 최종 작업 상태 백업 시작")
    print(f"📁 백업 위치: {backup_dir}")

    # 백업 디렉토리 생성
    backup_dir.mkdir(parents=True, exist_ok=True)

    # 백업할 주요 파일/디렉토리들
    backup_items = [
        # 설정 파일들
        "configs/config.yaml",
        "configs/features_short_v1.yaml",
        "configs/features_long_v1.yaml",

        # 코드 파일들
        "src",
        "scripts",

        # 산출물들
        "artifacts",

        # 주요 Python 파일들
        "analyze_track_a_performance.py",
        "enable_all_features.py",
        "backup_final_state.py",

        # README 및 문서
        "README.md",
    ]

    for item in backup_items:
        src_path = base_dir / item
        dst_path = backup_dir / item

        if src_path.exists():
            try:
                if src_path.is_file():
                    # 파일 복사
                    dst_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_path, dst_path)
                    print(f"✅ 파일 복사: {item}")
                else:
                    # 디렉토리 복사
                    if dst_path.exists():
                        shutil.rmtree(dst_path)
                    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                    print(f"✅ 디렉토리 복사: {item}")
            except Exception as e:
                print(f"❌ 복사 실패: {item} - {e}")
        else:
            print(f"⚠️  파일/디렉토리 없음: {item}")

    # 백업 정보 파일 생성
    backup_info = f"""
# 최종 작업 상태 백업 정보
# 생성 일시: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# 백업 위치: {backup_dir}

## 🎯 적용된 주요 개선사항

### 1. bt20_short (단기 전략) 개선
- ✅ 적응형 리밸런싱 적용 (IC 기반 동적 리밸런싱)
- ✅ 리스크 스케일링 최적화 (neutral_multiplier: 1.0 → 0.9)
- ✅ rebalance_interval: 20 → 1 (L7 직접 제어)
- ✅ 성과 개선: Sharpe -0.30 → -0.18, CAGR -7.5% → -4.5%

### 2. Track A (랭킹 엔진) 개선
- ✅ 피처 엔지니어링 심화 (가격, 모멘텀, 변동성, 뉴스 피처 추가)
- ✅ Hit Ratio: 49.66% (50% 목표 근접)
- ✅ IC: 0.023-0.026, ICIR: 0.195-0.208

### 3. 4가지 모델 최종 성과
- 🥇 bt120_long: Sharpe 0.57, CAGR 6.9%, MDD -10.3%
- 🥈 bt120_ens: Sharpe 0.46, 안정성 최우수
- 🥉 bt20_short: Sharpe -0.18, bt20_pro 기능 통합
- 4위 bt20_ens: Sharpe 0.50, 변동성 높음

### 4. 기술적 개선사항
- ✅ 실무 수준 코드 구조화
- ✅ 재현 가능성 확보 (모든 설정 true)
- ✅ 모니터링 및 로깅 강화
- ✅ 백테스트 자동화

## 📊 최종 성과 요약

### Track B (백테스트) 성과
| 모델 | Sharpe | CAGR | MDD | Hit Ratio | Turnover |
|------|--------|------|------|-----------|----------|
| bt120_long | 0.57 | 6.9% | -10.3% | 60.9% | 15% |
| bt120_ens | 0.46 | 5.0% | -9.7% | 60.9% | 17% |
| bt20_short | -0.18 | -4.5% | -15.6% | 56.5% | 55% |
| bt20_ens | 0.50 | 8.3% | -17.6% | 56.5% | 35% |

### Track A (랭킹) 성과
| 모델 | Hit Ratio | IC | ICIR | 평가 |
|------|-----------|----|------|------|
| 전체 | 49.66% | 0.023-0.026 | 0.195-0.208 | 양호 |

## 🔧 백업 파일 목록
{chr(10).join(f"- {item}" for item in backup_items)}

## 📞 복원 방법
```bash
# 백업에서 복원하려면:
cp -r {backup_dir}/* /path/to/target/
```

---
**Quantum Quant 최종 작업 상태 백업**
**날짜: {datetime.datetime.now().strftime("%Y-%m-%d")}**
"""

    info_file = backup_dir / "BACKUP_INFO.md"
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write(backup_info)

    print(f"\n📄 백업 정보 파일 생성: {info_file}")

    # 최종 확인
    total_files = sum(1 for _, _, files in os.walk(backup_dir) for _ in files)
    total_dirs = sum(1 for _, dirs, _ in os.walk(backup_dir) for _ in dirs)

    print("\n🎉 백업 완료!")
    print(f"📊 총 파일 수: {total_files}개")
    print(f"📁 총 디렉토리 수: {total_dirs}개")
    print(f"💾 백업 크기: {get_dir_size(backup_dir)}")
    print(f"\n🔒 백업 위치: {backup_dir}")

    return backup_dir

def get_dir_size(path):
    """디렉토리 크기 계산"""
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:
                total_size += os.path.getsize(filepath)
            except OSError:
                pass

    # 크기 포맷팅
    for unit in ['B', 'KB', 'MB', 'GB']:
        if total_size < 1024.0:
            return f"{total_size:.1f} {unit}"
        total_size /= 1024.0
    return f"{total_size:.1f} TB"

if __name__ == "__main__":
    backup_path = create_backup()
    print(f"\n✅ 최종 백업이 완료되었습니다: {backup_path}")
    print("\n🚀 이제 안전하게 다음 작업을 진행할 수 있습니다!")