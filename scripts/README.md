# Scripts 폴더

프로젝트 실행을 위한 메인 스크립트들이 모여있는 폴더입니다.

## 주요 스크립트들

### 실행 스크립트
- `run_multiple_tests.py`: 다중 백테스트 실행
- `run_track_a_multiple_tests.py`: Track A 다중 테스트 실행

### 분석 스크립트
- `analyze_*.py`: 각종 분석 스크립트들
- `check_*.py`: 데이터 검증 스크립트들
- `compare_*.py`: 성과 비교 스크립트들

### 최적화 스크립트
- `optimize_*.py`: 모델/파라미터 최적화
- `grid_search_*.py`: 그리드 서치 실행

### 유틸리티 스크립트
- `export_*.py`: 데이터 export
- `generate_*.py`: 데이터 생성
- `show_*.py`: 정보 표시

## 사용법

```bash
# 기본 실행
python scripts/run_multiple_tests.py

# Track A 실행
python scripts/run_track_a_multiple_tests.py
```

## 주의사항

- 이 스크립트들은 프로젝트의 핵심 실행 로직을 포함합니다
- 수정 시 baseline과 비교하여 검증하세요
- 새로운 스크립트 추가 시 이 폴더에 배치하세요
