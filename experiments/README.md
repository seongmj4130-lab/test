# Experiments 폴더

프로젝트 분석, 실험, 테스트를 위한 스크립트들이 모여있는 폴더입니다.

## 주요 스크립트들

### 성과 분석
- `analyze_track_a_performance.py`: Track A 성과 분석
- `calculate_*.py`: 성과 계산 및 메트릭 산출

### 데이터 추출/가공
- `extract_*.py`: 데이터 추출 및 가공
- `create_*.py`: 데이터 생성 스크립트

### 실험/테스트
- `test_*.py`: 테스트 스크립트들
- `temp_analysis.py`: 임시 분석
- `enable_all_features.py`: 피처 활성화 테스트

### 백업/관리
- `backup_final_state.py`: 상태 백업
- `create_baseline_backup.py`: Baseline 생성

## 사용법

```bash
# 성과 분석
python experiments/analyze_track_a_performance.py

# 데이터 추출
python experiments/extract_performance_metrics.py

# Baseline 생성
python experiments/create_baseline_backup.py
```

## 폴더 구조

```
experiments/
├── analyze_*.py      # 분석 스크립트
├── calculate_*.py    # 계산 스크립트
├── create_*.py       # 생성 스크립트
├── extract_*.py      # 추출 스크립트
├── test_*.py         # 테스트 스크립트
└── README.md
```

## 주의사항

- 이 폴더의 스크립트들은 실험/분석 목적입니다
- 프로덕션 코드에 영향을 주지 않도록 설계되었습니다
- 새로운 실험은 이 폴더에 추가하세요
- baseline에 영향을 주지 않도록 주의하세요