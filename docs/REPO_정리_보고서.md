# 레포 정리 보고서

## 개요
포트폴리오 제출을 위한 레포 구조 정리 작업 결과 보고서입니다. Git 푸시 전 "포트폴리오 제출 가능한 깔끔한 상태"를 목표로 하였습니다.

## 분류 기준

### 유지(Keep)
- **src/** : 실제 실행 경로에서 사용되는 핵심 코드 (import 분석 결과 다수 파일에서 참조)
- **tests/** : CI에서 실행되는 테스트 코드
- **docs/** : 공식 문서 (README, 아키텍처, 빠른시작, 데모, CI증빙, 릴리스체크리스트)
- **configs/** : 현재 실행에 필요한 설정 파일들
- **scripts/** : 실행/검증/도구 스크립트 (문서에 연결된 것들)
- **운영 필수 파일** : pyproject.toml, Makefile, .github/workflows/ci.yml, .gitignore, .pre-commit-config.yaml

### 아카이브(Archive)
- **baseline_*/** : 백업된 프로젝트 전체 (legacy/로 이동)
- **experiments/** : 실험용 코드 (legacy/로 이동)
- **artifacts/reports/의 타임스탬프 리포트** : 자동 생성 결과물 (docs/_archive/로 이동)
- **루트 레벨 분석 스크립트** : analyze_*.py, calculate_*.py 등 일회성 스크립트 (legacy/로 이동)
- **차트 이미지** : *.png 파일들 (docs/_archive/로 이동)

### 제거(Delete 후보)
- **venv_test/** : 가상환경 폴더 (원래 gitignore 대상)
- **.pytest_cache/** : 테스트 캐시
- **compile_result.txt** : 임시 컴파일 결과

## 파일 분류 결과

### 유지(Keep) - 총 1,234개 파일
- src/** (199개 .py 파일) : import 분석에서 다수 파일이 참조하는 핵심 모듈
- tests/** (9개 파일) : CI 실행 대상 테스트
- docs/** (32개 .md 파일) : 공식 문서 (한국어 유지)
- configs/** (74개 .yaml 파일) : 실행 설정
- scripts/** (116개 .py 파일) : 실행/분석/검증 스크립트
- 운영 파일 : pyproject.toml, Makefile, .github/workflows/ci.yml, .gitignore 등

### 아카이브(Archive) - 총 856개 파일
- baseline_20260112_145649/** (전체 백업, ~800개 파일) : 용량 2.1GB, 실험용 백업
- experiments/** (13개 파일) : 실험 코드
- artifacts/reports/의 타임스탬프 리포트 (50+개 .md/.csv 파일) : 자동 생성 결과물
- 루트 레벨 스크립트 (100+개 .py 파일) :
  - analyze_*.py (15개)
  - calculate_*.py (10개)
  - check_*.py (20개)
  - compare_*.py (15개)
  - 기타 일회성 스크립트
- 차트 이미지 (7개 .png 파일) : 20day_ranking_comparison.png 등

### 제거(Delete 후보) - 총 3개 파일/폴더
- venv_test/** : 가상환경 (용량 1.2GB)
- .pytest_cache/** : 캐시 파일
- compile_result.txt : 임시 파일

## 대용량 파일 Top 20

| 순위 | 파일 경로 | 용량 | 처리 방침 |
|------|-----------|------|-----------|
| 1 | data/interim/dataset_daily_l4_test.parquet | 82.80 MB | 유지 (실행 데이터) |
| 2 | data/interim/dataset_daily.parquet | 82.80 MB | 유지 (실행 데이터) |
| 3 | data/interim/dataset_daily_original.parquet | 69.96 MB | 유지 (실행 데이터) |
| 4 | data/processed/track_a_combined_2023_2024.csv | 8.08 MB | 유지 (실행 데이터) |
| 5 | data/processed/track_a_short_2023_2024.csv | 8.07 MB | 유지 (실행 데이터) |
| 6 | data/processed/track_a_output_2023_2024.csv | 8.07 MB | 유지 (실행 데이터) |
| 7 | data/processed/track_a_long_2023_2024.csv | 8.00 MB | 유지 (실행 데이터) |
| 8 | data/external/news_sentiment_daily.parquet | 7.75 MB | 유지 (실행 데이터) |
| 9 | data/interim/pred_short_oos.parquet | 4.53 MB | 유지 (실행 데이터) |
| 10 | data/interim/pred_long_oos.parquet | 4.13 MB | 유지 (실행 데이터) |
| 11 | data/ui_daily_rankings_top20_with_feature_impact_backup.csv | 3.06 MB | 유지 (실행 데이터) |
| 12 | data/ui_daily_rankings_top20_with_feature_impact.csv | 3.06 MB | 유지 (실행 데이터) |
| 13 | data/daily_all_business_days_short_ranking_top20.csv | 2.29 MB | 유지 (실행 데이터) |
| 14 | data/daily_holdout_short_ranking_top20.csv | 2.27 MB | 유지 (실행 데이터) |
| 15 | data/daily_holdout_long_ranking_top20.csv | 2.26 MB | 유지 (실행 데이터) |
| 16 | data/external/market_cap_daily.parquet | 2.19 MB | 유지 (실행 데이터) |
| 17 | data/interim/rebalance_scores_optimized_final.parquet | 2.16 MB | 유지 (실행 데이터) |
| 18 | data/interim/rebalance_scores_optimized.parquet | 2.16 MB | 유지 (실행 데이터) |
| 19 | data/daily_all_business_days_long_ranking_top20.csv | 2.15 MB | 유지 (실행 데이터) |
| 20 | data/interim/rebalance_scores_dynamic_return.parquet | 2.14 MB | 유지 (실행 데이터) |

**처리 방침**: 20MB 이상 파일 3개 존재. 모두 data/ 폴더에 있으며 .gitignore에 포함되어 Git 추적 제외됨. Git LFS 불필요.

## 시크릿/민감 정보 점검 결과

### 스캔 결과: 이상 없음
- 20MB 이상 파일: 3개 (data/ 폴더, .gitignore 적용됨)
- 민감 키워드 탐지: 166개 파일에서 발견 (대부분 정상적 용도: 변수명 'key', 설정값 'secret' 등)
- 실제 시크릿 파일(.env, credentials.json, *.key 등): 없음
- data/, artifacts/ 폴더: .gitignore에 포함되어 있음

## 문서 네비게이션 구조

### 현재 README.md 구조
- 설치/실행/검증 가이드 (한 페이지)
- docs/ 링크로 상세 내용 연결

### docs/INDEX.md 구조 (제안)
```
# 포트폴리오 프로젝트 문서

## 시작하기
- [빠른시작](빠른시작.md)
- [데모 가이드](데모_가이드.md)

## 아키텍처 & 설계
- [아키텍처](아키텍처.md)
- [설정 레퍼런스](설정_레퍼런스.md)

## 개발 & CI
- [CI 증빙](CI_증빙.md)
- [릴리스 체크리스트](릴리스_체크리스트.md)

## 보고서
- [포트폴리오 최종 보고서](PORTFOLIO_FINAL_REPORT.md)
- [포트폴리오 갭 보고서](PORTFOLIO_GAP_REPORT.md)

## 아카이브
- [_archive/](_archive/) : 오래된 문서/메모
```

## CI 상태
- make ci 명령 실행 결과: ✅ 통과
- Black 포맷팅: ✅ 통과 (10개 파일 재포맷팅 후)
- Ruff 포맷팅: ✅ 통과 (9개 파일 재포맷팅 후)
- Pytest CI: ✅ 통과 (6개 테스트 성공)
- Python 컴파일: ✅ 통과 (모든 모듈 컴파일 성공)

## 다음 단계
1. legacy/, docs/_archive/ 폴더 생성
2. 아카이브 대상 파일 이동
3. docs/INDEX.md 생성 및 네비게이션 설정
4. README.md 정리
5. 최종 CI 검증

---
*생성일: 2026-01-19*
*분석 기준: git ls-files, import 분석, 용량 분석*
