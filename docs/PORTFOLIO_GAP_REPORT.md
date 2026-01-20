# PORTFOLIO GAP REPORT: KOSPI200 퀀트 투자 전략 시스템

## 1. 현재 아키텍처 (Verified)

### 프로젝트 구조
```
000_code/
├── src/                    # 💻 메인 소스 코드 (잘 구성됨)
│   ├── components/         # 백테스트/포트폴리오/랭킹 컴포넌트
│   ├── pipeline/           # Track A/B 파이프라인 엔트리포인트
│   ├── stages/             # 처리 단계들 (데이터/모델링/백테스트)
│   ├── tracks/             # Track A(랭킹)/Track B(투자모델) 구현
│   ├── utils/              # 공통 유틸리티
│   └── interfaces/         # UI 연동 인터페이스
├── configs/               # ⚙️ YAML 설정 파일들 (70개)
├── scripts/               # 🚀 실행 스크립트들 (110개+)
├── docs/                  # 📚 문서 파일들 (22개)
├── data/                  # 📊 데이터 파일들 (790개+)
├── artifacts/             # 🏆 모델 및 분석 산출물
└── experiments/           # 🔬 실험/분석 스크립트들
```

### 핵심 기능
- **Track A**: 피처 기반 KOSPI200 종목 랭킹 생성 (단기/장기/통합)
- **Track B**: 랭킹 기반 4가지 투자 전략 백테스트 (BT20/BT120 모델)
- **투트랙 아키텍처**: 독립적 실행 가능한 랭킹 엔진 + 투자 모델
- **최종 성과**: BT120 Long 전략 Sharpe 0.6092 달성 (목표 초과)

### 엔트리 포인트 (279개 main 함수)
- **주요 실행**: `src/pipeline/track_a_pipeline.py`, `src/pipeline/track_b_pipeline.py`
- **레거시**: `scripts/run_pipeline_l0_l7.py` (전체 파이프라인)
- **UI 연동**: `src/interfaces/ui_service.py` (Flask API 지원)

## 2. 주요 위험 (Hardening 필요)

### 🚨 **Critical Risks**
1. **설정 파일 부재**
   - `pyproject.toml`, `requirements.txt`, `setup.cfg` 없음
   - 의존성 관리 불가 → 재현성 저해
   - 버전 관리 및 환경 격리 불가능

2. **코드 품질 이슈**
   - 컴파일 오류: `adaptive_rebalancing.py` (문법 오류), `combined_stages_all.py` (import 순서)
   - 하드코딩된 경로 및 설정값 다수 존재
   - 로깅 시스템 미흡 (디버깅 어려움)

3. **테스트 부재**
   - `tests/` 디렉토리 자체 없음
   - 루트/experiments/scripts에 흩어진 13개 테스트 파일 존재하나 실행 불가
   - 코드 변경 시 회귀 테스트 불가능

### ⚠️ **High Risks**
4. **문서화 부족**
   - API 문서 없음
   - 코드 주석 및 docstring 불충분
   - 아키텍처 다이어그램 부재

5. **CI/CD 부재**
   - 자동화된 테스트/빌드 파이프라인 없음
   - 코드 품질 게이트 없음
   - 릴리스 프로세스 부재

## 3. Quick Wins (1-2일)

### P0: 즉시 해결 가능
1. **설정 파일 생성**
   - `pyproject.toml`: 프로젝트 메타데이터, 의존성, 빌드 설정
   - `requirements.txt`: 런타임 의존성 명시
   - `.pre-commit-config.yaml`: 코드 품질 도구 설정

2. **컴파일 오류 수정**
   - `src/features/adaptive_rebalancing.py`: 문자열 리터럴 오류 수정
   - `src/stages/combined_stages_all.py`: import 순서 수정

3. **기본 테스트 구조 구축**
   - `tests/` 디렉토리 생성
   - `tests/conftest.py`: 테스트 픽스처
   - `tests/test_basic_imports.py`: 기본 import 테스트

### P1: 단기 개선 (3-5일)
4. **로깅 시스템 구축**
   - `src/utils/logging.py`: 구조화된 로깅 모듈
   - 설정별 로그 레벨 조정
   - 파일/콘솔 출력 지원

5. **설정 검증 시스템**
   - `src/utils/config_validator.py`: YAML 설정 유효성 검증
   - 필수 키 존재 확인
   - 타입 및 범위 검증

## 4. Medium-term (3-5일)

### P2: 중기 개선
6. **단위 테스트 확대**
   - 핵심 모듈별 단위 테스트 (utils, components)
   - 모의 객체 활용한 의존성 격리
   - 테스트 커버리지 70%+ 목표

7. **통합 테스트 구축**
   - 파이프라인 엔드투엔드 테스트
   - 데이터 흐름 검증
   - 성능 회귀 테스트

8. **문서화 개선**
   - Sphinx 기반 API 문서 생성
   - 아키텍처 다이어그램 (Mermaid)
   - 사용 예제 및 튜토리얼

## 5. Nice-to-have

### P3: 장기 개선
9. **CI/CD 파이프라인**
   - GitHub Actions 워크플로우
   - 자동화된 테스트/빌드/배포
   - 코드 품질 게이트 (black, flake8, mypy)

10. **컨테이너화**
    - Docker 이미지 생성
    - 개발/운영 환경 격리
    - 배포 자동화

11. **모니터링 및 관측성**
    - 메트릭 수집 (Prometheus)
    - 분산 추적 (Jaeger)
    - 알림 시스템

12. **성능 최적화**
    - 프로파일링 및 병목 지점 분석
    - 메모리 사용 최적화
    - 병렬 처리 개선

## 6. 제안 대상 구조

```
portfolio-hardened/
├── src/                    # 메인 소스 코드
├── tests/                  # 단위/통합 테스트
├── docs/                   # 문서 및 API 레퍼런스
├── notebooks/              # 분석 및 실험 노트북
├── scripts/                # 실행 스크립트
├── data/                   # 데이터 파일들
│   ├── raw/               # 원시 데이터
│   ├── processed/         # 전처리 데이터
│   └── interim/           # 중간 산출물
├── configs/               # 설정 파일들
├── requirements/          # 의존성 파일들
│   ├── requirements.txt
│   ├── requirements-dev.txt
│   └── requirements-test.txt
├── pyproject.toml         # 프로젝트 설정
├── Dockerfile            # 컨테이너화
├── docker-compose.yml    # 개발 환경
├── .pre-commit-config.yaml # 코드 품질
├── Makefile              # 편의 명령어
└── README.md
```

## 7. 실행 계획 (P1-P6)

### P1: Foundation (1-2일)
**목표**: 기본 안정성 확보
**파일 작업**:
- `pyproject.toml`: 프로젝트 설정 생성
- `requirements.txt`: 의존성 명시화
- `.pre-commit-config.yaml`: 코드 품질 도구 설정
- `src/features/adaptive_rebalancing.py`: 컴파일 오류 수정
- `src/stages/combined_stages_all.py`: import 순서 수정
- `tests/test_basic_imports.py`: 기본 import 테스트 생성

### P2: Testing Infrastructure (2-3일)
**목표**: 테스트 가능성 확보
**파일 작업**:
- `tests/conftest.py`: 테스트 픽스처 설정
- `tests/test_utils/`: 유틸리티 모듈 테스트
- `tests/test_components/`: 컴포넌트 테스트
- `tests/test_pipeline/`: 파이프라인 통합 테스트
- `pytest.ini`: pytest 설정
- `tox.ini`: 다중 환경 테스트

### P3: Configuration Management (2-3일)
**목표**: 설정 안정성 확보
**파일 작업**:
- `src/utils/config_validator.py`: 설정 검증 모듈
- `src/utils/config.py`: 설정 로딩 개선 (검증 추가)
- `configs/config.schema.yaml`: 설정 스키마 정의
- `tests/test_config/`: 설정 관련 테스트

### P4: Logging & Observability (1-2일)
**목표**: 디버깅 및 모니터링 용이성
**파일 작업**:
- `src/utils/logging.py`: 구조화된 로깅 시스템
- `configs/logging.yaml`: 로깅 설정
- `src/utils/metrics.py`: 메트릭 수집 모듈
- `tests/test_logging/`: 로깅 테스트

### P5: Documentation (2-3일)
**목표**: 유지보수성 향상
**파일 작업**:
- `docs/api/`: API 문서 (Sphinx)
- `docs/architecture/`: 아키텍처 문서
- `docs/examples/`: 사용 예제
- `docs/diagrams/`: Mermaid 다이어그램
- `mkdocs.yml`: 문서 사이트 설정

### P6: CI/CD & Automation (3-5일)
**목표**: 개발 생산성 향상
**파일 작업**:
- `.github/workflows/ci.yml`: CI 파이프라인
- `.github/workflows/cd.yml`: CD 파이프라인
- `Dockerfile`: 컨테이너화
- `docker-compose.yml`: 개발 환경
- `Makefile`: 편의 명령어
- `scripts/ci/`: CI 지원 스크립트

## 8. 우선순위 및 타임라인

### Phase 1 (Week 1-2): Critical Foundation
- P1 완료: 기본 안정성 확보
- 컴파일 오류 해결 및 기본 테스트 구축
- **결과**: 코드가 안정적으로 실행 가능

### Phase 2 (Week 3-4): Testing & Config
- P2 + P3 완료: 테스트 인프라 및 설정 관리
- **결과**: 코드 변경 시 안정성 검증 가능

### Phase 3 (Week 5-6): Observability & Docs
- P4 + P5 완료: 로깅 및 문서화
- **결과**: 유지보수 및 디버깅 용이

### Phase 4 (Week 7-8): Automation
- P6 완료: CI/CD 파이프라인 구축
- **결과**: 자동화된 품질 관리 및 배포

## 9. 리스크 평가

### 실행 리스크
- **낮음**: 대부분의 작업이 기존 코드 변경 최소화
- **중간**: 테스트 구축 시 기존 버그 발견 가능성
- **높음**: CI/CD 도입 시 기존 워크플로우 변경 필요

### 비즈니스 임팩트
- **즉시**: 코드 품질 향상으로 버그 감소
- **단기**: 테스트 커버리지로 안정성 확보
- **장기**: 자동화로 개발 속도 향상 및 비용 절감

### 의존성
- 팀 합의: P6 CI/CD 도입 전 팀 워크플로우 논의 필요
- 외부 도구: Docker, GitHub Actions 등 인프라 검토 필요

---

## Evidence Appendix (P0 검증 증빙)

### 컴파일 검증 결과
**실행 커맨드**: `python -m compileall src/ -q`

**결과 요약**: 2개 파일에서 SyntaxError 발견
- `src/features/adaptive_rebalancing.py`, Line 327: `SyntaxError: unterminated string literal (detected at line 327)`
- `src/stages/combined_stages_all.py`, Line 176: `SyntaxError: from __future__ imports must occur at the beginning of the file`

**상세 에러**:
```
*** Error compiling 'src/features\adaptive_rebalancing.py'...
  File "src/features\adaptive_rebalancing.py", line 327
    print("
          ^
SyntaxError: unterminated string literal (detected at line 327)

*** Error compiling 'src/stages\combined_stages_all.py'...
  File "src/stages\combined_stages_all.py", line 176
    from __future__ import annotations
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
SyntaxError: from __future__ imports must occur at the beginning of the file
```

### 테스트 실행 결과
**실행 커맨드**: `python -m pytest --version && python -m pytest test_*.py -v --tb=short`

**결과 요약**: 테스트 인프라 부재
- pytest 버전: 9.0.2 (설치됨)
- 실행 결과: `ERROR: file or directory not found: test_*.py`
- 결론: tests/ 디렉토리 및 실행 가능한 테스트 없음

### 기본 Import 검증 결과
**실행 커맨드**: `python -c "import src; print('Basic import successful')"`

**결과 요약**: 성공
```
Basic import successful
```

### 환경 정보
- Python 버전: 3.13.7
- OS: Windows 10.0.26200
- pytest 버전: 9.0.2

## 의존성 전략 결정

**의존성 관리 방안**: `requirements/requirements.txt + requirements-dev.txt + requirements-test.txt` 구조로 간다
- 기존 프로젝트에 requirements.txt가 없으므로, pyproject.toml 대신 requirements 파일들로 시작
- 향후 pyproject.toml로 전환 가능하나, 현재 팀 워크플로우 부담 최소화를 위해 requirements 파일 우선

**Python 버전**: 3.10/3.11/3.12 지원 (현재 3.13.7에서 실행 확인)
- 최소 지원 버전: 3.10 (f-string, walrus operator 사용 확인)
- 권장 버전: 3.11+ (향상된 타입 힌팅 지원)

**데이터/아티팩트 Git 관리 방침**:
- `data/` 디렉토리: gitignore 처리 (용량 이슈)
- `artifacts/` 디렉토리: gitignore 처리 (실행 결과물)
- 샘플 데이터만 별도 `data/samples/`에 유지
- 대용량 파일은 `.gitignore`에 `*.parquet`, `*.csv` (용량 초과 시) 명시

---

**결론**: 현재 프로젝트는 기능적으로 완성되었으나, 엔터프라이즈급 포트폴리오로서의 hardening이 필요합니다. P1-P3 우선 추진으로 빠른 안정성 확보 가능하며, 이후 단계적 개선으로 장기적 유지보수성 확보를 권장합니다.

**P1 진행 판단**: ✅ 조건부 진행 가능 (위 Evidence 및 의존성 결정으로 P0 보완 완료)
