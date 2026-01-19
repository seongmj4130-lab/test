# Docs 폴더

프로젝트 관련 모든 문서 파일들이 모여있는 폴더입니다.

## 주요 문서들

### 프레젠테이션 자료
- `ppt_report.md`: 최종 발표용 PPT 보고서 (완전판)
- `final_report.md`: 최종 프로젝트 보고서

### 상세 보고서들
- `final_backtest_report.md`: 백테스트 상세 결과
- `final_ranking_report.md`: 랭킹 엔진 분석 결과
- `final_easy_report.md`: 간단 버전 보고서

### 기술 문서들
- `PIPELINE_DOCUMENTATION_FOR_TEAM_PRESENTATION.md`: 파이프라인 기술 문서
- `TWO_TRACK_WITH_L5_BENEFITS.md`: 투트랙 아키텍처 설명
- `ENSEMBLE_RANKING_STRATEGY.md`: 앙상블 전략 설명

### 설정 및 사용법
- `CONFIG_PRIORITY_DOCUMENTATION.md`: 설정 우선순위 문서
- `CROSS_SECTIONAL_RANK_USAGE.md`: 랭킹 사용법
- `FINAL_METRICS_DEFINITION.md`: 평가 지표 정의

### 실험 및 개선
- `HIT_RATIO_IMPROVEMENT_PLAN.md`: Hit Ratio 개선 계획
- `L5_PREDICTION_TARGET.md`: 예측 타겟 설명
- `ALPHA_SHORT_EXPLANATION.md`: Alpha 파라미터 설명

## 폴더 구조

```
docs/
├── ppt_report.md                    # 🎯 최종 발표 자료
├── final_*.md                       # 📋 최종 보고서들
├── *_DOCUMENTATION.md               # 📖 기술 문서들
├── *_EXPLANATION.md                 # 💡 설명 문서들
├── *_PLAN.md                        # 📝 계획 문서들
├── *_DEFINITION.md                  # 📏 정의 문서들
└── README.md
```

## 사용법

### 발표 준비
```bash
# PPT 자료 확인
cat docs/ppt_report.md

# 최종 보고서 확인
cat docs/final_report.md
```

### 기술 문서 참조
```bash
# 파이프라인 문서
cat docs/PIPELINE_DOCUMENTATION_FOR_TEAM_PRESENTATION.md

# 평가 지표 정의
cat docs/FINAL_METRICS_DEFINITION.md
```

## 주의사항

- 이 폴더의 문서들은 프로젝트의 공식 문서입니다
- 수정 시 버전 관리를 철저히 하세요
- 중요한 변경사항은 팀과 공유하세요
- Baseline 문서와 비교하여 검증하세요