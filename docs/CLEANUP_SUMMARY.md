# 파일 정리 완료 보고서

## 작업 완료 일시
2026-01-06 15:20

## 작업 내용

### 1. 파일 정리 완료 ✅
README.md 기준으로 필요한 파일만 유지하고 나머지는 `05_backup` 폴더로 이동했습니다.

#### 유지된 항목:
- ✅ `src/` (전체 소스 코드)
- ✅ `configs/config.yaml` (메인 설정 파일)
- ✅ `data/raw/` (기초데이터)
- ✅ `data/external/` (기초데이터)
- ✅ `data/interim/` (기초데이터 + 산출물)
  - 기초데이터: `universe_k200_membership_monthly.*`, `ohlcv_daily.*`, `fundamentals_annual.*`, `panel_merged_daily.*`, `dataset_daily.*`, `cv_folds_*`
  - 산출물: `ranking_short_daily.*`, `ranking_long_daily.*`, `rebalance_scores_from_ranking.*`
- ✅ `scripts/run_pipeline_l0_l7.py` (기초데이터 수집용)
- ✅ `final_*.md` (4개 파일)
  - `final_report.md`
  - `final_easy_report.md`
  - `final_backtest_report.md`
  - `final_ranking_report.md`
- ✅ `README.md`

#### 이동된 항목:
- ✅ `backup/` → `05_backup/backup/`
- ✅ `backups/` → `05_backup/backups/`
- ✅ `ui/` → `05_backup/ui/`
- ✅ `tests/` → `05_backup/tests/`
- ✅ `docs/` → `05_backup/docs/`
- ✅ `artifacts/` → `05_backup/artifacts/`
- ✅ `configs/` (config.yaml 제외) → `05_backup/configs/`
- ✅ `scripts/` (run_pipeline_l0_l7.py 제외) → `05_backup/scripts/`
- ✅ `data/interim/` (기초데이터 + 산출물 제외) → `05_backup/data/interim/`
- ✅ `data/artifacts/` → `05_backup/data/artifacts/`
- ✅ `data/final_report/` → `05_backup/data/final_report/`
- ✅ `data/interim_backup_20251224/` → `05_backup/data/interim_backup_20251224/`
- ✅ `data/report_core_L0_L7/` → `05_backup/data/report_core_L0_L7/`
- ✅ `data/snapshots/` → `05_backup/data/snapshots/`

### 2. 산출물 재생성 ⚠️
**상태**: 네트워크 오류로 보류됨

**원인**: 
- Track A 파이프라인 실행 시 UI Payload Builder에서 pykrx 라이브러리의 외부 API 호출 실패
- Track B 파이프라인 실행 시 시장 국면 데이터 생성 중 외부 API 호출 실패

**조치 필요**:
1. 네트워크 연결 확인
2. pykrx 라이브러리 업데이트 또는 대체 방법 검토
3. 외부 API 호출이 필요한 부분을 캐시된 데이터로 우회하는 방법 검토

**재생성 필요 산출물**:
- Track A: `ranking_short_daily`, `ranking_long_daily` (이미 존재하지만 재생성 권장)
- Track B: 
  - `rebalance_scores_from_ranking` (이미 존재하지만 재생성 권장)
  - `bt_metrics_bt20_short`, `bt_metrics_bt20_ens`, `bt_metrics_bt120_long`, `bt_metrics_bt120_ens`
  - `bt_returns_*`, `bt_equity_curve_*`, `bt_positions_*` 등

## 현재 상태

### 03_code 디렉토리 구조
```
03_code/
├── src/                    # 소스 코드 (전체 유지)
├── configs/
│   └── config.yaml        # 메인 설정 파일만 유지
├── data/
│   ├── raw/               # 기초데이터 (유지)
│   ├── external/          # 기초데이터 (유지)
│   └── interim/            # 기초데이터 + 산출물 (필요한 것만 유지)
├── scripts/
│   └── run_pipeline_l0_l7.py  # 기초데이터 수집용만 유지
├── final_*.md             # 4개 파일 유지
└── README.md              # 유지
```

### 05_backup 디렉토리 구조
```
05_backup/
├── backup/
├── backups/
├── ui/
├── tests/
├── docs/
├── artifacts/
├── configs/               # config.yaml 제외한 모든 설정 파일
├── scripts/                # run_pipeline_l0_l7.py 제외한 모든 스크립트
└── data/
    ├── artifacts/
    ├── final_report/
    ├── interim_backup_20251224/
    ├── report_core_L0_L7/
    └── snapshots/
```

## 다음 단계

1. **네트워크 문제 해결 후 산출물 재생성**
   ```bash
   # Track A 실행
   python -m src.pipeline.track_a_pipeline
   
   # Track B 실행 (4가지 전략)
   python -m src.pipeline.track_b_pipeline bt20_short
   python -m src.pipeline.track_b_pipeline bt20_ens
   python -m src.pipeline.track_b_pipeline bt120_long
   python -m src.pipeline.track_b_pipeline bt120_ens
   ```

2. **재현성 검증**
   - 새로 생성된 산출물과 기존 산출물 비교
   - 데이터 일관성 확인

## 참고사항

- 모든 이동된 파일은 `05_backup` 폴더에 보관되어 있습니다.
- 필요시 `05_backup`에서 파일을 복원할 수 있습니다.
- 정리 스크립트는 `05_backup/scripts/cleanup_and_move.py`에 저장되어 있습니다.

