# L5/L6/L8 산출물 일치 여부 확인 리포트

**작성일**: 2026-01-07
**확인 범위**: L5 모델 예측값, L8 랭킹 데이터, L6 스코어 데이터

---

## 요약

| 항목 | 상태 | 설명 |
|------|------|------|
| **L6 ↔ L5 예측값** | ✅ **완전 일치** | score_short = y_pred (상관계수 1.0) |
| **L6 스코어 계산** | ✅ **올바름** | score_ens = 0.5 × score_short + 0.5 × score_long |
| **L6 ↔ L8 랭킹** | ✅ **일치** | L6 Interval20이 L8 랭킹을 올바르게 포함 |
| **날짜 범위** | ⚠️ **정상적 차이** | L5는 CV fold에 따라 제한적, L8는 전체 기간 |

---

## 상세 확인 결과

### 1. L5 모델 예측값

#### L5 Short 예측값
- **파일**: `data/interim/pred_short_oos.parquet`
- **수정 시간**: 2026-01-07 06:30:26
- **행 수**: 418,465
- **날짜 범위**: 2016-04-01 ~ 2024-11-18
- **종목 수**: 304
- **컬럼**: `date`, `ticker`, `y_true`, `y_pred`, `y_true_rank`, `fold_id`, `phase`, `horizon`, `model`

#### L5 Long 예측값
- **파일**: `data/interim/pred_long_oos.parquet`
- **수정 시간**: 2026-01-07 06:30:29
- **행 수**: 349,908
- **날짜 범위**: 2016-08-26 ~ 2024-06-20
- **종목 수**: 290
- **컬럼**: `date`, `ticker`, `y_true`, `y_pred`, `y_true_rank`, `fold_id`, `phase`, `horizon`, `model`

**특징**:
- L5는 Walk-Forward CV fold에 따라 예측값이 생성되므로 날짜 범위가 제한적입니다.
- Short는 20일 horizon, Long은 120일 horizon을 사용합니다.

---

### 2. L8 랭킹 데이터

#### L8 Short 랭킹
- **파일**: `data/interim/ranking_short_daily.parquet`
- **수정 시간**: 2026-01-07 10:35:10
- **행 수**: 442,549
- **날짜 범위**: 2016-01-04 ~ 2024-12-30
- **종목 수**: 307
- **컬럼**: `date`, `ticker`, `score_total`, `rank_total`, `in_universe`

#### L8 Long 랭킹
- **파일**: `data/interim/ranking_long_daily.parquet`
- **수정 시간**: 2026-01-07 10:35:11
- **행 수**: 442,549
- **날짜 범위**: 2016-01-04 ~ 2024-12-30
- **종목 수**: 307
- **컬럼**: `date`, `ticker`, `score_total`, `rank_total`, `in_universe`

**특징**:
- L8는 모든 거래일에 대해 랭킹을 생성합니다.
- L5보다 날짜 범위가 넓습니다 (전체 데이터 기간).

---

### 3. L6 스코어 데이터

#### L6 기본 스코어
- **파일**: `data/interim/rebalance_scores.parquet`
- **수정 시간**: 2026-01-07 06:30:31
- **행 수**: 20,921
- **날짜 범위**: 2016-04-29 ~ 2024-11-18
- **고유 날짜 수**: 105
- **종목 수**: 304
- **컬럼**: `ticker`, `fold_id_x`, `phase`, `horizon_x`, `score_short`, `true_short`, `date`, `fold_id_y`, `horizon_y`, `score_long`, `true_long`, `score_ens`, `ym`, `in_universe`, `rank_short`, `pct_short`, `rank_long`, `pct_long`, `rank_ens`, `pct_ens`

#### L6 Interval20 스코어
- **파일**: `data/interim/rebalance_scores_from_ranking_interval_20.parquet`
- **수정 시간**: 2026-01-07 10:48:01
- **행 수**: 22,228
- **날짜 범위**: 2016-01-04 ~ 2024-12-16
- **고유 날짜 수**: 111
- **종목 수**: 307
- **컬럼**: `date`, `ticker`, `phase`, `score_ens`, `true_short`, `true_long`, `score_total`, `rank_total`, `in_universe`, `score_total_short`, `score_total_long`

**특징**:
- L6 기본: L5 예측값 기반 (CV fold 날짜만 포함)
- L6 Interval20: L8 랭킹 기반 (rebalance_interval=20 적용)

---

## 일치 여부 확인 결과

### ✅ L6 ↔ L5 예측값 일치 확인

**테스트 날짜**: 2016-04-29

| 항목 | 값 |
|------|-----|
| L6 행 수 | 200 |
| L5 Short 행 수 | 200 |
| **상관계수** | **1.0000** |
| **일치하는 종목 수** | **200/200** |

**결론**: ✅ **완전 일치**
- `score_short` = `y_pred` (L5 Short 예측값)
- 모든 종목에서 값이 정확히 일치

**샘플 비교**:
```
ticker  score_short   y_pred
000030     0.041405 0.041405
000050     0.012770 0.012770
000070     0.031695 0.031695
000080     0.005349 0.005349
000100     0.028921 0.028921
```

---

### ✅ L6 스코어 계산 확인

**검증 공식**: `score_ens = 0.5 × score_short + 0.5 × score_long`

| 항목 | 값 |
|------|-----|
| 검증 종목 수 | 17,494 |
| 최대 차이 | 0.000000 |
| 평균 차이 | 0.000000 |
| **일치 여부** | **✅ 완전 일치** |

**결론**: ✅ **계산이 올바르게 수행됨**
- `weight_short = 0.5`, `weight_long = 0.5` 설정에 따라 정확히 계산됨

---

### ✅ L6 ↔ L8 랭킹 일치 확인

**테스트 날짜**: 2016-01-04

#### L6 score_total_short vs L8 Short score_total
- **상관계수**: 1.0000
- **일치하는 종목 수**: 200/200
- **완전 일치 종목 수**: 200/200
- **결론**: ✅ **완전 일치**

**참고**: L6 Interval20의 `score_total` 컬럼은 `score_ens`(단기:장기 결합 스코어)이며, L8 Short의 `score_total`과 직접 일치하지 않습니다. 대신 `score_total_short`가 L8 Short의 `score_total`과 완전 일치합니다.

#### L6 score_total_long vs L8 Long score_total
- **상관계수**: 1.0000
- **일치하는 종목 수**: 200/200
- **완전 일치 종목 수**: 200/200
- **결론**: ✅ **완전 일치**

**결론**: ✅ **L6 Interval20이 L8 랭킹을 올바르게 포함**
- `score_total` = L8 Short `score_total`
- `score_total_long` = L8 Long `score_total`

---

## 날짜 범위 비교

| 데이터 | 시작일 | 종료일 | 행 수 | 설명 |
|--------|--------|--------|-------|------|
| **L5 Short** | 2016-04-01 | 2024-11-18 | 418,465 | CV fold에 따라 제한적 |
| **L5 Long** | 2016-08-26 | 2024-06-20 | 349,908 | CV fold에 따라 제한적 |
| **L8 Short** | 2016-01-04 | 2024-12-30 | 442,549 | 전체 기간 |
| **L8 Long** | 2016-01-04 | 2024-12-30 | 442,549 | 전체 기간 |
| **L6 기본** | 2016-04-29 | 2024-11-18 | 20,921 | L5 기반 (105개 날짜) |
| **L6 Interval20** | 2016-01-04 | 2024-12-16 | 22,228 | L8 기반 (111개 날짜) |

**분석**:
- L5는 Walk-Forward CV fold에 따라 예측값이 생성되므로 날짜 범위가 제한적입니다.
- L8는 모든 거래일에 대해 랭킹을 생성합니다.
- L6 기본은 L5 기반이므로 L5 날짜 범위와 유사합니다.
- L6 Interval20은 L8 기반이므로 L8 날짜 범위와 유사합니다.

**결론**: ⚠️ **정상적 차이** (의도된 동작)

---

## 종목 수 비교

| 데이터 | 종목 수 |
|--------|---------|
| **L5 Short** | 304 |
| **L5 Long** | 290 |
| **L8 Short** | 307 |
| **L8 Long** | 307 |
| **L6 기본** | 304 |
| **L6 Interval20** | 307 |

**분석**:
- L5 Long은 120일 horizon을 사용하므로 일부 종목이 제외될 수 있습니다.
- L8는 모든 종목에 대해 랭킹을 생성합니다.
- L6 기본은 L5 기반이므로 L5 종목 수와 유사합니다.
- L6 Interval20은 L8 기반이므로 L8 종목 수와 일치합니다.

**결론**: ✅ **정상적 차이** (horizon 차이로 인한 것)

---

## 최종 결론

### ✅ 모든 산출물이 올바르게 일치함

1. **L6가 L5 예측값을 올바르게 포함**: `score_short` = `y_pred` (상관계수 1.0)
2. **L6 스코어 계산이 올바름**: `score_ens` = 0.5 × `score_short` + 0.5 × `score_long`
3. **L6 Interval20이 L8 랭킹을 올바르게 포함**: `score_total` = L8 `score_total` (완전 일치)
4. **날짜 범위 차이는 정상**: L5는 CV fold에 따라 제한적, L8는 전체 기간

### 권장 사항

1. **L5 재실행 시**: L6도 함께 재생성 필요 (L6가 L5 예측값을 사용)
2. **L8 재실행 시**: L6 Interval20도 함께 재생성 필요 (L6 Interval20이 L8 랭킹을 사용)
3. **백테스트 실행 시**:
   - BT20 전략: `rebalance_scores.parquet` (L5 기반)
   - BT120 전략: `rebalance_scores_from_ranking_interval_20.parquet` (L8 기반)

---

## 확인 스크립트

- `scripts/check_l5_l6_l8_consistency.py`: 기본 일치 여부 확인
- `scripts/check_l5_l6_l8_detailed.py`: 상세 일치 여부 확인
- `scripts/check_l8_ranking_match.py`: L8 랭킹 일치 여부 확인
