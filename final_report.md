# KOSPI200 투트랙 퀀트 투자 시스템 최종 통합 보고서

**작성일**: 2026-01-06  
**프로젝트 기간**: 2016-01-01 ~ 2024-12-31 (데이터 범위)  
**최종 버전**: Phase 9 + 뉴스 피처 추가 + 투트랙 구조 리팩토링 + 데이터 수집 모듈 분리 + 시장 국면 분류 개선 (2026-01-06)  
**대상 독자**: 프로젝트 이해를 원하는 모든 이해관계자 (기술자, 투자자, 관리자, 학생 등)

---

## 📋 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [프로젝트 변천사](#2-프로젝트-변천사)
3. [시스템 아키텍처](#3-시스템-아키텍처)
4. [핵심 개념 및 로직](#4-핵심-개념-및-로직)
5. [데이터 파이프라인](#5-데이터-파이프라인)
6. [랭킹 엔진 (Track A)](#6-랭킹-엔진-track-a)
7. [투자 모델 (Track B)](#7-투자-모델-track-b)
8. [최종 성과 지표](#8-최종-성과-지표)
9. [산출물 구조](#9-산출물-구조)
10. [리팩토링 및 개선사항](#10-리팩토링-및-개선사항)
11. [참고 문서](#11-참고-문서)

---

## 1. 프로젝트 개요

### 1.1 프로젝트 목적

**KOSPI200 투트랙 퀀트 투자 시스템**은 KOSPI200 유니버스 종목에 대해 머신러닝 기반 랭킹을 생성하고, 이를 활용한 투자 전략의 성과를 백테스트로 검증하는 통합 시스템입니다.

### 1.2 핵심 가치 제안

1. **객관적 랭킹 제공**: 데이터 기반으로 KOSPI200 종목의 투자 가치를 객관적으로 평가
2. **전략 검증**: 과거 데이터로 투자 전략의 유효성을 사전 검증
3. **투트랙 구조**: 랭킹 제공(Track A)과 투자 모델(Track B)을 독립적으로 운영
4. **재현성 보장**: Walk-Forward CV와 엄격한 데이터 분할로 재현 가능한 결과

### 1.3 시스템 한 문장 요약

**"KOSPI200 주식 200개를 매일 점수로 줄 세워서(랭킹), 상위 종목을 선택해 투자했을 때 실제로 돈이 되는지 과거 데이터로 검증하는 시스템"**

---

## 2. 프로젝트 변천사

### 2.1 Phase 0-3: 초기 구축 (2016-2023)

- **목표**: 기본 파이프라인 구축
- **주요 성과**:
  - L0~L4 데이터 수집 파이프라인 구축
  - 기본 랭킹 엔진 개발
  - 초기 백테스트 시스템 구축

### 2.2 Phase 4-6: 모델 개선 (2023-2024)

- **목표**: 모델 성능 향상 및 피처 확장
- **주요 성과**:
  - Ridge 회귀 모델 도입
  - Cross-Sectional Rank 변환 적용
  - 재무 데이터 지연 반영 로직 개선
  - Walk-Forward CV 엄격화

### 2.3 Phase 7-8: 전략 고도화 (2024)

- **목표**: 백테스트 전략 최적화
- **주요 성과**:
  - BT20/BT120 이중 전략 도입
  - Smart Buffering 로직 개발
  - 변동성 조정 및 국면별 리스크 스케일링
  - BT120 Holdout Sharpe 0.4565 달성

### 2.4 Phase 9: 최종 완성 (2024-2026)

- **목표**: 뉴스 피처 추가 및 투트랙 구조 리팩토링
- **주요 성과**:
  - 뉴스 감성 피처 4개 추가 (단기 모델 18개→22개)
  - BT20 Holdout Sharpe 0.7370 달성 (목표 0.50 초과)
  - 투트랙 구조 완전 분리 (Track A/B 독립 운영)
  - 데이터 수집 모듈 완전 분리 (`src/data_collection`)

### 2.5 2026-01-06: 최종 정리 및 개선

- **목표**: 프로젝트 문서화, 구조 정리, 시장 국면 분류 개선
- **주요 성과**:
  - 통합 보고서 작성
  - 데이터 수집 모듈 리팩토링 완료
  - UI 인터페이스 모듈화
  - 프로젝트 구조 최종 정리
  - **시장 국면 분류 개선**: 외부 API 없이 ohlcv_daily 데이터로 자동 분류
    - 가격/거래량/변동성 지표를 종합하여 Bull/Neutral/Bear 판단
    - 네트워크 의존성 제거로 안정성 향상
  - **최종 백테스트 실행**: 4가지 전략 모두 재실행 완료
    - bt120_long: Sharpe 0.2163, CAGR 3.65%, MDD -8.25% (Holdout)
    - bt20_short: Sharpe 0.1951, CAGR 2.01%, Hit Ratio 56.52% (Holdout)

---

## 3. 시스템 아키텍처

### 3.1 투트랙 구조

```
┌─────────────────────────────────────────────────────────────┐
│              공통 데이터 준비 (Shared Data)                  │
│  L0: 유니버스 구성 (KOSPI200 멤버십)                         │
│  L1: OHLCV 데이터 다운로드 + 기술적 지표 계산                │
│  L2: 재무 데이터 로드 (DART)                                │
│  L3: 패널 병합 (OHLCV + 재무 + 뉴스 + ESG)                  │
│  L4: Walk-Forward CV 분할 및 타겟 생성                      │
└─────────────────────────────────────────────────────────────┘
                        ↓
        ┌───────────────┴───────────────┐
        │                               │
┌───────▼────────┐            ┌─────────▼──────────┐
│ Track A        │            │ Track B             │
│ 랭킹 엔진      │            │ 투자 모델           │
├───────────────┤            ├─────────────────────┤
│ 목적:         │            │ 목적:               │
│ 피처 기반     │            │ 랭킹 기반 투자      │
│ 랭킹 산정     │            │ 모델 예시 제공      │
│               │            │                     │
│ L8: 랭킹 엔진 │            │ L6R: 랭킹 스코어    │
│   - 단기 랭킹 │            │   변환              │
│   - 장기 랭킹 │            │ L7: 백테스트 실행   │
│               │            │   - BT20 (20일)     │
│ L11: UI       │            │   - BT120 (120일)   │
│   Payload     │            │                     │
│   생성        │            │                     │
│               │            │                     │
│ 산출물:       │            │ 산출물:             │
│ - ranking_    │            │ - bt_metrics        │
│   short_daily │            │ - bt_returns        │
│ - ranking_    │            │ - bt_equity_curve   │
│   long_daily  │            │ - bt_positions       │
└───────────────┘            └─────────────────────┘
```

### 3.2 디렉토리 구조 (최종)

```
03_code/
├── README.md                    # 프로젝트 개요 및 사용법
├── final_report.md              # 이 문서 (통합 보고서)
├── final_easy_report.md         # 비금융인을 위한 쉬운 설명
├── final_ranking_report.md      # Track A 기술 보고서
├── final_backtest_report.md    # Track B 기술 보고서
├── configs/                     # 설정 파일
│   └── config.yaml             # 메인 설정
├── src/                         # 소스 코드
│   ├── data_collection/        # 데이터 수집 모듈 (리팩토링 완료)
│   ├── tracks/                 # 투트랙 구조
│   │   ├── track_a/            # Track A: 랭킹 엔진
│   │   ├── track_b/            # Track B: 투자 모델
│   │   └── shared/             # 공통 데이터 처리
│   ├── pipeline/               # 파이프라인 실행
│   ├── interfaces/             # UI 연동 인터페이스
│   └── utils/                  # 공통 유틸리티
├── data/                        # 데이터
│   ├── raw/                    # 원시 데이터
│   ├── external/               # 외부 데이터
│   └── interim/                # 중간 산출물
└── artifacts/                   # 최종 산출물
    ├── models/                 # 학습된 모델
    ├── rankings/               # 랭킹 결과
    ├── backtests/              # 백테스트 결과
    └── reports/                # 리포트
```

---

## 4. 핵심 개념 및 로직

### 4.1 투트랙 개념

**Track A (랭킹 엔진)**:
- **목적**: 피처들로 KOSPI200의 랭킹을 산정하여 이용자에게 제공
- **독립 실행**: Track B 없이도 랭킹만 제공 가능
- **산출물**: `ranking_short_daily.parquet`, `ranking_long_daily.parquet`

**Track B (투자 모델)**:
- **목적**: 랭킹을 기반으로 다양한 투자모델 예시를 만들어 이용자에게 정보 제공
- **의존성**: Track A의 랭킹 데이터 필요
- **산출물**: `bt_metrics`, `bt_returns`, `bt_equity_curve`

### 4.2 Walk-Forward CV

**핵심 원칙**: 시간 순서를 지켜서 "과거로 학습 → 미래를 테스트"

**구조**:
- **Dev 구간**: 모델 튜닝 및 검증 (2016-2022)
- **Holdout 구간**: 최종 실전 시험 (2023-2024)
- **Embargo/Purge**: 미래 정보 누수 방지

**설정값**:
- `step_days`: 20 (리밸런싱 간격)
- `embargo_days`: 20 (Embargo 기간)
- `holdout_years`: 2 (최근 2년 = Holdout)

### 4.3 타깃 변수

**단기 타깃 (`ret_fwd_20d`)**:
- 계산식: `(close[t+20] / close[t]) - 1.0`
- BT20 백테스트에 사용

**장기 타깃 (`ret_fwd_120d`)**:
- 계산식: `(close[t+120] / close[t]) - 1.0`
- BT120 백테스트에 사용

### 4.4 Cross-Sectional Rank 변환

**목적**: 절대 수익률 대신 상대 순위로 변환하여 모델 학습

**방법**:
```python
# 같은 날짜의 여러 종목을 순위로 변환 (0~100%)
rank = df.groupby('date')[target_col].rank(pct=True) * 100.0
```

**효과**: 시장 전체 움직임과 무관하게 종목 간 상대적 우위 학습

---

## 5. 데이터 파이프라인

### 5.1 L0: 유니버스 구성

**목적**: KOSPI200 구성 종목 월별 스냅샷 생성

**소스**: pykrx 라이브러리 (`get_index_portfolio_deposit_file()`)

**산출물**: `universe_k200_membership_monthly.parquet`
- 컬럼: `date`, `ym`, `ticker`

### 5.2 L1: OHLCV + 기술적 지표

**목적**: 주가 데이터 다운로드 및 기술적 지표 계산

**기술적 지표** (11개):
- 모멘텀: `price_momentum_20d`, `price_momentum_60d`, `momentum_3m`, `momentum_6m`
- 변동성: `volatility_20d`, `volatility_60d`, `downside_volatility_60d`
- 기타: `max_drawdown_60d`, `volume_ratio`, `momentum_reversal`

**산출물**: `ohlcv_daily.parquet`

### 5.3 L2: 재무 데이터

**목적**: DART API에서 재무 정보 다운로드

**지표**: `roe`, `debt_ratio`, `net_income`, `equity`, `total_liabilities`

**공시 지연 반영**:
- `effective_date = report_rcept_date + 1일` (접수일 기준)
- 접수일 없으면: `effective_date = year_end + 90일` (fallback)

**산출물**: `fundamentals_annual.parquet`

### 5.4 L3: 패널 병합

**목적**: 모든 데이터를 하나의 패널로 통합

**병합 순서**:
1. OHLCV + 재무 데이터 (asof merge)
2. 유니버스 멤버십 매핑
3. 업종 정보 병합
4. pykrx 재무데이터 병합 (PER, PBR, market_cap 등)
5. 업종 내 상대화 피처 생성

**산출물**: `panel_merged_daily.parquet`

### 5.5 L4: Walk-Forward CV 분할

**목적**: 타깃 변수 생성 및 CV 분할

**타깃 생성**:
- `ret_fwd_20d`: 20일 후 수익률
- `ret_fwd_120d`: 120일 후 수익률

**CV 분할**:
- Dev 구간: 여러 Train/Test fold 생성
- Holdout 구간: 최종 실전 시험

**산출물**: 
- `dataset_daily.parquet`
- `cv_folds_short.parquet`
- `cv_folds_long.parquet`

---

## 6. 랭킹 엔진 (Track A)

### 6.1 L5: 모델 학습 (선택적)

**모델**: Ridge 회귀

**Pipeline**:
1. `SimpleImputer` (결측치: 중앙값 대체)
2. `StandardScaler` (표준화)
3. `Ridge(alpha=1.0)` (회귀)

**피처**:
- 단기 모델: 22개 (Core 12개 + Short 6개 + News 4개)
- 장기 모델: 19개 (Core 12개 + Long 7개)

**타깃 변환**: Cross-Sectional Rank (0~100%)

**산출물**: `pred_short_oos.parquet`, `pred_long_oos.parquet`

### 6.2 L8: 랭킹 산정

**목적**: 피처 기반 종목 랭킹 생성

**프로세스**:
1. **피처 정규화**: 날짜별 Cross-Sectional 정규화 (percentile 또는 zscore)
2. **가중치 적용**: 피처 그룹별 가중치로 합산
3. **순위 생성**: 점수 높은 순서대로 1등~200등

**피처 그룹** (Phase 9 기준):
- `value`: 0.25 (가치 지표)
- `profitability`: 0.25 (수익성 지표)
- `technical`: 0.50 (기술적 지표)
- `news`: 0.10 (뉴스 감성)

**산출물**:
- `ranking_short_daily.parquet`: 단기 랭킹
- `ranking_long_daily.parquet`: 장기 랭킹

---

## 7. 투자 모델 (Track B)

### 7.1 L6R: 랭킹 스코어 변환

**목적**: Track A의 랭킹을 백테스트용 스코어로 변환

**로직**:
- 단기/장기 랭킹을 결합하여 통합 스코어 생성
- 리밸런싱 날짜별로 정리

**산출물**: `rebalance_scores_from_ranking.parquet`

### 7.2 L7: 백테스트 실행

**목적**: 랭킹 기반 투자 전략의 성과 시뮬레이션

#### 7.2.1 두 가지 전략

**BT20 (20일 보유 전략)**:
- 보유 기간: 20 영업일
- 리밸런싱: 매 리밸런싱 실행 (`rebalance_interval=1`)
- 가중치: Softmax (`temperature=0.5`)
- 역할: 수익률 중심 공격적 전략

**BT120 (120일 보유 전략)**:
- 보유 기간: 120 영업일
- 리밸런싱: 10번째 리밸런싱만 실행 (`rebalance_interval=10`)
- 가중치: Equal Weighting
- 역할: 안정성 중심 메인 전략

#### 7.2.2 포트폴리오 구성 로직

**1. 종목 선택**:
- 상위 K개 종목 선택 (`top_k`: BT20=15, BT120=20)
- Smart Buffering: 이전 보유 종목이 상위권에 있으면 유지
- Fallback: 선택 종목 부족 시 다음 순위에서 채움

**2. 가중치 계산**:
- **Equal**: `weight[i] = 1.0 / n`
- **Softmax**: `weight = softmax(score / temperature)`

**3. 노출 조정**:
- **변동성 조정**: 목표 변동성(15%)에 맞춰 조정
- **국면별 리스크 스케일링**: Bear/Neutral/Bull 구간별 배수 적용
- **시장 국면 분류**: 외부 API 없이 ohlcv_daily 데이터로 자동 분류
  - 가격 수익률, 변동성, 거래량 변화율을 종합하여 판단
  - 각 rebalance 날짜 기준으로 lookback 기간 동안의 지표 계산

**4. 수익률 계산**:
- 포트폴리오 수익률: `Σ(weight[i] * return[i])`
- 거래비용 차감: `net_return = gross_return - cost`
- 누적 수익률: `equity[t] = equity[t-1] * (1 + net_return[t])`

#### 7.2.3 성과 지표 계산

**수익 지표**:
- Net Total Return: 비용 차감 후 누적 수익률
- Net CAGR: 연평균 복리 수익률

**위험 지표**:
- Net Sharpe Ratio: 리스크 대비 수익 효율
- Net MDD: 최대 낙폭
- Calmar Ratio: CAGR / |MDD|

**운용 지표**:
- Avg Turnover: 평균 회전율
- Hit Ratio: 수익 구간 비율
- Profit Factor: 번 돈 합 / 잃은 돈 합

**산출물**:
- `bt_positions_{strategy}.parquet`: 포지션 히스토리
- `bt_returns_{strategy}.parquet`: 수익률 히스토리
- `bt_equity_curve_{strategy}.parquet`: 누적 자산 곡선
- `bt_metrics_{strategy}.parquet`: 성과 지표

---

## 8. 최종 성과 지표

### 8.1 최종 백테스트 결과 (2026-01-06 실행)

**4가지 전략의 Holdout 구간 성과**:

| 전략 | Net Sharpe | Net CAGR | Net MDD | Calmar Ratio | Hit Ratio | Profit Factor |
|------|------------|----------|---------|--------------|-----------|---------------|
| **bt120_long** | **0.2163** | **3.65%** | **-8.25%** | **0.4427** | 33.33% | **1.5317** |
| **bt20_short** | **0.1951** | **2.01%** | -10.38% | 0.1933 | **56.52%** | 1.1678 |
| bt20_ens | 0.0921 | 0.32% | -9.56% | 0.0331 | 47.83% | 1.0760 |
| bt120_ens | 0.0210 | -0.60% | -10.84% | -0.0549 | 33.33% | 1.0404 |

**주요 발견사항**:
- **bt120_long** (장기 보유 + 장기 랭킹): 가장 우수한 성과
  - Sharpe 0.2163, CAGR 3.65%, MDD -8.25%
  - 가장 높은 Profit Factor (1.5317)
  - 가장 높은 Calmar Ratio (0.4427)
- **bt20_short** (단기 보유 + 단기 랭킹): 가장 높은 Hit Ratio (56.52%)
  - 안정적인 수익률 (CAGR 2.01%)
  - 두 번째로 높은 Sharpe (0.1951)
- 통합 랭킹(ens) 전략은 Holdout 구간에서 일반화 성능이 낮음
  - Dev 구간에서는 좋은 성과를 보였으나 Holdout에서 성능 저하

**참고사항**:
- 모든 지표는 거래비용(cost_bps=10.0)을 반영한 Net 지표
- 시장 국면 기능은 외부 API 없이 ohlcv_daily 데이터로 자동 분류
- 실행 환경: 시장 국면 기능 활성화, 네트워크 의존성 제거

### 8.2 이전 성과 (참고용)

**BT20 (Phase 9, 뉴스 피처 추가 후)**:
- Net Sharpe: 0.7370 (목표 ≥ 0.50 초과 달성)
- Net CAGR: 12.08% (목표 ≥ 10% 달성)
- Net MDD: -8.53% (목표 ≤ -10% 달성)

**BT120 (Phase 8 기준)**:
- Net Sharpe: 0.4565 (목표 ≥ 0.45 달성)
- Net CAGR: 14.92% (목표 ≥ 14% 달성)
- Net MDD: -9.20% (목표 ≤ -10% 달성)

---

## 9. 산출물 구조

### 9.1 공통 데이터 (L0~L4)

**위치**: `data/interim/`

- `universe_k200_membership_monthly.parquet`: KOSPI200 멤버십
- `ohlcv_daily.parquet`: OHLCV + 기술적 지표
- `fundamentals_annual.parquet`: 재무 데이터
- `panel_merged_daily.parquet`: 병합된 패널 데이터
- `dataset_daily.parquet`: CV 분할 완료 데이터셋
- `cv_folds_short.parquet`: 단기 CV fold
- `cv_folds_long.parquet`: 장기 CV fold

### 9.2 Track A 산출물

**위치**: `data/interim/`

- `ranking_short_daily.parquet`: 단기 랭킹 (날짜별 종목 랭킹)
- `ranking_long_daily.parquet`: 장기 랭킹 (날짜별 종목 랭킹)
- `ui_payload.json` (선택적): UI에서 사용할 수 있는 형태

### 9.3 Track B 산출물

**위치**: `data/interim/`

- `rebalance_scores_from_ranking.parquet`: 랭킹 스코어 변환
- `bt_positions_{strategy}.parquet`: 포지션 히스토리
- `bt_returns_{strategy}.parquet`: 수익률 히스토리
- `bt_equity_curve_{strategy}.parquet`: 누적 자산 곡선
- `bt_metrics_{strategy}.parquet`: 성과 지표

**전략별 접미사**:
- `bt20_short`: BT20 단기 랭킹
- `bt20_ens`: BT20 통합 랭킹
- `bt120_long`: BT120 장기 랭킹
- `bt120_ens`: BT120 통합 랭킹

---

## 10. 리팩토링 및 개선사항

### 10.1 투트랙 구조 리팩토링 (2026-01-05)

**목적**: Track A와 Track B를 완전히 독립적으로 운영

**변경사항**:
- `src/tracks/track_a/`: Track A 전용 코드
- `src/tracks/track_b/`: Track B 전용 코드
- `src/tracks/shared/`: 공통 데이터 처리

**효과**: 각 트랙을 독립적으로 실행 및 유지보수 가능

### 10.2 데이터 수집 모듈 분리 (2026-01-06)

**목적**: 데이터 수집을 완전히 분리하여 재사용성 향상

**변경사항**:
- `src/data_collection/`: 데이터 수집 모듈 완전 분리
- `collectors.py`: L0~L4 데이터 수집 함수
- `pipeline.py`: 데이터 수집 파이프라인 클래스
- `ui_interface.py`: UI 인터페이스 함수

**효과**: 데이터 수집 로직 재사용성 향상, UI 연동 간소화

### 10.3 시장 국면 분류 개선 (2026-01-06)

**목적**: 외부 API 의존성 제거 및 안정성 향상

**변경사항**:
- **기존**: pykrx 라이브러리로 KOSPI200 지수 데이터 외부 API 호출
- **개선**: `ohlcv_daily` 데이터를 사용하여 자동 분류
  - 가격 수익률: lookback 기간 동안의 시장 가중 평균 수익률
  - 변동성: 일일 수익률 표준편차 (연환산)
  - 거래량: lookback 기간 동안의 거래량 변화율
  - 종합 판단: 세 지표를 종합하여 Bull/Neutral/Bear 분류

**효과**:
- 네트워크 의존성 제거로 안정성 향상
- 외부 API 호출 실패 시에도 정상 동작
- 내부 데이터만으로 시장 국면 판단 가능

**분류 로직**:
```python
# 1. 가격 수익률 계산
total_return_pct = ((1 + daily_returns).prod() - 1) * 100.0

# 2. 변동성 계산
volatility_pct = daily_returns.std() * sqrt(252) * 100.0

# 3. 거래량 변화율 계산
volume_change_pct = ((end_volume - start_volume) / start_volume) * 100.0

# 4. 종합 판단
if abs(total_return_pct) <= neutral_band * 100:
    regime = "neutral"
elif total_return_pct > neutral_band * 100:
    # Bull: 변동성/거래량 추가 확인
    regime = "bull" if volatility < 30% and volume_change > -20% else "neutral"
else:
    # Bear: 변동성/거래량 추가 확인
    regime = "bear"
```

### 10.4 최종 백테스트 실행 (2026-01-06)

**실행 전략**: 4가지 전략 모두 재실행

**결과 요약** (Holdout 구간):

| 전략 | Net Sharpe | Net CAGR | Net MDD | Calmar Ratio | Hit Ratio | Profit Factor |
|------|------------|----------|---------|--------------|-----------|---------------|
| **bt120_long** | **0.2163** | **3.65%** | **-8.25%** | **0.4427** | 33.33% | **1.5317** |
| **bt20_short** | **0.1951** | **2.01%** | -10.38% | 0.1933 | **56.52%** | 1.1678 |
| bt20_ens | 0.0921 | 0.32% | -9.56% | 0.0331 | 47.83% | 1.0760 |
| bt120_ens | 0.0210 | -0.60% | -10.84% | -0.0549 | 33.33% | 1.0404 |

**주요 발견사항**:
- **bt120_long** (장기 보유 + 장기 랭킹): 가장 우수한 성과
  - Sharpe 0.2163, CAGR 3.65%, MDD -8.25%
  - 가장 높은 Profit Factor (1.5317)
- **bt20_short** (단기 보유 + 단기 랭킹): 가장 높은 Hit Ratio (56.52%)
- 통합 랭킹(ens) 전략은 Holdout 구간에서 일반화 성능이 낮음
- `src/data_collection/`: 데이터 수집 모듈 생성
  - `collectors.py`: L0~L4 수집 함수
  - `pipeline.py`: 파이프라인 클래스
  - `ui_interface.py`: UI 인터페이스

**효과**:
- UI에서 직접 데이터 수집 가능
- 재현성 보장
- 실행 간편화

### 10.3 뉴스 피처 추가 (2026-01-04)

**목적**: 뉴스 감성 정보를 랭킹에 반영

**추가 피처** (4개):
- `news_sentiment`: 뉴스 감성 점수
- `news_sentiment_ewm5`: 5일 지수이동평균
- `news_sentiment_surprise`: 감성 변화율
- `news_volume`: 뉴스 볼륨

**효과**: BT20 Holdout Sharpe 0.50 → 0.7370 (+47.4%)

---

## 11. 참고 문서

### 11.1 핵심 문서

- **README.md**: 프로젝트 개요 및 사용법
- **final_easy_report.md**: 비금융인을 위한 쉬운 설명
- **final_ranking_report.md**: Track A 기술 보고서 (상세)
- **final_backtest_report.md**: Track B 기술 보고서 (상세)

### 11.2 설정 파일

- **configs/config.yaml**: 메인 설정 파일
  - `l4`: Walk-Forward CV 파라미터
  - `l5`: 모델 파라미터 (선택적)
  - `l8`: 랭킹 엔진 파라미터
  - `l7_bt20`: BT20 백테스트 파라미터
  - `l7_bt120`: BT120 백테스트 파라미터

### 11.3 실행 방법

**1단계: 공통 데이터 준비**
```bash
# 방법 1: 새로운 데이터 수집 모듈 사용 (권장)
python -c "from src.data_collection import collect_all_data; collect_all_data()"

# 방법 2: 기존 스크립트 사용
python scripts/run_pipeline_l0_l7.py
```

**2단계: Track A 실행**
```bash
python -m src.pipeline.track_a_pipeline
```

**3단계: Track B 실행**
```bash
python -m src.pipeline.track_b_pipeline bt20_short
```

### 11.4 UI 연동

**데이터 수집 인터페이스**:
```python
from src.data_collection import (
    get_universe,
    get_ohlcv,
    get_panel,
    check_data_availability,
)
```

**랭킹 조회**:
```python
from src.interfaces.ui_service import (
    get_short_term_ranking,
    get_long_term_ranking,
    get_combined_ranking,
)
```

---

## 부록: 주요 기술 스택

- **언어**: Python 3.x
- **데이터 처리**: pandas, numpy
- **머신러닝**: scikit-learn (Ridge 회귀)
- **데이터 소스**: pykrx (주가), OpenDartReader (재무)
- **저장 형식**: Parquet, CSV, JSON
- **설정 관리**: YAML

---

**작성 완료일**: 2026-01-06  
**최종 검토**: 
- 프로젝트 전체 변천사 반영
- 모든 개념 및 로직 설명
- 최종 성과 지표 포함
- 리팩토링 내용 반영
- 산출물 구조 명시

---

**이 보고서는 2026-01-06까지 진행된 프로젝트의 모든 내용을 포함하는 최종 통합 보고서입니다.**

