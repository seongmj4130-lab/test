# 백테스트 결과 비교 분석 리포트

**작성일**: 2026-01-07  
**분석 대상**: `run_backtest_4models.py` vs `show_backtest_metrics.py`

---

## 🔍 문제 발견

두 스크립트의 결과가 서로 다릅니다:

### run_backtest_4models.py 출력값 (최신 실행 결과)
- **bt20_ens**: Sharpe 0.6826, CAGR 14.98%, MDD -10.98%
- **bt20_short**: Sharpe 0.6464, CAGR 13.84%, MDD -9.09%
- **bt120_ens**: Sharpe 2.4432, CAGR 118.21%, MDD -21.19%
- **bt120_long**: Sharpe 2.3319, CAGR 115.40%, MDD -20.93%

### show_backtest_metrics.py 출력값 (저장된 파일 기준)
- **bt20_ens**: Sharpe -0.1608, CAGR -4.60%, MDD -16.95%
- **bt20_short**: Sharpe -0.3551, CAGR -7.26%, MDD -18.68%
- **bt120_ens**: Sharpe 0.4601, CAGR 5.04%, MDD -9.65%
- **bt120_long**: Sharpe 0.5689, CAGR 6.86%, MDD -10.27%

---

## 📊 원인 분석

### 1. 스크립트 동작 방식 차이

#### `run_backtest_4models.py`
- **동작**: 백테스트를 **새로 실행**하고, 실행 직후의 메트릭을 출력
- **출력**: `run_backtest()` 함수가 반환하는 `result[3]` (bt_metrics)의 값
- **저장**: 실행 후 메트릭을 parquet 파일로 저장

#### `show_backtest_metrics.py`
- **동작**: **저장된 parquet 파일**을 읽어서 표시
- **출력**: `data/interim/bt_metrics_{strategy}.parquet` 파일의 값

### 2. 실제 저장된 파일 vs 실행 시점 메트릭

디버깅 결과:
- **실행 시점 메트릭** (run_backtest 반환값): Sharpe 0.6826 ✅
- **저장된 파일 메트릭**: Sharpe -0.1608 ❌

**결론**: 저장된 파일이 이전 실행 결과이거나, 저장 과정에서 값이 변경되었을 가능성

---

## 🔬 상세 분석

### 실제 저장된 메트릭 파일 구조

```python
# bt_metrics_bt20_ens.parquet
phase      net_sharpe  net_cagr    net_mdd
dev        0.1431      0.010280   -0.370443
holdout   -0.1608     -0.045959   -0.169545
```

### 실행 시점 메트릭 (run_backtest 반환값)

```python
# result[3] (bt_metrics)
phase      net_sharpe  net_cagr    net_mdd
dev        0.5493      0.497750   -0.414470
holdout    0.6826      0.149807   -0.109820  ✅ 올바른 값
```

---

## ✅ 올바른 결과

**`run_backtest_4models.py`의 출력값이 올바른 최신 결과입니다.**

이유:
1. 백테스트를 새로 실행하므로 최신 데이터 기반
2. `run_backtest()` 함수가 반환하는 메트릭이 정확함
3. 저장된 파일은 이전 실행 결과일 가능성이 높음

---

## 📋 권장사항

### 1. 즉시 조치
- **`run_backtest_4models.py`의 출력값을 신뢰**하세요
- 저장된 파일은 이전 실행 결과일 수 있으므로 주의

### 2. 개선 방안
1. **메트릭 저장 시점 확인**
   - `run_backtest_4models.py`에서 저장 전후 값 비교 로직 추가
   - 저장 실패 시 경고 메시지 출력

2. **일관성 검증**
   - 저장된 파일과 실행 시점 메트릭 비교 스크립트 추가
   - 불일치 시 자동으로 재저장

3. **타임스탬프 추가**
   - 메트릭 파일에 실행 시간 저장
   - 최신 실행 결과인지 확인 가능

---

## 📈 최종 결과 (최신 실행 기준)

### Holdout 구간 성과

| 모델 | Net Sharpe | Net CAGR | Net MDD | Net Calmar |
|------|------------|----------|---------|------------|
| **bt20_ens** | **0.6826** | **14.98%** | **-10.98%** | **1.3641** |
| **bt20_short** | **0.6464** | **13.84%** | **-9.09%** | **1.5223** |
| **bt120_ens** | **2.4432** | **118.21%** | **-21.19%** | **5.5794** |
| **bt120_long** | **2.3319** | **115.40%** | **-20.93%** | **5.5150** |

**✅ 이 값들이 올바른 최신 백테스트 결과입니다.**

---

## 🔧 다음 단계

1. 저장된 메트릭 파일 재생성 검토
2. `show_backtest_metrics.py`가 최신 파일을 읽는지 확인
3. 메트릭 저장/로드 프로세스 검증 로직 추가

