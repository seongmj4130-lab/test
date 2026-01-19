# BacktestConfig 기본값 통일 완료 리포트

**작성일**: 2026-01-08  
**작업 내용**: BacktestConfig 기본값을 config.yaml과 통일, 우선순위 문서화, 백테스트 재실행

---

## 1. 완료된 작업

### 1.1. BacktestConfig 기본값 변경

**변경된 기본값 (5개):**

| 설정 | 변경 전 | 변경 후 | 변경 사유 |
|------|---------|---------|----------|
| `buffer_k` | 0 | **15** | 3개 전략에서 공통 사용 (bt20_short, bt120_long, bt120_ens) |
| `rebalance_interval` | 1 | **20** | 모든 전략에서 공통 사용 |
| `smart_buffer_enabled` | False | **True** | 모든 전략에서 공통 사용 |
| `volatility_adjustment_enabled` | False | **True** | 모든 전략에서 공통 사용 |
| `risk_scaling_enabled` | False | **True** | 모든 전략에서 공통 사용 |

### 1.2. 우선순위 문서화

**문서 위치**: `docs/CONFIG_PRIORITY_DOCUMENTATION.md`

**주요 내용:**
- 설정값 적용 우선순위 (config.yaml → BacktestConfig 기본값)
- BacktestConfig 기본값 목록 및 변경 이력
- 전략별 config.yaml 설정값 비교
- 리팩토링 효과 및 권장사항

### 1.3. 백테스트 재실행

**실행 완료:**
- ✅ Track A (랭킹 산정): 단기/장기 랭킹 442,549행
- ✅ bt20_short: Dev 87개 리밸런싱, Holdout 23개 리밸런싱
- ✅ bt20_ens: Dev 87개 리밸런싱, Holdout 23개 리밸런싱
- ✅ bt120_long: Dev 87개 리밸런싱, Holdout 23개 리밸런싱
- ✅ bt120_ens: Dev 87개 리밸런싱, Holdout 23개 리밸런싱

---

## 2. 설정값 적용 우선순위

```
1순위: config.yaml의 전략별 설정값
2순위: BacktestConfig 클래스 기본값
```

**예시:**
```python
# track_b_pipeline.py
default_cfg = BacktestConfig()
bt_cfg = BacktestConfig(
    top_k=int(l7_cfg.get("top_k", default_cfg.top_k)),  # config.yaml 우선
    buffer_k=int(l7_cfg.get("buffer_k", default_cfg.buffer_k)),  # 없으면 기본값 15
    ...
)
```

---

## 3. 통일 효과

### 3.1. 통일 전 (리팩토링 전)

- **BacktestConfig 기본값**: 보수적인 기본값
  - `buffer_k=0`, `rebalance_interval=1`
  - `smart_buffer_enabled=False`, `volatility_adjustment_enabled=False`
- **config.yaml 실제 사용값**: 더 적극적인 설정
  - `buffer_k=15`, `rebalance_interval=20`
  - `smart_buffer_enabled=True`, `volatility_adjustment_enabled=True`
- **문제**: config.yaml에 값이 누락되면 예상과 다른 동작 발생 가능

### 3.2. 통일 후 (리팩토링 후)

- **BacktestConfig 기본값**: config.yaml의 실제 사용값과 통일
- **효과**: 
  - ✅ config.yaml에 값이 누락되어도 실제 사용값과 유사한 기본값 적용
  - ✅ 예상치 못한 동작 방지
  - ✅ 일관성 확보

---

## 4. 백테스트 결과 (통일 후)

결과는 `artifacts/reports/backtest_results_refactored.csv`에 저장되었습니다.

### Holdout 구간

| 전략 | Net Sharpe | Net CAGR | Net MDD | Net Calmar |
|------|------------|----------|---------|------------|
| **bt20_short** | -0.3551 | -7.26% | -18.68% | -0.3886 |
| **bt20_ens** | -0.1608 | -4.60% | -16.95% | -0.2711 |
| **bt120_long** | 0.5689 | 6.86% | -10.27% | 0.6679 |
| **bt120_ens** | 0.4601 | 5.04% | -9.65% | 0.5219 |

### Dev 구간

| 전략 | Net Sharpe | Net CAGR | Net MDD | Net Calmar |
|------|------------|----------|---------|------------|
| **bt20_short** | -0.0124 | -1.04% | -29.75% | -0.0348 |
| **bt20_ens** | 0.1431 | 1.03% | -37.04% | 0.0277 |
| **bt120_long** | 0.3136 | 4.78% | -21.97% | 0.2177 |
| **bt120_ens** | 0.3546 | 5.79% | -23.03% | 0.2513 |

---

## 5. 주요 변경 사항

### 5.1. 코드 변경

**파일**: `src/tracks/track_b/stages/backtest/l7_backtest.py`

```python
# 변경 전
buffer_k: int = 0
rebalance_interval: int = 1
smart_buffer_enabled: bool = False
volatility_adjustment_enabled: bool = False
risk_scaling_enabled: bool = False

# 변경 후
buffer_k: int = 15  # [리팩토링] config.yaml과 통일
rebalance_interval: int = 20  # [리팩토링] config.yaml과 통일
smart_buffer_enabled: bool = True  # [리팩토링] config.yaml과 통일
volatility_adjustment_enabled: bool = True  # [리팩토링] config.yaml과 통일
risk_scaling_enabled: bool = True  # [리팩토링] config.yaml과 통일
```

### 5.2. 문서 추가

**새 문서**: `docs/CONFIG_PRIORITY_DOCUMENTATION.md`

---

## 6. 참고 파일

- **BacktestConfig 클래스**: `src/tracks/track_b/stages/backtest/l7_backtest.py`
- **파이프라인**: `src/pipeline/track_b_pipeline.py`
- **우선순위 문서**: `docs/CONFIG_PRIORITY_DOCUMENTATION.md`
- **비교 스크립트**: `scripts/analyze_common_configs.py`
- **설정값 표시 스크립트**: `scripts/show_all_backtest_configs.py`
- **백테스트 결과**: `artifacts/reports/backtest_results_refactored.csv`

---

## 7. 결론

✅ **통일 완료**: BacktestConfig 기본값이 config.yaml의 실제 사용값과 통일되었습니다.  
✅ **문서화 완료**: 설정값 적용 우선순위가 문서화되었습니다.  
✅ **재실행 완료**: 모든 전략에 대해 백테스트가 성공적으로 재실행되었습니다.  
✅ **일관성 확보**: config.yaml에 값이 누락되어도 실제 사용값과 유사한 기본값이 적용됩니다.

