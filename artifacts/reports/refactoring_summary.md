# 리팩토링 요약 리포트

**리팩토링 일시**: 2026-01-08
**리팩토링 대상**: 03_code
**리팩토링 목적**: config.yaml 우선 적용 구조로 변경 및 기본값 통일

## 리팩토링 완료 사항

### 1. track_b_pipeline.py 수정 완료

#### 변경 내용
- **기존**: `.get()` 메서드의 두 번째 인자로 하드코딩된 기본값 사용
- **변경**: `BacktestConfig` 클래스의 기본값을 참조하여 사용
- **우선순위**: config.yaml → BacktestConfig 기본값

#### 주요 변경사항
```python
# [리팩토링 전]
bt_cfg = BacktestConfig(
    top_k=int(l7_cfg.get("top_k", 12)),  # 하드코딩된 기본값
    buffer_k=int(l7_cfg.get("buffer_k", 15)),  # 하드코딩된 기본값
    smart_buffer_enabled=bool(l7_cfg.get("smart_buffer_enabled", True)),  # 하드코딩된 기본값
    ...
)

# [리팩토링 후]
default_cfg = BacktestConfig()
bt_cfg = BacktestConfig(
    top_k=int(l7_cfg.get("top_k", default_cfg.top_k)),  # BacktestConfig 기본값 사용
    buffer_k=int(l7_cfg.get("buffer_k", default_cfg.buffer_k)),  # BacktestConfig 기본값 사용
    smart_buffer_enabled=bool(l7_cfg.get("smart_buffer_enabled", default_cfg.smart_buffer_enabled)),  # BacktestConfig 기본값 사용
    ...
)
```

#### 통일된 기본값

| 설정 | 이전 기본값 | BacktestConfig 기본값 | 변경 후 |
|------|------------|---------------------|---------|
| `top_k` | 12 | **20** | ✅ 통일 |
| `buffer_k` | 15 | **0** | ✅ 통일 |
| `smart_buffer_enabled` | True | **False** | ✅ 통일 |
| `volatility_adjustment_enabled` | True | **False** | ✅ 통일 |
| `risk_scaling_enabled` | True | **False** | ✅ 통일 |
| `risk_scaling_bear_multiplier` | 0.8 | **0.7** | ✅ 통일 |
| `risk_scaling_neutral_multiplier` | 1.0 | **0.9** | ✅ 통일 |

### 2. 적용 우선순위

**변경 전**:
```
config.yaml → track_b_pipeline.py 하드코딩 기본값 → BacktestConfig 기본값 (사용 안 됨)
```

**변경 후**:
```
config.yaml → BacktestConfig 기본값
```

### 3. 실행 결과

#### 랭킹 산정부터 백테스트까지 재실행 완료

1. **Track A (랭킹 산정)**: ✅ 완료
   - 단기 랭킹: 442,549행
   - 장기 랭킹: 442,549행

2. **Track B (백테스트)**: ✅ 완료
   - bt20_short: ✅ 완료
   - bt20_ens: ✅ 완료
   - bt120_long: ✅ 완료
   - bt120_ens: ✅ 완료

### 4. 백테스트 결과 (리팩토링 후)

#### Holdout 구간

| 전략 | Net Sharpe | Net CAGR | Net MDD | Net Calmar |
|------|------------|----------|---------|------------|
| **bt20_short** | -0.3551 | -7.26% | -18.68% | -0.3886 |
| **bt20_ens** | -0.1608 | -4.60% | -16.95% | -0.2711 |
| **bt120_long** | 0.5689 | 6.86% | -10.27% | 0.6679 |
| **bt120_ens** | 0.4601 | 5.04% | -9.65% | 0.5219 |

#### Dev 구간

| 전략 | Net Sharpe | Net CAGR | Net MDD | Net Calmar |
|------|------------|----------|---------|------------|
| **bt20_short** | -0.0124 | -1.04% | -29.75% | -0.0348 |
| **bt20_ens** | 0.1431 | 1.03% | -37.04% | 0.0277 |
| **bt120_long** | 0.3136 | 4.78% | -21.97% | 0.2177 |
| **bt120_ens** | 0.3546 | 5.79% | -23.03% | 0.2513 |

### 5. 리팩토링 효과

#### 장점
1. **일관성 확보**: 모든 기본값이 `BacktestConfig` 클래스에서 단일 소스로 관리됨
2. **유지보수성 향상**: 기본값 변경 시 한 곳만 수정하면 됨
3. **명확한 우선순위**: config.yaml → BacktestConfig 기본값 순서가 명확
4. **오류 방지**: 기본값 불일치로 인한 예상치 못한 동작 방지

#### 주의사항
- **config.yaml 명시 필요**: 기본값이 변경되었으므로, 모든 설정값을 config.yaml에 명시적으로 정의하는 것이 권장됩니다.
- **기존 설정 영향**: config.yaml에 설정값이 없는 경우, 이전과 다른 기본값이 적용될 수 있습니다.

### 6. 권장사항

#### config.yaml에 모든 설정값 명시
리팩토링 후 기본값이 변경되었으므로, 다음 설정값들을 config.yaml에 명시적으로 정의하는 것이 권장됩니다:

```yaml
l7_bt20_short:
  top_k: 12  # config.yaml에 명시 (기본값 20에서 변경)
  buffer_k: 15  # config.yaml에 명시 (기본값 0에서 변경)
  smart_buffer_enabled: true  # config.yaml에 명시 (기본값 false에서 변경)
  volatility_adjustment_enabled: true  # config.yaml에 명시 (기본값 false에서 변경)
  risk_scaling_enabled: true  # config.yaml에 명시 (기본값 false에서 변경)
  risk_scaling_bear_multiplier: 0.8  # config.yaml에 명시 (기본값 0.7에서 변경)
  risk_scaling_neutral_multiplier: 1.0  # config.yaml에 명시 (기본값 0.9에서 변경)
  ...
```

### 7. 결론

✅ **리팩토링 완료**: `track_b_pipeline.py`의 `.get()` 기본값을 `BacktestConfig` 클래스 기본값으로 통일
✅ **실행 완료**: 랭킹 산정부터 백테스트까지 모든 전략 재실행 완료
✅ **결과 확인**: 모든 백테스트가 성공적으로 실행되어 결과 생성

### 8. 결과 파일 위치

- 백테스트 결과: `artifacts/reports/backtest_results_refactored.csv`
- Parquet 파일: `data/interim/bt_metrics_{strategy}.parquet`
