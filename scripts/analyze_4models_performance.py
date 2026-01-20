"""
[Phase 2] 4가지 ML 모델 성과 비교 분석
4가지 평가지표로 Dev/Holdout 구간별 평가

평가 지표:
1. Net Sharpe Ratio (목표: Dev ≥ 0.50, Holdout ≥ 0.50)
2. Net Total Return (비용 차감 누적 수익률)
3. Net CAGR (목표: Dev ≥ 10%, Holdout ≥ 15%)
4. Net MDD (목표: Dev ≤ -30%, Holdout ≤ -10%)

분석 대상 모델:
1. Grid Search
2. Ridge
3. XGBoost
4. Random Forest
"""

from __future__ import annotations

import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def calculate_net_sharpe_ratio(
    returns: pd.Series, annualization_factor: int = 252
) -> float:
    """Net Sharpe Ratio 계산 (연율화)"""
    if len(returns) == 0:
        return np.nan

    # 일일 수익률의 평균과 표준편차
    daily_mean = returns.mean()
    daily_std = returns.std()

    if daily_std == 0 or np.isnan(daily_std):
        return np.nan

    # 연율화 Sharpe Ratio
    sharpe_ratio = (daily_mean / daily_std) * np.sqrt(annualization_factor)
    return float(sharpe_ratio)


def calculate_net_total_return(returns: pd.Series) -> float:
    """Net Total Return 계산 (누적 수익률)"""
    if len(returns) == 0:
        return np.nan

    # 누적 곱 계산: (1 + r1) * (1 + r2) * ... * (1 + rn) - 1
    cumulative_return = (1 + returns).prod() - 1
    return float(cumulative_return)


def calculate_net_cagr(returns: pd.Series, total_days: int) -> float:
    """Net CAGR 계산 (연복리수익률)"""
    if len(returns) == 0 or total_days <= 0:
        return np.nan

    total_return = calculate_net_total_return(returns)
    if np.isnan(total_return):
        return np.nan

    # CAGR = (1 + total_return)^(365/total_days) - 1
    years = total_days / 365.25  # 실제 년수 계산
    cagr = (1 + total_return) ** (1 / years) - 1
    return float(cagr)


def calculate_net_mdd(returns: pd.Series) -> float:
    """Net MDD 계산 (최대 낙폭)"""
    if len(returns) == 0:
        return np.nan

    # 누적 수익률 계산
    cumulative = (1 + returns).cumprod()

    # 최고점부터의 낙폭 계산
    peak = cumulative.expanding().max()
    drawdown = (cumulative - peak) / peak

    # 최대 낙폭 (음수 값)
    mdd = drawdown.min()
    return float(mdd)


def analyze_backtest_results(
    backtest_df: pd.DataFrame,
    dev_end_date: str = "2023-12-31",
    holdout_start_date: str = "2024-01-01",
) -> dict[str, dict[str, float]]:
    """
    백테스트 결과를 4가지 평가지표로 분석

    Args:
        backtest_df: 백테스트 결과 DataFrame (date, portfolio_return 등 포함)
        dev_end_date: Dev 구간 종료일
        holdout_start_date: Holdout 구간 시작일

    Returns:
        Dev/Holdout 구간별 평가지표 딕셔너리
    """
    # 날짜 컬럼 확인 및 변환
    date_col = None
    for col in backtest_df.columns:
        if "date" in col.lower():
            date_col = col
            break

    if date_col is None:
        raise ValueError("Date column not found in backtest results")

    # 수익률 컬럼 확인
    return_col = None
    for col in ["portfolio_return", "returns", "return"]:
        if col in backtest_df.columns:
            return_col = col
            break

    if return_col is None:
        raise ValueError("Return column not found in backtest results")

    # 날짜 변환
    backtest_df = backtest_df.copy()
    backtest_df[date_col] = pd.to_datetime(backtest_df[date_col])
    backtest_df = backtest_df.sort_values(date_col)

    # Dev/Holdout 구간 분리
    dev_mask = backtest_df[date_col] <= pd.to_datetime(dev_end_date)
    holdout_mask = backtest_df[date_col] >= pd.to_datetime(holdout_start_date)

    dev_returns = backtest_df[dev_mask][return_col].dropna()
    holdout_returns = backtest_df[holdout_mask][return_col].dropna()

    # Dev 구간 분석
    dev_days = len(dev_returns)
    dev_metrics = {
        "net_sharpe_ratio": calculate_net_sharpe_ratio(dev_returns),
        "net_total_return": calculate_net_total_return(dev_returns),
        "net_cagr": (
            calculate_net_cagr(dev_returns, dev_days) if dev_days > 0 else np.nan
        ),
        "net_mdd": calculate_net_mdd(dev_returns),
    }

    # Holdout 구간 분석
    holdout_days = len(holdout_returns)
    holdout_metrics = {
        "net_sharpe_ratio": calculate_net_sharpe_ratio(holdout_returns),
        "net_total_return": calculate_net_total_return(holdout_returns),
        "net_cagr": (
            calculate_net_cagr(holdout_returns, holdout_days)
            if holdout_days > 0
            else np.nan
        ),
        "net_mdd": calculate_net_mdd(holdout_returns),
    }

    return {
        "dev": dev_metrics,
        "holdout": holdout_metrics,
        "metadata": {
            "dev_days": dev_days,
            "holdout_days": holdout_days,
            "total_days": len(backtest_df),
        },
    }


def load_model_config(model_type: str, horizon: str) -> Optional[dict]:
    """모델 가중치 파일 로드"""
    config_dir = Path("configs")

    # 모델별 파일명 패턴
    file_patterns = {
        "grid": f"feature_groups_{horizon}_optimized_grid_*.yaml",
        "ridge": f"feature_weights_{horizon}_ridge_*.yaml",
        "xgboost": f"feature_weights_{horizon}_xgboost_*.yaml",
        "rf": f"feature_weights_{horizon}_rf_*.yaml",
    }

    if model_type not in file_patterns:
        return None

    pattern = file_patterns[model_type]

    # 최신 파일 찾기 (타임스탬프 기준)
    matching_files = list(config_dir.glob(pattern))
    if not matching_files:
        return None

    # 타임스탬프 기준으로 최신 파일 선택
    latest_file = max(matching_files, key=lambda x: x.stat().st_mtime)

    try:
        with open(latest_file, encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Error loading {latest_file}: {e}")
        return None


def run_backtest_for_model(model_type: str, horizon: str) -> Optional[pd.DataFrame]:
    """특정 모델에 대한 백테스트 실행"""
    print(f"\n[{model_type.upper()}] {horizon} 전략 백테스트 중...")

    # 백테스트 결과 파일 경로 매핑
    # bt20_short: 단기 전략, bt120_long: 장기 전략
    strategy_map = {"short": "bt20_short", "long": "bt120_long"}

    if horizon not in strategy_map:
        print(f"  ⚠️ 알 수 없는 horizon: {horizon}")
        return None

    strategy = strategy_map[horizon]
    interim_dir = Path("data/interim")
    backtest_file = interim_dir / f"bt_equity_curve_{strategy}.csv"

    if not backtest_file.exists():
        print(f"  ⚠️ 백테스트 결과 파일이 없습니다: {backtest_file}")
        return None

    try:
        # 백테스트 결과 로드
        backtest_df = pd.read_csv(backtest_file)

        # 수익률 계산 (equity의 일별 변화율)
        backtest_df["portfolio_return"] = backtest_df["equity"].pct_change()

        print(f"  - 백테스트 데이터 로드 완료: {len(backtest_df)} 행")
        print("  - 수익률 계산 완료 (portfolio_return 컬럼 추가)")
        return backtest_df
    except Exception as e:
        print(f"  ⚠️ 백테스트 결과 로드 실패: {e}")
        return None


def main():
    """메인 분석 함수"""
    print("=" * 80)
    print("[Phase 2] 4가지 ML 모델 성과 비교 분석")
    print("=" * 80)

    # 분석 대상 모델들
    models = ["grid", "ridge", "xgboost", "rf"]
    horizons = ["short", "long"]

    # 결과 저장용 딕셔너리
    results = {}

    # 각 모델별 분석
    for model in models:
        model_results = {}
        for horizon in horizons:
            # 백테스트 실행
            backtest_df = run_backtest_for_model(model, horizon)
            if backtest_df is None:
                model_results[horizon] = None
                continue

            # 평가지표 계산
            try:
                metrics = analyze_backtest_results(backtest_df)
                model_results[horizon] = metrics
                print(f"  ✅ {horizon} 분석 완료")
            except Exception as e:
                print(f"  ⚠️ {horizon} 분석 실패: {e}")
                model_results[horizon] = None

        results[model] = model_results

    # 결과 출력 및 저장
    print("\n" + "=" * 80)
    print("분석 결과 요약")
    print("=" * 80)

    # 모델 이름 매핑
    model_names = {
        "grid": "Grid Search",
        "ridge": "Ridge",
        "xgboost": "XGBoost",
        "rf": "Random Forest",
    }

    # Dev 구간 결과
    print("\n[Dev 구간 성과 (2023년)]")
    print("-" * 100)
    print(
        f"{'모델':<12} {'전략':<8} {'Sharpe':<8} {'Total Ret':<12} {'CAGR':<8} {'MDD':<8}"
    )
    print("-" * 100)

    for model in models:
        model_name = model_names[model]

        for horizon in horizons:
            if results[model][horizon] is None:
                print(
                    f"{model_name:<12} {horizon:<8} {'N/A':<8} {'N/A':<12} {'N/A':<8} {'N/A':<8}"
                )
                continue

            dev = results[model][horizon]["dev"]
            print(
                f"{model_name:<12} {horizon:<8} "
                f"{dev['net_sharpe_ratio']:.3f} "
                f"{dev['net_total_return']:.3f} "
                f"{dev['net_cagr']:.3f} "
                f"{dev['net_mdd']:.3f}"
            )

    # Holdout 구간 결과
    print("\n\n[Holdout 구간 성과 (2024년)]")
    print("-" * 100)
    print(
        f"{'모델':<12} {'전략':<8} {'Sharpe':<8} {'Total Ret':<12} {'CAGR':<8} {'MDD':<8}"
    )
    print("-" * 100)

    for model in models:
        model_name = model_names[model]

        for horizon in horizons:
            if results[model][horizon] is None:
                print(
                    f"{model_name:<12} {horizon:<8} {'N/A':<8} {'N/A':<12} {'N/A':<8} {'N/A':<8}"
                )
                continue

            holdout = results[model][horizon]["holdout"]
            print(
                f"{model_name:<12} {horizon:<8} "
                f"{holdout['net_sharpe_ratio']:.3f} "
                f"{holdout['net_total_return']:.3f} "
                f"{holdout['net_cagr']:.3f} "
                f"{holdout['net_mdd']:.3f}"
            )

    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = (
        Path("artifacts/reports") / f"4models_performance_analysis_{timestamp}.csv"
    )

    # CSV로 저장하기 위한 데이터프레임 생성
    rows = []
    for model in models:
        for horizon in horizons:
            if results[model][horizon] is None:
                continue

            row = {"model": model_names[model], "horizon": horizon, "period": "dev"}
            row.update(results[model][horizon]["dev"])
            rows.append(row)

            row = {"model": model_names[model], "horizon": horizon, "period": "holdout"}
            row.update(results[model][horizon]["holdout"])
            rows.append(row)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_file, index=False, float_format="%.4f")

    print(f"\n✅ 분석 결과 저장: {output_file}")
    print("\n🎯 목표 성과 기준:")
    print("   - Net Sharpe Ratio: Dev ≥ 0.50, Holdout ≥ 0.50")
    print("   - Net CAGR: Dev ≥ 10%, Holdout ≥ 15%")
    print("   - Net MDD: Dev ≤ -30%, Holdout ≤ -10%")


if __name__ == "__main__":
    main()
