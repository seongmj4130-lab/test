# -*- coding: utf-8 -*-
"""
국면별 전략 모듈
- 시장 국면(bull/bear)에 따라 다른 팩터 가중치 적용
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional
import yaml
import pandas as pd

logger = logging.getLogger(__name__)


def get_market_regime(
    kospi200_returns: pd.Series,
    window: int = 63,
    threshold_pct: float = 0.0,
) -> str:
    """
    시장 국면 판정
    
    Args:
        kospi200_returns: KOSPI200 일일 수익률 시계열
        window: lookback 기간 (일수, 기본값: 63일)
        threshold_pct: bull/bear 분류 임계값 (기본값: 0.0)
    
    Returns:
        "bull" | "bear" | "neutral"
    """
    if len(kospi200_returns) < window:
        logger.warning(f"[Regime] 데이터 부족 ({len(kospi200_returns)} < {window}), neutral 반환")
        return "neutral"
    
    recent_returns = kospi200_returns[-window:]
    recent_return = recent_returns.sum()
    volatility = recent_returns.std()
    
    # 국면 분류
    if recent_return > threshold_pct:
        if recent_return > 0.05 and volatility < 0.015:
            return "bull_strong"
        elif recent_return > 0:
            return "bull"
        else:
            return "neutral"
    else:
        if recent_return < -0.05:
            return "bear_strong"
        else:
            return "bear"


def load_regime_weights(
    config_path: Optional[str] = None,
    base_dir: Optional[Path] = None,
) -> Dict[str, Dict[str, float]]:
    """
    국면별 가중치 로드
    
    Args:
        config_path: 가중치 설정 파일 경로 (기본값: configs/feature_weights_regime_aware.yaml)
        base_dir: 프로젝트 루트 디렉토리
    
    Returns:
        {
            "bull": {"feature1": weight1, ...},
            "bear": {"feature1": weight1, ...},
        }
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parent.parent.parent.parent
    
    if config_path is None:
        config_path = base_dir / "configs" / "feature_weights_regime_aware.yaml"
    else:
        config_path = base_dir / config_path
    
    if not config_path.exists():
        logger.warning(f"[Regime] 가중치 파일 없음: {config_path}, 기본 가중치 사용")
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        regime_weights = {}
        if "regime_weights" in data:
            for regime, regime_data in data["regime_weights"].items():
                if isinstance(regime_data, dict) and "weights" in regime_data:
                    regime_weights[regime] = regime_data["weights"]
                elif isinstance(regime_data, dict):
                    # weights 키가 없으면 직접 가중치 딕셔너리로 간주
                    regime_weights[regime] = regime_data
        
        logger.info(f"[Regime] 국면별 가중치 로드 완료: {list(regime_weights.keys())}")
        return regime_weights
    
    except Exception as e:
        logger.error(f"[Regime] 가중치 로드 실패: {e}, 기본 가중치 사용")
        return {}


def get_regime_weights_for_date(
    date: pd.Timestamp,
    market_regime_df: pd.DataFrame,
    regime_weights_config: Dict[str, Dict[str, float]],
    default_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    특정 날짜의 국면에 맞는 가중치 반환
    
    Args:
        date: 날짜
        market_regime_df: 시장 국면 DataFrame (date, regime 컬럼)
        regime_weights_config: 국면별 가중치 설정
        default_weights: 기본 가중치 (국면별 가중치가 없을 때 사용)
    
    Returns:
        {feature: weight} 딕셔너리
    """
    # 날짜에 해당하는 국면 조회
    regime_row = market_regime_df[market_regime_df["date"] == date]
    
    if len(regime_row) == 0:
        logger.debug(f"[Regime] {date.strftime('%Y-%m-%d')}: 국면 데이터 없음, 기본 가중치 사용")
        return default_weights or {}
    
    regime = regime_row.iloc[0]["regime"]
    
    # 국면별 가중치 조회
    if regime in regime_weights_config:
        weights = regime_weights_config[regime]
        logger.debug(f"[Regime] {date.strftime('%Y-%m-%d')}: {regime} 가중치 사용 ({len(weights)}개 팩터)")
        return weights
    
    # bull_strong -> bull, bear_strong -> bear 매핑
    if regime == "bull_strong" and "bull" in regime_weights_config:
        weights = regime_weights_config["bull"]
        logger.debug(f"[Regime] {date.strftime('%Y-%m-%d')}: bull_strong -> bull 가중치 사용")
        return weights
    
    if regime == "bear_strong" and "bear" in regime_weights_config:
        weights = regime_weights_config["bear"]
        logger.debug(f"[Regime] {date.strftime('%Y-%m-%d')}: bear_strong -> bear 가중치 사용")
        return weights
    
    # 기본 가중치 사용
    logger.debug(f"[Regime] {date.strftime('%Y-%m-%d')}: {regime} 가중치 없음, 기본 가중치 사용")
    return default_weights or {}


# 국면별 기본 가중치 (참고용)
REGIME_WEIGHTS_DEFAULT = {
    "bull": {
        "total_liabilities": 0.145,
        "equity": 0.144,
        "net_income": 0.114,
        "volume": 0.095,
        "volume_ratio": 0.090,
        "debt_ratio": 0.089,
        "debt_ratio_sector_z": 0.084,
        "turnover": 0.082,
        "price_momentum_20d": 0.079,
        "price_momentum": 0.079,
    },
    "bear": {
        "total_liabilities": 0.207,
        "equity": 0.196,
        "debt_ratio": 0.161,
        "volume": 0.159,
        "net_income": 0.145,
        "debt_ratio_sector_z": 0.132,
    },
    "neutral": {
        # 기본 균형 가중치 (전체 평균)
        "total_liabilities": 0.188,
        "equity": 0.183,
        "net_income": 0.141,
        "volume": 0.132,
        "debt_ratio": 0.128,
        "volume_ratio": 0.116,
        "debt_ratio_sector_z": 0.113,
    },
}

