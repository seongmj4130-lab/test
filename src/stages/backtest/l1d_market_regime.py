# src/stages/backtest/l1d_market_regime.py
# [개선안 16번] 트랙(A/B/공통) 폴더 재정리: 기존 import 경로 호환 래퍼

from src.tracks.shared.stages.regime.l1d_market_regime import build_market_regime

__all__ = ["build_market_regime"]
