# src/stages/backtest/l7_backtest.py
# [개선안 16번] 트랙(A/B/공통) 폴더 재정리: 기존 import 경로 호환 래퍼

from src.tracks.track_b.stages.backtest.l7_backtest import BacktestConfig, run_backtest

__all__ = ["BacktestConfig", "run_backtest"]
