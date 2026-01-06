# -*- coding: utf-8 -*-
# src/stages/backtest/l7c_benchmark.py
# [개선안 16번] 트랙(A/B/공통) 폴더 재정리: 기존 import 경로 호환 래퍼

from src.tracks.track_b.stages.backtest.l7c_benchmark import run_l7c_benchmark

__all__ = ["run_l7c_benchmark"]


