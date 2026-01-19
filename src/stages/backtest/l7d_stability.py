# -*- coding: utf-8 -*-
# src/stages/backtest/l7d_stability.py
# [개선안 16번] 트랙(A/B/공통) 폴더 재정리: 기존 import 경로 호환 래퍼

from src.tracks.track_b.stages.backtest.l7d_stability import (
    run_l7d_stability_from_artifacts,
)

__all__ = ["run_l7d_stability_from_artifacts"]
