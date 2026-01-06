# -*- coding: utf-8 -*-
# src/stages/data/l4_walkforward_split.py
# [개선안 16번] 트랙(A/B/공통) 폴더 재정리: 기존 import 경로 호환 래퍼

from src.tracks.shared.stages.data.l4_walkforward_split import build_inner_cv_folds, build_targets_and_folds

__all__ = ["build_inner_cv_folds", "build_targets_and_folds"]


