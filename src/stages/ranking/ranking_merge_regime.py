# src/stages/ranking/ranking_merge_regime.py
# [개선안 16번] 트랙(A/B/공통) 폴더 재정리: 기존 import 경로 호환 래퍼

from src.tracks.track_a.stages.ranking.ranking_merge_regime import (
    merge_regime_to_ranking,
)

__all__ = ["merge_regime_to_ranking"]
