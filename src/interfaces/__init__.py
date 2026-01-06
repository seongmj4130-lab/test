# -*- coding: utf-8 -*-
"""UI 연동 인터페이스 모듈"""

from .ui_service import (
    get_short_term_ranking,
    get_long_term_ranking,
    get_combined_ranking,
    RankingItem,
)

__all__ = [
    "get_short_term_ranking",
    "get_long_term_ranking",
    "get_combined_ranking",
    "RankingItem",
]

