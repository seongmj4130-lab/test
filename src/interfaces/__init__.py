"""UI 연동 인터페이스 모듈"""

from .ui_service import (
    RankingItem,
    get_combined_ranking,
    get_long_term_ranking,
    get_short_term_ranking,
)

__all__ = [
    "get_short_term_ranking",
    "get_long_term_ranking",
    "get_combined_ranking",
    "RankingItem",
]
