"""
L1B: pykrx 재무데이터 다운로드 (래퍼)
"""

from src.tracks.shared.stages.data.l1b_pykrx_fundamentals import (
    download_pykrx_fundamentals_daily,
)

__all__ = ["download_pykrx_fundamentals_daily"]
