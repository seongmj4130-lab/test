# -*- coding: utf-8 -*-
# src/stages/data/news_sentiment_features.py
# [개선안 16번] 트랙(A/B/공통) 폴더 재정리: 기존 import 경로 호환 래퍼

from src.tracks.shared.stages.data.news_sentiment_features import (
    NewsSentimentConfig,
    attach_news_sentiment_features,
)

__all__ = ["NewsSentimentConfig", "attach_news_sentiment_features"]


