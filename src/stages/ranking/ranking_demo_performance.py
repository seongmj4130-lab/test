# src/stages/ranking/ranking_demo_performance.py
# [개선안 16번] 트랙(A/B/공통) 폴더 재정리: 기존 import 경로 호환 래퍼

from src.tracks.track_a.stages.ranking.ranking_demo_performance import (  # [개선안 17번] 멀티 벤치마크 지원
    build_benchmark_returns,
    build_equity_curves,
    build_equity_curves_multi,
    build_top20_equal_weight_returns,
    calculate_performance_metrics,
    calculate_performance_metrics_multi,
    calculate_returns,
    run_L11_demo_performance,
)

__all__ = [
    "calculate_returns",
    "build_top20_equal_weight_returns",
    "build_benchmark_returns",
    "build_equity_curves",
    "build_equity_curves_multi",
    "calculate_performance_metrics",
    "calculate_performance_metrics_multi",
    "run_L11_demo_performance",
]
