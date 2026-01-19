"""현재 상태 확인"""
from pathlib import Path

import pandas as pd


def check_status():
    """현재 상태 확인"""
    print("=" * 80)
    print("현재 상태 요약")
    print("=" * 80)

    # 단기 랭킹 결과
    short_file = (
        "artifacts/reports/track_a_group_weights_grid_search_20260108_135117.csv"
    )
    if Path(short_file).exists():
        short_df = pd.read_csv(short_file)
        short_best = short_df.loc[short_df["objective_score"].idxmax()]
        print("\n[단기 랭킹]")
        print(f"  파일: {short_file}")
        print(f"  조합 수: {len(short_df)}개")
        print(f"  최적 Objective Score: {short_best['objective_score']:.4f}")
        print(f"  최적 Hit Ratio: {short_best['hit_ratio']*100:.2f}%")
        print(f"  최적 IC Mean: {short_best['ic_mean']:.4f}")
        print(f"  최적 ICIR: {short_best['icir']:.4f}")
        print(
            f"  가중치: technical={short_best['technical']:.2f}, value={short_best['value']:.2f}, profitability={short_best['profitability']:.2f}, news={short_best['news']:.2f}"
        )

    # 장기 랭킹 결과
    long_file = (
        "artifacts/reports/track_a_group_weights_grid_search_20260108_145118.csv"
    )
    if Path(long_file).exists():
        long_df = pd.read_csv(long_file)
        long_best = long_df.loc[long_df["objective_score"].idxmax()]
        print("\n[장기 랭킹]")
        print(f"  파일: {long_file}")
        print(f"  조합 수: {len(long_df)}개")
        print(f"  최적 Objective Score: {long_best['objective_score']:.4f}")
        print(f"  최적 Hit Ratio: {long_best['hit_ratio']*100:.2f}%")
        print(f"  최적 IC Mean: {long_best['ic_mean']:.4f}")
        print(f"  최적 ICIR: {long_best['icir']:.4f}")
        if "news" in long_best.index:
            print(
                f"  가중치: technical={long_best['technical']:.2f}, value={long_best['value']:.2f}, profitability={long_best['profitability']:.2f}, news={long_best['news']:.2f}"
            )
        else:
            print(
                f"  가중치: technical={long_best['technical']:.2f}, value={long_best['value']:.2f}, profitability={long_best['profitability']:.2f}"
            )

    # 비교
    if Path(short_file).exists() and Path(long_file).exists():
        print("\n" + "=" * 80)
        print("단기 vs 장기 랭킹 비교")
        print("=" * 80)
        print(
            f"단기: 조합 {len(short_df)}개, 최적 Score {short_best['objective_score']:.4f}, IC {short_best['ic_mean']:.4f}"
        )
        print(
            f"장기: 조합 {len(long_df)}개, 최적 Score {long_best['objective_score']:.4f}, IC {long_best['ic_mean']:.4f}"
        )
        print(
            f"\n차이: Score {long_best['objective_score'] - short_best['objective_score']:.4f}, IC {long_best['ic_mean'] - short_best['ic_mean']:.4f}"
        )


if __name__ == "__main__":
    check_status()
