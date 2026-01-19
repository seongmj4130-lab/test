# -*- coding: utf-8 -*-
"""
Grid Search 결과 기반 Dev/Holdout 구간 성과 비교 및 과적합 분석

Grid Search는 Dev 구간에서만 평가되었으므로:
1. Grid Search 결과 분석 (Dev 구간)
2. Holdout 구간 평가는 별도 실행 필요 (최적 가중치 적용 후)
3. 과적합 위험 분석
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def analyze_grid_search_results():
    """Grid Search 결과 분석"""
    results_dir = Path("artifacts/reports")
    
    # 단기 랭킹 결과
    short_file = results_dir / "track_a_group_weights_grid_search_20260108_135117.csv"
    # 장기 랭킹 결과
    long_file = results_dir / "track_a_group_weights_grid_search_20260108_145118.csv"
    
    results = {}
    
    for horizon, file_path in [("short", short_file), ("long", long_file)]:
        if not file_path.exists():
            continue
        
        df = pd.read_csv(file_path)
        best = df.loc[df['objective_score'].idxmax()]
        
        # 전체 조합 통계
        results[horizon] = {
            "n_combinations": len(df),
            "best_combination_id": int(best['combination_id']),
            "best_objective_score": float(best['objective_score']),
            "best_hit_ratio": float(best['hit_ratio']),
            "best_ic_mean": float(best['ic_mean']),
            "best_icir": float(best['icir']),
            "best_rank_ic_mean": float(best.get('rank_ic_mean', 0)),
            "best_rank_icir": float(best.get('rank_icir', 0)),
            "weights": {
                "technical": float(best['technical']),
                "value": float(best['value']),
                "profitability": float(best['profitability']),
                "news": float(best.get('news', 0)),
            },
            # 전체 조합 통계
            "all_combinations": {
                "objective_score_mean": float(df['objective_score'].mean()),
                "objective_score_std": float(df['objective_score'].std()),
                "hit_ratio_mean": float(df['hit_ratio'].mean()),
                "ic_mean_mean": float(df['ic_mean'].mean()),
                "ic_mean_std": float(df['ic_mean'].std()),
                "icir_mean": float(df['icir'].mean()),
            },
        }
    
    return results

def analyze_overfitting_risk(grid_results: dict) -> dict:
    """과적합 위험 분석 (Grid Search 결과 기반)"""
    
    analysis = {
        "grid_search_analysis": {},
        "overfitting_risk_indicators": {},
        "recommendations": [],
    }
    
    for horizon, result in grid_results.items():
        # Grid Search 결과 분석
        best_score = result["best_objective_score"]
        mean_score = result["all_combinations"]["objective_score_mean"]
        std_score = result["all_combinations"]["objective_score_std"]
        
        # 최적 조합이 평균 대비 얼마나 우수한지
        score_improvement = (best_score - mean_score) / std_score if std_score > 0 else 0
        
        # IC 변동성
        ic_std = result["all_combinations"]["ic_mean_std"]
        ic_mean = result["all_combinations"]["ic_mean_mean"]
        ic_cv = abs(ic_std / ic_mean) if ic_mean != 0 else 0  # 변동계수
        
        analysis["grid_search_analysis"][horizon] = {
            "best_score": best_score,
            "mean_score": mean_score,
            "std_score": std_score,
            "score_improvement_sigma": score_improvement,
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "ic_cv": ic_cv,
        }
        
        # 과적합 위험 지표
        risk_indicators = {}
        
        # 1. 최적 조합이 평균 대비 너무 우수한 경우 (3 sigma 이상)
        if score_improvement > 3.0:
            risk_indicators["high_score_improvement"] = {
                "level": "high",
                "value": score_improvement,
                "description": f"최적 조합이 평균 대비 {score_improvement:.2f}σ 우수 (과적합 가능성)",
            }
        elif score_improvement > 2.0:
            risk_indicators["high_score_improvement"] = {
                "level": "medium",
                "value": score_improvement,
                "description": f"최적 조합이 평균 대비 {score_improvement:.2f}σ 우수",
            }
        else:
            risk_indicators["high_score_improvement"] = {
                "level": "low",
                "value": score_improvement,
                "description": f"최적 조합이 평균 대비 {score_improvement:.2f}σ 우수 (정상 범위)",
            }
        
        # 2. IC 변동성이 높은 경우
        if ic_cv > 2.0:
            risk_indicators["high_ic_volatility"] = {
                "level": "high",
                "value": ic_cv,
                "description": f"IC 변동계수 {ic_cv:.2f} (높은 변동성, 불안정)",
            }
        elif ic_cv > 1.0:
            risk_indicators["high_ic_volatility"] = {
                "level": "medium",
                "value": ic_cv,
                "description": f"IC 변동계수 {ic_cv:.2f} (중간 변동성)",
            }
        else:
            risk_indicators["high_ic_volatility"] = {
                "level": "low",
                "value": ic_cv,
                "description": f"IC 변동계수 {ic_cv:.2f} (낮은 변동성, 안정적)",
            }
        
        # 3. IC 평균이 낮거나 음수인 경우
        if ic_mean < 0:
            risk_indicators["negative_ic"] = {
                "level": "high",
                "value": ic_mean,
                "description": f"IC 평균이 음수 ({ic_mean:.4f}) - 예측력 부족",
            }
        elif ic_mean < 0.01:
            risk_indicators["negative_ic"] = {
                "level": "medium",
                "value": ic_mean,
                "description": f"IC 평균이 매우 낮음 ({ic_mean:.4f})",
            }
        else:
            risk_indicators["negative_ic"] = {
                "level": "low",
                "value": ic_mean,
                "description": f"IC 평균 {ic_mean:.4f} (양수, 예측력 있음)",
            }
        
        analysis["overfitting_risk_indicators"][horizon] = risk_indicators
        
        # 종합 위험 평가
        high_risk_count = sum(1 for v in risk_indicators.values() if v["level"] == "high")
        medium_risk_count = sum(1 for v in risk_indicators.values() if v["level"] == "medium")
        
        if high_risk_count >= 2:
            overall_risk = "high"
            recommendations = [
                "⚠️ 높은 과적합 위험 감지",
                "  - Holdout 구간에서 성과 재평가 필수",
                "  - 정규화 강화 (Ridge alpha 증가) 검토",
                "  - 피처 수 감소 또는 피처 선택 강화",
                "  - 더 많은 조합 평가 (Grid Search 확장)",
            ]
        elif medium_risk_count >= 2 or high_risk_count >= 1:
            overall_risk = "medium"
            recommendations = [
                "⚠️ 중간 과적합 위험 감지",
                "  - Holdout 구간에서 성과 재평가 권장",
                "  - 정규화 조정 검토",
                "  - 추가 검증 데이터로 재평가",
            ]
        else:
            overall_risk = "low"
            recommendations = [
                "✅ 낮은 과적합 위험 (Grid Search 결과 기준)",
                "  - Holdout 구간 평가로 최종 확인 필요",
                "  - 현재 모델 설정 유지 가능",
            ]
        
        analysis["overfitting_risk_indicators"][horizon]["overall_risk"] = overall_risk
        analysis["overfitting_risk_indicators"][horizon]["recommendations"] = recommendations
    
    return analysis

def main():
    """메인 함수"""
    print("=" * 80)
    print("Grid Search 결과 기반 Dev/Holdout 구간 성과 비교 및 과적합 분석")
    print("=" * 80)
    
    # Grid Search 결과 분석
    print("\nGrid Search 결과 분석 중...")
    grid_results = analyze_grid_search_results()
    
    if not grid_results:
        print("❌ 분석할 Grid Search 결과가 없습니다.")
        return
    
    # 결과 출력
    for horizon, result in grid_results.items():
        print(f"\n[{horizon.upper()} 랭킹]")
        print("-" * 80)
        print(f"총 조합 수: {result['n_combinations']}개")
        print(f"최적 조합 ID: {result['best_combination_id']}")
        print(f"\n[최적 조합 성과] (Dev 구간)")
        print(f"  Objective Score: {result['best_objective_score']:.4f}")
        print(f"  Hit Ratio: {result['best_hit_ratio']*100:.2f}%")
        print(f"  IC Mean: {result['best_ic_mean']:.4f}")
        print(f"  ICIR: {result['best_icir']:.4f}")
        print(f"  Rank IC Mean: {result['best_rank_ic_mean']:.4f}")
        print(f"  Rank ICIR: {result['best_rank_icir']:.4f}")
        print(f"\n[최적 가중치]")
        for group, weight in result['weights'].items():
            print(f"  {group}: {weight:.2f}")
        print(f"\n[전체 조합 통계]")
        stats = result['all_combinations']
        print(f"  Objective Score: {stats['objective_score_mean']:.4f} ± {stats['objective_score_std']:.4f}")
        print(f"  Hit Ratio: {stats['hit_ratio_mean']*100:.2f}%")
        print(f"  IC Mean: {stats['ic_mean_mean']:.4f} ± {stats['ic_mean_std']:.4f}")
        print(f"  ICIR: {stats['icir_mean']:.4f}")
    
    # 과적합 분석
    print("\n" + "=" * 80)
    print("과적합 위험 분석")
    print("=" * 80)
    
    overfitting_analysis = analyze_overfitting_risk(grid_results)
    
    for horizon, analysis in overfitting_analysis["grid_search_analysis"].items():
        print(f"\n[{horizon.upper()} 랭킹]")
        print(f"  최적 Score: {analysis['best_score']:.4f}")
        print(f"  평균 Score: {analysis['mean_score']:.4f} ± {analysis['std_score']:.4f}")
        print(f"  개선도: {analysis['score_improvement_sigma']:.2f}σ")
        print(f"  IC 평균: {analysis['ic_mean']:.4f} ± {analysis['ic_std']:.4f}")
        print(f"  IC 변동계수: {analysis['ic_cv']:.2f}")
        
        risk_indicators = overfitting_analysis["overfitting_risk_indicators"][horizon]
        overall_risk = risk_indicators["overall_risk"]
        
        print(f"\n  [과적합 위험 지표]")
        for indicator_name, indicator in risk_indicators.items():
            if indicator_name == "overall_risk" or indicator_name == "recommendations":
                continue
            print(f"    {indicator_name}: {indicator['level']} - {indicator['description']}")
        
        print(f"\n  [종합 위험도]: {overall_risk.upper()}")
        print(f"  [권장사항]")
        for rec in risk_indicators["recommendations"]:
            print(f"    {rec}")
    
    # 결과 저장
    output_dir = Path("artifacts/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"grid_search_overfitting_analysis_{timestamp}.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("# Grid Search 결과 기반 과적합 분석\n\n")
        f.write(f"**생성일**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("=" * 80 + "\n\n")
        f.write("## ⚠️ 중요 참고사항\n\n")
        f.write("Grid Search는 **Dev 구간에서만 평가**되었습니다.\n\n")
        f.write("Holdout 구간 성과 비교를 위해서는:\n")
        f.write("1. 최적 가중치를 적용한 L8 랭킹 실행\n")
        f.write("2. Holdout 구간에서 성과 평가\n")
        f.write("3. Dev/Holdout 구간 성과 비교\n\n")
        f.write("=" * 80 + "\n\n")
        
        for horizon, result in grid_results.items():
            f.write(f"## {horizon.upper()} 랭킹\n\n")
            
            f.write("### 최적 조합 성과 (Dev 구간)\n\n")
            f.write(f"- Objective Score: {result['best_objective_score']:.4f}\n")
            f.write(f"- Hit Ratio: {result['best_hit_ratio']*100:.2f}%\n")
            f.write(f"- IC Mean: {result['best_ic_mean']:.4f}\n")
            f.write(f"- ICIR: {result['best_icir']:.4f}\n")
            f.write(f"- Rank IC Mean: {result['best_rank_ic_mean']:.4f}\n")
            f.write(f"- Rank ICIR: {result['best_rank_icir']:.4f}\n\n")
            
            f.write("### 최적 가중치\n\n")
            for group, weight in result['weights'].items():
                f.write(f"- {group}: {weight:.2f}\n")
            f.write("\n")
            
            # 과적합 분석
            analysis = overfitting_analysis["grid_search_analysis"][horizon]
            risk_indicators = overfitting_analysis["overfitting_risk_indicators"][horizon]
            
            f.write("### 과적합 위험 분석\n\n")
            f.write(f"**종합 위험도**: {risk_indicators['overall_risk'].upper()}\n\n")
            
            f.write("#### 위험 지표\n\n")
            for indicator_name, indicator in risk_indicators.items():
                if indicator_name in ["overall_risk", "recommendations"]:
                    continue
                f.write(f"- **{indicator_name}**: {indicator['level']} - {indicator['description']}\n")
            
            f.write("\n#### 권장사항\n\n")
            for rec in risk_indicators["recommendations"]:
                f.write(f"{rec}\n")
            
            f.write("\n" + "-" * 80 + "\n\n")
    
    print(f"\n✅ 결과 저장: {report_file}")
    
    return grid_results, overfitting_analysis

if __name__ == "__main__":
    main()
