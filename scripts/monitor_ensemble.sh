#!/bin/bash
# 앙상블 최적화 실시간 모니터링

echo "🔍 앙상블 최적화 모니터링 시작..."
echo "======================================"

# 실시간 로그 출력
tail -f logs/ensemble_*.log 2>/dev/null &
TAIL_PID=$!

# 진행 상황 모니터링
watch -n 10 -t -c '
clear
echo "╔════════════════════════════════════════════════════════════╗"
echo "║        앙상블 최적화 진행 상황 (10초마다 갱신)           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 최신 중간 결과 파일:"
ls -lht artifacts/reports/ensemble_*intermediate*.csv 2>/dev/null | head -3
echo ""
echo "📊 최신 최종 결과 파일:"
ls -lht artifacts/reports/ensemble_optimization_*.csv 2>/dev/null | head -2
echo ""
echo "⏱️  실행 중인 프로세스:"
ps aux | grep optimize_ensemble_weights | grep -v grep | grep -v monitor
echo ""
echo "💾 디스크 사용량 (artifacts/reports):"
du -sh artifacts/reports/ 2>/dev/null
echo ""
echo "📈 예상 진행률 (파일 크기 기준):"
LATEST=$(ls -t artifacts/reports/ensemble_*intermediate*.csv 2>/dev/null | head -1)
if [ -f "$LATEST" ]; then
    LINES=$(wc -l < "$LATEST")
    echo "  현재 평가 완료: $LINES개 조합"
fi
echo ""
echo "======================================"
echo "Ctrl+C로 모니터링 종료"
'

# 정리
kill $TAIL_PID 2>/dev/null
