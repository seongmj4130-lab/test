# 파이프라인을 로그 파일과 함께 실행하는 스크립트
# Usage: .\scripts\run_pipeline_with_log.ps1

param(
    [Parameter(Mandatory=$false)]
    [string]$FromStage = "L5",
    
    [Parameter(Mandatory=$false)]
    [string]$ToStage = "L7",
    
    [Parameter(Mandatory=$false)]
    [string]$Config = "configs/config.yaml"
)

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runTag = "phase4_alpha_tuning_$timestamp"
$logDir = "logs\$runTag"
$logFile = "$logDir\pipeline.log"

# 로그 디렉토리 생성
New-Item -ItemType Directory -Path $logDir -Force | Out-Null

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Phase 4 Alpha Tuning 파이프라인 실행" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Run Tag: $runTag" -ForegroundColor Yellow
Write-Host "From: $FromStage" -ForegroundColor Yellow
Write-Host "To: $ToStage" -ForegroundColor Yellow
Write-Host "로그 파일: $logFile" -ForegroundColor Yellow
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 파이프라인 실행 (로그 파일로 리다이렉트)
$command = "python -m src.core.pipeline --from $FromStage --to $ToStage --run-tag $runTag --skip-l2 --force --config $Config"

Write-Host "명령어 실행 중..." -ForegroundColor Green
Write-Host "실시간 로그 확인: Get-Content '$logFile' -Wait -Tail 20" -ForegroundColor Gray
Write-Host ""

# 표준 출력과 에러를 모두 로그 파일로 리다이렉트
& {
    $ErrorActionPreference = "Continue"
    Invoke-Expression $command 2>&1 | Tee-Object -FilePath $logFile
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "파이프라인 실행 완료" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "로그 파일: $logFile" -ForegroundColor Yellow





