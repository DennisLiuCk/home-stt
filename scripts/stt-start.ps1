#Requires -Version 5.1
<#
.SYNOPSIS
  Start the hold-to-talk voice-to-text daemon in the background.
.DESCRIPTION
  Launches stt-daemon.py via python.exe with stdout/stderr captured to
  %TEMP%\stt-daemon.log. Refuses to start a second copy if one is already
  running. PID is recorded next to this script.
#>
$ErrorActionPreference = 'Stop'

$daemonPath = Join-Path $PSScriptRoot 'stt-daemon.py'
$logPath    = Join-Path $env:TEMP    'stt-daemon.log'
$errPath    = Join-Path $env:TEMP    'stt-daemon.err.log'
$pidPath    = Join-Path $PSScriptRoot 'stt-daemon.pid'

if (-not (Test-Path -LiteralPath $daemonPath)) {
    Write-Error "Daemon script not found: $daemonPath"
    exit 1
}

# Already running?
$existing = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like '*stt-daemon.py*' }
if ($existing) {
    Write-Host "STT daemon already running (PID $($existing.ProcessId | Select-Object -First 1))."
    exit 0
}

$env:PYTHONIOENCODING = 'utf-8'
$proc = Start-Process python `
    -ArgumentList @('-u', $daemonPath) `
    -RedirectStandardOutput $logPath `
    -RedirectStandardError  $errPath `
    -WindowStyle Hidden `
    -PassThru

$proc.Id | Set-Content -LiteralPath $pidPath -Encoding ascii

Write-Host "STT daemon started (PID $($proc.Id))."
Write-Host "Log: $logPath"
Write-Host "Allow ~45s for model load + GPU warmup before first trigger key (v0.7.0+ loads Qwen3-ASR-0.6B + Qwen3-4B-Instruct-2507)."
