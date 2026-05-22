#Requires -Version 5.1
<#
.SYNOPSIS
  Stop the hold-to-talk voice-to-text daemon.
.DESCRIPTION
  Reads the recorded PID file first, falls back to scanning python.exe
  command lines for stt-daemon.py if the file is missing or stale.
#>
$ErrorActionPreference = 'SilentlyContinue'

$pidPath = Join-Path $PSScriptRoot 'stt-daemon.pid'
$killed     = $false
$primaryPid = $null

if (Test-Path -LiteralPath $pidPath) {
    $primaryPid = (Get-Content -LiteralPath $pidPath) -as [int]
    if ($primaryPid -and (Get-Process -Id $primaryPid -ErrorAction SilentlyContinue)) {
        Stop-Process -Id $primaryPid -Force
        Write-Host "Stopped daemon (PID $primaryPid)."
        $killed = $true
    }
    Remove-Item -LiteralPath $pidPath -ErrorAction SilentlyContinue
}

# Give the OS a beat to clear the process record so the fallback scan
# doesn't re-discover the PID we just killed.
Start-Sleep -Milliseconds 250

# Fallback: any orphan python running stt-daemon.py that wasn't the one
# we already killed (e.g. PID file was stale or a second daemon got
# launched some other way).
$procs = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -like '*stt-daemon.py*' -and $_.ProcessId -ne $primaryPid }
foreach ($p in $procs) {
    Stop-Process -Id $p.ProcessId -Force
    Write-Host "Stopped orphan daemon (PID $($p.ProcessId))."
    $killed = $true
}

if (-not $killed) {
    Write-Host "STT daemon was not running."
}
