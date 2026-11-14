<#
.SYNOPSIS
  RoleCraft-ZH 本地 llama-server 管理脚本（Qwen3.5-2B → :8088 / Qwen3.5-4B → :8089）。

.DESCRIPTION
  * 2B / 4B 均为本地 HF → bf16 GGUF → Q8_0 产物（E:\RBM\models\qwen-gguf）
  * 启动参数: -c <Ctx> -ngl 999 -fa on -t <Threads> -np <NParallel> --jinja --reasoning off
    （llama.cpp b11552 CUDA 预编译版；-c 会按 slot 均分，默认 32768/4 = 8192/slot）
  * 运行时需要 llama_cpp\cudart 在 PATH 中（脚本自动处理）
  * 日志与 pid 文件写入 logs/local/
  * 启动后轮询 /health，就绪才算成功

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File local/serve.ps1 -Action start
  powershell -NoProfile -ExecutionPolicy Bypass -File local/serve.ps1 -Action status
  powershell -NoProfile -ExecutionPolicy Bypass -File local/serve.ps1 -Action stop
  powershell -NoProfile -ExecutionPolicy Bypass -File local/serve.ps1 -Action start -Models 2b -NParallel 4
#>
[CmdletBinding()]
param(
    [ValidateSet('start', 'stop', 'restart', 'status')]
    [string]$Action = 'start',

    [ValidateSet('2b', '4b')]
    [string[]]$Models = @('2b', '4b'),

    # 注意: llama.cpp 把 -c 按 -np 均分给各 slot（日志 n_ctx_slot）。默认 32768/4 = 8192/slot，
    # 即每个会话 8K 上下文；单场景 system+5 轮对话约 1.5-2.5K token，8K/slot 足够。
    [int]$Ctx = 32768,
    [int]$NParallel = 4,
    [int]$Threads = 8,
    [int]$HealthTimeoutSec = 180,
    [switch]$NoWait,
    # v0.2 官方解码配置（Qwen3.5 README 非思考文本推荐，Plan A）：
    # temperature=1.0, top_p=1.0, top_k=20, min_p=0.0, presence_penalty=2.0, seed=42
    [switch]$Official
)

$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$LogDir      = Join-Path $ProjectRoot 'logs\local'
$BinDir      = 'E:\RBM\algorithms\draft\qwen3_5_infra_test\llama_cpp\bin'
$CudartDir   = 'E:\RBM\algorithms\draft\qwen3_5_infra_test\llama_cpp\cudart'
$GgufDir     = 'E:\RBM\models\qwen-gguf'
$ServerExe   = Join-Path $BinDir 'llama-server.exe'

$ModelSpecs = @{
    '2b' = @{ Slug = 'qwen3.5-2b'; Port = 8088; File = 'qwen35_2b-Q8_0.gguf' }
    '4b' = @{ Slug = 'qwen3.5-4b'; Port = 8089; File = 'qwen35_4b-Q8_0.gguf' }
}

function Get-PidFile([string]$key) { Join-Path $LogDir "llama-server_$key.pid" }

function Test-ServerHealth([int]$port) {
    try {
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -TimeoutSec 5 -ErrorAction Stop
        return ($r.status -eq 'ok')
    } catch {
        return $false
    }
}

function Stop-One([string]$key) {
    $spec = $ModelSpecs[$key]
    $port = [int]$spec.Port
    $pidFile = Get-PidFile $key
    $stopped = $false
    if (Test-Path $pidFile) {
        $serverPid = [int](Get-Content $pidFile -Raw).Trim()
        $p = Get-Process -Id $serverPid -ErrorAction SilentlyContinue
        if ($p -and $p.ProcessName -like 'llama-server*') {
            Write-Host "[$key] 停止 llama-server pid=$serverPid (port $port)"
            Stop-Process -Id $serverPid -Force -ErrorAction SilentlyContinue
            $stopped = $true
        }
        Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
    }
    # 兜底：按端口找监听进程
    try {
        $conn = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
        if ($conn) {
            foreach ($c in $conn) {
                $p = Get-Process -Id $c.OwningProcess -ErrorAction SilentlyContinue
                if ($p -and $p.ProcessName -like 'llama-server*') {
                    Write-Host "[$key] 按端口停止 llama-server pid=$($p.Id) (port $port)"
                    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
                    $stopped = $true
                }
            }
        }
    } catch { }
    if (-not $stopped) { Write-Host "[$key] 未发现运行中的 server (port $port)" }
}

function Start-One([string]$key) {
    $spec = $ModelSpecs[$key]
    $port = [int]$spec.Port
    $slug = $spec.Slug
    $gguf = Join-Path $GgufDir $spec.File
    $pidFile = Get-PidFile $key

    if (Test-ServerHealth $port) {
        Write-Host "[$key] 已在运行: http://127.0.0.1:$port (health ok)，跳过启动"
        return
    }
    if (-not (Test-Path $gguf)) {
        throw "[$key] GGUF 不存在: $gguf（先跑 local/convert.ps1 或检查 E:\RBM\models\qwen-gguf）"
    }

    # llama_server 需要 cudart 目录里的 cudart64_*.dll / cublas*.dll
    if (Test-Path $CudartDir) { $env:PATH = "$CudartDir;$env:PATH" }

    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
    $stdoutLog = Join-Path $LogDir "llama-server_$key.out.log"
    $stderrLog = Join-Path $LogDir "llama-server_$key.err.log"
    if (Test-Path $stdoutLog) { Remove-Item $stdoutLog -Force -ErrorAction SilentlyContinue }
    if (Test-Path $stderrLog) { Remove-Item $stderrLog -Force -ErrorAction SilentlyContinue }

    $argList = @(
        '-m', $gguf,
        '-c', "$Ctx",
        '-ngl', '999',
        '-fa', 'on',
        '-t', "$Threads",
        '-np', "$NParallel",
        '--host', '127.0.0.1',
        '--port', "$port",
        '--alias', $slug,
        '--jinja',
        # Qwen3.5 模板: enable_thinking=false → 空 <think></think> 块（非 thinking 模式）。
        # 必须显式 --reasoning off：auto/on 会让 server 传 enable_thinking=true，模型会先长篇思考。
        '--reasoning', 'off',
        '--no-webui'
    )
    if ($Official) {
        # 官方非思考文本推荐采样：presence_penalty（不是 repetition_penalty）+ 固定 seed
        $argList += @(
            '--temp', '1.0',
            '--top-k', '20',
            '--top-p', '1.0',
            '--min-p', '0.0',
            '--presence-penalty', '2.0',
            '--seed', '42'
        )
    }
    Write-Host "[$key] 启动: $ServerExe $($argList -join ' ')"
    Write-Host "[$key] 日志: $stdoutLog / $stderrLog"
    $proc = Start-Process -FilePath $ServerExe -ArgumentList $argList -WorkingDirectory $BinDir `
        -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog `
        -WindowStyle Hidden -PassThru
    Set-Content -Path $pidFile -Value $proc.Id -Encoding ascii
    Write-Host "[$key] pid=$($proc.Id), 等待 /health（最多 ${HealthTimeoutSec}s）..."

    if ($NoWait) { return }
    $deadline = (Get-Date).AddSeconds($HealthTimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (Test-ServerHealth $port) {
            $mode = if ($Official) { 'official(temp=1,top_k=20,top_p=1,min_p=0,presence=2,seed=42)' } else { 'default' }
            Write-Host "[$key] 就绪: http://127.0.0.1:$port (model=$slug, ctx=$Ctx, np=$NParallel, sampling=$mode)"
            return
        }
        $p = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
        if (-not $p) {
            $tail = if (Test-Path $stderrLog) { Get-Content $stderrLog -Tail 30 | Out-String } else { '' }
            throw "[$key] llama-server 进程已退出。stderr 末尾:`n$tail"
        }
        Start-Sleep -Seconds 2
    }
    $tail = if (Test-Path $stderrLog) { Get-Content $stderrLog -Tail 30 | Out-String } else { '' }
    throw "[$key] 等待 /health 超时（${HealthTimeoutSec}s）。stderr 末尾:`n$tail"
}

function Show-One([string]$key) {
    $spec = $ModelSpecs[$key]
    $port = [int]$spec.Port
    $pidFile = Get-PidFile $key
    $serverPid = $null
    if (Test-Path $pidFile) { $serverPid = (Get-Content $pidFile -Raw).Trim() }
    $ok = Test-ServerHealth $port
    $state = if ($ok) { 'READY' } else { 'DOWN ' }
    Write-Host ("[{0}] {1}  http://127.0.0.1:{2}  model={3}  pid={4}" -f $key, $state, $port, $spec.Slug, $serverPid)
}

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

switch ($Action) {
    'start' {
        foreach ($key in $Models) { Start-One $key }
        foreach ($key in $Models) { Show-One $key }
        Write-Host ''
        Write-Host '验证示例: curl.exe -s http://127.0.0.1:8088/health'
        Write-Host '基线示例: python local/run_baseline.py --model-slug qwen3.5-2b --base-url http://127.0.0.1:8088 --concurrency 4'
    }
    'stop' {
        foreach ($key in $Models) { Stop-One $key }
    }
    'restart' {
        foreach ($key in $Models) { Stop-One $key }
        Start-Sleep -Seconds 1
        foreach ($key in $Models) { Start-One $key }
        foreach ($key in $Models) { Show-One $key }
    }
    'status' {
        foreach ($key in $Models) { Show-One $key }
        try {
            $gpu = & nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>$null
            if ($gpu) { Write-Host "GPU 显存: $gpu" }
        } catch { }
    }
}
