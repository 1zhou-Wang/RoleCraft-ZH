<#
.SYNOPSIS
  MiniCPM5-2B 双 server 管理（greedy :8090 / official :8091）——RoleCraft-ZH 音频+文字 track 加测链路。

.DESCRIPTION
  * GGUF: E:\RBM\models\minicpm-gguf\MiniCPM5-2B-Q8_0.gguf（llama.cpp b11552 直接加载，LlamaForCausalLM 架构）
  * greedy :8090 —— server 不带采样 flag，请求端传 temperature=0（贪心）
  * official :8091 —— server 端带官方采样 flag：--temp 1.0 --top-p 0.95 --min-p 0.0 --seed 42
    （MiniCPM5 README：llama.cpp 默认 min_p=0.05 会加重读，必须 --min-p 0.0）
  * 两个 server 都尝试关闭 thinking：--reasoning off + --chat-template-kwargs '{"enable_thinking": false}'
  * 日志/pid 写 logs/minicpm5/
  * 运行时需要 llama_cpp\cudart 在 PATH（脚本自动处理）

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File local/serve_minicpm.ps1 -Action start
  powershell -NoProfile -ExecutionPolicy Bypass -File local/serve_minicpm.ps1 -Action status
  powershell -NoProfile -ExecutionPolicy Bypass -File local/serve_minicpm.ps1 -Action stop
#>
[CmdletBinding()]
param(
    [ValidateSet('start', 'stop', 'restart', 'status')]
    [string]$Action = 'start',

    [ValidateSet('greedy', 'official')]
    [string[]]$Models = @('greedy', 'official'),

    [int]$Ctx = 32768,
    [int]$NParallel = 4,
    [int]$Threads = 8,
    [int]$HealthTimeoutSec = 240,
    [switch]$NoWait,
    [switch]$EnableThinking
)

$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$LogDir      = Join-Path $ProjectRoot 'logs\minicpm5'
$BinDir      = 'E:\RBM\algorithms\draft\qwen3_5_infra_test\llama_cpp\bin'
$CudartDir   = 'E:\RBM\algorithms\draft\qwen3_5_infra_test\llama_cpp\cudart'
$Gguf        = 'E:\RBM\models\minicpm-gguf\MiniCPM5-2B-Q8_0.gguf'
$ServerExe   = Join-Path $BinDir 'llama-server.exe'

$Specs = @{
    'greedy'   = @{ Slug = 'minicpm5-2b-greedy';   Port = 8090; Official = $false }
    'official' = @{ Slug = 'minicpm5-2b-official'; Port = 8091; Official = $true }
}

function Get-PidFile([string]$key) { Join-Path $LogDir "llama-server-$key.pid" }

function Test-ServerHealth([int]$port) {
    try {
        $r = Invoke-RestMethod -Uri "http://127.0.0.1:$port/health" -TimeoutSec 5 -ErrorAction Stop
        return ($r.status -eq 'ok')
    } catch { return $false }
}

function Stop-One([string]$key) {
    $spec = $Specs[$key]
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
    $spec = $Specs[$key]
    $port = [int]$spec.Port
    $slug = $spec.Slug
    $pidFile = Get-PidFile $key

    if (Test-ServerHealth $port) {
        Write-Host "[$key] 已在运行: http://127.0.0.1:$port (health ok)，跳过启动"
        return
    }
    if (-not (Test-Path $Gguf)) {
        throw "[$key] GGUF 不存在: $Gguf（等待 lead 下载完成或从 ModelScope 镜像下载）"
    }
    if (Test-Path $CudartDir) { $env:PATH = "$CudartDir;$env:PATH" }

    New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
    $stdoutLog = Join-Path $LogDir "llama-server-$key.out.log"
    $stderrLog = Join-Path $LogDir "llama-server-$key.err.log"
    if (Test-Path $stdoutLog) { Remove-Item $stdoutLog -Force -ErrorAction SilentlyContinue }
    if (Test-Path $stderrLog) { Remove-Item $stderrLog -Force -ErrorAction SilentlyContinue }

    # thinking 关闭方式：用环境变量 LLAMA_ARG_CHAT_TEMPLATE_KWARGS 传 JSON，避免 Start-Process
    # -ArgumentList 的引号转义问题（命令行 JSON 带空格会被拆成多个 argv）。
    $env:LLAMA_ARG_CHAT_TEMPLATE_KWARGS = '{"enable_thinking":false}'
    $argList = @(
        '-m', $Gguf,
        '-c', "$Ctx",
        '-ngl', '999',
        '-fa', 'on',
        '-t', "$Threads",
        '-np', "$NParallel",
        '--host', '127.0.0.1',
        '--port', "$port",
        '--alias', $slug,
        '--jinja',
        '--no-webui'
    )
    if (-not $EnableThinking) {
        # role-play 评测与 Qwen 基线一致：关闭 thinking（模板变量 enable_thinking=false）
        $argList += @('--reasoning', 'off')
    }
    if ($spec.Official) {
        # 官方采样（min_p 必须 0，llama.cpp 默认 0.05 会加重读）+ repetition_penalty=1.05：
        # 实测 seed=42 下 rc01 偶发"快快快…"复读退化（1/10），按 README 建议补 1.05 惩罚
        $argList += @('--temp', '1.0', '--top-p', '0.95', '--min-p', '0.0', '--seed', '42',
                      '--repeat-penalty', '1.05')
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
            $mode = if ($spec.Official) { 'official(temp=1,top_p=0.95,min_p=0,seed=42)' } else { 'greedy' }
            Write-Host "[$key] 就绪: http://127.0.0.1:$port (model=$slug, ctx=$Ctx, np=$NParallel, sampling=$mode, thinking=$(-not $EnableThinking))"
            return
        }
        $p = Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
        if (-not $p) {
            $tail = if (Test-Path $stderrLog) { Get-Content $stderrLog -Tail 40 | Out-String } else { '' }
            throw "[$key] llama-server 进程已退出。stderr 末尾:`n$tail"
        }
        Start-Sleep -Seconds 2
    }
    $tail = if (Test-Path $stderrLog) { Get-Content $stderrLog -Tail 40 | Out-String } else { '' }
    throw "[$key] 等待 /health 超时（${HealthTimeoutSec}s）。stderr 末尾:`n$tail"
}

function Show-One([string]$key) {
    $spec = $Specs[$key]
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
        Write-Host '验证: curl.exe -s http://127.0.0.1:8090/health ; python local/audio_pipeline/minicpm_probe.py'
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
