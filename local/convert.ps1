<#
.SYNOPSIS
  Qwen3.5-2B / 4B 本地 HF → GGUF(bf16) → Q8_0 转换脚本（RoleCraft-ZH 本地推理链路）。

.DESCRIPTION
  * 转换脚本用 qwen3_5_infra_test 的 venv（torch 2.14 / transformers 5.19 / gguf），不要用项目 .venv
  * llama.cpp 源码: E:\RBM\algorithms\draft\qwen3_5_infra_test\llama_src\llama.cpp-b11552（b11552 原生支持 qwen35）
  * 关键参数 --no-mtp: checkpoint 带 mtp.* 权重，文本 baseline 不需要 MTP/speculative 层，直接排除
  * vision tower 权重（model.visual.*）由 convert 脚本的 TEXT 路线自动跳过（文本 track 只用文本）
  * 产物: E:\RBM\models\qwen-gguf\qwen35_{2b,4b}-bf16.gguf + qwen35_{2b,4b}-Q8_0.gguf
  * 日志: logs/local/convert_*.log

.EXAMPLE
  powershell -NoProfile -ExecutionPolicy Bypass -File local/convert.ps1 -Models 2b
  powershell -NoProfile -ExecutionPolicy Bypass -File local/convert.ps1 -Models all -SkipQuantize
#>
[CmdletBinding()]
param(
    [ValidateSet('2b', '4b', 'all')]
    [string]$Models = 'all',
    [switch]$SkipConvert,   # 跳过 HF→bf16
    [switch]$SkipQuantize,  # 跳过 bf16→Q8_0
    [switch]$KeepBf16       # 默认转换完 Q8_0 后删除 bf16 中间产物
)

$ErrorActionPreference = 'Stop'

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$LogDir      = Join-Path $ProjectRoot 'logs\local'
$Python      = 'E:\RBM\algorithms\draft\qwen3_5_infra_test\qwen_bench_win\.venv\Scripts\python.exe'
$LlamaSrc    = 'E:\RBM\algorithms\draft\qwen3_5_infra_test\llama_src\llama.cpp-b11552'
$LlamaBin    = 'E:\RBM\algorithms\draft\qwen3_5_infra_test\llama_cpp\bin'
$CudartDir   = 'E:\RBM\algorithms\draft\qwen3_5_infra_test\llama_cpp\cudart'
$GgufDir     = 'E:\RBM\models\qwen-gguf'

$Specs = @{
    '2b' = @{ HfDir = 'E:\RBM\models\Qwen3.5-2B'; Bf16 = 'qwen35_2b-bf16.gguf'; Q8 = 'qwen35_2b-Q8_0.gguf' }
    '4b' = @{ HfDir = 'E:\RBM\models\Qwen3.5-4B'; Bf16 = 'qwen35_4b-bf16.gguf'; Q8 = 'qwen35_4b-Q8_0.gguf' }
}

New-Item -ItemType Directory -Force -Path $LogDir, $GgufDir | Out-Null
if ($Models -eq 'all') { $todo = @('2b', '4b') } else { $todo = @($Models) }

foreach ($key in $todo) {
    $spec = $Specs[$key]
    $bf16Path = Join-Path $GgufDir $spec.Bf16
    $q8Path   = Join-Path $GgufDir $spec.Q8

    if (-not $SkipConvert -and -not (Test-Path $bf16Path)) {
        $log = Join-Path $LogDir "convert_${key}_bf16.log"
        Write-Host "[$key] HF→bf16: $($spec.HfDir) → $bf16Path"
        Push-Location $LlamaSrc
        try {
            & $Python (Join-Path $LlamaSrc 'convert_hf_to_gguf.py') $spec.HfDir `
                --outfile $bf16Path --outtype bf16 --no-mtp *>&1 |
                Tee-Object -FilePath $log
            if ($LASTEXITCODE -ne 0) { throw "[$key] convert 失败 (exit $LASTEXITCODE)，日志: $log" }
        } finally { Pop-Location }
    } elseif (Test-Path $bf16Path) {
        Write-Host "[$key] 已有 bf16，跳过转换: $bf16Path"
    }

    if (-not $SkipQuantize) {
        if (-not (Test-Path $bf16Path)) { throw "[$key] 缺少 bf16: $bf16Path" }
        $log = Join-Path $LogDir "quantize_${key}_q8.log"
        Write-Host "[$key] bf16→Q8_0: $q8Path"
        if (Test-Path $CudartDir) { $env:PATH = "$CudartDir;$env:PATH" }
        & (Join-Path $LlamaBin 'llama-quantize.exe') $bf16Path $q8Path Q8_0 *>&1 |
            Tee-Object -FilePath $log
        if ($LASTEXITCODE -ne 0) { throw "[$key] quantize 失败 (exit $LASTEXITCODE)，日志: $log" }
        if (-not $KeepBf16) {
            Write-Host "[$key] 删除 bf16 中间产物: $bf16Path"
            Remove-Item $bf16Path -Force
        }
        Get-Item $q8Path | Select-Object FullName, Length
    }
}
Write-Host '完成。用 local/serve.ps1 -Action start 启动服务。'
