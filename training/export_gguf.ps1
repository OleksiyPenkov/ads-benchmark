<#
.SYNOPSIS
Merges a LoRA adapter into its base model, converts the result to f16 GGUF and quantizes it with llama.cpp.

.PARAMETER StrataPack
When -AdapterDir contains output_hc_adapter.pt (trained by strata_hc_trainer.py), exports the patched
Strata pack via strata_hc_trainer.py --export-pack. Otherwise prints the quantized GGUF paths.

.PARAMETER WhatIf
Prints every command without running it. -DryRun does the same.
#>
[CmdletBinding(SupportsShouldProcess = $true)]
param(
    [Parameter(Mandatory = $true)][string]$AdapterDir,
    [Parameter(Mandatory = $true)][string]$BaseModel,
    [Parameter(Mandatory = $true)][string]$Name,
    [Parameter(Mandatory = $true)][string]$LlamaCppDir,
    [string]$OutDir = '',
    [string[]]$Quants = @('IQ3_S', 'IQ4_XS', 'Q4_K_M'),
    [string]$ImatrixFile,
    [string]$Python,
    [ValidateSet('bfloat16', 'float16', 'float32')][string]$MergeDtype = 'bfloat16',
    [switch]$DryRun,
    [switch]$StrataPack
)

$ErrorActionPreference = 'Stop'
$dry = $DryRun.IsPresent -or $WhatIfPreference
$repoRoot = Split-Path -Parent $PSScriptRoot
$computedRoot = if ($env:ADS_BENCH_COMPUTED_DIR) { $env:ADS_BENCH_COMPUTED_DIR } else { Join-Path $repoRoot 'computed' }
if (-not $OutDir) {
    $OutDir = Join-Path $computedRoot 'models\gguf'
}
$trainScript = Join-Path $PSScriptRoot 'train_sft_dpo.py'
$venvPython = Join-Path $computedRoot 'venv-train\Scripts\python.exe'
if (-not $Python) {
    $Python = if (Test-Path -LiteralPath $venvPython) { $venvPython } else { 'python' }
}

$missing = New-Object System.Collections.Generic.List[string]
function Require-Path([string]$Path, [string]$What) {
    if (Test-Path -LiteralPath $Path) { return }
    $missing.Add("$What not found: $Path")
    if (-not $dry) { throw $missing[$missing.Count - 1] }
}

Require-Path $trainScript 'train_sft_dpo.py'
Require-Path (Join-Path $AdapterDir 'adapter_config.json') 'adapter_config.json in -AdapterDir'
Require-Path (Join-Path $AdapterDir 'adapter_model.safetensors') 'adapter_model.safetensors in -AdapterDir'
if ((Test-Path -LiteralPath $BaseModel) -or ($BaseModel.StartsWith('.'))) {
    Require-Path $BaseModel '-BaseModel directory'
}
Require-Path $LlamaCppDir '-LlamaCppDir'
$convertScript = Join-Path $LlamaCppDir 'convert_hf_to_gguf.py'
Require-Path $convertScript 'convert_hf_to_gguf.py'
if ($ImatrixFile) { Require-Path $ImatrixFile '-ImatrixFile' }

$quantizeCandidates = @(
    (Join-Path $LlamaCppDir 'llama-quantize.exe'),
    (Join-Path $LlamaCppDir 'build\bin\Release\llama-quantize.exe'),
    (Join-Path $LlamaCppDir 'build\bin\llama-quantize.exe'),
    (Join-Path $LlamaCppDir 'bin\llama-quantize.exe'),
    (Join-Path $LlamaCppDir 'llama-quantize')
)
$quantize = $quantizeCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $quantize) {
    $quantize = $quantizeCandidates[0]
    $missing.Add("llama-quantize not found under -LlamaCppDir; looked in: $($quantizeCandidates -join '; ')")
    if (-not $dry) { throw $missing[$missing.Count - 1] }
}

if (-not $dry -and -not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    throw "Python interpreter not found: $Python"
}

$mergedDir = Join-Path $OutDir "$Name-merged"
$f16 = Join-Path $OutDir "$Name-f16.gguf"

function Invoke-Step([string]$Label, [string]$Exe, [string[]]$Arguments) {
    $rendered = ($Arguments | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join ' '
    Write-Host "[$Label] $Exe $rendered"
    if ($dry) { return }
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "$Label failed with exit code $LASTEXITCODE" }
}

if (-not $dry) { New-Item -ItemType Directory -Force -Path $OutDir | Out-Null }

Invoke-Step 'merge' $Python @($trainScript, '--merge-adapter', '--model-path', $BaseModel, '--adapter-dir', $AdapterDir, '--merged-dir', $mergedDir, '--merge-dtype', $MergeDtype)
Invoke-Step 'convert' $Python @($convertScript, $mergedDir, '--outtype', 'f16', '--outfile', $f16)

$outputs = @()
foreach ($quant in $Quants) {
    $target = Join-Path $OutDir "$Name-$quant.gguf"
    $arguments = @()
    if ($ImatrixFile) { $arguments += @('--imatrix', $ImatrixFile) }
    $arguments += @($f16, $target, $quant)
    Invoke-Step "quantize-$quant" $quantize $arguments
    $outputs += $target
}

if ($StrataPack) {
    $hcTrainer = Join-Path $PSScriptRoot 'strata_hc_trainer.py'
    $hcAdapter = Join-Path $AdapterDir 'output_hc_adapter.pt'
    if ((Test-Path -LiteralPath $hcTrainer) -and (Test-Path -LiteralPath $hcAdapter)) {
        Invoke-Step 'strata-pack' $Python @($hcTrainer, '--export-pack', '--adapter-out', $hcAdapter)
    } else {
        Write-Warning 'StrataPack: output_hc_adapter.pt not found in -AdapterDir; printing quantized GGUF files instead.'
        $outputs | ForEach-Object { Write-Host "  $_" }
    }
}

if ($missing.Count -gt 0) {
    $missing | ForEach-Object { Write-Warning "MISSING: $_" }
    exit 1
}
Write-Host ($(if ($dry) { 'dry run complete' } else { 'export complete' }))
