#Requires -Version 5.0
<#
.SYNOPSIS
    启动 NeMo Speech 本机验证 / ASR 演示（wanggang-run-oss）。
.DESCRIPTION
    主路径：无参运行后交互选择。单服务会跳过菜单。
.PARAMETER Mode
    direct = 本机直接运行；docker = 仅提示（本项目无 compose 服务）。
.PARAMETER Port
    本项目服务无需端口；保留兼容。
.PARAMETER Service
    内部用：跳过菜单，直接启动指定 Id。
#>
[CmdletBinding()]
param(
    [ValidateSet('direct', 'docker')]
    [string]$Mode = 'direct',

    [int]$Port = 0,

    [string]$Service = ''
)

$ErrorActionPreference = 'Stop'

if ($PSVersionTable.PSVersion.Major -lt 7) {
    $pwsh = Get-Command pwsh -ErrorAction SilentlyContinue
    if (-not $pwsh) {
        throw '未找到 PowerShell 7 (pwsh)。请先全局安装：winget install --id Microsoft.PowerShell -e'
    }
    $argList = @('-NoProfile', '-File', $PSCommandPath)
    foreach ($key in $PSBoundParameters.Keys) {
        $argList += "-$key"
        $val = $PSBoundParameters[$key]
        if ($val -isnot [System.Management.Automation.SwitchParameter]) {
            $argList += [string]$val
        }
    }
    & $pwsh.Source @argList
    exit $LASTEXITCODE
}

Set-Location $PSScriptRoot

$FfmpegBin = 'E:\Programs\ffmpeg-master-latest-win64-gpl\bin'
if (Test-Path $FfmpegBin) {
    $env:Path = "$FfmpegBin;$env:Path"
} else {
    Write-Warning "未找到本机 ffmpeg：$FfmpegBin"
}

if (-not $env:HF_ENDPOINT) {
    $env:HF_ENDPOINT = 'https://hf-mirror.com'
}
if (-not $env:UV_INDEX_URL) {
    $env:UV_INDEX_URL = 'https://pypi.tuna.tsinghua.edu.cn/simple'
}
if (-not $env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD) {
    # NeMo .nemo checkpoints may need full pickle load (trusted local/HF weights only).
    $env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD = '1'
}

function Show-GpuStatus {
    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $smi) {
        Write-Warning '未检测到 nvidia-smi，跳过 GPU 检查。'
        return
    }
    Write-Host '=== GPU 状态 ===' -ForegroundColor Cyan
    & nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu --format=csv
}

function Test-PortBusy {
    param([int]$TargetPort)
    $listener = $null
    try {
        $listener = New-Object System.Net.Sockets.TcpListener ([System.Net.IPAddress]::Loopback, $TargetPort)
        $listener.Start()
        return $false
    } catch {
        return $true
    } finally {
        if ($null -ne $listener) { $listener.Stop() }
    }
}

function Get-FreePort {
    param(
        [int]$StartPort,
        [int]$MaxTries = 50
    )
    if ($StartPort -lt 1) { $StartPort = 1024 }
    $end = $StartPort + $MaxTries - 1
    if ($end -gt 65535) { $end = 65535 }
    $p = $StartPort
    while ($p -le $end) {
        if (-not (Test-PortBusy -TargetPort $p)) {
            if ($p -ne $StartPort) {
                Write-Host "端口 $StartPort 已占用，顺延到 $p" -ForegroundColor Yellow
            }
            return $p
        }
        $p++
    }
    throw "从 $StartPort 起连续探测均被占用，放弃。"
}

function Select-ServicesInteractive {
    param([object[]]$AllServices)

    if ($AllServices.Count -eq 0) {
        throw '未配置 $Services，请按项目改写模板。'
    }
    if ($AllServices.Count -eq 1) {
        Write-Host "仅一个服务，直接启动：$($AllServices[0].Label)" -ForegroundColor Cyan
        return @($AllServices[0])
    }

    Write-Host ''
    Write-Host '=== 启动哪些服务？===' -ForegroundColor Cyan
    for ($i = 0; $i -lt $AllServices.Count; $i++) {
        $svc = $AllServices[$i]
        $portHint = if ($svc.NeedsPort) { "端口起点 $($svc.PreferredPort)" } else { '无需端口' }
        Write-Host ("  [{0}] {1}  ({2})" -f ($i + 1), $svc.Label, $portHint)
    }
    Write-Host ("  [{0}] 全部开" -f ($AllServices.Count + 1))
    Write-Host '  [0] 取消'
    Write-Host ''

    $defaultChoice = '1'
    $raw = Read-Host "请选择（可多选，逗号分隔，如 1,2；默认 $defaultChoice）"
    if ([string]::IsNullOrWhiteSpace($raw)) { $raw = $defaultChoice }

    if ($raw.Trim() -eq '0') {
        throw '已取消启动。'
    }

    $allIndex = $AllServices.Count + 1
    $parts = $raw.Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne '' }
    $selected = [System.Collections.Generic.List[object]]::new()

    foreach ($part in $parts) {
        $n = 0
        if (-not [int]::TryParse($part, [ref]$n)) {
            throw "无效选项：$part"
        }
        if ($n -eq $allIndex) {
            return @($AllServices)
        }
        if ($n -lt 1 -or $n -gt $AllServices.Count) {
            throw "选项超出范围：$n"
        }
        $selected.Add($AllServices[$n - 1])
    }

    if ($selected.Count -eq 0) {
        throw '未选择任何服务。'
    }

    $byId = [ordered]@{}
    foreach ($s in $selected) { $byId[$s.Id] = $s }
    return @($byId.Values)
}

function Assert-GpuOkForSelection {
    param([object[]]$Selected)

    $gpuServices = @($Selected | Where-Object { $_.UsesGpu })
    if ($gpuServices.Count -le 1) { return }

    $labels = ($gpuServices | ForEach-Object { $_.Label }) -join ', '
    Write-Host ''
    Write-Host "已选多个占卡服务：$labels" -ForegroundColor Yellow
    Write-Host '16GB 显存下同时加载多个推理服务容易 OOM。请确认剩余显存足够，或改回只开一个。' -ForegroundColor Yellow
    $confirm = Read-Host '仍要继续？[y/N]'
    if ($confirm -notmatch '^[yY]') {
        throw '已取消：多服务占卡未确认。'
    }
}

function Get-VenvPython {
    $py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path $py)) {
        throw @"
未找到 .venv。请先按 docs/RUN.md 安装（Windows 须用 +cu128 直链 wheel，勿裸装 index）：
  uv venv --python 3.12
  uv pip install https://download.pytorch.org/whl/cu128/torch-2.9.1%2Bcu128-cp312-cp312-win_amd64.whl https://download.pytorch.org/whl/cu128/torchaudio-2.9.1%2Bcu128-cp312-cp312-win_amd64.whl
  uv pip install -e `".[asr]`"
"@
    }
    return $py
}

$Services = @(
    [pscustomobject]@{
        Id            = 'verify'
        Label         = 'verify（导入检查 + CUDA）'
        PreferredPort = 0
        NeedsPort     = $false
        UsesGpu       = $false
    }
    [pscustomobject]@{
        Id            = 'asr-demo'
        Label         = 'asr-demo（Parakeet 0.6B 转写）'
        PreferredPort = 0
        NeedsPort     = $false
        UsesGpu       = $true
    }
)

function Ensure-AsrDemoScript {
    $dir = Join-Path $PSScriptRoot 'scripts_local'
    if (-not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir | Out-Null
    }
    $py = Join-Path $dir 'asr_demo.py'
    if (Test-Path $py) { return $py }

    $content = @'
"""One-shot ASR demo for local NeMo Speech (wanggang-run-oss)."""
from __future__ import annotations

import os
import urllib.request
from pathlib import Path

SAMPLE_URL = "https://hf-mirror.com/datasets/Narsil/asr_dummy/resolve/main/1.flac"
SAMPLE_PATH = Path(__file__).resolve().parent / "asr_dummy_1.flac"
MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"


def main() -> None:
    if not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

    if not SAMPLE_PATH.is_file():
        print(f"Downloading sample: {SAMPLE_URL}")
        req = urllib.request.Request(SAMPLE_URL, headers={"User-Agent": "nemo-speech-local-demo"})
        with urllib.request.urlopen(req, timeout=120) as resp, open(SAMPLE_PATH, "wb") as out:
            out.write(resp.read())

    import torch
    import nemo.collections.asr as nemo_asr

    print("cuda:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    print(f"Loading {MODEL_NAME} ...")
    asr_model = nemo_asr.models.ASRModel.from_pretrained(MODEL_NAME)
    transcript = asr_model.transcribe([str(SAMPLE_PATH)])[0].text
    print("transcript:", transcript)


if __name__ == "__main__":
    main()
'@
    [System.IO.File]::WriteAllText($py, $content, [System.Text.UTF8Encoding]::new($false))
    return $py
}

function Start-ProjectService {
    param(
        [Parameter(Mandatory)]
        [object]$Service,

        [Parameter(Mandatory)]
        [string]$RunMode,

        [int]$ListenPort = 0
    )

    if ($RunMode -eq 'docker') {
        throw '本项目未配置 docker compose 服务。请用 -Mode direct，或自行 docker pull nvcr.io/nvidia/nemo-speech:26.07.00'
    }

    # 勿用 `uv run`：会按 uv.lock 同步并在 Windows 上换成 CPU torch。
    $python = Get-VenvPython

    switch ($Service.Id) {
        'verify' {
            Write-Host '运行 verify...' -ForegroundColor Cyan
            & $python -c "import torch; import nemo.collections.asr as nemo_asr; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None); print('torch', torch.__version__); print('NeMo Speech ASR OK')"
            if ($LASTEXITCODE -ne 0) { throw "verify 失败，退出码 $LASTEXITCODE" }
            Write-Host 'verify 通过' -ForegroundColor Green
            return
        }
        'asr-demo' {
            $script = Ensure-AsrDemoScript
            Write-Host "运行 asr-demo: $script" -ForegroundColor Cyan
            & $python $script
            if ($LASTEXITCODE -ne 0) { throw "asr-demo 失败，退出码 $LASTEXITCODE" }
            Write-Host 'asr-demo 完成' -ForegroundColor Green
            return
        }
        default {
            throw "未知服务：$($Service.Id)"
        }
    }
}

Show-GpuStatus

if ($Service) {
    $match = @($Services | Where-Object { $_.Id -eq $Service })
    if ($match.Count -eq 0) {
        throw "未知服务 Id：$Service。可选：$($Services.Id -join ', ')"
    }
    $chosen = $match
} else {
    $chosen = Select-ServicesInteractive -AllServices $Services
    Assert-GpuOkForSelection -Selected $chosen
}

$multi = $chosen.Count -gt 1

if ($multi) {
    $started = @()
    foreach ($svc in $chosen) {
        $listen = 0
        if ($svc.NeedsPort) {
            $listen = Get-FreePort -StartPort $svc.PreferredPort
            Write-Host "$($svc.Label) 使用端口 $listen" -ForegroundColor Cyan
        }

        $argList = [System.Collections.Generic.List[string]]::new()
        $argList.AddRange([string[]]@('-NoProfile', '-File', $PSCommandPath, '-Mode', $Mode, '-Service', $svc.Id))
        if ($listen -gt 0) {
            $argList.Add('-Port')
            $argList.Add("$listen")
        }

        $p = Start-Process -FilePath 'pwsh' -ArgumentList $argList -PassThru -WorkingDirectory $PSScriptRoot
        $started += [pscustomobject]@{ Id = $svc.Id; Label = $svc.Label; Port = $listen; Pid = $p.Id }
        Write-Host "已后台启动 $($svc.Label) PID=$($p.Id)" -ForegroundColor Green
    }

    Write-Host ''
    Write-Host '=== 已启动 ===' -ForegroundColor Cyan
    foreach ($s in $started) {
        $portInfo = if ($s.Port -gt 0) { " port=$($s.Port)" } else { '' }
        Write-Host ("- {0}{1} pid={2}" -f $s.Label, $portInfo, $s.Pid)
    }
    Write-Host '各服务在独立进程中运行；结束请自行停对应 PID。' -ForegroundColor Yellow
    return
}

$svc = $chosen[0]
$listen = 0
if ($svc.NeedsPort) {
    $startPort = $svc.PreferredPort
    if ($Port -gt 0) { $startPort = $Port }
    $listen = Get-FreePort -StartPort $startPort
    Write-Host "$($svc.Label) 使用端口 $listen" -ForegroundColor Cyan
}

Start-ProjectService -Service $svc -RunMode $Mode -ListenPort $listen
