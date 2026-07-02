param(
    [switch]$SkipInstall,
    [switch]$SkipDemo,
    [switch]$SkipTests,
    [switch]$BuildManual,
    [switch]$InstallPytest,
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [string]$Label,
        [scriptblock]$Action
    )
    Write-Host ""
    Write-Host "==> $Label" -ForegroundColor Cyan
    & $Action
}

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

Write-Host "QuantumTransportEOM automation runner" -ForegroundColor Green
Write-Host "Project root: $projectRoot"
Write-Host "Python command: $Python"

if (-not $SkipInstall) {
    Invoke-Step "Installing package in editable mode" {
        & $Python -m pip install -e .
    }
}

$env:PYTHONPATH = Join-Path $projectRoot "src"
Write-Host "PYTHONPATH=$env:PYTHONPATH"

if (-not $SkipDemo) {
    Invoke-Step "Running demo" {
        & $Python examples/demo.py
    }
}

$runTests = -not $SkipTests
if ($runTests) {
    $pytestOk = $true
    try {
        & $Python -c "import pytest"
    } catch {
        $pytestOk = $false
    }

    if (-not $pytestOk -and $InstallPytest) {
        Invoke-Step "Installing pytest" {
            & $Python -m pip install pytest
        }
        $pytestOk = $true
    }

    if ($pytestOk) {
        Invoke-Step "Running full test suite" {
            & $Python -m pytest tests
        }
    } else {
        Write-Warning "pytest is not installed. Re-run with -InstallPytest to install it automatically."
    }
}

if ($BuildManual) {
    $docsDir = Join-Path $projectRoot "docs"
    if (-not (Test-Path $docsDir)) {
        Write-Warning "docs directory not found: $docsDir"
    } else {
        $pdflatex = Get-Command pdflatex -ErrorAction SilentlyContinue
        if ($null -eq $pdflatex) {
            Write-Warning "pdflatex not found. Install TeX (MiKTeX or TeX Live) to build PDF."
        } else {
            Invoke-Step "Building user manual PDF" {
                Push-Location $docsDir
                pdflatex -interaction=nonstopmode -halt-on-error user_manual.tex
                pdflatex -interaction=nonstopmode -halt-on-error user_manual.tex
                Pop-Location
            }
        }
    }
}

Write-Host ""
Write-Host "All requested steps completed." -ForegroundColor Green
