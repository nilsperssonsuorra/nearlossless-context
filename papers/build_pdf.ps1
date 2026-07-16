# Build papers/main.pdf with MiKTeX / TeX Live pdflatex.
# Usage (from repo root or papers/):
#   powershell -File papers/build_pdf.ps1
#   cd papers; .\build_pdf.ps1

$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Here

$pdflatex = Get-Command pdflatex -ErrorAction SilentlyContinue
if (-not $pdflatex) {
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe",
        "C:\Program Files\MiKTeX\miktex\bin\x64\pdflatex.exe",
        "C:\texlive\2024\bin\windows\pdflatex.exe"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { $pdflatex = $c; break }
    }
}
if (-not $pdflatex) {
    Write-Error "pdflatex not found. Install MiKTeX or TeX Live and re-run."
}

Write-Host "Using: $pdflatex" -ForegroundColor Cyan
Write-Host "Building main.tex (2 passes)..." -ForegroundColor Cyan

# MiKTeX may write warnings to stderr; treat success by PDF presence + log.
$prevEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $pdflatex -interaction=nonstopmode main.tex | Out-Host
& $pdflatex -interaction=nonstopmode main.tex | Out-Host
$ErrorActionPreference = $prevEap
if (-not (Test-Path (Join-Path $Here "main.pdf"))) {
    throw "pdflatex failed (no main.pdf)"
}

$out = Join-Path $Here "main.pdf"
if (-not (Test-Path $out)) { throw "main.pdf was not produced" }

# Open if interactive
Write-Host "OK: $out" -ForegroundColor Green
try {
    $size = (Get-Item $out).Length
    Write-Host ("Size: {0:N1} KB" -f ($size / 1KB))
} catch {}
