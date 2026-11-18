# Build the ICME 2027 draft without latexmk (no perl on this machine).
# Usage:  powershell -ExecutionPolicy Bypass -File tools\build.ps1 [-Clean]
param([switch]$Clean)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if ($Clean) {
    Get-ChildItem -File -Include *.aux,*.bbl,*.blg,*.log,*.out,*.synctex.gz |
        Remove-Item -Force -ErrorAction SilentlyContinue
    "cleaned"
}

"[1/3] pdflatex ..."
pdflatex -interaction=nonstopmode -halt-on-error main.tex | Out-Null
if ($LASTEXITCODE -ne 0) { "FAILED at pass 1 - see main.log"; exit 1 }

"[2/3] bibtex ..."
if (Test-Path main.aux) { bibtex main | Out-Null }

"[3/3] pdflatex x2 ..."
pdflatex -interaction=nonstopmode main.tex | Out-Null
pdflatex -interaction=nonstopmode main.tex | Out-Null

$log = Join-Path $root "main.log"
$pages = "-"
if (Test-Path $log) {
    $m = Select-String -Path $log -Pattern "Output written on main\.pdf \((\d+) pages?" |
        Select-Object -Last 1
    if ($m) { $pages = $m.Matches[0].Groups[1].Value }
}
$size = "-"
$pdf = Join-Path $root "main.pdf"
if (Test-Path $pdf) { $size = "{0:N0} KB" -f ((Get-Item $pdf).Length / 1KB) }
"OK  main.pdf  $pages pages  $size"
