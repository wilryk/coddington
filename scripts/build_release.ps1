<#
.SYNOPSIS
    Builds the Coddington desktop app and Windows installer, start to finish.

.DESCRIPTION
    One command from a clean checkout to a finished installer:

      1. Creates a fresh virtual environment (skip with -ReuseVenv) and
         installs the package plus its web and packaging dependencies.
      2. Runs PyInstaller against packaging/desktop/coddington.spec.
      3. Smoke-tests the frozen build: launches it, waits for it to answer,
         and checks that it serves the real app -- not a stale bundle --
         and that nothing in the page points outside the machine.
      4. Builds the Windows installer with Inno Setup, named and versioned
         from the package's own version string.

    Run from anywhere; it locates the repository root itself.

.PARAMETER SkipInstaller
    Build and smoke-test the app but skip the installer step. Useful if
    Inno Setup isn't installed, or you only want the app bundle.

.PARAMETER ReuseVenv
    Reuse an existing .build-venv instead of recreating it from scratch.
    Faster for repeat builds; the default rebuilds it every time so a
    build never runs against stale or hand-edited dependencies.

.EXAMPLE
    powershell -File scripts\build_release.ps1

.EXAMPLE
    powershell -File scripts\build_release.ps1 -ReuseVenv -SkipInstaller
#>

param(
    [switch]$SkipInstaller,
    [switch]$ReuseVenv
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Write-Step($text) {
    Write-Host ""
    Write-Host "==> $text" -ForegroundColor Cyan
}

# ---------------------------------------------------------------------------
# 1. Virtual environment

$venvDir = Join-Path $root ".build-venv"
if (-not $ReuseVenv) {
    if (Test-Path $venvDir) {
        Write-Step "Removing existing .build-venv"
        Remove-Item -Recurse -Force $venvDir
    }
}
if (-not (Test-Path $venvDir)) {
    Write-Step "Creating a virtual environment for the build"
    python -m venv $venvDir
    if ($LASTEXITCODE -ne 0) { throw "Could not create the build virtual environment." }
}
$venvPython = Join-Path $venvDir "Scripts\python.exe"

Write-Step "Installing heliostat[web] and PyInstaller"
& $venvPython -m pip install --upgrade pip -q
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }
& $venvPython -m pip install -e "$root[web]" pyinstaller -q
if ($LASTEXITCODE -ne 0) { throw "Dependency install failed." }

$version = (& $venvPython -c "import heliostat; print(heliostat.__version__)").Trim()
Write-Host "    heliostat $version"

# ---------------------------------------------------------------------------
# 2. PyInstaller

Write-Step "Running PyInstaller"
& $venvPython -m PyInstaller "packaging\desktop\coddington.spec" --noconfirm
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed." }

$distDir = Join-Path $root "dist\Coddington"
$exePath = Join-Path $distDir "Coddington.exe"
if (-not (Test-Path $exePath)) { throw "Build finished but $exePath is missing." }

# ---------------------------------------------------------------------------
# 3. Smoke test: launch the frozen build and check what it actually serves

Write-Step "Smoke-testing the frozen build"
$port = 8493
$proc = Start-Process -FilePath $exePath `
    -ArgumentList @("serve", "--no-browser", "--port", $port) `
    -PassThru -WindowStyle Hidden

try {
    $healthy = $false
    for ($i = 0; $i -lt 60; $i++) {
        try {
            $resp = Invoke-WebRequest -Uri "http://127.0.0.1:$port/api/health" `
                -UseBasicParsing -TimeoutSec 2
            if ($resp.StatusCode -eq 200) { $healthy = $true; break }
        } catch {
            Start-Sleep -Milliseconds 500
        }
    }
    if (-not $healthy) {
        throw "The frozen build never answered on port $port -- see the console window it opened."
    }

    $index = Invoke-WebRequest -Uri "http://127.0.0.1:$port/" -UseBasicParsing

    if ($index.Content -notmatch 'apptab-workspace') {
        throw "The served page is not the workspace shell -- check the static-file bundling in coddington.spec."
    }
    if ($index.Content -match 'https?://(?!www\.w3\.org)') {
        throw "The served page references an external URL -- the offline guarantee is broken."
    }

    Write-Host "    OK: serves the workspace shell on http://127.0.0.1:$port/, no external references"
} finally {
    if ($proc -and -not $proc.HasExited) {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    }
}

# ---------------------------------------------------------------------------
# 4. Installer

if ($SkipInstaller) {
    Write-Step "Skipping installer (-SkipInstaller was passed)"
} else {
    $iscc = (Get-Command iscc.exe -ErrorAction SilentlyContinue).Source
    if (-not $iscc) {
        foreach ($candidate in @(
            "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
            "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
            "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
        )) {
            if (Test-Path $candidate) { $iscc = $candidate; break }
        }
    }

    if (-not $iscc) {
        Write-Warning "Inno Setup (iscc) was not found, so no installer was built."
        Write-Warning "Install it with: winget install --id JRSoftware.InnoSetup -e"
        Write-Warning "or from https://jrsoftware.org/isinfo.php, then re-run this script."
    } else {
        Write-Step "Building the installer with Inno Setup"
        New-Item -ItemType Directory -Force -Path (Join-Path $root "dist\installer") | Out-Null
        & $iscc "packaging\desktop\coddington.iss" "/DMyAppVersion=$version"
        if ($LASTEXITCODE -ne 0) { throw "Inno Setup compile failed." }
    }
}

# ---------------------------------------------------------------------------
# 5. Summary

Write-Step "Done"
$exeSize = "{0:N0}" -f ((Get-Item $exePath).Length / 1MB)
Write-Host "    App:       $exePath ($exeSize MB launcher; dist\Coddington\ is the full folder to zip)"
$installer = Join-Path $root "dist\installer\Coddington-Setup-$version.exe"
if (Test-Path $installer) {
    $instSize = "{0:N0}" -f ((Get-Item $installer).Length / 1MB)
    Write-Host "    Installer: $installer ($instSize MB)"
}
