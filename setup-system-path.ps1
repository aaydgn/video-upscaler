$ErrorActionPreference = "Stop"

$uvSource = Join-Path $env:USERPROFILE ".local\bin"
$uvTarget = "C:\Tools\uv\bin"
$ffmpegBin = "C:\Tools\ffmpeg\bin"

New-Item -ItemType Directory -Force -Path $uvTarget | Out-Null

foreach ($name in @("uv.exe", "uvx.exe", "uvw.exe")) {
    $source = Join-Path $uvSource $name
    if (Test-Path -LiteralPath $source) {
        Copy-Item -LiteralPath $source -Destination (Join-Path $uvTarget $name) -Force
    }
}

foreach ($required in @((Join-Path $uvTarget "uv.exe"), (Join-Path $ffmpegBin "ffmpeg.exe"))) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Missing required executable: $required"
    }
}

$machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
$parts = @($machinePath -split ";" | Where-Object { $_ })

foreach ($path in @($uvTarget, $ffmpegBin)) {
    if (-not ($parts | Where-Object { $_.TrimEnd("\") -ieq $path.TrimEnd("\") })) {
        $parts += $path
    }
}

[Environment]::SetEnvironmentVariable("Path", ($parts -join ";"), "Machine")

Write-Host "Added to system PATH:"
Write-Host "  $uvTarget"
Write-Host "  $ffmpegBin"
Write-Host "Open a new terminal before relying on the updated system PATH."
