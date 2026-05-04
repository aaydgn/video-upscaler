$ErrorActionPreference = "Stop"

$pathsToAdd = @(
    "$env:USERPROFILE\.local\bin",
    "C:\Tools\ffmpeg\bin",
    "C:\Tools\rife-ncnn-vulkan"
)

$current = [Environment]::GetEnvironmentVariable("Path", "User")
$parts = @($current -split ";" | Where-Object { $_ })

foreach ($path in $pathsToAdd) {
    if ($parts -notcontains $path) {
        $parts += $path
    }
}

$newPath = $parts -join ";"
[Environment]::SetEnvironmentVariable("Path", $newPath, "User")

Write-Host "Updated user PATH:"
Write-Host $newPath
Write-Host "Open a new terminal before relying on the updated PATH."
