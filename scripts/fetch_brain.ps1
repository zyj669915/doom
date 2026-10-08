# fetch_brain.ps1 — 从 GitHub Releases 拉取果蝇脑图版本
# 用法：
#   .\scripts\fetch_brain.ps1                      # 默认拉 brain-v0-original
#   .\scripts\fetch_brain.ps1 -Tag brain-v1        # 拉指定版本（训练后新脑）
#   .\scripts\fetch_brain.ps1 -OutDir D:\fly\doomfly\outputs\doom\malecns_v1
#
# Releases 页面：https://github.com/zyj669915/doom/releases
param(
    [string]$Tag = "brain-v0-original",
    [string]$OutDir = "D:\fly\doomfly\outputs\doom\malecns_v1",
    [string]$Repo  = "zyj669915/doom"
)

$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

# 版本包含的文件（与 Releases 上传时一致）
$files = @(
    @{ Name = "graph.npz";          SizeMB = 239 },
    @{ Name = "edges.feather";      SizeMB = 1003 },
    @{ Name = "edges.arrow";         SizeMB = 293 }
)

$base = "https://github.com/$Repo/releases/download/$Tag"
Write-Host "拉取脑图版本: $Tag" -ForegroundColor Cyan
Write-Host "目标目录: $OutDir`n"

foreach ($f in $files) {
    $url  = "$base/$($f.Name)"
    $dest = Join-Path $OutDir $f.Name
    # edges.arrow 放到 normalized 子目录
    if ($f.Name -eq "edges.arrow") {
        $dest = Join-Path $OutDir "normalized\$($f.Name)"
        New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null
    }
    if (Test-Path $dest) {
        $sz = [math]::Round((Get-Item $dest).Length/1MB, 0)
        if ($sz -ge ($f.SizeMB * 0.9)) {
            Write-Host "  [跳过] $($f.Name) 已存在 (${sz} MB)" -ForegroundColor DarkGray
            continue
        }
    }
    Write-Host "  [下载] $($f.Name) (~$($f.SizeMB) MB) ..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
    $sz = [math]::Round((Get-Item $dest).Length/1MB, 0)
    Write-Host "         完成 (${sz} MB)" -ForegroundColor Green
}
Write-Host "`n完成。脑图就绪，可运行训练脚本。" -ForegroundColor Cyan
