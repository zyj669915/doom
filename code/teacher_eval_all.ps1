$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = 'D:\fly\doomfly'
$py = 'D:\fly\doomfly\.venv-neural\Scripts\python.exe'
$code = 'D:\fly\experiment\code'
$phase3 = Join-Path $code 'outputs\doom\phase3'
Set-Location $code
$items = @(
  @{ n = 'teacher-v2-combat';   p = (Join-Path $code 'outputs\doom\teacher-v2-combat\teacher.pt') },
  @{ n = 'teacher-v2-corridor'; p = (Join-Path $code 'outputs\doom\teacher-v2-corridor\teacher.pt') },
  @{ n = 'teacher-v1';          p = (Join-Path $code 'outputs\doom\teacher-v1\teacher.pt') }
)
foreach ($it in $items) {
  Write-Output "=== eval $($it.n) @ $(Get-Date -Format HH:mm:ss) ==="
  & $py eval_v2_600.py --skip-student --teacher-path $it.p
  if ($LASTEXITCODE -ne 0) { throw "eval failed: $($it.n)" }
  Copy-Item (Join-Path $phase3 'evaluation.json') (Join-Path $phase3 "teacher_$($it.n).json") -Force
  Copy-Item (Join-Path $phase3 'summary.json')   (Join-Path $phase3 "summary_$($it.n).json")   -Force
}
Write-Output '=== ALL TEACHER EVALS DONE ==='
