# Score the 9 already-trained M2 runs (no training): writes reports\scores\<ckpt name>.parquet. GPU inference, ~1 min each.
#   Set-ExecutionPolicy -Scope Process Bypass; .\scripts\rescore_m2.ps1
New-Item -ItemType Directory -Force logs | Out-Null
$cfgs = @("configs/m2.yaml") + (Get-Content configs\queue\order.txt | ForEach-Object { "configs/queue/$_.yaml" })
foreach ($c in $cfgs) {
    "$(Get-Date -f HH:mm:ss) score $c"
    python -u -W ignore -m src.train.run_cnn --config $c --score-only 2>&1 | Select-Object -Last 1
}
"$(Get-Date -f HH:mm:ss) DONE"
