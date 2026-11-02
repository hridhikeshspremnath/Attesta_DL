# Unattended M2b queue (normalized-bytes runs). Run from the project root in the activated .venv-dl shell:
#   Set-ExecutionPolicy -Scope Process Bypass; .\scripts\queue_m2b.ps1
# Resumable like queue_m2.ps1. Logs: logs\queue_norm\<name>.log, reports: reports\queue_norm\<name>.md,
# per-sample scores: reports\scores\q_<name>.parquet
New-Item -ItemType Directory -Force logs\queue_norm, reports\queue_norm | Out-Null
python -W ignore -m scripts.make_queue_norm | Out-Null
foreach ($name in Get-Content configs\queue_norm\order.txt) {
    if (Test-Path "reports\queue_norm\$name.md") { "$(Get-Date -f HH:mm:ss) skip $name (report exists)"; continue }
    "$(Get-Date -f HH:mm:ss) start $name"
    for ($try = 1; $try -le 3; $try++) {
        python -u -W ignore -m src.train.run_cnn --config "configs/queue_norm/$name.yaml" 2>&1 | Tee-Object -FilePath "logs\queue_norm\$name.log" -Append
        if ($LASTEXITCODE -eq 0) { break }
        "$(Get-Date -f HH:mm:ss) $name failed (try $try), retrying"; Start-Sleep 20
    }
    "$(Get-Date -f HH:mm:ss) end $name (exit $LASTEXITCODE)"
}
"$(Get-Date -f HH:mm:ss) QUEUE DONE"
