# Unattended M2 queue. Run from the project root in the activated .venv-dl shell:
#   .\scripts\queue_m2.ps1
# Resumable: rerun it and finished runs (early-stopped or complete) only redo their final eval; a failed run
# does not stop the queue. Per-run logs: logs\queue\<name>.log, reports: reports\queue\<name>.md
New-Item -ItemType Directory -Force logs\queue, reports\queue | Out-Null
python -W ignore -m scripts.make_queue | Out-Null
foreach ($name in Get-Content configs\queue\order.txt) {
    if (Test-Path "reports\queue\$name.md") { "$(Get-Date -f HH:mm:ss) skip $name (report exists)"; continue }
    "$(Get-Date -f HH:mm:ss) start $name"
    for ($try = 1; $try -le 3; $try++) {  # a crash (e.g. CUDA OOM) resumes from the last epoch checkpoint
        python -u -W ignore -m src.train.run_cnn --config "configs/queue/$name.yaml" 2>&1 | Tee-Object -FilePath "logs\queue\$name.log" -Append
        if ($LASTEXITCODE -eq 0) { break }
        "$(Get-Date -f HH:mm:ss) $name failed (try $try), retrying"; Start-Sleep 20
    }
    "$(Get-Date -f HH:mm:ss) end $name (exit $LASTEXITCODE)"
}
"$(Get-Date -f HH:mm:ss) QUEUE DONE"
