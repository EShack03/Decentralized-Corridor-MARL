# =====================================================================
# run_all_evals.ps1
# Runs the full canonical evaluation set for the MEng corridor-MARL
# project, one after another, logging each to its own file.
#
# Usage (from the repo root, venv active):
#     .\run_all_evals.ps1
#
# Each run writes:
#   * its own result .txt in the repo root (produced by the eval script)
#   * a full console log to  eval_logs\<run-name>.log
# A per-run banner is printed with start/finish time and exit status.
# If one run fails it is recorded and the batch CONTINUES, so a single
# crash never costs you the rest of the night. A final summary lists
# which runs succeeded and which failed.
# =====================================================================

$ErrorActionPreference = "Continue"   # don't abort the batch on one failure

# --- common settings shared by every run -----------------------------
$PY       = "python"
$SCRIPT   = "onpolicy/scripts/eval_cbf_stage2.py"
$COMMON   = @(
    "--env_name", "GraphMPE",
    "--algorithm_name", "rmappo",
    "--scenario_name", "three_phase_graph_sequential",
    "--dynamics_type", "air_taxi",
    "--world_size", "5",
    "--episode_length", "200",
    "--render_episodes", "50",
    "--n_rollout_threads", "8",
    "--cbf_barrier", "cbf_stage1_barrier.pt"
)

$ORIGINAL = "model_weights"
$MODEL_3B = "onpolicy/results/GraphMPE/working_three_phase_graph/rmappo/stage3b_full_v2/run1/models"
$MODEL_3A = "onpolicy/results/GraphMPE/three_phase_graph_updated/rmappo/stage3a_full/run2/models"

# --- the six runs, as (name, argument-array) pairs -------------------
# Density sweeps use the single operating point alpha=1.0, horizon=3
# (one baseline + one filtered cell per density) across N = 5,8,10,
# repeated over seeds 1,2,3. The design sweep (F) explores the full
# alpha x horizon x fallback grid at N=5 on a single seed.
$runs = @(
    @{ name = "A_original_static";   args = @("--model_dir", $ORIGINAL,  "--seeds", "1,2,3", "--cbf_neighbour_model", "static",            "--eval_tag", "original", "--cbf_density_sweep", "--cbf_density_Ns", "5,8,10", "--cbf_sweep_alphas", "1.0", "--cbf_sweep_horizons", "3") },
    @{ name = "B_original_cv";       args = @("--model_dir", $ORIGINAL,  "--seeds", "1,2,3", "--cbf_neighbour_model", "constant_velocity", "--eval_tag", "original", "--cbf_density_sweep", "--cbf_density_Ns", "5,8,10", "--cbf_sweep_alphas", "1.0", "--cbf_sweep_horizons", "3") },
    @{ name = "C_3B_static";         args = @("--model_dir", $MODEL_3B,  "--seeds", "1,2,3", "--cbf_neighbour_model", "static",            "--eval_tag", "3B",       "--cbf_density_sweep", "--cbf_density_Ns", "5,8,10", "--cbf_sweep_alphas", "1.0", "--cbf_sweep_horizons", "3") },
    @{ name = "D_3B_cv";             args = @("--model_dir", $MODEL_3B,  "--seeds", "1,2,3", "--cbf_neighbour_model", "constant_velocity", "--eval_tag", "3B",       "--cbf_density_sweep", "--cbf_density_Ns", "5,8,10", "--cbf_sweep_alphas", "1.0", "--cbf_sweep_horizons", "3") },
    @{ name = "E_3A_cv_N5";          args = @("--model_dir", $MODEL_3A,  "--seeds", "1,2,3", "--cbf_neighbour_model", "constant_velocity", "--eval_tag", "3A",       "--num_agents", "5", "--num_landmarks", "5", "--cbf_alpha", "1.0", "--cbf_horizon", "3") },
    @{ name = "F_3B_cv_designsweep_N5"; args = @("--model_dir", $MODEL_3B, "--seeds", "1", "--cbf_neighbour_model", "constant_velocity", "--eval_tag", "3B", "--num_agents", "5", "--num_landmarks", "5", "--cbf_sweep", "--cbf_sweep_alphas", "0.1,0.3,0.5,1.0", "--cbf_sweep_horizons", "1,3,5", "--cbf_sweep_fallbacks", "least_unsafe,unfiltered") }
)

# --- set up logging --------------------------------------------------
$logDir = "eval_logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir | Out-Null }

$batchStart = Get-Date
Write-Host ""
Write-Host ("#" * 70)
Write-Host "# BATCH START: $batchStart"
Write-Host "# $($runs.Count) runs -> logs in $logDir\"
Write-Host ("#" * 70)

$results = @()

foreach ($run in $runs) {
    $name   = $run.name
    $argv   = @($SCRIPT) + $COMMON + $run.args
    $logPath = Join-Path $logDir "$name.log"
    $start  = Get-Date

    Write-Host ""
    Write-Host ("=" * 70)
    Write-Host ">>> RUN: $name"
    Write-Host ">>> start : $start"
    Write-Host ">>> log   : $logPath"
    Write-Host ">>> cmd   : $PY $($argv -join ' ')"
    Write-Host ("=" * 70)

    # Run python, streaming output to BOTH the console and the log file.
    & $PY @argv 2>&1 | Tee-Object -FilePath $logPath
    $code = $LASTEXITCODE
    $end  = Get-Date
    $dur  = $end - $start

    $status = if ($code -eq 0) { "OK" } else { "FAILED (exit $code)" }
    Write-Host ""
    Write-Host ">>> $name  ->  $status   (elapsed $([int]$dur.TotalMinutes) min)"

    $results += [pscustomobject]@{
        Run    = $name
        Status = $status
        Minutes = [int]$dur.TotalMinutes
        Log    = $logPath
    }
}

# --- final summary ---------------------------------------------------
$batchEnd = Get-Date
Write-Host ""
Write-Host ("#" * 70)
Write-Host "# BATCH COMPLETE: $batchEnd"
Write-Host "# total elapsed: $([int]($batchEnd - $batchStart).TotalMinutes) min"
Write-Host ("#" * 70)
$results | Format-Table -AutoSize
$failed = $results | Where-Object { $_.Status -ne "OK" }
if ($failed) {
    Write-Host ""
    Write-Host "!! $($failed.Count) run(s) FAILED - inspect their logs:" -ForegroundColor Yellow
    $failed | ForEach-Object { Write-Host "   $($_.Run)  ->  $($_.Log)" -ForegroundColor Yellow }
} else {
    Write-Host ""
    Write-Host "All runs completed successfully." -ForegroundColor Green
}
