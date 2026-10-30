# Close every pre-report gap in one go. Run inside (bgl) from the repository root:
#   Set-ExecutionPolicy -Scope Process Bypass; .\scripts\finish_gaps.ps1
$ErrorActionPreference = "Continue"
if (-not $env:CONDA_PREFIX) { Write-Host "Run 'conda activate bgl' first." -ForegroundColor Red; exit 1 }
$py = Join-Path $env:CONDA_PREFIX "python.exe"
function Step($title, $block) { Write-Host "`n=== $title ===" -ForegroundColor Cyan; & $block; if ($LASTEXITCODE -ne 0) { Write-Host "  -> step reported an error (see above)" -ForegroundColor Yellow } }

Step "1/7 experiment log: apply reviewed 'learned' statements and unsuccessful outcomes" { & $py -m src.utils.apply_annotations }
Step "2/7 learning curves of the final model (three seeds)" { & $py -m src.models.plot_curves --runs sem_final_N5_s11 sem_final_N5_s22 sem_final_N5_s33 --out sem_final_N5 }
Step "3/7 error analysis by input type / data quality (validation)" { & $py -m src.models.error_buckets --run-prefix sem_final --N 5 }
Step "4/7 XAI case studies incl. 4th case (your interpretations are preserved in results\xai_cases_interpreted.md)" {
  if (Test-Path results\xai_cases.md) { Copy-Item results\xai_cases.md results\xai_cases_interpreted.md -Force }
  & $py -m src.xai.cases --run sem_final_N5_s11
}
Step "5/7 optimisation-study summary regenerated (points to docs\phase5_findings.md)" { & $py -m src.models.study_summary }
Step "6/7 test suite" { & $py -m pytest --basetemp=E:\pytest-temp -q }
Step "7/7 housekeeping" {
  if (Test-Path notebooks) { Remove-Item notebooks -Recurse -Force; Write-Host "  removed empty notebooks\" }
  foreach ($f in "results\figures\curves_sem_final_N5.png", "results\error_buckets.md", "results\xai_cases.md", "results\xai_cases_interpreted.md", "results\optimisation_study.md", "docs\phase5_findings.md", "docs\architecture_sem_final.md", "report\figures\architecture_sem_final.png", "report\draft_Q1_Q2.md") {
    if (Test-Path $f) { Write-Host ("  OK      {0}" -f $f) -ForegroundColor Green } else { Write-Host ("  MISSING {0}" -f $f) -ForegroundColor Red }
  }
}
Write-Host "`nNow: git add -A; git commit -m 'Pre-report gap fixes'; then .\scripts\make_submission.ps1" -ForegroundColor Green
Write-Host "Send me: results\error_buckets.md, the 'Case 4' block of results\xai_cases.md, and the first line printed by step 1."
