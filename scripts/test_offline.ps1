$ErrorActionPreference = 'Stop'
$python = Join-Path (Get-Location) '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) { $python = 'python' }
$suites = @(
    'security_tests.phase1_identity_boundary',
    'security_tests.phase2_authorization_evaluation',
    'security_tests.phase3_answer_quality',
    'security_tests.phase4_reliability',
    'security_tests.phase5_source_inspection',
    'security_tests.issue_regressions',
    'security_tests.seller_balanced_retrieval',
    'security_tests.portfolio_answer_scope',
    'security_tests.api_liveness',
    'security_tests.react_api_contracts',
    'security_tests.evaluation_v11',
    'security_tests.targeted_evaluation_retry',
    'security_tests.jina_rotation',
    'security_tests.groq_rotation',
    'security_tests.remediation_v12',
    'security_tests.execution_errors',
    'security_tests.current_evaluation'
)
foreach ($suite in $suites) {
    & $python -B scripts/run_offline_suite.py $suite
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
