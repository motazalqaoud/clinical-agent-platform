from clinagent.evaluation.runner import CaseResult, compute_metrics, gate


def _r(cat, **checks):
    return CaseResult(
        id="x", category=cat, passed=all(checks.values()), checks=checks, latency_ms=1.0
    )


def test_gate_fails_on_any_phi_leak():
    results = [_r("answerable", answered=True, cited_expected=True, no_phi_in_answer=False)]
    failures = gate(compute_metrics(results), {"phi_leaks": {"max": 0}})
    assert failures and "phi_leaks" in failures[0]


def test_gate_reports_unmeasured_metric():
    results = [_r("answerable", answered=True, cited_expected=True, no_phi_in_answer=True)]
    assert gate(compute_metrics(results), {"injection_quarantine_rate": {"min": 1.0}})


def test_gate_passes_when_within_thresholds():
    results = [_r("answerable", answered=True, cited_expected=True, no_phi_in_answer=True)]
    assert gate(compute_metrics(results), {"answer_rate": {"min": 1.0}}) == []
