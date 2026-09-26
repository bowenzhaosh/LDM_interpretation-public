from pfn_dag_verify.validation import validate_bootstrap_coverage


def test_bootstrap_coverage_validator_schema_smoke():
    result = validate_bootstrap_coverage(datasets_per_slope=2, bootstraps=20, groups=16)
    assert set(result["results"]) == {"0.8", "1.0", "1.2"}
    assert result["datasets_per_slope"] == 2
    assert result["bootstraps_per_dataset"] == 20
