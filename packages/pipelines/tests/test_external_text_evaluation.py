from faulttrace_pipelines.external_text_evaluation import (
    aggregate_nli_predictions,
    aggregate_support_status,
    answer_exact_match,
    answer_token_f1,
    certificate_metrics,
    lexical_grounding_score,
    select_certificate_threshold,
)


def test_answer_metrics_use_standard_normalization():
    assert answer_exact_match("The Paris!", "Paris") == 1.0
    assert answer_token_f1("Paris is in France", "Paris") == 0.4


def test_support_aggregation_handles_conflict_and_abstention():
    assert aggregate_support_status([]) == "insufficient_evidence"
    assert aggregate_support_status(["supported"]) == "supported"
    assert aggregate_support_status(["supported", "unsupported"]) == "conflicting"


def test_nli_aggregation_uses_strongest_non_neutral_document():
    assert aggregate_nli_predictions([("neutral", 0.9)]) == "insufficient_evidence"
    assert aggregate_nli_predictions(
        [("entailment", 0.7), ("contradiction", 0.8), ("neutral", 0.95)]
    ) == "unsupported"


def test_lexical_grounding_enforces_numeric_fidelity():
    grounded = lexical_grounding_score(
        ["The trial enrolled 100 patients."],
        ["A total of 100 patients were enrolled in the trial."],
    )
    invented_number = lexical_grounding_score(
        ["The trial enrolled 900 patients."],
        ["A total of 100 patients were enrolled in the trial."],
    )

    assert grounded.numeric_fidelity is True
    assert grounded.score > 0
    assert invented_number.numeric_fidelity is False
    assert invented_number.score == 0.0


def test_certificate_threshold_is_calibrated_without_test_labels():
    calibration = select_certificate_threshold(
        scores=[0.9, 0.8, 0.7, 0.1],
        labels=[True, True, False, False],
        target_fcr=0.0,
    )

    assert calibration["threshold"] == 0.8
    assert calibration["certified"] == 2
    metrics = certificate_metrics([0.85, 0.2], [True, False], 0.8)
    assert metrics["coverage"] == 0.5
    assert metrics["false_certification_rate"] == 0.0
