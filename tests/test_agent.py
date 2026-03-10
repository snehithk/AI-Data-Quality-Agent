from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_data_quality_agent import DataQualityAgent, create_sample_data


def test_sample_data_detects_core_issues():
    agent = DataQualityAgent()
    df = create_sample_data()

    issues = agent.run_quality_checks(
        df,
        semantic_context=agent.infer_dataset_semantics(df, agent.profile_dataset(df)),
        check_plan=agent.build_check_plan(agent.profile_dataset(df), agent.infer_dataset_semantics(df, agent.profile_dataset(df)), None),
    )
    issue_types = {issue.issue_type for issue in issues}

    assert "MISSING_VALUES" in issue_types
    assert "DUPLICATE_ROWS" in issue_types
    assert "INFINITE_VALUES" in issue_types
    assert "EMPTY_STRINGS" in issue_types
    assert "OUTLIERS" in issue_types
    assert "NEGATIVE_VALUES" in issue_types


def test_analysis_exposes_semantic_context_and_plan():
    agent = DataQualityAgent()
    analysis = agent.analyze(create_sample_data())

    assert analysis["semantic_context"]["dataset_type"] == "financial_timeseries"
    assert analysis["semantic_context"]["analysis_mode"] == "heuristic"
    assert analysis["check_plan"]["checks"]
    assert analysis["check_plan"]["column_rules"]["volume"]["non_negative"] is True


def test_expected_schema_flags_invalid_values():
    agent = DataQualityAgent()
    df = pd.DataFrame(
        {
            "status": ["active", "paused"],
            "amount": [10, 20],
        }
    )
    schema = {
        "status": {"dtype": "object", "nullable": False, "allowed_values": ["active", "inactive"]},
        "amount": {"dtype": "int64", "nullable": False},
        "owner": {"dtype": "object", "nullable": False},
    }

    analysis = agent.analyze(df, expected_schema=schema)
    issue_types = [issue["issue_type"] for issue in analysis["issues"]]

    assert "ENUM_VIOLATION" in issue_types
    assert "MISSING_COLUMN" in issue_types


def test_analysis_returns_structured_assessment_without_api_key():
    agent = DataQualityAgent()
    analysis = agent.analyze(create_sample_data())

    assert analysis["assessment"]["analysis_mode"] == "heuristic"
    assert isinstance(analysis["assessment"]["recommendations"], list)
    assert isinstance(analysis["assessment"]["suggested_fixes"], list)
    assert 0 <= analysis["quality_score"] <= 100
