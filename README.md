# AI Data Quality Agent

Python CLI for profiling tabular data, running quality checks, and generating a structured diagnosis with an LLM.

## Scope

The current implementation does three things:

1. profiles a CSV or pandas DataFrame
2. runs deterministic quality checks
3. uses an LLM to infer dataset semantics and summarize findings

If no API key is configured, the tool falls back to heuristic planning and assessment.

## Checks

Implemented checks:

- missing values
- duplicate rows
- infinite numeric values
- blank strings
- non-negative checks for metric columns
- IQR outlier detection
- schema validation from JSON
- allowed-value validation for categorical columns

## Output

The tool prints a text report and writes a JSON report containing:

- dataset profile
- semantic context
- check plan
- detected issues
- quality score
- severity
- assessment

## Files

```text
.
├── ai_data_quality_agent.py
├── examples/
│   └── expected_schema.json
├── tests/
│   └── test_agent.py
├── README.md
├── ROADMAP.md
└── requirements.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run the built-in sample dataset:

```bash
python3 ai_data_quality_agent.py --sample-data
```

Run against a CSV:

```bash
python3 ai_data_quality_agent.py --input your_data.csv --output report.json
```

Run with schema validation:

```bash
python3 ai_data_quality_agent.py \
  --input your_data.csv \
  --expected-schema examples/expected_schema.json \
  --output report.json
```

Run with OpenRouter:

```bash
export OPENROUTER_API_KEY="your_key_here"
python3 ai_data_quality_agent.py --sample-data
```

## Report Shape

```json
{
  "semantic_context": {
    "analysis_mode": "llm",
    "dataset_type": "financial_time_series",
    "dataset_purpose": "Track daily price and volume movements of listed stocks for market analysis."
  },
  "check_plan": {
    "checks": [
      {
        "name": "missing_values",
        "priority": "HIGH",
        "reason": "Completeness is a baseline quality gate."
      }
    ]
  },
  "quality_score": 20,
  "severity": "CRITICAL",
  "assessment": {
    "analysis_mode": "llm",
    "executive_summary": "The dataset exhibits multiple critical quality violations.",
    "recommendations": [
      "Add pre-load validation to reject invalid rows."
    ],
    "suggested_fixes": [
      "pandas: df.drop_duplicates(...)"
    ]
  }
}
```

## Tests

```bash
pytest
```

Current test coverage includes:

- issue detection on sample data
- semantic inference and check-plan generation
- schema validation behavior
- fallback behavior without an API key

## Notes

- The current implementation is pandas-based and suited to small or medium local datasets.
- For larger production datasets, the useful pattern is to compute metrics outside the LLM and send only summaries, failed checks, and samples to the model.
- The LLM output is advisory. Deterministic checks should remain the source of truth for enforcement.

## License

MIT
