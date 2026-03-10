# AI Data Quality Agent Roadmap

## Current State

The repository currently includes:

- class-based `DataQualityAgent`
- CLI entry point
- deterministic data profiling and rule checks
- optional OpenRouter-backed diagnosis
- JSON schema validation
- pytest coverage for core paths

## Next Improvements

### Short Term

- Add column-level range rules in schema files
- Support Excel and Parquet inputs
- Export Markdown and HTML reports
- Add drift comparison against a baseline report
- Improve issue grouping so related findings are clustered

### Mid Term

- Provider abstraction for OpenAI, Anthropic, and local models
- Historical run storage for trend analysis
- Auto-remediation suggestions with generated SQL or pandas fixes
- Streamlit dashboard for report inspection
- GitHub Actions workflow for tests and linting

### Longer Term

- Integrate with Airflow or Prefect
- Add Great Expectations interoperability
- Support data contracts and richer schema rules
- Add lineage-aware root cause hints
- Add dataset-specific check packs for finance, marketing, and operations data

## Principles For Future Work

- Keep deterministic checks first-class
- Use LLMs for diagnosis and prioritization, not for basic arithmetic
- Prefer structured JSON outputs over prose-only reports
- Make the agent usable both as a demo and as a pipeline building block
