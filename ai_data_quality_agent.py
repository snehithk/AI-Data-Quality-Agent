#!/usr/bin/env python3
"""
AI Data Quality Agent

Agent-style data quality analysis for CSV or pandas DataFrames. The workflow is:
1. Profile the data
2. Infer dataset semantics and a prioritized check plan
3. Execute deterministic checks
4. Use an LLM to explain root causes and remediation
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "nvidia/nemotron-3-nano-30b-a3b:free"
DEFAULT_OUTPUT_PATH = "quality_report.json"
DEFAULT_TIMEOUT_SECONDS = 30

SEVERITY_RANK = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}


@dataclass
class DataIssue:
    issue_type: str
    column: str
    severity: str
    count: int
    description: str
    evidence: dict[str, Any]


def create_sample_data() -> pd.DataFrame:
    """Create a sample stock dataset with intentional data quality issues."""
    np.random.seed(42)
    date_range = pd.date_range(start="2024-01-01", periods=120, freq="D")

    data = {
        "date": date_range,
        "stock_symbol": ["AAPL"] * 120,
        "open_price": np.random.uniform(150, 200, 120),
        "close_price": np.random.uniform(150, 200, 120),
        "volume": np.random.randint(1_000_000, 10_000_000, 120),
        "market": ["NASDAQ"] * 120,
    }

    data["close_price"][10] = None
    data["volume"][20] = -100
    data["open_price"][30] = np.inf
    data["stock_symbol"][40] = ""
    data["close_price"][55] = 4999.99
    data["market"][60] = "NASDQ"

    df = pd.DataFrame(data)
    duplicate_rows = df.iloc[[5, 6]].copy()
    return pd.concat([df, duplicate_rows], ignore_index=True)


class DataQualityAgent:
    """Agent-style orchestrator for profiling, planning, checks, and AI reasoning."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.model = model
        self.timeout = timeout

    def analyze(
        self, df: pd.DataFrame, expected_schema: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        profile = self.profile_dataset(df)
        semantic_context = self.infer_dataset_semantics(df, profile)
        check_plan = self.build_check_plan(profile, semantic_context, expected_schema)
        issues = self.run_quality_checks(
            df,
            expected_schema=expected_schema,
            semantic_context=semantic_context,
            check_plan=check_plan,
        )
        quality_score = self.compute_quality_score(profile, issues)
        assessment = self.generate_ai_assessment(
            profile=profile,
            semantic_context=semantic_context,
            check_plan=check_plan,
            issues=issues,
            quality_score=quality_score,
        )

        return {
            "generated_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "agent": {
                "name": "AI Data Quality Agent",
                "model": self.model,
                "used_llm": any(
                    mode == "llm"
                    for mode in (
                        semantic_context.get("analysis_mode"),
                        assessment.get("analysis_mode"),
                    )
                ),
            },
            "dataset_profile": profile,
            "semantic_context": semantic_context,
            "check_plan": check_plan,
            "issues": [asdict(issue) for issue in issues],
            "quality_score": quality_score,
            "severity": assessment["severity"],
            "assessment": assessment,
        }

    def profile_dataset(self, df: pd.DataFrame) -> dict[str, Any]:
        duplicate_rows = int(df.duplicated().sum())
        missing_by_column = {
            col: int(count) for col, count in df.isnull().sum().items() if count > 0
        }

        numeric_summary: dict[str, Any] = {}
        for column in df.select_dtypes(include=[np.number]).columns:
            series = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if series.notna().sum() == 0:
                continue
            numeric_summary[column] = {
                "mean": round(float(series.mean()), 4),
                "median": round(float(series.median()), 4),
                "std": round(float(series.std(ddof=0)), 4),
                "min": round(float(series.min()), 4),
                "max": round(float(series.max()), 4),
            }

        categorical_summary: dict[str, Any] = {}
        for column in df.select_dtypes(include=["object", "string", "category"]).columns:
            top_values = df[column].fillna("<NULL>").astype(str).value_counts().head(5)
            categorical_summary[column] = {
                "unique_values": int(df[column].nunique(dropna=True)),
                "top_values": top_values.to_dict(),
            }

        sample_rows = df.head(5).replace([np.inf, -np.inf], None).astype(object).where(pd.notnull(df.head(5)), None)

        return {
            "row_count": int(len(df)),
            "column_count": int(len(df.columns)),
            "columns": list(df.columns),
            "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
            "missing_by_column": missing_by_column,
            "duplicate_rows": duplicate_rows,
            "numeric_summary": numeric_summary,
            "categorical_summary": categorical_summary,
            "sample_rows": sample_rows.to_dict(orient="records"),
        }

    def infer_dataset_semantics(
        self, df: pd.DataFrame, profile: dict[str, Any]
    ) -> dict[str, Any]:
        if not self.api_key:
            return self._heuristic_semantic_context(df, profile)

        prompt = self._build_semantic_inference_prompt(profile)
        try:
            raw_response = self._call_openrouter(prompt)
            parsed = self._parse_llm_json(
                raw_response,
                required_keys={
                    "dataset_purpose",
                    "dataset_type",
                    "business_rules",
                    "column_semantics",
                    "priority_checks",
                },
            )
            if parsed:
                parsed = self._normalize_assessment_payload(parsed)
                parsed["analysis_mode"] = "llm"
                return parsed
        except Exception as exc:
            fallback = self._heuristic_semantic_context(df, profile)
            fallback["analysis_mode"] = "fallback_after_llm_error"
            fallback["llm_error"] = str(exc)
            return fallback

        fallback = self._heuristic_semantic_context(df, profile)
        fallback["analysis_mode"] = "fallback_after_invalid_llm_response"
        return fallback

    def build_check_plan(
        self,
        profile: dict[str, Any],
        semantic_context: dict[str, Any],
        expected_schema: dict[str, Any] | None,
    ) -> dict[str, Any]:
        column_semantics = semantic_context.get("column_semantics", {})
        business_rules = semantic_context.get("business_rules", [])

        checks: list[dict[str, Any]] = [
            {"name": "missing_values", "priority": "HIGH", "reason": "Completeness is a baseline quality gate."},
            {"name": "duplicate_rows", "priority": "HIGH", "reason": "Duplicate records corrupt downstream metrics."},
            {"name": "infinite_values", "priority": "HIGH", "reason": "Infinite values break aggregations and models."},
            {"name": "blank_strings", "priority": "MEDIUM", "reason": "Blank categorical fields hide missingness."},
            {"name": "outliers", "priority": "MEDIUM", "reason": "Extreme values can indicate bad ingestion or business anomalies."},
        ]

        planned_columns: dict[str, dict[str, Any]] = {}
        for column in profile["columns"]:
            semantics = column_semantics.get(column, {})
            if semantics.get("should_be_non_negative"):
                planned_columns.setdefault(column, {})["non_negative"] = True
            if semantics.get("allowed_values"):
                planned_columns.setdefault(column, {})["allowed_values"] = semantics["allowed_values"]
            if semantics.get("is_identifier"):
                planned_columns.setdefault(column, {})["identifier"] = True

        if expected_schema:
            checks.append(
                {"name": "schema_validation", "priority": "HIGH", "reason": "An expected schema was supplied by the caller."}
            )

        for suggested in semantic_context.get("priority_checks", []):
            checks.append(
                {
                    "name": suggested.get("name", "ai_suggested_check"),
                    "priority": suggested.get("priority", "MEDIUM"),
                    "reason": suggested.get("reason", "AI-suggested dataset-specific check."),
                }
            )

        return {
            "analysis_mode": semantic_context.get("analysis_mode", "heuristic"),
            "dataset_purpose": semantic_context.get("dataset_purpose", "Unknown"),
            "business_rules": business_rules,
            "checks": self._dedupe_checks(checks),
            "column_rules": planned_columns,
        }

    def run_quality_checks(
        self,
        df: pd.DataFrame,
        expected_schema: dict[str, Any] | None = None,
        semantic_context: dict[str, Any] | None = None,
        check_plan: dict[str, Any] | None = None,
    ) -> list[DataIssue]:
        semantic_context = semantic_context or {}
        check_plan = check_plan or {}
        issues: list[DataIssue] = []

        missing_counts = df.isnull().sum()
        for column, count in missing_counts.items():
            if count:
                issues.append(
                    DataIssue(
                        issue_type="MISSING_VALUES",
                        column=column,
                        severity="HIGH",
                        count=int(count),
                        description=f"Column '{column}' contains {int(count)} missing value(s).",
                        evidence={"missing_ratio": round(float(count / len(df)), 4)},
                    )
                )

        duplicate_count = int(df.duplicated().sum())
        if duplicate_count:
            issues.append(
                DataIssue(
                    issue_type="DUPLICATE_ROWS",
                    column="__row__",
                    severity="HIGH",
                    count=duplicate_count,
                    description=f"Dataset contains {duplicate_count} duplicated row(s).",
                    evidence={"duplicate_ratio": round(float(duplicate_count / len(df)), 4)},
                )
            )

        column_rules = check_plan.get("column_rules", {})
        column_semantics = semantic_context.get("column_semantics", {})

        for column in df.select_dtypes(include=[np.number]).columns:
            series = pd.to_numeric(df[column], errors="coerce")
            infinite_count = int(np.isinf(series).sum())
            if infinite_count:
                issues.append(
                    DataIssue(
                        issue_type="INFINITE_VALUES",
                        column=column,
                        severity="HIGH",
                        count=infinite_count,
                        description=f"Column '{column}' contains {infinite_count} infinite value(s).",
                        evidence={},
                    )
                )

            clean_series = series.replace([np.inf, -np.inf], np.nan).dropna()
            if clean_series.empty:
                continue

            if column_rules.get(column, {}).get("non_negative"):
                negative_count = int((clean_series < 0).sum())
                if negative_count:
                    issues.append(
                        DataIssue(
                            issue_type="NEGATIVE_VALUES",
                            column=column,
                            severity="MEDIUM",
                            count=negative_count,
                            description=f"Column '{column}' has {negative_count} unexpected negative value(s).",
                            evidence={"min_value": round(float(clean_series.min()), 4)},
                        )
                    )

            outlier_count, bounds = self._count_iqr_outliers(clean_series)
            if outlier_count:
                severity = "HIGH" if outlier_count / len(clean_series) > 0.1 else "MEDIUM"
                issues.append(
                    DataIssue(
                        issue_type="OUTLIERS",
                        column=column,
                        severity=severity,
                        count=outlier_count,
                        description=f"Column '{column}' has {outlier_count} value(s) outside the IQR bounds.",
                        evidence={"lower_bound": round(bounds[0], 4), "upper_bound": round(bounds[1], 4)},
                    )
                )

        for column in df.select_dtypes(include=["object", "string", "category"]).columns:
            series = df[column].fillna("").astype(str)
            empty_count = int(series.str.strip().eq("").sum())
            if empty_count:
                issues.append(
                    DataIssue(
                        issue_type="EMPTY_STRINGS",
                        column=column,
                        severity="MEDIUM",
                        count=empty_count,
                        description=f"Column '{column}' contains {empty_count} blank string value(s).",
                        evidence={},
                    )
                )

            allowed_values = column_rules.get(column, {}).get("allowed_values")
            if allowed_values:
                invalid_mask = ~df[column].isin(allowed_values)
                invalid_count = int(invalid_mask.sum())
                if invalid_count:
                    sample_values = df.loc[invalid_mask, column].dropna().astype(str).unique().tolist()[:5]
                    issues.append(
                        DataIssue(
                            issue_type="ENUM_VIOLATION",
                            column=column,
                            severity="MEDIUM",
                            count=invalid_count,
                            description=f"Column '{column}' contains values outside the allowed set.",
                            evidence={"sample_invalid_values": sample_values},
                        )
                    )

            if column_semantics.get(column, {}).get("is_identifier"):
                duplicate_ids = int(df[column].dropna().duplicated().sum())
                if duplicate_ids:
                    issues.append(
                        DataIssue(
                            issue_type="IDENTIFIER_DUPLICATES",
                            column=column,
                            severity="HIGH",
                            count=duplicate_ids,
                            description=f"Identifier-like column '{column}' contains duplicate values.",
                            evidence={},
                        )
                    )

        if expected_schema:
            issues.extend(self._validate_expected_schema(df, expected_schema))

        issues = self._dedupe_issues(issues)
        issues.sort(key=lambda item: (SEVERITY_RANK.get(item.severity, 4), item.issue_type, item.column))
        return issues

    def compute_quality_score(self, profile: dict[str, Any], issues: list[DataIssue]) -> int:
        score = 100.0
        row_count = max(profile["row_count"], 1)
        severity_penalties = {"HIGH": 8.0, "MEDIUM": 4.0, "LOW": 2.0}

        for issue in issues:
            base_penalty = severity_penalties.get(issue.severity, 2.0)
            ratio_penalty = min((issue.count / row_count) * 20.0, 10.0)
            score -= base_penalty + ratio_penalty

        return max(0, min(100, int(round(score))))

    def generate_ai_assessment(
        self,
        profile: dict[str, Any],
        semantic_context: dict[str, Any],
        check_plan: dict[str, Any],
        issues: list[DataIssue],
        quality_score: int,
    ) -> dict[str, Any]:
        if not self.api_key:
            return self._heuristic_assessment(profile, semantic_context, check_plan, issues, quality_score)

        prompt = self._build_assessment_prompt(profile, semantic_context, check_plan, issues, quality_score)
        try:
            raw_response = self._call_openrouter(prompt)
            parsed = self._parse_llm_json(
                raw_response,
                required_keys={
                    "severity",
                    "executive_summary",
                    "key_issues",
                    "root_causes",
                    "recommendations",
                    "next_checks",
                    "suggested_fixes",
                },
            )
            if parsed:
                parsed["analysis_mode"] = "llm"
                return parsed
        except Exception as exc:
            fallback = self._heuristic_assessment(profile, semantic_context, check_plan, issues, quality_score)
            fallback["analysis_mode"] = "fallback_after_llm_error"
            fallback["llm_error"] = str(exc)
            return fallback

        fallback = self._heuristic_assessment(profile, semantic_context, check_plan, issues, quality_score)
        fallback["analysis_mode"] = "fallback_after_invalid_llm_response"
        return fallback

    def render_text_report(self, analysis: dict[str, Any]) -> str:
        profile = analysis["dataset_profile"]
        semantics = analysis["semantic_context"]
        plan = analysis["check_plan"]
        assessment = analysis["assessment"]

        lines = [
            "=" * 80,
            "AI DATA QUALITY AGENT",
            "=" * 80,
            f"Generated: {analysis['generated_at']}",
            f"Rows: {profile['row_count']} | Columns: {profile['column_count']}",
            f"Quality Score: {analysis['quality_score']} ({analysis['severity']})",
            f"Semantic Inference Mode: {semantics['analysis_mode']}",
            f"Assessment Mode: {assessment['analysis_mode']}",
            f"Dataset Type: {semantics.get('dataset_type', 'Unknown')}",
            f"Dataset Purpose: {semantics.get('dataset_purpose', 'Unknown')}",
            "",
            "Planned Checks:",
        ]

        for check in plan["checks"][:6]:
            lines.append(f"- [{check['priority']}] {check['name']}: {check['reason']}")

        lines.extend(["", "Key Issues:"])
        for issue in analysis["issues"][:8]:
            lines.append(
                f"- [{issue['severity']}] {issue['issue_type']} in {issue['column']}: {issue['description']}"
            )

        lines.extend(["", "Executive Summary:", assessment["executive_summary"], "", "Recommendations:"])
        for item in assessment["recommendations"]:
            lines.append(f"- {item}")

        if assessment.get("suggested_fixes"):
            lines.extend(["", "Suggested Fixes:"])
            for item in assessment["suggested_fixes"][:4]:
                lines.append(f"- {item}")

        return "\n".join(lines)

    def _build_semantic_inference_prompt(self, profile: dict[str, Any]) -> str:
        payload = {
            "columns": profile["columns"],
            "dtypes": profile["dtypes"],
            "numeric_summary": profile["numeric_summary"],
            "categorical_summary": profile["categorical_summary"],
            "sample_rows": profile["sample_rows"],
        }
        return (
            "You are an AI data quality agent. Infer what this dataset likely represents and "
            "which quality rules matter most. Respond with valid JSON only.\n\n"
            "Required JSON keys: dataset_purpose, dataset_type, business_rules, column_semantics, priority_checks.\n"
            "Rules:\n"
            "- business_rules must be an array of short strings\n"
            "- column_semantics must be an object keyed by column name\n"
            "- each column semantic object may include semantic_type, should_be_non_negative, is_identifier, allowed_values\n"
            "- priority_checks must be an array of objects with keys name, priority, reason\n\n"
            f"Dataset payload:\n{json.dumps(payload, indent=2, default=str)}"
        )

    def _build_assessment_prompt(
        self,
        profile: dict[str, Any],
        semantic_context: dict[str, Any],
        check_plan: dict[str, Any],
        issues: list[DataIssue],
        quality_score: int,
    ) -> str:
        payload = {
            "quality_score": quality_score,
            "profile": profile,
            "semantic_context": semantic_context,
            "check_plan": check_plan,
            "issues": [asdict(issue) for issue in issues],
        }
        return (
            "You are an AI data quality agent reviewing the output of a profiling and validation run. "
            "Respond with valid JSON only.\n\n"
            "Required JSON keys: severity, executive_summary, key_issues, root_causes, "
            "recommendations, next_checks, suggested_fixes.\n"
            "Rules:\n"
            "- severity must be one of LOW, MEDIUM, HIGH, CRITICAL\n"
            "- recommendations and suggested_fixes must be concrete engineering actions\n"
            "- suggested_fixes can include SQL, pandas, or ETL-oriented fix ideas in plain text\n\n"
            f"Analysis payload:\n{json.dumps(payload, indent=2, default=str)}"
        )

    def _call_openrouter(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }

        response = requests.post(
            OPENROUTER_API_URL,
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def _parse_llm_json(self, raw_response: str, required_keys: set[str]) -> dict[str, Any] | None:
        try:
            parsed = json.loads(raw_response)
        except json.JSONDecodeError:
            return None
        if not required_keys.issubset(parsed.keys()):
            return None
        return parsed

    def _heuristic_semantic_context(self, df: pd.DataFrame, profile: dict[str, Any]) -> dict[str, Any]:
        column_semantics: dict[str, Any] = {}
        business_rules: list[str] = []

        for column in profile["columns"]:
            lowered = column.lower()
            semantic = {"semantic_type": "generic_field"}

            if "date" in lowered or pd.api.types.is_datetime64_any_dtype(df[column]):
                semantic["semantic_type"] = "event_date"
            elif "symbol" in lowered:
                semantic["semantic_type"] = "ticker_symbol"
                semantic["allowed_values"] = [value for value in df[column].dropna().astype(str).unique().tolist()[:5] if value]
            elif "market" in lowered:
                semantic["semantic_type"] = "exchange_code"
                semantic["allowed_values"] = ["NASDAQ", "NYSE", "AMEX"]
            elif any(token in lowered for token in ("price", "amount", "volume", "count", "qty", "total")):
                semantic["semantic_type"] = "metric"
                semantic["should_be_non_negative"] = True
            elif lowered.endswith("_id") or lowered == "id":
                semantic["semantic_type"] = "identifier"
                semantic["is_identifier"] = True

            column_semantics[column] = semantic

        if "volume" in df.columns:
            business_rules.append("Trading or transaction volume should not be negative.")
        if "stock_symbol" in df.columns:
            business_rules.append("Ticker symbols should be non-blank valid exchange-listed codes.")
        if "date" in df.columns:
            business_rules.append("Time series records should preserve usable date values.")

        return {
            "analysis_mode": "heuristic",
            "dataset_purpose": "Time-series market data quality monitoring",
            "dataset_type": "financial_timeseries",
            "business_rules": business_rules,
            "column_semantics": column_semantics,
            "priority_checks": [
                {"name": "non_negative_metrics", "priority": "HIGH", "reason": "Metrics like price and volume should not be negative."},
                {"name": "categorical_domain_validation", "priority": "MEDIUM", "reason": "Exchange and ticker codes should stay within expected domains."},
            ],
        }

    def _heuristic_assessment(
        self,
        profile: dict[str, Any],
        semantic_context: dict[str, Any],
        check_plan: dict[str, Any],
        issues: list[DataIssue],
        quality_score: int,
    ) -> dict[str, Any]:
        severity = self._score_to_severity(quality_score)
        key_issues = [issue.description for issue in issues[:5]] or ["No major issues detected."]

        root_causes = []
        if profile["duplicate_rows"]:
            root_causes.append("Duplicate ingestion or missing record-level uniqueness controls.")
        if profile["missing_by_column"]:
            root_causes.append("Upstream validation is not enforcing completeness rules before load.")
        if any(issue.issue_type == "ENUM_VIOLATION" for issue in issues):
            root_causes.append("Reference domain validation is missing for categorical fields.")
        if any(issue.issue_type == "NEGATIVE_VALUES" for issue in issues):
            root_causes.append("Metric columns are not protected by range checks at ingest time.")
        if not root_causes:
            root_causes.append("Current rule set did not identify a dominant failure mode.")

        recommendations = [
            "Run the AI planning stage against every new dataset to infer dataset-specific quality rules before checks execute.",
            "Enforce schema and nullability checks before data lands in downstream tables.",
            "Persist quality reports per run so the agent can compare drift and regressions over time.",
        ]

        next_checks = [
            "Add historical baselines so the agent can flag distribution drift instead of only row-level issues.",
            "Generate dataset-specific fix scripts from the agent output and require review before execution.",
            "Integrate the agent with batch orchestration so failed quality thresholds can block promotion.",
        ]

        suggested_fixes = [
            "pandas: df = df.drop_duplicates()",
            "pandas: df['close_price'] = df['close_price'].fillna(df['close_price'].median())",
            "pandas: df = df[df['volume'] >= 0]",
            "ETL: validate stock_symbol and market against reference dimensions before load.",
        ]

        return {
            "analysis_mode": "heuristic",
            "severity": severity,
            "executive_summary": (
                f"The agent classified this dataset as {semantic_context.get('dataset_type', 'generic tabular data')} "
                f"used for {semantic_context.get('dataset_purpose', 'data quality monitoring')}. "
                f"After profiling {profile['row_count']} rows and executing {len(check_plan.get('checks', []))} planned checks, "
                f"it found {len(issues)} issue(s) and assigned a quality score of {quality_score}/100."
            ),
            "key_issues": key_issues,
            "root_causes": root_causes,
            "recommendations": recommendations,
            "next_checks": next_checks,
            "suggested_fixes": suggested_fixes,
        }

    def _validate_expected_schema(self, df: pd.DataFrame, expected_schema: dict[str, Any]) -> list[DataIssue]:
        issues: list[DataIssue] = []
        for column, rules in expected_schema.items():
            if column not in df.columns:
                issues.append(
                    DataIssue(
                        issue_type="MISSING_COLUMN",
                        column=column,
                        severity="HIGH",
                        count=1,
                        description=f"Expected column '{column}' is missing from the dataset.",
                        evidence={"expected_rules": rules},
                    )
                )
                continue

            expected_dtype = rules.get("dtype")
            if expected_dtype:
                actual_dtype = str(df[column].dtype)
                if not self._dtype_matches(actual_dtype, expected_dtype):
                    issues.append(
                        DataIssue(
                            issue_type="SCHEMA_MISMATCH",
                            column=column,
                            severity="HIGH",
                            count=1,
                            description=f"Column '{column}' has dtype '{actual_dtype}' but expected '{expected_dtype}'.",
                            evidence={"actual_dtype": actual_dtype, "expected_dtype": expected_dtype},
                        )
                    )

            if rules.get("nullable") is False:
                missing_count = int(df[column].isnull().sum())
                if missing_count:
                    issues.append(
                        DataIssue(
                            issue_type="NON_NULL_VIOLATION",
                            column=column,
                            severity="HIGH",
                            count=missing_count,
                            description=f"Column '{column}' is required but contains nulls.",
                            evidence={},
                        )
                    )
                if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
                    blank_count = int(df[column].fillna("").astype(str).str.strip().eq("").sum())
                    if blank_count:
                        issues.append(
                            DataIssue(
                                issue_type="NON_NULL_VIOLATION",
                                column=column,
                                severity="HIGH",
                                count=blank_count,
                                description=f"Column '{column}' is required but contains blank values.",
                                evidence={},
                            )
                        )

            allowed_values = rules.get("allowed_values")
            if allowed_values:
                invalid_mask = ~df[column].isin(allowed_values)
                invalid_count = int(invalid_mask.sum())
                if invalid_count:
                    sample_values = df.loc[invalid_mask, column].dropna().astype(str).unique().tolist()[:5]
                    issues.append(
                        DataIssue(
                            issue_type="ENUM_VIOLATION",
                            column=column,
                            severity="MEDIUM",
                            count=invalid_count,
                            description=f"Column '{column}' contains values outside the allowed set.",
                            evidence={"sample_invalid_values": sample_values},
                        )
                    )
        return issues

    def _count_iqr_outliers(self, series: pd.Series) -> tuple[int, tuple[float, float]]:
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        if iqr == 0:
            return 0, (q1, q3)
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return int(((series < lower_bound) | (series > upper_bound)).sum()), (lower_bound, upper_bound)

    def _score_to_severity(self, score: int) -> str:
        if score >= 90:
            return "LOW"
        if score >= 75:
            return "MEDIUM"
        if score >= 50:
            return "HIGH"
        return "CRITICAL"

    def _dtype_matches(self, actual_dtype: str, expected_dtype: str) -> bool:
        actual = actual_dtype.lower()
        expected = expected_dtype.lower()
        aliases = {
            "object": ("object", "string", "str"),
            "str": ("object", "string", "str"),
            "string": ("object", "string", "str"),
            "int": ("int", "int64", "int32"),
            "float": ("float", "float64", "float32"),
            "datetime64": ("datetime64", "datetime64[ns]"),
        }
        accepted = aliases.get(expected, (expected,))
        return any(token in actual for token in accepted)

    def _normalize_assessment_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        for key in ("key_issues", "root_causes", "recommendations", "next_checks", "suggested_fixes"):
            value = payload.get(key)
            if isinstance(value, str):
                payload[key] = [line.strip("- ").strip() for line in value.splitlines() if line.strip()]
            elif not isinstance(value, list):
                payload[key] = [str(value)] if value is not None else []
        return payload

    def _dedupe_checks(self, checks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        seen: set[str] = set()
        deduped: list[dict[str, Any]] = []
        for check in checks:
            name = check["name"]
            if name in seen:
                continue
            seen.add(name)
            deduped.append(check)
        deduped.sort(key=lambda item: (SEVERITY_RANK.get(item["priority"], 4), item["name"]))
        return deduped

    def _dedupe_issues(self, issues: list[DataIssue]) -> list[DataIssue]:
        deduped: list[DataIssue] = []
        seen: set[tuple[str, str, str]] = set()
        for issue in issues:
            key = (issue.issue_type, issue.column, issue.description)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(issue)
        return deduped


def load_expected_schema(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_dataframe(input_path: str | None, sample_data: bool) -> pd.DataFrame:
    if sample_data or not input_path:
        return create_sample_data()
    return pd.read_csv(input_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agent-style AI data quality analysis")
    parser.add_argument("--input", help="Path to input CSV file")
    parser.add_argument("--expected-schema", help="Path to JSON schema rules used for validation")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help=f"Output JSON path (default: {DEFAULT_OUTPUT_PATH})")
    parser.add_argument("--sample-data", action="store_true", help="Use the built-in sample dataset with intentional quality issues")
    parser.add_argument("--api-key", help="OpenRouter API key. Falls back to OPENROUTER_API_KEY env var.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"OpenRouter model name (default: {DEFAULT_MODEL})")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_dataframe(args.input, sample_data=args.sample_data)
    expected_schema = load_expected_schema(args.expected_schema)

    agent = DataQualityAgent(api_key=args.api_key, model=args.model)
    analysis = agent.analyze(df, expected_schema=expected_schema)

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(analysis, handle, indent=2, default=str)

    print(agent.render_text_report(analysis))
    print("")
    print(f"Saved JSON report to {output_path.resolve()}")


if __name__ == "__main__":
    main()
