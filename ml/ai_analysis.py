"""
Claude-powered churn indicator analysis for Sparkify dataset.

Loads Sparkify data, extracts features, and calls Claude to identify
the strongest leading indicators of churn based on aggregate statistics.
Outputs correlation_report.json with Claude's insights and risk multipliers.

Usage:
    python ml/ai_analysis.py

Environment:
    ANTHROPIC_API_KEY — Required. Claude API key.
    (optional) DATA_PATH — Path to sparkify_mini.json. Defaults to ml/data/sparkify_mini.json

Output:
    ml/artifacts/correlation_report.json — Structured report with top indicators and insights
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

# Optional: load .env if python-dotenv is installed
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv(Path(__file__).parent.parent / ".env")
except Exception:
    pass

try:
    from anthropic import Anthropic
except Exception:  # anthropic optional at import time
    Anthropic = None  # type: ignore

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ml.data_loader import get_sparkify_data
from ml.feature_engineering import extract_features, FEATURE_NAMES

logger = logging.getLogger(__name__)


def compute_feature_statistics(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
) -> Dict[str, Any]:
    """
    Compute aggregate statistics for each feature, split by churn status.

    Args:
        X: Feature matrix of shape (n_users, n_features)
        y: Churn labels of shape (n_users,)
        feature_names: List of feature names

    Returns:
        Dictionary with statistics for each feature:
        {
            "feature_name": {
                "churned": {
                    "mean": float,
                    "median": float,
                    "std": float,
                    "count": int,
                },
                "active": {
                    "mean": float,
                    "median": float,
                    "std": float,
                    "count": int,
                },
                "overall": {
                    "mean": float,
                    "median": float,
                    "std": float,
                },
                "churn_rate_percentiles": [
                    {"percentile": 10, "value": float, "churn_rate": float},
                    ...
                ]
            }
        }
    """
    stats = {}

    for i, feature_name in enumerate(feature_names):
        feature_values = X[:, i]
        churned_mask = y == 1
        active_mask = y == 0

        churned_values = feature_values[churned_mask]
        active_values = feature_values[active_mask]

        # Overall statistics
        overall_stats = {
            "mean": float(np.mean(feature_values)),
            "median": float(np.median(feature_values)),
            "std": float(np.std(feature_values)),
        }

        # Churned users statistics
        churned_stats = {
            "mean": float(np.mean(churned_values)) if len(churned_values) > 0 else 0.0,
            "median": float(np.median(churned_values)) if len(churned_values) > 0 else 0.0,
            "std": float(np.std(churned_values)) if len(churned_values) > 0 else 0.0,
            "count": int(len(churned_values)),
        }

        # Active users statistics
        active_stats = {
            "mean": float(np.mean(active_values)) if len(active_values) > 0 else 0.0,
            "median": float(np.median(active_values)) if len(active_values) > 0 else 0.0,
            "std": float(np.std(active_values)) if len(active_values) > 0 else 0.0,
            "count": int(len(active_values)),
        }

        # Churn rate at percentiles
        percentiles = [10, 25, 50, 75, 90]
        churn_rate_percentiles = []

        for percentile in percentiles:
            threshold = float(np.percentile(feature_values, percentile))
            mask_below = feature_values <= threshold
            churn_rate_below = float(np.mean(y[mask_below])) if np.sum(mask_below) > 0 else 0.0

            churn_rate_percentiles.append({
                "percentile": percentile,
                "value": round(threshold, 2),
                "churn_rate": round(churn_rate_below, 4),
            })

        stats[feature_name] = {
            "churned": churned_stats,
            "active": active_stats,
            "overall": overall_stats,
            "churn_rate_percentiles": churn_rate_percentiles,
        }

    return stats


def format_statistics_for_claude(
    stats: Dict[str, Any],
    overall_churn_rate: float,
) -> str:
    """
    Format feature statistics as a readable text block for Claude analysis.

    Args:
        stats: Feature statistics dictionary from compute_feature_statistics()
        overall_churn_rate: Overall churn rate across all users

    Returns:
        Formatted text string for Claude prompt
    """
    lines = []
    lines.append("SPARKIFY DATASET FEATURE STATISTICS")
    lines.append("=" * 80)
    lines.append(f"Overall Churn Rate: {overall_churn_rate:.2%}")
    lines.append("")

    for feature_name, feature_stats in stats.items():
        lines.append(f"\nFeature: {feature_name}")
        lines.append("-" * 80)

        churned = feature_stats["churned"]
        active = feature_stats["active"]
        overall = feature_stats["overall"]

        lines.append(f"  Overall:  mean={round(overall['mean'], 2)}, median={round(overall['median'], 2)}, std={round(overall['std'], 2)}")
        lines.append(f"  Churned:  mean={round(churned['mean'], 2)}, median={round(churned['median'], 2)}, std={round(churned['std'], 2)} (n={churned['count']})")
        lines.append(f"  Active:   mean={round(active['mean'], 2)}, median={round(active['median'], 2)}, std={round(active['std'], 2)} (n={active['count']})")

        # Compute difference in means
        if overall['std'] > 0:
            mean_diff = churned['mean'] - active['mean']
            lines.append(f"  Mean Difference (Churned - Active): {round(mean_diff, 2)}")

        # Churn rate at percentiles
        lines.append("  Churn rate at percentiles:")
        for pct_data in feature_stats["churn_rate_percentiles"]:
            lines.append(
                f"    {pct_data['percentile']:>2d}th percentile "
                f"(value={pct_data['value']:>8.2f}): churn_rate={pct_data['churn_rate']:.2%}"
            )

    return "\n".join(lines)


def parse_claude_response(response_text: str) -> Dict[str, Any]:
    """
    Parse Claude's response into structured JSON.

    Expects Claude to return a JSON block with:
    {
        "top_indicators": [
            {"feature": "...", "insight": "...", "risk_multiplier": X.XX},
            ...
        ],
        "summary": "...",
        "recommended_features": ["...", ...]
    }

    Args:
        response_text: Claude's response text

    Returns:
        Parsed JSON dictionary

    Raises:
        ValueError: If response cannot be parsed as valid JSON
    """
    try:
        # Try to extract JSON from response
        # Claude may wrap it with markdown code blocks
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            json_str = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            json_str = response_text[start:end].strip()
        else:
            # Try to find JSON object directly
            start = response_text.find("{")
            end = response_text.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = response_text[start:end]
            else:
                json_str = response_text

        parsed = json.loads(json_str)
        return parsed
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse Claude's response as JSON: {e}\n\nResponse:\n{response_text}")


def deterministic_correlation_fallback(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
) -> Dict[str, Any]:
    """
    Deterministic fallback when Claude is unreachable.

    Computes Pearson correlation of each feature against the churn label
    and synthesizes a recommended-features list ordered by |corr|.
    """
    insights = []
    for i, name in enumerate(feature_names):
        col = X[:, i].astype(np.float64)
        if np.std(col) == 0 or np.std(y) == 0:
            corr = 0.0
        else:
            corr = float(np.corrcoef(col, y.astype(np.float64))[0, 1])
        # Estimate a risk multiplier: churn rate among above-median vs overall
        median = float(np.median(col))
        mask_high = col >= median
        rate_high = float(np.mean(y[mask_high])) if mask_high.any() else 0.0
        overall = float(np.mean(y)) if len(y) else 0.0
        risk_multiplier = (rate_high / overall) if overall > 0 else 1.0

        direction = "positively" if corr > 0 else "negatively"
        insights.append(
            {
                "feature": name,
                "insight": (
                    f"Correlates {direction} with churn (r={corr:+.3f})."
                ),
                "risk_multiplier": round(risk_multiplier, 3),
                "_abs_corr": abs(corr),
            }
        )

    insights.sort(key=lambda d: d["_abs_corr"], reverse=True)
    top = insights[: min(5, len(insights))]
    for d in insights:
        d.pop("_abs_corr", None)

    return {
        "top_indicators": top,
        "summary": (
            "Deterministic fallback (Claude unavailable): top indicators ranked "
            "by |Pearson correlation| with the churn label."
        ),
        "recommended_features": [d["feature"] for d in top],
    }


def analyze_with_claude(statistics_text: str) -> Dict[str, Any]:
    """
    Call Claude to analyze feature statistics and identify churn indicators.

    Args:
        statistics_text: Formatted feature statistics for Claude

    Returns:
        Parsed response with top indicators and insights

    Raises:
        ValueError: If ANTHROPIC_API_KEY not set
        Exception: If Claude API call fails
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable not set. "
            "Please set it and try again."
        )

    try:
        client = Anthropic(api_key=api_key)
    except TypeError as e:
        if "proxies" in str(e):
            import httpx
            client = Anthropic(api_key=api_key, http_client=httpx.Client())
        else:
            raise

    system_prompt = (
        "You are a data analyst specializing in user retention and churn prediction. "
        "Given behavioral statistics from a music streaming service, your task is to identify "
        "the strongest leading indicators of churn. For each indicator, explain why it predicts churn "
        "and provide a risk multiplier (e.g., 2.4x means users with that signal are 2.4x more likely to churn). "
        "Return your analysis as valid JSON with the following structure:\n"
        "{\n"
        '  "top_indicators": [\n'
        '    {"feature": "feature_name", "insight": "explanation of why it predicts churn", "risk_multiplier": 2.4},\n'
        '    ...\n'
        '  ],\n'
        '  "summary": "overall summary of key insights",\n'
        '  "recommended_features": ["feature1", "feature2", ...]\n'
        "}\n"
        "Focus on:\n"
        "1. Features where churned users have significantly different values than active users\n"
        "2. Features where churn rate changes dramatically at different percentiles\n"
        "3. Behavioral patterns that suggest disengagement or dissatisfaction\n"
        "Be precise with risk multipliers (estimate as churn_rate_for_signal / overall_churn_rate)"
    )

    user_message = (
        f"Analyze the following Sparkify dataset statistics and identify the strongest "
        f"leading indicators of churn:\n\n{statistics_text}\n\n"
        f"Provide your response as valid JSON only, with no additional text before or after the JSON block."
    )

    logger.info("Calling Claude API for churn indicator analysis...")

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_message}
        ],
    )

    response_text = response.content[0].text
    logger.info(f"Received response from Claude: {len(response_text)} characters")

    parsed_response = parse_claude_response(response_text)
    return parsed_response


def print_human_readable_summary(
    stats: Dict[str, Any],
    overall_churn_rate: float,
    claude_analysis: Dict[str, Any],
) -> None:
    """
    Print a human-readable summary of statistics and Claude's insights.

    Args:
        stats: Feature statistics dictionary
        overall_churn_rate: Overall churn rate
        claude_analysis: Claude's analysis response
    """
    print("\n" + "=" * 80)
    print("SPARKIFY DATASET CHURN INDICATOR ANALYSIS")
    print("=" * 80)

    print(f"\nOverall Churn Rate: {overall_churn_rate:.2%}")
    print(f"Analysis Timestamp: {datetime.now().isoformat()}")

    # Feature statistics table
    print("\n" + "-" * 80)
    print("FEATURE STATISTICS")
    print("-" * 80)
    print(
        f"{'Feature':<30} | "
        f"{'Churned Mean':<15} | "
        f"{'Active Mean':<15} | "
        f"{'Difference':<15}"
    )
    print("-" * 80)

    for feature_name, feature_stats in stats.items():
        churned_mean = feature_stats["churned"]["mean"]
        active_mean = feature_stats["active"]["mean"]
        diff = churned_mean - active_mean

        print(
            f"{feature_name:<30} | "
            f"{churned_mean:>14.2f} | "
            f"{active_mean:>14.2f} | "
            f"{diff:>14.2f}"
        )

    # Claude's top indicators
    print("\n" + "-" * 80)
    print("TOP CHURN INDICATORS (Claude Analysis)")
    print("-" * 80)

    top_indicators = claude_analysis.get("top_indicators", [])
    for i, indicator in enumerate(top_indicators, 1):
        feature = indicator.get("feature", "Unknown")
        insight = indicator.get("insight", "No insight provided")
        risk_mult = indicator.get("risk_multiplier", 0)

        print(f"\n{i}. {feature}")
        print(f"   Risk Multiplier: {risk_mult:.2f}x")
        print(f"   Insight: {insight}")

    # Summary
    print("\n" + "-" * 80)
    print("SUMMARY")
    print("-" * 80)
    summary = claude_analysis.get("summary", "No summary provided")
    # Wrap summary text at 80 characters
    for line in summary.split('\n'):
        if len(line) > 80:
            words = line.split()
            current_line = ""
            for word in words:
                if len(current_line) + len(word) + 1 <= 80:
                    current_line += word + " "
                else:
                    if current_line:
                        print(current_line.rstrip())
                    current_line = word + " "
            if current_line:
                print(current_line.rstrip())
        else:
            print(line)

    # Recommended features
    recommended = claude_analysis.get("recommended_features", [])
    if recommended:
        print("\n" + "-" * 80)
        print("RECOMMENDED FEATURES FOR TRAINING")
        print("-" * 80)
        for feat in recommended:
            print(f"  - {feat}")

    print("\n" + "=" * 80)
    print(f"Report saved to: ml/artifacts/correlation_report.json")
    print("=" * 80 + "\n")


def main():
    """
    Main entry point: load data, compute statistics, call Claude, save report.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        # Step 1: Load Sparkify dataset
        logger.info("Step 1: Loading Sparkify dataset...")
        try:
            df = get_sparkify_data()
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            print("\nTo run this analysis, download the Sparkify dataset:")
            print("  1. Download sparkify_mini.json from Kaggle or Udacity")
            print("  2. Create directory: mkdir -p ml/data")
            print("  3. Place file at: ml/data/sparkify_mini.json")
            sys.exit(1)

        logger.info(f"Loaded {len(df):,} events from {df['userId'].nunique():,} users")

        # Step 2: Extract features
        logger.info("Step 2: Extracting features...")
        X, y, feature_names = extract_features(df)
        logger.info(f"Extracted features: {X.shape} with churn rate {y.mean():.4f}")

        # Step 3: Compute statistics
        logger.info("Step 3: Computing aggregate statistics...")
        stats = compute_feature_statistics(X, y, feature_names)
        overall_churn_rate = float(np.mean(y))
        logger.info(f"Statistics computed for {len(stats)} features")

        # Step 4: Format for Claude
        logger.info("Step 4: Formatting statistics for Claude...")
        statistics_text = format_statistics_for_claude(stats, overall_churn_rate)

        # Step 5: Call Claude (with deterministic fallback if unreachable)
        logger.info("Step 5: Calling Claude API for analysis...")
        try:
            if Anthropic is None:
                raise RuntimeError("anthropic package not installed")
            claude_analysis = analyze_with_claude(statistics_text)
            logger.info("Claude analysis complete")
        except Exception as e:
            logger.warning(
                "Claude unreachable (%s) — using deterministic correlation fallback",
                e,
            )
            claude_analysis = deterministic_correlation_fallback(X, y, feature_names)

        # Step 6: Save report
        logger.info("Step 6: Saving correlation report...")
        artifacts_dir = Path(__file__).parent / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)

        report = {
            "timestamp": datetime.now().isoformat(),
            "dataset_summary": {
                "total_events": int(len(df)),
                "total_users": int(df['userId'].nunique()),
                "churn_rate": round(overall_churn_rate, 4),
            },
            "feature_statistics": stats,
            "claude_analysis": claude_analysis,
        }

        report_path = artifacts_dir / "correlation_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved to {report_path}")

        # Step 7: Print summary
        print_human_readable_summary(stats, overall_churn_rate, claude_analysis)

        return 0

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())
