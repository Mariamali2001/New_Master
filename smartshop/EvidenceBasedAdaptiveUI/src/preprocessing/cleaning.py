"""Data cleaning and validation utilities."""

from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)

BFI10_RAW_COLUMNS: list[str] = [
    "trait_introversion",
    "trait_trust",
    "trait_low_conscientiousness",
    "trait_emotional_stability",
    "trait_low_openness",
    "trait_extraversion",
    "trait_agreeableness_reverse",
    "trait_conscientiousness",
    "trait_neuroticism",
    "trait_openness",
]

BIG_FIVE_TRAITS: list[str] = [
    "Extraversion",
    "Agreeableness",
    "Conscientiousness",
    "Neuroticism",
    "Openness",
]


def dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return dtype, missing count, and unique count for each column."""
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing": df.isna().sum(),
            "unique": df.nunique(),
        }
    )


# Expected correct answers for the five embedded attention checkers.
ATTENTION_CHECK_EXPECTED: dict[str, object] = {
    "check1": 3,
    "check2": 3,
    "check3": "Rounded Corners",
    "check4": 3,
    "check5": 3,
}


def score_attention_checks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score check1–check5 against expected answers.

    Adds:
      - attention_n_correct (0–5)
      - attention_n_wrong   (0–5)
    """
    scored = df.copy()
    check_columns = list(ATTENTION_CHECK_EXPECTED.keys())
    missing = [c for c in check_columns if c not in scored.columns]
    if missing:
        raise KeyError(f"Missing attention-check columns: {missing}")

    correct = None
    for column, expected in ATTENTION_CHECK_EXPECTED.items():
        hit = (scored[column] == expected).astype(int)
        correct = hit if correct is None else correct + hit

    scored["attention_n_correct"] = correct
    scored["attention_n_wrong"] = len(check_columns) - scored["attention_n_correct"]
    return scored


def filter_attention_checks(
    df: pd.DataFrame,
    attention_column: str = "attention_check",
    expected_attention_value: int = 3,
    min_correct: int = 4,
) -> pd.DataFrame:
    """
    Remove responses that fail attention-check questions.

    Rule for check1–check5 (thesis quality filter):
      - Keep if at least `min_correct` of 5 checkers are right
        (default: 4/5 correct → at most 1 wrong is tolerated).
      - Drop if more than one checker is wrong (likely rushed / low-effort).

    Also supports an optional single `attention_check` column when present.
    """
    filtered = df.copy()
    initial_rows = len(filtered)

    if attention_column in filtered.columns:
        filtered = filtered[filtered[attention_column] == expected_attention_value]
        logger.info(
            "Removed %s rows via %s filter",
            initial_rows - len(filtered),
            attention_column,
        )
    elif attention_column != "attention_check":
        # Only warn when a custom column was requested and is missing.
        logger.warning("Attention column '%s' not found.", attention_column)
    # else: survey uses check1–check5 (not a single attention_check column) — OK

    check_columns = list(ATTENTION_CHECK_EXPECTED.keys())
    if all(column in filtered.columns for column in check_columns):
        before_checks = len(filtered)
        scored = score_attention_checks(filtered)
        max_wrong = len(check_columns) - min_correct
        keep_mask = scored["attention_n_correct"] >= min_correct
        filtered = scored.loc[keep_mask].drop(
            columns=["attention_n_correct", "attention_n_wrong"]
        ).copy()

        n_wrong_dist = scored["attention_n_wrong"].value_counts().sort_index()
        logger.info(
            "Attention checkers before filter: %s rows | wrong-count distribution: %s",
            before_checks,
            n_wrong_dist.to_dict(),
        )
        logger.info(
            "Keep rule: >= %s/5 correct (max %s wrong). Kept %s, removed %s",
            min_correct,
            max_wrong,
            len(filtered),
            before_checks - len(filtered),
        )
    elif any(column in filtered.columns for column in check_columns):
        present = [c for c in check_columns if c in filtered.columns]
        logger.warning(
            "Only partial attention checkers present (%s); skipping check1-check5 filter.",
            present,
        )

    return filtered


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace, lowercase, and replace spaces with underscores."""
    cleaned = df.copy()
    cleaned.columns = (
        cleaned.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    )
    return cleaned


def compute_big_five_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Reverse-score negative BFI-10 items and compute Big Five trait scores."""
    scored = df.copy()

    scored["trait_introversion_r"] = 6 - scored["trait_introversion"]
    scored["trait_agreeableness_reverse_r"] = 6 - scored["trait_agreeableness_reverse"]
    scored["trait_low_conscientiousness_r"] = 6 - scored["trait_low_conscientiousness"]
    scored["trait_emotional_stability_r"] = 6 - scored["trait_emotional_stability"]
    scored["trait_low_openness_r"] = 6 - scored["trait_low_openness"]

    scored["Extraversion"] = (
        scored["trait_extraversion"] + scored["trait_introversion_r"]
    ) / 2
    scored["Agreeableness"] = (
        scored["trait_trust"] + scored["trait_agreeableness_reverse_r"]
    ) / 2
    scored["Conscientiousness"] = (
        scored["trait_conscientiousness"] + scored["trait_low_conscientiousness_r"]
    ) / 2
    scored["Neuroticism"] = (
        scored["trait_neuroticism"] + scored["trait_emotional_stability_r"]
    ) / 2
    scored["Openness"] = (
        scored["trait_openness"] + scored["trait_low_openness_r"]
    ) / 2

    return scored


def categorize_trait(score: float) -> str:
    """Map a continuous trait score to Low, Medium, or High."""
    if score <= 2.5:
        return "Low"
    if score <= 3.5:
        return "Medium"
    return "High"


def add_big_five_levels(
    df: pd.DataFrame,
    traits: Iterable[str] = BIG_FIVE_TRAITS,
) -> pd.DataFrame:
    """Add categorical Big Five level columns."""
    leveled = df.copy()
    for trait in traits:
        leveled[f"{trait}_Level"] = leveled[trait].apply(categorize_trait)
    return leveled


def drop_bfi10_raw_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop original BFI-10 questionnaire item columns."""
    existing = [column for column in BFI10_RAW_COLUMNS if column in df.columns]
    return df.drop(columns=existing)


def categorical_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize unique values and missing counts for object columns."""
    rows = []
    for column in df.select_dtypes(include="object").columns:
        rows.append(
            {
                "column": column,
                "unique_values": df[column].nunique(),
                "missing": df[column].isna().sum(),
            }
        )
    return pd.DataFrame(rows)
