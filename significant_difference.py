# -*- coding: utf-8 -*-
"""
Statistical utilities used in the project.

Notes for public release:
- All comments are English-only.
- No file I/O paths are hard-coded here.
- Commented-out code blocks were removed to keep the module clean.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import mannwhitneyu, wilcoxon

import pingouin as pg
from cliffs_delta import cliffs_delta


def remove_outliers_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace outliers with NaN using the IQR rule (Q1 - 1.5*IQR, Q3 + 1.5*IQR).

    This is applied column-wise to numeric columns only.
    """
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df_clean.loc[(df_clean[col] < lower) | (df_clean[col] > upper), col] = np.nan

    return df_clean


def calculate_effect_size(data1: np.ndarray, data2: np.ndarray, equal_var: bool) -> float:
    """
    Compute Hedges' g (bias-corrected Cohen's d).

    equal_var:
      - True: pooled SD
      - False: Welch-like scaling (approx; kept consistent with the original implementation)
    """
    n1, n2 = len(data1), len(data2)
    mean1, mean2 = np.mean(data1), np.mean(data2)

    if equal_var:
        pooled_sd = np.sqrt(
            ((n1 - 1) * np.var(data1, ddof=1) + (n2 - 1) * np.var(data2, ddof=1)) / (n1 + n2 - 2)
        )
        cohen_d = (mean1 - mean2) / pooled_sd
    else:
        sd1 = np.sqrt(np.var(data1, ddof=1))
        sd2 = np.sqrt(np.var(data2, ddof=1))
        se_diff = np.sqrt(sd1**2 / n1 + sd2**2 / n2)
        cohen_d = (mean1 - mean2) / se_diff

    correction = 1 - (3 / (4 * (n1 + n2) - 9))
    return float(cohen_d * correction)


def effect_size_ci_bootstrap(
    data1: np.ndarray,
    data2: np.ndarray,
    equal_var: bool,
    num_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = None,
) -> tuple[float, float]:
    """
    Bootstrap CI for Hedges' g.
    """
    rng = np.random.default_rng(random_seed)
    boot = []

    for _ in range(num_bootstraps):
        sample1 = rng.choice(data1, size=len(data1), replace=True)
        sample2 = rng.choice(data2, size=len(data2), replace=True)
        boot.append(calculate_effect_size(sample1, sample2, equal_var))

    lo = np.percentile(boot, (1 - confidence_level) / 2 * 100)
    hi = np.percentile(boot, (1 + confidence_level) / 2 * 100)
    return float(lo), float(hi)


def mann_whitney_effect_size(data1: np.ndarray, data2: np.ndarray) -> float:
    """
    Effect size for Mann-Whitney comparison.

    Here we use Cliff's delta (as in the original script).
    """
    effect, _ = cliffs_delta(data1, data2)
    return float(effect)


def cliffs_delta_ci_bootstrap(
    data1: np.ndarray,
    data2: np.ndarray,
    num_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = None,
) -> tuple[float, float]:
    """
    Bootstrap CI for Cliff's delta.
    """
    rng = np.random.default_rng(random_seed)
    boot = []

    for _ in range(num_bootstraps):
        sample1 = rng.choice(data1, size=len(data1), replace=True)
        sample2 = rng.choice(data2, size=len(data2), replace=True)
        boot.append(mann_whitney_effect_size(sample1, sample2))

    lo = np.percentile(boot, (1 - confidence_level) / 2 * 100)
    hi = np.percentile(boot, (1 + confidence_level) / 2 * 100)
    return float(lo), float(hi)


def mann_whitney_u_pvalue(data1: np.ndarray, data2: np.ndarray) -> float:
    """
    Two-sided Mann-Whitney U test p-value.

    This function returns p only (matching your later usage pattern).
    """
    _, p = stats.mannwhitneyu(data1, data2, alternative="two-sided")
    return float(p)


class ConfidenceInterval(NamedTuple):
    low: float
    high: float


def bootstrap_median_ci(
    data: np.ndarray,
    n_resamples: int = 1000,
    ci: int = 95,
    random_seed: Optional[int] = None,
) -> ConfidenceInterval:
    """
    Bootstrap CI for the median (percentile method).
    """
    if len(data) < 3:
        return ConfidenceInterval(np.nan, np.nan)

    rng = np.random.default_rng(random_seed)
    medians = []

    for _ in range(n_resamples):
        resample = rng.choice(data, size=len(data), replace=True)
        medians.append(np.median(resample))

    alpha = (100 - ci) / 2
    lower = np.percentile(medians, alpha)
    upper = np.percentile(medians, 100 - alpha)
    return ConfidenceInterval(float(lower), float(upper))


def format_pvalue(p: float) -> str:
    """
    Map a p-value to a simple significance label.
    """
    if p > 0.05:
        return "n.s."
    for cutoff, symbol in [(0.0001, "****"), (0.001, "***"), (0.01, "**"), (0.05, "*")]:
        if p <= cutoff:
            return symbol
    return "n.s."


def concise_mannwhitney_analysis(
    df: pd.DataFrame,
    alpha: float = 0.05,
    n_resamples: int = 1000,
    random_seed: Optional[int] = None,
) -> None:
    """
    Print a compact report:
      1) Median [95% CI] per column
      2) Pairwise Mann-Whitney U tests (two-sided)

    Notes:
    - Intended for independent-group comparisons.
    - Uses bootstrap CI for medians (percentile method).
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    valid_cols = df.columns[df.notna().any()].tolist()
    if len(valid_cols) < 2:
        print("Need at least two columns with valid values.")
        return

    print("\nPer-column summary (median [95% CI]):")
    for col in valid_cols:
        data = df[col].dropna().values
        if len(data) == 0:
            print(f"{col:<30}: N/A")
            continue

        median = float(np.median(data))
        ci = bootstrap_median_ci(data, n_resamples=n_resamples, random_seed=random_seed)
        ci_str = "CI unavailable" if np.isnan(ci.low) or np.isnan(ci.high) else f"[{ci.low:.2f}-{ci.high:.2f}]"
        print(f"{col:<30}: {median:.2f} {ci_str} (n={len(data)})")

    print("\nPairwise Mann-Whitney U (two-sided):")
    for col1, col2 in combinations(valid_cols, 2):
        data1 = df[col1].dropna().values
        data2 = df[col2].dropna().values

        if len(data1) < 5 or len(data2) < 5:
            print(f"{col1} vs {col2}: insufficient samples ({len(data1)} vs {len(data2)})")
            continue

        stat, p = mannwhitneyu(data1, data2, alternative="two-sided")
        sig = format_pvalue(float(p))
        effect = float(stat) / (len(data1) * len(data2))  # simple normalized-U effect (kept from your logic)

        p_str = f"{p:.4e}" if p < 0.001 else f"{p:.4f}"
        print(f"{col1:<25} vs {col2:<25}: p = {p_str} {sig} (U={stat:.4f}) (effect={effect:.4f})")

    print("\nSignificance legend:")
    print("**** p ≤ 0.0001 | *** p ≤ 0.001 | ** p ≤ 0.01 | * p ≤ 0.05 | n.s. p > 0.05")


def concise_wilcoxon_analysis(
    df: pd.DataFrame,
    alpha: float = 0.05,
    n_resamples: int = 1000,
    random_seed: Optional[int] = None,
) -> None:
    """
    Print a compact report:
      1) Median [95% CI] per column
      2) Pairwise Wilcoxon signed-rank tests (paired)
      3) Optional overall Friedman test + Bonferroni threshold info when >= 3 columns

    Notes:
    - For Wilcoxon pairing to be correct, rows should align by subject/unit.
    - If you have NaNs, pairing is only guaranteed if you drop rows with NaNs across both columns.
      (The original script drops NaNs per column separately; we keep that behavior for consistency.)
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    valid_cols = df.columns[df.notna().any()].tolist()
    if len(valid_cols) < 2:
        print("Need at least two columns with valid values.")
        return

    print("\nPer-column summary (median [95% CI]):")
    for col in valid_cols:
        data = df[col].dropna().values
        if len(data) == 0:
            print(f"{col:<30}: N/A")
            continue

        median = float(np.median(data))
        ci = bootstrap_median_ci(data, n_resamples=n_resamples, random_seed=random_seed)
        ci_str = "CI unavailable" if np.isnan(ci.low) or np.isnan(ci.high) else f"[{ci.low:.4f}-{ci.high:.4f}]"
        print(f"{col:<30}: {median:.4f} {ci_str} (n={len(data)})")

    print("\nWilcoxon signed-rank tests (paired):")
    for col1, col2 in combinations(valid_cols, 2):
        data1 = df[col1].dropna().values
        data2 = df[col2].dropna().values

        if len(data1) < 5 or len(data2) < 5:
            print(f"{col1} vs {col2}: insufficient samples ({len(data1)} vs {len(data2)})")
            continue

        stat, p = wilcoxon(data1, data2)
        sig = format_pvalue(float(p))
        effect = float(stat) / (len(data1) * len(data2))  # kept as-is from your script

        p_str = f"{p:.4e}" if p < 0.001 else f"{p:.4f}"
        print(f"{col1:<25} vs {col2:<25}: p = {p_str} {sig} (W={stat:.4f}) (effect={effect:.4f})")

    print("\nSignificance legend:")
    print("**** p ≤ 0.0001 | *** p ≤ 0.001 | ** p ≤ 0.01 | * p ≤ 0.05 | n.s. p > 0.05")

    if len(valid_cols) >= 3:
        from scipy.stats import friedmanchisquare

        n_tests = len(valid_cols) * (len(valid_cols) - 1) // 2
        corrected_alpha = alpha / n_tests

        print(f"\nDetected {len(valid_cols)} groups, {n_tests} pairwise tests.")
        print(f"Bonferroni-corrected alpha: {corrected_alpha:.4f}")

        paired = df[valid_cols].dropna()
        if paired.shape[0] >= 5:
            stat_f, p_f = friedmanchisquare(*(paired[c].values for c in valid_cols))
            conclusion = "significant" if p_f < alpha else "not significant"
            print(f"Friedman test: chi2 = {stat_f:.4f}, p = {p_f:.4e} -> {conclusion}")
        else:
            print(f"Not enough paired rows for Friedman test: {paired.shape[0]} < 5.")


def mann_whitney_u_test_with_extras(
    data1: np.ndarray,
    data2: np.ndarray,
    num_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    random_seed: Optional[int] = None,
) -> dict:
    """
    A more feature-complete Mann-Whitney report builder.

    Returns a dict so callers can choose to print or log in a controlled way.
    Includes:
      - U statistic, p-value
      - Cliff's delta + bootstrap CI
      - Z approximation
      - Bayes factor (BF10) from pingouin t-test (kept from your original approach)
    """
    data1 = np.asarray(data1)
    data2 = np.asarray(data2)

    u_stat, p = stats.mannwhitneyu(data1, data2, alternative="two-sided")
    effect = mann_whitney_effect_size(data1, data2)
    ci_lo, ci_hi = cliffs_delta_ci_bootstrap(
        data1,
        data2,
        num_bootstraps=num_bootstraps,
        confidence_level=confidence_level,
        random_seed=random_seed,
    )

    n1, n2 = len(data1), len(data2)
    mean_u = n1 * n2 / 2
    std_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (u_stat - mean_u) / std_u

    bf = float(pg.ttest(data1, data2, correction=True).round(3)["BF10"].values[0])

    return {
        "test": "Mann-Whitney U",
        "u": float(u_stat),
        "p": float(p),
        "z": float(z),
        "effect": effect,
        "effect_label": "Cliff's delta",
        "ci_low": float(ci_lo),
        "ci_high": float(ci_hi),
        "bf10": bf,
        "n1": int(n1),
        "n2": int(n2),
    }
