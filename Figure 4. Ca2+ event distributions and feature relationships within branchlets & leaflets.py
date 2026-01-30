# -*- coding: utf-8 -*-
 

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter
from scipy.stats import gaussian_kde

from significant_difference import (
    remove_outliers_df,
    number_of_cell,
    Mann_Whitney_U_test,
    concise_mannwhitney_analysis,
    concise_wilcoxon_analysis,
)


# =============================================================================
# Config
# =============================================================================

@dataclass(frozen=True)
class Paths:
    # Replace these with your real paths
    data_csv: Path = Path("PATH/TO/events_with_loc_label.csv")
    vae_pickle: Path = Path("PATH/TO/vae_z_values_young_with_distance.pkl")
    out_dir: Path = Path("PATH/TO/output_figures")


@dataclass(frozen=True)
class UnitConversion:
    # Keep your original constants, but put them in one place
    area_scale: float = 1
    duration_scale: float = 1


# =============================================================================
# Data prep helpers
# =============================================================================

def load_events_table(
    csv_path: Path,
    usecols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Load the main events table (CSV)."""
    return pd.read_csv(csv_path, usecols=usecols)


def add_static_dynamic_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add event_static/event_dynamic based on two landmark columns.
    Convention (original):
      static if away==0 and toward==0
    """
    df = df.copy()
    df["event_static"] = 0
    static_mask = (df["Landmark_event_away_from_landmark_landmark_1"] == 0) & (
        df["Landmark_event_toward_landmark_landmark_1"] == 0
    )
    df.loc[static_mask, "event_static"] = 1
    df["event_dynamic"] = 1 - df["event_static"]
    return df


def apply_unit_conversions(df: pd.DataFrame, uc: UnitConversion) -> pd.DataFrame:
    """Apply unit conversions for area and duration."""
    df = df.copy()
    if "Basic_Area" in df.columns:
        df["Basic_Area"] = df["Basic_Area"] * uc.area_scale
    if "Curve_Duration_10_to_10" in df.columns:
        df["Curve_Duration_10_to_10"] = df["Curve_Duration_10_to_10"] * uc.duration_scale
    return df


def filter_young_wt_and_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Original filter:
      (wt==0) & (primary_label>=0) & (secondary_label>=0)
    """
    df = df.copy()
    return df[(df["wt"] == 0) & (df["primary_label"] >= 0) & (df["secondary_label"] >= 0)].reset_index(drop=True)


def assign_3bin_label(
    series: pd.Series,
    thresholds: Tuple[float, float],
    labels: Tuple[int, int, int] = (1, 2, 3),
) -> np.ndarray:
    """
    Convert a numeric series into 3 bins:
      <= t1 -> labels[0]
      (t1, t2) -> labels[1]
      >= t2 -> labels[2]
    """
    t1, t2 = thresholds
    cond1 = series <= t1
    cond2 = (series > t1) & (series < t2)
    cond3 = series >= t2
    return np.select([cond1, cond2, cond3], labels, default=0)


def compute_cellwise_label_percentages(
    df: pd.DataFrame,
    label_col: str,
    cell_col: str = "cell_id",
    keep_label_values: Sequence[int] = (1, 2, 3),
    prefix: str = "",
) -> pd.DataFrame:
    """
    For each cell_id, compute the within-cell percentages of label values.

    Returns a wide table:
      cell_id | <prefix>label_1 | <prefix>label_2 | ...
    """
    out = (
        df.groupby(cell_col)[label_col]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .reindex(columns=list(keep_label_values), fill_value=0)
        .rename(columns=lambda x: f"{prefix}{label_col}_{x}")
        .reset_index()
    )
    return out


# =============================================================================
# Plotting helpers
# =============================================================================

def plot_paired_violin_scatter_box(
    values_wide: pd.DataFrame,
    save_path: Path,
    violin_palette: Sequence[str],
    scatter_palette: Sequence[str],
    figsize: Tuple[float, float] = (8, 6),
    scatter_size: float = 5.0,
    x_scatter_shift: float = 0.11,
    extend_y_top_ratio: float = 0.10,
    extend_y_bottom_ratio: float = 0.0,
    start_y_at: Optional[float] = None,
) -> None:
    """
    Half-violin + paired scatter + connecting lines + boxplot overlay.

    Notes:
    - Expects a wide DataFrame with columns as groups and rows as paired samples.
    - Keeps your original 'half violin' visual by clipping to half width.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=figsize)
    plt.rcParams["font.size"] = 12
    plt.rcParams["font.weight"] = "bold"

    ax = sns.violinplot(
        data=values_wide,
        dodge=False,
        scale="width",
        inner=None,
        palette=list(violin_palette),
        cut=0,
        clip_on=True,
    )

    # Clip each violin to half width (keep your original style)
    for violin in ax.collections:
        bbox = violin.get_paths()[0].get_extents()
        x0, y0, width, height = bbox.bounds
        violin.set_clip_path(
            plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData)
        )

    # Scatter
    old_len = len(ax.collections)
    sns.stripplot(
        data=values_wide,
        dodge=False,
        ax=ax,
        size=scatter_size,
        palette=list(scatter_palette),
    )

    # Shift scatter x (to match half violin)
    scatter_cols = ax.collections[old_len:]
    for col in scatter_cols:
        offs = col.get_offsets().copy()
        offs[:, 0] += x_scatter_shift
        col.set_offsets(offs)

    # Draw paired lines
    offsets_per_group = [col.get_offsets() for col in scatter_cols]
    n_samples = values_wide.shape[0]
    for i in range(n_samples):
        xs = [offs[i, 0] for offs in offsets_per_group]
        ys = [offs[i, 1] for offs in offsets_per_group]
        ax.plot(xs, ys, color="gray", alpha=0.3, linewidth=0.4, zorder=1)

    # Boxplot overlay
    sns.boxplot(
        data=values_wide,
        saturation=1,
        showfliers=False,
        width=0.3,
        boxprops={"zorder": 3, "facecolor": "none", "linewidth": 4, "edgecolor": "black"},
        whiskerprops={"linewidth": 4, "color": "black", "zorder": 10},
        capprops={"linewidth": 4, "color": "black", "zorder": 10},
        medianprops={"linewidth": 4, "color": "red", "zorder": 10},
        ax=ax,
    )

    # Y limits
    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    new_ymax = ymax + extend_y_top_ratio * yspan
    new_ymin = ymin - extend_y_bottom_ratio * yspan
    if start_y_at is not None:
        ax.set_ylim([start_y_at, new_ymax])
    else:
        ax.set_ylim([new_ymin, new_ymax])

    # Cosmetics
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.set_xticks([])
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(save_path, format="eps", bbox_inches="tight")
    plt.show()
    plt.close()


# =============================================================================
# Example pipelines (keeps your original analysis intent)
# =============================================================================

def pipeline_cellwise_percentages_and_violin(
    df: pd.DataFrame,
    label_col: str,
    keep_values: Sequence[int],
    out_path: Path,
    violin_palette: Sequence[str],
    scatter_palette: Sequence[str],
    start_y_at: Optional[float] = None,
) -> None:
    """
    Generic pipeline:
    - cellwise percentages
    - outlier removal
    - stats (cell count + paired Wilcoxon)
    - plot
    """
    result = compute_cellwise_label_percentages(
        df=df,
        label_col=label_col,
        keep_label_values=keep_values,
        prefix="",  # keep names close to original
    )

    result = remove_outliers_df(result)
    number_of_cell(result)

    result = result.dropna(axis=0)
    concise_wilcoxon_analysis(result.iloc[:, 1:])

    plot_paired_violin_scatter_box(
        values_wide=result.iloc[:, 1:],
        save_path=out_path,
        violin_palette=violin_palette,
        scatter_palette=scatter_palette,
        start_y_at=start_y_at,
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    paths = Paths()
    uc = UnitConversion()

    # -----------------------------
    # Part 1: Percentages by area/duration/dynamic within a subcellular region
    # -----------------------------
    df = load_events_table(paths.data_csv)
    df = filter_young_wt_and_labels(df)
    df = apply_unit_conversions(df, uc)
    df = add_static_dynamic_flags(df)
 

    # Area label
    df["area_label"] = assign_3bin_label(df["Basic_Area"], thresholds=(10, 100))
    pipeline_cellwise_percentages_and_violin(
        df=df,
        label_col="area_label",
        keep_values=(1, 2, 3),
        out_path=paths.out_dir / "fig_area_label_percentages.eps",
        violin_palette=("#a7ccd2", "#a7ccd2", "#a7ccd2"),
        scatter_palette=("#e67a6e", "#8a7ca4", "#4a726b"),
    )

    # Duration label
    df["duration_label"] = assign_3bin_label(df["Curve_Duration_10_to_10"], thresholds=(1, 10))
    pipeline_cellwise_percentages_and_violin(
        df=df,
        label_col="duration_label",
        keep_values=(1, 2, 3),
        out_path=paths.out_dir / "fig_duration_label_percentages.eps",
        violin_palette=("#7b9fa4", "#7b9fa4", "#7b9fa4"),
        scatter_palette=("#e67a6e", "#8a7ca4", "#4a726b"),
    )

    # Dynamic label (binary)
    df["dynamic_label"] = np.select(
        [df["event_dynamic"] == 0, df["event_dynamic"] == 1],
        [1, 2],
        default=0,
    )
    pipeline_cellwise_percentages_and_violin(
        df=df,
        label_col="dynamic_label",
        keep_values=(1, 2),
        out_path=paths.out_dir / "fig_dynamic_label_percentages.eps",
        violin_palette=("#a7ccd2", "#7b9fa4"),
        scatter_palette=("#e67a6e", "#8a7ca4", "#4a726b"),
        start_y_at=0.0,
    )
 

    print("Done. Figures saved to:", paths.out_dir)


if __name__ == "__main__":
    main()
