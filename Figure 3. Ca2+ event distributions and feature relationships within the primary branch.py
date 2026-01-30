# -*- coding: utf-8 -*-
 

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, ListedColormap
from matplotlib.ticker import PercentFormatter

# ---- External utilities (keep your original module) ----
from significant_difference import (
    remove_outliers_df,
    number_of_cell,
    Mann_Whitney_U_test,
    concise_mannwhitney_analysis,
    concise_wilcoxon_analysis,
)
 
@dataclass(frozen=True)
class Config: 
    data_csv: Path = Path("DATA/young_mouse_events_with_loc_labels.csv")
    all_mouse_cleaned_csv: Path = Path("DATA/all_mouse_cleaned.csv")
    vae_pickle: Path = Path("DATA/vae_z_values_young_with_distance.pkl")
    out_dir: Path = Path("OUTPUT/figures")

    # Constants used in your unit conversion
    area_scale: float = 1
    duration_scale: float = 1

CFG = Config()
CFG.out_dir.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Small helpers
# =============================================================================

def save_then_show(fig: plt.Figure, out_path: Path, dpi: int | None = None) -> None:
    """Save before showing to avoid empty/cleared figures on some backends."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, format=out_path.suffix.lstrip("."), bbox_inches="tight", dpi=dpi)
    plt.show()


def add_half_violin_clip(ax: plt.Axes) -> None:
    """Clip violin bodies to show only half (right half removed)."""
    for violin in ax.collections:
        # violin bodies are PathCollections with paths
        if not hasattr(violin, "get_paths") or len(violin.get_paths()) == 0:
            continue
        bbox = violin.get_paths()[0].get_extents()
        x0, y0, w, h = bbox.bounds
        violin.set_clip_path(
            plt.Rectangle((x0, y0), w / 2, h, transform=ax.transData)
        )


def paired_strip_lines(ax: plt.Axes, loc_perc: pd.DataFrame, x_shift: float = 0.11,
                       line_color: str = "gray", line_alpha: float = 0.3,
                       line_width: float = 0.4) -> None:
    """
    After stripplot is drawn, offset the scatter groups and draw paired lines
    assuming each column has same sample order (paired per row).
    """
    old_len = len(ax.collections)
    sns.stripplot(data=loc_perc, dodge=False, ax=ax, size=5,
                  palette=["#6c8fa3", "#9d8fb8", "#e96d6d"])
    scatter_cols = ax.collections[old_len:]

    # Shift x positions (useful for half-violin style)
    for col in scatter_cols:
        offs = col.get_offsets().copy()
        offs[:, 0] += x_shift
        col.set_offsets(offs)

    # Connect paired points row-wise
    offsets_per_group = [col.get_offsets() for col in scatter_cols]
    n = loc_perc.shape[0]
    for i in range(n):
        xs = [offs[i, 0] for offs in offsets_per_group]
        ys = [offs[i, 1] for offs in offsets_per_group]
        ax.plot(xs, ys, color=line_color, alpha=line_alpha, linewidth=line_width, zorder=1)


def violin_scatter_box(
    loc_perc: pd.DataFrame,
    out_path: Path,
    violin_palette: list[str],
    start_from: float | None = None,
    y_expand_ratio: float = 0.10,
    fig_size: tuple[float, float] = (8, 6),
) -> None:
    """
    Half violin + paired scatter + boxplot overlay.
    - loc_perc: columns are groups, rows are paired samples
    - start_from: if not None, y-axis starts from this value
    """
    fig, ax = plt.subplots(figsize=fig_size)

    plt.rcParams["font.size"] = 12
    plt.rcParams["font.weight"] = "bold"

    sns.violinplot(
        data=loc_perc,
        dodge=False,
        scale="width",
        inner=None,
        palette=violin_palette,
        cut=0,
        clip_on=True,
        ax=ax,
    )
    add_half_violin_clip(ax)

    paired_strip_lines(ax, loc_perc, x_shift=0.11)

    sns.boxplot(
        data=loc_perc,
        saturation=1,
        showfliers=False,
        width=0.3,
        boxprops={"zorder": 3, "facecolor": "none", "linewidth": 4, "edgecolor": "black"},
        whiskerprops={"linewidth": 4, "color": "black", "zorder": 10},
        capprops={"linewidth": 4, "color": "black", "zorder": 10},
        medianprops={"linewidth": 4, "color": "red", "zorder": 10},
        ax=ax,
    )

    # Axis styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.set_xticks([])
    ax.grid(False)

    ymin, ymax = ax.get_ylim()
    new_ymax = ymax + y_expand_ratio * (ymax - ymin)
    if start_from is not None:
        ax.set_ylim([start_from, new_ymax])
    else:
        ax.set_ylim([ymin, new_ymax])

    save_then_show(fig, out_path)


# =============================================================================
# Data preparation
# =============================================================================

def load_primary_branch_events(cfg: Config) -> pd.DataFrame:
    """Load and filter events for the primary branch region (your original condition)."""
    df = pd.read_csv(cfg.data_csv)
    df = df[(df["wt"] == 0) & (df["primary_label"] > 0) & (df["secondary_label"] == -1)].reset_index(drop=True)

    # Unit conversion
    df["Basic_Area"] = df["Basic_Area"] * cfg.area_scale
    df["Curve_Duration_10_to_10"] = df["Curve_Duration_10_to_10"] * cfg.duration_scale

    # Static / dynamic
    df["event_static"] = 0
    static_mask = (df["Landmark_event_away_from_landmark_landmark_1"] == 0) & \
                  (df["Landmark_event_toward_landmark_landmark_1"] == 0)
    df.loc[static_mask, "event_static"] = 1
    df["event_dynamic"] = 1 - df["event_static"]

    return df


def add_three_level_label(df: pd.DataFrame, col: str, cuts: tuple[float, float],
                          new_col: str) -> pd.DataFrame:
    """Create 3-level label: <=cuts[0] -> 1, (cuts[0], cuts[1]) -> 2, >=cuts[1] -> 3."""
    a, b = cuts
    cond1 = df[col] <= a
    cond2 = (df[col] > a) & (df[col] < b)
    cond3 = df[col] >= b
    df[new_col] = np.select([cond1, cond2, cond3], [1, 2, 3], default=0)
    return df


def cellwise_label_proportion(df: pd.DataFrame, label_col: str, labels: list[int]) -> pd.DataFrame:
    """
    For each cell_id: compute proportion of each label value in `labels`.
    Output: cell_id + columns like f"{label_col}_{k}"
    """
    out = (
        df.groupby("cell_id")[label_col]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .reindex(columns=labels, fill_value=0)
        .rename(columns=lambda x: f"{label_col}_{x}")
        .reset_index()
    )
    return out


# =============================================================================
# Section A: proportions within subcellular structure
# =============================================================================

def run_proportion_plots(cfg: Config) -> None:
    df = load_primary_branch_events(cfg)

    # ---- area proportion ----
    df = add_three_level_label(df, col="Basic_Area", cuts=(10, 100), new_col="area_label")
    prop = cellwise_label_proportion(df, "area_label", [1, 2, 3])
    prop = remove_outliers_df(prop).dropna(axis=0)
    number_of_cell(prop)
    concise_wilcoxon_analysis(prop.iloc[:, 1:])

    violin_scatter_box(
        prop.iloc[:, 1:],
        out_path=cfg.out_dir / "fig3_primary_branch_area_label_proportion.eps",
        violin_palette=["#ffde97"] * 3,
    )

    # ---- duration proportion ----
    df = add_three_level_label(df, col="Curve_Duration_10_to_10", cuts=(1, 10), new_col="duration_label")
    prop = cellwise_label_proportion(df, "duration_label", [1, 2, 3])
    prop = remove_outliers_df(prop).dropna(axis=0)
    number_of_cell(prop)
    concise_wilcoxon_analysis(prop.iloc[:, 1:])

    violin_scatter_box(
        prop.iloc[:, 1:],
        out_path=cfg.out_dir / "fig3_primary_branch_duration_label_proportion.eps",
        violin_palette=["#e1a964"] * 3,
    )

    # ---- dynamic proportion (2 groups) ----
    df["dynamic_label"] = np.select([df["event_dynamic"] == 0, df["event_dynamic"] == 1], [1, 2], default=0)
    prop = cellwise_label_proportion(df, "dynamic_label", [1, 2])
    prop = remove_outliers_df(prop).dropna(axis=0)
    number_of_cell(prop)
    concise_wilcoxon_analysis(prop.iloc[:, 1:])

    violin_scatter_box(
        prop.iloc[:, 1:],
        out_path=cfg.out_dir / "fig3_primary_branch_dynamic_label_proportion.eps",
        violin_palette=["#ffde97", "#e1a964"],
        start_from=0.0,
    )


# =============================================================================
# Section B: correlation difference between groups
# =============================================================================

def build_pairwise_labels_for_correlation(cfg: Config) -> pd.DataFrame:
    """
    Load your pickle merged_df, merge event_dynamic / duration / area / loc labels
    by (cell_id, signal_order), then create pairwise labels for (signal_a, signal_b).
    """
    import pickle

    with open(cfg.vae_pickle, "rb") as f:
        merged = pickle.load(f)["merged_df"]

    # 1) Merge event_dynamic from cleaned csv (for both a and b)
    base = pd.read_csv(
        cfg.all_mouse_cleaned_csv,
        usecols=["cell_id", "Landmark_event_away_from_landmark_landmark_1", "Landmark_event_toward_landmark_landmark_1"],
    )
    base["event_static"] = 0
    mask_static = (base["Landmark_event_away_from_landmark_landmark_1"] == 0) & (base["Landmark_event_toward_landmark_landmark_1"] == 0)
    base.loc[mask_static, "event_static"] = 1
    base["event_dynamic"] = 1 - base["event_static"]

    base["signal_order"] = base.groupby("cell_id").cumcount()

    merged = merged.merge(base[["cell_id", "signal_order", "event_dynamic"]],
                          left_on=["cell_id", "signal_a_order"], right_on=["cell_id", "signal_order"], how="left") \
                   .rename(columns={"event_dynamic": "signal_a_event_dynamic"}) \
                   .drop(columns=["signal_order"])

    merged = merged.merge(base[["cell_id", "signal_order", "event_dynamic"]],
                          left_on=["cell_id", "signal_b_order"], right_on=["cell_id", "signal_order"], how="left") \
                   .rename(columns={"event_dynamic": "signal_b_event_dynamic"}) \
                   .drop(columns=["signal_order"])

    # 2) Merge duration/area/loc labels from event csv (for both a and b)
    ev = pd.read_csv(cfg.data_csv, usecols=["cell_id", "Curve_Duration_10_to_10", "Basic_Area", "primary_label", "secondary_label"])
    ev["Curve_Duration_10_to_10"] = ev["Curve_Duration_10_to_10"] * cfg.duration_scale
    ev["Basic_Area"] = ev["Basic_Area"] * cfg.area_scale
    ev["signal_order"] = ev.groupby("cell_id").cumcount()

    def merge_feature(feature: str, new_a: str, new_b: str) -> None:
        nonlocal merged
        merged = merged.merge(ev[["cell_id", "signal_order", feature]],
                              left_on=["cell_id", "signal_a_order"], right_on=["cell_id", "signal_order"], how="left") \
                       .rename(columns={feature: new_a}) \
                       .drop(columns=["signal_order"])
        merged = merged.merge(ev[["cell_id", "signal_order", feature]],
                              left_on=["cell_id", "signal_b_order"], right_on=["cell_id", "signal_order"], how="left") \
                       .rename(columns={feature: new_b}) \
                       .drop(columns=["signal_order"])

    merge_feature("Curve_Duration_10_to_10", "signal_a_duration", "signal_b_duration")
    merge_feature("Basic_Area", "signal_a_area", "signal_b_area")
    merge_feature("primary_label", "signal_a_primary", "signal_b_primary")
    merge_feature("secondary_label", "signal_a_secondary", "signal_b_secondary")

    # 3) Pairwise labels
    # dynamic_label: both static(0)->1; both dynamic(1)->2
    merged["dynamic_label"] = np.select(
        [(merged["signal_a_event_dynamic"] == 0) & (merged["signal_b_event_dynamic"] == 0),
         (merged["signal_a_event_dynamic"] == 1) & (merged["signal_b_event_dynamic"] == 1)],
        [1, 2],
        default=0,
    )

    # duration_label: (<=1, 1-10, >=10) for both a and b
    merged["duration_label"] = np.select(
        [(merged["signal_a_duration"] <= 1) & (merged["signal_b_duration"] <= 1),
         (merged["signal_a_duration"] > 1) & (merged["signal_a_duration"] < 10) & (merged["signal_b_duration"] > 1) & (merged["signal_b_duration"] < 10),
         (merged["signal_a_duration"] >= 10) & (merged["signal_b_duration"] >= 10)],
        [1, 2, 3],
        default=0,
    )

    # area_label: (<=10, 10-100, >=100) for both a and b
    merged["area_label"] = np.select(
        [(merged["signal_a_area"] <= 10) & (merged["signal_b_area"] <= 10),
         (merged["signal_a_area"] > 10) & (merged["signal_a_area"] < 100) & (merged["signal_b_area"] > 10) & (merged["signal_b_area"] < 100),
         (merged["signal_a_area"] >= 100) & (merged["signal_b_area"] >= 100)],
        [1, 2, 3],
        default=0,
    )

    # loc_label: your 3-way logic
    merged["loc_label"] = np.select(
        [
            (merged["signal_a_primary"] == -1) & (merged["signal_a_secondary"] == -1) & (merged["signal_b_primary"] == -1) & (merged["signal_b_secondary"] == -1),
            (merged["signal_a_primary"] > 0) & (merged["signal_a_secondary"] == -1) & (merged["signal_b_primary"] > 0) & (merged["signal_b_secondary"] == -1),
            (merged["signal_a_primary"] >= 0) & (merged["signal_a_secondary"] >= 0) & (merged["signal_b_primary"] >= 0) & (merged["signal_b_secondary"] >= 0),
        ],
        [1, 2, 3],
        default=0,
    )

    return merged


def run_correlation_group_compare(cfg: Config,
                                 aim_y: str = "cosine_corr",
                                 aim_label: str = "dynamic_label") -> None:
    """
    For loc_label==2, compute per-cell median correlation per label,
    pivot to paired columns, do Wilcoxon, and plot.
    """
    data = build_pairwise_labels_for_correlation(cfg)
    data = data[data["loc_label"] == 2].reset_index(drop=True)

    palette_map = {
        "area_label": ["#ffde97"] * 3,
        "duration_label": ["#e1a964"] * 3,
        "dynamic_label": ["#fec37d"] * 3,  # (kept from your original)
    }
    violin_palette = palette_map.get(aim_label, ["#fec37d"] * 3)

    cell_median = data.groupby(["cell_id", aim_label])[aim_y].median().reset_index()
    cell_median = cell_median[cell_median[aim_label] > 0].reset_index(drop=True)

    pivot = cell_median.pivot(index="cell_id", columns=aim_label, values=aim_y).reset_index()
    # Rename numeric columns to stable names
    pivot.columns = [f"group_{c}" if isinstance(c, (int, np.integer)) else c for c in pivot.columns]

    pivot = remove_outliers_df(pivot).dropna(axis=0)
    number_of_cell(pivot)
    concise_wilcoxon_analysis(pivot.iloc[:, 1:])

    violin_scatter_box(
        pivot.iloc[:, 1:],
        out_path=cfg.out_dir / f"fig3_primary_branch_corr_by_{aim_label}.eps",
        violin_palette=violin_palette,
        fig_size=(8, 6),
    )


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    # A) proportions
    run_proportion_plots(CFG)

    # B) correlation compare (choose one)
    run_correlation_group_compare(CFG, aim_y="cosine_corr", aim_label="dynamic_label")
    # run_correlation_group_compare(CFG, aim_y="cosine_corr", aim_label="area_label")
    # run_correlation_group_compare(CFG, aim_y="cosine_corr", aim_label="duration_label")


if __name__ == "__main__":
    main()
