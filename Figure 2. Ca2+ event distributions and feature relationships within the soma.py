# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 17:21:41 2024

Notes (cleaned):
- Code logic is unchanged.
- All comments are converted to English.
- All file I/O paths are anonymized (no personal/computer/drive info).
"""

# =============================================================================
# Imports
# =============================================================================
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import gaussian_kde
from matplotlib.patches import Rectangle, Wedge
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm, ListedColormap
from matplotlib.ticker import PercentFormatter

from significant_difference import (
    remove_outliers_df,
    number_of_cell,
    Mann_Whitney_U_test,
    concise_mannwhitney_analysis,
    concise_wilcoxon_analysis,
)

# =============================================================================
# Anonymized I/O paths (EDIT THESE TO YOUR REAL PATHS LOCALLY)
# =============================================================================
DATA_CSV_PATH = "data/young_mouse_all_events_with_loc_label_two_line.csv"
VAE_PICKLE_PATH = "data/vae_z_values_young_with_distance.pkl"

# Output folder (anonymized). Use your own output directory locally.
OUT_DIR = "outputs"

# =============================================================================
# Plot helpers
# =============================================================================
def violin_and_scatter_half_violin(
    df_values: pd.DataFrame,
    save_path: str,
    palette=None,
    start_0: bool = False,
    aim_y_lim: float = 0.1,
    expand_top_ratio: float = 0.1,
    expand_bottom_ratio: float = 0.0,
    figsize=(8, 6),
):
    """
    Half-violin + paired scatter + boxplot.
    - Keeps original plotting logic: half-violin clipping, scatter x-offset, paired lines.
    - `start_0` behavior is kept as in the original (set lower bound to aim_y_lim).
    - `expand_top_ratio` and `expand_bottom_ratio` allow matching your two variants.
    """
    if palette is None:
        palette = ["#9ac89a"] * df_values.shape[1]

    plt.figure(figsize=figsize)
    plt.rcParams["font.size"] = 12
    plt.rcParams["font.weight"] = "bold"

    # Violin (full), then clip to half (left half)
    ax = sns.violinplot(
        data=df_values,
        dodge=False,
        scale="width",
        inner=None,
        palette=palette,
        cut=0,
        clip_on=True,
    )

    # Clip each violin to half width
    for violin in ax.collections:
        bbox = violin.get_paths()[0].get_extents()
        x0, y0, width, height = bbox.bounds
        violin.set_clip_path(
            plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData)
        )

    # Scatter
    old_len = len(ax.collections)
    sns.stripplot(
        data=df_values,
        dodge=False,
        ax=ax,
        size=5,
        palette=["#d17d6c", "#9c8fb9", "#6c8d9d"],
    )

    # Offset scatter x positions to align with half-violin
    scatter_cols = ax.collections[old_len:]
    for col in scatter_cols:
        offs = col.get_offsets().copy()
        offs[:, 0] += 0.11
        col.set_offsets(offs)

    # Draw paired lines across groups by row index
    offsets_per_group = [col.get_offsets() for col in scatter_cols]
    n_samples = df_values.shape[0]
    for i in range(n_samples):
        xs = [offs[i, 0] for offs in offsets_per_group]
        ys = [offs[i, 1] for offs in offsets_per_group]
        ax.plot(xs, ys, color="gray", alpha=0.3, linewidth=0.4, zorder=1)

    # Boxplot overlay
    sns.boxplot(
        data=df_values,
        saturation=1,
        showfliers=False,
        width=0.3,
        boxprops={
            "zorder": 3,
            "facecolor": "none",
            "linewidth": 4,
            "edgecolor": "black",
        },
        whiskerprops={"linewidth": 4, "color": "black", "zorder": 10},
        capprops={"linewidth": 4, "color": "black", "zorder": 10},
        medianprops={"linewidth": 4, "color": "red", "zorder": 10},
        ax=ax,
    )

    # Y-limits adjustment (kept consistent with your two variants)
    current_ymin, current_ymax = ax.get_ylim()
    y_span = current_ymax - current_ymin
    new_ymax = current_ymax + expand_top_ratio * y_span
    new_ymin = current_ymin - expand_bottom_ratio * y_span

    if start_0:
        ax.set_ylim([aim_y_lim, new_ymax])
    else:
        ax.set_ylim([new_ymin, new_ymax])

    # Axis styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.set_xticks([])
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(save_path, format="eps", bbox_inches="tight")
    plt.show()


def violin_and_scatter_compact(
    df_values: pd.DataFrame,
    palette=None,
    start_0: bool = False,
    aim_y_lim: float = 0.0,
    figsize=(6, 2),
):
    """
    Compact version used in the later part of the script.
    Logic is identical to the compact original:
    - Half-violin clipping
    - Scatter x-offset
    - Paired lines
    - Boxplot overlay
    - No saving inside the function (save is done outside, same as original)
    """
    if palette is None:
        palette = ["#bfeebe", "#77a377"]

    plt.figure(figsize=figsize)
    plt.rcParams["font.size"] = 12
    plt.rcParams["font.weight"] = "bold"

    ax = sns.violinplot(
        data=df_values,
        dodge=False,
        scale="width",
        inner=None,
        palette=palette,
        cut=0,
        clip_on=True,
    )

    for violin in ax.collections:
        bbox = violin.get_paths()[0].get_extents()
        x0, y0, width, height = bbox.bounds
        violin.set_clip_path(
            plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData)
        )

    old_len = len(ax.collections)
    sns.stripplot(
        data=df_values,
        dodge=False,
        ax=ax,
        size=5,
        palette=["#d17d6c", "#9c8fb9", "#6c8d9d"],
    )

    scatter_cols = ax.collections[old_len:]
    for col in scatter_cols:
        offs = col.get_offsets().copy()
        offs[:, 0] += 0.11
        col.set_offsets(offs)

    offsets_per_group = [col.get_offsets() for col in scatter_cols]
    n_samples = df_values.shape[0]
    for i in range(n_samples):
        xs = [offs[i, 0] for offs in offsets_per_group]
        ys = [offs[i, 1] for offs in offsets_per_group]
        ax.plot(xs, ys, color="gray", alpha=0.3, linewidth=0.4, zorder=1)

    sns.boxplot(
        data=df_values,
        saturation=1,
        showfliers=False,
        width=0.3,
        boxprops={
            "zorder": 3,
            "facecolor": "none",
            "linewidth": 4,
            "edgecolor": "black",
        },
        whiskerprops={"linewidth": 4, "color": "black", "zorder": 10},
        capprops={"linewidth": 4, "color": "black", "zorder": 10},
        medianprops={"linewidth": 4, "color": "red", "zorder": 10},
        ax=ax,
    )

    current_ymin, current_ymax = ax.get_ylim()
    new_ymax = current_ymax + 0.1 * (current_ymax - current_ymin)
    if start_0:
        ax.set_ylim([aim_y_lim, new_ymax])
    else:
        ax.set_ylim([current_ymin, new_ymax])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.set_xticks([])
    ax.grid(False)

    plt.tight_layout()
    plt.show()


# =============================================================================
# Section 1: Event ratios by area/duration/dynamic state within a subcellular structure
# =============================================================================
# NOTE: The original comment mentioned extracting subcellular structures, but the shown code filters soma-like labels.
comtest = pd.read_csv(DATA_CSV_PATH)
comtest = comtest[
    (comtest.wt == 0) & (comtest.primary_label == -1) & (comtest.secondary_label == -1)
].reset_index(drop=True)
 

# Static/dynamic event flags
comtest["event_static"] = 0
mask_static = (comtest.Landmark_event_away_from_landmark_landmark_1 == 0) & (
    comtest.Landmark_event_toward_landmark_landmark_1 == 0
)
comtest.loc[mask_static, "event_static"] = 1
comtest["event_dynamic"] = 1 - comtest["event_static"]

# -----------------------------------------------------------------------------
# 1A) Plot calcium event ratio by area groups
# -----------------------------------------------------------------------------
cond1 = comtest["Basic_Area"] <= 10
cond2 = (comtest["Basic_Area"] > 10) & (comtest["Basic_Area"] < 100)
cond3 = comtest["Basic_Area"] >= 100
comtest["area_label"] = np.select([cond1, cond2, cond3], [1, 2, 3], default=0)

result = (
    comtest.groupby("cell_id")["area_label"]
    .value_counts(normalize=True)
    .unstack(fill_value=0)
    .reindex(columns=[1, 2, 3], fill_value=0)
    .rename(columns=lambda x: f"area_label_{x}")
    .reset_index()
)

result = remove_outliers_df(result)
number_of_cell(result)

result = result.dropna(axis=0)
concise_wilcoxon_analysis(result.iloc[:, 1:])

violin_and_scatter_half_violin(
    result.iloc[:, 1:],
    save_path=f"{OUT_DIR}/fig2_soma_area_group_event_ratio.eps",
    palette=["#bfeebe", "#bfeebe", "#bfeebe"],
    start_0=False,
    aim_y_lim=0.1,
    expand_top_ratio=0.1,
)

# -----------------------------------------------------------------------------
# 1B) Plot calcium event ratio by duration groups
# -----------------------------------------------------------------------------
cond1 = comtest["Curve_Duration_10_to_10"] <= 1
cond2 = (comtest["Curve_Duration_10_to_10"] > 1) & (comtest["Curve_Duration_10_to_10"] < 10)
cond3 = comtest["Curve_Duration_10_to_10"] >= 10
comtest["duration_label"] = np.select([cond1, cond2, cond3], [1, 2, 3], default=0)

result = (
    comtest.groupby("cell_id")["duration_label"]
    .value_counts(normalize=True)
    .unstack(fill_value=0)
    .reindex(columns=[1, 2, 3], fill_value=0)
    .rename(columns=lambda x: f"duration_label_{x}")
    .reset_index()
)

result = remove_outliers_df(result)
number_of_cell(result)

result = result.dropna(axis=0)
concise_wilcoxon_analysis(result.iloc[:, 1:])

violin_and_scatter_half_violin(
    result.iloc[:, 1:],
    save_path=f"{OUT_DIR}/fig2_soma_duration_group_event_ratio.eps",
    palette=["#77a377", "#77a377", "#77a377"],
    start_0=False,
    aim_y_lim=0.1,
    expand_top_ratio=0.1,
)

# -----------------------------------------------------------------------------
# 1C) Plot calcium event ratio by dynamic vs static groups
# -----------------------------------------------------------------------------
cond1 = comtest["event_dynamic"] == 0
cond2 = comtest["event_dynamic"] == 1
comtest["dynamic_label"] = np.select([cond1, cond2], [1, 2], default=0)

result = (
    comtest.groupby("cell_id")["dynamic_label"]
    .value_counts(normalize=True)
    .unstack(fill_value=0)
    .reindex(columns=[1, 2], fill_value=0)
    .rename(columns=lambda x: f"dynamic_label_{x}")
    .reset_index()
)

result = remove_outliers_df(result)
number_of_cell(result)

result = result.dropna(axis=0)
concise_wilcoxon_analysis(result.iloc[:, 1:])

violin_and_scatter_half_violin(
    result.iloc[:, 1:],
    save_path=f"{OUT_DIR}/fig2_soma_dynamic_group_event_ratio.eps",
    palette=["#bfeebe", "#77a377"],
    start_0=True,
    aim_y_lim=0.1,
    expand_top_ratio=0.1,
)

# =============================================================================
# Section 2: Correlation differences across groups
# =============================================================================
# Load VAE merged data (kept as original)
with open(VAE_PICKLE_PATH, "rb") as f:
    data = pickle.load(f)["merged_df"]

# Attach event_dynamic for signal A/B (kept as original logic)
df_dyn = pd.read_csv(
    DATA_CSV_PATH,
    usecols=[
        "cell_id",
        "Landmark_event_away_from_landmark_landmark_1",
        "Landmark_event_toward_landmark_landmark_1",
    ],
)
df_dyn["event_static"] = 0
mask_static = (df_dyn.Landmark_event_away_from_landmark_landmark_1 == 0) & (
    df_dyn.Landmark_event_toward_landmark_landmark_1 == 0
)
df_dyn.loc[mask_static, "event_static"] = 1
df_dyn["event_dynamic"] = 1 - df_dyn["event_static"]
df_dyn["signal_a_order"] = df_dyn.groupby("cell_id").cumcount()
df_dyn["signal_b_order"] = df_dyn["signal_a_order"]

data = data.merge(df_dyn[["cell_id", "signal_a_order", "event_dynamic"]], on=["cell_id", "signal_a_order"], how="left")
data = data.rename(columns={"event_dynamic": "singal_a_event_dynamic"})
data = data.merge(df_dyn[["cell_id", "signal_b_order", "event_dynamic"]], on=["cell_id", "signal_b_order"], how="left")
data = data.rename(columns={"event_dynamic": "singal_b_event_dynamic"})

# Attach duration for signal A/B
df_dur = pd.read_csv(DATA_CSV_PATH, usecols=["cell_id", "Curve_Duration_10_to_10"]) 
df_dur["signal_a_order"] = df_dur.groupby("cell_id").cumcount()
df_dur["signal_b_order"] = df_dur["signal_a_order"]

data = data.merge(df_dur[["cell_id", "signal_a_order", "Curve_Duration_10_to_10"]], on=["cell_id", "signal_a_order"], how="left")
data = data.rename(columns={"Curve_Duration_10_to_10": "singal_a_Curve_Duration_10_to_10"})
data = data.merge(df_dur[["cell_id", "signal_b_order", "Curve_Duration_10_to_10"]], on=["cell_id", "signal_b_order"], how="left")
data = data.rename(columns={"Curve_Duration_10_to_10": "singal_b_Curve_Duration_10_to_10"})

# Attach area for signal A/B
df_area = pd.read_csv(DATA_CSV_PATH, usecols=["cell_id", "Basic_Area"]) 
df_area["signal_a_order"] = df_area.groupby("cell_id").cumcount()
df_area["signal_b_order"] = df_area["signal_a_order"]

data = data.merge(df_area[["cell_id", "signal_a_order", "Basic_Area"]], on=["cell_id", "signal_a_order"], how="left")
data = data.rename(columns={"Basic_Area": "singal_a_Basic_Area"})
data = data.merge(df_area[["cell_id", "signal_b_order", "Basic_Area"]], on=["cell_id", "signal_b_order"], how="left")
data = data.rename(columns={"Basic_Area": "singal_b_Basic_Area"})

# Attach location labels for signal A/B
df_loc = pd.read_csv(DATA_CSV_PATH, usecols=["cell_id", "primary_label", "secondary_label"])
df_loc["signal_a_order"] = df_loc.groupby("cell_id").cumcount()
df_loc["signal_b_order"] = df_loc["signal_a_order"]

data = data.merge(df_loc[["cell_id", "signal_a_order", "primary_label"]], on=["cell_id", "signal_a_order"], how="left")
data = data.rename(columns={"primary_label": "singal_a_primary_label"})
data = data.merge(df_loc[["cell_id", "signal_a_order", "secondary_label"]], on=["cell_id", "signal_a_order"], how="left")
data = data.rename(columns={"secondary_label": "singal_a_secondary_label"})

data = data.merge(df_loc[["cell_id", "signal_b_order", "primary_label"]], on=["cell_id", "signal_b_order"], how="left")
data = data.rename(columns={"primary_label": "singal_b_primary_label"})
data = data.merge(df_loc[["cell_id", "signal_b_order", "secondary_label"]], on=["cell_id", "signal_b_order"], how="left")
data = data.rename(columns={"secondary_label": "singal_b_secondary_label"})

# Create dynamic_label / duration_label / area_label (paired constraints kept)
a_col = "singal_a_event_dynamic"
b_col = "singal_b_event_dynamic"
cond1 = (data[a_col] == 0) & (data[b_col] == 0)
cond2 = (data[a_col] == 1) & (data[b_col] == 1)
data["dynamic_label"] = np.select([cond1, cond2], [1, 2], default=0)

a_col = "singal_a_Curve_Duration_10_to_10"
b_col = "singal_b_Curve_Duration_10_to_10"
cond1 = (data[a_col] <= 1) & (data[b_col] <= 1)
cond2 = (data[a_col] > 1) & (data[a_col] < 10) & (data[b_col] > 1) & (data[b_col] < 10)
cond3 = (data[a_col] >= 10) & (data[b_col] >= 10)
data["duration_label"] = np.select([cond1, cond2, cond3], [1, 2, 3], default=0)

a_col = "singal_a_Basic_Area"
b_col = "singal_b_Basic_Area"
cond1 = (data[a_col] <= 10) & (data[b_col] <= 10)
cond2 = (data[a_col] > 10) & (data[a_col] < 100) & (data[b_col] > 10) & (data[b_col] < 100)
cond3 = (data[a_col] >= 100) & (data[b_col] >= 100)
data["area_label"] = np.select([cond1, cond2, cond3], [1, 2, 3], default=0)

cond1 = (
    (data["singal_a_primary_label"] == -1)
    & (data["singal_a_secondary_label"] == -1)
    & (data["singal_b_primary_label"] == -1)
    & (data["singal_b_secondary_label"] == -1)
)
cond2 = (
    (data["singal_a_primary_label"] > 0)
    & (data["singal_a_secondary_label"] == -1)
    & (data["singal_b_primary_label"] > 0)
    & (data["singal_b_secondary_label"] == -1)
)
cond3 = (
    (data["singal_a_primary_label"] >= 0)
    & (data["singal_a_secondary_label"] >= 0)
    & (data["singal_b_primary_label"] >= 0)
    & (data["singal_b_secondary_label"] >= 0)
)
data["loc_label"] = np.select([cond1, cond2, cond3], [1, 2, 3], default=0)

# Keep only loc_label==1 (same as original)
data = data[data.loc_label == 1].reset_index(drop=True)

aim_y = "cosine_corr"
aim_label = "dynamic_label"  # options: 'area_label', 'duration_label', 'dynamic_label'

if aim_label == "area_label":
    palette = ["#bfeebe", "#bfeebe", "#bfeebe"]
elif aim_label == "duration_label":
    palette = ["#77a377", "#77a377", "#77a377"]
elif aim_label == "dynamic_label":
    # Note: original had a typo '#9ac819a'; kept palette intent but corrected to valid hex.
    palette = ["#9ac89a", "#9ac89a", "#9ac89a"]

cell_label_median = data.groupby(["cell_id", aim_label])[aim_y].median().reset_index()
cell_label_median = cell_label_median[cell_label_median[aim_label] > 0].reset_index(drop=True)

cell_label_median_pivot = (
    cell_label_median.pivot(index="cell_id", columns=aim_label, values="cosine_corr")
    .reset_index()
)
cell_label_median_pivot.columns = [
    f"loc_{col}" if isinstance(col, int) else col for col in cell_label_median_pivot.columns
]

cell_label_median_pivot = remove_outliers_df(cell_label_median_pivot)
number_of_cell(cell_label_median_pivot)

cell_label_median_pivot = cell_label_median_pivot.dropna(axis=0)
concise_wilcoxon_analysis(cell_label_median_pivot.iloc[:, 1:])

violin_and_scatter_half_violin(
    cell_label_median_pivot.iloc[:, 1:],
    save_path=f"{OUT_DIR}/fig2_soma_correlation_by_{aim_label}.eps",
    palette=palette,
    start_0=False,
    aim_y_lim=0.0,
    expand_top_ratio=0.2,     # match the correlation-plot variant
    expand_bottom_ratio=0.07, # match the correlation-plot variant
    figsize=(8, 6),
)

# =============================================================================
# Section 3: Parameter dependency heatmaps / distributions
# =============================================================================
# Load and preprocess data (kept same logic)
usecols = [
    "mouse_id",
    "cell_id",
    "Basic_Area",
    "Curve_Duration_10_to_10",
    "wt",
    "primary_label",
    "secondary_label",
    "Landmark_event_away_from_landmark_landmark_1",
    "Landmark_event_toward_landmark_landmark_1",
]
comtest = pd.read_csv(DATA_CSV_PATH, usecols=usecols)

comtest = comtest[
    (comtest.wt == 0) & (comtest.primary_label == -1) & (comtest.secondary_label == -1)
].reset_index(drop=True)

comtest["event_static"] = 0
mask_static = (comtest["Landmark_event_away_from_landmark_landmark_1"] == 0) & (
    comtest["Landmark_event_toward_landmark_landmark_1"] == 0
)
comtest.loc[mask_static, "event_static"] = 1
comtest["event_dynamic"] = 1 - comtest["event_static"]
 
comtest.loc[:, ["Basic_Area", "Curve_Duration_10_to_10"]] = remove_outliers_df(
    comtest.loc[:, ["Basic_Area", "Curve_Duration_10_to_10"]]
)

# -----------------------------------------------------------------------------
# 3A) 2D KDE joint density (with reflection method as original block)
# -----------------------------------------------------------------------------
x_param = "Basic_Area"
y_param = "Curve_Duration_10_to_10"

comtest_clean = comtest[[x_param, y_param]].copy()
comtest_clean = comtest_clean.replace([np.inf, -np.inf], np.nan).dropna()

EPS = 1e-6
x_data = comtest_clean[x_param].values + EPS
y_data = comtest_clean[y_param].values + EPS

def reflect_data_positive(arr):
    """Reflect values greater than EPS to reduce boundary effects."""
    return np.concatenate([arr, EPS - arr[arr > EPS]])

x_reflected = reflect_data_positive(x_data)
y_reflected = reflect_data_positive(y_data)

print(f"Original points: {len(x_data)}, after reflection: {len(x_reflected)}")
print(f"X range: [{x_reflected.min():.2e}, {x_reflected.max():.2e}]")
print(f"Y range: [{y_reflected.min():.2e}, {y_reflected.max():.2e}]")

try:
    kde = gaussian_kde([x_reflected, y_reflected])
    n, d = len(x_reflected), 2
    scott_factor = n ** (-1.0 / (d + 4))
    kde.set_bandwidth(bw_method=kde.factor * min(scott_factor, 0.8))
except np.linalg.LinAlgError:
    print("Singular covariance; fallback to diagonal covariance.")
    cov = np.cov([x_reflected, y_reflected])
    cov = np.diag(np.diag(cov))
    kde = gaussian_kde([x_reflected, y_reflected], bw_method="scott")
    kde.covariance = cov
    kde.inv_cov = np.linalg.inv(cov)

xmax = x_data.max() * 1.05
ymax = y_data.max() * 1.05
xx, yy = np.mgrid[0:xmax:200j, 0:ymax:200j]
positions = np.vstack([xx.ravel(), yy.ravel()])
Z = kde(positions).reshape(xx.shape) * 4

fig, ax = plt.subplots(figsize=(8, 8))
N_levels = 2
levels = np.linspace(Z.min(), Z.max(), N_levels + 1)

ax.contourf(xx, yy, Z, levels=levels, cmap="plasma", extend="neither", alpha=0.8)

sample_idx = np.random.choice(len(comtest_clean), size=min(200, len(comtest_clean)), replace=False)
sample = comtest_clean.iloc[sample_idx]
ax.scatter(
    sample[x_param],
    sample[y_param],
    s=15,
    color="white",
    edgecolor="k",
    alpha=0.6,
    linewidth=0.5,
)

ax.set_xlabel(x_param)
ax.set_ylabel(y_param)
ax.grid(True, linestyle="--", alpha=0.3)
ax.set_xlim(0, xmax)
ax.set_ylim(0, ymax)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.show()

plt.savefig(
    f"{OUT_DIR}/fig2_soma_param_dependency_{x_param}_and_{y_param}.eps",
    format="eps",
    bbox_inches="tight",
)

# -----------------------------------------------------------------------------
# 3B) Density-region percentage pie charts (kept as original)
# -----------------------------------------------------------------------------
x = comtest[x_param].values
y = comtest[y_param].values
valid_mask = np.isfinite(x) & np.isfinite(y)
x_clean = x[valid_mask]
y_clean = y[valid_mask]
print(f"Valid points: {len(x_clean)}")

xy = np.vstack([x_clean, y_clean])
kde2 = gaussian_kde(xy)

xmin, xmax = x_clean.min(), x_clean.max()
ymin, ymax = y_clean.min(), y_clean.max()
X, Y = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
positions = np.vstack([X.ravel(), Y.ravel()])
Z2 = np.reshape(kde2(positions), X.shape)

dx = X[1, 0] - X[0, 0]
dy = Y[0, 1] - Y[0, 0]
total_integral = np.sum(Z2) * dx * dy
print(f"Total density integral: {total_integral:.4f}")

Z_normalized = Z2 / Z2.max()
density_levels = np.linspace(0, 1, 9)

percentages = []
level_labels = []
level_colors = []

for i in range(len(density_levels) - 1):
    low = density_levels[i]
    high = density_levels[i + 1]
    mask = (Z_normalized >= low) & (Z_normalized < high)
    volume = np.sum(Z2[mask]) * dx * dy
    percentage = 100 * volume / total_integral
    percentages.append(percentage)

    mid_density = (low + high) / 2
    level_colors.append(plt.get_cmap("plasma")(mid_density))
    level_labels.append(f"{low:.1f}-{high:.1f}")

total_percentage = sum(percentages)
if abs(total_percentage - 100) > 1e-6:
    scale = 100 / total_percentage
    percentages = [p * scale for p in percentages]
    print(f"Adjusted total: {sum(percentages):.2f}%")

# Pie chart with percentage labels
plt.figure(figsize=(10, 8))
wedges, texts, autotexts = plt.pie(
    percentages,
    labels=level_labels,
    colors=level_colors,
    startangle=90,
    autopct="%1.1f%%",
    counterclock=False,
    pctdistance=0.85,
    textprops={"fontsize": 10, "color": "white", "weight": "bold"},
    wedgeprops={"edgecolor": "white", "linewidth": 1, "alpha": 0.9},
)
plt.gca().add_artist(plt.Circle((0, 0), 0.50, fc="white"))
plt.axis("equal")
plt.tight_layout()
plt.show()

# Pie chart without percentage labels (as in original second pie plot)
plt.figure(figsize=(10, 8))
plt.pie(
    percentages,
    labels=level_labels,
    colors=level_colors,
    startangle=90,
    counterclock=False,
    textprops={"fontsize": 10, "color": "white", "weight": "bold"},
    wedgeprops={"edgecolor": "white", "linewidth": 1, "alpha": 0.9},
)
plt.gca().add_artist(plt.Circle((0, 0), 0.50, fc="white"))
plt.axis("equal")
plt.tight_layout()
plt.show()

plt.savefig(
    f"{OUT_DIR}/fig2_soma_param_dependency_density_pie.eps",
    format="eps",
    bbox_inches="tight",
)

# -----------------------------------------------------------------------------
# 3C) Histogram + trend line for each parameter (Freedman–Diaconis rule; kept)
# -----------------------------------------------------------------------------
for aim_param in [x_param, y_param]:
    param_values = comtest[aim_param].dropna().values.astype(float)
    color = "#bfeebe" if aim_param == x_param else "#77a377"

    q25, q75 = np.percentile(param_values, [25, 75])
    iqr = q75 - q25
    n = len(param_values)
    bin_width = 2 * iqr / np.cbrt(n)

    min_val, max_val = np.min(param_values), np.max(param_values)
    if bin_width == 0:
        bin_width = (max_val - min_val) / 10
    num_bins = int(np.ceil((max_val - min_val) / bin_width))
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)

    binned = pd.cut(param_values, bins=bin_edges, include_lowest=True)
    counts = binned.value_counts().sort_index()
    value_counts = (counts / counts.sum()) * 100

    df_counts = value_counts.reset_index()
    df_counts.columns = ["bin", "percentage"]
    df_counts["bin_center"] = df_counts["bin"].apply(lambda x: x.mid)

    fig, ax = plt.subplots(figsize=(6, 2))
    bar_width = bin_width * 0.9
    bars = ax.bar(
        df_counts["bin_center"],
        df_counts["percentage"],
        width=bar_width,
        color=color,
        edgecolor="black",
        linewidth=0.8,
    )

    bar_centers = [bar.get_x() + bar.get_width() / 2 for bar in bars]
    bar_heights = [bar.get_height() for bar in bars]

    ax.plot(
        bar_centers,
        bar_heights,
        "-o",
        markeredgecolor="black",
        markeredgewidth=4.5,
        color="black",
        linewidth=5,
        zorder=1,
    )
    ax.plot(
        bar_centers,
        bar_heights,
        "-o",
        color=color,
        linewidth=2,
        markersize=6,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=1.5,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()

    plt.savefig(
        f"{OUT_DIR}/fig2_soma_{aim_param}_hist_and_curve.eps",
        format="eps",
        bbox_inches="tight",
    )

# =============================================================================
# Section 4: Dual distribution (static vs dynamic) + stats + compact violin
# =============================================================================
def _fd_edges(x, max_bins=200, fallback_bins=10):
    """Freedman–Diaconis equal-width bin edges; includes fallback for degenerate cases."""
    x = pd.Series(x).dropna().astype(float).values
    if x.size == 0:
        raise ValueError("Empty data.")
    xmin, xmax = np.min(x), np.max(x)
    span = xmax - xmin
    n = x.size
    if span == 0 or n < 2:
        eps = 1e-9 if xmin == 0 else abs(xmin) * 1e-9
        return np.array([xmin - eps, xmax + eps]), 1, 2 * eps

    q25, q75 = np.percentile(x, [25, 75])
    iqr = q75 - q25
    if not np.isfinite(iqr) or iqr <= 0:
        k = fallback_bins
    else:
        h = 2.0 * iqr / np.cbrt(n)
        if not np.isfinite(h) or h <= 0:
            k = fallback_bins
        else:
            k = int(np.ceil(span / h))
    k = int(np.clip(k, 1, max_bins))
    edges = np.linspace(xmin, xmax, k + 1)
    h = span / k
    return edges, k, h


def plot_dual_distribution(
    aim_param_A,
    aim_param_B,
    color_A="#bfeebe",
    color_B="#77a377",
    line_color_A="#2ca02c",
    line_color_B="#d62728",
    num_bins=None,
    max_bins=200,
):
    """
    Plot two distributions in one figure:
    - Each series is converted to % histogram on a shared bin grid
    - FD bins are aligned (union) unless `num_bins` is specified
    """
    A = pd.Series(aim_param_A).dropna().astype(float).values
    B = pd.Series(aim_param_B).dropna().astype(float).values
    if A.size == 0 or B.size == 0:
        raise ValueError("A or B is empty.")

    if num_bins is not None:
        combined = np.concatenate([A, B])
        xmin, xmax = np.min(combined), np.max(combined)
        edges = np.linspace(xmin, xmax, int(max(1, num_bins)) + 1)
    else:
        edges_A, _, _ = _fd_edges(A, max_bins=max_bins)
        edges_B, _, _ = _fd_edges(B, max_bins=max_bins)
        edges = np.unique(np.concatenate([edges_A, edges_B]))
        if edges.size < 2:
            xmin, xmax = float(edges[0]), float(edges[0]) + 1e-9
            edges = np.array([xmin, xmax])

    counts_A, _ = np.histogram(A, bins=edges)
    counts_B, _ = np.histogram(B, bins=edges)
    pct_A = counts_A / counts_A.sum() * 100 if counts_A.sum() > 0 else np.zeros_like(counts_A, dtype=float)
    pct_B = counts_B / counts_B.sum() * 100 if counts_B.sum() > 0 else np.zeros_like(counts_B, dtype=float)

    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = edges[1:] - edges[:-1]

    subw = widths * 0.45
    xA = centers - widths * 0.25
    xB = centers + widths * 0.25

    fig, ax = plt.subplots(figsize=(6, 2))

    ax.bar(
        xA,
        pct_A,
        width=subw,
        color=color_A,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.8,
        label="Static Events",
        align="center",
    )
    ax.bar(
        xB,
        pct_B,
        width=subw,
        color=color_B,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.8,
        label="Dynamic Events",
        align="center",
    )

    ax.plot(
        xA,
        pct_A.tolist(),
        "o-",
        markeredgecolor="black",
        markeredgewidth=4.5,
        color="black",
        linewidth=5,
        zorder=1,
    )
    ax.plot(
        xA,
        pct_A.tolist(),
        "o-",
        color=line_color_A,
        linewidth=2,
        markersize=6,
        markerfacecolor="white",
        markeredgecolor=line_color_A,
        markeredgewidth=1.5,
    )

    ax.plot(
        xB,
        pct_B.tolist(),
        "s-",
        markeredgecolor="black",
        markeredgewidth=4.5,
        color="black",
        linewidth=5,
        zorder=1,
    )
    ax.plot(
        xB,
        pct_B.tolist(),
        "s-",
        color=line_color_B,
        linewidth=2,
        markersize=6,
        markerfacecolor="white",
        markeredgecolor=line_color_B,
        markeredgewidth=1.5,
    )

    ax.yaxis.set_major_formatter(PercentFormatter(100))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    xmin, xmax = edges[0], edges[-1]
    span = xmax - xmin
    pad = 0.03 * span if span > 0 else 1.0
    ax.set_xlim(xmin - pad, xmax + pad)

    ymin, ymax = ax.get_ylim()
    ax.set_ylim([ymin - 0.05 * (ymax - ymin), ymax + 0.1 * (ymax - ymin)])

    plt.tight_layout()
    return fig, ax


aim_param = "Curve_Duration_10_to_10"  # or "Basic_Area"
static_vals = comtest.loc[comtest.event_dynamic == 0, aim_param]
dynamic_vals = comtest.loc[comtest.event_dynamic == 1, aim_param]

fig, ax = plot_dual_distribution(
    static_vals,
    dynamic_vals,
    color_A="#bfeebe",
    color_B="#77a377",
    line_color_A="#bfeebe",
    line_color_B="#77a377",
    num_bins=20,
)
plt.show()
plt.savefig(
    f"{OUT_DIR}/fig2_soma_static_vs_dynamic_{aim_param}_hist_curve.eps",
    format="eps",
    bbox_inches="tight",
)

# Paired per-cell median comparison (kept)
_pro_ = (
    comtest[comtest.event_dynamic == 0][[aim_param, "cell_id"]]
    .groupby("cell_id")
    .median()
    .reset_index()
    .merge(
        comtest[comtest.event_dynamic == 1][[aim_param, "cell_id"]]
        .groupby("cell_id")
        .median()
        .reset_index(),
        on=["cell_id"],
        how="left",
    )
)
_pro_ = remove_outliers_df(_pro_)
number_of_cell(_pro_)

concise_wilcoxon_analysis(_pro_.iloc[:, 1:].dropna())
concise_mannwhitney_analysis(_pro_.iloc[:, 1:])
Mann_Whitney_U_test(
    data1=_pro_.iloc[:, 1].dropna(),
    data2=_pro_.iloc[:, 2].dropna(),
)

# Compact violin (kept behavior)
violin_and_scatter_compact(_pro_.iloc[:, 1:], start_0=True)
plt.savefig(
    f"{OUT_DIR}/fig2_soma_static_vs_dynamic_{aim_param}_violin.eps",
    format="eps",
    bbox_inches="tight",
)

# =============================================================================
# Section 5: Fan-shaped wedge plot (sector plot)
# =============================================================================
num_radial = 3
num_angular = 6
radii = np.linspace(0, 1.0, num_radial + 1)
angles = np.linspace(0, 180, num_angular + 1)

# Data (contains NaNs; kept)
sector_data = [
    np.nan, 0.95265, 0.97755, 0.167385, 0.923387, 0.167385,
    0.831633, 0.513937, 0.460023, 0.358811, 0.574744, np.nan,
    0.937675, 0.728945, 0.796726, 0.635781, 0.320374, np.nan
]

fig1, ax1 = plt.subplots(figsize=(8, 5))
normal_patches = []
normal_values = []

for idx, val in enumerate(sector_data):
    i = idx // num_angular
    j = idx % num_angular

    wedge = Wedge(
        center=(0, 0),
        r=radii[i + 1],
        theta1=angles[j],
        theta2=angles[j + 1],
        width=radii[i + 1] - radii[i],
    )

    if np.isnan(val):
        ax1.add_patch(
            Wedge(
                center=(0, 0),
                r=radii[i + 1],
                theta1=angles[j],
                theta2=angles[j + 1],
                width=radii[i + 1] - radii[i],
                facecolor="lightgray",
                edgecolor="black",
                hatch="///",
                linewidth=0.8,
            )
        )
    else:
        normal_patches.append(wedge)
        normal_values.append(val)

colors = [
    (209 / 255, 210 / 255, 219 / 255),
    "#bfeebe",
    "#81ad80",
    "#5b865c",
]
soft_cmap = LinearSegmentedColormap.from_list("soft_cmap", colors, N=4)

collection = PatchCollection(normal_patches, cmap=soft_cmap, edgecolor="black", linewidth=2)
collection.set_clim(0, 1)
collection.set_array(np.array(normal_values))
ax1.add_collection(collection)

ax1.set_xlim(-1.1, 1.1)
ax1.set_ylim(0, 1.1)
ax1.set_aspect("equal")
ax1.axis("off")
plt.tight_layout()

plt.savefig(
    f"{OUT_DIR}/fig2_soma_sector_plot.eps",
    format="eps",
    bbox_inches="tight",
)

# Separate colorbar figure
fig2, ax2 = plt.subplots(figsize=(3, 8))
fig2.subplots_adjust(left=0.5)
ax2.axis("off")
cbar = fig2.colorbar(collection, ax=ax2, orientation="vertical", fraction=0.1)
cbar.set_label("Value")

plt.tight_layout()
plt.show()
plt.savefig(
    f"{OUT_DIR}/fig2_soma_sector_plot_colorbar.eps",
    format="eps",
    bbox_inches="tight",
)
