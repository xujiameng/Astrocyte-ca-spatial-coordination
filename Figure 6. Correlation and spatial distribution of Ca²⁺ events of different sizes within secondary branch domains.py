# -*- coding: utf-8 -*-

import gc
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap


# -----------------------
# Define plotting function
def violin_and_scatter(loc_perc, save_path, start_0=False, aim_y_lim=0):
    plt.figure(figsize=(6, 6))
    plt.rcParams.update({"font.size": 12, "font.weight": "bold"})

    # Draw violin plot
    ax = sns.violinplot(
        data=loc_perc,
        dodge=False,
        scale="width",
        inner=None,
        palette=["#c7e1e6", "#91b5bb", "#5c8b94"],
        cut=0,
        clip_on=True,
    )

    # Adjust violin plot: show only half for each violin
    for violin in ax.collections:
        if violin.get_paths():
            bbox = violin.get_paths()[0].get_extents()
            x0, y0, width, height = bbox.bounds
            violin.set_clip_path(
                plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData)
            )

    old_len_collections = len(ax.collections)
    sns.stripplot(
        data=loc_perc,
        dodge=False,
        ax=ax,
        size=5,
        palette=["#e67a6e", "#8a7ca4", "#4a726b"],
    )

    # # Offset scatter points
    # for dots in ax.collections[old_len_collections:]:
    #     dots.set_offsets(dots.get_offsets() + np.array([0.11, 0]))

    # Extract newly-added PathCollections (one per group)
    scatter_cols = ax.collections[old_len_collections:]

    # 3. Manually apply a unified X offset (to match the half-violin layout)
    for col in scatter_cols:
        offs = col.get_offsets().copy()
        offs[:, 0] += 0.11
        col.set_offsets(offs)

    # 4. Draw paired connecting lines (row-wise pairing across groups)
    offsets_per_group = [col.get_offsets() for col in scatter_cols]
    n_samples = loc_perc.shape[0]
    for i in range(n_samples):
        xs = [offs[i, 0] for offs in offsets_per_group]
        ys = [offs[i, 1] for offs in offsets_per_group]
        ax.plot(xs, ys, color="gray", alpha=0.3, linewidth=0.4, zorder=1)

    sns.boxplot(
        data=loc_perc,
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

    # Adjust y-axis range (core part)
    current_ymin, current_ymax = ax.get_ylim()
    new_ymax = current_ymax + 0.2 * (current_ymax - current_ymin)  # expand upper bound
    if start_0:
        ax.set_ylim([aim_y_lim, new_ymax])
    else:
        ax.set_ylim([current_ymin, new_ymax])  # keep lower bound unchanged

    # ax.set_ylim([current_ymin, new_ymax])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.set_xticks([])  # hide x-axis tick labels
    ax.grid(False)  # do not draw grid lines

    plt.tight_layout()
    plt.savefig(save_path, format="eps", bbox_inches="tight")
    plt.show()


# -----------------------
# 1. Load data from pickle and drop unused columns
PKL_PATH = "<DATA_DIR>/vae_z_values_young_with_distance.pkl"
# PKL_PATH = "<DATA_DIR>/vae_z_values_young_with_distance_1_point_1.pkl"

with open(PKL_PATH, "rb") as f:
    data = pickle.load(f)["merged_df"]

cols_to_drop = [
    "signal_a_start_point_x",
    "signal_a_start_point_y",
    "signal_a_start_frame",
    "signal_b_start_point_x",
    "signal_b_start_point_y",
    "signal_b_start_frame",
    "correlation_matrix",
]
data.drop(columns=cols_to_drop, inplace=True)

# -----------------------
# 2. Merge 1: load calcium event base info CSV (only needed columns) and convert dtypes
CSV_PATH = "<DATA_DIR>/young_mouse_all_events_with_loc_label_two_line.csv"

df1 = pd.read_csv(
    CSV_PATH,
    usecols=[
        "cell_id",
        "Landmark_event_away_from_landmark_landmark_1",
        "Landmark_event_toward_landmark_landmark_1",
    ],
)

# df1["cell_id"] = df1["cell_id"].astype(np.int32)

# Convert event type to 0/1
df1["event_static"] = (
    (df1["Landmark_event_away_from_landmark_landmark_1"] == 0)
    & (df1["Landmark_event_toward_landmark_landmark_1"] == 0)
).astype(np.int8)
df1["event_dynamic"] = (1 - df1["event_static"]).astype(np.int8)

df1["signal_a_order"] = df1.groupby("cell_id").cumcount().astype(np.int32)
df1["signal_b_order"] = df1["signal_a_order"]

data = data.merge(
    df1[["cell_id", "signal_a_order", "event_dynamic"]],
    on=["cell_id", "signal_a_order"],
    how="left",
)
data.rename(columns={"event_dynamic": "singal_a_event_dynamic"}, inplace=True)

data = data.merge(
    df1[["cell_id", "signal_b_order", "event_dynamic"]],
    on=["cell_id", "signal_b_order"],
    how="left",
)
data.rename(columns={"event_dynamic": "singal_b_event_dynamic"}, inplace=True)

del df1
gc.collect()

# -----------------------
# 3. Merge 2: load branch labels and convert to memory-friendly dtypes
df2 = pd.read_csv(CSV_PATH, usecols=["cell_id", "primary_label", "secondary_label"])

# df2["cell_id"] = df2["cell_id"].astype(np.int32)

# Convert primary/secondary labels to integers to avoid float precision artifacts
df2["primary_label"] = df2["primary_label"].round(0).astype(np.int32)
df2["secondary_label"] = df2["secondary_label"].round(0).astype(np.int32)

df2["signal_a_order"] = df2.groupby("cell_id").cumcount().astype(np.int32)
df2["signal_b_order"] = df2["signal_a_order"]

data = data.merge(
    df2[["cell_id", "signal_a_order", "primary_label"]],
    on=["cell_id", "signal_a_order"],
    how="left",
)
data.rename(columns={"primary_label": "singal_a_primary_label"}, inplace=True)

data = data.merge(
    df2[["cell_id", "signal_a_order", "secondary_label"]],
    on=["cell_id", "signal_a_order"],
    how="left",
)
data.rename(columns={"secondary_label": "singal_a_secondary_label"}, inplace=True)

data = data.merge(
    df2[["cell_id", "signal_b_order", "primary_label"]],
    on=["cell_id", "signal_b_order"],
    how="left",
)
data.rename(columns={"primary_label": "singal_b_primary_label"}, inplace=True)

data = data.merge(
    df2[["cell_id", "signal_b_order", "secondary_label"]],
    on=["cell_id", "signal_b_order"],
    how="left",
)
data.rename(columns={"secondary_label": "singal_b_secondary_label"}, inplace=True)

del df2
gc.collect()

# -----------------------
# 4. Merge 3: load curve duration and apply scalar transform
df3 = pd.read_csv(CSV_PATH, usecols=["cell_id", "Curve_Duration_10_to_10"])

# df3["cell_id"] = df3["cell_id"].astype(np.int32)

df3["Curve_Duration_10_to_10"] = df3["Curve_Duration_10_to_10"] * 2 / 5
df3["signal_a_order"] = df3.groupby("cell_id").cumcount().astype(np.int32)
df3["signal_b_order"] = df3["signal_a_order"]

data = data.merge(
    df3[["cell_id", "signal_a_order", "Curve_Duration_10_to_10"]],
    on=["cell_id", "signal_a_order"],
    how="left",
)
data.rename(
    columns={"Curve_Duration_10_to_10": "singal_a_Curve_Duration_10_to_10"},
    inplace=True,
)

data = data.merge(
    df3[["cell_id", "signal_b_order", "Curve_Duration_10_to_10"]],
    on=["cell_id", "signal_b_order"],
    how="left",
)
data.rename(
    columns={"Curve_Duration_10_to_10": "singal_b_Curve_Duration_10_to_10"},
    inplace=True,
)

del df3
gc.collect()

# -----------------------
# 5. Merge 4: load Basic_Area and apply conversion
df4 = pd.read_csv(CSV_PATH, usecols=["cell_id", "Basic_Area"])

# df4["cell_id"] = df4["cell_id"].astype(np.int32)
 

df4["signal_a_order"] = df4.groupby("cell_id").cumcount().astype(np.int32)
df4["signal_b_order"] = df4["signal_a_order"]

data = data.merge(
    df4[["cell_id", "signal_a_order", "Basic_Area"]],
    on=["cell_id", "signal_a_order"],
    how="left",
)
data.rename(columns={"Basic_Area": "singal_a_Basic_Area"}, inplace=True)

data = data.merge(
    df4[["cell_id", "signal_b_order", "Basic_Area"]],
    on=["cell_id", "signal_b_order"],
    how="left",
)
data.rename(columns={"Basic_Area": "singal_b_Basic_Area"}, inplace=True)

del df4
gc.collect()

# -----------------------
# 6. Discuss primary vs secondary branch relationship
# NOTE: cast primary_label to int to avoid extra unique values from float precision
data["singal_a_primary_label"] = data["singal_a_primary_label"].astype(np.int32)
data["singal_b_primary_label"] = data["singal_b_primary_label"].astype(np.int32)

# Filter according to conditions
data_copy = data.copy()

# data = data_copy.copy()
# -----------------------  Discuss primary branch relationship
# data = data[
#     (data["singal_a_primary_label"] > 0)
#     & (data["singal_a_secondary_label"] == 0)
#     & (data["singal_b_primary_label"] > 0)
#     & (data["singal_b_secondary_label"] == 0)
# ]
# data_same_primary_branch = data[data["singal_a_primary_label"] == data["singal_b_primary_label"]]
# data_same_primary_branch["loc_label"] = 1
# data_diff_primary_branch = data[data["singal_a_primary_label"] != data["singal_b_primary_label"]]
# data_diff_primary_branch["loc_label"] = 2
# data = pd.concat([data_same_primary_branch, data_diff_primary_branch], axis=0)
# del data_same_primary_branch, data_diff_primary_branch

# -----------------------  Discuss secondary branch relationship
data = data[
    (data["singal_a_primary_label"] > 0)
    & (data["singal_a_secondary_label"] >= 0)
    & (data["singal_b_primary_label"] > 0)
    & (data["singal_b_secondary_label"] >= 0)
]

data_same_secondary = data[
    (data["singal_a_primary_label"] == data["singal_b_primary_label"])
    & (data["singal_a_secondary_label"] == data["singal_b_secondary_label"])
    & (data["singal_a_secondary_label"] > 0)
    & (data["singal_b_secondary_label"] > 0)
].copy()
data_same_secondary.loc[:, "loc_label"] = 1

data_diff_secondary = data[
    (data["singal_a_primary_label"] == data["singal_b_primary_label"])
    & (data["singal_a_secondary_label"] != data["singal_b_secondary_label"])
    & (data["singal_a_secondary_label"] > 0)
    & (data["singal_b_secondary_label"] > 0)
].copy()
data_diff_secondary.loc[:, "loc_label"] = 2

# data_other_secondary = data[
#     (data["singal_a_primary_label"] == data["singal_b_primary_label"])
#     & (data["singal_a_secondary_label"] > 0)
#     & (data["singal_b_secondary_label"] == 0)
# ].copy()
# data_other_secondary.loc[:, "loc_label"] = 3
# data = pd.concat([data_same_secondary, data_diff_secondary, data_other_secondary], axis=0)
# del data_same_secondary, data_diff_secondary, data_other_secondary

data = pd.concat([data_same_secondary, data_diff_secondary], axis=0)
del data_same_secondary, data_diff_secondary
gc.collect()

# -----------------------
# 7. Filter by curve duration (remove extreme values)
# _data_ = data[(data["singal_a_event_dynamic"] == 1) & (data["singal_a_event_dynamic"] == 1)]
# _data_ = data[(data["singal_a_event_dynamic"] == 0) & (data["singal_a_event_dynamic"] == 0)]

# _data_ = data[(data["singal_a_Basic_Area"] < 10) & (data["singal_b_Basic_Area"] < 10)]
# _data_ = data[
#     (data["singal_a_Basic_Area"] > 10)
#     & (data["singal_b_Basic_Area"] > 10)
#     & (data["singal_a_Basic_Area"] < 100)
#     & (data["singal_b_Basic_Area"] < 100)
# ]
# _data_ = data[(data["singal_a_Basic_Area"] > 100) & (data["singal_b_Basic_Area"] > 100)]

_data_ = data[
    (data["singal_a_Curve_Duration_10_to_10"] > 1)
    & (data["singal_b_Curve_Duration_10_to_10"] > 1)
    & (data["singal_a_Curve_Duration_10_to_10"] < 10)
    & (data["singal_b_Curve_Duration_10_to_10"] < 10)
]

# _data_ = data[(data["singal_a_Curve_Duration_10_to_10"] < 1) & (data["singal_b_Curve_Duration_10_to_10"] < 1)]
# _data_ = data[(data["singal_a_Curve_Duration_10_to_10"] > 10) & (data["singal_b_Curve_Duration_10_to_10"] > 10)]
# _data_ = data

# -----------------------
# 8. Compute per-cell medians for the two loc_label classes (1, 2)
aim_y = "cosine_corr"  # NOTE: ensure this column exists in the pickle data
# aim_y = "pearson_corr"
# aim_y = "euclidean_sim"
# aim_y = "manhattan_sim"
# aim_y = "spearman_corr"
# aim_y = "normalized_correlation_matrix"

aim_label = "loc_label"
if aim_y not in _data_.columns:
    raise KeyError(f"Column '{aim_y}' is missing. Please check your data.")

# Force loc_label into an integer/categorical type (only 1 and 2 expected)
_data_[aim_label] = _data_[aim_label].astype(np.int8)

cell_label_median = _data_.groupby(["cell_id", aim_label])[aim_y].median().reset_index()
cell_label_median = cell_label_median[cell_label_median[aim_label] > 0].reset_index(drop=True)

# Pivot: since loc_label has only two values, the pivot does not explode columns
cell_label_median_pivot = cell_label_median.pivot(
    index="cell_id",
    columns=aim_label,
    values=aim_y,
).reset_index()

cell_label_median_pivot.columns = [
    f"loc_{col}" if isinstance(col, (int, np.integer)) else col
    for col in cell_label_median_pivot.columns
]

print("Pivot shape:", cell_label_median_pivot.iloc[:, 1:].shape)

# -----------------------
# 9. Call statistical analysis helpers (assumed memory-optimized)
from significant_difference import (
    remove_outliers_df,
    number_of_cell,
    Mann_Whitney_U_test,
    concise_mannwhitney_analysis,
    concise_wilcoxon_analysis,
)

cell_label_median_pivot = remove_outliers_df(cell_label_median_pivot)
cell_label_median_pivot.dropna(axis=0, inplace=True)
number_of_cell(cell_label_median_pivot)

# Wilcoxon signed-rank test (paired data)
concise_wilcoxon_analysis(cell_label_median_pivot.iloc[:, 1:].dropna(axis=0))

# concise_mannwhitney_analysis(cell_label_median_pivot.iloc[:, 1:])

# -----------------------
# 10. Plot
# violin_and_scatter(
#     cell_label_median_pivot.iloc[:, 1:],
#     "<OUTPUT_DIR>/fig_secondary_branch_corr_area_lt_10.eps",
# )
# violin_and_scatter(
#     cell_label_median_pivot.iloc[:, 1:],
#     "<OUTPUT_DIR>/fig_secondary_branch_corr_area_10_100.eps",
# )

violin_and_scatter(
    cell_label_median_pivot.iloc[:, 1:],
    "<OUTPUT_DIR>/fig_secondary_branch_corr_area_gt_100.eps",
    start_0=True,
    aim_y_lim=0.1,
)


#%% ####
# Colorbar for area-bucket plots

# Define custom hex colors
colors = ["#7895c1", "#f5ebae", "#e3625d"]  # e.g., blue, yellow, red

# Create a custom colormap
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

# Create discrete levels
levels = np.linspace(0, 1, 4)  # 4 points -> 3 intervals
print("Using levels:", levels)

# Create discrete normalization
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

# Create ScalarMappable
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

# Create colorbar with discrete ticks
plt.figure()
cbar = plt.colorbar(sm, boundaries=levels, ticks=levels)
cbar.ax.yaxis.set_ticks_position("left")

# Hide axes
plt.axis("off")
plt.tight_layout()
plt.savefig("<OUTPUT_DIR>/fig_area_event_colorbar.eps", format="eps", bbox_inches="tight")
plt.show()


#%% Supplementary plot: proportion differences (primary branch vs secondary branch)

# Re-imports kept to match the original script structure
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


# Define plotting function
def violin_and_scatter(loc_perc, save_path, start_0=False, aim_y_lim=0):
    plt.figure(figsize=(6, 6))
    plt.rcParams.update({"font.size": 12, "font.weight": "bold"})

    # Draw violin plot
    ax = sns.violinplot(
        data=loc_perc,
        dodge=False,
        scale="width",
        inner=None,
        palette=["#c7e1e6", "#91b5bb", "#5c8b94"],
        cut=0,
        clip_on=True,
    )

    # Adjust violin plot: show only half for each violin
    for violin in ax.collections:
        if violin.get_paths():
            bbox = violin.get_paths()[0].get_extents()
            x0, y0, width, height = bbox.bounds
            violin.set_clip_path(
                plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData)
            )

    old_len_collections = len(ax.collections)
    sns.stripplot(
        data=loc_perc,
        dodge=False,
        ax=ax,
        size=5,
        palette=["#e67a6e", "#8a7ca4", "#4a726b"],
    )

    # Extract newly-added PathCollections (one per group)
    scatter_cols = ax.collections[old_len_collections:]

    # 3. Manually apply a unified X offset (to match the half-violin layout)
    for col in scatter_cols:
        offs = col.get_offsets().copy()
        offs[:, 0] += 0.11
        col.set_offsets(offs)

    # 4. Draw paired connecting lines (row-wise pairing across groups)
    offsets_per_group = [col.get_offsets() for col in scatter_cols]
    n_samples = loc_perc.shape[0]
    for i in range(n_samples):
        xs = [offs[i, 0] for offs in offsets_per_group]
        ys = [offs[i, 1] for offs in offsets_per_group]
        ax.plot(xs, ys, color="gray", alpha=0.3, linewidth=0.4, zorder=1)

    sns.boxplot(
        data=loc_perc,
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

    # Adjust y-axis range (core part)
    current_ymin, current_ymax = ax.get_ylim()
    new_ymax = current_ymax + 0.15 * (current_ymax - current_ymin)  # expand upper bound
    if start_0:
        ax.set_ylim([aim_y_lim, new_ymax])
    else:
        ax.set_ylim([current_ymin, new_ymax])  # keep lower bound unchanged

    # ax.set_ylim([current_ymin, new_ymax])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.set_xticks([])  # hide x-axis tick labels
    ax.grid(False)  # do not draw grid lines

    plt.tight_layout()
    plt.savefig(save_path, format="eps", bbox_inches="tight")
    plt.show()


# Load data
data = pd.read_csv(CSV_PATH, usecols=["cell_id", "Basic_Area", "primary_label", "secondary_label"]) 

# Define subsets
p_b = data[(data["primary_label"] > 0) & (data["secondary_label"] == -1)]
s_b = data[(data["primary_label"] > 0) & (data["secondary_label"] > 0)]

# Compute proportions and merge results
result_df = (
    p_b.groupby("cell_id")["Basic_Area"]
    # .apply(lambda x: (x < 10).mean())
    # .apply(lambda x: ((x > 10) & (x < 100)).mean())
    .apply(lambda x: (x > 100).mean())
    .rename("total_proportion_lt_10")
    .to_frame()
    .merge(
        s_b.groupby("cell_id")["Basic_Area"]
        # .apply(lambda x: (x < 10).mean())
        # .apply(lambda x: ((x > 10) & (x < 100)).mean())
        .apply(lambda x: (x > 100).mean())
        .rename("p_s_proportion_lt_10"),
        on="cell_id",
        how="outer",
    )
    .fillna(0)
    .reset_index()
)

# Call statistical analysis helpers (assumed memory-optimized)
from significant_difference import (
    remove_outliers_df,
    number_of_cell,
    Mann_Whitney_U_test,
    concise_mannwhitney_analysis,
    concise_wilcoxon_analysis,
)

result_df = remove_outliers_df(result_df)
result_df.dropna(axis=0, inplace=True)
number_of_cell(result_df)

# Wilcoxon signed-rank test (paired data)
concise_wilcoxon_analysis(result_df.iloc[:, 1:])

# violin_and_scatter(
#     result_df.iloc[:, 1:],
#     "<OUTPUT_DIR>/fig_primary_vs_secondary_branch_area_lt_10.eps",
#     start_0=True,
#     aim_y_lim=0.43,
# )

violin_and_scatter(
    result_df.iloc[:, 1:],
    "<OUTPUT_DIR>/fig_primary_vs_secondary_branch_area_10_100.eps",
    start_0=True,
    aim_y_lim=0.11,
)

# violin_and_scatter(
#     result_df.iloc[:, 1:],
#     "<OUTPUT_DIR>/fig_primary_vs_secondary_branch_area_gt_100.eps",
# )
