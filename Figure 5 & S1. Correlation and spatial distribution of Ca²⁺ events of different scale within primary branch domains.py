# -*- coding: utf-8 -*-

import gc
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap


# -----------------------
# Plotting utilities
def violin_and_scatter(
    loc_perc: pd.DataFrame,
    save_path: str,
    start_0: bool = False,
    aim_y_lim: float = 0.0,
    ymax_expand_ratio: float = 0.2,
) -> None:
    """
    Draw a half-violin + paired scatter + connecting lines + boxplot overlay.

    Parameters
    ----------
    loc_perc:
        A DataFrame where each column is a group and each row is a paired sample.
    save_path:
        Output path for the figure (EPS).
    start_0:
        If True, force the y-axis lower bound to `aim_y_lim`.
    aim_y_lim:
        The y-axis lower bound used when `start_0=True`.
    ymax_expand_ratio:
        Expand the upper y-limit by this fraction of the current y-range.
    """
    plt.figure(figsize=(5, 6))
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

    # Clip each violin to show only half
    for violin in ax.collections:
        if violin.get_paths():
            bbox = violin.get_paths()[0].get_extents()
            x0, y0, width, height = bbox.bounds
            violin.set_clip_path(
                plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData)
            )

    old_len_collections = len(ax.collections)

    # Draw scatter points
    sns.stripplot(
        data=loc_perc,
        dodge=False,
        ax=ax,
        size=5,
        palette=["#e67a6e", "#8a7ca4", "#4a726b"],
    )

    # Newly-added PathCollections correspond to scatter groups (one per column)
    scatter_cols = ax.collections[old_len_collections:]

    # Shift scatter x positions to align with half-violin
    for col in scatter_cols:
        offs = col.get_offsets().copy()
        offs[:, 0] += 0.11
        col.set_offsets(offs)

    # Draw paired connecting lines across groups (row-wise pairing)
    offsets_per_group = [col.get_offsets() for col in scatter_cols]
    n_samples = loc_perc.shape[0]
    for i in range(n_samples):
        xs = [offs[i, 0] for offs in offsets_per_group]
        ys = [offs[i, 1] for offs in offsets_per_group]
        ax.plot(xs, ys, color="gray", alpha=0.3, linewidth=0.4, zorder=1)

    # Boxplot overlay
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

    # Adjust y-axis range
    current_ymin, current_ymax = ax.get_ylim()
    new_ymax = current_ymax + ymax_expand_ratio * (current_ymax - current_ymin)

    if start_0:
        ax.set_ylim([aim_y_lim, new_ymax])
    else:
        ax.set_ylim([current_ymin, new_ymax])

    # Cosmetic styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.set_xticks([])
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(save_path, format="eps", bbox_inches="tight")
    plt.show()


# -----------------------
# Data helpers
def add_signal_order(
    df: pd.DataFrame, cell_col: str = "cell_id", order_col: str = "signal_a_order"
) -> pd.DataFrame:
    """
    Add per-cell sequential order index to match event rows across tables.
    """
    df = df.copy()
    df[order_col] = df.groupby(cell_col).cumcount().astype(np.int32)
    return df


def load_base_from_pickle(pkl_path: str) -> pd.DataFrame:
    """
    Load the merged DataFrame from a pickle and drop unused columns.
    """
    with open(pkl_path, "rb") as f:
        df = pickle.load(f)["merged_df"]

    cols_to_drop = [
        "signal_a_start_point_x",
        "signal_a_start_point_y",
        "signal_a_start_frame",
        "signal_b_start_point_x",
        "signal_b_start_point_y",
        "signal_b_start_frame",
        "correlation_matrix",
    ]
    df.drop(columns=cols_to_drop, inplace=True)
    return df


def merge_event_dynamic(data: pd.DataFrame, csv_all: pd.DataFrame) -> pd.DataFrame:
    """
    Merge event dynamic/static labels into pairwise data for signal A and B.
    """
    df1 = csv_all[
        [
            "cell_id",
            "Landmark_event_away_from_landmark_landmark_1",
            "Landmark_event_toward_landmark_landmark_1",
        ]
    ].copy()

    df1["event_static"] = (
        (df1["Landmark_event_away_from_landmark_landmark_1"] == 0)
        & (df1["Landmark_event_toward_landmark_landmark_1"] == 0)
    ).astype(np.int8)
    df1["event_dynamic"] = (1 - df1["event_static"]).astype(np.int8)

    df1 = add_signal_order(df1, order_col="signal_a_order")
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
    return data


def merge_branch_labels(data: pd.DataFrame, csv_all: pd.DataFrame) -> pd.DataFrame:
    """
    Merge primary/secondary branch labels for signal A and B.
    """
    df2 = csv_all[["cell_id", "primary_label", "secondary_label"]].copy()
    df2["primary_label"] = df2["primary_label"].round(0).astype(np.int32)
    df2["secondary_label"] = df2["secondary_label"].round(0).astype(np.int32)

    df2 = add_signal_order(df2, order_col="signal_a_order")
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
    return data


def merge_curve_duration(data: pd.DataFrame, csv_all: pd.DataFrame) -> pd.DataFrame:
    """
    Merge curve duration, applying the original scalar transform (duration * 2/5).
    """
    df3 = csv_all[["cell_id", "Curve_Duration_10_to_10"]].copy()
    df3["Curve_Duration_10_to_10"] = df3["Curve_Duration_10_to_10"] * 2 / 5

    df3 = add_signal_order(df3, order_col="signal_a_order")
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
    return data


def merge_basic_area(data: pd.DataFrame, csv_all: pd.DataFrame) -> pd.DataFrame:
    """
    Merge basic area for signal A and B.
    """
    df4 = csv_all[["cell_id", "Basic_Area"]].copy()
    df4 = add_signal_order(df4, order_col="signal_a_order")
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
    return data


def label_primary_branch_relation(data: pd.DataFrame) -> pd.DataFrame:
    """
    Label pairs as same primary branch (loc_label=1) or different (loc_label=2).
    """
    data = data.copy()
    data["singal_a_primary_label"] = data["singal_a_primary_label"].astype(np.int32)
    data["singal_b_primary_label"] = data["singal_b_primary_label"].astype(np.int32)

    data = data[(data["singal_a_primary_label"] > 0) & (data["singal_b_primary_label"] > 0)]

    same_df = data[data["singal_a_primary_label"] == data["singal_b_primary_label"]].copy()
    same_df.loc[:, "loc_label"] = 1

    diff_df = data[data["singal_a_primary_label"] != data["singal_b_primary_label"]].copy()
    diff_df.loc[:, "loc_label"] = 2

    out = pd.concat([same_df, diff_df], axis=0)
    del same_df, diff_df
    gc.collect()
    return out


def filter_by_area_range(
    data: pd.DataFrame,
    min_area: float,
    max_area: float,
) -> pd.DataFrame:
    """
    Filter rows by Basic_Area range for both signal A and signal B.
    """
    return data[
        (data["singal_a_Basic_Area"] > min_area)
        & (data["singal_b_Basic_Area"] > min_area)
        & (data["singal_a_Basic_Area"] < max_area)
        & (data["singal_b_Basic_Area"] < max_area)
    ]


def compute_cellwise_median_pivot(
    data: pd.DataFrame,
    value_col: str,
    label_col: str = "loc_label",
) -> pd.DataFrame:
    """
    Compute per-cell median for each label, then pivot to wide format.

    Output columns:
        cell_id, loc_1, loc_2
    """
    if value_col not in data.columns:
        raise KeyError(f"Column '{value_col}' is missing. Please check your pickle content.")

    tmp = data.copy()
    tmp[label_col] = tmp[label_col].astype(np.int8)

    cell_label_median = (
        tmp.groupby(["cell_id", label_col])[value_col].median().reset_index()
    )
    cell_label_median = cell_label_median[cell_label_median[label_col] > 0].reset_index(
        drop=True
    )

    pivot_df = (
        cell_label_median.pivot(index="cell_id", columns=label_col, values=value_col)
        .reset_index()
    )

    pivot_df.columns = [
        f"loc_{col}" if isinstance(col, (int, np.integer)) else col for col in pivot_df.columns
    ]

    print("Pivot shape:", pivot_df.iloc[:, 1:].shape)
    return pivot_df


#%% -----------------------
# Analysis + figure: primary-branch correlation comparison (area 10-100 example)

# NOTE: Replace these placeholders with your real local paths.
PKL_PATH = "<DATA_DIR>/vae_z_values_young_with_distance.pkl"
CSV_PATH = "<DATA_DIR>/young_mouse_all_events_with_loc_label_two_line.csv"
FIG_OUT_CORR = "<OUTPUT_DIR>/fig_primary_branch_corr_area_10_100.eps"

data = load_base_from_pickle(PKL_PATH)

# Load all required CSV columns once (reduces repeated disk IO)
csv_all = pd.read_csv(
    CSV_PATH,
    usecols=[
        "cell_id",
        "Landmark_event_away_from_landmark_landmark_1",
        "Landmark_event_toward_landmark_landmark_1",
        "primary_label",
        "secondary_label",
        "Curve_Duration_10_to_10",
        "Basic_Area",
    ],
)

data = merge_event_dynamic(data, csv_all)
data = merge_branch_labels(data, csv_all)
data = merge_curve_duration(data, csv_all)
data = merge_basic_area(data, csv_all)
del csv_all
gc.collect()

# Keep a copy if you need to switch back to other filtering logic later
data_copy = data.copy()

# Label same/different primary branch
data = label_primary_branch_relation(data)

# Area filter (10-100)
_data_ = filter_by_area_range(data, min_area=10, max_area=100)

# Compute per-cell medians for each loc_label
aim_y = "cosine_corr"
cell_label_median_pivot = compute_cellwise_median_pivot(_data_, value_col=aim_y)

# Statistical tests (assumes your module is available)
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

# Paired Wilcoxon signed-rank test
cell_label_median_pivot = cell_label_median_pivot.dropna(axis=0)
concise_wilcoxon_analysis(cell_label_median_pivot.iloc[:, 1:])

# Plot
violin_and_scatter(
    cell_label_median_pivot.iloc[:, 1:],
    FIG_OUT_CORR,
    start_0=True,
    aim_y_lim=-0.2,
    ymax_expand_ratio=0.2,
)


#%% -----------------------
# Example plotting section: primary-branch correlation heatmap preparation (single cell)

# NOTE: Replace these placeholders with your real local paths.
FIG_OUT_COLORBAR = "<OUTPUT_DIR>/fig_area_event_colorbar.eps"

data = load_base_from_pickle(PKL_PATH)

# Merge only what is needed here (branch labels + area)
df2 = pd.read_csv(CSV_PATH, usecols=["cell_id", "primary_label", "secondary_label"])
df2["primary_label"] = df2["primary_label"].round(0).astype(np.int32)
df2["secondary_label"] = df2["secondary_label"].round(0).astype(np.int32)
df2 = add_signal_order(df2, order_col="signal_a_order")
df2["signal_b_order"] = df2["signal_a_order"]

data = data.merge(
    df2[["cell_id", "signal_a_order", "primary_label"]],
    on=["cell_id", "signal_a_order"],
    how="left",
).rename(columns={"primary_label": "singal_a_primary_label"})

data = data.merge(
    df2[["cell_id", "signal_a_order", "secondary_label"]],
    on=["cell_id", "signal_a_order"],
    how="left",
).rename(columns={"secondary_label": "singal_a_secondary_label"})

data = data.merge(
    df2[["cell_id", "signal_b_order", "primary_label"]],
    on=["cell_id", "signal_b_order"],
    how="left",
).rename(columns={"primary_label": "singal_b_primary_label"})

data = data.merge(
    df2[["cell_id", "signal_b_order", "secondary_label"]],
    on=["cell_id", "signal_b_order"],
    how="left",
).rename(columns={"secondary_label": "singal_b_secondary_label"})

del df2
gc.collect()

df4 = pd.read_csv(CSV_PATH, usecols=["cell_id", "Basic_Area"])
df4 = add_signal_order(df4, order_col="signal_a_order")
df4["signal_b_order"] = df4["signal_a_order"]

data = data.merge(
    df4[["cell_id", "signal_a_order", "Basic_Area"]],
    on=["cell_id", "signal_a_order"],
    how="left",
).rename(columns={"Basic_Area": "singal_a_Basic_Area"})

data = data.merge(
    df4[["cell_id", "signal_b_order", "Basic_Area"]],
    on=["cell_id", "signal_b_order"],
    how="left",
).rename(columns={"Basic_Area": "singal_b_Basic_Area"})

del df4
gc.collect()

# Primary branch discussion (strict subset)
data["singal_a_primary_label"] = data["singal_a_primary_label"].astype(np.int32)
data["singal_b_primary_label"] = data["singal_b_primary_label"].astype(np.int32)

data_copy = data.copy()

data = data[
    (data["singal_a_primary_label"] > 0)
    & (data["singal_a_secondary_label"] == 0)
    & (data["singal_b_primary_label"] > 0)
    & (data["singal_b_secondary_label"] == 0)
]

# Anonymized example cell selector (replace with your actual ID)
data = data[data["cell_id"] == "<EXAMPLE_CELL_ID>"]

same_df = data[data["singal_a_primary_label"] == data["singal_b_primary_label"]].copy()
same_df.loc[:, "loc_label"] = 1

diff_df = data[data["singal_a_primary_label"] != data["singal_b_primary_label"]].copy()
diff_df.loc[:, "loc_label"] = 2

# Example area bucket (keep one active at a time)
same_df = same_df[(same_df["singal_a_Basic_Area"] > 100) & (same_df["singal_b_Basic_Area"] > 100)]
diff_df = diff_df[(diff_df["singal_a_Basic_Area"] > 100) & (diff_df["singal_b_Basic_Area"] > 100)]

data = pd.concat([same_df, diff_df], axis=0)
del same_df, diff_df

# Make pairs symmetric by swapping A/B
data_swap = data.copy()
data_swap[["signal_a_order", "signal_b_order"]] = data_swap[["signal_b_order", "signal_a_order"]]
data = pd.concat([data, data_swap], axis=0)
del data_swap
gc.collect()


#%% -----------------------
# Colorbar figure (discrete levels)

# Custom hex colors
colors = ["#7895c1", "#f5ebae", "#e3625d"]

# Custom colormap
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

# Discrete levels
levels = np.linspace(0, 1, 4)
print("Using levels:", levels)

# Discrete normalization
norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

# ScalarMappable for colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])

plt.figure()
cbar = plt.colorbar(sm, boundaries=levels, ticks=levels)
cbar.ax.yaxis.set_ticks_position("left")

plt.axis("off")
plt.tight_layout()
plt.savefig(FIG_OUT_COLORBAR, format="eps", bbox_inches="tight")
plt.show()


#%% -----------------------
# Supplementary plot: proportions in p+s vs all events (per-cell paired comparison)

# NOTE: Replace this placeholder output path.
FIG_OUT_PROP = "<OUTPUT_DIR>/fig_cell_vs_primary_structure_area_lt_10.eps"

data = pd.read_csv(CSV_PATH, usecols=["cell_id", "Basic_Area", "primary_label", "secondary_label"])

# p+s subset
p_s = data[(data["primary_label"] > 0) & (data["secondary_label"] != 0)]

# Proportion calculator
result_df = (
    data.groupby("cell_id")["Basic_Area"]
    .apply(lambda x: (x < 10).mean())
    # .apply(lambda x: ((x > 10) & (x < 100)).mean())
    # .apply(lambda x: (x > 100).mean())
    .rename("total_proportion_lt_10")
    .to_frame()
    .merge(
        p_s.groupby("cell_id")["Basic_Area"]
        .apply(lambda x: (x < 10).mean())
        # .apply(lambda x: ((x > 10) & (x < 100)).mean())
        # .apply(lambda x: (x > 100).mean())
        .rename("p_s_proportion_lt_10"),
        on="cell_id",
        how="outer",
    )
    .fillna(0)
    .reset_index()
)

result_df = remove_outliers_df(result_df)
result_df.dropna(axis=0, inplace=True)
number_of_cell(result_df)

# Paired Wilcoxon signed-rank test
concise_wilcoxon_analysis(result_df.iloc[:, 1:])

violin_and_scatter(
    result_df.iloc[:, 1:],
    FIG_OUT_PROP,
    start_0=True,
    aim_y_lim=0.56,
    ymax_expand_ratio=0.15,
)
