# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 17:21:41 2024

@author: Administrator
"""

#%%  subcellular structural localization
import os
import ast

import scipy.io as sio
import pandas as pd
import numpy as np
import cv2
from shapely.geometry import Point, Polygon
from tqdm import tqdm


def subcellular_structural_localization(mat_path: str, soma_path: str) -> pd.DataFrame:
    """
    Assign each calcium event's start location to a subcellular region.

    Returns a DataFrame with two columns:
      - primary_label:
          -1 = soma
           0 = remain/outside defined branch regions
          >0 = major-branch region index (1-based)
      - secondary_label:
          -1 = major branch (within a branch region)
           0 = microdomain (within a branch region but not inside major branch/branchlet polygons)
          >0 = branchlet index (1-based within the selected region)
    """
    # Derive related .mat paths from the provided landmark path
    cell_path = soma_path.replace("_LandMark.mat", "_Cell.mat")
    mb_reg_path = soma_path.replace("_LandMark.mat", "_mb_reg.mat")
    mb_path = soma_path.replace("_LandMark.mat", "_mb.mat")
    blet_path = soma_path.replace("_LandMark.mat", "_blet.mat")

    # AQuA outputs used to compute event start-frame coordinates
    x3d_path = mat_path.replace("_AQuA.mat", "_res_fts_loc_x3D.mat")
    sz_path = mat_path.replace("_AQuA.mat", "_res_opts_sz.mat")
    t0_path = mat_path.replace("_AQuA.mat", "_res_fts_loc_t0.mat")
    t1_path = mat_path.replace("_AQuA.mat", "_res_fts_loc_t1.mat")

    if not (os.path.isfile(x3d_path) and os.path.isfile(sz_path) and os.path.isfile(t0_path) and os.path.isfile(t1_path)):
        raise FileNotFoundError("Missing one or more required AQuA output .mat files for location extraction.")

    # Load event voxel indices and volume size (Fortran order) used by AQuA
    x3d = sio.loadmat(x3d_path)["x3D"]  # object array: each entry holds linear indices
    sz = sio.loadmat(sz_path)["sz"]
    t0 = sio.loadmat(t0_path)["t0"]
    t1 = sio.loadmat(t1_path)["t1"]

    # Load subcellular structure boundaries (polygons)
    soma_loc = sio.loadmat(soma_path)["bd0"][0][0][0][0][0][0]
    cell_loc = sio.loadmat(cell_path)["bd0"][0][0][0][0][0][0]  # kept for completeness (not used below)
    mb_reg_loc = sio.loadmat(mb_reg_path)["bd0"]
    mb_loc = sio.loadmat(mb_path)["bd0"]
    blet_loc = sio.loadmat(blet_path)["bd0"]

    # Convert each event's linear indices into (x, y, t) and keep only its earliest frame pixels
    locations = [None] * x3d.shape[1]
    for i in range(x3d.shape[1]):
        lin_idx = x3d[0, i].astype(np.int32)
        x, y, t = np.unravel_index(lin_idx, sz[0].astype(np.int32), order="F")
        t_min = int(t.min())
        locations[i] = np.array([x[t == t_min], y[t == t_min], t[t == t_min]]).T

    # Compute the centroid of the earliest-frame pixels (per event)
    event_centroids = np.zeros((len(locations), 2), dtype=float)
    for i, loc in enumerate(locations):
        if loc.shape[0] == 1:
            centroid = loc[:, :2].astype(float)
        else:
            centroid = np.mean(loc[:, :2], axis=0).astype(float)
        event_centroids[i, :] = centroid

    # Default labels: (0, 0) means "remain" (not in soma / branch region / branch structures)
    ca_event_loc_label = pd.DataFrame(
        np.zeros((len(locations), 2), dtype=float),
        columns=["primary_label", "secondary_label"],
    )

    # For each major-branch region, assign events to:
    #   - region (primary_label = j+1)
    #   - major branch polygon within region (secondary_label = -1)
    #   - branchlet polygons within region (secondary_label = branchlet index)
    for j in range(mb_reg_loc.shape[1]):
        region_poly = mb_reg_loc[0, j][0][0][0][0]

        in_region = np.array([Point(p).within(Polygon(region_poly)) for p in event_centroids])
        ca_event_loc_label.iloc[in_region, 0] = j + 1

        # Find which major-branch polygon belongs to this region by checking its centroid
        mb_centroids = np.array(
            [np.mean(mb_loc[0, i][0][0][0][0], axis=0) for i in range(mb_loc.shape[1])]
        )
        mb_in_region = np.array([Point(p).within(Polygon(region_poly)) for p in mb_centroids])

        # NOTE: original code assumes exactly one major branch is selected here
        aim_mb_loc = mb_loc[0, mb_in_region][0][0][0][0][0]
        in_major_branch = np.array([Point(p).within(Polygon(aim_mb_loc)) for p in event_centroids])
        ca_event_loc_label.iloc[in_major_branch, 1] = -1

        # Find branchlets in this region, then label events inside each branchlet polygon
        blet_centroids = np.array(
            [np.mean(blet_loc[0, i][0][0][0][0], axis=0) for i in range(blet_loc.shape[1])]
        )
        blet_in_region = np.array([Point(p).within(Polygon(region_poly)) for p in blet_centroids])

        aim_blet_loc = blet_loc[0, blet_in_region]
        for n_blet in range(aim_blet_loc.shape[0]):
            blet_poly = aim_blet_loc[n_blet][0][0][0][0]
            in_blet = np.array([Point(p).within(Polygon(blet_poly)) for p in event_centroids])
            ca_event_loc_label.iloc[in_blet, 1] = n_blet + 1

    # Soma has the highest priority: override labels for events inside soma polygon
    in_soma = np.array([Point(p).within(Polygon(soma_loc)) for p in event_centroids])
    ca_event_loc_label.iloc[in_soma, 0] = -1
    ca_event_loc_label.iloc[in_soma, 1] = -1

    # Optional: keep these around if you want to later export timing info
    # (These were loaded because the original code expected them.)
    _ = t0, t1, cell_loc

    return ca_event_loc_label


def load_paths_from_txt(txt_path: str) -> list[str]:
    """
    Load a list of paths from a text file where each line contains a Python-like string.
    Safer than eval(): uses ast.literal_eval().
    """
    paths: list[str] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            paths.append(ast.literal_eval(line))
    return paths


if __name__ == "__main__":
    # Read input path lists
    soma_paths = load_paths_from_txt("soma_path.txt")
    mat_paths = load_paths_from_txt("mat_path.txt")

    # Accumulate labels for all recordings
    all_labels = pd.DataFrame(columns=["primary_label", "secondary_label"])

    for i in tqdm(range(len(mat_paths))):
        labels = subcellular_structural_localization(mat_paths[i], soma_paths[i])
        all_labels = pd.concat([all_labels, labels], axis=0, ignore_index=True)

 




#%%  

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from significant_difference import (
    remove_outliers_df,
    number_of_cell,
    concise_wilcoxon_analysis,
)


def violin_and_scatter(df: pd.DataFrame, save_path: str) -> None:
    """
    Make a half-violin + scatter + paired-lines + boxplot figure.

    Expected input:
      - df: rows are paired samples (e.g., cells), columns are conditions (e.g., soma/branch/...).
    """
    plt.figure(figsize=(5, 5))
    plt.rcParams["font.size"] = 12
    plt.rcParams["font.weight"] = "bold"

    ax = sns.violinplot(
        data=df,
        dodge=False,
        scale="width",
        inner=None,
        palette=["#9ac89a", "#fec37d", "#91b5bb"],
        cut=0,
        clip_on=True,
    )

    for violin in ax.collections:
        bbox = violin.get_paths()[0].get_extents()
        x0, y0, width, height = bbox.bounds
        violin.set_clip_path(
            plt.Rectangle((x0, y0), width / 2, height, transform=ax.transData)
        )

    old_len_collections = len(ax.collections)
    sns.stripplot(
        data=df,
        dodge=False,
        ax=ax,
        size=5,
        palette=["#9ac89a", "#fec37d", "#91b5bb"],
    )

    scatter_cols = ax.collections[old_len_collections:]

    for col in scatter_cols:
        offs = col.get_offsets().copy()
        offs[:, 0] += 0.11
        col.set_offsets(offs)

    offsets_per_group = [col.get_offsets() for col in scatter_cols]
    n_samples = df.shape[0]
    for i in range(n_samples):
        xs = [offs[i, 0] for offs in offsets_per_group]
        ys = [offs[i, 1] for offs in offsets_per_group]
        ax.plot(xs, ys, color="gray", alpha=0.3, linewidth=0.4, zorder=1)

    sns.boxplot(
        data=df,
        saturation=1,
        showfliers=False,
        width=0.3,
        boxprops={"zorder": 3, "facecolor": "none", "linewidth": 4, "edgecolor": "black"},
        whiskerprops={"linewidth": 4, "color": "black", "zorder": 10},
        capprops={"linewidth": 4, "color": "black", "zorder": 10},
        medianprops={"linewidth": 4, "color": "red", "zorder": 10},
        ax=ax,
    )

    ymin, ymax = ax.get_ylim()
    pad = 0.1 * (ymax - ymin)
    ax.set_ylim([ymin - pad, ymax + pad])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.set_xticks([])
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(save_path, format="eps", bbox_inches="tight")
    plt.show()


def build_wide_by_structure(
    comtest: pd.DataFrame,
    value_col: str,
    is_percentage: bool = False,
) -> pd.DataFrame:
    """
    Convert event-level values into a wide table with 3 columns:
      - soma
      - primary_branch
      - branchlets_and_leaflets

    Rules (based on your labeling scheme):
      soma:          primary_label == -1 and secondary_label == -1
      primary:       primary_label > 0  and secondary_label == -1
      branchlets+:   primary_label >= 0 and secondary_label >= 0

    If is_percentage=True:
      - for each cell_id, compute mean(value_col) * 100 within each structure
        (your original code used sum/count * 100).
    Otherwise:
      - keep raw values aligned to original rows (NaN where not in structure).
    """
    primary = comtest["primary_label"].values
    secondary = comtest["secondary_label"].values
    values = comtest[value_col].values

    masks = [
        (primary == -1) & (secondary == -1),
        (primary > 0) & (secondary == -1),
        (primary >= 0) & (secondary >= 0),
    ]

    data = {}
    for i, mask in enumerate(masks, start=1):
        data[f"cond_{i}"] = np.where(mask, values, np.nan)

    wide = pd.DataFrame(data, index=comtest.index).rename(
        columns={
            "cond_1": "soma",
            "cond_2": "primary_branch",
            "cond_3": "branchlets_and_leaflets",
        }
    )

    if not is_percentage:
        return wide

    grouped = wide.groupby(comtest["cell_id"])
    percentage = (grouped.sum() / grouped.count()) * 100
    return percentage.fillna(0)


def print_sample_counts(comtest: pd.DataFrame, wide: pd.DataFrame) -> None:
    """
    Print how many unique cells/mice contribute non-NaN values to each column.
    Helpful for sanity-checking inclusion after filtering/outlier removal.
    """
    for col in wide.columns:
        valid_idx = wide[col].dropna().index
        valid_cells = comtest.loc[valid_idx, "cell_id"].unique()
        valid_mice = comtest.loc[valid_idx, "mouse_id"].unique()
        print(
            f"Column '{col}': cell num = {len(valid_cells)}, mice num = {len(valid_mice)}, mice={valid_mice}"
        )


if __name__ == "__main__":
    comtest = pd.read_csv(
        "YoungMouse_AllCalciumEvents_AllParameters_WithEventLocationLabels.csv"
    )

    comtest["event_static"] = 0
    comtest["event_static"] = 0
    static_mask = (
        (comtest.Landmark_event_away_from_landmark_landmark_1 == 0)
        & (comtest.Landmark_event_toward_landmark_landmark_1 == 0)
    )
    comtest.loc[static_mask, "event_static"] = 1
    comtest["event_dynamic"] = 1 - comtest["event_static"]

    # 1) Event count percentage per structure (per cell)
    loc_perc = pd.DataFrame(
        columns=[
            "soma_ca_event_perc",
            "primary_branch_ca_event_perc",
            "branchlets_and_leaflets_ca_event_perc",
        ]
    )

    for cell_id in comtest["cell_id"].drop_duplicates():
        pro = comtest[comtest["cell_id"] == cell_id]
        soma_num = ((pro["primary_label"] == -1) & (pro["secondary_label"] == -1)).sum()
        primary_num = ((pro["primary_label"] > 0) & (pro["secondary_label"] == -1)).sum()
        branchlets_num = len(pro) - soma_num - primary_num

        temp = pd.DataFrame(
            {
                "soma_ca_event_perc": [soma_num / len(pro)],
                "primary_branch_ca_event_perc": [primary_num / len(pro)],
                "branchlets_and_leaflets_ca_event_perc": [branchlets_num / len(pro)],
            }
        )
        loc_perc = pd.concat([loc_perc, temp], ignore_index=True)

    loc_perc = remove_outliers_df(loc_perc).dropna(axis=0)
    number_of_cell(loc_perc)
    concise_wilcoxon_analysis(loc_perc)

    violin_and_scatter(
        loc_perc,
        "G:/fig1.Proportion of calcium event counts across different subcellular compartments.eps",
    )

    # 2) Event area (median per cell, paired)
    param = "Basic_Area"
    wide = build_wide_by_structure(comtest, param, is_percentage=False)
    wide = remove_outliers_df(wide)
    print_sample_counts(comtest, wide)

    wide_cell = wide.groupby(comtest["cell_id"]).median().reset_index()
    wide_cell = wide_cell.dropna(axis=0)

    concise_wilcoxon_analysis(wide_cell.iloc[:, 1:])
    violin_and_scatter(
        wide_cell.iloc[:, 1:],
        f"./fig1.Param of calcium events {param} across different subcellular compartments.eps",
    )

    # 3) Event duration (median per cell, paired)
    param = "Curve_Duration_10_to_10"
    wide = build_wide_by_structure(comtest, param, is_percentage=False)
    wide = remove_outliers_df(wide)
    print_sample_counts(comtest, wide)

    wide_cell = wide.groupby(comtest["cell_id"]).median().reset_index()
    wide_cell = wide_cell.dropna(axis=0)

    concise_wilcoxon_analysis(wide_cell.iloc[:, 1:])
    violin_and_scatter(
        wide_cell.iloc[:, 1:],
        f"./fig1.Param of calcium events {param} across different subcellular compartments.eps",
    )

    # 4) Dynamic percentage per structure (per cell)
    param = "event_dynamic"
    wide_pct = build_wide_by_structure(comtest, param, is_percentage=True)
    wide_pct = remove_outliers_df(wide_pct)
    number_of_cell(wide_pct)

    wide_pct = wide_pct.dropna(axis=0)
    concise_wilcoxon_analysis(wide_pct)
    violin_and_scatter(
        wide_pct,
        f"./fig1.Proportion of calcium events {param} across different subcellular compartments.eps",
    )

    # 5) Static percentage per structure (per cell)
    param = "event_static"
    wide_pct = build_wide_by_structure(comtest, param, is_percentage=True)
    wide_pct = remove_outliers_df(wide_pct)
    number_of_cell(wide_pct)

    wide_pct = wide_pct.dropna(axis=0)
    concise_wilcoxon_analysis(wide_pct)
    violin_and_scatter(
        wide_pct,
        f"./fig1.Proportion of calcium events {param} across different subcellular compartments.eps",
    )
