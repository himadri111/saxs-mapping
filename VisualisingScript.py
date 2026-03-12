#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 17:18:51 2025

@author: lauraforster
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import os
import matplotlib

# Force inline backend when running in Spyder (avoids FigureCanvasAgg warnings)
if "spyder_kernels" in sys.modules:
    matplotlib.use("module://matplotlib_inline.backend_inline")
matplotlib.use("Agg")  # important when running overnight / no GUI; prevents figure windows

def heatmap(Filenumber, Output_directoryCSV, HeatmapPlot, AngularOverlay, WhatPlot, threshold_saxs_intensity, threshold_area, threshold_Dperiod, zmin, zmax, perc_above_baseline, max_intensity):
    
    print(f"\nBeginning heatmap generation for File {Filenumber}...")

    iq_path = os.path.join(Output_directoryCSV, f"{Filenumber} IQ_fitting.csv")
    # Step 1: Check and load IQ CSV
    if not os.path.isfile(iq_path):
        print(f"No IQ CSV found at: {iq_path}")
        return
    
    iq_df = pd.read_csv(iq_path)

    if len(iq_df) <= 5:
        print(f"IQ CSV found but has insufficient data ({len(iq_df)} rows).")
        return
    
    print(f"IQ CSV loaded with {len(iq_df)} rows.")

    # Step 2: Extract coordinates
    if not {'x', 'y'}.issubset(iq_df.columns):
        print("IQ CSV missing required 'x' or 'y' columns.")
        return
    
    x = iq_df['x'].astype(int)
    y = iq_df['y'].astype(int)

    # Step 3: Select column based on WhatPlot
    if WhatPlot == 'SAXS':
        column = 'total SAXS intensity'
        threshold = threshold_saxs_intensity
        default_z = (0, 2000)
    elif WhatPlot == 'SAXS_norm':
        column = 'total_SAXS_norm_0_1'
        threshold = 0
        default_z = (0, 1)
    elif WhatPlot == 'curvearea':
        column = 'area under third order curve'
        threshold = threshold_area
        default_z = (0, 0.05)
    elif WhatPlot == 'curvearea_norm':
        column = 'collagen_third_norm_0_1'
        threshold = 0
        default_z = (0, 1)
    elif WhatPlot == 'Dperiod':
        column = 'D_period_lu'
        threshold = threshold_Dperiod
        default_z = (63, 69)
    elif WhatPlot == 'wMu':
        column = 'wMu'
        threshold = 0
        default_z = (0, 0.99)
    elif WhatPlot == 'fibril_radius':
        column = 'fibril_radius'
        threshold = 0
        default_z = (0, 100)
    else:
        print(f"Invalid WhatPlot value: {WhatPlot}")
        return

    if column not in iq_df.columns:
        print(f"Column '{column}' not found in IQ CSV.")
        return
    
    if zmin is None or zmax is None:
        zmin, zmax = default_z

    values = iq_df[column].to_numpy()
    values_all = iq_df[column].to_numpy(dtype=float)
    # ------------------------------------------------------------------
    #  A) base filter on the main metric (SAXS / area / D_period / etc.)
    # ------------------------------------------------------------------
    mask_main = values > threshold

    # ------------------------------------------------------------------
    #  B) percentage_above_baseline filter (if enabled & present)
    # ------------------------------------------------------------------
    if perc_above_baseline is not None:
        if 'percentage_above_baseline' not in iq_df.columns:
            print("percentage_above_baseline threshold requested but column "
                  "'percentage_above_baseline' not found. Skipping that filter.")
            mask_pct = np.ones(len(iq_df), dtype=bool)
        else:
            pct = iq_df['percentage_above_baseline'].to_numpy(dtype=float)
            # assume this column is in %, so compare directly to e.g. 20
            mask_pct = np.isfinite(pct) & (pct >= perc_above_baseline)
    else:
        mask_pct = np.ones(len(iq_df), dtype=bool)


    # ------------------------------------------------------------------
    #  C) Collagen-based bottom-X% filter (independent of WhatPlot)
    # ------------------------------------------------------------------
    if max_intensity is not None:
        if 'total_SAXS_norm_0_1' not in iq_df.columns:
            print("total_SAXS_norm_0_1 column not found. Skipping intensity filter.")
            mask_int = np.ones(len(iq_df), dtype=bool)
        else:
            collagen_vals = iq_df['total_SAXS_norm_0_1'].to_numpy(dtype=float)
            valid = np.isfinite(collagen_vals)
    
            if valid.sum() == 0:
                print("No valid total_SAXS_norm_0_1 values. Skipping filter.")
                mask_int = np.ones(len(iq_df), dtype=bool)
            else:
                cutoff = np.quantile(collagen_vals[valid], max_intensity)
                mask_int = valid & (collagen_vals >= cutoff)
    
                print(f"[Collagen filter] cutoff={cutoff:.4f} "
                      f"(dropping bottom {max_intensity*100:.1f}%)")
    else:
        mask_int = np.ones(len(iq_df), dtype=bool)
    
    # ------------------------------------------------------------------
    #  Combine all three: main metric + % above baseline + max intensity
    # ------------------------------------------------------------------
    combined_mask = mask_main & mask_pct & mask_int

    filtered_df = iq_df[combined_mask]
    if filtered_df.empty:
        print("No values left after applying thresholds "
              f"(column={column}, perc_above_baseline={perc_above_baseline}, "
              f"max_intensity={max_intensity}).")
        return

    x = filtered_df['x'].astype(int)
    y = filtered_df['y'].astype(int)
    values = filtered_df[column].to_numpy()

    # ------------------------------------------------------------------
    #  Build image array and plot
    # ------------------------------------------------------------------
    max_x = x.max() + 1
    max_y = y.max() + 1
    
    if not HeatmapPlot and not AngularOverlay:
        print("Both HeatmapPlot and AngularOverlay are False. Nothing to plot.")
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))

    if HeatmapPlot:
        image = np.full((max_y, max_x), np.nan)

        for xi, yi, vi in zip(x, y, values):
            image[yi, xi] = vi
    
        im = ax.imshow(
            image,
            origin='lower',
            cmap='jet',
            vmin=zmin,
            vmax=zmax,
            extent=(0, max_x, 0, max_y)
        )
    
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(column)
        ax.set_title(f"{WhatPlot} Heatmap for File {Filenumber}")
    else:
        ax.set_title(f"Angular Overlay Only for File {Filenumber}")
    
    ax.set_xticks(np.arange(0, max_x, step=5))
    ax.set_yticks(np.arange(0, max_y, step=5))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(False)

    if AngularOverlay:
        try:
            overlay_orientation_arrows(Filenumber, Output_directoryCSV, ax, max_x, max_y)
            ax.set_title(f"{WhatPlot} Heatmap for File {Filenumber} with Angular Overlay")
        except Exception as e:
            print(f"Angular overlay failed: {e}")

    plt.tight_layout()
    plt.show()
    
    # --------------------------------------
    # Histogram before/after extra filtering
    # --------------------------------------
    vals_before = values_all[np.isfinite(values_all)]
    vals_after = values[np.isfinite(values)]

    if vals_before.size and vals_after.size:
        fig_h, ax_h = plt.subplots(figsize=(7, 5))
        ax_h.hist(vals_before, bins=50, alpha=0.5, label="All (above main threshold)")
        ax_h.hist(vals_after,  bins=50, alpha=0.5, label="After perc/max filters")
        ax_h.set_xlabel(column)
        ax_h.set_ylabel("Count")
        ax_h.set_title(f"Distribution of {column} (File {Filenumber})")
        ax_h.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("[heatmap] Not enough valid data to plot histograms.")

    
from matplotlib.collections import LineCollection

def overlay_orientation_arrows(
    Filenumber, Output_directoryCSV, ax, max_x, max_y,
    min_rsq=0.2,
    min_len=0.05,
    max_len=1.
):
    """
    Draw orientation lines on top of the heatmap.
    Angle comes from peak positions.
    Line length is determined by AP (alignment parameter):
        length = min_len + norm(AP) * (max_len - min_len)
    """

    ichi_csv = os.path.join(Output_directoryCSV, f"{Filenumber} IChi_fitting.csv")
    if not os.path.exists(ichi_csv):
        print(f"[AngularOverlay] IChi file not found: {ichi_csv}")
        return

    df = pd.read_csv(ichi_csv)
    if len(df) < 5:
        print("[AngularOverlay] IChi file has fewer than 5 rows. Skipping overlay.")
        return

    # Column helpers (support new/old names)
    def pick(*names):
        for n in names:
            if n in df.columns:
                return n
        return None

    col_p1 = pick('wm1_p1', 'peak_position', 'peak_position_first')
    col_p2 = pick('wm1_p2', 'peak_position2', 'peak_position_second')  # optional fallback
    col_ap = pick('AP_WM', 'AP', 'alignment_param', 'alignment_parameter')

    if col_p1 is None and col_p2 is None:
        print("[AngularOverlay] No peak position columns found.")
        return
    if col_ap is None:
        print("[AngularOverlay] AP column not found. Expected name 'AP'.")
        return

    rows = []
    for _, r in df.iterrows():
        xi, yi = int(r['x']), int(r['y'])
        if not (0 <= xi < max_x and 0 <= yi < max_y):
            continue

        # Get angle (choose smaller peak, exactly like before)
        a1 = r.get(col_p1, np.nan)
        a2 = r.get(col_p2, np.nan)

        if np.isnan(a1) and np.isnan(a2):
            continue

        a1 = r.get(col_p1, np.nan)
        a2 = r.get(col_p2, np.nan)
        
        if np.isnan(a1) and np.isnan(a2):
            continue
        angle_deg = (float(a1) if np.isfinite(a1) else float(a2)) % 360.0
        
        rsq = r.get("rsq_gaussian_fit", 1.0)
        if pd.isna(rsq) or rsq < min_rsq:
            continue

        ap = r.get(col_ap, np.nan)
        if pd.isna(ap):
            continue

        rows.append((xi, yi, angle_deg, float(ap)))

    if not rows:
        print("[AngularOverlay] No valid rows for orientation overlay.")
        return

    # ------------------------------
    # Normalise AP → line length
    # ------------------------------
    APs = np.array([ap for *_, ap in rows], float)
    APmin, APmax = np.nanmin(APs), np.nanmax(APs)

    if APmax > APmin:
        APnorm = (APs - APmin) / (APmax - APmin)
    else:
        APnorm = np.full_like(APs, 0.5)

    lengths = min_len + APnorm * (max_len - min_len)

    # Direction
    angles = np.deg2rad([ang for _, _, ang, _ in rows])
    dx = lengths * np.sin(angles)
    dy = lengths * np.cos(angles)

    # Build line segments centered at pixels
    xs = np.array([x for x, *_ in rows], float) + 0.5
    ys = np.array([y for _, y, *_ in rows], float) + 0.5
    x0 = xs - dx / 2.0; y0 = ys - dy / 2.0
    x1 = xs + dx / 2.0; y1 = ys + dy / 2.0

    segments = np.stack(
        [np.column_stack([x0, y0]), np.column_stack([x1, y1])],
        axis=1
    )

    lc = LineCollection(segments, colors='gray', linewidths=0.8, zorder=10)
    ax.add_collection(lc)
    print("min:", lengths.min(), "max:", lengths.max(), "mean:", lengths.mean())


    print(
        f"[AngularOverlay] Plotted {len(rows)} orientation lines "
        f"(AP ∈ [{APmin:.3g},{APmax:.3g}], lengths ∈ [{lengths.min():.2f},{lengths.max():.2f}])."
    )

def _plot_modelfit_heatmap(df, ax, x_col="x", y_col="y", fit_col="fit_mode",
                           title="Model fit (categorical)"):
    """
    Plot a categorical heatmap of model choice per (x,y).
    Colors:
      none=black, skipped=grey, Norm=red, skew=blue, WM=green, blanks=white.
    """
    # Normalise strings and map to integer codes
    def _norm(v):
        if pd.isna(v): return np.nan
        v = str(v).strip()
        if v == "": return np.nan
        vlow = v.lower()
        if vlow == "none": return 0
        if vlow == "skipped": return 1
        if vlow in ("norm", "gauss", "gaussian", "normal"): return 2
        if vlow in ("skew", "skewnorm", "skewed", "skewedgauss"): return 3
        if vlow in ("wm", "weightedmoment", "weighted_moment"): return 4
        return np.nan  # unknown -> blank

    if fit_col not in df.columns:
        raise ValueError(f"Expected column '{fit_col}' not found in dataframe.")
    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Expected columns '{x_col}', '{y_col}' not found in dataframe.")

    df_local = df[[x_col, y_col, fit_col]].copy()
    df_local["_modelfit_code"] = df_local[fit_col].map(_norm)

    # Pivot to 2D grid
    pivot = df_local.pivot_table(index=y_col, columns=x_col, values="_modelfit_code", aggfunc="first")

    # Colormap: 0=black, 1=grey, 2=red, 3=blue, 4=green; NaN -> white
    cmap = ListedColormap(["black", "grey", "red", "blue", "green"]).with_extremes(bad="white")
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], ncolors=5, clip=False)

    im = ax.imshow(pivot.values, origin="lower", cmap=cmap, norm=norm, aspect="equal")

    # Axis ticks (sparse if many)
    x_vals = pivot.columns.to_numpy()
    y_vals = pivot.index.to_numpy()

    def _nice_ticks(vals, max_ticks=10):
        if len(vals) <= max_ticks:
            idx = np.arange(len(vals))
        else:
            idx = np.linspace(0, len(vals) - 1, num=max_ticks, dtype=int)
        return idx, vals[idx]

    xti, xtl = _nice_ticks(x_vals)
    yti, ytl = _nice_ticks(y_vals)

    ax.set_xticks(xti); ax.set_xticklabels([f"{v}" for v in xtl], rotation=45, ha="right")
    ax.set_yticks(yti); ax.set_yticklabels([f"{v}" for v in ytl])
    ax.set_xlabel(x_col); ax.set_ylabel(y_col)
    ax.set_title(title)

    # Legend
    legend_patches = [
        Patch(facecolor="black", edgecolor="black", label="none"),
        Patch(facecolor="grey", edgecolor="grey", label="skipped"),
        Patch(facecolor="red", edgecolor="red", label="Norm"),
        Patch(facecolor="blue", edgecolor="blue", label="skew"),
        Patch(facecolor="green", edgecolor="green", label="WM"),
        Patch(facecolor="white", edgecolor="black", label="(blank)"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", frameon=True, title="fit_mode")

    return im, pivot
