#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 14:10:37 2025

@author: lauraforster
"""
import h5py
from h5py import File
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from Utils import progress_bar
from scipy.ndimage import gaussian_filter1d
from matplotlib.backends.backend_pdf import PdfPages
import time
import re
import logging
from itertools import product
import gc
import warnings
from lmfit.models import GaussianModel, ConstantModel
from scipy.signal import peak_widths



logging.getLogger("pyFAI").setLevel(logging.ERROR)
warnings.filterwarnings(
    "ignore",
    message="Using UFloat objects with std_dev==0 may give unexpected results.",
    module="uncertainties.core"
)
warnings.filterwarnings(
    "ignore",
    message="AffineScalarFunc.error_components()",
    category=FutureWarning
)

warnings.filterwarnings(
    "ignore",
    message="AffineScalarFunc.derivatives()",
    category=FutureWarning
)

# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------ IQ Fitting Functions ------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------
# -------    Initialisation    ---------
# ------------------------------------------------------------------------------------------------------------------------

def CheckBSD(scan_no, Output_directoryCSV, base_output_path, Output_directorybsd, sample_loc, file_start, file_end, identifier):
    bsd_file_path = os.path.join(Output_directorybsd, f'BSDiodes_data_{scan_no}.npz')
    sample_name = f"i22-{scan_no}.nxs"
    sample_path = os.path.join(sample_loc + sample_name)

    bs2diodeAll, arrayShape = load_or_extract_bsd(scan_no, bsd_file_path, sample_path, identifier, file_start, file_end, sample_loc)
    print(f"BSDiodes shape: {bs2diodeAll.shape}, total points: {bs2diodeAll.size}, expected: {arrayShape[0] * arrayShape[1]}")

    print('\n')
    ny, nx = arrayShape[0], arrayShape[1]
    print('Checking BSDiode Shape:', 'x:', ny, 'y:', nx)
    return

def CheckIqCSV(Filenumber, Output_directoryCSV):
    iq_path = os.path.join(Output_directoryCSV, f"{Filenumber} IQ_fitting.csv")
    iq_df = pd.read_csv(iq_path)
    # Step 2: Extract coordinates
    if not {'x', 'y'}.issubset(iq_df.columns):
        print("IQ CSV missing required 'x' or 'y' columns.")
        return
    
    x = iq_df['x'].astype(int)
    y = iq_df['y'].astype(int)
    print('Checking Iq CSV shape:', np.max(y)+1, np.max(x)+1)
    
def extract_bsd_data_split(start, end, sample_loc):
    diode_all = []
    lengths = []
    for file_number in range(start, end + 1):
        sample_path = os.path.join(sample_loc, f"i22-{file_number}.nxs")
        with File(sample_path, "r") as f:
            bsd = f["entry/BSDIODES/data"]  # shape: (frames, channels, 2)
            mean_vals = np.mean(bsd[:, :, 1], axis=1)  # average over diode channels
            diode_all.append(mean_vals)  # shape: (n_frames_per_file,)
            lengths.append(len(mean_vals))
            # print(f"File: i22-{file_number}.nxs → {len(mean_vals)} frames")
        
    
    # Determine the maximum number of frames across all files.
    max_frames = max(lengths)
    n_files = len(diode_all)
    print(f"Maximum frames per file: {max_frames}, Number of files: {n_files}")
    
    # Pad each 1D array to have length = max_frames. (Padding with NaN)
    padded = [
        np.pad(arr, (0, max_frames - len(arr)), mode='constant', constant_values=np.nan)
        for arr in diode_all
    ]
   
    # Stack the padded arrays into a 2D array
    bs2diodeAll = np.stack(padded, axis=0)
    arrayShape = (n_files, max_frames)
    print(f"BSDiodes shape from file: {bs2diodeAll.shape}, expected shape: {n_files}, {max_frames}")
    
    return bs2diodeAll, arrayShape

def extract_bsd_data(scan_no, sample_path):
    SAMPLE_PATH = Path(sample_path)

    with h5py.File(SAMPLE_PATH, 'r') as sample_file:
        entry = list(sample_file.keys())[0]
        # arrayShape = sample_file[entry + "/SAXS/data"].shape
        arrayShape = sample_file[entry + "/BSDIODES/data"].shape[:2]  # (ny, nx)

        bs2diodeAll = []

        for xpos in range(arrayShape[0]):
            for ypos in range(arrayShape[1]):
                frames_trans_flux = np.mean(sample_file[entry + "/BSDIODES/data"][xpos, ypos, :, 1])
                bs2diodeAll.append(frames_trans_flux)
    
    return np.asarray(bs2diodeAll).reshape(arrayShape), arrayShape

def extract_number(filename):
    match = re.search(r'_(\d+)\.dat$', filename)
    if match:
        return int(match.group(1))
    return float('inf')

def load_or_extract_bsd(scan_no, bsd_file_path, sample_path, identifier=None, file_start=None, file_end=None, sample_loc=None):
    start_time_BSD = time.time()
    print('\n')
    bsd_dir = os.path.dirname(bsd_file_path)
    os.makedirs(bsd_dir, exist_ok=True)

    regenerate = False

    if os.path.exists(bsd_file_path):
        print(f"Extracting BSDiodes data from {bsd_file_path} (.npz file)")
        data = np.load(bsd_file_path)
        bs2diodeAll = data['bs2diodeAll']
        arrayShape = data['arrayShape']

        expected_shape = (arrayShape[0], arrayShape[1])
        expected_total = arrayShape[0] * arrayShape[1]
        actual_shape = bs2diodeAll.shape
        actual_total = bs2diodeAll.size
        
        if actual_total != expected_total:
            print(f"❌ BSDiodes size mismatch — regenerating: expected {expected_total}, got {actual_total}")
            regenerate = True
        elif actual_shape != expected_shape:
            try:
                bs2diodeAll = bs2diodeAll.reshape(expected_shape)
            except Exception as e:
                regenerate = True
        if bs2diodeAll.shape != expected_shape:
            print(f"⚠️  Final BSDiodes shape mismatch: got {bs2diodeAll.shape}, expected {expected_shape}")
    else:
        regenerate = True

    if regenerate:
        
        if identifier == 'Split':
            
            bs2diodeAll, arrayShape = extract_bsd_data_split(file_start, file_end, sample_loc)
        else:
            bs2diodeAll, arrayShape = extract_bsd_data(scan_no, sample_path)

        np.savez(bsd_file_path, bs2diodeAll=bs2diodeAll, arrayShape=arrayShape)
        mid_time_BSD = time.time() 
        elapsed_midtime_BSD = mid_time_BSD - start_time_BSD
        minutes, seconds = divmod(elapsed_midtime_BSD, 60)
        print(f"✅ BSDiodes saved to {bsd_file_path}, Time: {int(minutes)}m {np.round(seconds, 2)}s")
        bs2diodeAll = bs2diodeAll.reshape(arrayShape) 
    return bs2diodeAll, arrayShape

# ------------------------------------------------------------------------------------------------------------------------
# -------    Fitting Iq    ---------
# ------------------------------------------------------------------------------------------------------------------------

def ProcessIQFitting(scan_no, Output_directoryCSV, base_output_path, Output_directorybsd, sample_loc,
                     file_start, file_end, identifier, order_no, x_peak_wid, xborderLL, xborderLR,
                     xborderRL, xborderRR, order_position, minimum_area_threshold, rsq_min,
                     A, b, d, PlotIqFit, xcoords, ycoords):

    TEST_MODE = (int(scan_no) == 1111)
        
    df_lookup, moment_mat = _load_lookup_table()
    
    def _legend_if_any(ax=None):
        """Only add a legend if there are labeled artists."""
        ax = ax or plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        # ignore private labels starting with underscore (Matplotlib behavior)
        labels = [lb for lb in labels if lb and not lb.startswith('_')]
        if labels:
            ax.legend()

    def _append_skipped():
        area_third_raw.append(np.nan)
        
        pct_above_baseline_list.append(np.nan)
        max_intensity_list.append(np.nan)
        
        firstmoment.append(np.nan)
        secondmoment.append(np.nan)
        thirdmoment.append(np.nan)
        skewness.append(np.nan)
        
        q0.append(np.nan)
        deltaq0.append(np.nan)
        wMu.append(np.nan)
        firstmoment_lu.append(np.nan)
        D_period.append(np.nan)
        secondmoment_lu.append(np.nan)
        thirdmoment_lu.append(np.nan)
        skewness_lu.append(np.nan)
        
        D_period_lu.append(np.nan)
        peak_width.append(np.nan)
        wp.append(np.nan)
        peak_amplitude.append(np.nan)
        fibril_radius.append(np.nan)
        
    
    def _hist_with_gaussian(values, title, xlabel, pdf_pages, bins=50):
        # calculates a histogram for the PDF output
        from scipy.stats import norm  # local import to avoid touching your global imports
        vals = np.asarray(values, float)
        vals = vals[np.isfinite(vals)]
        if vals.size < 5:
            return
        mu, sigma = norm.fit(vals)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(vals, bins=bins, density=True, alpha=0.6, edgecolor='k')
        xs = np.linspace(vals.min(), vals.max(), 400)
        ax.plot(xs, norm.pdf(xs, mu, sigma), lw=2, label=f'Gaussian fit μ={mu:.3g}, σ={sigma:.3g}')
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.legend()
        pdf_pages.savefig(fig)
        plt.close(fig)

    def _append_fit(area, pct_above, max_intensity, wm_vals, lu_vals, calc_vals):
        area_third_raw.append(np.round(area, 6))
        
        pct_above_baseline_list.append(pct_above)
        max_intensity_list.append(max_intensity)
        
        firstmoment.append(wm_vals.get("firstmoment", np.nan))
        secondmoment.append(wm_vals.get("secondmoment", np.nan))
        thirdmoment.append(wm_vals.get("thirdmoment", np.nan))
        skewness.append(wm_vals.get("skewness", np.nan))
    
        q0.append(lu_vals.get("q0", np.nan))
        deltaq0.append(lu_vals.get("deltaq0", np.nan))
        wMu.append(lu_vals.get("wMu", np.nan))
        firstmoment_lu.append(lu_vals.get("firstmoment_lu", np.nan))
        D_period.append(calc_vals.get("D_period", np.nan))
        secondmoment_lu.append(lu_vals.get("secondmoment_lu", np.nan))
        thirdmoment_lu.append(lu_vals.get("thirdmoment_lu", np.nan))
        skewness_lu.append(lu_vals.get("skewness_lu", np.nan))
    
        D_period_lu.append(calc_vals.get("D_period_lu", np.nan))
        peak_width.append(calc_vals.get("peak_width", np.nan))
        wp.append(calc_vals.get("wp", np.nan))
        peak_amplitude.append(calc_vals.get("peak_amplitude", np.nan))
        fibril_radius.append(calc_vals.get("fibril_radius", np.nan))

    def _finalise_fig(fig):
        """Send this figure either to PDF or to the screen, then close it."""
        if PlotIqFit:
            plt.show()
        else:
            pdf_pages.savefig(fig)
        plt.close(fig)
        
    def _plot_skipped_peak(x_full, y_full, x_idx, y_idx, reason):
        """
        Plot the raw peak-region data for a skipped point in a distinct colour,
        with a title explaining why it was skipped.
        """
        fig, ax = plt.subplots()
        if y_full is not None and len(y_full):
            try:
                # Cut to the same peak window definition used elsewhere
                x_whole, y_whole, *_ = Cutting(
                    order_position,
                    x_full, y_full,
                    xborderLL, xborderLR, xborderRL, xborderRR,
                    x_peak_wid
                )
                if len(x_whole):
                    y_whole  = gaussian_filter1d(y_whole,  2)
                    ax.plot(x_whole, y_whole, color='tab:gray', label='raw (skipped)')
                    ax.scatter(x_whole, y_whole, color='black',s=5, label='raw (skipped)')
                    _legend_if_any(ax)
            except Exception:
                # If Cutting fails for any reason, just fall back to an empty plot with the reason
                pass
        ax.set_title(f"SKIPPED ({x_idx},{y_idx}): {reason}")
        ax.set_ylim(0,0.2)
        _finalise_fig(fig)

    # read in various file paths
    dat_nxs_file = f'{base_output_path}/i22-{scan_no}/i22-{scan_no}_iq.nxs'
    bsd_file_path = os.path.join(Output_directorybsd, f'BSDiodes_data_{scan_no}.npz')
    sample_name = f"i22-{scan_no}.nxs"
    sample_path = os.path.join(sample_loc + sample_name)
    outputname = os.path.join(Output_directoryCSV, f"{scan_no} IQ_fitting.csv")

    # make some directories
    os.makedirs(Output_directoryCSV, exist_ok=True)
    os.chdir(Output_directoryCSV)
    pdf_pages = PdfPages(os.path.join(Output_directoryCSV, f"{scan_no} IQ_fitting_outputplots.pdf"))

    bs2diodeAll, arrayShape = load_or_extract_bsd(scan_no, bsd_file_path, sample_path, identifier, file_start, file_end, sample_loc)

    # storage
    ixt, iyt = [], []
    total_saxs_raw, total_saxs_norm = [], []
    
    # legacy single set (kept for Normal/Skew modes)
    area_third_raw, area_third_norm = [], []
    
    pct_above_baseline_list, max_intensity_list = [],[]
    
    # Initial WM calculated values
    firstmoment, secondmoment, thirdmoment, skewness = [],[],[],[]
    
    # from lookup table
    q0, deltaq0, wMu = [],[],[]
    firstmoment_lu, D_period, secondmoment_lu, thirdmoment_lu, skewness_lu = [],[],[], [], []
    
    # Calculate values
    D_period_lu, peak_width, peak_amplitude, wp, fibril_radius = [],[],[],[],[]
    
    # keep track of percentage completed 
    count_exp = count_line = count_noneBS = 0
    count_prebaseline_skip = 0
    count_total = 0
    
    processed = 0
    start = time.time()

    with h5py.File(dat_nxs_file, 'r') as nxs_file:
        iq_data = nxs_file['iq'][:]
        coords = nxs_file['coords'][:]
        num_frames = iq_data.shape[0]
        q_values = iq_data[0, :, 0]

        if coords.shape[0] != num_frames:
            raise ValueError(f"Coordinate array shape mismatch: {coords.shape[0]} coords vs {num_frames} frames.")

        if xcoords is not None and ycoords is not None:
            requested = set(zip(ycoords, xcoords))  # coords are (y, x)
            requested = set(product(ycoords, xcoords))
            frame_indices = [i for i, (y, x) in enumerate(coords) if (y, x) in requested]
        else:
            frame_indices = list(range(num_frames))

        # PASS 1: total SAXS raw + normalise scan-level
        total_raw_by_idx = np.full(num_frames, np.nan, dtype=float)
        
        total = len(frame_indices)  # frames we’ll iterate over
        
        for i in frame_indices:
            x = q_values
            y = iq_data[i, :, 1]
            if y is None or len(y) == 0:
                continue
            # print("x:", x)
            # print("y:", y)
            # print("x type:", type(x), "y type:", type(y))
            # print("x shape:", np.shape(x), "y shape:", np.shape(y))

            area_all = float(np.trapezoid(y, x))
            if area_all < 0:
                area_all = 0.0
            total_raw_by_idx[i] = area_all

        valid_mask = ~np.isnan(total_raw_by_idx)
        vals = total_raw_by_idx[valid_mask]
        
        if vals.size:
            # max point + coords
            idx_max_local = np.nanargmax(total_raw_by_idx)
            y_max, x_max = coords[idx_max_local]
            # print(f"[Total SAXS] Max value: {total_raw_by_idx[idx_max_local]:.6g} at (x={x_max}, y={y_max})")

            # mean of top 1%
            val=0.05
            k = max(1, int(val * vals.size))
            top_mean = float(np.mean(np.partition(vals, -k)[-k:]))
            # print(f"[Total SAXS] Mean of top {val*100}% values (n={k}): {top_mean:.6g}")

            # histogram + Gaussian fit into the PDF
            _hist_with_gaussian(vals, title=f"Total SAXS distribution — scan {scan_no}",
                                xlabel="integrated intensity", pdf_pages=pdf_pages)

            # # normalise (current linear 0..1)
            tmin = float(np.nanmin(vals))
            tmax = top_mean
            denom = (tmax - tmin) if (tmax - tmin) > 0 else 1.0
            # total_norm_by_idx = (total_raw_by_idx - tmin) / denom
            total_norm_by_idx = np.clip((total_raw_by_idx - tmin) / denom, 0, 1)

        else:
            total_norm_by_idx = total_raw_by_idx

        # PASS 2: do per-frame work (skip/fit), while ALWAYS appending one row
        for frame_index in frame_indices:
            count_total += 1
            y_idx, x_idx = coords[frame_index]
            x = np.array(q_values)
            y = np.array(iq_data[frame_index, :, 1])
            
            processed += 1
            start = progress_bar(processed, total,
                                 prefix=f"[{scan_no}] IQ fitting",
                                 start_time=start)

            # coordinates & totals (append early to keep lengths aligned)
            ixt.append(int(x_idx))
            iyt.append(int(y_idx))
            total_raw = total_raw_by_idx[frame_index]
            total_norm = total_norm_by_idx[frame_index]
            total_saxs_raw.append(np.round(total_raw, 6))
            total_saxs_norm.append(np.round(total_norm, 6))

            # threshold on scan-level total SAXS
            # if not np.isfinite(total_norm):
            #     total_norm = 0.0
            # qualifies = (total_norm >= local_min_area_threshold)

            # if not np.isfinite(total_norm) or y is None or len(y) == 0:
            #     # no data → mark skipped
            #     _plot_skipped_peak(x, y,x_idx, y_idx,reason="no data / invalid total SAXS")
            #     _append_skipped()
            #     continue

            # if not qualifies:
            #     _plot_skipped_peak(x, y,x_idx, y_idx,reason=f"below SAXS threshold ({total_norm:.3g} < {local_min_area_threshold:.3g})")

            #     _append_skipped()
            #     continue
            
            # Only skip if the frame genuinely has no I(q) data
            if y is None or len(y) == 0:
                _plot_skipped_peak(x, y, x_idx, y_idx, reason="no data")
                _append_skipped()
                count_prebaseline_skip += 1
                continue
            
            try:
                diode_val = bs2diodeAll[y_idx, x_idx]
            except IndexError:
                print(f"BSD index out of bounds for ({y_idx}, {x_idx}), skipping...")
                _plot_skipped_peak(x, y,x_idx, y_idx,reason="BSD index out of bounds")

                _append_skipped()
                count_prebaseline_skip += 1
                continue
    
            fig, ax = plt.subplots()
           
            baselineflag, area, pct_above, max_intensity, wm_vals, lu_vals, calc_vals, x, y, x_peak, y_baseline_corrected, bkg_mode = process_order_peak(
                order_no, order_position, df_lookup, moment_mat, diode_val, x, y,
                xborderLL, xborderLR, xborderRL, xborderRR,
                x_peak_wid, minimum_area_threshold,  
                rsq_min, A, b, d, TEST_MODE, x_idx, y_idx, ax=ax
            )
            
            if bkg_mode == "exp":
                count_exp += 1
            elif bkg_mode == "lin":
                count_line += 1
            elif bkg_mode == "none":
                count_noneBS += 1
            else:  # "pre" or anything else
                count_prebaseline_skip += 1

            if baselineflag == 0:
                ax.set_title(f"{x_idx, y_idx}  (baseline fallback)")
            else:
                ax.set_title(f"{x_idx, y_idx}")
            
            if TEST_MODE:
                ax.plot(x_peak, y_baseline_corrected, color='blue', label='TEST raw data', zorder=0)
                ax.scatter(x_peak, y_baseline_corrected, color='blue')

            plt.title(f'{x_idx, y_idx}')
            _legend_if_any(ax)
            _finalise_fig(fig)

            _append_fit(area, pct_above, max_intensity, wm_vals, lu_vals, calc_vals)

    area_third_arr = np.array(area_third_raw, dtype=float)
    valid_c = np.isfinite(area_third_arr)
    
    # default out
    area_third_norm = [np.nan] * len(area_third_raw)
    
    if np.any(valid_c):
        vals = area_third_arr[valid_c]
    
        # (keep your existing diagnostics)
        idx_max_area = np.nanargmax(area_third_arr)
        x_m, y_m = ixt[idx_max_area], iyt[idx_max_area]
        # print(f"[Collagen area] Max value: {area_third_arr[idx_max_area]:.6g} at (x={x_m}, y={y_m})")
    
        # plot histogram of the raw distribution into the PDF
        _hist_with_gaussian(
            vals,
            title=f"Collagen (third-order) area distribution — scan {scan_no}",
            xlabel="area under third-order peak",
            pdf_pages=pdf_pages
        )
    
        # --- cap the top 5% at their mean ---
        p = 0.05
        k = max(1, int(np.ceil(p * vals.size)))
        top5_mean = float(np.mean(np.partition(vals, -k)[-k:]))
        num_capped = int(np.sum(vals > top5_mean))
    
        # print(f"[Collagen area] Top {int(p*100)}% cap mean: {top5_mean:.6g} (capping {num_capped} values)")
    
        vals_capped = np.minimum(vals, top5_mean)
    
        # --- distribution of remaining (capped) points ---
        mu = float(np.mean(vals_capped))
        sigma = float(np.std(vals_capped, ddof=0))
        # print(f"[Collagen area] μ (capped) = {mu:.6g}, σ (capped) = {sigma:.6g}")

        # --- normalise using ±3σ around μ, constrained to [0, top5_mean] ---
        lo = max(0.0, mu - 3.0 * sigma)
        hi = min(top5_mean, mu + 3.0 * sigma)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            q_lo = float(np.percentile(vals_capped, 5))
            q_hi = float(np.percentile(vals_capped, 95))
            lo, hi = max(0.0, q_lo), min(top5_mean, q_hi)
            if hi <= lo:
                hi = lo + 1e-9
        
        # scale and CLIP to [0, 1], then give a tiny epsilon floor
        norm_vals = (vals_capped - lo) / (hi - lo)
        norm_vals = np.clip(norm_vals, 0.0, 1.0)
        norm_vals = np.maximum(norm_vals, 1e-9)   # avoids filtering out at > threshold
        # print(f"[Collagen area] Normalisation window: lo={lo:.6g}, hi={hi:.6g}")

        area_third_norm_arr = np.full_like(area_third_arr, np.nan, dtype=float)
        area_third_norm_arr[valid_c] = norm_vals
        area_third_norm = area_third_norm_arr.tolist()


    else:
        area_third_norm = [np.nan] * len(area_third_raw)
    
    
    pdf_pages.close()
        
    # Base (legacy) DataFrame (always present)
    df_dict = {
        'x': ixt,
        'y': iyt,
        
        'total SAXS intensity': total_saxs_raw,
        'total_SAXS_norm_0_1': total_saxs_norm,
        'area under third order curve': area_third_raw,
        'collagen_third_norm_0_1': area_third_norm,
        
        'percentage_above_baseline': pct_above_baseline_list,
        'max_recorded_intensity':    max_intensity_list,
        
        'firstmoment': firstmoment,
        'D_period': D_period,
        'secondmoment': secondmoment,
        'thirdmoment': thirdmoment,
        'skewness': skewness,
        
        'q0': q0,
        'deltaq0': deltaq0,
        'wMu': wMu,
        'firstmoment_lu': firstmoment_lu,
        'secondmoment_lu': secondmoment_lu,
        'thirdmoment_lu': thirdmoment_lu,
        'skewness_lu': skewness_lu,
        
        'D_period_lu': D_period_lu,
        'peak_width': peak_width,
        'wp': wp,
        'peak_amplitude': peak_amplitude,
        'fibril_radius': fibril_radius,
    }
    

    df = pd.DataFrame(df_dict)
    df.to_csv(outputname, index=False)

    denom = max(count_total, 1)
    print(
        f"Iq Fitting Finished — "
        f"Exponential BG: {100*count_exp/denom:.2f}%, "
        f"Linear BG: {100*count_line/denom:.2f}%, "
        f"No/other BG: {100*count_noneBS/denom:.2f}%"
    )

    progress_bar(total, total, prefix=f"[{scan_no}] IQ fitting",
                 start_time=start, end=True)

    try:
        return x, y, x_peak, y_baseline_corrected
    except UnboundLocalError:
        return None, None, None, None

def process_order_peak(order_no, order_position, df_lookup, moment_mat, diode_val, x, y,
                       xborderLL, xborderLR, xborderRL, xborderRR, x_peak_wid,
                       minimum_area_threshold, rsq_min, A, b, d, TEST_MODE, x_idx, y_idx, ax):
    baselineflag = 1
    # --- cut windows & baseline correction (unchanged) ---
    x_peak, y_peak, x_sides, y_sides, x_L, x_R, y_L, y_R = Cutting(
        order_position, x, y, xborderLL, xborderLR, xborderRL, xborderRR, x_peak_wid
    )
    
    def _downsample_xy(x_arr, y_arr, factor=4):
        """
        Bin-average x and y in chunks of 'factor'.
        If there are fewer than 'factor' points, return as-is.
        """
        x_arr = np.asarray(x_arr, dtype=float)
        y_arr = np.asarray(y_arr, dtype=float)

        m = min(len(x_arr), len(y_arr))
        if m == 0 or m < factor:
            return x_arr, y_arr

        # Trim to a multiple of 'factor'
        m_trunc = (m // factor) * factor
        x_arr = x_arr[:m_trunc]
        y_arr = y_arr[:m_trunc]

        x_ds = x_arr.reshape(-1, factor).mean(axis=1)
        y_ds = y_arr.reshape(-1, factor).mean(axis=1)
        return x_ds, y_ds

    # Downsample peak window, side bands and the left/right subsets
    x_peak,  y_peak  = _downsample_xy(x_peak,  y_peak,  factor=4)
    x_sides, y_sides = _downsample_xy(x_sides, y_sides, factor=4)
    x_L,     y_L     = _downsample_xy(x_L,     y_L,     factor=4)
    x_R,     y_R     = _downsample_xy(x_R,     y_R,     factor=4)

    # smooth lightly before baselining
    y_peak_filtered  = gaussian_filter1d(y_peak,  2)
    y_sides_filtered = gaussian_filter1d(y_sides, 2)
        
    # --- after BaselineCorrection(...) ---
    # >>> SKIP BASELINE COMPLETELY IN TEST MODE <<<
    if TEST_MODE:
        # just pass the raw (smoothed) peak window through as “baseline corrected”
        y_baseline_corrected = y_peak_filtered
        y_bgr_fit = np.zeros_like(y_peak_filtered)  # cosmetic (flat baseline)
        rsq_bkg = 1.0
        bkg_mode = "skip"
        passed = 1
    else:
        # --- normal baseline correction path ---
        y_baseline_corrected, y_bgr_fit, passed, rsq_bkg, bkg_mode = BaselineCorrection(
            x_sides, y_sides_filtered, x_peak, y_peak_filtered, A, b, d, x_L, x_R, y_L, y_R, ax
        )
    # ------------------------------------------------------------------
    # Basic checks: if Cutting gave nothing, we truly cannot proceed
    # ------------------------------------------------------------------
    if (not hasattr(x_peak, "__len__")) or (len(x_peak) < 5):
        baselineflag = 0
        return baselineflag, 0.0, 0.0, 0.0, {}, {}, {}, x, y, np.array([]), np.array([]),  "pre"
    
    # Ensure arrays
    x_peak = np.asarray(x_peak, dtype=float)
    y_peak_filtered = np.asarray(y_peak_filtered, dtype=float)
    
    # If baseline correction failed, FALL BACK to a simple straight-line baseline
    # between the ends of the peak window.
    if passed == 0 or (not hasattr(y_baseline_corrected, "__len__")) or (len(y_baseline_corrected) != len(x_peak)):
        baselineflag = 0
        y0 = float(y_peak_filtered[0])
        y1 = float(y_peak_filtered[-1])
        bline = np.linspace(y0, y1, x_peak.size)
        y_baseline_corrected = y_peak_filtered - bline
    else:
        baselineflag = 1
        y_baseline_corrected = np.asarray(y_baseline_corrected, dtype=float)

    # ------------------------------------------------------------------
    # Area under baseline-corrected curve
    # If negative, force to 0, but still compute other metrics.
    # ------------------------------------------------------------------
    y_bc = np.asarray(y_baseline_corrected, dtype=float)
    finite = np.isfinite(x_peak) & np.isfinite(y_bc)
    if np.count_nonzero(finite) < 5:
        baselineflag = 0
        return baselineflag, 0.0, 0.0, 0.0, {}, {}, {}, x, y, x_peak, y_baseline_corrected
    
    x_f = x_peak[finite]
    y_f = y_bc[finite]
    
    # raw signed area (can be negative if baseline overshoots)
    try:
        area_signed = float(np.trapezoid(y=y_f, x=x_f))
    except Exception:
        area_signed = 0.0
    
    # output area: never negative (but we keep y_f unchanged for WM)
    area = max(0.0, area_signed)
    
    y_pos = np.clip(y_f, 0.0, None)
    y_neg = np.clip(y_f, None, 0.0)
    
    try:
        area_pos = float(np.trapezoid(y=y_pos, x=x_f))
    except Exception:
        area_pos = 0.0
    try:
        area_neg = float(np.trapezoid(y=y_neg, x=x_f))
    except Exception:
        area_neg = 0.0
    
    total_abs_area = area_pos - area_neg
    pct_above = (100.0 * area_pos / total_abs_area) if total_abs_area > 0 else 0.0
    max_intensity = float(np.nanmax(y_f)) if y_f.size else 0.0
    # ------------------------------------------------------------------
    # 4) WeightedMoment, Lookup, Calculate_parameters as before
    # ------------------------------------------------------------------
    wm_vals = WeightedMoment(x_peak, y_baseline_corrected,  ax=ax)
    lu_vals = Lookup(wm_vals["firstmoment"], wm_vals["secondmoment"], wm_vals["thirdmoment"], wm_vals["skewness"], df_lookup, moment_mat,)
    calc_vals = Calculate_parameters(wm_vals["firstmoment"], lu_vals["q0"], lu_vals["deltaq0"], lu_vals["wMu"], lu_vals["firstmoment_lu"], lu_vals["secondmoment_lu"], lu_vals["thirdmoment_lu"], lu_vals["skewness_lu"])
            
    return baselineflag, area, pct_above, max_intensity, wm_vals, lu_vals, calc_vals, x, y, x_peak, y_baseline_corrected, bkg_mode

# ------------------------------------------------------------------------------------------------------------------------
# -------    Baseline correction    ---------
# ------------------------------------------------------------------------------------------------------------------------

def Cutting(order, x, y, xborderLL, xborderLR, xborderRL, xborderRR, x_peak_wid):
    # Perform element-wise comparisons
    x_whole = x[(x >= order - x_peak_wid) & (x < order + x_peak_wid)]
    x_L = x[(x >= order - xborderLL) & (x < order - xborderLR)]
    x_R = x[(x >= order + xborderRL) & (x < order + xborderRR)]

    y_whole = y[(x >= order - x_peak_wid) & (x < order + x_peak_wid)]
    y_L = y[(x >= order - xborderLL) & (x < order - xborderLR)]
    y_R = y[(x >= order + xborderRL) & (x < order + xborderRR)]

    x_sides = np.concatenate((x_L, x_R))
    y_sides = np.concatenate((y_L, y_R))

    return x_whole, y_whole, x_sides, y_sides, x_L, x_R, y_L, y_R


def BaselineCorrection(x_sides, y_sides, x_whole, y_whole, A, b, d,
                       x_L, x_R, y_L, y_R, ax):
    """
    Baseline correction using:
      1) Exponential + constant fit to SIDE-BANDS (preferred)
      2) Linear fit to SIDE-BANDS (fallback)
      3) Fail -> return zeros + annotate plot

    Returns:
      y_baseline_corrected, y_background_fit, ok_flag(1/0), r2, method("exp"/"lin"/"none")
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from lmfit.models import ExponentialModel, ConstantModel
    from sklearn.metrics import r2_score

    ax = ax or plt.gca()

    # -------------------------
    # Clean + basic guards
    # -------------------------
    x_sides = np.asarray(x_sides, float)
    y_sides = np.asarray(y_sides, float)
    x_whole = np.asarray(x_whole, float)
    y_whole = np.asarray(y_whole, float)

    m = np.isfinite(x_sides) & np.isfinite(y_sides)
    x_sides, y_sides = x_sides[m], y_sides[m]

    mW = np.isfinite(x_whole) & np.isfinite(y_whole)
    xW, yW = x_whole[mW], y_whole[mW]

    def _annot_fail(msg="background fit failed"):
        ax.plot(x_whole, y_whole, label="whole (no bg)")
        ax.text(0.02, 0.98, msg, transform=ax.transAxes,
                va="top", ha="left",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
        return 0, 0, 0, 0, "none"

    if x_sides.size < 5 or xW.size < 5:
        return _annot_fail("background fit failed (insufficient data)")

    # -------------------------
    # Helpers (concise)
    # -------------------------
    def _r2(y_true, y_pred):
        yt = np.asarray(y_true, float)
        yp = np.asarray(y_pred, float)
        mm = np.isfinite(yt) & np.isfinite(yp)
        if np.count_nonzero(mm) < 5:
            return -np.inf
        yt, yp = yt[mm], yp[mm]
        # if yt is constant-ish, r2_score can be unstable; just treat as bad
        if np.ptp(yt) < 1e-12:
            return -np.inf
        return float(r2_score(yt, yp))

    def _lin_fit(xs, ys):
        # returns m,c
        m_, c_ = np.polyfit(xs, ys, 1)
        return float(m_), float(c_)

    def _span_ok(y_fit, y_ref, min_frac=0.5):
        # require fit to span at least min_frac of the side-band span
        y_fit = np.asarray(y_fit, float)
        y_ref = np.asarray(y_ref, float)
        rf = float(np.ptp(y_fit[np.isfinite(y_fit)])) if np.isfinite(y_fit).any() else 0.0
        rr = float(np.ptp(y_ref[np.isfinite(y_ref)])) if np.isfinite(y_ref).any() else 0.0
        rr = max(rr, 1e-12)
        return (rf / rr) >= float(min_frac)

    def _slope_ok(m_, min_abs=1e-6):
        # your baseline is always decreasing; reject flat-ish or positive
        return (m_ < 0.0) and (abs(m_) >= float(min_abs))

    # -------------------------
    # Side-band linear guess (always)
    # -------------------------
    m_side, c_side = _lin_fit(x_sides, y_sides)
    y_side_span = float(np.ptp(y_sides))
    y_side_span = max(y_side_span, 1e-12)
    x_span = float(np.ptp(x_sides))
    x_span = max(x_span, 1e-12)

    # If sides themselves are not decreasing, don't trust exp at all
    # (still allow linear fallback because it may capture local trend)
    sides_decreasing = (m_side < 0.0)

    # -------------------------
    # 1) Exponential + constant
    # -------------------------
    # Model: y = A * exp(-x/decay) + C
    # Sensible initial guesses from sides:
    #   - C around the minimum side value
    #   - A around (max-min)
    #   - decay not too small/flat: tie it to x-span
    exp_ok = False
    best = None  # (method, y_bg, y_bc, r2)

    # Exp acceptance thresholds (tweak here)
    MIN_DECAY_FRAC = 0.20      # decay must be >= 0.2*x_span (avoid ultra-fast / weird)
    MAX_DECAY_FRAC = 20.0      # prevent totally-flat exp
    MIN_R2_EXP     = 0.20
    MIN_SPAN_FRAC  = 0.50      # fit must span at least 50% of side-band span

    if sides_decreasing:
        try:
            exp_model = ExponentialModel(prefix="exp_")
            const_model = ConstantModel(prefix="c_")
            model = exp_model + const_model

            A0 = float(np.max(y_sides) - np.min(y_sides))
            A0 = max(A0, 1e-9)
            C0 = float(np.min(y_sides))  # conservative, keeps exp from drifting up

            decay0 = x_span
            decay_min = max(1e-6, MIN_DECAY_FRAC * x_span)
            decay_max = min(1e6, MAX_DECAY_FRAC * x_span)

            params = model.make_params()
            params["exp_amplitude"].set(value=A0, min=0.0, max=float(200.0 * y_side_span))
            params["exp_decay"].set(value=decay0, min=decay_min, max=decay_max)
            params["c_c"].set(value=C0,
                              min=float(np.min(y_sides) - 3.0 * y_side_span),
                              max=float(np.max(y_sides) + 3.0 * y_side_span))

            result = model.fit(y_sides, params, x=x_sides)

            y_bg_sides = model.eval(result.params, x=x_sides)
            y_bg_whole = model.eval(result.params, x=x_whole)

            r2_exp = _r2(y_sides, y_bg_sides)

            # extra exp sanity:
            # - decay should not hit bounds hard
            decay_fit = float(result.params["exp_decay"].value)
            decay_close_to_bounds = (abs(decay_fit - decay_min) / (decay_min + 1e-12) < 0.05) or \
                                    (abs(decay_fit - decay_max) / (decay_max + 1e-12) < 0.05)

            # - exp background should not be increasing overall on the window
            m_bg, _ = _lin_fit(xW, y_bg_whole[mW] if y_bg_whole.size == y_whole.size else np.asarray(y_bg_whole)[mW])
            bg_decreasing = (m_bg < 0.0)

            exp_ok = (
                (r2_exp >= MIN_R2_EXP) and
                _span_ok(y_bg_sides, y_sides, min_frac=MIN_SPAN_FRAC) and
                (not decay_close_to_bounds) and
                bg_decreasing
            )

            if exp_ok:
                y_bc = y_whole - y_bg_whole
                best = ("exp", y_bg_whole, y_bc, r2_exp)

        except Exception:
            exp_ok = False

    # -------------------------
    # 2) Linear fallback
    # -------------------------
    # Linear acceptance thresholds (tweak here)
    MIN_R2_LIN      = 0.20
    MIN_SLOPE_ABS   = 1e-6

    if best is None:
        try:
            m_lin, c_lin = _lin_fit(x_sides, y_sides)

            y_bg_whole = (m_lin * x_whole + c_lin).astype(float)
            y_bg_sides = (m_lin * x_sides + c_lin).astype(float)

            r2_lin = _r2(y_sides, y_bg_sides)

            lin_ok = (
                _slope_ok(m_lin, min_abs=MIN_SLOPE_ABS) and
                (r2_lin >= MIN_R2_LIN) and
                _span_ok(y_bg_sides, y_sides, min_frac=MIN_SPAN_FRAC)
            )

            if lin_ok:
                y_bc = y_whole - y_bg_whole
                best = ("lin", y_bg_whole, y_bc, r2_lin)

        except Exception:
            best = None

    # -------------------------
    # 3) Plot + return
    # -------------------------
    if best is None:
        return _annot_fail("background fit failed")

    method, y_bg, y_bc, r2v = best

    ax.plot(x_whole, y_whole, label="whole")
    ax.plot(x_whole, y_bg, label=f"{method} bg")
    ax.plot(x_whole, y_bc, label=f"{method} bg subtracted")

    return y_bc, y_bg, 1, float(r2v), method

# ------------------------------------------------------------------------------------------------------------------------
# -------    Weighted Moment Calculation    ---------
# ------------------------------------------------------------------------------------------------------------------------

def WeightedMoment(x_peak, y_baseline_corrected, ax=None):

    ax = ax or plt.gca()

    x = np.asarray(x_peak, dtype=float)
    y = np.asarray(y_baseline_corrected, dtype=float)

    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], np.maximum(y[m], 0.0)

    Iy   = np.sum(y)
    
    if Iy <= 0 or not np.isfinite(Iy):
        return dict(
            firstmoment=np.nan,
            secondmoment=np.nan,
            thirdmoment=np.nan,
            skewness=np.nan,
        )

    qm1   = np.sum(x * y) / Iy               # weighted mean (center)
    qm2   = np.sum((x**2) * y) / Iy
    qm3   = np.sum((x**3) * y) / Iy
    var  = max(qm2 - qm1**2, 0.0)
    sigma = np.sqrt(var)
    
    term = qm2 - (qm1**2)
    
    if term <= 0 or not np.isfinite(term):
        return dict(
            firstmoment=np.nan,
            secondmoment=np.nan,
            thirdmoment=np.nan,
            skewness=np.nan,
        )
    
    skew_exp = (qm3 - (3*qm1*qm2) + 2*(qm1**3) ) / (term**(3/2)) #https://en.wikipedia.org/wiki/Skewness
    
    E_skew = skew_exp
    
    return dict(
        firstmoment=round(qm1, 6),
        secondmoment=round(qm2, 4),
        thirdmoment=round(qm3, 6),
        skewness=round(E_skew, 6),
    )

def _load_lookup_table():

    script_dir = Path(__file__).resolve().parent
    lookup_path = script_dir / "Lookup_Table.csv"

    if not lookup_path.exists():
        raise FileNotFoundError(f"Lookup table not found at: {lookup_path}")

    df_lookup = pd.read_csv(lookup_path)
    df_lookup["skewness"] = df_lookup["skewness"].clip(lower=0.0)

    required_cols = ["q0", "deltaQ0", "wMu",
                     "firstmoment", "secondmoment", "thirdmoment", "skewness"]
    missing = [c for c in required_cols if c not in df_lookup.columns]
    if missing:
        raise ValueError(f"Lookup table is missing columns: {missing}")

    # moments used for the distance calculation
    moment_mat = df_lookup[["firstmoment", "secondmoment", "thirdmoment", "skewness"]].to_numpy(dtype=float)

    return df_lookup, moment_mat


def Lookup(firstmoment, secondmoment, thirdmoment, skewness, df_lookup, moment_mat):
    """
    Staged lookup with priority:
      1) firstmoment (always)
      2) skewness (if usable; cap negatives to 0)
      3) secondmoment
      4) thirdmoment

    If skewness was negative originally, we cap it to 0 and *de-emphasise* skew by
    switching to (first, second, third) matching.
    """
    # --- Validate inputs ---
    fm = float(firstmoment) if np.isfinite(firstmoment) else np.nan
    sm = float(secondmoment) if np.isfinite(secondmoment) else np.nan
    tm = float(thirdmoment) if np.isfinite(thirdmoment) else np.nan
    sk = float(skewness) if np.isfinite(skewness) else np.nan

    if not np.isfinite(fm):
        return {
            "q0": np.nan, "deltaq0": np.nan, "wMu": np.nan,
            "firstmoment_lu": np.nan, "secondmoment_lu": np.nan,
            "thirdmoment_lu": np.nan, "skewness_lu": np.nan,
        }

    # Cap negative skew to 0, and decide whether to trust skew as a matching dimension
    skew_was_negative = (np.isfinite(sk) and sk < 0.0)
    if np.isfinite(sk) and sk < 0.0:
        sk = 0.0

    # --- Pull lookup columns in a known order ---
    # moment_mat may be in any order depending on how you built it previously, so be safe:
    # We'll build arrays from df_lookup directly.
    fm_lu = df_lookup["firstmoment"].to_numpy(dtype=float)
    sk_lu = df_lookup["skewness"].to_numpy(dtype=float)
    sm_lu = df_lookup["secondmoment"].to_numpy(dtype=float)
    tm_lu = df_lookup["thirdmoment"].to_numpy(dtype=float)

    # --- Stage 1: shortlist by closest firstmoment ---
    # Keep a small candidate set so refinements don't get distorted by far-away rows.
    # Size scales a bit with table size but stays bounded.
    n = fm_lu.size
    k = int(np.clip(max(25, n * 0.01), 25, 400))  # 1% of table, 25..400
    d_fm = np.abs(fm_lu - fm)
    cand_idx = np.argpartition(d_fm, k-1)[:k]

    # --- Stage 2+: rank candidates by priority ---
    # If skew was negative, do NOT use skew heavily; use 2nd/3rd instead.
    # Otherwise use skew as the 2nd priority dimension.
    # Use robust scaling so one dimension can't dominate due to units/magnitude.
    def _robust_scale(v):
        v = np.asarray(v, float)
        vv = v[np.isfinite(v)]
        if vv.size < 10:
            return 1.0
        s = float(np.nanpercentile(vv, 75) - np.nanpercentile(vv, 25))
        return s if s > 1e-12 else 1.0

    # scales on candidate subset (good enough and faster)
    s_fm = _robust_scale(fm_lu[cand_idx])
    s_sk = _robust_scale(sk_lu[cand_idx])
    s_sm = _robust_scale(sm_lu[cand_idx])
    s_tm = _robust_scale(tm_lu[cand_idx])

    # Always include firstmoment distance (already shortlisted, but still matters)
    score = (np.abs(fm_lu[cand_idx] - fm) / s_fm)

    # Next: skew if usable and provided
    if (not skew_was_negative) and np.isfinite(sk):
        score += 0.75 * (np.abs(sk_lu[cand_idx] - sk) / s_sk)  # strong, but less than fm

    # Then: second moment if provided
    if np.isfinite(sm):
        score += 0.35 * (np.abs(sm_lu[cand_idx] - sm) / s_sm)

    # Then: third moment if provided
    if np.isfinite(tm):
        score += 0.20 * (np.abs(tm_lu[cand_idx] - tm) / s_tm)

    best_local = int(np.argmin(score))
    idx = int(cand_idx[best_local])

    row = df_lookup.iloc[idx]
    return {
        "q0":              float(row["q0"]),
        "deltaq0":         float(row["deltaQ0"]),
        "wMu":             float(row["wMu"]),
        "firstmoment_lu":  float(row["firstmoment"]),
        "secondmoment_lu": float(row["secondmoment"]),
        "thirdmoment_lu":  float(row["thirdmoment"]),
        "skewness_lu":     float(row["skewness"]),
    }

def Calculate_parameters(firstmoment_exp, q0, deltaq0, wMu, firstmoment_lu, secondmoment_lu, thirdmoment_lu, skewness_lu):
    
    D_period_lu = ( (2*np.pi) / q0 ) * 3
    
    # D-period from experimental weighted mean (qm1)
    D_period = ((2*np.pi) / firstmoment_exp) * 3 if np.isfinite(firstmoment_exp) and firstmoment_exp > 0 else np.nan

    peak_width = deltaq0
    linearfactor = 200/75
    wp = wMu*q0
    peak_amplitude = 1
    fibril_radius = linearfactor*(1/wp)
    return {
        "D_period_lu":      D_period_lu,
        "D_period":    D_period,
        "peak_width":    peak_width,
        "wp":   wp,
        "peak_amplitude":peak_amplitude,
        "fibril_radius": fibril_radius,
    }
    
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------ ICHI Fitting Functions ------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------
# -------    Fit IChi data    ---------
# ------------------------------------------------------------------------------------------------------------------------

def ProcessICHIFitting(
    Filenumber, outputfolder, base_output_path, Output_directoryCSV,
    area_threshold, Plot, IChi_EntireFit,
    xcoords=None, ycoords=None
):
    print(f"\nBeginning I_Chi Fitting — {'Entire' if IChi_EntireFit else 'Rings'} mode\n")

    # DEBUG_ICH = True
    # DEBUG_ONLY_FRAMES = {(26, 0), (27, 0)}  # (x,y) pairs; empty set = print none

    centre_fn = os.path.join(base_output_path, outputfolder, f"i22-{Filenumber}_ichi.nxs")
    iqCSVfile = os.path.join(Output_directoryCSV, f"{Filenumber} IQ_fitting.csv")

    os.makedirs(Output_directoryCSV, exist_ok=True)
    pdf_path = os.path.join(Output_directoryCSV, f"{Filenumber} IChi_fitting_outputplots.pdf")
    pdf = PdfPages(pdf_path)

    try:
        iq_df = pd.read_csv(iqCSVfile)
    except FileNotFoundError:
        print(f"Error: IQ fitting file not found: {iqCSVfile}")
        pdf.close()
        return

    iq_lookup = {
        (int(r["x"]), int(r["y"])): {
            "total SAXS intensity": r.get("total SAXS intensity", np.nan),
            "total_SAXS_norm_0_1": r.get("total_SAXS_norm_0_1", np.nan),
            "area under third order curve": r.get("area under third order curve", np.nan),
            "collagen_third_norm_0_1": r.get("collagen_third_norm_0_1", np.nan),
        }
        for _, r in iq_df.iterrows()
    }

    with File(centre_fn, "r") as f:
        centre_data = f["ichi"][:]
        coords = [tuple(map(int, c)) for c in f["coords"][:]]

    chi_vals = centre_data[0, :, 0]
    n_frames = len(coords)

    # -----------------------------
    # Frame selection
    # -----------------------------
    if xcoords is not None and ycoords is not None:
        xset = set(map(int, xcoords))
        yset = set(map(int, ycoords))
        requested = set(product(yset, xset))  # coords stored as (y,x)
        frames = [i for i, c in enumerate(coords) if c in requested]
    elif xcoords is not None:
        xset = set(map(int, xcoords))
        frames = [i for i, (_, x) in enumerate(coords) if x in xset]
    elif ycoords is not None:
        yset = set(map(int, ycoords))
        frames = [i for i, (y, _) in enumerate(coords) if y in yset]
    else:
        frames = list(range(n_frames))

    # -----------------------------
    # Helpers
    # -----------------------------
    def _odd_leq(n, target):
        if n < 5:
            return 5
        t = min(target, n if n % 2 else n - 1)
        return t if t % 2 else max(5, t - 1)

    def _repair_notch(raw, chi, center_deg=180.0, halfwin_deg=6.0, dip_frac=0.25, min_depth=0.01):
        raw = np.asarray(raw, float)
        chi = np.asarray(chi, float)

        d = np.abs((chi - center_deg + 180) % 360 - 180)
        win = d <= halfwin_deg
        if np.count_nonzero(win) < 3:
            return raw, False

        sh = (d > halfwin_deg) & (d <= 2 * halfwin_deg)
        if np.count_nonzero(sh) < 4:
            return raw, False

        valley = np.nanmedian(raw[win])
        shoulder = np.nanmedian(raw[sh])

        depth = shoulder - valley
        if not np.isfinite(depth) or depth < min_depth:
            return raw, False
        if depth / (np.abs(shoulder) + 1e-12) < dip_frac:
            return raw, False

        idx = np.arange(raw.size)
        good = ~win & np.isfinite(raw)
        if np.count_nonzero(good) < 5:
            return raw, False

        fixed = raw.copy()
        fixed[win] = np.interp(idx[win], idx[good], raw[good])
        return fixed, True

    # def dbg_for(x_idx, y_idx):
    #     return DEBUG_ICH and (not DEBUG_ONLY_FRAMES or (x_idx, y_idx) in DEBUG_ONLY_FRAMES)

    def compute_stats(hv, sv):
        hv_f = hv[np.isfinite(hv)]
        if hv_f.size < 10:
            return None

        low = hv_f[hv_f <= np.nanpercentile(hv_f, 30)]
        if low.size < 5:
            low = hv_f

        b0 = float(np.nanmedian(low))
        mad = float(np.nanmedian(np.abs(low - b0)))
        sigma = 1.4826 * mad if np.isfinite(mad) and mad > 0 else float(np.nanstd(low))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 1e-6

        y_min = float(np.nanmin(hv_f))
        y_max = float(np.nanmax(hv_f))
        yspan = float((y_max - y_min) + 1e-12)

        pos = np.clip(hv - b0, 0.0, None)
        total_area = float(np.trapz(np.clip(hv, 0.0, None), chi_vals))
        peak_area = float(np.trapz(pos, chi_vals))
        peak_frac = (peak_area / total_area) if total_area > 0 else 0.0

        return dict(b0=b0, sigma=sigma, yspan=yspan, peak_frac=peak_frac)

    def select_dominant_peak(hv, stats):
        # tunables (same meaning as your current code)
        min_peak_frac = 0.05
        snr_min, rprom_min, rrise_min = 5.0, 0.1, 0.05
        min_width_deg = 30.0
        dom_rise_min = 0.3
        pair_tol_deg = 35.0
        pair_frac_min = 0.25

        if stats["peak_frac"] < min_peak_frac:
            return None, "weak_overall_peaks"

        peaks, props = find_peaks(hv, prominence=0.0, distance=15)
        if len(peaks) == 0:
            return None, "no_peaks"

        proms = props.get("prominences", np.zeros(len(peaks), dtype=float)).astype(float)
        heights = hv[peaks].astype(float)

        b0, sigma, yspan = stats["b0"], stats["sigma"], stats["yspan"]

        snr = proms / sigma
        rprom = proms / yspan
        rrise = (heights - b0) / yspan

        good = (snr >= snr_min) & (rprom >= rprom_min) & (rrise >= rrise_min)
        kept_idx = np.where(good)[0]
        if kept_idx.size == 0:
            return None, "no_peaks"

        dom_local_i = int(np.argmax(proms[kept_idx]))
        dom_i = int(kept_idx[dom_local_i])
        dom_peak = int(peaks[dom_i])
        dom_angle = float(chi_vals[dom_peak]) % 360.0

        widths, _, left_ips, right_ips = peak_widths(hv, peaks, rel_height=0.5)
        li = int(max(0, np.floor(left_ips[dom_i])))
        ri = int(min(len(hv) - 1, np.ceil(right_ips[dom_i])))

        dchi = float(np.nanmedian(np.diff(chi_vals))) if chi_vals.size > 1 else 2.0
        width_deg = float(widths[dom_i] * dchi)

        if width_deg < min_width_deg:
            return None, "too_narrow"

        dom_rise = float((hv[dom_peak] - b0) / yspan)
        if (not np.isfinite(dom_rise)) or (dom_rise < dom_rise_min):
            return None, "weak_dom_rise"

        pair_target = (dom_angle - 180.0) % 360.0
        d = (chi_vals - pair_target + 180.0) % 360.0 - 180.0
        pair_mask = np.abs(d) <= pair_tol_deg
        if not np.any(pair_mask):
            return None, "no_pair_peak"

        pair_height = float(np.nanmax(hv[pair_mask]))
        pair_prom_proxy = pair_height - b0
        dom_prom_proxy = float(hv[dom_peak]) - b0

        if (not np.isfinite(pair_prom_proxy)) or (dom_prom_proxy <= 0) or \
           (pair_prom_proxy / (dom_prom_proxy + 1e-12) < pair_frac_min):
            return None, "weak_pair"

        info = dict(
            peaks=peaks, kept_peaks=peaks[kept_idx],
            dom_peak=dom_peak, dom_angle=dom_angle,
            li=li, ri=ri, width_deg=width_deg
        )
        return info, ""

    def weighted_moments(sv, li, ri):
        xw = chi_vals[li:ri+1].astype(float)
        yw = sv[li:ri+1].astype(float)

        yw0 = yw - float(np.nanmedian(yw))
        yw0 = np.clip(yw0, 0.0, None)

        wsum = float(np.nansum(yw0))
        if not np.isfinite(wsum) or wsum <= 0:
            return None

        wm1 = float(np.nansum(xw * yw0) / wsum)
        wm2 = float(np.sqrt(np.nansum(((xw - wm1) ** 2) * yw0) / wsum))
        return wm1, wm2
    
    def _angdiff_deg(a, b):
        """Shortest signed angular difference a-b in degrees in [-180, 180)."""
        return (a - b + 180.0) % 360.0 - 180.0
    
    def wm_stats_in_window(chi_deg, y, center_deg, halfwidth_deg, dchi=None):
        """
        Weighted moments on a circular window around center.
        Uses baseline-subtracted y (recommended), clips weights to >=0.
        Returns WM1(abs angle), WM2(std), WM3(3rd central moment), skew, area.
        """
        chi = np.asarray(chi_deg, float)
        y = np.asarray(y, float)
        if dchi is None:
            dchi = float(np.nanmedian(np.diff(chi))) if chi.size > 1 else 1.0
    
        # window mask on circle
        off = _angdiff_deg(chi, center_deg)
        m = np.abs(off) <= halfwidth_deg
        if not np.any(m) or np.count_nonzero(np.isfinite(y[m])) < 5:
            return None
    
        w = np.clip(y[m], 0.0, None)
        if not np.any(np.isfinite(w)) or np.nansum(w) <= 0:
            return None
    
        off_m = off[m]
        wsum = float(np.nansum(w))
    
        mu = float(np.nansum(off_m * w) / wsum)          # mean offset (deg)
        var = float(np.nansum(((off_m - mu) ** 2) * w) / wsum)
        sig = float(np.sqrt(max(var, 0.0)))
    
        m3 = float(np.nansum(((off_m - mu) ** 3) * w) / wsum)
        skew = float(m3 / (sig**3 + 1e-12)) if sig > 0 else np.nan
    
        wm1_abs = (center_deg + mu) % 360.0
        area = float(np.nansum(w) * dchi)  # robust to wrap / non-contiguous mask
    
        return {
            "wm1": wm1_abs,
            "wm2": sig,
            "wm3": m3,
            "skew": skew,
            "area": area,
        }

    # -----------------------------
    # Output
    # -----------------------------
    output_rows = []
    total = len(frames)
    processed = 0
    start = time.time()

    for frame_idx in frames:
        y_idx, x_idx = coords[frame_idx]
        raw_intensity = centre_data[frame_idx, :, 1]

        processed += 1
        start = progress_bar(processed, total, prefix=f"[{Filenumber}] Iχ fitting", start_time=start)

        iq_vals = iq_lookup.get((x_idx, y_idx), {})
        row_data = {
            "x": x_idx, "y": y_idx,
            "total SAXS intensity": iq_vals.get("total SAXS intensity", np.nan),
            "total_SAXS_norm_0_1": iq_vals.get("total_SAXS_norm_0_1", np.nan),
            "area under third order curve": iq_vals.get("area under third order curve", np.nan),
            "collagen_third_norm_0_1": iq_vals.get("collagen_third_norm_0_1", np.nan),

            "peak_position": np.nan, "peak_position2": np.nan,
            "peak_width": np.nan, "peak_width2": np.nan,
            "peak_amplitude": np.nan, "peak_amplitude2": np.nan,
            "peak_height": np.nan, "peak_height2": np.nan,
            "rsq_gaussian_fit": np.nan,
            "SM": np.nan, "AP": np.nan,
            "area_fit": np.nan, "area_peaks": np.nan,
            "bg_c": np.nan,

            # "NotchRepaired": 0,
            "FailReason": "",
            
            # WM-derived peak stats (baseline-subtracted, fit-windowed)
            "wm1_p1": np.nan, "wm2_p1": np.nan, "wm3_p1": np.nan, "wm_skew_p1": np.nan, "wm_area_p1": np.nan,
            "wm1_p2": np.nan, "wm2_p2": np.nan, "wm3_p2": np.nan, "wm_skew_p2": np.nan, "wm_area_p2": np.nan,
            "wm1_p3": np.nan, "wm2_p3": np.nan, "wm3_p3": np.nan, "wm_skew_p3": np.nan, "wm_area_p3": np.nan,
            
            "AP_WM": np.nan,
            "wm_area_sum": np.nan,
            "area_total_bs": np.nan,
        }

        # dbg_this = dbg_for(x_idx, y_idx)

        def fail(reason):
            row_data["FailReason"] = reason
            return reason

        # ---- smoothing (and optional notch repair) ----
        wl_h = _odd_leq(len(chi_vals), 31)
        wl_l = _odd_leq(len(chi_vals), 15)

        raw_repaired, did_repair = _repair_notch(
            raw_intensity, chi_vals, center_deg=180.0, halfwin_deg=6.0, dip_frac=0.12, min_depth=0.005
        )
        if did_repair:
            raw_intensity = raw_repaired
            # row_data["NotchRepaired"] = 1

        heavy = savgol_filter(raw_intensity, wl_h, 3)
        smooth = savgol_filter(raw_intensity, wl_l, 3)

        # ---- threshold gate ----
        area_norm = row_data["total_SAXS_norm_0_1"]
        # if dbg_this:
        #     print("\n" + "-"*70)
            # print(f"[ICH DEBUG] frame={frame_idx} (x={x_idx}, y={y_idx}) total_SAXS_norm_0_1={area_norm}")

        if not np.isfinite(area_norm):
            fail("nan_exist")
        elif area_norm < area_threshold:
            fail("under_thresh")

        # ---- compute stats + peak selection ----
        stats = None
        peak_info = None
        if row_data["FailReason"] == "":
            hv = np.asarray(heavy, float)
            sv = np.asarray(smooth, float)
            stats = compute_stats(hv, sv)
            if stats is None:
                fail("nan_exist")

        if row_data["FailReason"] == "":
            peak_info, reason = select_dominant_peak(np.asarray(heavy, float), stats)
            if reason:
                fail(reason)

        # ---- WM + fit ----
        fit_res = None
        if row_data["FailReason"] == "":
            wm = weighted_moments(np.asarray(smooth, float), peak_info["li"], peak_info["ri"])
            if wm is None:
                fail("wm_fail")
            else:
                wm1, wm2 = wm
                # if dbg_this:
                    # print(f"[ICH DEBUG] WM1={wm1:.2f}°  WM2={wm2:.2f}°")

                fit_res = fit_multi_gaussian_periodic(chi_vals, np.asarray(smooth, float), peak_info["dom_angle"])
                if fit_res is None:
                    fail("fail_gauss")
                else:
                    
                    # --- baseline-subtract using lmfit constant ---
                    bg_c = float(fit_res["bg_c"])
                    smooth_bs = np.asarray(smooth, float) - bg_c
                    heavy_bs  = np.asarray(heavy, float)  - bg_c
                    yfit_bs   = np.asarray(fit_res["y_plot"], float) - bg_c
                    
                    # --- define peak windows from lmfit sigma ---
                    # NOTE: your fit_res["peak_width"] is sigma in degrees (not FWHM).
                    sigma_deg = float(fit_res["peak_width"])
                    
                    # If you want wider windows, change halfwidth:
                    # halfwidth = sigma_deg              # ±1σ (what you asked for)
                    # halfwidth = 2.0 * sigma_deg        # ±2σ (often nicer for noisy peaks)
                    # halfwidth = 0.5 * (2.355*sigma_deg)  # ±FWHM/2
                    halfwidth = sigma_deg
                    
                    dchi = float(np.nanmedian(np.diff(chi_vals))) if len(chi_vals) > 1 else 1.0
                    
                    # peaks to measure (2 by default; include 3rd if you add it later)
                    centers = [
                        float(fit_res["peak_position"]),
                        float(fit_res["peak_position2"]),
                    ]
                    # Optional 3rd peak support if you ever return peak_position3 from fitting:
                    if "peak_position3" in fit_res and np.isfinite(fit_res["peak_position3"]):
                        centers.append(float(fit_res["peak_position3"]))
                    
                    wm_results = []
                    for c in centers:
                        wm = wm_stats_in_window(chi_vals, smooth_bs, c, halfwidth, dchi=dchi)
                        wm_results.append(wm)
                    
                    # write WM outputs
                    def _put_wm(idx, wm):
                        k = idx + 1
                        if wm is None:
                            return
                        row_data[f"wm1_p{k}"] = wm["wm1"]
                        row_data[f"wm2_p{k}"] = wm["wm2"]
                        row_data[f"wm3_p{k}"] = wm["wm3"]
                        row_data[f"wm_skew_p{k}"] = wm["skew"]
                        row_data[f"wm_area_p{k}"] = wm["area"]
                    
                    for i, wm in enumerate(wm_results[:3]):
                        _put_wm(i, wm)
                    
                    # --- AP_WM: fraction of baseline-subtracted positive area captured in windows ---
                    area_total_bs = float(np.nansum(np.clip(smooth_bs, 0.0, None)) * dchi)
                    wm_area_sum = float(np.nansum([wm["area"] for wm in wm_results if wm is not None]))
                    
                    row_data["area_total_bs"] = area_total_bs
                    row_data["wm_area_sum"] = wm_area_sum
                    row_data["AP_WM"] = (wm_area_sum / area_total_bs) if area_total_bs > 0 else np.nan
                    row_data.update({
                        "peak_position": fit_res["peak_position"],
                        "peak_position2": fit_res["peak_position2"],
                        "peak_width": fit_res["peak_width"],
                        "peak_width2": fit_res["peak_width2"],
                        "peak_amplitude": fit_res["peak_amplitude"],
                        "peak_amplitude2": fit_res["peak_amplitude2"],
                        "peak_height": fit_res["peak_height"],
                        "peak_height2": fit_res["peak_height2"],
                        "rsq_gaussian_fit": fit_res["rsq"],
                        "SM": fit_res["SM"],
                        "AP": fit_res["AP"],
                        "area_fit": fit_res["area_fit"],
                        "area_peaks": fit_res["area_peaks"],
                        "bg_c": fit_res["bg_c"],
                    })

        if row_data["FailReason"] == "" and np.all(np.isnan([row_data["peak_position"], row_data["peak_position2"]])):
            row_data["FailReason"] = "other"

        # ---- plotting (single place) ----
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(chi_vals, smooth, color="black", lw=2, label="Smoothed Iχ")
        ax.plot(chi_vals, heavy, color="red", ls=":", alpha=0.5, label="Heavy smooth")

        if peak_info is not None:
            ax.scatter(chi_vals[peak_info["peaks"]], np.asarray(heavy)[peak_info["peaks"]],
                       marker="x", s=60, linewidths=2, color="tab:purple", label="find_peaks (raw)")
            if peak_info["kept_peaks"].size:
                ax.scatter(chi_vals[peak_info["kept_peaks"]], np.asarray(heavy)[peak_info["kept_peaks"]],
                           marker="x", s=90, linewidths=3, color="tab:green", label="kept peaks")

        if fit_res is not None:
            bg_c = float(fit_res["bg_c"])
            smooth_bs = np.asarray(smooth, float) - bg_c
            heavy_bs  = np.asarray(heavy, float)  - bg_c
            yfit_bs   = np.asarray(fit_res["y_plot"], float) - bg_c
        
            # original-scale baseline
            ax.axhline(bg_c, color="tab:gray", ls="--", lw=1.5, label="lmfit baseline (bg_c)")
        
            # baseline-subtracted curves (same axes; just shifted)
            ax.plot(chi_vals, smooth_bs, color="tab:olive", lw=2, alpha=0.8, label="Smoothed Iχ (baseline-sub)")
            ax.plot(chi_vals, heavy_bs,  color="tab:brown", ls=":", alpha=0.6, label="Heavy smooth (baseline-sub)")
        
            # baseline-subtracted fit peaks
            ax.plot(chi_vals, yfit_bs, color="tab:cyan", lw=2, alpha=0.9, label="Fit (baseline-sub)")
            ax.axhline(0.0, color="tab:gray", ls=":", lw=1.0, alpha=0.8, label="0 (baseline-sub)")

        ax.set_xlim(0, 360)
        ax.set_xlabel("Azimuthal angle (°)")
        ax.set_ylabel("Intensity (a.u.)")
        ax.set_title(f"Iχ frame {frame_idx} (x={x_idx}, y={y_idx})  Fail={row_data['FailReason']}")
        ax.legend()
        plt.tight_layout()

        if Plot:
            plt.show()
        else:
            pdf.savefig(fig)
            plt.close(fig)

        output_rows.append(row_data)

    pdf.close()
    gc.collect()

    df_out = pd.DataFrame(output_rows)
    df_out.to_csv(os.path.join(Output_directoryCSV, f"{Filenumber} IChi_fitting.csv"), index=False)

    del df_out, output_rows
    gc.collect()

    progress_bar(total, total, prefix=f"[{Filenumber}] Iχ fitting", start_time=start, end=True)




def fit_multi_gaussian_periodic(chi_deg, intensity, dominant_deg,
                                sigma0=None, sigma_min=1.0, sigma_max=60.0,
                                bg0=None, bg_max=None,
                                amp0=None, amp_max=None):
    
    chi = np.asarray(chi_deg, dtype=float)
    y   = np.asarray(intensity, dtype=float)

    if chi.size < 10 or not np.any(np.isfinite(y)):
        return None

    # --------------------------------------------------
    # Peak centres (fixed)
    # --------------------------------------------------
    x0 = dominant_deg % 360.0
    centers = np.array([
        x0 - 360.0,
        x0 - 180.0,
        x0,
        x0 + 180.0,
        x0 + 360.0
    ])

    # --------------------------------------------------
    # Model construction
    # --------------------------------------------------
    models = []
    for i in range(5):
        models.append(GaussianModel(prefix=f"g{i}_"))
    bg = ConstantModel(prefix="bg_")

    model = models[0]
    for m in models[1:]:
        model += m
    model += bg

    params = model.make_params()

    # --------------------------------------------------
    # Shared sigma (FWHM ≈ 15–20° typical)
    # --------------------------------------------------
    sigma0 = 30.0 / 2.355
    params["g0_sigma"].set(value=sigma0, min=1.0, max=60.0)

    for i in range(1, 5):
        params[f"g{i}_sigma"].expr = "g0_sigma"

    # --------------------------------------------------
    # Amplitude + center constraints (SHARED amplitude)
    # --------------------------------------------------
    y_peak = np.nanmax(y)
    amp0 = max(1e-6, y_peak * sigma0 * np.sqrt(2 * np.pi))
    
    for i, c in enumerate(centers):
        params[f"g{i}_center"].set(value=c, vary=False)
    
    # master amplitude (central peak)
    params["g2_amplitude"].set(value=amp0, min=0.0)
    
    # tie all other amplitudes to the central one
    for i in [0, 1, 3, 4]:
        params[f"g{i}_amplitude"].expr = "g2_amplitude"


    # --------------------------------------------------
    # Robust background estimate (ignore peaks)
    # --------------------------------------------------
    mask = np.ones_like(chi, dtype=bool)
    for c in centers:
        mask &= np.abs(chi - c) > 25.0

    if np.any(mask):
        bg0 = np.nanmedian(y[mask])
        mad = np.nanmedian(np.abs(y[mask] - bg0))
    else:
        bg0 = np.nanmedian(y)
        mad = np.nanstd(y)

    if not np.isfinite(bg0):
        bg0 = 0.0
    if not np.isfinite(mad) or mad <= 0:
        mad = 0.05 * (np.nanmax(y) - np.nanmin(y) + 1e-6)

    params["bg_c"].set(
        value=bg0,
        min=0.0,
        max=bg0 + 5.0 * mad
    )

    # --------------------------------------------------
    # Weights (robust to spikes)
    # --------------------------------------------------
    y0 = y - np.nanmin(y)
    w = 1.0 / np.sqrt(np.maximum(y0, 1e-3))

    # --------------------------------------------------
    # Fit
    # --------------------------------------------------
    try:
        result = model.fit(
            y,
            params,
            x=chi,
            weights=w,
            method="least_squares",
            fit_kws={"loss": "soft_l1", "f_scale": 0.1}
        )
    except Exception:
        return None

    if not result.success:
        return None

    p = result.params

    # --------------------------------------------------
    # Extract parameters
    # --------------------------------------------------
    c_main = p["g2_center"].value
    sigma  = p["g0_sigma"].value

    a_main = p["g2_amplitude"].value
    a_pair = p["g1_amplitude"].value

    h_main = models[2].eval(p, x=np.array([c_main]))[0]
    h_pair = models[1].eval(p, x=np.array([c_main - 180]))[0]

    # --------------------------------------------------
    # Goodness of fit (global R²)
    # --------------------------------------------------
    y_hat = result.best_fit
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - np.nanmean(y)) ** 2)
    rsq = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # --------------------------------------------------
    # Alignment parameters (your existing helper)
    # --------------------------------------------------
    try:
        SM, AP, area_fit, area_peaks = calculate_alignment_param(
            y_hat,                 # fitted Iχ
            p["bg_c"].value,       # constant background
            chi                    # 0–360 domain
        )
    except Exception:
        SM, AP, area_fit, area_peaks = np.nan, np.nan,  np.nan, np.nan

    # --------------------------------------------------
    # Return dictionary
    # --------------------------------------------------
    return {
        "y_plot": y_hat,

        "peak_position":   c_main % 360.0,
        "peak_position2":  (c_main - 180.0) % 360.0,

        "peak_width":      sigma,
        "peak_width2":     sigma,

        "peak_amplitude":  a_main,
        "peak_amplitude2": a_pair,

        "peak_height":     h_main,
        "peak_height2":    h_pair,

        "rsq":             rsq,
        "SM":              SM,
        "AP":              AP,
        "area_fit":        area_fit,
        "area_peaks":      area_peaks,
        "bg_c":            p["bg_c"].value
    }

def calculate_alignment_param(
    y_fit,
    bg_c,
    chi_deg
):

    # Ensure numpy arrays
    y_fit = np.asarray(y_fit, dtype=float)
    chi   = np.asarray(chi_deg, dtype=float)

    # -----------------------------
    # Guard conditions
    # -----------------------------
    if y_fit.size < 5 or not np.any(np.isfinite(y_fit)):
        return np.nan, np.nan

    # -----------------------------
    # Structural Metric (SM)
    # -----------------------------
    mean_val = np.nanmean(y_fit)
    std_val  = np.nanstd(y_fit)

    if mean_val > 0:
        SM = std_val / mean_val
    else:
        SM = np.nan

    # -----------------------------
    # Alignment Parameter (AP)
    # -----------------------------
    try:
        # Total area under fitted curve
        area_fit = float(np.trapz(y_fit, chi))

        # Background contribution (constant × full domain)
        bg_area = bg_c * 360.0

        if area_fit > 0:
            AP = (area_fit - bg_area) / area_fit
            area_peaks = area_fit - bg_area
        else:
            AP = np.nan
            area_peaks = np.nan
        

    except Exception:
        AP = np.nan
        area_fit = np.nan
        area_peaks = np.nan

    return SM, AP, area_fit, area_peaks






