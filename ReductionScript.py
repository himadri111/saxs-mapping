#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 17:19:18 2025

@author: lauraforster
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import logging
import h5py
from pyFAI.integrator.azimuthal import AzimuthalIntegrator
from pyFAI import units
from Utils import progress_bar
from pathlib import Path
from h5py import File
import time
import warnings

logging.getLogger("pyFAI").setLevel(logging.ERROR)

warnings.filterwarnings(
    "ignore",
    message="Using UFloat objects with std_dev==0 may give unexpected results.",
    module="uncertainties.core"
)

logging.getLogger("pyFAI.geometry.core").setLevel(logging.ERROR)

from dataclasses import dataclass

# ------------------------------------------------------------------------------------------------------------------------
# -------    Mask and Calibration    ---------
# ------------------------------------------------------------------------------------------------------------------------

@dataclass
class CalibrationInfo:
    mask: np.ndarray
    beam_center_x: float
    beam_center_y: float
    x_pixel_size: float
    y_pixel_size: float
    sample_detector_separation: float
    wavelength: float

def MaskCalib(MASK_PATH, CALIBRANT_PATH):
    with File(MASK_PATH) as maskFile:
        try:
            mask = np.array(maskFile["entry/mask/mnask"])
            print(f"Loaded mask as mnask: {MASK_PATH}")
        except KeyError:
            try:
                mask = np.array(maskFile["entry/mask/mask"])
                print(f"Loaded mask as mask: {MASK_PATH}")
            except KeyError:
                raise KeyError("Could not find 'mnask' or 'mask' in mask file!")

    with File(CALIBRANT_PATH) as calib_file:
        beam_center_x = float(calib_file["entry1/instrument/detector/beam_center_x"][()])
        beam_center_y = float(calib_file["entry1/instrument/detector/beam_center_y"][()])
        x_pixel_size = float(calib_file["entry1/instrument/detector/x_pixel_size"][()])
        y_pixel_size = float(calib_file["entry1/instrument/detector/y_pixel_size"][()])
        sample_detector_separation = float(calib_file["entry1/instrument/detector/distance"][()])
        wavelength = float(calib_file["entry1/calibration_sample/beam/incident_wavelength"][()]) / 1e10
        print(f"Loaded calibration: {CALIBRANT_PATH}")
    print('\n')
    return CalibrationInfo(mask, beam_center_x, beam_center_y, x_pixel_size,
                           y_pixel_size, sample_detector_separation, wavelength)
       
# ------------------------------------------------------------------------------------------------------------------------
# -------    Reduce Iq    ---------
# ------------------------------------------------------------------------------------------------------------------------
 
def ReductionIQ(Filenumber, sample_loc, output_path, outputfolder,sample_beginning, sample_end, identifier, frame_index, nq, nchi, q_range, chi_range, calib, Plot,file_start=None, file_end=None, x_list=None, y_list=None):
    print('Beginning I_q integration')
    a1 = AzimuthalIntegrator(wavelength = calib.wavelength)        
    a1.setFit2D(
        directDist=calib.sample_detector_separation, 
        centerX=calib.beam_center_x / calib.x_pixel_size, 
        centerY=calib.beam_center_y / calib.y_pixel_size, 
        pixelX=calib.x_pixel_size * 1000, 
        pixelY=calib.y_pixel_size * 1000
    )
    iq_data = []
    coords_list = []
    
    processed = 0
    start = time.time()
    
    if identifier == 'Split':
        assert file_start is not None and file_end is not None, "file_start and file_end must be provided for Split scans"
        
        filelist = np.arange(file_start, file_end + 1)
        
        if x_list is not None:
            x_list = list(x_list)
        if y_list is not None:
            y_list = list(y_list)

        coords_to_process = None
        if x_list is not None and y_list is not None:
            coords_to_process = set(zip(y_list, x_list))  # (row, col)
        elif x_list is not None:
            coords_to_process = 'x_only'
        elif y_list is not None:
            coords_to_process = 'y_only'
        else:
            coords_to_process = None
        
        with File(Path(sample_loc + f"i22-{filelist[0]}.nxs")) as first_file:
            total = len(filelist) * first_file["entry/SAXS/data"].shape[0]

        for y_index, file_number in enumerate(filelist):
            sample_name = f"i22-{file_number}.nxs"
            sample_path = Path(sample_loc + sample_name)
    
            with File(sample_path) as sample_file:
                dset = sample_file["entry/SAXS/data"]
                
                x_len = dset.shape[0]  # number of frames in this line scan
    
                for x_index in range(x_len):
                    if coords_to_process == 'x_only' and x_index not in x_list:
                        continue
                    if coords_to_process == 'y_only' and y_index not in y_list:
                        continue
                    if isinstance(coords_to_process, set) and (y_index, x_index) not in coords_to_process:
                        continue
    
                    frame = dset[x_index, :, :]
                    # print(f"Processing iq reduction frame x(col) = {x_index}, y(row) = {y_index} ")
    
                    test_iq = a1.integrate1d(frame, nq, radial_range=q_range, azimuth_range=chi_range,
                                             mask=calib.mask, error_model="poisson", correctSolidAngle=False)
    
                    iq_data.append(np.stack((test_iq[0], test_iq[1]), axis=-1))
                    coords_list.append((y_index, x_index))  # Save coords
                    
                    processed += 1
                    start = progress_bar(processed, total, prefix=f"[{Filenumber}] IQ reduction", start_time=start)

    
                    if Plot:
                        plt.plot(test_iq[0], test_iq[1])
                        plt.title(f'I_q plot for frame x(col) = {x_index}, y(row) = {y_index}')
                        plt.xlabel('q (inv nm)')
                        plt.ylabel('Intensity (a.u.)')
                        plt.show()
        
        # Save combined .nxs file
        iq_data = np.array(iq_data)
        coords_array = np.array(coords_list, dtype=int)
        
        nxs_dir = f"{output_path}/{outputfolder}"
        os.makedirs(nxs_dir, exist_ok=True)

        output_file = f"{nxs_dir}/i22-{Filenumber}_iq.nxs"
        with File(output_file, "w") as f:
            f.create_dataset("iq", data=iq_data)
            f.create_dataset("coords", data=coords_array) 
            
        progress_bar(total, total, prefix=f"[{Filenumber}] IQ reduction", start_time=start, end=True)

    else:
                
        sample_name = f"{sample_beginning}{Filenumber}{sample_end}"
        sample_path = Path(sample_loc + sample_name)
        
        with File(sample_path) as sample_file:
            dset = sample_file["entry/SAXS/data"]
            rows, cols = dset.shape[:2]
            
            total = rows * cols


            if x_list is not None and y_list is not None:
                coords_to_process = set(zip(y_list, x_list))  # (row, col)
            elif x_list is not None:
                coords_to_process = 'x_only'
            elif y_list is not None:
                coords_to_process = 'y_only'
            else:
                coords_to_process = None
        
    
            for ycoord in range(rows):
                for xcoord in range(cols):
                    if coords_to_process == 'x_only' and xcoord not in x_list:
                        continue
                    if coords_to_process == 'y_only' and ycoord not in y_list:
                        continue
                    if isinstance(coords_to_process, set) and (ycoord, xcoord) not in coords_to_process:
                        continue
                    # print("SAXS dataset shape:", dset.shape, " ndim:", dset.ndim)

                    frame = dset[ycoord, xcoord, :, :]
                    # print(f"Processing iq reduction frame x(col) = {xcoord}, y(row) = {ycoord}")
    
                    test_iq = a1.integrate1d(frame, nq, radial_range=q_range, azimuth_range=chi_range,
                                             mask=calib.mask, error_model="poisson", correctSolidAngle=False)
                        
                    iq_data.append(np.stack((test_iq[0], test_iq[1]), axis=-1))
                    coords_list.append((ycoord, xcoord))
                    
                    processed += 1
                    start = progress_bar(processed, total, prefix=f"[{Filenumber}] IQ reduction", start_time=start)

    
                    if Plot:
                        plt.plot(test_iq[0], test_iq[1])
                        plt.title(f'I_q plot for frame x(col)={xcoord}, y(row)={ycoord}')
                        plt.xlabel('q (inv nm)')
                        plt.ylabel('Intensity (a.u.)')
                        plt.show()
                    
 # Save to .nxs files
        iq_data = np.array(iq_data)
        coords_array = np.array(coords_list, dtype=int)

        nxs_dir = f"{output_path}/{outputfolder}"
        os.makedirs(nxs_dir, exist_ok=True)

        with File(f"{nxs_dir}/i22-{Filenumber}_iq.nxs", "w") as f:
            f.create_dataset("iq", data=iq_data)
            f.create_dataset("coords", data=coords_array)  # Add coords here
        progress_bar(total, total, prefix=f"[{Filenumber}] IQ reduction", start_time=start, end=True)

    return 

def CheckIQRed(scan_no, Output_directoryCSV, base_output_path, sample_loc, file_start, file_end, identifier):
    dat_nxs_file = f'{base_output_path}/i22-{scan_no}/i22-{scan_no}_iq.nxs'        
    with h5py.File(dat_nxs_file, 'r') as nxs_file:
        iq_data = nxs_file['iq'][:]  # shape: (n_frames, 2)
        coords = nxs_file['coords'][:]  # shape: (n_frames, 2), in (y, x) order
        num_frames = iq_data.shape[0]
        # q_values = iq_data[0, :, 0]  # all frames share the same q values
    print('Checking Iq Reduction Shape:', 'x:', coords[-1][0], 'y:', coords[-1][1], 'Number of Frames:', num_frames)
    return

# ------------------------------------------------------------------------------------------------------------------------
# -------    Reduce IChi (Not rings)   ---------
# ------------------------------------------------------------------------------------------------------------------------

def ReductionICHI(Filenumber, sample_loc, output_path, outputfolder,sample_beginning, sample_end, identifier, frame_index, nq, nchi, q_range, q_range_centre, chi_range, inner_q, outer_q, calib, Plot, IChi_EntireReduct, file_start=None, file_end=None, x_list=None, y_list=None):
    if IChi_EntireReduct == False:
        ReductICHI_Rings(Filenumber, sample_loc, output_path, outputfolder,sample_beginning, sample_end, identifier, frame_index, nq, nchi, q_range, q_range_centre, chi_range, inner_q, outer_q, calib, Plot, file_start=file_start, file_end=file_end, x_list=x_list, y_list=y_list)
    elif IChi_EntireReduct == True:
        ReductICHI_Entire(Filenumber, sample_loc, output_path, outputfolder,sample_beginning, sample_end, identifier, frame_index,nq, nchi, q_range, q_range_centre, chi_range,inner_q, outer_q, calib, Plot,file_start=file_start, file_end=file_end,x_list=x_list, y_list=y_list)
        
        
def ReductICHI_Entire(Filenumber, sample_loc, output_path, outputfolder,sample_beginning, sample_end, identifier, frame_index, nq, nchi, q_range, q_range_centre, chi_range, inner_q, outer_q, calib, Plot, file_start=None, file_end=None, x_list=None, y_list=None):
    
    print('\n')
    print('Beginning I_Chi integration (centre-only)')
    
    a1 = AzimuthalIntegrator(wavelength = calib.wavelength)        
    a1.setFit2D(
        directDist=calib.sample_detector_separation, 
        centerX=calib.beam_center_x / calib.x_pixel_size, 
        centerY=calib.beam_center_y / calib.y_pixel_size, 
        pixelX=calib.x_pixel_size * 1000, 
        pixelY=calib.y_pixel_size * 1000
    )
    
    ichi_data = []
    coords_list = []
    
    processed = 0
    start = time.time()
    
    if identifier == 'Split':
        assert file_start is not None and file_end is not None
        filelist = np.arange(file_start, file_end + 1)
        
        if x_list is not None:
            x_list = list(x_list)
        if y_list is not None:
            y_list = list(y_list)
        
        coords_to_process = None
        if x_list is not None and y_list is not None:
            coords_to_process = set(zip(y_list, x_list))  # (row, col)
        elif x_list is not None:
            coords_to_process = 'x_only'
        elif y_list is not None:
            coords_to_process = 'y_only'
        else:
            coords_to_process = None
            
        with File(Path(sample_loc + f"i22-{filelist[0]}.nxs")) as first_file:
            dset0 = first_file["entry/SAXS/data"]
            frames_per_file = dset0.shape[0]  # line scan length
        total = len(filelist) * frames_per_file

            
        for y_index, file_number in enumerate(filelist):
            sample_name = f"i22-{Filenumber}.nxs"
            sample_path = Path(sample_loc + sample_name)
            
            with File(sample_path) as sample_file:
                dset = sample_file["entry/SAXS/data"]
                x_len = dset.shape[0]
                
                for x_index in range(x_len):
                    if coords_to_process == 'x_only' and x_index not in x_list:
                        continue
                    if coords_to_process == 'y_only' and y_index not in y_list:
                        continue
                    if isinstance(coords_to_process, set) and (y_index, x_index) not in coords_to_process:
                        continue

                    frame = dset[x_index, :, :]
                    # print(f"Processing ichi reduction frame x(row) = {x_index}, y(col) = {y_index})")
                    
                    #Centre
                    chi_vals, chi = a1.integrate_radial(frame,npt=nchi,unit=units.CHI_DEG, radial_range=(q_range_centre[0], q_range_centre[1]),mask=calib.mask,  method="splitpixel")
                    
                    chi_vals = np.mod(chi_vals, 360)
                    sort_idx = np.argsort(chi_vals)
                    chi_vals = chi_vals[sort_idx].flatten()
                    chi = chi[sort_idx].flatten()
                    
                    ichi_data.append(np.column_stack((chi_vals, chi)))
                    coords_list.append((y_index, x_index))
                    
                    processed += 1
                    start = progress_bar(processed, total,
                                         prefix=f"[{Filenumber}] Iχ reduction",
                                         start_time=start)

                    
                    if Plot:
                        plt.figure(figsize=(8, 4))
                        plt.plot(chi_vals, chi, label='centre raw')
                        plt.title(f'I_χ frame at x(col) = {x_index}, y(row) = {y_index}')
                        plt.xlabel('Azimuthal angle (°)')
                        plt.ylabel('Intensity (a.u.)')
                        plt.legend()
                        plt.tight_layout()
                        plt.show()
                        
    else:
        sample_name = f"{sample_beginning}{Filenumber}{sample_end}"
        sample_path = Path(sample_loc + sample_name)
        with File(sample_path) as sample_file:
            dset = sample_file["entry/SAXS/data"]
            rows, cols = dset.shape[:2]
            total = rows * cols
            
            coords_to_process = None
            if x_list is not None and y_list is not None:
                coords_to_process = set(zip(y_list, x_list))  # (row, col)
            elif x_list is not None:
                coords_to_process = 'x_only'
            elif y_list is not None:
                coords_to_process = 'y_only'
            else:
                coords_to_process = None
        
            for ycoord in range(rows):
                for xcoord in range(cols):
                    if coords_to_process == 'x_only' and xcoord not in x_list:
                        continue
                    if coords_to_process == 'y_only' and ycoord not in y_list:
                        continue
                    if isinstance(coords_to_process, set) and (ycoord, xcoord) not in coords_to_process:
                        continue
                    frame = dset[ycoord, xcoord, :, :]
                    # print(f"Processing ichi reduction frame x(col) = {xcoord}, y(row) = {ycoord}")
                    
                    #Centre
                    chi_vals, chi = a1.integrate_radial(frame,npt=nchi,unit=units.CHI_DEG, radial_range=(q_range_centre[0], q_range_centre[1]), mask=calib.mask, method="splitpixel")
                    chi_vals = np.mod(chi_vals, 360)
                    sort_idx = np.argsort(chi_vals)
                    chi_vals = chi_vals[sort_idx].flatten()
                    chi = chi[sort_idx].flatten()
                    ichi_data.append(np.column_stack((chi_vals, chi)))
                    coords_list.append((ycoord, xcoord))
                    
                    processed += 1
                    start = progress_bar(processed, total,
                                         prefix=f"[{Filenumber}] Iχ reduction",
                                         start_time=start)
                    
                    # Quick plot for sanity check
                    if Plot:
                        plt.figure(figsize=(8, 4))
                        plt.plot(chi_vals, chi, label='centre raw')
                        plt.title(f'I_χ frame at x(col) = {xcoord}, y(row) = {ycoord}')
                        plt.xlabel('Azimuthal angle (°)')
                        plt.ylabel('Intensity (a.u.)')
                        plt.legend()
                        plt.tight_layout()
                        plt.show()
                        
    # Save .nxs files
    ichi_data = np.stack(ichi_data, axis=0)  # (n_frames, n_chi, 2)
    coords_array = np.array(coords_list, dtype=int)
    
    nxs_dir = f"{output_path}/{outputfolder}"
    os.makedirs(nxs_dir, exist_ok=True)
    
    with File(f"{nxs_dir}/i22-{Filenumber}_ichi.nxs", "w") as f:
        f.create_dataset("ichi", data=ichi_data)
        f.create_dataset("coords", data=coords_array)
        
    progress_bar(total, total, prefix=f"[{Filenumber}] Iχ reduction",
                     start_time=start, end=True)
    return


# ------------------------------------------------------------------------------------------------------------------------
# -------    Reduce IChi (Using three rings)    ---------
# ------------------------------------------------------------------------------------------------------------------------

    
def ReductICHI_Rings(Filenumber, sample_loc, output_path, outputfolder,sample_beginning, sample_end, identifier, frame_index, nq, nchi, q_range, q_range_centre, chi_range, inner_q, outer_q, calib, Plot, file_start=None, file_end=None, x_list=None, y_list=None):
    print('\n')
    print('Beginning I_Chi integration (three ring integration)')

    a1 = AzimuthalIntegrator(wavelength = calib.wavelength)        
    a1.setFit2D(
        directDist=calib.sample_detector_separation, 
        centerX=calib.beam_center_x / calib.x_pixel_size, 
        centerY=calib.beam_center_y / calib.y_pixel_size, 
        pixelX=calib.x_pixel_size * 1000, 
        pixelY=calib.y_pixel_size * 1000
    )
    
    ichi_data, ichi_inner_data, ichi_outer_data = [], [], []
    coords_list = []
    
    processed = 0
    start = time.time()
    
    if identifier == 'Split':
        assert file_start is not None and file_end is not None
        filelist = np.arange(file_start, file_end + 1)
        
        # Compute total frames across files ONCE
        with File(Path(sample_loc + f"i22-{filelist[0]}.nxs")) as first_file:
            dset0 = first_file["entry/SAXS/data"]
            frames_per_file = dset0.shape[0]  # line scan length
        total = len(filelist) * frames_per_file
        
        if x_list is not None:
            x_list = list(x_list)
        if y_list is not None:
            y_list = list(y_list)
        
        coords_to_process = None
        if x_list is not None and y_list is not None:
            coords_to_process = set(zip(y_list, x_list))  # (row, col)
        elif x_list is not None:
            coords_to_process = 'x_only'
        elif y_list is not None:
            coords_to_process = 'y_only'
        else:
            coords_to_process = None

        
        for y_index, file_number in enumerate(filelist):
            sample_name = f"i22-{Filenumber}.nxs"
            sample_path = Path(sample_loc + sample_name)
            
            with File(sample_path) as sample_file:
                dset = sample_file["entry/SAXS/data"]
                x_len = dset.shape[0]
                
                for x_index in range(x_len):
                    if coords_to_process == 'x_only' and x_index not in x_list:
                        continue
                    if coords_to_process == 'y_only' and y_index not in y_list:
                        continue
                    if isinstance(coords_to_process, set) and (y_index, x_index) not in coords_to_process:
                        continue

                    frame = dset[x_index, :, :]
                    # print(f"Processing ichi reduction frame x(row) = {x_index}, y(col) = {y_index})")
                    
                    #Centre
                    chi_vals, chi = a1.integrate_radial(frame,npt=nchi,unit=units.CHI_DEG, radial_range=(q_range_centre[0], q_range_centre[1]),mask=calib.mask,  method="splitpixel")
                    
                    # Inner
                    chi_vals_inner, chi_inner = a1.integrate_radial(frame,npt=nchi,unit=units.CHI_DEG, radial_range=(inner_q[0], inner_q[1]),mask=calib.mask, method="splitpixel")
                    
                    # Outer
                    chi_vals_outer, chi_outer = a1.integrate_radial(frame,npt=nchi,unit=units.CHI_DEG, radial_range=(outer_q[0], outer_q[1]), mask=calib.mask, method="splitpixel")
                    
                    chi_vals = np.mod(chi_vals, 360)
                    sort_idx = np.argsort(chi_vals)
                    chi_vals = chi_vals[sort_idx].flatten()
                    chi = chi[sort_idx].flatten()
                    # Ensure inner/outer chi_vals are also wrapped and sorted
                    chi_vals_inner = np.mod(chi_vals_inner, 360)
                    chi_vals_outer = np.mod(chi_vals_outer, 360)
                    
                    sort_idx_inner = np.argsort(chi_vals_inner)
                    sort_idx_outer = np.argsort(chi_vals_outer)
                    
                    chi_inner_sorted = chi_inner[sort_idx_inner].flatten()
                    chi_outer_sorted = chi_outer[sort_idx_outer].flatten()
                    
                    ichi_data.append(np.column_stack((chi_vals, chi)))
                    ichi_inner_data.append(np.column_stack((chi_vals_inner[sort_idx_inner], chi_inner_sorted)))
                    ichi_outer_data.append(np.column_stack((chi_vals_outer[sort_idx_outer], chi_outer_sorted)))
                    
                    coords_list.append((y_index, x_index))
                    
                    processed += 1
                    start = progress_bar(processed, total,
                                         prefix=f"[{Filenumber}] Iχ reduction",
                                         start_time=start)

                    if Plot:
                        plt.figure(figsize=(8, 4))
                        plt.plot(chi_vals, chi, label='centre raw')
                        plt.plot(chi_vals_inner[sort_idx_inner], chi_inner_sorted, '--', label='inner ring', alpha=0.6)
                        plt.plot(chi_vals_outer[sort_idx_outer], chi_outer_sorted, '--', label='outer ring', alpha=0.6)
                        plt.title(f'I_χ frame at x(col) = {x_index}, y(row) = {y_index}')
                        plt.xlabel('Azimuthal angle (°)')
                        plt.ylabel('Intensity (a.u.)')
                        plt.legend()
                        plt.tight_layout()
                        plt.show()
      
    else:
        sample_name = f"{sample_beginning}{Filenumber}{sample_end}"
        sample_path = Path(sample_loc + sample_name)
        
        with File(sample_path) as sample_file:
            dset = sample_file["entry/SAXS/data"]
            rows, cols = dset.shape[:2]
            total = rows * cols
            
            coords_to_process = None
            if x_list is not None and y_list is not None:
                coords_to_process = set(zip(y_list, x_list))  # (row, col)
            elif x_list is not None:
                coords_to_process = 'x_only'
            elif y_list is not None:
                coords_to_process = 'y_only'
            else:
                coords_to_process = None

            for ycoord in range(rows):
                for xcoord in range(cols):
                    if coords_to_process == 'x_only' and xcoord not in x_list:
                        continue
                    if coords_to_process == 'y_only' and ycoord not in y_list:
                        continue
                    if isinstance(coords_to_process, set) and (ycoord, xcoord) not in coords_to_process:
                        continue
                    frame = dset[ycoord, xcoord, :, :]
                    print(f"Processing ichi reduction frame x(col) = {xcoord}, y(row) = {ycoord}")
                    
                    #Centre
                    chi_vals, chi = a1.integrate_radial(frame,npt=nchi,unit=units.CHI_DEG, radial_range=(q_range_centre[0], q_range_centre[1]), mask=calib.mask, method="splitpixel")
                    
                    # Inner
                    chi_vals_inner, chi_inner = a1.integrate_radial(frame,npt=nchi,unit=units.CHI_DEG, radial_range=(inner_q[0], inner_q[1]), mask=calib.mask, method="splitpixel")
                    
                    # Outer
                    chi_vals_outer, chi_outer = a1.integrate_radial(frame,npt=nchi,unit=units.CHI_DEG, radial_range=(outer_q[0], outer_q[1]), mask=calib.mask, method="splitpixel")
                    
                    chi_vals = np.mod(chi_vals, 360)
                    sort_idx = np.argsort(chi_vals)
                    chi_vals = chi_vals[sort_idx].flatten()
                    chi = chi[sort_idx].flatten()
                    # Ensure inner/outer chi_vals are also wrapped and sorted
                    chi_vals_inner = np.mod(chi_vals_inner, 360)
                    chi_vals_outer = np.mod(chi_vals_outer, 360)
                    
                    sort_idx_inner = np.argsort(chi_vals_inner)
                    sort_idx_outer = np.argsort(chi_vals_outer)
                    
                    chi_inner_sorted = chi_inner[sort_idx_inner].flatten()
                    chi_outer_sorted = chi_outer[sort_idx_outer].flatten()
                    
                    ichi_data.append(np.column_stack((chi_vals, chi)))
                    ichi_inner_data.append(np.column_stack((chi_vals_inner[sort_idx_inner], chi_inner_sorted)))
                    ichi_outer_data.append(np.column_stack((chi_vals_outer[sort_idx_outer], chi_outer_sorted)))
                    
                    coords_list.append((ycoord, xcoord))
                    
                    processed += 1
                    start = progress_bar(processed, total,
                                         prefix=f"[{Filenumber}] Iχ reduction",
                                         start_time=start)

                    
                    # Quick plot for sanity check
                    if Plot:
                        plt.figure(figsize=(8, 4))
                        plt.plot(chi_vals, chi, label='centre raw')
                        plt.plot(chi_vals_inner[sort_idx_inner], chi_inner_sorted, '--', label='inner ring', alpha=0.6)
                        plt.plot(chi_vals_outer[sort_idx_outer], chi_outer_sorted, '--', label='outer ring', alpha=0.6)
                        plt.title(f'I_χ frame at x(col) = {xcoord}, y(row) = {ycoord}')
                        plt.xlabel('Azimuthal angle (°)')
                        plt.ylabel('Intensity (a.u.)')
                        plt.legend()
                        plt.tight_layout()
                        plt.show()

    # Save .nxs files
    ichi_data = np.stack(ichi_data, axis=0)  # (n_frames, n_chi, 2)
    ichi_inner_data = np.stack(ichi_inner_data, axis=0)
    ichi_outer_data = np.stack(ichi_outer_data, axis=0)
    coords_array = np.array(coords_list, dtype=int)

    nxs_dir = f"{output_path}/{outputfolder}"
    os.makedirs(nxs_dir, exist_ok=True)

    with File(f"{nxs_dir}/i22-{Filenumber}_ichi.nxs", "w") as f:
        f.create_dataset("ichi", data=ichi_data)
        f.create_dataset("coords", data=coords_array)
    with File(f"{nxs_dir}/i22-{Filenumber}_ichi_inner.nxs", "w") as f:
        f.create_dataset("ichi_inner", data=ichi_inner_data)
        f.create_dataset("coords", data=coords_array)
    with File(f"{nxs_dir}/i22-{Filenumber}_ichi_outer.nxs", "w") as f:
        f.create_dataset("ichi_outer", data=ichi_outer_data)
        f.create_dataset("coords", data=coords_array)
        
    progress_bar(total, total, prefix=f"[{Filenumber}] Iχ reduction",
                     start_time=start, end=True)
    return 

def PlotDynamicChiRanges(Filenumber, avg_peak, x_peak, y_baseline_corrected, xplot, yplot, q_range_centre, inner_q, outer_q):
    
    print(f"[Dynamic IChi] File {Filenumber}: Average peak position = {avg_peak:.4f}, "
          f"q_centre=({q_range_centre[0]:.3f}, {q_range_centre[1]:.3f}), "
          f"q_inner=({inner_q[0]:.3f}, {inner_q[1]:.3f}), "
          f"q_outer=({outer_q[0]:.3f}, {outer_q[1]:.3f})")
    
    plt.plot(x_peak, y_baseline_corrected, label='BG corrected')
    plt.plot(xplot,yplot, label='raw')
    plt.axvline(q_range_centre[0], linestyle = '--', color='black')
    plt.axvline(q_range_centre[1], linestyle = '--', color='black', label='centre')
    plt.axvline(inner_q[0], linestyle = '--', color='green')
    plt.axvline(inner_q[1], linestyle = '--', color='green', label='inner')
    plt.axvline(outer_q[0], linestyle = '--', color='red')
    plt.axvline(outer_q[1], linestyle = '--', color='red', label='outer')
    plt.xlim(inner_q[0]-0.01, outer_q[1]+0.01)
    plt.legend()
    plt.ylim(-2,10)
    plt.title('Dynamic Ranges for IChi fit')
    plt.xlabel('q')
    plt.xlabel('I')
    plt.show()
