#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 17:19:09 2025

@author: lauraforster
"""
import time
import os 
import numpy as np
from pathlib import Path
import Utils as Utils
import pandas as pd

import ReductionScript as RS
import FittingScript as FS
import VisualisingScript as VS

start_time = time.time()

# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------ Inputs -----------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------------------------------------------
Experiment = 'Dec23' #or Dec23, or Feb22 or July23 or Feb25 or May24

# Note: During some experiments where grid scanning was unavailable - line scans were taken using a custom python script and can be stitched together into 2D images using this script. For normal files ignore 'Split' logic
Start = 725339		   #Split Files ONLY
End   = 725395        #Split Files ONLY

Filelist = [692820, 692821]
# ------------------------------------------------------------------------------------------------------------------------
# Define frame numbers to process 
# xframe = [1,2,3,4]
# yframe = [5,6,7]

# OR process all frames
xframe = None
yframe = None
# ------------------------------------------------------------------------------------------------------------------------
# REDUCTION
ProcessIqRed   = False           #Iq Reduction
ProcessIChiRed = False           #IChi Reduction

# FITTING 
ProcessIqFit   = True            #Iq Fitting
ProcessIChiFit = False           #IChi Fitting

# VISUALSE ANALYSIS 
Visualise      = True           #Visualise parameters
HeatmapPlot    = True           #View IQ parameter heatmap
AngularOverlay = True           #View IChi parameter heatmap
# ------------------------------------------------------------------------------------------------------------------------
# ICHI - Ring vs Entire ANALYSIS
DynamicIchi       = False       #Dynamically create the IChi ring positions in q from IQ data (If False use define chi ring positions)
IChi_EntireReduct = True        #True = Reduce IChi over single set q range /// False = Reduce to three rings (one for region of interest and two for inner and outer background)
IChi_EntireFit    = True        #True = Reduce IChi over single set q range /// False = Fit using three rings (one for region of interest and two for inner and outer background)

# Check Shapes - Check the shape of the resultant output to ensure it matches
CheckIqRed = False              #Check Iq reduction shape
CheckBSD   = False              #Check BSDiodes output shape
CheckIqCSV = False              #Check Iq fit paramete output shape
# ------------------------------------------------------------------------------------------------------------------------
# PLOTS DISPLAY
#Select True to display inline plots for each processed Frame, False will produce a PDF 
PlotIqRed   = False             #Iq Reduction
PlotIqFit   = False             #Iq Fitting
PlotIchiRed = False             #Ichi Reduction
PlotIchiFit = False             #Ichi Fitting

chiRangesPlot = False           #Ichi ranges for rings
# ------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------Define reduction Parameters------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# IQ REDUCTION
nq = 4000                       #No. points in q
chi_range = [0,360]             #Radial range
q_range = [0.1,1.1]             #q range

# ICHI REDUCTION
nchi = 180                      #No. points in chi
chi_range_chi = [0,360]         #Radial range
q_range_centre = [0.1,1.1]      #q range

# If using rings method (which are not defined by the q positioning)
inner_q = [0.26,0.27]           #Inner Ichi range
outer_q = [0.31,0.32]           #Outer Ichi range
cent, inn, outt = 0.01, 0.02, 0.03 #Dynamic Ranges for centre, inner and outer rings per file

# ------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------DefineFitting Parameters---------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# ******************************************************  IQ FIT  ***********************************************************
order = 3                       # Peak order number to fit
order_position = 0.292          # Approximate q position for the peak
x_peak_wid = 0.02               # Approximate width of the peak
xborderLL, xborderLR = 0.02, 0.012  # LEFT of the peak max and min, for baseline subtraction
xborderRL, xborderRR = 0.012, 0.02  # RIGHT of the peak max and min, for baseline subtraction
minimum_area_threshold = 0.05   # Minimum peak area % 0.05
rsq_min = 0.2                   # Minimum Rsq for CSV saving
A,b,d = 200, -30, 0.1           # Fit values

# ******************************************************  ICHI FIT ********************************************************

# ******************************************************  HEATMAP  *****************************************************
WhatPlot = 'Dperiod' #'SAXS','SAXS_norm' 'curvearea', 'curvearea_norm' , 'Dperiod'. 'wMu'

threshold_saxs_intensity = 0.0  # Minimum threshold for total SAXS intensity
threshold_area = 0.0            # Minimum threshold for total col intensity
threshold_Dperiod = 60          # Minimum threshold for D-period
perc_above_baseline = 0         # Percentage of points above the baseline to be included
max_intensity = 0.3             # Minumum intensity for Visualisation - bottom percentage of total SAXS not visualised
threshold_areaIChi = 0.15       # Minimum area threshold for IChi

if WhatPlot == 'SAXS':
    zmin, zmax = 0, 2000
if WhatPlot == 'SAXS_norm' or 'curvearea_norm':
    zmin, zmax = 0, 1
if WhatPlot == 'curvearea':
    zmin, zmax = 0, 0.05
if WhatPlot == 'Dperiod':
    zmin, zmax = 64, 68
if WhatPlot == 'wMu':
    zmin, zmax = 0, 0.15
if WhatPlot == 'fibril_radius':
    zmin, zmax = 0, 400

# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------Paths -----------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
sample_beginning = "i22-"
sample_end = ".nxs"

if Experiment == 'May24':
    identifier = 'Single' 
    MASK_PATH = Path('/Volumes/Expansion/Calib Files/sm36684May24/SAXS_mask.nxs')
    CALIBRANT_PATH = Path('/Volumes/Expansion/Calib Files/sm36684May24/SAXS_calibration.nxs')
    sample_loc = "/Volumes/Seagate/dls/sm36684-1_LauraForster/"
    
if Experiment == 'Dec23':
    identifier = 'Split' 
    MASK_PATH = Path('/Volumes/Expansion/Calib Files/sm34636Dec23/SAXS_mask.nxs')
    CALIBRANT_PATH = Path('/Volumes/Expansion/Calib Files/sm34636Dec23/SAXS_calibration.nxs')
    Filenumbers = (End - Start) + 1
    filelist = np.linspace(Start, End, Filenumbers, dtype=int)
    sample_loc = "/Volumes/Seagate/sm34636-1/sm34636-1/" 
    
if Experiment == 'Feb22':
    identifier = 'Single' 
    MASK_PATH = Path('/Volumes/Expansion/Calib Files/smxxxxFeb22/SAXS_mask.nxs')
    CALIBRANT_PATH = Path('/Volumes/Expansion/Calib Files/smxxxxFeb22/SAXS_calibration.nxs')
    sample_loc = "/Volumes/Seagate/sm29784-5/" 

if Experiment == 'July23':
    identifier = 'Single' 
    MASK_PATH = Path('/Volumes/Expansion/Calib Files/sm33398July23/SAXS_mask.nxs')
    CALIBRANT_PATH = Path('/Volumes/Expansion/Calib Files/sm33398July23/SAXS_calibration.nxs')
    sample_loc = "/Volumes/Seagate/sm33398-1/" 

if Experiment == 'Feb25':
    identifier = 'Single' 
    MASK_PATH = Path('/Volumes/LauraDrive/Calib Files/sm38399Feb25/SAXS_mask.nxs')
    CALIBRANT_PATH = Path('/Volumes/LauraDrive/Calib Files/sm38399Feb25/SAXS_calibration.nxs')
    sample_loc = "/Volumes/Seagate/dls/sm38399-1_LauraForster/" 
    
if Experiment == 'Test':
    identifier = 'Single' 
    MASK_PATH = Path('/Volumes/Expansion/Calib Files/sm38399Feb25/SAXS_mask.nxs')
    CALIBRANT_PATH = Path('/Volumes/Expansion/Calib Files/sm38399Feb25/SAXS_calibration.nxs')
    sample_loc = "/Volumes/Seagate/dls/sm38399-1_LauraForster/" 
    
base_output_path = f'/Volumes/LauraDrive/DLS visits/{Experiment}' 
Output_directoryCSV = f'/Users/lauraforster/Documents/Uni/3 - PhD/SAXS/DLS visits/{Experiment}/CSVs/'
Output_directorybsd = f'/Users/lauraforster/Documents/Uni/3 - PhD/SAXS/DLS visits/{Experiment}/BSDs/'
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------ Reduction & Fitting ---------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# --------Load mask  and calibration files
if ProcessIqRed == True or ProcessIChiRed == True:
    calib = RS.MaskCalib(MASK_PATH, CALIBRANT_PATH)

# *********************************************************Split Files****************************************************
# --------Reduce Iq and IChi for Split files (lots of line scans stacked into grid scan)
if identifier == 'Split':
    Filenumbers = (End - Start) + 1
    filelist = np.linspace(Start, End, Filenumbers, dtype=int)
    outputfolder = f"{sample_beginning}{Start}"
    
    Iq_folder = os.path.join(base_output_path, "Iq")
    Ichi_folder = os.path.join(base_output_path, "Ichi")

    frame_idxIQ,frame_idxICHI = 0, 0
    
    file_start, file_end = Start,End
    Filenumber = str(Start)

# --------Reduce Iq 
    if ProcessIqRed == True:
        RS.ReductionIQ(Filenumber, sample_loc, base_output_path, outputfolder, sample_beginning, sample_end, identifier, frame_idxIQ, nq, nchi, q_range,chi_range, calib, PlotIqRed, file_start, file_end, x_list=xframe, y_list=yframe)
        print(f'I_q Reduction file outputted to {base_output_path} as .nxs file')
        # Check Sizes of outputs
    if CheckIqRed == True:
        RS.CheckIQRed(Filenumber, Output_directoryCSV, base_output_path, sample_loc, file_start, file_end, identifier)
# --------Fit Iq    
    if ProcessIqFit == True:
        Filenumber = str(Filenumber)
        xplot,yplot, x_peak, y_baseline_corrected = FS.ProcessIQFitting(Filenumber, Output_directoryCSV, base_output_path, Output_directorybsd, sample_loc, file_start, file_end, identifier, order, x_peak_wid, xborderLL, xborderLR, xborderRL, xborderRR, order_position, minimum_area_threshold, rsq_min,A,b,d, PlotIqFit, xcoords=xframe, ycoords=yframe)
        # Check Sizes of outputs
    if CheckBSD == True:
        FS.CheckBSD(Filenumber, Output_directoryCSV, base_output_path, Output_directorybsd, sample_loc, file_start, file_end, identifier)
    if CheckIqCSV == True:
        FS.CheckIqCSV(Filenumber, Output_directoryCSV)
# --------Generate IChi rings from Iq data
    if DynamicIchi == True:
        # Check that the CSV file has already been generated
        csv_path = os.path.join(Output_directoryCSV, f"{Filenumber} IQ_fitting.csv")
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            valid_peaks = df['peak_position_third'][df['peak_position_third'] > 0]
            if len(valid_peaks) > 0:
                avg_peak = valid_peaks.mean()
                # Define q ranges
                q_range_centre = [round(avg_peak - cent, 3), round(avg_peak + cent, 3)]
                inner_q = [round(avg_peak - outt, 3), round(avg_peak - inn, 3)]
                outer_q = [round(avg_peak + inn, 3), round(avg_peak + outt, 3)]
                if chiRangesPlot == True:
                    RS.PlotDynamicChiRanges(Filenumber, avg_peak, x_peak, y_baseline_corrected, xplot, yplot, q_range_centre, inner_q, outer_q)
            else:
                print('\n')
                print(f"No valid peaks found in {csv_path}, skipping dynamic ichi.")
        else:
            print(f"CSV file for {Filenumber} not found at {csv_path}, skipping dynamic ichi setup.")
    else:
        try:
            if chiRangesPlot == True:
                RS.PlotDynamicChiRanges(Filenumber, avg_peak, x_peak, y_baseline_corrected, xplot, yplot, q_range_centre, inner_q, outer_q)
        except:
                print()
# --------Reduce Ichi
    if ProcessIChiRed == True:
        RS.ReductionICHI(Filenumber, sample_loc, base_output_path, outputfolder, sample_beginning, sample_end,identifier, frame_idxICHI, nq, nchi, q_range, q_range_centre, chi_range_chi, inner_q, outer_q, calib, PlotIchiRed, IChi_EntireReduct, file_start, file_end, x_list=xframe, y_list=yframe)
        if IChi_EntireReduct == True:
            print(f'I_Chi Reduction (centre files) outputted to {base_output_path} as .nxs file')
        else:
            print(f'I_Chi Reduction (inner, outer and centre files) outputted to {base_output_path} as .nxs file')

# --------Fit Ichi 
    if ProcessIChiFit == True:
        FS.ProcessICHIFitting(Filenumber, outputfolder, base_output_path, Output_directoryCSV, threshold_areaIChi, PlotIchiFit, IChi_EntireFit, xcoords=xframe, ycoords=yframe)
        print(f'I_Chi Fitting outputted to {Output_directoryCSV} as .csv file')


# *********************************************************Single Files****************************************************
# --------Reduce Iq and IChi for Single files (single grid scan)
else:
    for Filenumber in Filelist:
        Filenumber = str(Filenumber)
        Start = End = Filenumber
        file_start, file_end = Start,End
        outputfolder = str(sample_beginning + str(Filenumber))
        
        Iq_folder = base_output_path + "/Iq"
        Ichi_folder = base_output_path + "/Ichi"
        
        frame_idxIQ,frame_idxICHI = 0, 0
 # --------Reduce Iq   
        if ProcessIqRed == True:
            RS.ReductionIQ(Filenumber, sample_loc, base_output_path, outputfolder, sample_beginning, sample_end, identifier, frame_idxIQ, nq, nchi, q_range,chi_range, calib, PlotIqRed, file_start=None, file_end=None, x_list=xframe, y_list=yframe)
            print(f'I_q Reduction file outputted to {base_output_path} as .nxs file')
        # Check Sizes of outputs
        if CheckIqRed == True:
            RS.CheckIQRed(Filenumber, Output_directoryCSV, base_output_path, sample_loc, file_start, file_end, identifier)
# --------Fit Iq  
        if ProcessIqFit == True:
            xplot,yplot, x_peak, y_baseline_corrected = FS.ProcessIQFitting(Filenumber, Output_directoryCSV, base_output_path, Output_directorybsd, sample_loc, file_start, file_end, identifier, order, x_peak_wid, xborderLL, xborderLR, xborderRL, xborderRR, order_position, minimum_area_threshold, rsq_min, A,b,d,PlotIqFit, xcoords=xframe, ycoords=yframe)
            print(f'I_Q Fitting outputted to {Output_directoryCSV} as .csv file')
        # Check Sizes of outputs
        if CheckBSD == True:
            FS.CheckBSD(Filenumber, Output_directoryCSV, base_output_path, Output_directorybsd, sample_loc, file_start, file_end, identifier)
        if CheckIqCSV == True:
            FS.CheckIqCSV(Filenumber, Output_directoryCSV)
# --------Generate IChi rings from Iq data          
        if DynamicIchi == True:
            # Check that the CSV file has already been generated
            csv_path = os.path.join(Output_directoryCSV, f"{Filenumber} IQ_fitting.csv")
            
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                valid_peaks = df['peak_position_third'][df['peak_position_third'] > 0]
                if len(valid_peaks) > 0:
                    avg_peak = valid_peaks.mean()
                    q_range_centre = [round(avg_peak - cent, 3), round(avg_peak + cent, 3)]
                    inner_q = [round(avg_peak - outt, 3), round(avg_peak - inn, 3)]
                    outer_q = [round(avg_peak + inn, 3), round(avg_peak + outt, 3)]
                    if chiRangesPlot == True:
                        try:
                            RS.PlotDynamicChiRanges(Filenumber, avg_peak, x_peak, y_baseline_corrected, xplot, yplot, q_range_centre, inner_q, outer_q)
                        except:
                            print()
                else:
                    print('\n')
                    print(f"No valid peaks found in {csv_path}, skipping dynamic ichi.")
            else:
                print(f"CSV file for {Filenumber} not found at {csv_path}, skipping dynamic ichi setup.")
        else:
            try:
                if chiRangesPlot == True:
                    RS.PlotDynamicChiRanges(Filenumber, avg_peak, x_peak, y_baseline_corrected, xplot, yplot, q_range_centre, inner_q, outer_q)
            except:
                print()
# --------Reduce Ichi
        if ProcessIChiRed == True:
            RS.ReductionICHI(Filenumber, sample_loc, base_output_path, outputfolder, sample_beginning, sample_end,identifier, frame_idxICHI, nq, nchi, q_range, q_range_centre, chi_range_chi, inner_q, outer_q, calib, PlotIchiRed, IChi_EntireReduct, file_start=None, file_end=None, x_list=xframe, y_list=yframe)
            if IChi_EntireReduct == True:
                print(f'I_Chi Reduction (centre files) outputted to {base_output_path} as .nxs file')
            else:
                print(f'I_Chi Reduction (inner, outer and centre files) outputted to {base_output_path} as .nxs file')
# --------Fit Ichi 
        if ProcessIChiFit == True:
            FS.ProcessICHIFitting(Filenumber, outputfolder, base_output_path, Output_directoryCSV, threshold_areaIChi, PlotIchiFit, IChi_EntireFit, xcoords=xframe, ycoords=yframe)
            print(f'I_Chi Fitting outputted to {Output_directoryCSV} as .csv file')
        
        Utils.cleanup()
        Utils.aggressive_cleanup()
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------ Heatmap plotting ------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------

# --------Visualise via heatmap
if Visualise == True:
    VS.heatmap(Filenumber, Output_directoryCSV, HeatmapPlot, AngularOverlay, WhatPlot, threshold_saxs_intensity, threshold_area, threshold_Dperiod, zmin, zmax, perc_above_baseline, max_intensity)

# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------ End -------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------------------------------------


print('\n')
end_time = time.time() 
elapsed_time = end_time - start_time

# Calculate minutes and seconds
minutes, seconds = divmod(elapsed_time, 60)

# Print the result
print(f"Analysis of SAXS finished, Time Elapsed: {int(minutes)} minutes {np.round(seconds, 2)} seconds")

# Play a tune at the end of the code
# os.system('afplay /Users/lauraforster/Desktop/chime-and-chomp-84419.mp3')






