# saxs-mapping
Pipeline for analyzing synchrotron SAXS maps (2D raster maps of 2D SAXS patterns) including pyFAI integration, model fitting, nanostructural parameter extraction, and visualisation

## Overview

This repository contains a pipeline for analysing synchrotron SAXS raster maps
(2D grids of SAXS patterns). The workflow includes:

1. Azimuthal integration using pyFAI
2. Peak fitting of I(q) curves
3. Extraction of nanoscale structural parameters
4. Visualisation as spatial heatmaps

## Main scripts

- `SAXS_DriverFile.py` – main pipeline controller
- `ReductionScript.py` – pyFAI integration of detector images
- `FittingScript.py` – model fitting of I(q) curves
- `VisualisingScript.py` – generation of parameter maps
- `Utils.py` – helper functions

## Requirements

Install dependencies with:
pip install -r requirements.txt

## Status

This repository contains research analysis code and is under active development.
