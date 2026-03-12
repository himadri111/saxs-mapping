#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 21 13:48:06 2025

@author: lauraforster
"""

import sys, time
import gc
import matplotlib.pyplot as plt

def _fmt_time(sec):
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    if h:   return f"{h:d}h{m:02d}m{s:02d}s"
    if m:   return f"{m:d}m{s:02d}s"
    return  f"{s:d}s"

def progress_bar(current, total, prefix="", width=30, start_time=None, end=False):
    """Single-line progress bar with ETA and elapsed. Overwrites the same line."""
    if total <= 0:
        total = 1
    current = max(0, min(current, total))
    frac = current / total
    filled = int(width * frac)
    bar = "█" * filled + "─" * (width - filled)
    pct = f"{frac*100:5.1f}%"

    # times
    now = time.time()
    if start_time is None:
        start_time = now
    elapsed = now - start_time
    if current > 0:
        eta = elapsed * (total / current - 1)
    else:
        eta = 0.0

    msg = f"\r{prefix} [{bar}] {pct} | {current}/{total} | elapsed { _fmt_time(elapsed) } | eta { _fmt_time(eta) }"
    sys.stdout.write(msg)
    sys.stdout.flush()

    if end or current >= total:
        sys.stdout.write("\n")
        sys.stdout.flush()

    return start_time  # so callers can keep passing it back

def cleanup():
    plt.close('all')
    gc.collect()

def aggressive_cleanup():
    # Close any figures that might have been left open
    try:
        plt.close("all")
    except Exception:
        pass

    # Garbage collect multiple times (helps with cyclic refs)
    gc.collect()
    gc.collect()
    
    
    
    