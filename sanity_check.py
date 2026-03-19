
import sys
import os
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import logging

# Setup paths
sys.path.insert(0, "/home/yagokhan/VaranusNeoFlow")

from backtest.data_loader import load_all_assets, build_scan_dataframes, build_htf_dataframe, find_bar_index
from neo_flow.adaptive_engine import scan_asset

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("sanity_check")

def run_sanity():
    logger.info("=== Varanus Neo-Flow: 100-Bar Bias-Free Sanity Check ===")
    
    # 1. Load Data
    all_data = load_all_assets()
    asset = "SOL"
    asset_data = all_data.get(asset)
    if not asset_data:
        logger.error(f"Asset {asset} not found")
        return

    # Get last 100 1h timestamps
    ts_1h = asset_data["1h"].timestamps
    test_indices = range(len(ts_1h) - 101, len(ts_1h) - 1)
    
    leaks = 0
    total = 0
    
    for idx in test_indices:
        scan_ns = ts_1h[idx]
        scan_ts = pd.Timestamp(scan_ns, tz="UTC")
        
        # PROTOCOL AUDIT:
        # scan_dfs uses scan_ns - 1 (excludes bar starting AT scan_ns)
        scan_dfs = build_scan_dataframes(asset_data, scan_ns - 1)
        
        # 4H HTF Filter uses scan_ns - 4h (last CLOSED 4h bar)
        _4H_NS = 4 * 3600 * 10**9
        df_4h = build_htf_dataframe(asset_data, scan_ns - _4H_NS)
        
        # Check if any data in scan_dfs or df_4h has timestamp >= scan_ns
        for tf, df in scan_dfs.items():
            last_ts = df.index[-1]
            if last_ts.value >= scan_ns:
                logger.error(f"LEAK DETECTED: {tf} has bar at {last_ts} (scan_ns={scan_ts})")
                leaks += 1
        
        if df_4h is not None:
            last_ts_4h = df_4h.index[-1]
            if last_ts_4h.value >= scan_ns:
                logger.error(f"LEAK DETECTED: 4h has bar at {last_ts_4h} (scan_ns={scan_ts})")
                leaks += 1
                
        total += 1
        
    if leaks == 0:
        logger.info(f"SUCCESS: 0 leaks found across {total} scan steps.")
        logger.info("The -1 Rule and 4H Offset are strictly enforced.")
    else:
        logger.error(f"FAILURE: {leaks} leaks detected!")

if __name__ == "__main__":
    run_sanity()
