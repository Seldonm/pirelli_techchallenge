#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Author  : Mauro Losciale
# Email: losciale.mauro@gmail.com
# Version: 1.0.0
# =============================================================================
"""Script for feature engineering """
# =============================================================================
# Imports
import pandas as pd
import argparse
from datetime import datetime
import os

# =============================================================================
def valid_date(s):
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        msg = "not a valid date: {0!r}".format(s)
        raise argparse.ArgumentTypeError(msg)
        
parser = argparse.ArgumentParser(description="Feature engineering")
parser.add_argument("start_date", type=valid_date, help="Start date")
parser.add_argument("end_date",  type=valid_date, help="End date")

#MACHINE_ID = "m1"
#AREPA_TYPE = "a1"
OUTPUT_ROOT = "outputs/phase1"
KITCHEN_ID = "k1"
PHASE_ID = "phase1"
SOURCE_PATH = "source"
parse_dates = ['timestamp', 'start_time']
dtypes = {'metric_1':'float', 'metric_2':'float'}

def load_dataset(filename, delimiter=";", decimal=",", parse_dates=[]):
    try:
        return pd.read_csv(filename, delimiter=delimiter, decimal=decimal, parse_dates=parse_dates)
    except FileNotFoundError: 
        print(f"{filename} dataset not found")
        
def filter_input_date(cooking_metrics: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    
    metrics = cooking_metrics[cooking_metrics.timestamp.between(start_date, end_date)]
    
    if (metrics.shape[0] == 0):
        return pd.DataFrame()
    else:
        return metrics
        
def build_features(metrics, batch_registry, faulty_intervals, machine_id: str=None, arepa_type: str=None) -> pd.DataFrame:
        
    if (metrics.empty):
        print("No rows available for the specified date range")
        return pd.DataFrame()  
    
    with_faulty = metrics.merge(batch_registry, on='batch_id') \
                        .merge(faulty_intervals, on='machine_id')
                        
    with_faulty['faulty'] = (with_faulty['timestamp'].between(with_faulty['start_time'], with_faulty['end_time'])).astype(int)
    aggr_dict = {
        'timestamp': 'first',
        'machine_id': 'first',
        'batch_id': 'first',
        'metric_1': 'first',
        'metric_2': 'first',
        'arepa_type': 'first',
        'faulty': 'sum'
    }
    
    no_faulty = with_faulty.groupby(['timestamp', 'machine_id', 'batch_id']) \
                    .agg(aggr_dict) \
                    .query("faulty == 0") \
                    .drop('faulty', axis=1) \
                    .reset_index(drop=True) \
                    .set_index('timestamp')
    
    h_sampled = no_faulty.groupby(['machine_id', 'arepa_type']) \
                    .resample('H') \
                    .mean(numeric_only=True) \
                    .reset_index()

    h_sampled.set_index('timestamp', inplace=True)
    
    if (h_sampled.shape[0] > 0):
        h_sampled.index = h_sampled.index.strftime("%Y-%m-%dT%H:%M:%S")
        
        if not (machine_id) == None:
            h_sampled = h_sampled.loc[h_sampled['machine_id'] == machine_id]
            
        if not (arepa_type) == None:
            h_sampled = h_sampled.loc[h_sampled['arepa_type'] == arepa_type]
        
        return h_sampled
    else:
        print("No features available for the specified date range")
        return pd.DataFrame()
    
def write_results(results: pd.DataFrame, start_date: datetime, end_date: datetime, machine_id: str=None, arepa_type: str=None):
    st_date = start_date.strftime("%Y%m%dT%H%M%S")
    ed_date = end_date.strftime("%Y%m%dT%H%M%S")
    OUTPUT_ROOT = f"outputs/{PHASE_ID}"
    
    outpath = f"{OUTPUT_ROOT}/{st_date}_{ed_date}"
    
    if not os.path.exists(outpath):
        os.makedirs(outpath)
        
    if not (machine_id) == None and not (arepa_type) == None:
        results.to_csv(f"{outpath}/{KITCHEN_ID}_{machine_id}_{arepa_type}.csv")
    elif not (machine_id) == None:
        results.to_csv(f"{outpath}/{KITCHEN_ID}_{machine_id}_all_arepas.csv")
    else:
        results.to_csv(f"{outpath}/{KITCHEN_ID}_all_machines_{arepa_type}.csv")
    
if __name__ == "__main__":
    
    MACHINE_ID = "m1"
    AREPA_TYPE = "a1"  
    
    args = parser.parse_args()
    cooking_metrics = load_dataset(filename=f"{SOURCE_PATH}/cooking_metrics.csv", parse_dates=['timestamp'])
    batch_registry = load_dataset(filename=f"{SOURCE_PATH}/batch_registry.csv")
    faulty_intervals = load_dataset(filename=f"{SOURCE_PATH}/faulty_intervals.csv", parse_dates=['start_time', 'end_time'])
    
    filtered = filter_input_date(cooking_metrics, args.start_date, args.end_date)
    result = build_features(cooking_metrics, batch_registry, faulty_intervals, MACHINE_ID, AREPA_TYPE)
    
    if not (result.empty):
        write_results(result, args.start_date, args.end_date, MACHINE_ID, AREPA_TYPE)
    