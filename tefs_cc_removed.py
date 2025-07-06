import numpy as np
import pandas as pd
import glob
from tqdm import tqdm
import time
import os
from utils import estimate_conditional_transfer_entropy
from New_tefs import TEFS
from collections import Counter
import pickle

# Analysis parameters
ICE_NAME = 'sic'
PRED_SEASON = 'ONDJF'
TARG_SEASON = 'F'
CYCLE_LEN = 6
LAG_FEATURES = [1,2,3,4,5]
LAG_TARGET = [1]
DIRECTION = 'forward'
OUTPUT_DIR = f'tefs_results/{ICE_NAME}_{TARG_SEASON}_nocc'
PARALLEL = True
N_JOBS = 32
os.makedirs(OUTPUT_DIR, exist_ok=True)

if TARG_SEASON == 'S':
    targ_idx = 20
elif TARG_SEASON == 'F':
    targ_idx = 25

# Load ice data
print("Loading sea ice data...")
sit = np.load('data/SIT_ease.npy')
sit = (sit - np.nanmin(sit)) / (np.nanmax(sit) - np.nanmin(sit))
sic = np.load('data/SIC_ease.npy')
sic = (sic - np.nanmin(sic)) / (np.nanmax(sic) - np.nanmin(sic))
ice = {'sic': sic, 'sit': sit}
ice_var = ice[ICE_NAME]
k = ice_var.shape[0] // (10*CYCLE_LEN)

def index_timeseries(index_name):
    df = pd.read_csv(f'data/{index_name}.csv')
    as_arr = np.array(df)
    flattened = np.ravel(as_arr[:, 1:])
    normalized = (flattened - np.min(flattened)) / (np.max(flattened) - np.min(flattened))
    return normalized

def extract_index_name(filename):
    return filename.split('\\')[1][:-4]

# Load index data
print("Loading teleconnection indices...")
index_files = glob.glob('data/*.csv')
index_names = [extract_index_name(filename) for filename in index_files]
indexes = {index_name: index_timeseries(index_name) for index_name in index_names}
df = pd.DataFrame(indexes)
features_names = list(df.columns)

# Create a mapping where order doesn't matter
def get_sorted_tuple(feat_list):
    """Convert a list of features to a sorted tuple (where order doesn't matter)"""
    return tuple(sorted(feat_list))

# Initialize result array to store selected features for each pixel
feature_selections = np.empty((ice_var.shape[1], ice_var.shape[2]), dtype=object)
feature_selections.fill(None)  # Initialize with None for pixels we don't process

# Function to process a single pixel
def process_pixel(lat, lon):
    try:
        ice_var_ = ice_var[:, lat, lon]
        if np.isnan(ice_var_).any():
            return lat, lon, None
        
        # Set up dataset
        df_copy = df.copy()
        df_copy[ICE_NAME] = ice_var_
        df_redux = pd.DataFrame(columns=df_copy.columns)
        num_cycles = (len(df_copy) // 12) - 2  # 2 years sacrificed
        df_slices = []
        
        for i in range(num_cycles):
            ids = np.array([9,10,11,12,13,targ_idx]) + i*12
            df_slice = df_copy.loc[ids, :]
            df_slices.append(df_slice)
        
        df_redux = pd.concat(df_slices)
        df_redux.reset_index(inplace=True,drop=True)
        features = df_redux.drop(columns=[ICE_NAME])
        target = df_redux[ICE_NAME]
        
        # Run feature selection
        fs = TEFS(
            features=features.values,
            target=target.values,
            k=k,
            lag_features=LAG_FEATURES,
            lag_target=LAG_TARGET,
            direction=DIRECTION,
            cycle_len=CYCLE_LEN,
            verbose=0,
            var_names=features_names,
            n_jobs=1  # Use 1 for the inner loop since we parallelize the outer loop
        )
        fs.fit()
        selected_features = fs.select_features(threshold=np.inf)
        
        # Store sorted tuple (since order doesn't matter for our purposes)
        if selected_features:
            return lat, lon, get_sorted_tuple(selected_features)
        else:
            return lat, lon, tuple()  # Empty tuple if nothing selected
    except Exception as e:
        print(f"Error processing pixel ({lat}, {lon}): {e}")
        return lat, lon, None

# Function to run the analysis and save results
def run_feature_selection():
    start_time = time.time()
    
    # Get valid pixels (non-NaN)
    valid_pixels = []
    for lat in range(ice_var.shape[1]):
        for lon in range(ice_var.shape[2]):
            if not np.isnan(ice_var[:, lat, lon]).any():
                valid_pixels.append((lat, lon))
    
    print(f"Processing {len(valid_pixels)} valid pixels...")
    
    # Process pixels (parallel or sequential)
    results = []
    if PARALLEL:
        from joblib import Parallel, delayed
        batch_size = max(1, len(valid_pixels) // 100)  # Create ~100 batches for progress tracking
        
        # Process in parallel with progress tracking
        results = Parallel(n_jobs=N_JOBS, verbose=10)(
            delayed(process_pixel)(lat, lon) 
            for lat, lon in valid_pixels
        )
    else:
        # Sequential processing with tqdm progress bar
        for lat, lon in tqdm(valid_pixels, desc="Processing pixels"):
            results.append(process_pixel(lat, lon))
    
    # Fill the result array
    for lat, lon, selected in results:
        if selected is not None:
            feature_selections[lat, lon] = selected
    
    # Calculate some basic statistics
    non_empty_selections = sum(1 for lat in range(ice_var.shape[1]) 
                              for lon in range(ice_var.shape[2]) 
                              if feature_selections[lat, lon] is not None and 
                                 len(feature_selections[lat, lon]) > 0)
    
    total_valid = sum(1 for lat in range(ice_var.shape[1]) 
                      for lon in range(ice_var.shape[2]) 
                      if feature_selections[lat, lon] is not None)
    
    # Count unique combinations
    combos = Counter()
    for lat in range(ice_var.shape[1]):
        for lon in range(ice_var.shape[2]):
            if feature_selections[lat, lon] is not None:
                combos[feature_selections[lat, lon]] += 1
    
    # Save the results
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\nFeature selection completed in {elapsed_time:.2f} seconds")
    print(f"Valid pixels: {total_valid}")
    print(f"Pixels with at least one feature selected: {non_empty_selections} ({non_empty_selections/total_valid*100:.2f}%)")
    print(f"Unique feature combinations: {len(combos)}")
    
    # Print most common combinations
    print("\nTop 10 most common feature combinations:")
    for combo, count in combos.most_common(10):
        if combo == tuple():
            combo_str = "No indices selected"
        else:
            combo_str = ", ".join(combo)
        percentage = (count / total_valid) * 100
        print(f"{combo_str}: {count} pixels ({percentage:.2f}%)")
    
    # Save full results array
    result_file = os.path.join(OUTPUT_DIR, f'feature_selections.pkl')
    with open(result_file, 'wb') as f:
        pickle.dump(feature_selections, f)
    print(f"Full feature selection results saved to {result_file}")
    
    # Save a more detailed statistics file for later analysis
    stats = {
        'run_time': elapsed_time,
        'total_valid_pixels': total_valid,
        'pixels_with_features': non_empty_selections,
        'feature_combos': dict(combos),
        'parameters': {
            'ice_var': ICE_NAME,
            'pred_season': PRED_SEASON,
            'targ_season': TARG_SEASON,
            'cycle_len': CYCLE_LEN,
            'lag_features': LAG_FEATURES,
            'lag_target': LAG_TARGET,
            'direction': DIRECTION,
            'k': k
        }
    }
    
    stats_file = os.path.join(OUTPUT_DIR, f'selection_stats.pkl')
    with open(stats_file, 'wb') as f:
        pickle.dump(stats, f)
    print(f"Selection statistics saved to {stats_file}")
    
    # Save a NumPy version for easier loading in different environments
    np_file = os.path.join(OUTPUT_DIR, f'feature_selections.npz')
    
    # Convert to a format that can be saved by NumPy
    # We'll use integers to represent each unique combination
    combo_to_id = {combo: i for i, combo in enumerate(combos.keys())}
    id_to_combo = {i: combo for combo, i in combo_to_id.items()}
    
    # Create integer array (pixels with no results will be -1)
    int_array = np.full((ice_var.shape[1], ice_var.shape[2]), -1, dtype=np.int32)
    for lat in range(ice_var.shape[1]):
        for lon in range(ice_var.shape[2]):
            if feature_selections[lat, lon] is not None:
                int_array[lat, lon] = combo_to_id[feature_selections[lat, lon]]
    
    # Save both the integer array and the mapping
    #np.savez(np_file, 
    #         int_array=int_array, 
    #         id_to_combo_keys=np.array(list(id_to_combo.keys())),
    #         id_to_combo_values=np.array([id_to_combo[i] for i in id_to_combo.keys()], dtype=object))
    
    print(f"NumPy version saved to {np_file}")
    return feature_selections, stats

if __name__ == "__main__":
    print(f"Starting feature selection for {ICE_NAME} ({TARG_SEASON})")
    feature_selections, stats = run_feature_selection()
    print("Done!")