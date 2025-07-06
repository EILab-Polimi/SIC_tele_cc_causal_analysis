import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed
from itertools import product
from scipy.signal import detrend

# Custom module imports
from New_tefs.estimation import estimate_conditional_transfer_entropy

# ---- Configuration ----
COND_NAME = 'GISTEMP_monthly_temp'
PRED_SEASON = 'ONDJF'
CYCLE_LEN = 6
LAG_FEATURES = [1, 2, 3, 4, 5]
LAG_TARGET = [1]

PREDICTORS = ['ice', 'tele']
ICE_NAMES = ['sic', 'sit']
CONDITIONING = [True, False]
TARG_SEASON = ['F', 'S']


# ---- Utility Functions ----
def detrend_data(array):
    """
    Detrend along the time dimension (axis=0).
    NaNs will be preserved.
    """
    detrended_array = np.full(array.shape, np.nan)
    lat_dim, lon_dim = array.shape[1], array.shape[2]

    for lat in range(lat_dim):
        for lon in range(lon_dim):
            pixel_series = array[:, lat, lon]
            if not np.isnan(pixel_series).any():
                detrended_array[:, lat, lon] = detrend(pixel_series, type='linear')
    return detrended_array

def index_timeseries(index_name):
    """
    Load and normalize a timeseries from a CSV file.
    
    Args:
        index_name (str): Base name of the CSV file (without path or extension).
    
    Returns:
        np.ndarray: Normalized flattened timeseries.
    """
    df = pd.read_csv(f'data/{index_name}.csv')
    # Skip the first column (assumed to be non-data) and flatten the rest.
    data = np.array(df)[:, 1:]
    flattened = data.ravel()
    normalized = (flattened - np.min(flattened)) / (np.max(flattened) - np.min(flattened))
    return normalized


def extract_index_name(filepath):
    """
    Extract the index name from a file path.
    
    Args:
        filepath (str): Full path to the file.
    
    Returns:
        str: File name without extension.
    """
    base = os.path.basename(filepath)
    name, _ = os.path.splitext(base)
    return name


def load_and_normalize_ice_data():
    """
    Load and normalize ice data from numpy files.
    
    Returns:
        dict: Dictionary with keys 'sit' and 'sic' containing normalized data.
    """
    sit = np.load('data/SIT_ease.npy')
    sic = np.load('data/SIC_ease.npy')

    sit = detrend_data(sit)
    sic = detrend_data(sic)
    
    sit = (sit - np.nanmin(sit)) / (np.nanmax(sit) - np.nanmin(sit))
    sic = (sic - np.nanmin(sic)) / (np.nanmax(sic) - np.nanmin(sic))
    return {'sit': sit, 'sic': sic}


def process_grid_point(lat, lon, df_index, ice_var, ice_name, slice_indices,
                       index_name, predictor, conditioning, cond_var, k):
    """
    Process a single grid point for transfer entropy calculation.
    
    Args:
        lat (int): Latitude index.
        lon (int): Longitude index.
        df_index (pd.DataFrame): DataFrame containing index timeseries.
        ice_var (np.ndarray): 3D array of ice data.
        ice_name (str): Name of the ice variable ('sic' or 'sit').
        slice_indices (list): List of numpy arrays with DataFrame slice indices.
        index_name (str): Name of the index variable.
        predictor (str): Either 'ice' or 'tele' indicating the predictor.
        conditioning (bool): Whether to include the conditioning variable.
        cond_var (np.ndarray or None): Conditioning variable timeseries.
        k (int): Parameter computed from the time series length.
        
    Returns:
        float: Computed transfer entropy or np.nan if the grid point is invalid.
    """
    ice_series = ice_var[:, lat, lon]
    if np.isnan(ice_series).any():
        return np.nan

    # Add ice data to a copy of the index DataFrame.
    df = df_index.copy()
    df[ice_name] = ice_series
    if conditioning:
        df['GISTEMP'] = cond_var
        
    # Concatenate the pre-sliced portions.
    df_redux = pd.concat([df.iloc[ids] for ids in slice_indices]).reset_index(drop=True)
    
    if predictor == 'tele':
        X = df_redux[index_name].values
        Y = df_redux[ice_name].values 
    else:
        X = df_redux[ice_name].values
        Y = df_redux[index_name].values
    
    if conditioning:
        Z = df_redux['GISTEMP'].values
        lag_conditioning = LAG_FEATURES
    else:
        Z = df_redux[list()].values

    te = estimate_conditional_transfer_entropy(
        X=X,
        Y=Y,
        Z=Z,
        k=k,
        lag_features=LAG_FEATURES,
        lag_target=LAG_TARGET,
        lag_conditioning=None,
        cycle_len=CYCLE_LEN,
    )
    return max(te, 0)


def run_experiment(predictor, ice_name, conditioning, targ_season,
                   ice, cond_var, index_files, k):
    """
    Run a single experiment configuration over all index files.
    
    Args:
        predictor (str): 'ice' or 'tele'.
        ice_name (str): 'sic' or 'sit'.
        conditioning (bool): Whether to include the conditioning variable.
        targ_season (str): Target season ('F' or 'S').
        ice (dict): Dictionary with ice data.
        cond_var (np.ndarray): Conditioning variable timeseries.
        index_files (list): List of CSV file paths for index data.
        k (int): Parameter computed from time series length.
    """
    # Determine predictand based on predictor choice.
    predictand = [pred for pred in PREDICTORS if pred != predictor][0]
    ice_var = ice[ice_name]
    path_suffix = '_CC' if conditioning else ''
    cond_var_ = cond_var if conditioning else None
    targ_idx = 25 if targ_season == 'F' else 20

    for filepath in index_files:
        index_name = extract_index_name(filepath)
        if predictor == 'ice':
            predictor_name = ice_name
            predictand_name = index_name
            path = f'teleconnections_results/{ice_name}-tele_SEASON{path_suffix}'
            savename = f'{ice_name}_{PRED_SEASON}.{index_name}_{targ_season}'
        else:
            predictor_name = index_name
            predictand_name = ice_name
            path = f'teleconnections_results/tele-{ice_name}_SEASON{path_suffix}'
            savename = f'{index_name}_{PRED_SEASON}.{ice_name}_{targ_season}'
        
        print(f'Now processing: {predictor_name} > {predictand_name} (season: {targ_season}, cond: {conditioning})')

        # Build the index DataFrame from the timeseries.
        df_index = pd.DataFrame({index_name: index_timeseries(index_name)})

        # Precompute slice indices based on the DataFrame length.
        num_cycles = (len(df_index) // 12) - 2  # sacrificing 2 years
        slice_indices = [np.array([9, 10, 11, 12, 13, targ_idx]) + i * 12 for i in range(num_cycles)]

        lat_dim, lon_dim = ice_var.shape[1], ice_var.shape[2]

        # Compute TE for each grid point in parallel.
        results = Parallel(n_jobs=-1)(
            delayed(process_grid_point)(
                lat, lon, df_index, ice_var, ice_name, slice_indices,
                index_name, predictor, conditioning, cond_var_, k
            )
            for lat in range(lat_dim)
            for lon in range(lon_dim)
        )
        te_results = np.array(results).reshape(lat_dim, lon_dim)

        # Plotting the results.
        fig, ax = plt.subplots(figsize=(20, 10))
        img = ax.imshow(te_results, cmap='Blues_r')
        cbar_ax = fig.add_axes([0.72, 0.15, 0.03, 0.7])
        fig.colorbar(img, cax=cbar_ax)
        ax.axis('off')

        # Save results.
        #np.save(f'{path}/arrays/{savename}.npy', te_results)
        #fig.savefig(f'{path}/plots/{savename}.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    # Load and normalize ice data.
    ice = load_and_normalize_ice_data()
    # Compute parameter k from SIT data.
    k = ice['sit'].shape[0] // (10 * CYCLE_LEN)

    # Filter index files (only include those with 'nino' and without 'orig').
    index_files = glob.glob('data/*.csv')
    #index_files = [f for f in index_files if 'nino' in f and 'orig' not in f]
    index_files = [f for f in index_files if 'orig' not in f and 'GISTEMP' not in f]
    
    # Load the conditioning variable.
    cond_var = index_timeseries(COND_NAME)

    # Run experiments for each combination of settings.
    experiments = product(PREDICTORS, ICE_NAMES, CONDITIONING, TARG_SEASON)
    for predictor, ice_name, conditioning, targ_season in experiments:
        run_experiment(predictor, ice_name, conditioning, targ_season, ice, cond_var, index_files, k)


if __name__ == '__main__':
    main()