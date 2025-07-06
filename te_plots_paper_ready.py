import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed
from itertools import product
from matplotlib.colorbar import ColorbarBase
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import get_cmap
from scipy.stats import entropy
from scipy.signal import detrend

INDEX = 'nino4'
SIC_TELE_PATH = 'teleconnections_results/sic-tele_SEASON_CC/arrays/'
TELE_SIC_PATH = 'teleconnections_results/tele-sic_SEASON_CC/arrays/'
SAVE_PATH = 'teleconnections_results/paper_ready'

te_results = []
te_max = []
savenames = []
for path in ['sic_tele', 'tele_sic']:
    for season in ['S', 'F']:
        if path == 'sic_tele':
            total_path = SIC_TELE_PATH + f'sic_ONDJF.{INDEX}_{season}.npy'
            savenames.append(f'sic_{INDEX}_{season}')
        else:
            total_path = TELE_SIC_PATH + f'{INDEX}_ONDJF.sic_{season}.npy'
            savenames.append(f'{INDEX}_sic_{season}')
        te_results_ = np.load(total_path)
        te_results.append(te_results_)
        te_max.append(np.nanmax(te_results_))
te_max = np.max(te_max)

for results, savename in zip(te_results, savenames):
    # Plotting the results.
    fig, ax = plt.subplots(figsize=(20, 10))
    img = ax.imshow(results, cmap='Blues_r', vmax=te_max, origin='lower')
    cbar_ax = fig.add_axes([0.72, 0.15, 0.03, 0.7])
    fig.colorbar(img, cax=cbar_ax)
    ax.axis('off')
    
    # Save results.
    #fig.savefig(f'{SAVE_PATH}/{savename}.png', dpi=300, bbox_inches='tight')
    plt.show()