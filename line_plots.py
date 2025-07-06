import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
import matplotlib.ticker as ticker

ICE_VAR = 'sic'
METRIC = 'sum'

indexes = [
            'amon',
            'ao',
            'nao',
            'nino12',
            'nino3',
            'nino34',
            'nino4',
            'pdo',
            'soi',
            'tpi',
        ]

indexes_paper = [
            'AMO',
            'AO',
            'NAO',
            'Niño1+2',
            'Niño3',
            'Niño3.4',
            'Nino4',
            'PDO',
            'SOI',
            'TPI',
        ]

ice_tele_dir = f'teleconnections_results/{ICE_VAR}-tele_SEASON/arrays/'
tele_ice_dir = f'teleconnections_results/tele-{ICE_VAR}_SEASON/arrays/'
ice_tele_no_cc_dir = f'teleconnections_results/{ICE_VAR}-tele_SEASON_CC/arrays/'
tele_ice_no_cc_dir = f'teleconnections_results/tele-{ICE_VAR}_SEASON_CC/arrays/'
savename = f'teleconnections_results/lineplots_{ICE_VAR}_{METRIC}.png'
        
        
# Create dataframes for both datasets
df = pd.DataFrame()
df_no_cc = pd.DataFrame()
        
for index in indexes:
    for targ_season in ['S', 'F']:
        # Original data (with climate change effects)
        x1 = np.load(ice_tele_dir + f'{ICE_VAR}_ONDJF.{index}_{targ_season}.npy').flatten()
        x1 = x1[~np.isnan(x1)]
        x2 = np.load(tele_ice_dir + f'{index}_ONDJF.{ICE_VAR}_{targ_season}.npy').flatten()
        x2 = x2[~np.isnan(x2)]
                
        # No climate change data
        x1_no_cc = np.load(ice_tele_no_cc_dir + f'{ICE_VAR}_ONDJF.{index}_{targ_season}.npy').flatten()
        x1_no_cc = x1_no_cc[~np.isnan(x1_no_cc)]
        x2_no_cc = np.load(tele_ice_no_cc_dir + f'{index}_ONDJF.{ICE_VAR}_{targ_season}.npy').flatten()
        x2_no_cc = x2_no_cc[~np.isnan(x2_no_cc)]
                
        if METRIC == 'sum':
            df.loc[targ_season, f'{ICE_VAR}>{index}'] = np.sum(x1)
            df.loc[targ_season, f'{index}>{ICE_VAR}'] = np.sum(x2)
            df_no_cc.loc[targ_season, f'{ICE_VAR}>{index}'] = np.sum(x1_no_cc)
            df_no_cc.loc[targ_season, f'{index}>{ICE_VAR}'] = np.sum(x2_no_cc)
        elif METRIC == 'max':
            df.loc[targ_season, f'{ICE_VAR}>{index}'] = np.max(x1)
            df.loc[targ_season, f'{index}>{ICE_VAR}'] = np.max(x2)
            df_no_cc.loc[targ_season, f'{ICE_VAR}>{index}'] = np.max(x1_no_cc)
            df_no_cc.loc[targ_season, f'{index}>{ICE_VAR}'] = np.max(x2_no_cc)
        elif METRIC == 'q99':
            df.loc[targ_season, f'{ICE_VAR}>{index}'] = np.quantile(x1, 0.99)
            df.loc[targ_season, f'{index}>{ICE_VAR}'] = np.quantile(x2, 0.99)
            df_no_cc.loc[targ_season, f'{ICE_VAR}>{index}'] = np.quantile(x1_no_cc, 0.99)
            df_no_cc.loc[targ_season, f'{index}>{ICE_VAR}'] = np.quantile(x2_no_cc, 0.99)
        
def create_positive_colormap():
    """Create a colormap for positive values (reds)"""
    colors = ['#f7f7f7', '#fddbc7', '#f4a582', '#d6604d', '#b2182b', '#67001f']
    return LinearSegmentedColormap.from_list("positive", colors, N=128)
        
def create_negative_colormap():
    """Create a colormap for negative values (blues)"""
    colors = ['#053061', '#2166ac', '#4393c3', '#92c5de', '#d1e5f0', '#f7f7f7']
    return LinearSegmentedColormap.from_list("negative", colors, N=128)
        
def plot_gradient_bar(ax, x, height, cmap, norm, width):
    """Plot filled gradient bars for climate change data"""
    num_sections = 30  # Fewer sections for a smoother gradient
    for i in range(num_sections):
        y_bottom = i * height / num_sections
        y_top = (i + 1) * height / num_sections
        color = cmap(norm(y_top))
        ax.bar(x, y_top - y_bottom, bottom=y_bottom, width=width, color=color, edgecolor=None)
        
def plot_outline_bar(ax, x, height, width, color='black', filled=False, fill_color=None):
    """Plot the outline of a bar with visible dashing"""
    if filled and fill_color:
        # Add a semi-transparent fill - fixed Rectangle parameters
        rect = Rectangle((x - width/2, 0), width, height, facecolor=fill_color, alpha=0.15)
        ax.add_patch(rect)
            
    # Plot the rectangle outline with more visible dashes  
    # Top line
    ax.plot([x - width/2, x + width/2], [height, height], color=color, 
            linestyle='--', linewidth=1, dashes=(4, 3))
            
    # Left line
    ax.plot([x - width/2, x - width/2], [0, height], color=color, 
            linestyle='--', linewidth=1, dashes=(4, 3))
            
    # Right line
    ax.plot([x + width/2, x + width/2], [0, height], color=color, 
            linestyle='--', linewidth=1, dashes=(4, 3))
        
def create_plot(df, df_no_cc):
    indices = [col.split('>')[1] for col in df.columns if '>' in col][::2]
    lead_times = df.index
    x = np.arange(len(indices))
        
    fig, ax = plt.subplots(figsize=(20, 12))
    # Define maximum value for scale
    if METRIC == 'sum':
        max_value = 800
        max_cc_value = 100
    elif METRIC == 'q99':
        max_value = 0.12
        max_cc_value = 0.03
            
    # Create separate color maps for positive and negative values
    pos_cmap = create_positive_colormap()
    neg_cmap = create_negative_colormap()
            
    # Find the max absolute value in df_no_cc for better normalization    
    # Use a separate normalization just for the CC data
    # This ensures the colors will be more visible
    pos_norm = plt.Normalize(0, max_cc_value)
    neg_norm = plt.Normalize(-max_cc_value, 0)
            
    # Fixed colors for outlines
    pos_outline_color = '#8B0000'  # Dark red
    neg_outline_color = '#00008B'  # Dark blue
            
    # Used for shading the outlines
    pos_fill_color = '#F6BDC0'  # Light red
    neg_fill_color = '#AFDCEB'  # Light blue
        
    bar_width = 0.15
    offset = bar_width * (len(lead_times) - 1) / 2
            
    for i, lead_time in enumerate(lead_times):
        # Get values for both datasets
        index_to_sit = [df.loc[lead_time, f'{index}>{ICE_VAR}'] for index in indices]
        sit_to_index = [df.loc[lead_time, f'{ICE_VAR}>{index}'] for index in indices]
                
        index_to_sit_no_cc = [df_no_cc.loc[lead_time, f'{index}>{ICE_VAR}'] for index in indices]
        sit_to_index_no_cc = [df_no_cc.loc[lead_time, f'{ICE_VAR}>{index}'] for index in indices]
                
        # Plot bars for each index
        for j in range(len(indices)):
            x_pos = x[j] - offset + i*bar_width
                    
            # Plot climate change data (filled gradient bars)
            if index_to_sit_no_cc[j] > 0:
                plot_gradient_bar(ax, x_pos, index_to_sit_no_cc[j], pos_cmap, pos_norm, bar_width)
                    
            if sit_to_index_no_cc[j] > 0:
                plot_gradient_bar(ax, x_pos, -sit_to_index_no_cc[j], neg_cmap, neg_norm, bar_width)
                    
            # Plot no climate change data as outlined bars
            if index_to_sit[j] > 0:
                plot_outline_bar(ax, x_pos, index_to_sit[j], bar_width, 
                                    color=pos_outline_color, filled=True, fill_color=pos_fill_color)
                    
            if sit_to_index[j] > 0:
                plot_outline_bar(ax, x_pos, -sit_to_index[j], bar_width, 
                                    color=neg_outline_color, filled=True, fill_color=neg_fill_color)
        
    # Create custom legend with fixed entries
    legend_elements = [
        Rectangle((0, 0), 1, 1, facecolor='#CD5352', 
                    label='CC signal removed (filled-in bars)'),
        Rectangle((0, 0), 1, 1, fill=False, linestyle='--', edgecolor=pos_outline_color, 
                    linewidth=1.5, label='CC signal not removed (dashed-outline bars)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=26, facecolor='#FAFAFA')
            
    # Add title and labels
    #ax.set_title(f'Transfer Entropy between {ICE_VAR.upper()} and Climate Indices\nComparison with and without Climate Change effects', fontsize=24)
    ax.set_xticks(x)
    ax.set_xticklabels(indexes_paper, fontsize=26, rotation=30, ha='center')
            
    # Set up y-axis limits and grid lines
    ax.set_ylim(-max_value, max_value)  # Changed from -max_value * 1.3, max_value * 1.3
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.8)
    ax.axhline(y=-max_value, color='gray', linestyle='-', linewidth=0.3, alpha=0.8)
    ax.axhline(y=-max_value*3/4, color='gray', linestyle='-', linewidth=0.3, alpha=0.8)
    ax.axhline(y=-max_value*2/4, color='gray', linestyle='-', linewidth=0.3, alpha=0.8)
    ax.axhline(y=-max_value*1/4, color='gray', linestyle='-', linewidth=0.3, alpha=0.8)
    ax.axhline(y=max_value*1/4, color='gray', linestyle='-', linewidth=0.3, alpha=0.8)
    ax.axhline(y=max_value*2/4, color='gray', linestyle='-', linewidth=0.3, alpha=0.8)
    ax.axhline(y=max_value*3/4, color='gray', linestyle='-', linewidth=0.3, alpha=0.8)
    ax.axhline(y=max_value, color='gray', linestyle='-', linewidth=0.3, alpha=0.8)  # Added this line
    
    y_levels = [
        -max_value,
        -max_value*3/4,
        -max_value*2/4,
        -max_value*1/4,
         0,
         max_value*1/4,
         max_value*2/4,
         max_value*3/4,
         max_value,  # Added this line
    ]
    
    # choose a formatting function based on your METRIC
    fmt = "{:.0f}".format if METRIC == 'sum' else "{:.2f}".format

    for y in y_levels:
        # x=1.01 in axis‐fraction coords, y in data coords
        ax.text(
            1.01, y, fmt(abs(y)),
            transform=ax.get_yaxis_transform(),
            va='center', ha='left',
            fontsize=22,
            color='gray',
            clip_on=False
        )
    
    # Remove y-axis ticks and some spines
    ax.yaxis.set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
        
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave room for the colorbars
    fig.savefig(savename, dpi=300, bbox_inches='tight')
    plt.show()
        
# Run the plot creation
create_plot(df, df_no_cc)