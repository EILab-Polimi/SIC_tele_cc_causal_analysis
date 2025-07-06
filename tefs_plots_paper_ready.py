import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from collections import Counter

ICE_VAR = 'sic'
TARG_SEASON = 'F'
savename = f'tefs_results/tefs_results_nocc_{ICE_VAR}_{TARG_SEASON}.jpg'

# -------------------------------
# 1. Load the saved feature selection data
# -------------------------------
output_dir = f'tefs_results/{ICE_VAR}_{TARG_SEASON}_nocc'
npz_file = os.path.join(output_dir, 'feature_selections.npz')
data = np.load(npz_file, allow_pickle=True)
int_array = data['int_array']
id_to_combo_keys = data['id_to_combo_keys']
id_to_combo_values = data['id_to_combo_values']

# -------------------------------
# 2. Modify combinations according to the GISTEMP rule
# -------------------------------
def modify_combo(combo):
    """
    Apply the GISTEMP rule:
      - If the combo contains ONLY "GISTEMP", return an empty tuple (i.e. no teleconnection indices)
      - If the combo contains "GISTEMP" along with others, remove "GISTEMP"
    """
    combo = tuple(combo)
    if "GISTEMP_monthly_temp" in combo:
        if len(combo) == 1:
            return tuple()  # recode as "No indices"
        else:
            return tuple(sorted(x for x in combo if x != "GISTEMP_monthly_temp"))
    return combo

# Map original integer IDs to modified combos
id_to_mod_combo = {}
for key, combo in zip(id_to_combo_keys, id_to_combo_values):
    id_to_mod_combo[key] = modify_combo(combo)

# -------------------------------
# 3. Build an array of modified combos (per pixel)
# -------------------------------
height, width = int_array.shape
mod_combo_array = np.empty((height, width), dtype=object)
for i in range(height):
    for j in range(width):
        code = int_array[i, j]
        if code == -1:
            mod_combo_array[i, j] = None  # pixel with NaN data
        else:
            mod_combo_array[i, j] = id_to_mod_combo.get(code)

# -------------------------------
# 4. Convert combos to final labels ("No indices", top-10 combos, "Other")
# -------------------------------
# First, count the frequency of each modified combo (excluding NaN).
from collections import Counter
combo_counter = Counter()
for i in range(height):
    for j in range(width):
        if mod_combo_array[i, j] is not None:
            combo_tuple = mod_combo_array[i, j]
            label = "No indices" if combo_tuple == tuple() else ", ".join(combo_tuple)
            combo_counter[label] += 1

# Exclude "No indices" from top-10 ranking
total_valid = sum(combo_counter.values()) - combo_counter["No indices"]
non_no_indices = {lab: cnt for lab, cnt in combo_counter.items() if lab != "No indices"}

# Top 10 combos (if fewer than 10, you'll just get them all)
top_10 = sorted(non_no_indices.items(), key=lambda x: x[1], reverse=True)[:10]
top_10_combos = [lab for lab, cnt in top_10]

# -------------------------------
# 5. Build a classification array (num_array) for plotting
# -------------------------------
# We define an ordered list of categories for the final plot:
#   1. "No indices"
#   2. each top-10 combo
#   3. "Other combinations" for everything else
ordered_categories = []
if "No indices" in combo_counter:
    ordered_categories.append("No indices")
ordered_categories.extend(top_10_combos)
ordered_categories.append("Other combinations")

# Map each category to an integer
label_to_int = {cat: i for i, cat in enumerate(ordered_categories)}

# Create an integer-coded array with these final labels
plot_array = np.full((height, width), -1, dtype=int)  # -1 will remain NaN
for i in range(height):
    for j in range(width):
        if mod_combo_array[i, j] is not None:
            combo_tuple = mod_combo_array[i, j]
            label = "No indices" if combo_tuple == tuple() else ", ".join(combo_tuple)
            if label not in ordered_categories[:-1]:  # i.e. not "No indices" or in top-10
                label = "Other combinations"
            plot_array[i, j] = label_to_int[label]

# -------------------------------
# 6. Choose distinct colors
# -------------------------------
# We'll pick a color for "No indices", a color for "Other combinations", and
# unique colors for the top-10 combos from a qualitative colormap.
color_mapping = {}
cmap = plt.get_cmap('tab20c')

idx = 0
if "No indices" in ordered_categories:
    color_mapping["No indices"] = '#DEDEDE'  # gray
    idx += 1

for combo in top_10_combos:
    color_mapping[combo] = cmap(idx % 10)
    idx += 1

# Pick a distinctly different color for "Other combinations"
color_mapping["Other combinations"] = '#00008B'  # bright magenta (for example)

# -------------------------------
# 7. Build the ListedColormap and BoundaryNorm
# -------------------------------
color_list = [color_mapping[cat] for cat in ordered_categories]
cmap_custom = ListedColormap(color_list)
norm = BoundaryNorm(np.arange(-0.5, len(ordered_categories)+0.5, 1), len(ordered_categories))

# -------------------------------
# 8. Recount the final classification for the legend
# -------------------------------
final_counter = Counter(plot_array.flatten())
# -1 is the masked (NaN) class, so skip it
valid_pixels = sum(count for code, count in final_counter.items() if code != -1)

# -------------------------------
# 9. Plot
# -------------------------------
fig, ax = plt.subplots(figsize=(20, 12))

masked_array = np.ma.masked_where(plot_array == -1, plot_array)
im = plt.imshow(masked_array, cmap=cmap_custom, norm=norm)
#plt.title("Teleconnection Feature Selection Map")
plt.axis('off')

# -------------------------------
# 10. Build legend with correct percentages
# -------------------------------
legend_handles = []
for i, cat in enumerate(ordered_categories):
    count_cat = final_counter[i]  # how many pixels are labeled with i
    pct_cat = (count_cat / valid_pixels * 100) if valid_pixels > 0 else 0
    color = color_mapping[cat]
    # clean up labels for paper-ready version
    if cat == 'amon':
        cat = 'AMO'
    elif cat == 'nino12':
        cat = 'Niño1+2'
    elif cat == 'nino34':
        cat = 'Niño3.4'
    elif cat == 'nino3':
        cat = 'Niño3'
    elif cat == 'nino4':
        cat = 'Niño4'
    elif cat == 'amon, nino12':
        cat = 'AMO, Niño1+2'
    elif cat == 'nino12, pdo':
        cat = 'Niño1+2, PDO'
    elif cat in ['pdo', 'nao', 'ao', 'soi', 'tpi']:
        cat = cat.upper()
    legend_handles.append(
        mpatches.Patch(color=color, label=f"{cat}: {pct_cat:.2f}%")
    )

# Add a patch for truly NaN (plot_array == -1)
legend_handles.append(mpatches.Patch(color='white', label="NaN"))

#plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=24)
plt.legend(
    handles=legend_handles, 
    bbox_to_anchor=(0.5, -0.1),  # for example, place legend below the plot
    loc='upper center', 
    ncol=3,  # 2 or 3 columns, depending on how many categories you have
    fontsize=18
)

plt.tight_layout()
plt.show()

#fig.savefig(savename, dpi=300, bbox_inches='tight')