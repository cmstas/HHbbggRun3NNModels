import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec  # for layout with separate table axis
import pandas as pd
import numpy as np
import os
import re
import pyarrow.parquet as pq
import pyarrow.dataset as ds
#import mplhep
#plt.style.use(mplhep.style.CMS)  # Turned off the CMS style
plt.minorticks_on()
def plot_fatjet_properties_stacked(df, pt_ranges=[(250, 400), (400, 600), (600, 1000)]):
    out_dir = "plots_stacked"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Get background processes (skip "Data").
    processes = [proc for proc in df['proc'].unique() if proc != 'Data']
    print(f"Stacking plots for processes: {processes}")
    # Separate out the background processes (exclude signal 'GluGluToHH') for stacking.
    background_procs = [proc for proc in processes if proc != 'GluGluToHH']
    
    # Set up a simple color map.
    colors = plt.cm.tab10.colors
    color_map = {proc: colors[i % len(colors)] for i, proc in enumerate(processes)}
    
    # Get all columns corresponding to the selected fatjet properties.
    plot_cols = [
        c for c in df.columns
        if (c.startswith("fatjet_selected_") and "gen" not in c)
        or c.startswith("delta")
    ]

    # 2) If 'fatjet_selected_pt' is present, put it at the front.
    if "fatjet_selected_pt" in plot_cols:
        plot_cols.remove("fatjet_selected_pt")
        plot_cols.insert(0, "fatjet_selected_pt")

    fatjet_cols = plot_cols
    
    # Loop over each variable.
    for col in fatjet_cols:
        # Determine if the variable is a probability variable.
        global_max = df[col].max() if not df[col].empty else 0
        is_probability_var = (0.9 <= global_max <= 1.1)
        
        if is_probability_var:
            bins = np.linspace(0, 1, 51)
        else:
            global_min = df[col].min()
            global_max = df[col].max()
            bins = np.linspace(global_min, global_max, 51)
        
        # Prepare data for stacking from each background process (all pT).
        hist_data = []
        hist_weights = []
        labels = []
        colors_list = []
        weighted_sums = []  # store weighted sum for each sample
        
        for proc in background_procs:
            proc_df = df[df['proc'] == proc]
            if proc_df.empty:
                continue
            
            if is_probability_var:
                valid_mask = proc_df[col].between(0, 1) & (proc_df[col] > -100)
                proc_data = proc_df.loc[valid_mask, col]
                proc_weights = proc_df.loc[valid_mask, 'weight']
            else:
                proc_data = proc_df[col]
                proc_weights = proc_df['weight']
                mask = proc_data > -100
                proc_data = proc_data[mask]
                proc_weights = proc_weights[mask]
            
            if proc_data.empty:
                continue
            
            ws = proc_weights.sum()  # compute weighted sum for this sample
            weighted_sums.append(ws)
            
            hist_data.append(proc_data)
            hist_weights.append(proc_weights)
            labels.append(proc)
            colors_list.append(color_map[proc])
        
        if len(hist_data) == 0:
            continue
        # ***** CHANGED LINES: Sort background samples by weighted sum (ascending order) for histogram stacking.
        sort_indices = np.argsort(weighted_sums)
        hist_data = [hist_data[i] for i in sort_indices]
        hist_weights = [hist_weights[i] for i in sort_indices]
        labels_sorted = [labels[i] for i in sort_indices]
        colors_list = [colors_list[i] for i in sort_indices]
        weighted_sums_sorted = [weighted_sums[i] for i in sort_indices]
        
        # Plot the stacked histogram on ax.
        fig = plt.figure(figsize=(10, 6))
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
        ax = fig.add_subplot(gs[0])
        ax.hist(hist_data, bins=bins, weights=hist_weights,
                stacked=True, label=labels_sorted, color=colors_list,  # CHANGED: use sorted labels
                histtype='stepfilled', linewidth=2)
        
        # Overlay signal sample.
        signal_weight = None
        if 'GluGluToHH' in df['proc'].unique():
            sig_df = df[df['proc'] == 'GluGluToHH']
            if not sig_df.empty:
                if is_probability_var:
                    valid_mask = sig_df[col].between(0, 1) & (sig_df[col] > -100)
                    sig_data = sig_df.loc[valid_mask, col]
                    sig_weights = sig_df.loc[valid_mask, 'weight']
                else:
                    sig_data = sig_df[col]
                    sig_weights = sig_df['weight']
                    mask = sig_data > -100
                    sig_data = sig_data[mask]
                    sig_weights = sig_weights[mask]
                if not sig_data.empty:
                    ax.hist(sig_data, bins=bins, weights=sig_weights,
                            histtype='step', linewidth=2, label='GluGluToHH', color='blue')
                    signal_weight = sig_weights.sum()
        
        ax.set_yscale('log')
        ax.legend(ncol=3, fontsize=10, labelspacing=0.2, columnspacing=0.5, handletextpad=0.3, frameon=False)
        ax.set_xlabel(f'FatJet Selected {col.replace("fatjet_selected_", "").replace("particleNet", "pNet")}', fontsize=16, loc="right")
        ax.set_ylabel('Weighted Events', fontsize=16, loc="top")
        ax.tick_params(axis='both', which='major', direction='in', length=8, width=1,
                       labelsize=16, top=True, right=True)
        ax.tick_params(axis='both', which='minor', direction='in', length=4, width=1,
                       top=True, right=True)
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10)/10))        
        # ***** CHANGED LINES: Reverse sorted background samples for table ordering.
        table_data = [[f"{ws:.2f}"] for ws in reversed(weighted_sums_sorted)]
        row_labels = list(reversed(labels_sorted))
        if signal_weight is not None:
            row_labels.append('GluGluToHH')
            # For the signal sample, show three decimal places instead of 2.
            table_data.append([f"{signal_weight:.3f}"])
        table_ax = fig.add_subplot(gs[1])
        table_ax.axis('off')
        table_ax.table(cellText=table_data, rowLabels=row_labels, colLabels=["Weighted Sum"],
                       loc='center', cellLoc='center')
        
        fig.tight_layout()
        filename = os.path.join(out_dir, f"fatjet_selected_{col.replace('fatjet_selected_','').replace('particleNet','pNet')}_all_pt.png")
        fig.savefig(filename)
        plt.close(fig)
        
        # Now produce stacked plots for each specified pT range.
        for pt_min, pt_max in pt_ranges:
            # Prepare data for this pT range.
            hist_data = []
            hist_weights = []
            labels = []
            colors_list = []
            weighted_sums = []
            
            for proc in background_procs:
                proc_df = df[(df['proc'] == proc) & 
                             (df["fatjet_selected_pt"] >= pt_min) & 
                             (df["fatjet_selected_pt"] < pt_max)]
                if proc_df.empty:
                    continue
                
                if is_probability_var:
                    valid_mask = proc_df[col].between(0, 1) & (proc_df[col] > -100)
                    proc_data = proc_df.loc[valid_mask, col]
                    proc_weights = proc_df.loc[valid_mask, 'weight']
                else:
                    proc_data = proc_df[col]
                    proc_weights = proc_df['weight']
                    mask = proc_data > -100
                    proc_data = proc_data[mask]
                    proc_weights = proc_weights[mask]
                
                if proc_data.empty:
                    continue
                
                ws = proc_weights.sum()
                weighted_sums.append(ws)
                
                hist_data.append(proc_data)
                hist_weights.append(proc_weights)
                labels.append(proc)
                colors_list.append(color_map[proc])
            
            if len(hist_data) == 0:
                continue
            
            # ***** CHANGED LINES: Sort background samples by weighted sum (ascending order) for histogram stacking.
            sort_indices = np.argsort(weighted_sums)
            hist_data = [hist_data[i] for i in sort_indices]
            hist_weights = [hist_weights[i] for i in sort_indices]
            labels_sorted = [labels[i] for i in sort_indices]
            colors_list = [colors_list[i] for i in sort_indices]
            weighted_sums_sorted = [weighted_sums[i] for i in sort_indices]
            # Compute dynamic bins for this pT range.
            if is_probability_var:
                local_bins = np.linspace(0, 1, 51)
            else:
                combined_series = pd.concat(hist_data)
                if 'GluGluToHH' in df['proc'].unique():
                    sig_df = df[(df['proc'] == 'GluGluToHH') & 
                                (df["fatjet_selected_pt"] >= pt_min) & 
                                (df["fatjet_selected_pt"] < pt_max)]
                    if not sig_df.empty:
                        if is_probability_var:
                            valid_mask = sig_df[col].between(0, 1) & (sig_df[col] > -100)
                            sig_data = sig_df.loc[valid_mask, col]
                        else:
                            sig_data = sig_df[col]
                            mask = sig_data > -100
                            sig_data = sig_data[mask]
                        if not sig_data.empty:
                            combined_series = pd.concat([combined_series, sig_data])
                local_min = combined_series.min()
                local_max = combined_series.max()
                if local_min == local_max:
                    local_bins = np.linspace(local_min - 1, local_max + 1, 51)
                else:
                    local_bins = np.linspace(local_min, local_max, 51)
            
            # Create figure with two subplots for this pT range.
            fig = plt.figure(figsize=(10, 6))
            gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
            ax = fig.add_subplot(gs[0])
            ax.hist(hist_data, bins=local_bins, weights=hist_weights,
                    stacked=True, label=labels_sorted, color=colors_list,  # CHANGED: use sorted labels
                    histtype='stepfilled', linewidth=2)
            
            signal_weight = None
            if 'GluGluToHH' in df['proc'].unique():
                sig_df = df[(df['proc'] == 'GluGluToHH') & 
                            (df["fatjet_selected_pt"] >= pt_min) & 
                            (df["fatjet_selected_pt"] < pt_max)]
                if not sig_df.empty:
                    if is_probability_var:
                        valid_mask = sig_df[col].between(0, 1) & (sig_df[col] > -100)
                        sig_data = sig_df.loc[valid_mask, col]
                        sig_weights = sig_df.loc[valid_mask, 'weight']
                    else:
                        sig_data = sig_df[col]
                        sig_weights = sig_df['weight']
                        mask = sig_data > -100
                        sig_data = sig_data[mask]
                        sig_weights = sig_weights[mask]
                    if not sig_data.empty:
                        ax.hist(sig_data, bins=local_bins, weights=sig_weights,
                                histtype='step', linewidth=2, label='GluGluToHH', color='blue')
                        signal_weight = sig_weights.sum()
            
            ax.set_yscale('log')
            ax.legend(ncol=3, fontsize=10, labelspacing=0.2, columnspacing=0.5, handletextpad=0.3, frameon=False)
            ax.set_xlabel(f'FatJet Selected {col.replace("fatjet_selected_", "").replace("particleNet", "pNet")} [GeV]', fontsize=16, loc="right")
            ax.set_ylabel('Weighted Events', fontsize=16, loc="top")
            ax.tick_params(axis='both', which='major', direction='in', length=8, width=1,
                           labelsize=16, top=True, right=True)
            ax.tick_params(axis='both', which='minor', direction='in', length=4, width=1,
                           top=True, right=True)
            ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10)/10))
            ax.set_title(f'(pT: {pt_min}-{pt_max} GeV)', fontsize=16)
            
            # ***** CHANGED LINES: Reverse sorted background samples for table ordering.
            table_data = [[f"{ws:.2f}"] for ws in reversed(weighted_sums_sorted)]
            row_labels = list(reversed(labels_sorted))
            if signal_weight is not None:
                row_labels.append('GluGluToHH')
                table_data.append([f"{signal_weight:.3f}"])
            table_ax = fig.add_subplot(gs[1])
            table_ax.axis('off')
            table_ax.table(cellText=table_data, rowLabels=row_labels, colLabels=["Weighted Sum"],
                           loc='center', cellLoc='center')
            
            fig.tight_layout()
            filename = os.path.join(out_dir, f"fatjet_selected_{col.replace('fatjet_selected_','').replace('particleNet','pNet')}_pt_{pt_min}_{pt_max}.png")
            fig.savefig(filename)
            plt.close(fig)

# ----------------------
# Main part of the file
# ----------------------

# File path for the merged parquet file.
#file_path = '/home/users/xuyan/HH2ggbb/For_Brunella/Results_WP90_noDijetMass_PNetJets_DijetReg_onlyNonRes_eval.parquet'
file_path = 'Results_noDijetMass_PNetJets_DijetReg_onlyNonRes_eval_kl_boosted_addSingleH_mbbtrimmed_withFatjets.parquet'

# Open the parquet file to extract column names.
parquet_file = pq.ParquetFile(file_path)
all_columns = parquet_file.schema.names

#print(all_columns)

# Filter for columns related to fatjets.
fatjet_columns = [col for col in all_columns if 'fatjet' in col]
Res_columns = [col for col in all_columns if 'Res' in col]
# Essential columns needed for analysis and plotting.
essential_columns = [
    'proc',
    'weight',
    'mass',  
    'pt',        # gg system pt
    'eta',       # gg system eta
    'phi',       # gg system phi
    'lead_eta',  # lead photon eta
    'lead_phi',  # lead photon phi
    'sublead_eta',  # sublead photon eta
    'sublead_phi',  # sublead photon phi
    'n_fatjets',
    'n_leptons',
    'Res_CosThetaStar_gg',
    'Res_pholead_PtOverM',
    'Res_phosublead_PtOverM',
    "lead_hoe",
    "sublead_hoe",
    #'lead_photon_EnergyErrOverE',
    #"sublead_photon_EnergyErrOverE",
    #"MET_PNet_all",
    "sigma_m_over_m_smeared_decorr",
    'puppiMET_phi',
    "puppiMET_sumEt",
    'n_jets',
    'event', 'lumi', 'run', 'jet1_genMatched_Hbb', 'jet2_genMatched_Hbb','Res_mjj_regressed','lead_isScEtaEB','sublead_isScEtaEB','lead_mvaID','sublead_mvaID','lead_isScEtaEE','sublead_isScEtaEE'
]

# Combine fatjet columns with essential columns.
columns_to_load = list(set(fatjet_columns + essential_columns+ Res_columns))
print(f"Loading {len(columns_to_load)} columns out of {len(all_columns)} total columns")
dataset = ds.dataset(file_path, format="parquet")

table = dataset.to_table(filter=(ds.field("n_fatjets") != 0), columns=columns_to_load) 
n_total_rows = table.num_rows
print(f"Total rows (no filter): {n_total_rows}")

# 🔹 Carica solo le righe con n_fatjets != 0
#table_filtered = dataset.to_table(filter=(ds.field("n_fatjets") != 0), columns=columns_to_load)
#n_filtered_rows = table_filtered.num_rows
#print(f"Filtered rows (n_fatjets != 0): {n_filtered_rows}")

# Load only the selected columns from the parquet file,
# filtering out rows where n_fatjets == 0.
#dataset = ds.dataset(file_path, format="parquet")
#table = table_all #dataset.to_table(filter=(ds.field("n_fatjets") != 0), columns=columns_to_load)
df = table.to_pandas()
lead_mvaID_condition = ((df["lead_isScEtaEB"] == True) & (df['lead_mvaID'] > 0.0439603)) | ((df["lead_isScEtaEE"] == True) & (df['lead_mvaID'] > -0.249526))
sublead_mvaID_condition = ((df["sublead_isScEtaEB"] == True) & (df['sublead_mvaID'] > 0.0439603)) | ((df["sublead_isScEtaEE"] == True) & (df['sublead_mvaID'] > -0.249526))
df = df[(lead_mvaID_condition) & (sublead_mvaID_condition)]

print(f"Successfully loaded filtered data with {len(df)} rows")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

print(df["proc"].unique())

print(list(df.columns))

# ----------------------
# Vectorized fatjet selection
# ----------------------

# Identify and sort all msoftdrop columns by fatjet index.
mass_cols = sorted(
    [col for col in df.columns if re.match(r'fatjet\d+_msoftdrop', col)],
    key=lambda x: int(re.search(r'fatjet(\d+)_msoftdrop', x).group(1))
)
print("finish sorting")
if len(mass_cols) == 0:
    raise ValueError("No fatjet mass columns found.")

# Create a NumPy array of msoftdrop values.
mass_arr = df[mass_cols].to_numpy()  # shape: (num_events, num_fatjets)
eligible = (mass_arr > 20)# & (mass_arr < 210)

# eligible = eligible & (((df['jet1_genMatched_Hbb'].values != 1) | (df['jet2_genMatched_Hbb'].values != 1))[:, None])

# Reject any fatjets with default subjet masses (-999).
subjet1_mass_cols = sorted(
    [col for col in df.columns if re.match(r'fatjet\d+_subjet1_mass', col)],
    key=lambda x: int(re.search(r'fatjet(\d+)_subjet1_mass', x).group(1))
)
subjet2_mass_cols = sorted(
    [col for col in df.columns if re.match(r'fatjet\d+_subjet2_mass', col)],
    key=lambda x: int(re.search(r'fatjet(\d+)_subjet2_mass', x).group(1))
)
subjet1_mass_arr = df[subjet1_mass_cols].to_numpy()  # shape: (num_events, num_fatjets)
subjet2_mass_arr = df[subjet2_mass_cols].to_numpy()  # shape: (num_events, num_fatjets)
eligible = eligible & (subjet1_mass_arr != -999) & (subjet2_mass_arr != -999)


subjet1_btagDeepB_cols = sorted(
    [col for col in df.columns if re.match(r'fatjet\d+_subjet1_btagDeepB', col)],
    key=lambda x: int(re.search(r'fatjet(\d+)_subjet1_btagDeepB', x).group(1))
)
subjet2_btagDeepB_cols = sorted(
    [col for col in df.columns if re.match(r'fatjet\d+_subjet2_btagDeepB', col)],
    key=lambda x: int(re.search(r'fatjet(\d+)_subjet2_btagDeepB', x).group(1))
)
subjet1_btagDeepB_arr = df[subjet1_btagDeepB_cols].to_numpy()  # shape: (num_events, num_fatjets)
subjet2_btagDeepB_arr = df[subjet2_btagDeepB_cols].to_numpy()  # shape: (num_events, num_fatjets)
eligible = eligible & (subjet2_btagDeepB_arr >= 0.2)

# Apply tau ratio cut: require fatjet tau2/tau1 < 0.75
tau1_cols = sorted(
    [col for col in df.columns if re.match(r'fatjet\d+_tau1', col)],
    key=lambda x: int(re.search(r'fatjet(\d+)_tau1', x).group(1))
)
tau2_cols = sorted(
    [col for col in df.columns if re.match(r'fatjet\d+_tau2', col)],
    key=lambda x: int(re.search(r'fatjet(\d+)_tau2', x).group(1))
)
tau1_arr = df[tau1_cols].to_numpy()  # shape: (num_events, num_fatjets)
tau2_arr = df[tau2_cols].to_numpy()  # shape: (num_events, num_fatjets)
# Avoid division by zero: set ratio to infinity where tau1==0.
tau_ratio_arr = np.where(tau1_arr == 0, np.inf, tau2_arr / tau1_arr)
eligible = eligible & (tau_ratio_arr < 0.75)

# Identify and sort all particleNet_XbbVsQCD columns by fatjet index.
particleNet_cols = sorted(
    [col for col in df.columns if re.match(r'fatjet\d+_particleNet_XbbVsQCD', col)],
    key=lambda x: int(re.search(r'fatjet(\d+)_particleNet_XbbVsQCD', x).group(1))
)
if len(particleNet_cols) == 0:
    raise ValueError("No fatjet particleNet_XbbVsQCD columns found.")

# Create a NumPy array for the particleNet scores.
particleNet_arr = df[particleNet_cols].to_numpy()  # shape: (num_events, num_fatjets)
eligible = eligible & (particleNet_arr > 0.4)

# Mask non-eligible jets by replacing their particleNet scores with -infinity.
masked_scores = np.where(eligible, particleNet_arr, -np.inf)

# For each event, select the fatjet index with the highest particleNet score among eligible jets.
best_idx = np.argmax(masked_scores, axis=1)

n_fatjets_final = eligible.sum(axis=1)
df['n_fatjets_final'] = n_fatjets_final

# Filter to only events with at least one eligible fatjet.
row_has = eligible.any(axis=1)
#df = df[row_has].copy()
#best_idx = best_idx[row_has]

# Build a dictionary mapping each fatjet property to its corresponding columns.
prop_dict = {}
pattern = re.compile(r'fatjet(\d+)_(.+)')
for col in df.columns:
    m = pattern.match(col)
    if m:
        jet_index = int(m.group(1))
        prop = m.group(2)
        prop_dict.setdefault(prop, []).append((jet_index, col))

# For each property, create a new 'fatjet_selected_{prop}' column by taking the value
# from the column corresponding to the best (highest particleNet score) eligible fatjet.
for prop, jets in prop_dict.items():
    jets_sorted = sorted(jets, key=lambda x: x[0])
    col_names = [col for (_, col) in jets_sorted]
    values = df[col_names].to_numpy()
    selected_vals = np.take_along_axis(values, best_idx[:, np.newaxis], axis=1).flatten()
    new_col = f'fatjet_selected_{prop}'
    df[new_col] = selected_vals

print(f"After msoftdrop cut and best particleNet selection, data has {len(df)} rows")

df['fatjet_selected_tau2tau1_ratio'] = df['fatjet_selected_tau2'] / df['fatjet_selected_tau1']

# Helper function to wrap angles to the [-pi, pi] range
def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

# Compute the corrected fatjet mass by multiplying the fatjet mass with its mass correction factor
df['fatjet_selected_mass_corrected'] = df['fatjet_selected_mass'] * df['fatjet_selected_particleNet_massCorr']

print("Define shorthand references")
# Define shorthand references (just for readability here):
g1_eta  = df['lead_eta']
g1_phi  = df['lead_phi']
g2_eta  = df['sublead_eta']
g2_phi  = df['sublead_phi']
gg_eta  = df['eta']  # entire diphoton system
gg_phi  = df['phi']  # entire diphoton system
fj_eta  = df['fatjet_selected_eta']
fj_phi  = df['fatjet_selected_phi']
subj1_eta = df['fatjet_selected_subjet1_eta']
subj1_phi = df['fatjet_selected_subjet1_phi']
subj2_eta = df['fatjet_selected_subjet2_eta']
subj2_phi = df['fatjet_selected_subjet2_phi']

# 1) deltaEta_g1_g2, deltaPhi_g1_g2
df['deltaEta_g1_g2'] = g1_eta - g2_eta
df['deltaPhi_g1_g2'] = wrap_angle(g1_phi - g2_phi)

# 2) deltaEta_gg_fj, deltaPhi_gg_fj, deltaR_gg_fj
df['deltaEta_gg_fj'] = gg_eta - fj_eta
df['deltaPhi_gg_fj'] = wrap_angle(gg_phi - fj_phi)
df['deltaR_gg_fj']   = np.sqrt(df['deltaEta_gg_fj']**2 + df['deltaPhi_gg_fj']**2)

# 3) deltaEta_g1_fj, deltaPhi_g1_fj, deltaR_g1_fj
df['deltaEta_g1_fj'] = g1_eta - fj_eta
df['deltaPhi_g1_fj'] = wrap_angle(g1_phi - fj_phi)
df['deltaR_g1_fj']   = np.sqrt(df['deltaEta_g1_fj']**2 + df['deltaPhi_g1_fj']**2)

# 4) deltaEta_g2_fj, deltaPhi_g2_fj, deltaR_g2_fj
df['deltaEta_g2_fj'] = g2_eta - fj_eta
df['deltaPhi_g2_fj'] = wrap_angle(g2_phi - fj_phi)
df['deltaR_g2_fj']   = np.sqrt(df['deltaEta_g2_fj']**2 + df['deltaPhi_g2_fj']**2)

# 5) deltaEta_subj1_gg, deltaPhi_subj1_gg, deltaR_subj1_gg
df['deltaEta_subj1_gg'] = subj1_eta - gg_eta
df['deltaPhi_subj1_gg'] = wrap_angle(subj1_phi - gg_phi)
df['deltaR_subj1_gg']   = np.sqrt(df['deltaEta_subj1_gg']**2 + df['deltaPhi_subj1_gg']**2)

# 6) deltaEta_subj2_gg, deltaPhi_subj2_gg, deltaR_subj2_gg
df['deltaEta_subj2_gg'] = subj2_eta - gg_eta
df['deltaPhi_subj2_gg'] = wrap_angle(subj2_phi - gg_phi)
df['deltaR_subj2_gg']   = np.sqrt(df['deltaEta_subj2_gg']**2 + df['deltaPhi_subj2_gg']**2)

# 7) deltaEta_subj1_subj2, deltaPhi_subj1_subj2, deltaR_subj1_subj2
df['deltaEta_subj1_subj2'] = subj1_eta - subj2_eta
df['deltaPhi_subj1_subj2'] = wrap_angle(subj1_phi - subj2_phi)
df['deltaR_subj1_subj2']   = np.sqrt(df['deltaEta_subj1_subj2']**2 + df['deltaPhi_subj1_subj2']**2)

# 8) deltaEta_g1_subj1, deltaPhi_g1_subj1, deltaR_g1_subj1
df['deltaEta_g1_subj1'] = g1_eta - subj1_eta
df['deltaPhi_g1_subj1'] = wrap_angle(g1_phi - subj1_phi)
df['deltaR_g1_subj1']   = np.sqrt(df['deltaEta_g1_subj1']**2 + df['deltaPhi_g1_subj1']**2)

# 9) deltaEta_g1_subj2, deltaPhi_g1_subj2, deltaR_g1_subj2
df['deltaEta_g1_subj2'] = g1_eta - subj2_eta
df['deltaPhi_g1_subj2'] = wrap_angle(g1_phi - subj2_phi)
df['deltaR_g1_subj2']   = np.sqrt(df['deltaEta_g1_subj2']**2 + df['deltaPhi_g1_subj2']**2)

# 10) deltaEta_g2_subj1, deltaPhi_g2_subj1, deltaR_g2_subj1
df['deltaEta_g2_subj1'] = g2_eta - subj1_eta
df['deltaPhi_g2_subj1'] = wrap_angle(g2_phi - subj1_phi)
df['deltaR_g2_subj1']   = np.sqrt(df['deltaEta_g2_subj1']**2 + df['deltaPhi_g2_subj1']**2)

# 11) deltaEta_g2_subj2, deltaPhi_g2_subj2, deltaR_g2_subj2
df['deltaEta_g2_subj2'] = g2_eta - subj2_eta
df['deltaPhi_g2_subj2'] = wrap_angle(g2_phi - subj2_phi)
df['deltaR_g2_subj2']   = np.sqrt(df['deltaEta_g2_subj2']**2 + df['deltaPhi_g2_subj2']**2)

df["subjet1_pt"] = df["fatjet_selected_subjet1_pt"]# / df['fatjet_selected_mass_corrected']
df["subjet1_mass"] = df["fatjet_selected_subjet1_mass"]# / df['fatjet_selected_mass_corrected']

df["subjet2_pt"] = df["fatjet_selected_subjet2_pt"] #/ df['fatjet_selected_mass_corrected']
df["subjet2_mass"] = df["fatjet_selected_subjet2_mass"] #/ df['fatjet_selected_mass_corrected']

df["DeltaR_jg_min"] = df[["deltaR_g1_subj1", "deltaR_g1_subj2", "deltaR_g2_subj1", "deltaR_g2_subj2"]].min(axis=1)

df["deltaPhi_puppiMET_subjet1"] = wrap_angle(df["puppiMET_phi"] - df["fatjet_selected_subjet1_phi"])
df["deltaPhi_puppiMET_subjet2"] = wrap_angle(df["puppiMET_phi"] - df["fatjet_selected_subjet2_phi"])

df_ALL = df.copy()
df = df[row_has].copy()
best_idx = best_idx[row_has]

# Filter out signal events without a matched Hbb fatjet.
signal_mask = df['proc'] == 'GluGluToHH'
df_without_genMatched = df
df = df[~signal_mask | (signal_mask & (df['fatjet_selected_genMatched_Hbb'] == 1))]

print(list(df.columns))

# plot_fatjet_properties_stacked(df)

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({'font.size': 16})

# =============================================================================
# Common Setup
# =============================================================================
# Define the features (same as before).
features = [
    #'lead_hoe',
    #'sublead_hoe',
    'fatjet_selected_msoftdrop',
    'fatjet_selected_tau2tau1_ratio',
    'fatjet_selected_subjet1_btagDeepB',
    'fatjet_selected_subjet2_btagDeepB',
    'fatjet_selected_particleNet_XbbVsQCD',
    'subjet1_pt',
    'subjet1_mass',
    'subjet2_pt',
    'subjet2_mass',
    'lead_eta',
    'sublead_eta',
    'eta',
    'n_leptons',
    'Res_CosThetaStar_gg',
    'Res_pholead_PtOverM',
    'Res_phosublead_PtOverM',
    #"deltaPhi_puppiMET_subjet1",
    #"deltaPhi_puppiMET_subjet2",
    #'MET_ptPNetCorr',
    #"sigma_m_over_m_smeared_decorr",
    #"puppiMET_sumEt",
    'DeltaR_jg_min',
    'deltaEta_g1_g2',
    'deltaPhi_g1_g2',
    'deltaEta_gg_fj',
    'deltaPhi_gg_fj',
    'deltaR_gg_fj',
    'deltaEta_g1_fj',
    'deltaPhi_g1_fj',
    'deltaR_g1_fj',
    'deltaEta_g2_fj',
    'deltaPhi_g2_fj',
    'deltaR_g2_fj',
    'deltaEta_subj1_gg',
    'deltaPhi_subj1_gg',
    'deltaR_subj1_gg',
    'deltaEta_subj2_gg',
    'deltaPhi_subj2_gg',
    'deltaR_subj2_gg',
    'deltaEta_subj1_subj2',
    'deltaPhi_subj1_subj2',
    'deltaR_subj1_subj2',
    'deltaEta_g1_subj1',
    'deltaPhi_g1_subj1',
    'deltaR_g1_subj1',
    'deltaEta_g1_subj2',
    'deltaPhi_g1_subj2',
    'deltaR_g1_subj2',
    'deltaEta_g2_subj1',
    'deltaPhi_g2_subj1',
    'deltaR_g2_subj1',
    'deltaEta_g2_subj2',
    'deltaPhi_g2_subj2',
    'deltaR_g2_subj2',
    'n_jets',
    'n_fatjets',
    'n_fatjets_final'
]
print("Features:")
print(features)
# Define the signal process.
signal_proc = "GluGluToHH"
# For a unified BDT we now combine the allowed processes from your two setups.
# Here we mimic your original training/test split: exclude the processes that you used only for testing.
# Non-test-only for training:
train_allowed = ["GluGluToHH", "TTGG", "GGJets", "ttHToGG", "VHToGG", "GluGluHToGG"]
# Processes to be reserved for testing only:
test_only = ["DDQCDGJets", "VBFHToGG", "BBHto2G"]

# More efficient filtering using sequential masks
print("Create the combined training dataset.")
mask_train = (
    (df['proc'] != 'Data') &
    (df['proc'].isin(train_allowed)) &
    (df['weight'] >= 0) &
    (df['n_fatjets'] > 0)
)
df_train = df[mask_train]

print("Create an extra test dataset from the test-only processes.")
mask_test_extra = (
    (df_without_genMatched['proc'] != 'Data') &
    (df_without_genMatched['proc'].isin(test_only)) &
    (df_without_genMatched['weight'] >= 0)
)
df_test_extra = df_without_genMatched[mask_test_extra]

print("Training event counts:")
print(df_train['proc'].value_counts())
print("\nTest-only event counts:")
print(df_test_extra['proc'].value_counts())

# Create a binary target: 1 for signal, 0 for background.
df_train['target'] = (df_train['proc'] == signal_proc).astype(int)
df_test_extra['target'] = (df_test_extra['proc'] == signal_proc).astype(int)

# =============================================================================
# Create Unified Train/Test Split
# =============================================================================

# Use the training dataset to create a split.
X = df_train[features].dropna()
y = df_train['target'].loc[X.index]
proc_train = df_train['proc'].loc[X.index]

# Split into training and an internal test set (stratified by target).
X_train, X_test_internal, y_train, y_test_internal, proc_train_split, proc_test_internal = train_test_split(
    X, y, proc_train, test_size=0.2, stratify=y, random_state=42
)

# Prepare the extra test set from the test-only processes.
X_test_extra = df_test_extra[features].dropna()
y_test_extra = df_test_extra['target'].loc[X_test_extra.index]
proc_test_extra = df_test_extra['proc'].loc[X_test_extra.index]

# Combine the internal test set with the extra test events.
X_test = pd.concat([X_test_internal, X_test_extra])
y_test = pd.concat([y_test_internal, y_test_extra])
proc_test = pd.concat([proc_test_internal, proc_test_extra])

print(f"Combined training set size: {len(y_train)}")
print(f"Combined test set size: {len(y_test)}")

# =============================================================================
# Train and Evaluate VH-specific BDT (Signal: GluGluToHH vs Background: VHToGG)
# =============================================================================

# Select only events that are either the signal (GluGluToHH) or the VH background (VHToGG)
vh_mask = df_train['proc'].isin([signal_proc, "VHToGG"])
df_train_vh = df_train[vh_mask]
print("VH-specific training event counts:")
print(df_train_vh['proc'].value_counts())

# Create binary target: 1 for signal, 0 for VHToGG
df_train_vh['target'] = (df_train_vh['proc'] == signal_proc).astype(int)

# Select features and drop events with missing values.
X_vh = df_train_vh[features].dropna()
y_vh = df_train_vh['target'].loc[X_vh.index]
proc_vh = df_train_vh['proc'].loc[X_vh.index]

# Split the VH-specific dataset into training and test sets (stratified by target)
X_train_vh, X_test_vh, y_train_vh, y_test_vh, proc_train_vh, proc_test_vh = train_test_split(
    X_vh, y_vh, proc_vh, test_size=0.2, stratify=y_vh, random_state=42
)

print(f"VH-specific training set size: {len(y_train_vh)}")
print(f"VH-specific test set size: {len(y_test_vh)}")

# Train the VH-specific BDT with the same hyperparameters
clf_vh = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42,
    max_depth=3,
    learning_rate=0.01,
    n_estimators=200,
    reg_lambda=10,
    reg_alpha=1.0,
    subsample=0.6,
    colsample_bytree=0.6,
)
clf_vh.fit(X_train_vh, y_train_vh)

import joblib
joblib.dump(clf_vh, "clf_vh_model.pkl")

# Evaluate the VH-specific BDT
y_pred_vh = clf_vh.predict(X_test_vh)
print("\nVH-specific BDT Classification Report:")
print(classification_report(y_test_vh, y_pred_vh))
print("VH-specific BDT Test Accuracy:", accuracy_score(y_test_vh, y_pred_vh))

# Get predicted probabilities for both train and test sets
y_proba_train_vh = clf_vh.predict_proba(X_train_vh)[:, 1]
y_proba_test_vh = clf_vh.predict_proba(X_test_vh)[:, 1]

# Compute ROC curve and AUC for the training set
fpr_train_vh, tpr_train_vh, _ = roc_curve(y_train_vh, y_proba_train_vh)
roc_auc_train_vh = auc(fpr_train_vh, tpr_train_vh)

# Compute ROC curve and AUC for the test set
fpr_test_vh, tpr_test_vh, _ = roc_curve(y_test_vh, y_proba_test_vh)
roc_auc_test_vh = auc(fpr_test_vh, tpr_test_vh)

# Plot ROC curves for both train and test sets
plt.figure(figsize=(8, 6))
plt.plot(fpr_test_vh, tpr_test_vh, lw=2, label=f'VH BDT Test (AUC = {roc_auc_test_vh:.3f})', color='blue')
plt.plot(fpr_train_vh, tpr_train_vh, lw=2, linestyle='--',
         label=f'VH BDT Train (AUC = {roc_auc_train_vh:.3f})', color='blue')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("VH-specific BDT ROC Curves (Train vs Test)")
plt.legend(loc="best", fontsize=13)
plt.grid(True)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.savefig("ROC_VH.png")


import matplotlib.pyplot as plt
import numpy as np
# ---------------------------------------------
# VH-specific BDT Feature Importances (Top 15)
# ---------------------------------------------
# Get raw importances from the VH-specific classifier.
importances_vh = clf_vh.feature_importances_
feature_names_vh = X_train_vh.columns

# Sort in descending order.
desc_idx_vh = np.argsort(importances_vh)[::-1]
# Select top 15 indices.
top15_idx_vh = desc_idx_vh[:15]
top15_features_vh = feature_names_vh[top15_idx_vh]
top15_importances_vh = importances_vh[top15_idx_vh]

plt.rcParams.update({'font.size': 10})
plt.figure(figsize=(8, 6))
# Reverse order for a horizontal bar plot (largest at the top)
plt.barh(top15_features_vh[::-1], top15_importances_vh[::-1], height=0.6)
plt.xlabel("Feature Importance")
plt.title("VH-specific BDT Top 15 Feature Importances")
plt.tight_layout()
plt.savefig("Feature_Importance_VH.png")


# ---------------------------
# BDT Score plotting
# ---------------------------
X_testPlusTrain = df_without_genMatched[features].dropna()
proc_all = df_without_genMatched['proc'].loc[X_testPlusTrain.index]
weights_all = df_without_genMatched['weight'].loc[X_testPlusTrain.index]
background_procs = ["TTGG", "GGJets", "ttHToGG", "VHToGG", "GluGluHToGG", "VBFHToGG","DDQCDGJets","BBHto2G"]
y_proba = clf_vh.predict_proba(X_testPlusTrain)[:,1]
plt.xlim([0.0, 1.0])
plt.clf()
def plot_score_distribution(y_proba, proc, proc_list, title, weights):
    plt.figure(figsize=(10, 6))
    # Contatori per ognuno dei background
    for bkg in proc_list:
        mask = proc == "VHToGG"
        y_proba_bkg = y_proba[mask]
        weights_bkg = weights[mask]
        # Plot stacked histogram per ogni background e segnale.
        plt.hist(y_proba_bkg, bins=50,range=(0, 1.0), stacked=True, label=f'{bkg} (Train+Test)', histtype='stepfilled', linewidth=2, alpha=0.7,weights=weights_bkg)
    # Aggiungi il segnale
    mask_signal = proc == "GluGluToHH"
    y_proba_signal = y_proba[mask_signal]
    weights_signal = weights[mask_signal]
    plt.hist(y_proba_signal, bins=50,range=(0, 1.0), stacked=False, label=f'{signal_proc} (Train+Test)', histtype='step', linewidth=2, color='blue',weights=weights_signal)
    # Configura l'asse e il grafico
    plt.xlabel('XGBoost BDT Probability Score')
    plt.ylabel('Weighted Events')
    plt.title("")
    plt.yscale('log')
    plt.legend(loc='best',fontsize=7)
    plt.grid(True)
# Do the plot for test+training datasets
plot_score_distribution(y_proba, proc_all, background_procs, 'Test+Train Set Score Distribution', weights_all)
plt.savefig("BDT_VH_Score.png")


# =============================================================================
# Train Unified BDT with VH Score as an Additional Input Feature
# =============================================================================

# Compute the VH-specific BDT output ("vh_score") for every event in the unified training and test sets.
vh_score_train = clf_vh.predict_proba(X_train)[:, 1]
vh_score_test = clf_vh.predict_proba(X_test)[:, 1]

# Create new training and test datasets that include the "vh_score" as an extra feature.
X_train_plus_vh = X_train.copy()
X_train_plus_vh['vh_score'] = vh_score_train

X_test_plus_vh = X_test.copy()
X_test_plus_vh['vh_score'] = vh_score_test

# Train a new unified BDT using the augmented feature set.
clf_all_plus_vh = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    random_state=42,
    max_depth=3,
    learning_rate=0.01,
    n_estimators=1000,
    reg_lambda=10,
    reg_alpha=1.0,
    subsample=0.6,
    colsample_bytree=0.6,
)
clf_all_plus_vh.fit(X_train_plus_vh, y_train)

joblib.dump(clf_all_plus_vh, "clf_all_plus_vh.pkl")

# =============================================================================
# Evaluate the New Unified BDT with VH Score Feature
# =============================================================================

y_pred_all_plus_vh = clf_all_plus_vh.predict(X_test_plus_vh)
print("\nUnified BDT with VH Score Feature Classification Report:")
print(classification_report(y_test, y_pred_all_plus_vh))
print("Unified BDT with VH Score Feature Test Accuracy:", accuracy_score(y_test, y_pred_all_plus_vh))

# Get predicted probabilities for ROC curve plotting.
y_proba_train_all_plus_vh = clf_all_plus_vh.predict_proba(X_train_plus_vh)[:, 1]
y_proba_test_all_plus_vh = clf_all_plus_vh.predict_proba(X_test_plus_vh)[:, 1]

# =============================================================================
# Plot ROC Curves by Process (Train vs Test) for the New Unified Classifier
# =============================================================================

# Use the same color mapping as before.
all_procs = [proc for proc in df['proc'].unique() if proc != 'Data']
color_map = {proc: plt.cm.tab10.colors[i % len(plt.cm.tab10.colors)] for i, proc in enumerate(all_procs)}

plt.figure(figsize=(8, 6))

# Plot ROC curves for each background process in the test set.
background_procs_test = np.unique(proc_test[y_test == 0])
for bkg in background_procs_test:
    if ((bkg != "BBHto2G") & (bkg != "VBFHToGG") & (bkg != "DDQCDGJets")):
        mask_test = (proc_test == signal_proc) | ((proc_test == bkg))
        y_test_subset = y_test[mask_test]
        y_proba_subset = y_proba_test_all_plus_vh[mask_test]
        fpr, tpr, _ = roc_curve(y_test_subset, y_proba_subset)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{bkg} Test (AUC = {roc_auc:.3f})', color=color_map[bkg])

# Plot ROC curves for each background process in the training set (dashed lines).
background_procs_train = np.unique(proc_train_split[y_train == 0])
for bkg in background_procs_train:
    if ((bkg != "BBHto2G") & (bkg != "VBFHToGG") & (bkg != "DDQCDGJets") & (bkg != "VHToGG")) :
        mask_train = (proc_train_split == signal_proc) | ((proc_train_split == bkg))
        y_train_subset = y_train[mask_train]
        y_proba_subset = y_proba_train_all_plus_vh[mask_train]
        fpr, tpr, _ = roc_curve(y_train_subset, y_proba_subset)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, linestyle='--', label=f'{bkg} Train (AUC = {roc_auc:.3f})', color=color_map[bkg])

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("Unified BDT with VH Score ROC Curves (Train vs Test)")
plt.legend(loc="best", fontsize=13)
plt.grid(True)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.savefig("BDT_Merged_ROC.png")

# =============================================================================
# Plot Feature Importances for the New Unified Classifier (Top 15)
# =============================================================================

# Extract and sort feature importances.
importances_all_plus_vh = clf_all_plus_vh.feature_importances_
feature_names_all_plus_vh = X_train_plus_vh.columns
desc_idx_all_plus_vh = np.argsort(importances_all_plus_vh)[::-1]
top15_idx_all_plus_vh = desc_idx_all_plus_vh[:15]
top15_features_all_plus_vh = feature_names_all_plus_vh[top15_idx_all_plus_vh]
top15_importances_all_plus_vh = importances_all_plus_vh[top15_idx_all_plus_vh]
plt.clf()
plt.rcParams.update({'font.size': 10})
plt.figure(figsize=(8, 6))
plt.barh(top15_features_all_plus_vh[::-1], top15_importances_all_plus_vh[::-1], height=0.6)
plt.xlabel("Feature Importance")
plt.title("Unified BDT with VH Score Feature - Top 15 Feature Importances")
plt.tight_layout()
plt.savefig("Feature_Importance_Merged.png")

# =============================================================================
# Apply a 70% Signal Efficiency Cut on the Full Sample and Sum Weights
# =============================================================================

# Combine the full sample used in training and testing.
df_full = df_ALL #df_without_genMatched

# We only consider events with complete feature information.
X_full = df_full[features]#.dropna()
df_full['vh_score'] = clf_vh.predict_proba(X_full)[:,1]

X_full['vh_score'] = df_full['vh_score']
proc_all = df_full['proc'].loc[X_full.index]
weights_all = df_full['weight'].loc[X_full.index]
background_procs = ["TTGG", "GGJets", "ttHToGG", "VHToGG", "GluGluHToGG", "VBFHToGG","DDQCDGJets"]
y_proba = clf_all_plus_vh.predict_proba(X_full)[:,1]
plt.xlim([0.0, 1.0])

def plot_score_distribution(y_proba, proc, proc_list, title, weights):
    plt.clf()
    plt.figure(figsize=(10, 6))
    # Contatori per ognuno dei background                                                                                                                     
    for bkg in proc_list:
        mask = proc == bkg
        y_proba_bkg = y_proba[mask]
        weights_bkg = weights[mask]
        # Plot stacked histogram per ogni background e segnale.
        plt.hist(y_proba_bkg, bins=50,range=(0, 1.0), stacked=True, label=f'{bkg} (Train+Test)', histtype='stepfilled', linewidth=2, alpha=0.7,weights=weights_bkg)
    # Aggiungi il segnale                                                                                                                                     
    mask_signal = proc == "GluGluToHH"
    y_proba_signal = y_proba[mask_signal]
    weights_signal = weights[mask_signal]
    plt.hist(y_proba_signal, bins=50,range=(0, 1.0), stacked=False, label=f'{signal_proc} (Train+Test)', histtype='step', linewidth=2, color='blue',weights=weights_signal)
    # Configura l'asse e il grafico
    plt.xlabel('XGBoost BDT Probability Score')
    plt.ylabel('Weighted Events')
    plt.title("")
    plt.yscale('log')
    plt.legend(loc='best',fontsize=7)
    plt.grid(True)
# Do the plot for test+training datasets                                                                                                                      
plot_score_distribution(y_proba, proc_all, background_procs, 'Test+Train Set Score Distribution', weights_all)
plt.savefig("BDT_Merged_Score.png")


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_multiple_features(df, features, proc_col='proc', weight_col='weight', signal_proc='GluGluHH', background_procs=None, normalize=True, label='features'):
    """
    Plots multiple feature distributions for signal and background.

    Parameters:
    - df: DataFrame containing the dataset
    - features: List of feature names to plot
    - proc_col: Column name indicating the process type (signal/background)
    - weight_col: Column name for event weights
    - signal_proc: Name of the signal process to highlight
    - background_procs: List of background processes (if None, all except signal are considered background)
    - normalize: If True, normalizes the distributions to unity
    """

    # If background_procs is not provided, use all processes except the signal
    if background_procs is None:
        background_procs = df[proc_col].unique()
        background_procs = [p for p in background_procs if p != signal_proc]  # Exclude signal
    
    # Define the number of rows/columns for subplots
    num_features = len(features)
    num_cols = 4  # Number of columns in the plot grid
    num_rows = int(np.ceil(num_features / num_cols))  # Calculate needed rows

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 4))
    axes = axes.flatten()  # Flatten to make indexing easier

    for i, feature in enumerate(features):
        ax = axes[i]
        # Create masks to filter signal and background events
        mask_signal = df[proc_col] == signal_proc
        #mask_background = df[proc_col].isin(background_procs)
        # Extract values and weights for signal and background
        signal_values = df.loc[mask_signal, feature]
        signal_weights = df.loc[mask_signal, weight_col]
        #background_values = df.loc[mask_background, feature]
        #background_weights = df.loc[mask_background, weight_col]
        # Plot background distribution (blue filled histogram)
        #ax.hist(background_values, bins=50, range=(df[feature].min(), df[feature].max()), weights=background_weights,
        #         density=normalize, alpha=0.5, label='Background', histtype='stepfilled', linewidth=2, color='blue')
        # Plot signal distribution (red line)                                                                            
        #ax.hist(signal_values, bins=50, range=(df[feature].min(), df[feature].max()), weights=signal_weights,
        #density=normalize, alpha=0.7, label=f'Signal ({signal_proc})', histtype='step', linewidth=2, color='red')
        bkg_values_list = []
        bkg_weights_list = []
        labels = []
        for bkg in background_procs:
            mask = df[proc_col] == bkg
            bkg_values = df.loc[mask, feature]
            bkg_weights = df.loc[mask, weight_col]
            bkg_values_list.append(bkg_values)
            bkg_weights_list.append(bkg_weights)
            labels.append(bkg)
        ax.hist(signal_values,
                bins=50,
                range=(df[feature].min(), df[feature].max()),
                weights=signal_weights * 1000,
                histtype="step",
                color="black",
                linewidth=2,
                label="GluGluToHH(x1000)",density=False
        )
        # Plot dello stack dei background
        ax.hist(bkg_values_list,bins=50,range=(df[feature].min(), df[feature].max()),stacked=True,weights=bkg_weights_list,label=labels,  alpha=0.8, density=False)
        # Aggiungi il segnale (non stacked)
        
        """
        ax.hist(signal_values,
                bins=50,
                range=(df[feature].min(), df[feature].max()),
                weights=signal_weights * 1000,
                histtype="step",
                color="black",
                linewidth=2,
                label="GluGluToHH(x1000)",density=False,zorder=3
        )
        """
        # Label axes and title
        ax.set_xlabel(feature)
        ax.set_ylabel('Normalized Events' if normalize else 'Events')
        ax.set_title(f'Distribution of {feature}')
        ax.legend()
        ax.grid()

    # Hide unused subplots if any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig(f"{label}.png")

# Example usage:
features = features + ['fatjet_selected_mass_corrected','fatjet_selected_msoftdrop']
features = features + ['Res_mjj_regressed']
# plot_multiple_features(df_full, features=features, signal_proc="GluGluHH", background_procs=background_procs, label='Features_beforeCut')


# Create a Series of weights corresponding to the events with non-null features.
weights_full = df_full.loc[X_full.index, "weight"]
# Compute predicted signal probabilities on the full sample.
y_proba_full = clf_all_plus_vh.predict_proba(X_full)[:, 1]
y_proba_full_series = pd.Series(y_proba_full, index=X_full.index)
# For the signal events in the full sample, determine the threshold that retains 70% of them.
# "Signal efficiency" here is the fraction of signal events with predicted probability >= threshold.
df_full['target']= (df_full['proc'] == signal_proc).astype(int)
signal_mask = (df_full.loc[X_full.index, "target"] == 1)
signal_probs = y_proba_full_series[signal_mask]

# The threshold is the 30th percentile of the signal probability distribution
# (so that 70% of signal events have probability above this threshold).
threshold = np.percentile(signal_probs, 30)
print(f"70% signal efficiency threshold: {threshold:.3f}")

# Apply the cut: keep events with predicted signal probability >= threshold.
df_full_cut = df_full #.loc[X_full.index].copy()
df_full_cut["y_proba"] = y_proba_full_series
df_full_cut.loc[df_full["n_fatjets_final"] == 0, "y_proba"] = -99
df_full_cut.to_parquet("df_full_cut_with_y_proba_forGiacomo.parquet", index=False)
y_proba_thresholds = [0.10, 0.2, 0.3, 0.4, 0.5,0.6,0.7,0.8,0.9,0.92,0.93]

for thr in y_proba_thresholds:
    # Filtro il DataFrame
    df_cut = df_full_cut[df_full_cut["y_proba"] >= thr]
    
    # Genero un label dinamico
    label = f"Features_yProbaCut{int(thr*100)}"

    print(f"Plotting for y_proba ≥ {thr}, events: {len(df_cut)}")
    
    # Chiamo la funzione di plot
    plot_multiple_features(
        df_cut,
        features=features,
        signal_proc="GluGluHH",
        background_procs=background_procs,
        label=label
     )
#df_full_cut = df_full_cut[df_full_cut["y_proba"]>=0.92]
#plot_multiple_features(df_full_cut, features=features, signal_proc="GluGluHH", background_procs=background_procs, label=#'Features_')
#df_full_cut = df_full_cut[df_full_cut["y_proba"]>=0.92]
#plot_multiple_features(df_full_cut, features=features, signal_proc="GluGluHH", background_procs=background_procs, label='Features_afterCut')

weight_sum_by_proc = df_full_cut.groupby("proc")["weight"].sum()
print("\nWeighted sum by process BEFORE cut:")
print(weight_sum_by_proc)

# Definisci un intervallo di soglie
thresholds = np.linspace(0, 1.0, 50)  # 100 valori tra 0.90 e 0.99
best_threshold = None
best_s_over_sqrt_b = 0

results = []  # Per salvare i risultati

for threshold in thresholds:
    df_cut = df_full_cut[df_full_cut["y_proba"] >= threshold]
    # Somma pesata per ogni processo
    weight_sum_by_proc = df_cut.groupby("proc")["weight"].sum()
    # Somma totale dei background
    background_weight_total = weight_sum_by_proc.loc[weight_sum_by_proc.index != signal_proc].sum()
    # Somma pesata del segnale
    signal_weight = weight_sum_by_proc.get(signal_proc, 0.0)
    # Calcolo S/sqrt(B)
    s_over_sqrt_b = signal_weight / np.sqrt(background_weight_total) if background_weight_total > 0 else 0
    results.append((threshold, s_over_sqrt_b))
    # Aggiorna il valore ottimale
    if s_over_sqrt_b > best_s_over_sqrt_b:
        best_s_over_sqrt_b = s_over_sqrt_b
        best_threshold = threshold

# Print best result
print(f"Optimal threshold: {best_threshold:.3f} with S/sqrt(B) = {best_s_over_sqrt_b:.3f}")
df_full_cut = df_full_cut[df_full_cut["y_proba"] >= best_threshold]
weight_sum_by_proc = df_full_cut.groupby("proc")["weight"].sum()
print(f"\nWeighted sum by process AFTER cut ({best_threshold:.3f}):")
print(weight_sum_by_proc)

            
import matplotlib.pyplot as plt
plt.clf()
thresholds, s_sqrt_b_values = zip(*results)
plt.plot(thresholds, s_sqrt_b_values, marker='o', linestyle='-')
plt.xlabel("Threshold (y_proba)")
plt.ylabel("S/sqrt(B)")
plt.title("Optimization of the threshold S/sqrt(B)")
plt.grid()
plt.savefig("Threshold_Significance_BDT_VHFeeded.png")
