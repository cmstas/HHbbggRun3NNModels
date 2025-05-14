import pandas as pd

# Load the two datasets
df_resolved = pd.read_parquet("resolved_category_trimmed4finalfit.parquet")
df_boosted = pd.read_parquet("boosted_category_trimmed4finalfit.parquet")

# Add 'mjj' and 'is_boosted' columns
df_resolved = df_resolved[(df_resolved['Res_mjj_regressed'] > 70) & (df_resolved['Res_mjj_regressed'] < 190)]
df_resolved['mjj'] = df_resolved['Res_mjj_regressed']
df_resolved['is_boosted'] = 0

df_boosted['mjj'] = df_boosted['fatjet_selected_mass_corrected']
df_boosted['is_boosted'] = 1

# Combine the two datasets
df_combined = pd.concat([df_resolved, df_boosted], ignore_index=True)

# Save the combined dataset
df_combined.to_parquet("combined_resolved_boosted_trimmed4finalfit.parquet")

print("Combined parquet file saved as: combined_resolved_boosted.parquet")