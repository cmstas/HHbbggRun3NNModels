import pandas as pd

df1 = pd.read_parquet("df_full_cut_with_y_proba_forGiacomo.parquet")
df2 = pd.read_parquet("Results_noDijetMass_PNetJets_DijetReg_onlyNonRes_eval_kl_boosted_addSingleH_mbbtrimmed_withFatjets.parquet")

print(df1["mass"])
print(df2["mass"])
for col in df1.columns:
    if "selected" in col:
        print(col)
# Check which indices in df2 are not present in df1
missing_indices = df2.index.difference(df1.index)

# Print missing indices
if not missing_indices.empty:
    print("Missing indices in df1:")
    print(missing_indices.tolist())

df2["y_proba"] = df1["y_proba"].reindex(df2.index).fillna(-99)

df2["y_proba"] = df2["y_proba"].astype("float32")

df2["fatjet_selected_msoftdrop"] = df1["fatjet_selected_msoftdrop"].reindex(df2.index).fillna(-99)
df2["fatjet_selected_msoftdrop"] = df2["fatjet_selected_msoftdrop"].astype("float32")

df2.to_parquet("Results_noDijetMass_PNetJets_DijetReg_onlyNonRes_eval_kl_boosted_addSingleH_mbbtrimmed_withFatjets_boostedBDT.parquet", index=True)

print("Column 'y_proba' copied into df2. Missing entries filled with -99.")
