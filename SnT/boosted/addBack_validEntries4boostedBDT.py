import pandas as pd
import numpy as np
df_with_fatjets = pd.read_parquet("Results_noDijetMass_PNetJets_DijetReg_onlyNonRes_eval_kl_boosted_addSingleH_mbbtrimmed_withFatjets_boostedBDT.parquet")
df_no_fatjets = pd.read_parquet("Results_noDijetMass_PNetJets_DijetReg_onlyNonRes_eval_kl_boosted_addSingleH_mbbtrimmed_noFatjets.parquet")

df_no_fatjets["y_proba"] = -99.0

df_combined = pd.concat([df_with_fatjets, df_no_fatjets], ignore_index=True)

#Only needed columns for final fit
columns_to_keep = ["eta","event", "run", "lumi", "y_proba", "mass", "weight","score_GluGluToHH","pred_GluGluToHH","Res_mjj_regressed","proc","fatjet_selected_mass_corrected"]

output_file = "Results_noDijetMass_PNetJets_DijetReg_onlyNonRes_eval_kl_boosted_addSingleH_mbbtrimmed_boosteBDT_ALL_trimmed.parquet"
# Save only selected columns 
df_combined[columns_to_keep].to_parquet(output_file, index=False)

print(f"Successfully combined the two datasets.")
print(f"Output saved to: {output_file}")
print(f"Total events: {len(df_combined)}")
