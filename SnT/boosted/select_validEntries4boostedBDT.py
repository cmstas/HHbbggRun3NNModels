import pandas as pd
import os

input_file = "Results_noDijetMass_PNetJets_DijetReg_onlyNonRes_eval_kl_boosted_addSingleH_mbbtrimmed.parquet"

df = pd.read_parquet(input_file)

# Split based on n_fatjets > 0
df_has_fatjets = df[df["n_fatjets"] > 0]
df_no_fatjets = df[df["n_fatjets"] <= 0]

base, ext = os.path.splitext(input_file)
out_with_fatjets = f"{base}_withFatjets{ext}"
out_without_fatjets = f"{base}_noFatjets{ext}"

df_has_fatjets.to_parquet(out_with_fatjets, index=False)
df_no_fatjets.to_parquet(out_without_fatjets, index=False)

print(f"Saved events with n_fatjets > 0 to: {out_with_fatjets} ({len(df_has_fatjets)} rows)")
print(f"Saved remaining events to: {out_without_fatjets} ({len(df_no_fatjets)} rows)")
