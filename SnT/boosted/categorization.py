import pandas as pd

# === 1. Load Parquet file ===
input_file = "Results_noDijetMass_PNetJets_DijetReg_onlyNonRes_eval_kl_boosted_addSingleH_mbbtrimmed_boosteBDT_ALL_trimmed.parquet"
df = pd.read_parquet(input_file)

# === 2. Define selection thresholds ===
BDT_boosted_cut = 0.939
DNN_nonRes_cut_SR0 = 0.998439
DNN_nonRes_cut_SR1 = 0.983356
DNN_Res_cut_SR0 = 0.95406
DNN_Res_cut_SR1 = 0.845485

# === 3. Define selection masks ===
sel_resolved_SR0 = (df["score_GluGluToHH"] >= DNN_nonRes_cut_SR0) & (df["pred_GluGluToHH"] >= DNN_Res_cut_SR0)
sel_resolved_SR1 = (
    (df["score_GluGluToHH"] >= DNN_nonRes_cut_SR1)
    & (df["pred_GluGluToHH"] >= DNN_Res_cut_SR1)
    & ~sel_resolved_SR0
)
sel_resolved = sel_resolved_SR0 | sel_resolved_SR1
sel_boosted = df["y_proba"] >= BDT_boosted_cut

# === 4. Categorize each event ===
category = pd.Series("other", index=df.index)
category[sel_resolved & ~sel_boosted] = "resolved"
category[sel_boosted] = "boosted"
df["category"] = category

for name in df.columns:
  print(name)
  
print(df[sel_resolved_SR0]["weight"])

# === 5. Global Selection Statistics ===
n_resolved_SR0 = df[sel_resolved_SR0]["weight"].sum()
n_resolved_SR1 = df[sel_resolved_SR1]["weight"].sum()
n_resolved_all = df[sel_resolved]["weight"].sum()
n_boosted_raw = df[sel_boosted]["weight"].sum()
n_both = df[(sel_resolved & sel_boosted)]["weight"].sum()
n_boosted = df[(df["category"] == "boosted")]["weight"].sum()
n_resolved = df[(df["category"] == "resolved")]["weight"].sum()
n_other = df[(df["category"] == "other")]["weight"].sum()

print("\n=== Global Selection Statistics ===")
print(f"Events passing boosted cut    : {n_boosted_raw:.2f}")
print(f"Events passing resolved SR0   : {n_resolved_SR0:.2f}")
print(f"Events passing resolved SR1   : {n_resolved_SR1:.2f}")
print(f"Events passing both selections: {n_both:.2f}")
print(f"Boosted category events       : {n_boosted:.2f}")
print(f"Resolved category events      : {n_resolved:.2f}")
print(f"Other category events         : {n_other:.2f}")

print("\n=== Per-process Breakdown by Category (Yield + Fraction of Total per Process) ===")
for proc in sorted(df["proc"].unique()):
    n_total = df[(df["proc"] == proc)]["weight"].sum()
    n_boosted = df[((df["proc"] == proc) & (df["category"] == "boosted"))]["weight"].sum()
    n_resolved = df[((df["proc"] == proc) & (df["category"] == "resolved"))]["weight"].sum()
    n_other = df[((df["proc"] == proc) & (df["category"] == "other"))]["weight"].sum()
    n_both = df[((df["proc"] == proc) & (sel_resolved & sel_boosted))]["weight"].sum()  # NEW: Events in both categories

    f_boosted = n_boosted / n_total if n_total else 0
    f_resolved = n_resolved / n_total if n_total else 0
    f_other = n_other / n_total if n_total else 0
    f_both = n_both / n_total if n_total else 0  # NEW: Fraction in both categories

    print(f"{proc:<20} | Total: {n_total:10.2f} | "
    f"Boosted: {n_boosted:10.3f} ({f_boosted:.3%}) | "
    f"Resolved: {n_resolved:10.3f} ({f_resolved:.3%}) | "
    f"Both: {n_both:10.3f} ({f_both:.3%})")  # NEW: Added "Both" column
    f"Other: {n_other:10.3f} ({f_other:.3%}) | "



# === 7. Output categorized data ===
df[df["category"] == "boosted"].to_parquet("boosted_category_trimmed4finalfit.parquet", index=False)
df[df["category"] == "resolved"].to_parquet("resolved_category_trimmed4finalfit.parquet", index=False)
df[df["category"] == "other"].to_parquet("other_category_trimmed4finalfit.parquet", index=False)

print("\n✅ Output saved: boosted_category.parquet, resolved_category.parquet, other_category.parquet")
