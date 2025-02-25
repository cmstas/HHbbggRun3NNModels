import numpy as np
import math
import pandas as pd
import common
from common import deltaPhi, deltaR

# If df==None, it creates a list of the variables needed in the dataframe.
# If df!=None, it creates a dataframe with all the needed variables.
def compositeVariables(input_variables, df_columns, df=None):
  skimmedVariables = []

  # Define dependencies and calculation methods in a dictionary
  variable_definitions = {
    "boosted":{
      "dependencies": ["fatjet1_pt", "fatjet2_pt", "fatjet3_pt", "fatjet4_pt"],
      "calculation": lambda df: ((df[['fatjet1_pt', 'fatjet2_pt', 'fatjet3_pt', 'fatjet4_pt']] > 600).any(axis=1)).astype(int)
    },
     "VBF": {
       "dependencies": ["VBF_first_jet_eta", "VBF_second_jet_eta", "VBF_dijet_mass"],
       "calculation": lambda df: (
         ((df["VBF_first_jet_eta"] - df["VBF_second_jet_eta"]).abs() > 3.0) & 
         (df["VBF_dijet_mass"] > 500)
       ).astype(int)
     },
    "nonRes_lead_bjet_massOverMjj_PNet_all": {
      "dependencies": ["nonRes_lead_bjet_mass_PNet_all", "nonRes_dijet_mass_PNet_all"],
      "calculation": lambda df: df['nonRes_lead_bjet_mass_PNet_all'] / df['nonRes_dijet_mass_PNet_all']
    },
    "nonRes_lead_bjet_ptOverMjj_PNet_all": {
      "dependencies": ["nonRes_lead_bjet_pt_PNet_all", "nonRes_dijet_mass_PNet_all"],
      "calculation": lambda df: df['nonRes_lead_bjet_pt_PNet_all'] / df['nonRes_dijet_mass_PNet_all']
    },
    "nonRes_sublead_bjet_massOverMjj_PNet_all": {
      "dependencies": ["nonRes_sublead_bjet_mass_PNet_all", "nonRes_dijet_mass_PNet_all"],
      "calculation": lambda df: df['nonRes_sublead_bjet_mass_PNet_all'] / df['nonRes_dijet_mass_PNet_all']
    },
    "nonRes_sublead_bjet_ptOverMjj_PNet_all": {
      "dependencies": ["nonRes_sublead_bjet_pt_PNet_all", "nonRes_dijet_mass_PNet_all"],
      "calculation": lambda df: df['nonRes_sublead_bjet_pt_PNet_all'] / df['nonRes_dijet_mass_PNet_all']
    },
    "nonRes_dijet_PtOverMggjj_PNet_all": {
      "dependencies": ["nonRes_dijet_pt_PNet_all", "nonRes_HHbbggCandidate_mass"],
      "calculation": lambda df: df['nonRes_dijet_pt_PNet_all'] / df['nonRes_HHbbggCandidate_mass']
    },
    "nonRes_dijet_massOverMggjj_PNet_all": {
      "dependencies": ["nonRes_dijet_mass_PNet_all", "nonRes_HHbbggCandidate_mass"],
      "calculation": lambda df: df['nonRes_dijet_mass_PNet_all'] / df['nonRes_HHbbggCandidate_mass']
    },
    "lead_bjet_massOverMjj": {
      "dependencies": ["nonRes_lead_bjet_mass", "nonRes_dijet_mass"],
      "calculation": lambda df: df['nonRes_lead_bjet_mass'] / df['nonRes_dijet_mass']
    },
    "sublead_bjet_massOverMjj": {
      "dependencies": ["nonRes_sublead_bjet_mass", "nonRes_dijet_mass"],
      "calculation": lambda df: df['nonRes_sublead_bjet_mass'] / df['nonRes_dijet_mass']
    },
    "dipho_PtOverMggjj": {
      "dependencies": ["pt", "nonRes_HHbbggCandidate_mass"],
      "calculation": lambda df: df['pt'] / df['nonRes_HHbbggCandidate_mass']
    },
    "dipho_PtOverMgg": {
      "dependencies": ["pt", "mass"],
      "calculation": lambda df: df['pt'] / df['mass']
    },
    "dijet_PtOverMggjj": {
      "dependencies": ["nonRes_dijet_pt", "nonRes_HHbbggCandidate_mass"],
      "calculation": lambda df: df['nonRes_dijet_pt'] / df['nonRes_HHbbggCandidate_mass']
    },
    "dijet_PtOverMjj": {
      "dependencies": ["nonRes_dijet_pt", "nonRes_dijet_mass"],
      "calculation": lambda df: df['nonRes_dijet_pt'] / df['nonRes_dijet_mass']
    },
    "dijet_massOverMggjj": {
      "dependencies": ["nonRes_dijet_mass", "nonRes_HHbbggCandidate_mass"],
      "calculation": lambda df: df['nonRes_dijet_mass'] / df['nonRes_HHbbggCandidate_mass']
    },
    "deltaPhi_g1_g2": {
      "dependencies": ['sublead_phi', 'lead_phi'],
      "calculation": lambda df: df.apply(lambda row: deltaPhi(row["sublead_phi"], row["lead_phi"]), axis=1)
    },
    "deltaPhi_g1_j1": {
      "dependencies": ['nonRes_lead_bjet_phi', 'lead_phi'],
      "calculation": lambda df: df.apply(lambda row: deltaPhi(row["nonRes_lead_bjet_phi"], row["lead_phi"]), axis=1)
    },
    "deltaPhi_g1_j2": {
      "dependencies": ['nonRes_sublead_bjet_phi', 'lead_phi'],
      "calculation": lambda df: df.apply(lambda row: deltaPhi(row["nonRes_sublead_bjet_phi"], row["lead_phi"]), axis=1)
    },
    "lead_eta_mirrored": {
      "dependencies": ["lead_eta"],
      "calculation": lambda df: df["lead_eta"] * np.sign(df["lead_eta"])
    },
    "sublead_eta_mirrored": {
      "dependencies": ["sublead_eta", "lead_eta"],
      "calculation": lambda df: df["sublead_eta"] * np.sign(df["lead_eta"])
    },
    "lead_bjet_eta_mirrored": {
      "dependencies": ["nonRes_lead_bjet_eta", "lead_eta"],
      "calculation": lambda df: df["nonRes_lead_bjet_eta"] * np.sign(df["lead_eta"])
    },
    "sublead_bjet_eta_mirrored": {
      "dependencies": ["nonRes_sublead_bjet_eta", "lead_eta"],
      "calculation": lambda df: df["nonRes_sublead_bjet_eta"] * np.sign(df["lead_eta"])
    },
    "deltaR_photons": {
      "dependencies": ["lead_eta", "lead_phi", "sublead_eta", "sublead_phi"],
      "calculation": lambda df: df.apply(lambda row: deltaR(row["lead_eta"], row["lead_phi"], row["sublead_eta"], row["sublead_phi"]), axis=1)
    },
    "deltaEta_photons": {
      "dependencies": ["lead_eta", "sublead_eta"],
      "calculation": lambda df: abs(df["lead_eta"] - df["sublead_eta"])
    },
    "deltaR_jets": {
      "dependencies": ["nonRes_lead_bjet_eta", "nonRes_lead_bjet_phi", "nonRes_sublead_bjet_eta", "nonRes_sublead_bjet_phi"],
      "calculation": lambda df: df.apply(lambda row: deltaR(row["nonRes_lead_bjet_eta"], row["nonRes_lead_bjet_phi"], row["nonRes_sublead_bjet_eta"], row["nonRes_sublead_bjet_phi"]), axis=1)
    },
    "deltaEta_jets": {
      "dependencies": ["nonRes_lead_bjet_eta", "nonRes_sublead_bjet_eta"],
      "calculation": lambda df: abs(df["nonRes_lead_bjet_eta"] - df["nonRes_sublead_bjet_eta"])
    },
    "deltaR_gg_jj": {
      "dependencies": ["eta", "phi", "nonRes_dijet_eta", "nonRes_dijet_phi"],
      "calculation": lambda df: df.apply(lambda row: deltaR(row["eta"], row["phi"], row["nonRes_dijet_eta"], row["nonRes_dijet_phi"]), axis=1)
    },
    "deltaR_g1_jj": {
      "dependencies": ["lead_eta", "lead_phi", "nonRes_dijet_eta", "nonRes_dijet_phi"],
      "calculation": lambda df: df.apply(lambda row: deltaR(row["lead_eta"], row["lead_phi"], row["nonRes_dijet_eta"], row["nonRes_dijet_phi"]), axis=1)
    },
    "deltaR_g2_jj": {
      "dependencies": ["sublead_eta", "sublead_phi", "nonRes_dijet_eta", "nonRes_dijet_phi"],
      "calculation": lambda df: df.apply(lambda row: deltaR(row["sublead_eta"], row["sublead_phi"], row["nonRes_dijet_eta"], row["nonRes_dijet_phi"]), axis=1)
    },
    "deltaR_j1_gg": {
      "dependencies": ["nonRes_lead_bjet_eta", "nonRes_lead_bjet_phi", "eta", "phi"],
      "calculation": lambda df: df.apply(lambda row: deltaR(row["nonRes_lead_bjet_eta"], row["nonRes_lead_bjet_phi"], row["eta"], row["phi"]), axis=1)
    },
    "deltaR_j2_gg": {
      "dependencies": ["nonRes_sublead_bjet_eta", "nonRes_sublead_bjet_phi", "eta", "phi"],
      "calculation": lambda df: df.apply(lambda row: deltaR(row["nonRes_sublead_bjet_eta"], row["nonRes_sublead_bjet_phi"], row["eta"], row["phi"]), axis=1)
    },
    "deltaEta_gg_jj": {
      "dependencies": ["eta", "nonRes_dijet_eta"],
      "calculation": lambda df: abs(df["eta"] - df["nonRes_dijet_eta"])
    },
    "deltaEta_g1_g2": {
      "dependencies": ["lead_eta", "sublead_eta"],
      "calculation": lambda df: abs(df["lead_eta"] - df["sublead_eta"])
    },
    "deltaEta_j1_j2": {
      "dependencies": ["nonRes_lead_bjet_eta", "nonRes_sublead_bjet_eta"],
      "calculation": lambda df: abs(df["nonRes_lead_bjet_eta"] - df["nonRes_sublead_bjet_eta"])
    },
    "deltaEta_g1_jj": {
      "dependencies": ["lead_eta", "nonRes_dijet_eta"],
      "calculation": lambda df: abs(df["lead_eta"] - df["nonRes_dijet_eta"])
    },
    "deltaEta_g2_jj": {
      "dependencies": ["sublead_eta", "nonRes_dijet_eta"],
      "calculation": lambda df: abs(df["sublead_eta"] - df["nonRes_dijet_eta"])
    },
    "deltaEta_j1_gg": {
      "dependencies": ["nonRes_lead_bjet_eta", "eta"],
      "calculation": lambda df: abs(df["nonRes_lead_bjet_eta"] - df["eta"])
    },
    "deltaEta_j2_gg": {
      "dependencies": ["nonRes_sublead_bjet_eta", "eta"],
      "calculation": lambda df: abs(df["nonRes_sublead_bjet_eta"] - df["eta"])
    },
    "deltaPhi_gg_jj": {
      "dependencies": ["phi", "nonRes_dijet_phi"],
      "calculation": lambda df: df.apply(lambda row: deltaPhi(row["phi"], row["nonRes_dijet_phi"]), axis=1)
    },
    "deltaPhi_g1_jj": {
      "dependencies": ["lead_phi", "nonRes_dijet_phi"],
      "calculation": lambda df: df.apply(lambda row: deltaPhi(row["lead_phi"], row["nonRes_dijet_phi"]), axis=1)
    },
    "deltaPhi_g2_jj": {
      "dependencies": ["sublead_phi", "nonRes_dijet_phi"],
      "calculation": lambda df: df.apply(lambda row: deltaPhi(row["sublead_phi"], row["nonRes_dijet_phi"]), axis=1)
    },
    "deltaPhi_j1_gg": {
      "dependencies": ["nonRes_lead_bjet_phi", "phi"],
      "calculation": lambda df: df.apply(lambda row: deltaPhi(row["nonRes_lead_bjet_phi"], row["phi"]), axis=1)
    },
    "deltaPhi_j2_gg": {
      "dependencies": ["nonRes_sublead_bjet_phi", "phi"],
      "calculation": lambda df: df.apply(lambda row: deltaPhi(row["nonRes_sublead_bjet_phi"], row["phi"]), axis=1)
    },
    "Max_mvaID": {
      "dependencies": ["lead_mvaID", "sublead_mvaID"],
      "calculation": lambda df: np.max([df["lead_mvaID"], df["sublead_mvaID"]], axis = 0)
    },
    "Min_mvaID": {
      "dependencies": ["lead_mvaID", "sublead_mvaID"],
      "calculation": lambda df: np.min([df["lead_mvaID"], df["sublead_mvaID"]], axis = 0)
    },
    "HHbbggCandidate_ptoverMggjj": {
      "dependencies": ["nonRes_HHbbggCandidate_pt", "nonRes_HHbbggCandidate_mass"],
      "calculation": lambda df: df['nonRes_HHbbggCandidate_pt'] / df['nonRes_HHbbggCandidate_mass']
    },
    "HHbbggCandidate_mass_mXv1": {
      "dependencies": ["nonRes_HHbbggCandidate_mass", "nonRes_dijet_mass_PNet_all", "mass"],
      "calculation": lambda df: df["nonRes_HHbbggCandidate_mass"] - df["nonRes_dijet_mass_PNet_all"] - df["mass"] + 2 * common.HIGGS_MASS
    },
    "HHbbggCandidate_mass_mXv2": {
      "dependencies": ["nonRes_HHbbggCandidate_mass", "nonRes_dijet_mass_PNet_all", "mass"],
      "calculation": lambda df: df["nonRes_HHbbggCandidate_mass"] - df["nonRes_dijet_mass_PNet_all"] + common.HIGGS_MASS
    },
    "puppiMET_sumEtoverMggjj": {
      "dependencies": ["puppiMET_sumEt", "nonRes_HHbbggCandidate_mass"],
      "calculation": lambda df: df['puppiMET_sumEt'] / df['nonRes_HHbbggCandidate_mass']
    },
    "puppiMET_ptoverMggjj": {
      "dependencies": ["puppiMET_pt", "nonRes_HHbbggCandidate_mass"],
      "calculation": lambda df: df['puppiMET_pt'] / df['nonRes_HHbbggCandidate_mass']
    },
    "lead_bjet_btagTight": {
      "dependencies": ["nonRes_lead_bjet_btagPNetB"],
      "calculation": lambda df: (df['nonRes_lead_bjet_btagPNetB'] > 0.6915).astype(int)
    },
    "sublead_bjet_btagTight": {
      "dependencies": ["nonRes_sublead_bjet_btagPNetB"],
      "calculation": lambda df: (df['nonRes_sublead_bjet_btagPNetB'] > 0.6915).astype(int)
    },
    "lead_bjet_btagMedium": {
      "dependencies": ["nonRes_lead_bjet_btagPNetB"],
      "calculation": lambda df: (df['nonRes_lead_bjet_btagPNetB'] > 0.26).astype(int)
    },
    "sublead_bjet_btagMedium": {
      "dependencies": ["nonRes_sublead_bjet_btagPNetB"],
      "calculation": lambda df: (df['nonRes_sublead_bjet_btagPNetB'] > 0.26).astype(int)
    },
    "lead_bjet_btagLoose": {
      "dependencies": ["nonRes_lead_bjet_btagPNetB"],
      "calculation": lambda df: (df['nonRes_lead_bjet_btagPNetB'] > 0.0499).astype(int)
    },
    "sublead_bjet_btagLoose": {
      "dependencies": ["nonRes_sublead_bjet_btagPNetB"],
      "calculation": lambda df: (df['nonRes_sublead_bjet_btagPNetB'] > 0.0499).astype(int)
    },
    "lead_MVAID_WP80": {
      "dependencies": ["lead_isScEtaEB", 'lead_mvaID', "lead_isScEtaEE"],
      "calculation": lambda df: (((df["lead_isScEtaEB"] == True) & (df['lead_mvaID'] > 0.420473)) | ((df["lead_isScEtaEE"] == True) & (df['lead_mvaID'] > 0.203451))).astype(int)
    },
    "sublead_MVAID_WP80": {
      "dependencies": ["sublead_isScEtaEB", 'sublead_mvaID', "sublead_isScEtaEE"],
      "calculation": lambda df: (((df["sublead_isScEtaEB"] == True) & (df['sublead_mvaID'] > 0.420473)) | ((df["sublead_isScEtaEE"] == True) & (df['sublead_mvaID'] > 0.203451))).astype(int)
    },
    "lead_MVAID_WP90": {
      "dependencies": ["lead_isScEtaEB", 'lead_mvaID', "lead_isScEtaEE"],
      "calculation": lambda df: (((df["lead_isScEtaEB"] == True) & (df['lead_mvaID'] > 0.0439603)) | ((df["lead_isScEtaEE"] == True) & (df['lead_mvaID'] > -0.249526))).astype(int)
    },
    "sublead_MVAID_WP90": {
      "dependencies": ["sublead_isScEtaEB", 'sublead_mvaID', "sublead_isScEtaEE"],
      "calculation": lambda df: (((df["sublead_isScEtaEB"] == True) & (df['sublead_mvaID'] > 0.0439603)) | ((df["sublead_isScEtaEE"] == True) & (df['sublead_mvaID'] > -0.249526))).astype(int)
    },
    # NW variables
    "M_chi": {
      "dependencies": ["nonRes_HHbbggCandidate_mass", 'mass', "nonRes_dijet_mass"],
      "calculation": lambda df: df['nonRes_HHbbggCandidate_mass'] - df['mass'] - df['nonRes_dijet_mass'] + 2*124.9
    },
    "CosThetaStar_CS_abs": {
      "dependencies": ["nonRes_CosThetaStar_CS"],
      "calculation": lambda df: np.abs(df['nonRes_CosThetaStar_CS'])
    },
    "CosThetaStar_gg_abs": {
      "dependencies": ["nonRes_CosThetaStar_gg"],
      "calculation": lambda df: np.abs(df['nonRes_CosThetaStar_gg'])
    },
    "CosThetaStar_jj_abs": {
      "dependencies": ["nonRes_CosThetaStar_jj"],
      "calculation": lambda df: np.abs(df['nonRes_CosThetaStar_jj'])
    },
    "HHbbggCandidate_eta_abs": {
      "dependencies": ["nonRes_HHbbggCandidate_eta"],
      "calculation": lambda df: np.abs(df['nonRes_HHbbggCandidate_eta'])
    },
    "lead_eta_abs": {
      "dependencies": ["lead_eta"],
      "calculation": lambda df: np.abs(df['lead_eta'])
    },
    "sublead_eta_abs": {
      "dependencies": ["sublead_eta"],
      "calculation": lambda df: np.abs(df['sublead_eta'])
    },
    "eta_abs": {
      "dependencies": ["eta"],
      "calculation": lambda df: np.abs(df['eta'])
    },
    "lead_bjet_eta_abs": {
      "dependencies": ["nonRes_lead_bjet_eta"],
      "calculation": lambda df: np.abs(df['nonRes_lead_bjet_eta'])
    },
    "sublead_bjet_eta_abs": {
      "dependencies": ["nonRes_sublead_bjet_eta"],
      "calculation": lambda df: np.abs(df['nonRes_sublead_bjet_eta'])
    },
    "dijet_eta_abs": {
      "dependencies": ["nonRes_dijet_eta"],
      "calculation": lambda df: np.abs(df['nonRes_dijet_eta'])
    },
    "gg_pT_OverHHcand_mass": {
      "dependencies": ["pt", "nonRes_HHbbggCandidate_mass"],
      "calculation": lambda df: df['pt'] / df['nonRes_HHbbggCandidate_mass']
    },
    "jj_pT_OverHHcand_mass": {
      "dependencies": ["dijet_pt", "nonRes_HHbbggCandidate_mass"],
      "calculation": lambda df: df['nonRes_dijet_pt'] / df['nonRes_HHbbggCandidate_mass']
    },
    "lead_g_pT_OverHggcand_mass": {
      "dependencies": ["lead_pt", "nonRes_HHbbggCandidate_mass"],
      "calculation": lambda df: df['lead_pt'] / df['mass']
    },
    "lead_j_pT_OverHbbcand_mass": {
      "dependencies": ["nonRes_lead_bjet_pt", "nonRes_dijet_mass"],
      "calculation": lambda df: df['nonRes_lead_bjet_pt'] / df['nonRes_dijet_mass']
    },
    "sublead_g_pT_OverHggcand_mass": {
      "dependencies": ["sublead_pt", "nonRes_HHbbggCandidate_mass"],
      "calculation": lambda df: df['sublead_pt'] / df['mass']
    },
    "sublead_j_pT_OverHbbcand_mass": {
      "dependencies": ["nonRes_sublead_bjet_pt", "nonRes_dijet_mass"],
      "calculation": lambda df: df['nonRes_sublead_bjet_pt'] / df['nonRes_dijet_mass']
    }
  }
  for col in df_columns:
    if "VBF" in col or "fatjet" in col:
        skimmedVariables.append(col)
  # Loop over all input variables
  for var in input_variables:
    non_res_var = "nonRes_" + var
    res_var = "Res_" + var
    if var in df_columns:
      skimmedVariables.append(var)
      continue
    if non_res_var in df_columns:
      skimmedVariables.append(non_res_var)
      continue
    if res_var in df_columns:
      skimmedVariables.append(res_var)
      continue
    if var in variable_definitions:
      dependencies = variable_definitions[var]["dependencies"]
      if df is not None:
        # Calculate the variable in the DataFrame
        df[var] = variable_definitions[var]["calculation"](df)
      else:
        # Add dependencies if the DataFrame is not provided
        for dep in dependencies:
          if dep not in skimmedVariables:
            skimmedVariables.append(dep)
    else:
      print(f"Variable {var} not recognized, please add it in compositeVariables.")

  # Return either the list of dependencies or the dataframe with the new variables.
  if df is None:
    return skimmedVariables
  else:
    return df
