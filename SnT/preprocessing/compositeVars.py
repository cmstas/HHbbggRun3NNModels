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
    "lead_bjet_massOverMjj": {
      "dependencies": ["lead_bjet_mass", "dijet_mass"],
      "calculation": lambda df: df['lead_bjet_mass'] / df['dijet_mass']
    },
    "sublead_bjet_massOverMjj": {
      "dependencies": ["sublead_bjet_mass", "dijet_mass"],
      "calculation": lambda df: df['sublead_bjet_mass'] / df['dijet_mass']
    },
    "dipho_PtOverMggjj": {
      "dependencies": ["pt", "HHbbggCandidate_mass"],
      "calculation": lambda df: df['pt'] / df['HHbbggCandidate_mass']
    },
    "dipho_PtOverMgg": {
      "dependencies": ["pt", "mass"],
      "calculation": lambda df: df['pt'] / df['mass']
    },
    "dijet_PtOverMggjj": {
      "dependencies": ["dijet_pt", "HHbbggCandidate_mass"],
      "calculation": lambda df: df['dijet_pt'] / df['HHbbggCandidate_mass']
    },
    "dijet_PtOverMjj": {
      "dependencies": ["dijet_pt", "dijet_mass"],
      "calculation": lambda df: df['dijet_pt'] / df['dijet_mass']
    },
    "dijet_massOverMggjj": {
      "dependencies": ["dijet_mass", "HHbbggCandidate_mass"],
      "calculation": lambda df: df['dijet_mass'] / df['HHbbggCandidate_mass']
    },
    "deltaPhi_g1_g2": {
      "dependencies": ['sublead_phi', 'lead_phi'],
      "calculation": lambda df: df.apply(lambda row: deltaPhi(row["sublead_phi"], row["lead_phi"]), axis=1)
    },
    "deltaPhi_g1_j1": {
      "dependencies": ['lead_bjet_phi', 'lead_phi'],
      "calculation": lambda df: df.apply(lambda row: deltaPhi(row["lead_bjet_phi"], row["lead_phi"]), axis=1)
    },
    "deltaPhi_g1_j2": {
      "dependencies": ['sublead_bjet_phi', 'lead_phi'],
      "calculation": lambda df: df.apply(lambda row: deltaPhi(row["sublead_bjet_phi"], row["lead_phi"]), axis=1)
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
      "dependencies": ["lead_bjet_eta", "lead_eta"],
      "calculation": lambda df: df["lead_bjet_eta"] * np.sign(df["lead_eta"])
    },
    "sublead_bjet_eta_mirrored": {
      "dependencies": ["sublead_bjet_eta", "lead_eta"],
      "calculation": lambda df: df["sublead_bjet_eta"] * np.sign(df["lead_eta"])
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
      "dependencies": ["lead_bjet_eta", "lead_bjet_phi", "sublead_bjet_eta", "sublead_bjet_phi"],
      "calculation": lambda df: df.apply(lambda row: deltaR(row["lead_bjet_eta"], row["lead_bjet_phi"], row["sublead_bjet_eta"], row["sublead_bjet_phi"]), axis=1)
    },
    "deltaEta_jets": {
      "dependencies": ["lead_bjet_eta", "sublead_bjet_eta"],
      "calculation": lambda df: abs(df["lead_bjet_eta"] - df["sublead_bjet_eta"])
    },
    "deltaR_gg_jj": {
      "dependencies": ["eta", "phi", "dijet_eta", "dijet_phi"],
      "calculation": lambda df: df.apply(lambda row: deltaR(row["eta"], row["phi"], row["dijet_eta"], row["dijet_phi"]), axis=1)
    },
    "deltaR_g1_jj": {
      "dependencies": ["lead_eta", "lead_phi", "dijet_eta", "dijet_phi"],
      "calculation": lambda df: df.apply(lambda row: deltaR(row["lead_eta"], row["lead_phi"], row["dijet_eta"], row["dijet_phi"]), axis=1)
    },
    "deltaR_g2_jj": {
      "dependencies": ["sublead_eta", "sublead_phi", "dijet_eta", "dijet_phi"],
      "calculation": lambda df: df.apply(lambda row: deltaR(row["sublead_eta"], row["sublead_phi"], row["dijet_eta"], row["dijet_phi"]), axis=1)
    },
    "deltaR_j1_gg": {
      "dependencies": ["lead_bjet_eta", "lead_bjet_phi", "eta", "phi"],
      "calculation": lambda df: df.apply(lambda row: deltaR(row["lead_bjet_eta"], row["lead_bjet_phi"], row["eta"], row["phi"]), axis=1)
    },
    "deltaR_j2_gg": {
      "dependencies": ["sublead_bjet_eta", "sublead_bjet_phi", "eta", "phi"],
      "calculation": lambda df: df.apply(lambda row: deltaR(row["sublead_bjet_eta"], row["sublead_bjet_phi"], row["eta"], row["phi"]), axis=1)
    },
    "deltaEta_gg_jj": {
      "dependencies": ["eta", "dijet_eta"],
      "calculation": lambda df: abs(df["eta"] - df["dijet_eta"])
    },
    "deltaEta_g1_g2": {
      "dependencies": ["lead_eta", "sublead_eta"],
      "calculation": lambda df: abs(df["lead_eta"] - df["sublead_eta"])
    },
    "deltaEta_j1_j2": {
      "dependencies": ["lead_bjet_eta", "sublead_bjet_eta"],
      "calculation": lambda df: abs(df["lead_bjet_eta"] - df["sublead_bjet_eta"])
    },
    "deltaEta_g1_jj": {
      "dependencies": ["lead_eta", "dijet_eta"],
      "calculation": lambda df: abs(df["lead_eta"] - df["dijet_eta"])
    },
    "deltaEta_g2_jj": {
      "dependencies": ["sublead_eta", "dijet_eta"],
      "calculation": lambda df: abs(df["sublead_eta"] - df["dijet_eta"])
    },
    "deltaEta_j1_gg": {
      "dependencies": ["lead_bjet_eta", "eta"],
      "calculation": lambda df: abs(df["lead_bjet_eta"] - df["eta"])
    },
    "deltaEta_j2_gg": {
      "dependencies": ["sublead_bjet_eta", "eta"],
      "calculation": lambda df: abs(df["sublead_bjet_eta"] - df["eta"])
    },
    "deltaPhi_gg_jj": {
      "dependencies": ["phi", "dijet_phi"],
      "calculation": lambda df: df.apply(lambda row: deltaPhi(row["phi"], row["dijet_phi"]), axis=1)
    },
    "deltaPhi_g1_jj": {
      "dependencies": ["lead_phi", "dijet_phi"],
      "calculation": lambda df: df.apply(lambda row: deltaPhi(row["lead_phi"], row["dijet_phi"]), axis=1)
    },
    "deltaPhi_g2_jj": {
      "dependencies": ["sublead_phi", "dijet_phi"],
      "calculation": lambda df: df.apply(lambda row: deltaPhi(row["sublead_phi"], row["dijet_phi"]), axis=1)
    },
    "deltaPhi_j1_gg": {
      "dependencies": ["lead_bjet_phi", "phi"],
      "calculation": lambda df: df.apply(lambda row: deltaPhi(row["lead_bjet_phi"], row["phi"]), axis=1)
    },
    "deltaPhi_j2_gg": {
      "dependencies": ["sublead_bjet_phi", "phi"],
      "calculation": lambda df: df.apply(lambda row: deltaPhi(row["sublead_bjet_phi"], row["phi"]), axis=1)
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
      "dependencies": ["HHbbggCandidate_pt", "HHbbggCandidate_mass"],
      "calculation": lambda df: df['HHbbggCandidate_pt'] / df['HHbbggCandidate_mass']
    },
    "HHbbggCandidate_mass_mXv1": {
      "dependencies": ["HHbbggCandidate_mass", "dijet_mass", "mass"],
      "calculation": lambda df: df["HHbbggCandidate_mass"] - df["dijet_mass"] - df["mass"] + 2 * common.HIGGS_MASS
    },
    "HHbbggCandidate_mass_mXv2": {
      "dependencies": ["HHbbggCandidate_mass", "dijet_mass", "mass"],
      "calculation": lambda df: df["HHbbggCandidate_mass"] - df["dijet_mass"] + common.HIGGS_MASS
    },
    "puppiMET_sumEtoverMggjj": {
      "dependencies": ["puppiMET_sumEt", "HHbbggCandidate_mass"],
      "calculation": lambda df: df['puppiMET_sumEt'] / df['HHbbggCandidate_mass']
    },
    "puppiMET_ptoverMggjj": {
      "dependencies": ["puppiMET_pt", "HHbbggCandidate_mass"],
      "calculation": lambda df: df['puppiMET_pt'] / df['HHbbggCandidate_mass']
    },
    "lead_bjet_btagTight": {
      "dependencies": ["lead_bjet_btagPNetB"],
      "calculation": lambda df: (df['lead_bjet_btagPNetB'] > 0.6915).astype(int)
    },
    "sublead_bjet_btagTight": {
      "dependencies": ["sublead_bjet_btagPNetB"],
      "calculation": lambda df: (df['sublead_bjet_btagPNetB'] > 0.6915).astype(int)
    },
    "lead_bjet_btagMedium": {
      "dependencies": ["lead_bjet_btagPNetB"],
      "calculation": lambda df: (df['lead_bjet_btagPNetB'] > 0.26).astype(int)
    },
    "sublead_bjet_btagMedium": {
      "dependencies": ["sublead_bjet_btagPNetB"],
      "calculation": lambda df: (df['sublead_bjet_btagPNetB'] > 0.26).astype(int)
    },
    "lead_bjet_btagLoose": {
      "dependencies": ["lead_bjet_btagPNetB"],
      "calculation": lambda df: (df['lead_bjet_btagPNetB'] > 0.0499).astype(int)
    },
    "sublead_bjet_btagLoose": {
      "dependencies": ["sublead_bjet_btagPNetB"],
      "calculation": lambda df: (df['sublead_bjet_btagPNetB'] > 0.0499).astype(int)
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
      "dependencies": ["HHbbggCandidate_mass", 'mass', "dijet_mass"],
      "calculation": lambda df: df['HHbbggCandidate_mass'] - df['mass'] - df['dijet_mass'] + 2*124.9
    },
    "CosThetaStar_CS_abs": {
      "dependencies": ["CosThetaStar_CS"],
      "calculation": lambda df: np.abs(df['CosThetaStar_CS'])
    },
    "CosThetaStar_gg_abs": {
      "dependencies": ["CosThetaStar_gg"],
      "calculation": lambda df: np.abs(df['CosThetaStar_gg'])
    },
    "CosThetaStar_jj_abs": {
      "dependencies": ["CosThetaStar_jj"],
      "calculation": lambda df: np.abs(df['CosThetaStar_jj'])
    },
    "HHbbggCandidate_eta_abs": {
      "dependencies": ["HHbbggCandidate_eta"],
      "calculation": lambda df: np.abs(df['HHbbggCandidate_eta'])
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
      "dependencies": ["lead_bjet_eta"],
      "calculation": lambda df: np.abs(df['lead_bjet_eta'])
    },
    "sublead_bjet_eta_abs": {
      "dependencies": ["sublead_bjet_eta"],
      "calculation": lambda df: np.abs(df['sublead_bjet_eta'])
    },
    "dijet_eta_abs": {
      "dependencies": ["dijet_eta"],
      "calculation": lambda df: np.abs(df['dijet_eta'])
    },
    "gg_pT_OverHHcand_mass": {
      "dependencies": ["pt", "HHbbggCandidate_mass"],
      "calculation": lambda df: df['pt'] / df['HHbbggCandidate_mass']
    },
    "jj_pT_OverHHcand_mass": {
      "dependencies": ["dijet_pt", "HHbbggCandidate_mass"],
      "calculation": lambda df: df['dijet_pt'] / df['HHbbggCandidate_mass']
    },
    "lead_g_pT_OverHggcand_mass": {
      "dependencies": ["lead_pt", "HHbbggCandidate_mass"],
      "calculation": lambda df: df['lead_pt'] / df['mass']
    },
    "lead_j_pT_OverHbbcand_mass": {
      "dependencies": ["lead_bjet_pt", "dijet_mass"],
      "calculation": lambda df: df['lead_bjet_pt'] / df['dijet_mass']
    },
    "sublead_g_pT_OverHggcand_mass": {
      "dependencies": ["sublead_pt", "HHbbggCandidate_mass"],
      "calculation": lambda df: df['sublead_pt'] / df['mass']
    },
    "sublead_j_pT_OverHbbcand_mass": {
      "dependencies": ["sublead_bjet_pt", "dijet_mass"],
      "calculation": lambda df: df['sublead_bjet_pt'] / df['dijet_mass']
    }
  }

  # Loop over all input variables
  for var in input_variables:
    if var in df_columns:
      skimmedVariables.append(var)
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
