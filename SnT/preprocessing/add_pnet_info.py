import pandas as pd
import argparse
from ROOT.Math import PtEtaPhiMVector, Polar2DVector
import ROOT

placeholder_val = -999  # value assigned to erroneous/undefined calculations


def cross2D(v1, v2):
    return v1.x() * v2.y() - v1.y() * v2.x()


# Use MET to obtain an ad-hoc correction of the jet pt/mass
# [met], [lead_jet], [sublead_jet] are transverse 2D vectors (eta = 0)
# Returns scale factor associated with this ad-hoc correction
# SHOULD NOT BE TRUSTED since this decomposition often fails since
# there is generally more than 2 jets/other components to the MET that are
# unaccounted for
def get_sf(met, lead_jet, sublead_jet):
    # decompose met into: met = alpha * lead_jet + beta * sublead_jet
    if cross2D(sublead_jet, lead_jet) == 0:
        return 1, 1, False
    alpha = cross2D(sublead_jet, met) / cross2D(sublead_jet, lead_jet)
    beta = cross2D(lead_jet, met) / cross2D(lead_jet, sublead_jet)

    return 1 + alpha, 1 + beta, abs(alpha) < 0.5 and abs(beta) < 0.5


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True, help="Path to input parquet", type=str)
parser.add_argument(
    "--AN-type-prefix",
    required=False,
    help="'nonRes' or 'Res' for v2 parquets; '' (blank) for v1 parquets",
    default="nonRes",
    type=str,
)
args = parser.parse_args()

an_prefix = args.AN_type_prefix + "_" if args.AN_type_prefix else ""

pq = pd.read_parquet(args.input)
# number of stored jets -- this is 10 for HiggsDNAv2
num_stored_jets = len(
    pq[
        pq.columns[
            (pq.columns.str.startswith("jet")) & (pq.columns.str.endswith("_pt"))
        ]
    ].columns
)

# Add PNet info for individual jets
for i in range(1, num_stored_jets + 1):
    pq[f"jet{i}_PNet_sf_vis"] = (
        (1 - pq[f"jet{i}_rawFactor"]) * pq[f"jet{i}_PNetRegPtRawCorr"]
    ).where(pq[f"jet{i}_pt"] != placeholder_val, placeholder_val)

    pq[f"jet{i}_pt_PNet_vis"] = (pq[f"jet{i}_pt"] * pq[f"jet{i}_PNet_sf_vis"]).where(
        pq[f"jet{i}_pt"] != placeholder_val, placeholder_val
    )

    pq[f"jet{i}_mass_PNet_vis"] = (
        pq[f"jet{i}_mass"] * pq[f"jet{i}_PNet_sf_vis"]
    ).where(pq[f"jet{i}_pt"] != placeholder_val, placeholder_val)

    pq[f"jet{i}_PNet_sf_all"] = (
        (1 - pq[f"jet{i}_rawFactor"])
        * pq[f"jet{i}_PNetRegPtRawCorr"]
        * pq[f"jet{i}_PNetRegPtRawCorrNeutrino"]
    ).where(pq[f"jet{i}_pt"] != placeholder_val, placeholder_val)

    pq[f"jet{i}_pt_PNet_all"] = (pq[f"jet{i}_pt"] * pq[f"jet{i}_PNet_sf_all"]).where(
        pq[f"jet{i}_pt"] != placeholder_val, placeholder_val
    )

    pq[f"jet{i}_mass_PNet_all"] = (
        pq[f"jet{i}_mass"] * pq[f"jet{i}_PNet_sf_all"]
    ).where(pq[f"jet{i}_pt"] != placeholder_val, placeholder_val)

# Add PNet info for selected lead and sublead jets
for prefix in ["lead", "sublead"]:
    prefix = an_prefix + prefix

    pq[f"{prefix}_bjet_PNet_sf_vis"] = (
        (1 - pq[f"{prefix}_bjet_rawFactor"]) * pq[f"{prefix}_bjet_PNetRegPtRawCorr"]
    ).where(pq[f"{prefix}_bjet_pt"] != placeholder_val, placeholder_val)

    pq[f"{prefix}_bjet_pt_PNet_vis"] = (
        pq[f"{prefix}_bjet_pt"] * pq[f"{prefix}_bjet_PNet_sf_vis"]
    ).where(pq[f"{prefix}_bjet_pt"] != placeholder_val, placeholder_val)

    pq[f"{prefix}_bjet_mass_PNet_vis"] = (
        pq[f"{prefix}_bjet_mass"] * pq[f"{prefix}_bjet_PNet_sf_vis"]
    ).where(pq[f"{prefix}_bjet_pt"] != placeholder_val, placeholder_val)

    pq[f"{prefix}_bjet_PNet_sf_all"] = (
        (1 - pq[f"{prefix}_bjet_rawFactor"])
        * pq[f"{prefix}_bjet_PNetRegPtRawCorr"]
        * pq[f"{prefix}_bjet_PNetRegPtRawCorrNeutrino"]
    ).where(pq[f"{prefix}_bjet_pt"] != placeholder_val, placeholder_val)

    pq[f"{prefix}_bjet_pt_PNet_all"] = (
        pq[f"{prefix}_bjet_pt"] * pq[f"{prefix}_bjet_PNet_sf_all"]
    ).where(pq[f"{prefix}_bjet_pt"] != placeholder_val, placeholder_val)

    pq[f"{prefix}_bjet_mass_PNet_all"] = (
        pq[f"{prefix}_bjet_mass"] * pq[f"{prefix}_bjet_PNet_sf_all"]
    ).where(pq[f"{prefix}_bjet_pt"] != placeholder_val, placeholder_val)

###### Add additional columns derived from PNet information
# rough type-1 PNet MET obtained by removing lead+sublead PuppiJets and inserting lead+sublead PNet jets (with visible PNet corr only)
met_pnet_vis = []

# rough type-1 PNet MET obtained by removing lead+sublead PuppiJets and inserting lead+sublead PNet jets (full visible + neutrino corrections)
met_pnet_all = []

# dijet pt of lead+sublead PNet jets (with visible PNet corr only)
dijet_pt_PNet_vis = []

# dijet pt of lead+sublead PNet jets (full visible + neutrino corrections)
dijet_pt_PNet_all = []

# dijet mass of lead+sublead PNet jets (with visible PNet corr only)
dijet_mass_PNet_vis = []

# dijet mass of lead+sublead PNet jets (full visible + neutrino corrections)
dijet_mass_PNet_all = []

# dijet mass of lead+sublead PNet jets (full visible + neutrino corrections applied to jet pt only, mass left the same as given by Puppi)
dijet_mass_PNet_mass_uncorr = []

# dijet mass of lead+sublead PNet jets (full visible + neutrino corrections on jets, with ad-hoc correction using rough type-1 PNet MET projection)
dijet_mass_PNet_all_with_MET_corr = []

# dijet_pt_PNet_all / (mass of bbgg calculated using PNet jets with full visible+neutrino corrections applied)
dijet_pt_PNet_all_normalized = []

# dijet_pt_PNet_all_with_MET_corr / (mass of bbgg calculated using PNet jets with full visible+neutrino corrections and ad-hoc type-1 PNet MET correction applied)
dijet_pt_PNet_all_with_MET_corr_normalized = []

for (
    lead_pt,
    lead_eta,
    lead_phi,
    lead_mass,
    lead_pt_PNet_vis,
    lead_pt_PNet_all,
    lead_mass_PNet_vis,
    lead_mass_PNet_all,
    sublead_pt,
    sublead_eta,
    sublead_phi,
    sublead_mass,
    sublead_pt_PNet_vis,
    sublead_pt_PNet_all,
    sublead_mass_PNet_vis,
    sublead_mass_PNet_all,
    puppi_pt,
    puppi_phi,
    puppi_sumEt,
    njets,
    lead_pho_pt,
    lead_pho_eta,
    lead_pho_phi,
    sublead_pho_pt,
    sublead_pho_eta,
    sublead_pho_phi,
) in zip(
    pq[an_prefix + "lead_bjet_pt"],
    pq[an_prefix + "lead_bjet_eta"],
    pq[an_prefix + "lead_bjet_phi"],
    pq[an_prefix + "lead_bjet_mass"],
    pq[an_prefix + "lead_bjet_pt_PNet_vis"],
    pq[an_prefix + "lead_bjet_pt_PNet_all"],
    pq[an_prefix + "lead_bjet_mass_PNet_vis"],
    pq[an_prefix + "lead_bjet_mass_PNet_all"],
    pq[an_prefix + "sublead_bjet_pt"],
    pq[an_prefix + "sublead_bjet_eta"],
    pq[an_prefix + "sublead_bjet_phi"],
    pq[an_prefix + "sublead_bjet_mass"],
    pq[an_prefix + "sublead_bjet_pt_PNet_vis"],
    pq[an_prefix + "sublead_bjet_pt_PNet_all"],
    pq[an_prefix + "sublead_bjet_mass_PNet_vis"],
    pq[an_prefix + "sublead_bjet_mass_PNet_all"],
    pq["puppiMET_pt"],
    pq["puppiMET_phi"],
    pq["puppiMET_sumEt"],
    pq["n_jets"],
    pq["lead_pt"],
    pq["lead_eta"],
    pq["lead_phi"],
    pq["sublead_pt"],
    pq["sublead_eta"],
    pq["sublead_phi"],
):
    if lead_pt == placeholder_val or sublead_pt == placeholder_val:
        met_pnet_vis.append(placeholder_val)
        met_pnet_all.append(placeholder_val)
        dijet_mass_PNet_vis.append(placeholder_val)
        dijet_mass_PNet_all.append(placeholder_val)
        dijet_pt_PNet_vis.append(placeholder_val)
        dijet_pt_PNet_all.append(placeholder_val)
        dijet_mass_PNet_mass_uncorr.append(placeholder_val)
        dijet_mass_PNet_all_with_MET_corr.append(placeholder_val)
        dijet_pt_PNet_all_normalized.append(placeholder_val)
        dijet_pt_PNet_all_with_MET_corr_normalized.append(placeholder_val)
        continue

    met_2D = Polar2DVector(puppi_pt, puppi_phi)

    lead_2D = Polar2DVector(lead_pt, lead_phi)
    sublead_2D = Polar2DVector(sublead_pt, sublead_phi)

    lead_PNet_2D_vis = Polar2DVector(lead_pt_PNet_vis, lead_phi)
    sublead_PNet_2D_vis = Polar2DVector(sublead_pt_PNet_vis, sublead_phi)

    met_PNet_2D_vis = (
        met_2D + lead_2D + sublead_2D - (lead_PNet_2D_vis + sublead_PNet_2D_vis)
    )
    met_pnet_vis.append(met_PNet_2D_vis.R())

    lead_PNet_2D_all = Polar2DVector(lead_pt_PNet_all, lead_phi)
    sublead_PNet_2D_all = Polar2DVector(sublead_pt_PNet_all, sublead_phi)

    met_PNet_2D_all = (
        met_2D + lead_2D + sublead_2D - (lead_PNet_2D_all + sublead_PNet_2D_all)
    )
    met_pnet_all.append(met_PNet_2D_all.R())

    lead_PNet_vis = PtEtaPhiMVector(
        lead_pt_PNet_vis, lead_eta, lead_phi, lead_mass_PNet_vis
    )
    sublead_PNet_vis = PtEtaPhiMVector(
        sublead_pt_PNet_vis, sublead_eta, sublead_phi, sublead_mass_PNet_vis
    )
    dijet_mass_PNet_vis.append((lead_PNet_vis + sublead_PNet_vis).M())
    dijet_pt_PNet_vis.append((lead_PNet_vis + sublead_PNet_vis).Pt())

    lead_PNet_all = PtEtaPhiMVector(
        lead_pt_PNet_all,
        lead_eta,
        lead_phi,
        lead_mass_PNet_all,
    )
    sublead_PNet_all = PtEtaPhiMVector(
        sublead_pt_PNet_all,
        sublead_eta,
        sublead_phi,
        sublead_mass_PNet_all,
    )
    dijet_mass_PNet_all.append((lead_PNet_all + sublead_PNet_all).M())
    dijet_pt_PNet_all.append((lead_PNet_all + sublead_PNet_all).Pt())

    lead_PNet_mass_uncorr = PtEtaPhiMVector(
        lead_pt_PNet_all,
        lead_eta,
        lead_phi,
        lead_mass,
    )
    sublead_PNet_mass_uncorr = PtEtaPhiMVector(
        sublead_pt_PNet_all,
        sublead_eta,
        sublead_phi,
        sublead_mass,
    )
    dijet_mass_PNet_mass_uncorr.append(
        (lead_PNet_mass_uncorr + sublead_PNet_mass_uncorr).M()
    )

    # lead/sublead PNet jets scale factor obtained by subtracting off MET projections
    lead_PNet_all_MET_sf, sublead_PNet_all_MET_sf, success = get_sf(
        met_PNet_2D_all, lead_PNet_2D_all, sublead_PNet_2D_all
    )

    lead_PNet_all_with_MET_corr = PtEtaPhiMVector(
        lead_pt_PNet_all * lead_PNet_all_MET_sf,
        lead_eta,
        lead_phi,
        lead_mass_PNet_all * lead_PNet_all_MET_sf,
    )
    sublead_PNet_all_with_MET_corr = PtEtaPhiMVector(
        sublead_pt_PNet_all * sublead_PNet_all_MET_sf,
        sublead_eta,
        sublead_phi,
        sublead_mass_PNet_all * sublead_PNet_all_MET_sf,
    )
    dijet_mass_PNet_all_with_MET_corr.append(
        (lead_PNet_all_with_MET_corr + sublead_PNet_all_with_MET_corr).M()
        if success
        else placeholder_val
    )

    lead_photon = PtEtaPhiMVector(lead_pho_pt, lead_pho_eta, lead_pho_phi, 0)
    sublead_photon = PtEtaPhiMVector(
        sublead_pho_pt, sublead_pho_eta, sublead_pho_phi, 0
    )
    dijet_pt_PNet_all_normalized.append(
        (lead_PNet_all + sublead_PNet_all).Pt()
        / (lead_PNet_all + sublead_PNet_all + lead_photon + sublead_photon).M()
    )
    dijet_pt_PNet_all_with_MET_corr_normalized.append(
        (lead_PNet_all_with_MET_corr + sublead_PNet_all_with_MET_corr).Pt()
        / (
            lead_PNet_all_with_MET_corr
            + sublead_PNet_all_with_MET_corr
            + lead_photon
            + sublead_photon
        ).M()
        if success
        else placeholder_val
    )

pq["MET_PNet_vis"] = met_pnet_vis
pq["MET_PNet_all"] = met_pnet_all
pq[an_prefix + "dijet_mass_PNet_vis"] = dijet_mass_PNet_vis
pq[an_prefix + "dijet_mass_PNet_all"] = dijet_mass_PNet_all
pq[an_prefix + "dijet_pt_PNet_vis"] = dijet_pt_PNet_vis
pq[an_prefix + "dijet_pt_PNet_all"] = dijet_pt_PNet_all
pq[an_prefix + "dijet_mass_PNet_mass_uncorr"] = dijet_mass_PNet_mass_uncorr
pq[an_prefix + "dijet_mass_PNet_all_with_MET_corr"] = dijet_mass_PNet_all_with_MET_corr
pq[an_prefix + "dijet_pt_PNet_all_normalized"] = dijet_pt_PNet_all_normalized
pq[an_prefix + "dijet_pt_PNet_all_with_MET_corr_normalized"] = (
    dijet_pt_PNet_all_with_MET_corr_normalized
)

pq.to_parquet(args.input.removesuffix(".parquet") + "_with_full_PNet_info.parquet")
