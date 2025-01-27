import pandas as pd
import argparse
from ROOT.Math import PtEtaPhiMVector, Polar2DVector
import ROOT

placeholder_val = -999

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
num_stored_jets = len(
    pq[
        pq.columns[
            (pq.columns.str.startswith("jet")) & (pq.columns.str.endswith("_pt"))
        ]
    ].columns
)

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

met_pnet_vis = []
met_pnet_all = []
dijet_pt_PNet_vis = []
dijet_pt_PNet_all = []
dijet_mass_PNet_vis = []
dijet_mass_PNet_all = []

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
):
    if lead_pt == placeholder_val or sublead_pt == placeholder_val:
        met_pnet_vis.append(placeholder_val)
        met_pnet_all.append(placeholder_val)
        dijet_mass_PNet_vis.append(placeholder_val)
        dijet_mass_PNet_all.append(placeholder_val)
        dijet_pt_PNet_vis.append(placeholder_val)
        dijet_pt_PNet_all.append(placeholder_val)
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

pq["MET_PNet_vis"] = met_pnet_vis
pq["MET_PNet_all"] = met_pnet_all
pq[an_prefix + "dijet_mass_PNet_vis"] = dijet_mass_PNet_vis
pq[an_prefix + "dijet_mass_PNet_all"] = dijet_mass_PNet_all
pq[an_prefix + "dijet_pt_PNet_vis"] = dijet_pt_PNet_vis
pq[an_prefix + "dijet_pt_PNet_all"] = dijet_pt_PNet_all

pq.to_parquet(args.input.removesuffix(".parquet") + "_with_full_PNet_info.parquet")
