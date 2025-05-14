# Resolved 2D fit, use https://gitlab.cern.ch/hhtobbgg_nu/flashggFinalFit/-/tree/higgsdnafinalfit?ref_type=heads
# cmsrel CMSSW_14_1_0_pre4
# pip3 install pyarrow

# In the config file you need to update
# Save path for original parquet files
#original_parquet_save_path: /eos/user/b/bdanzi/EarlyRun3HHbbgg/2D/CMSSW_14_1_0_pre4/src/flashggFinalFit/Parquet/2DFit_Resolved
# Save path for unfinished parquet and ROOT files during execution
#script_working_save_path: /eos/user/b/bdanzi/EarlyRun3HHbbgg/2D/CMSSW_14_1_0_pre4/src/flashggFinalFit/Parquet/2DFit_Resolved/intermediateFiles
# Save path for finished parquet files ready for Trees2WS
#finished_root_save_path: /eos/user/b/bdanzi/EarlyRun3HHbbgg/2D/CMSSW_14_1_0_pre4/src/flashggFinalFit/Parquet/2DFit_Resolved/rootFiles
# Additional save path for finished parquet files ready for Trees2WS
#   Useful for saving to public/shared directories
#alt_finished_root_save_paths:
#  - /eos/user/b/bdanzi/EarlyRun3HHbbgg

cd $CMSSW_BASE/src/flashggFinalFit/Parquet

rm /eos/home-b/bdanzi/EarlyRun3HHbbgg/2D/CMSSW_14_1_0_pre4/src/flashggFinalFit/Parquet/2DFit_Resolved/rootFiles/Results_noDijetMass_PNetJets_DijetReg_onlyNonRes_eval_kl_boosted_addSingleH_mbbtrimmed_boosteBDT_ALL_newcategory/*
python3 processParquet.py --dataset "Results_noDijetMass_PNetJets_DijetReg_onlyNonRes_eval_kl_boosted_addSingleH_mbbtrimmed_boosteBDT_ALL_newcategory" \
--input "combined_resolved_boosted_trimmed4finalfit.parquet" \
--cat0 "score_GluGluToHH >= 0.998439 & pred_GluGluToHH >= 0.95406 & is_boosted == 0" \
--cat1 "score_GluGluToHH >= 0.983356 and pred_GluGluToHH >= 0.845485 and (score_GluGluToHH < 0.998439 or pred_GluGluToHH < 0.95406) & is_boosted == 0" \
--replace "Res_mjj_regressed=dijet_mass,weight=weight_tot,proc=sample" \
--logLevel INFO --years "2022+2023"

#need to:
#remove any conda env
cd $CMSSW_BASE/src/flashggFinalFit/Trees2WS
rm -r $CMSSW_BASE/src/flashggFinalFit/Trees2WS/inputs/bbgg/*
python3 trees2ws.py --inputConfig config_bbgg.py --inputTreeFile /eos/home-b/bdanzi/EarlyRun3HHbbgg/2D/CMSSW_14_1_0_pre4/src/flashggFinalFit/Parquet/2DFit_Resolved/rootFiles/Results_noDijetMass_PNetJets_DijetReg_onlyNonRes_eval_kl_boosted_addSingleH_mbbtrimmed_boosteBDT_ALL_newcategory/Data_13TeV_2022+2023.root --outputDir $CMSSW_BASE/src/flashggFinalFit/Trees2WS/inputs/bbgg/ws_data --year 2022+2023 --productionMode gghh --logLevel INFO
python3 trees2ws.py --inputConfig config_bbgg.py --inputTreeFile /eos/home-b/bdanzi/EarlyRun3HHbbgg/2D/CMSSW_14_1_0_pre4/src/flashggFinalFit/Parquet/2DFit_Resolved/rootFiles/Results_noDijetMass_PNetJets_DijetReg_onlyNonRes_eval_kl_boosted_addSingleH_mbbtrimmed_boosteBDT_ALL_newcategory/output_GluGluToHH_M125_13TeV.root --outputDir $CMSSW_BASE/src/flashggFinalFit/Trees2WS/inputs/bbgg/ws_GGHH --year 2022+2023 --productionMode gghh --logLevel INFO
python3 trees2ws.py --inputConfig config_bbgg.py --inputTreeFile /eos/home-b/bdanzi/EarlyRun3HHbbgg/2D/CMSSW_14_1_0_pre4/src/flashggFinalFit/Parquet/2DFit_Resolved/rootFiles/Results_noDijetMass_PNetJets_DijetReg_onlyNonRes_eval_kl_boosted_addSingleH_mbbtrimmed_boosteBDT_ALL_newcategory/output_ttHToGG_M125_13TeV.root --outputDir $CMSSW_BASE/src/flashggFinalFit/Trees2WS/inputs/bbgg/ws_TTH --year 2022+2023 --productionMode tth --logLevel INFO
python3 trees2ws.py --inputConfig config_bbgg.py --inputTreeFile /eos/home-b/bdanzi/EarlyRun3HHbbgg/2D/CMSSW_14_1_0_pre4/src/flashggFinalFit/Parquet/2DFit_Resolved/rootFiles/Results_noDijetMass_PNetJets_DijetReg_onlyNonRes_eval_kl_boosted_addSingleH_mbbtrimmed_boosteBDT_ALL_newcategory/output_VHToGG_M125_13TeV.root --outputDir $CMSSW_BASE/src/flashggFinalFit/Trees2WS/inputs/bbgg/ws_VH --year 2022+2023 --productionMode vh --logLevel INFO
python3 trees2ws.py --inputConfig config_bbgg.py --inputTreeFile /eos/home-b/bdanzi/EarlyRun3HHbbgg/2D/CMSSW_14_1_0_pre4/src/flashggFinalFit/Parquet/2DFit_Resolved/rootFiles/Results_noDijetMass_PNetJets_DijetReg_onlyNonRes_eval_kl_boosted_addSingleH_mbbtrimmed_boosteBDT_ALL_newcategory/output_VBFHToGG_M125_13TeV.root --outputDir $CMSSW_BASE/src/flashggFinalFit/Trees2WS/inputs/bbgg/ws_VBF --year 2022+2023 --productionMode vbf --logLevel INFO
python3 trees2ws.py --inputConfig config_bbgg.py --inputTreeFile /eos/home-b/bdanzi/EarlyRun3HHbbgg/2D/CMSSW_14_1_0_pre4/src/flashggFinalFit/Parquet/2DFit_Resolved/rootFiles/Results_noDijetMass_PNetJets_DijetReg_onlyNonRes_eval_kl_boosted_addSingleH_mbbtrimmed_boosteBDT_ALL_newcategory/output_GluGluHToGG_M125_13TeV.root --outputDir $CMSSW_BASE/src/flashggFinalFit/Trees2WS/inputs/bbgg/ws_GG2H --year 2022+2023 --productionMode ggh --logLevel INFO


cd $CMSSW_BASE/src/flashggFinalFit/Background
# ttH fTest and fit (mgg and mjj), update manually config_tth.py etc.
rm -r $CMSSW_BASE/src/flashggFinalFit/Background/outdir_tth/resonantFit/output
rm -r $CMSSW_BASE/src/flashggFinalFit/Background/outdir_tth/resonantFit/
mkdir $CMSSW_BASE/src/flashggFinalFit/Background/outdir_tth/resonantFit/
mkdir $CMSSW_BASE/src/flashggFinalFit/Background/outdir_tth/resonantFit/output
python3 RunBackgroundScripts.py --inputConfig config_tth.py --mode fTestParallel --fitType mgg --resonant --modeOpts="--doPlots --threshold 1 --nGaussMax 5 --template NGauss" --logLevel INFO
python3 RunBackgroundScripts.py --inputConfig config_tth.py --mode fTestParallel --fitType mjj --resonant --modeOpts="--doPlots --threshold 1 --nGaussMax 5 --template NGauss" --logLevel INFO
python3 RunBackgroundScripts.py --inputConfig config_tth.py --mode resonantFit --fitType mgg --resonant --modeOpts="--doPlots --skipSystematics --replacementThreshold 0 --template NGauss" --groupBackgroundFitJobsByCat --logLevel INFO
python3 RunBackgroundScripts.py --inputConfig config_tth.py --mode resonantFit --fitType mjj --resonant --modeOpts="--doPlots --skipSystematics --replacementThreshold 0 --template NGauss" --groupBackgroundFitJobsByCat --logLevel INFO --noClean
python3 ../tools/CollectModels.py --json $CMSSW_BASE/src/flashggFinalFit/Background/outdir_tth/resonantFit/output/models.json --output $CMSSW_BASE/src/flashggFinalFit/Background/outdir_tth/resonantFit/output/CMS-2D_multipdf_tth_%PROC_%YEAR_%CAT.root --wsType bkg-res --logLevel INFO

# VH fTest and fit
rm -r $CMSSW_BASE/src/flashggFinalFit/Background/outdir_vh/resonantFit/output
rm -r $CMSSW_BASE/src/flashggFinalFit/Background/outdir_vh/resonantFit/
mkdir $CMSSW_BASE/src/flashggFinalFit/Background/outdir_vh/resonantFit/
mkdir $CMSSW_BASE/src/flashggFinalFit/Background/outdir_vh/resonantFit/output
python3 RunBackgroundScripts.py --inputConfig config_vh.py --mode fTestParallel --fitType mgg --resonant --modeOpts="--doPlots --threshold 1 --nGaussMax 5 --template NGauss" --logLevel INFO
python3 RunBackgroundScripts.py --inputConfig config_vh.py --mode fTestParallel --fitType mjj --resonant --modeOpts="--doPlots --threshold 1 --nGaussMax 5 --template NGauss" --logLevel INFO
python3 RunBackgroundScripts.py --inputConfig config_vh.py --mode resonantFit --fitType mgg --resonant --modeOpts="--doPlots --skipSystematics --replacementThreshold 0 --template NGauss" --groupBackgroundFitJobsByCat --logLevel INFO
python3 RunBackgroundScripts.py --inputConfig config_vh.py --mode resonantFit --fitType mjj --resonant --modeOpts="--doPlots --skipSystematics --replacementThreshold 0 --template NGauss" --groupBackgroundFitJobsByCat --logLevel INFO --noClean
python3 ../tools/CollectModels.py --json $CMSSW_BASE/src/flashggFinalFit/Background/outdir_vh/resonantFit/output/models.json --output $CMSSW_BASE/src/flashggFinalFit/Background/outdir_vh/resonantFit/output/CMS-2D_multipdf_vh_%PROC_%YEAR_%CAT.root --wsType bkg-res --logLevel INFO

# VBF fTest and fit
rm -r $CMSSW_BASE/src/flashggFinalFit/Background/outdir_vbf/resonantFit/output
rm -r $CMSSW_BASE/src/flashggFinalFit/Background/outdir_vbf/resonantFit
mkdir $CMSSW_BASE/src/flashggFinalFit/Background/outdir_vbf/resonantFit/
mkdir $CMSSW_BASE/src/flashggFinalFit/Background/outdir_vbf/resonantFit/output
python3 RunBackgroundScripts.py --inputConfig config_vbf.py --mode fTestParallel --fitType mgg --resonant --modeOpts="--doPlots --threshold 1 --nGaussMax 5 --template NGauss" --logLevel INFO
python3 RunBackgroundScripts.py --inputConfig config_vbf.py --mode fTestParallel --fitType mjj --resonant --modeOpts="--doPlots --threshold 1 --nGaussMax 5 --template NGauss" --logLevel INFO
python3 RunBackgroundScripts.py --inputConfig config_vbf.py --mode resonantFit --fitType mgg --resonant --modeOpts="--doPlots --skipSystematics --replacementThreshold 0 --template NGauss" --groupBackgroundFitJobsByCat --logLevel INFO
python3 RunBackgroundScripts.py --inputConfig config_vbf.py --mode resonantFit --fitType mjj --resonant --modeOpts="--doPlots --skipSystematics --replacementThreshold 0 --template NGauss" --groupBackgroundFitJobsByCat --logLevel INFO --noClean
python3 ../tools/CollectModels.py --json $CMSSW_BASE/src/flashggFinalFit/Background/outdir_vbf/resonantFit/output/models.json --output $CMSSW_BASE/src/flashggFinalFit/Background/outdir_vbf/resonantFit/output/CMS-2D_multipdf_vbf_%PROC_%YEAR_%CAT.root --wsType bkg-res --logLevel INFO

# ggH fTest and fit
rm -r $CMSSW_BASE/src/flashggFinalFit/Background/outdir_ggh/resonantFit/output
rm -r $CMSSW_BASE/src/flashggFinalFit/Background/outdir_ggh/resonantFit
mkdir $CMSSW_BASE/src/flashggFinalFit/Background/outdir_ggh/resonantFit/
mkdir $CMSSW_BASE/src/flashggFinalFit/Background/outdir_ggh/resonantFit/output
python3 RunBackgroundScripts.py --inputConfig config_ggh.py --mode fTestParallel --fitType mgg --resonant --modeOpts="--doPlots --threshold 1 --nGaussMax 5 --template NGauss" --logLevel INFO
python3 RunBackgroundScripts.py --inputConfig config_ggh.py --mode fTestParallel --fitType mjj --resonant --modeOpts="--doPlots --threshold 1 --nGaussMax 5 --template NGauss" --logLevel INFO
python3 RunBackgroundScripts.py --inputConfig config_ggh.py --mode resonantFit --fitType mgg --resonant --modeOpts="--doPlots --skipSystematics --replacementThreshold 0 --template NGauss" --groupBackgroundFitJobsByCat --logLevel INFO
python3 RunBackgroundScripts.py --inputConfig config_ggh.py --mode resonantFit --fitType mjj --resonant --modeOpts="--doPlots --skipSystematics --replacementThreshold 0 --template NGauss" --groupBackgroundFitJobsByCat --logLevel INFO --noClean
python3 ../tools/CollectModels.py --json $CMSSW_BASE/src/flashggFinalFit/Background/outdir_ggh/resonantFit/output/models.json --output $CMSSW_BASE/src/flashggFinalFit/Background/outdir_ggh/resonantFit/output/CMS-2D_multipdf_ggh_%PROC_%YEAR_%CAT.root --wsType bkg-res --logLevel INFO

cd $CMSSW_BASE/src/flashggFinalFit/Background
make clean
make -j 8
python3 RunBackgroundScripts.py --inputConfig config_bbgg.py --mode fTestParallel --fitType mgg --logLevel INFO
python3 RunBackgroundScripts.py --inputConfig config_bbgg.py --mode fTestParallel --fitType mjj --logLevel INFO


cd $CMSSW_BASE/src/flashggFinalFit/Signal
python3 RunSignalScripts.py --inputConfig config_bbgg.py --mode fTest --fitType mgg --modeOpts="--doPlots --nGaussMax 5 --template NGauss" --logLevel INFO
python3 RunSignalScripts.py --inputConfig config_bbgg.py --mode fTest --fitType mjj --modeOpts="--doPlots --nGaussMax 5 --template NGauss" --logLevel INFO
python3 RunSignalScripts.py --inputConfig config_bbgg.py --mode signalFit --fitType mgg --modeOpts="--doPlots --skipSystematics --replacementThreshold 0 --template NGauss" --groupSignalFitJobsByCat --logLevel INFO
python3 RunSignalScripts.py --inputConfig config_bbgg.py --mode signalFit --fitType mjj --modeOpts="--doPlots --skipSystematics --replacementThreshold 0 --template NGauss" --groupSignalFitJobsByCat --logLevel INFO --noClean
python3 ../tools/CollectModels.py --json $CMSSW_BASE/src/flashggFinalFit/Signal/outdir_bbgg_2022+2023/signalFit/output/models.json --output $CMSSW_BASE/src/flashggFinalFit/Signal/outdir_bbgg_2022+2023/signalFit/output/CMS-2D_sigfit_%PROC_%YEAR_%CAT.root --wsType signal --logLevel INFO


cd $CMSSW_BASE/src/flashggFinalFit/Datacard
python3 make2DWorkspaces.py --cats cat0,cat1 --years 2022+2023 --logLevel INFO
python3 RunYields.py --inputWSDirMap 2022+2023=/eos/home-b/bdanzi/EarlyRun3HHbbgg/2D/CMSSW_14_1_0_pre4/src/flashggFinalFit/Trees2WS/inputs/bbgg/ws_GGHH \
--sigModelWSDir /eos/home-b/bdanzi/EarlyRun3HHbbgg/2D/CMSSW_14_1_0_pre4/src/flashggFinalFit/Datacard/outdir_2D/signal \
--sigModelExt GGHH \
--bkgModelWSDir /eos/home-b/bdanzi/EarlyRun3HHbbgg/2D/CMSSW_14_1_0_pre4/src/flashggFinalFit/Datacard/outdir_2D/bkg-nonres \
--bkgModelExt bkg-nonres \
--resBkgModelWSDir /eos/home-b/bdanzi/EarlyRun3HHbbgg/2D/CMSSW_14_1_0_pre4/src/flashggFinalFit/Datacard/outdir_2D/bkg-res \
--resBkgModelExt bkg-res \
--resProcs tth,vh,vbf,ggh --cats cat0,cat1 --procs auto --ext bbgg --fitType 2D --logLevel INFO --batch local --queue espresso
python3 makeDatacard.py --ext bbgg --years 2022+2023 --analysis bbgg --prune --doTrueYield --skipCOWCorr --doMCStatUncertainty --saveDataFrame --output Datacard_bbgg --logLevel INFO

cd $CMSSW_BASE/src/flashggFinalFit/Combine
rm -r Models/signal
rm -r Models/background
mkdir -p Models/signal
mkdir -p Models/background
cp ../Datacard/outdir_2D/signal/*.root Models/signal/.
cp ../Datacard/outdir_2D/bkg-nonres/*.root Models/background/.
cp ../Datacard/outdir_2D/bkg-res/*.root Models/background/.
cp ../Datacard/Datacard_bbgg.txt .

python3 RunText2Workspace.py --mode mu_inclusive --batch local --ext _bbgg
python3 RunFits.py --inputJson inputs_bbgg/inputs_bbgg_bestfit_syst.json --mode mu_inclusive --ext _bbgg --mass 125.38 --batch condor --queue espresso --dryRun --logLevel INFO
./runFits_bbgg_mu_inclusive/condor_bestfit_syst_r.sh 0
combine -M AsymptoticLimits Datacard_bbgg_mu_inclusive.root -t -1 --X-rtd TMCSO_AdaptivePseudoAsimov=0 --X-rtd TMCSO_PseudoAsimov=0 --cminDefaultMinimizerStrategy 0 --cminFallbackAlgo Minuit2,Migrad,0:0.1 --X-rt MINIMIZER_freezeDisassociatedParams --X-rtd MINIMIZER_multiMin_hideConstants --X-rtd MINIMIZER_multiMin_maskConstraints --X-rtd MINIMIZER_multiMin_maskChannels=2 --verbose 2 --freezeParameters var{.*} --floatParameters var{MCStat_.*} --setParameters=MH=125

# 2 cat, no mjj DNN
-- AsymptoticLimits ( CLs ) --
Observed Limit: r < 10.0275
Expected  2.5%: r < 4.6630
Expected 16.0%: r < 6.6341
Expected 50.0%: r < 10.0312
Expected 84.0%: r < 15.5492
Expected 97.5%: r < 23.0848

__________________________________
# 2 cat, no mjj DNN
 # -- AsymptoticLimits ( CLs ) --
# Observed Limit: r < 9.9787
# Expected  2.5%: r < 4.7656
# Expected 16.0%: r < 6.7285
# Expected 50.0%: r < 10.0000
# Expected 84.0%: r < 15.1420
# Expected 97.5%: r < 22.0000

