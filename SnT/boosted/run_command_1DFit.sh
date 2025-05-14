# Boosted 1D fit, use https://gitlab.cern.ch/hhtobbgg_nu/flashggFinalFit/-/tree/higgsdnafinalfit?ref_type=heads

cd $CMSSW_BASE/src/flashggFinalFit/Parquet
_____________________________________________
rm -r 2DFit_Resolved/rootFiles/Results_noDijetMass_PNetJets_DijetReg_onlyNonRes_eval_kl_boosted_addSingleH_mbbtrimmed_boosteBDT_ALL_newcategory_boosted1D
python3 processParquet.py --dataset "Results_noDijetMass_PNetJets_DijetReg_onlyNonRes_eval_kl_boosted_addSingleH_mbbtrimmed_boosteBDT_ALL_newcategory_boosted1D" \
--input "combined_resolved_boosted_trimmed4finalfit.parquet" \
--cat2 "is_boosted == 1" \
--replace "mjj=dijet_mass,weight=weight_tot,proc=sample" \
--logLevel DEBUG \
--years "2022+2023"

cd ../Trees2WS
python3 trees2ws.py --inputConfig config_bbgg.py --inputTreeFile /eos/user/b/bdanzi/EarlyRun3HHbbgg/2D/CMSSW_14_1_0_pre4/src/flashggFinalFit/Parquet/2DFit_Resolved/rootFiles/Results_noDijetMass_PNetJets_DijetReg_onlyNonRes_eval_kl_boosted_addSingleH_mbbtrimmed_boosteBDT_ALL_newcategory_boosted1D/Data_13TeV_2022+2023.root --outputDir $CMSSW_BASE/src/flashggFinalFit/Trees2WS/inputs/bbgg_1D/ws_data --year 2022+2023 --productionMode gghh --logLevel INFO
python3 trees2ws.py --inputConfig config_bbgg.py --inputTreeFile /eos/user/b/bdanzi/EarlyRun3HHbbgg/2D/CMSSW_14_1_0_pre4/src/flashggFinalFit/Parquet/2DFit_Resolved/rootFiles/Results_noDijetMass_PNetJets_DijetReg_onlyNonRes_eval_kl_boosted_addSingleH_mbbtrimmed_boosteBDT_ALL_newcategory_boosted1D/output_GluGluToHH_M125_13TeV.root --outputDir $CMSSW_BASE/src/flashggFinalFit/Trees2WS/inputs/bbgg_1D/ws_GGHH --year 2022+2023 --productionMode gghh --logLevel INFO
python3 trees2ws.py --inputConfig config_bbgg.py --inputTreeFile /eos/user/b/bdanzi/EarlyRun3HHbbgg/2D/CMSSW_14_1_0_pre4/src/flashggFinalFit/Parquet/2DFit_Resolved/rootFiles/Results_noDijetMass_PNetJets_DijetReg_onlyNonRes_eval_kl_boosted_addSingleH_mbbtrimmed_boosteBDT_ALL_newcategory_boosted1D/output_ttHToGG_M125_13TeV.root --outputDir $CMSSW_BASE/src/flashggFinalFit/Trees2WS/inputs/bbgg_1D/ws_TTH --year 2022+2023 --productionMode tth --logLevel INFO
python3 trees2ws.py --inputConfig config_bbgg.py --inputTreeFile /eos/user/b/bdanzi/EarlyRun3HHbbgg/2D/CMSSW_14_1_0_pre4/src/flashggFinalFit/Parquet/2DFit_Resolved/rootFiles/Results_noDijetMass_PNetJets_DijetReg_onlyNonRes_eval_kl_boosted_addSingleH_mbbtrimmed_boosteBDT_ALL_newcategory_boosted1D/output_VHToGG_M125_13TeV.root --outputDir $CMSSW_BASE/src/flashggFinalFit/Trees2WS/inputs/bbgg_1D/ws_VH --year 2022+2023 --productionMode vh --logLevel INFO
python3 trees2ws.py --inputConfig config_bbgg.py --inputTreeFile /eos/user/b/bdanzi/EarlyRun3HHbbgg/2D/CMSSW_14_1_0_pre4/src/flashggFinalFit/Parquet/2DFit_Resolved/rootFiles/Results_noDijetMass_PNetJets_DijetReg_onlyNonRes_eval_kl_boosted_addSingleH_mbbtrimmed_boosteBDT_ALL_newcategory_boosted1D/output_VBFHToGG_M125_13TeV.root --outputDir $CMSSW_BASE/src/flashggFinalFit/Trees2WS/inputs/bbgg_1D/ws_VBF --year 2022+2023 --productionMode vbf --logLevel INFO
python3 trees2ws.py --inputConfig config_bbgg.py --inputTreeFile /eos/user/b/bdanzi/EarlyRun3HHbbgg/2D/CMSSW_14_1_0_pre4/src/flashggFinalFit/Parquet/2DFit_Resolved/rootFiles/Results_noDijetMass_PNetJets_DijetReg_onlyNonRes_eval_kl_boosted_addSingleH_mbbtrimmed_boosteBDT_ALL_newcategory_boosted1D/output_GluGluHToGG_M125_13TeV.root --outputDir $CMSSW_BASE/src/flashggFinalFit/Trees2WS/inputs/bbgg_1D/ws_GG2H --year 2022+2023 --productionMode ggh --logLevel INFO

# change to bbgg_1D
cd ../Background
python3 RunBackgroundScripts.py --inputConfig config_tth.py --mode fTestParallel --fitType mgg --resonant --modeOpts="--doPlots --threshold 5 --nGaussMax 5 --template NGauss" --logLevel INFO; python3 RunBackgroundScripts.py --inputConfig config_tth.py --mode resonantFit --fitType mgg --modeOpts="--doPlots --skipSystematics --replacementThreshold 0 --template NGauss" --resonant --groupBackgroundFitJobsByCat --logLevel INFO
python3 RunBackgroundScripts.py --inputConfig config_vh.py --mode fTestParallel --fitType mgg --resonant --modeOpts="--doPlots --threshold 5 --nGaussMax 5 --template NGauss" --logLevel INFO; python3 RunBackgroundScripts.py --inputConfig config_vh.py --mode resonantFit --fitType mgg --modeOpts="--doPlots --skipSystematics --replacementThreshold 0 --template NGauss" --resonant --groupBackgroundFitJobsByCat --logLevel INFO
python3 RunBackgroundScripts.py --inputConfig config_vbf.py --mode fTestParallel --fitType mgg --resonant --modeOpts="--doPlots --threshold 0 --nGaussMax 5 --template NGauss" --logLevel INFO; python3 RunBackgroundScripts.py --inputConfig config_vbf.py --mode resonantFit --fitType mgg --modeOpts="--doPlots --skipSystematics --replacementThreshold 0 --template NGauss" --resonant --groupBackgroundFitJobsByCat --logLevel INFO
python3 RunBackgroundScripts.py --inputConfig config_ggh.py --mode fTestParallel --fitType mgg --resonant --modeOpts="--doPlots --threshold 5 --nGaussMax 5 --template NGauss" --logLevel INFO; python3 RunBackgroundScripts.py --inputConfig config_ggh.py --mode resonantFit --fitType mgg --modeOpts="--doPlots --skipSystematics --replacementThreshold 0 --template NGauss" --resonant --groupBackgroundFitJobsByCat --logLevel INFO

python3 RunPackager.py --cats cat2 --exts tth,vh,ggh,vbf --mergeYears --batch local --massPoints 125 --logLevel INFO

python3 RunBackgroundScripts.py --inputConfig config_bbgg.py --mode fTestParallel --fitType mgg --logLevel INFO

# change to bbgg_1D
cd ../Signal
python3 RunSignalScripts.py --inputConfig config_bbgg.py --mode fTest --fitType mgg --modeOpts="--doPlots --nGaussMax 5 --template NGauss" --logLevel INFO
python3 RunSignalScripts.py --inputConfig config_bbgg.py --mode signalFit --fitType mgg --modeOpts="--doPlots --skipSystematics --replacementThreshold 0 --template NGauss" --groupSignalFitJobsByCat --logLevel INFO

cd ../Datacard
python3 RunYields.py --inputWSDirMap 2022+2023=/eos/user/b/bdanzi/EarlyRun3HHbbgg/2D/CMSSW_14_1_0_pre4/src/flashggFinalFit/Trees2WS/inputs/bbgg_1D/ws_GGHH --sigModelWSDir /eos/user/b/bdanzi/EarlyRun3HHbbgg/2D/CMSSW_14_1_0_pre4/src/flashggFinalFit/Signal/outdir_bbgg_2022+2023/signalFit/output --sigModelExt bbgg_2022+2023_GGHH_2022+2023 --bkgModelWSDir /eos/user/b/bdanzi/EarlyRun3HHbbgg/2D/CMSSW_14_1_0_pre4/src/flashggFinalFit/Background/outdir_bbgg --resBkgModelWSDir /eos/user/b/bdanzi/EarlyRun3HHbbgg/2D/CMSSW_14_1_0_pre4/src/flashggFinalFit/Background/outdir_packaged --resBkgModelExt multipdf_packaged --resProcs tth,vh,vbf,ggh --cats cat2 --procs auto --ext bbgg --skipCOWCorr --batch local --queue espresso --fitType mgg --logLevel INFO --mergeYears
python3 makeDatacard.py --ext bbgg --years 2022+2023 --analysis bbgg --prune --doTrueYield --skipCOWCorr --doMCStatUncertainty --saveDataFrame --output Datacard_bbgg --logLevel INFO

cd ../Combine
mkdir -p Models/signal
mkdir -p Models/background
cp ../Signal/outdir_bbgg_2022+2023/signalFit/output/CMS-HGG_sigfit_bbgg_2022+2023_GGHH_2022+2023_*.root Models/signal/.
cp ../Background/outdir_bbgg/CMS-HGG_multipdf_*.root Models/background/.
cp ../Background/outdir_packaged/CMS-HGG_multipdf_packaged_cat*.root Models/background/.
cp ../Datacard/Datacard_bbgg.txt .

python3 RunText2Workspace.py --mode mu_inclusive --batch local --ext _bbgg
python3 RunFits.py --inputJson inputs_bbgg/inputs_bbgg_bestfit_syst.json --mode mu_inclusive --ext _bbgg --mass 125.38 --batch condor --queue espresso --dryRun --logLevel INFO
./runFits_bbgg_mu_inclusive/condor_bestfit_syst_r.sh 0
combine -M AsymptoticLimits Datacard_bbgg_mu_inclusive.root -t -1 --setParameters=MH=125 -m 125 --freezeParameters var{frac_.\*},var{sigma_.\*},var{mean_.\*}

