# Results_noDijetMass_PNetJets_DijetReg_onlyNonRes_eval_kl_boosted_addSingleH_mbbtrimmed.parquet, this is the parquet file scored with nonRes and singleH DNN scores. It is trimmed (throwing away un-needed variables)

#1. use select_validEntries4boostedBDT.py to pick out events that pass n_fatjets > 0, it produces two parquet files _withFatjets.parquet and _noFatjets.parquet
python3 select_validEntries4boostedBDT.py

#2. train and evaluate boosted BDT using _withFatjets.parquet. This makes sure only n_fatjets > 0 events are trained and scored
python3 BDT_training_complete_WithVHFeeding.py

#3. once done, copy the score from the scored parquet back to the input parquet to keep the same file structure as _noFatjets.parquet(this is a lazy approach, but it works:))
python3 sticher.py

#4. merged two parquet togethr again, with only features which are essential for final fit
python3 addBack_validEntries4boostedBDT.py

#5. categorization between boosted and no boosted
python3 categorization.py

#6. combine booosted and resolved for final fit
python3 prepare_comb_parquet_4finalfit.py
#done

#7. 2D fit for resolved
# Look at run_command_2DFit.sh

#8. 1D fit for boosted
# Look at run_command_1DFit.sh

