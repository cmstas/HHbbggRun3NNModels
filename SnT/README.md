# SnT DNN model

For the first time, one needs to first create the appropriate environment with:
```
bash make_env.sh
```
Then, each time, one needs to run:
```
source setup.sh
```

To preprocess and merge input parquet files, run (if run on lxplus):
```
python3 preprocessing/process_inputs.py -o . -v configs/variables_phoMVApreCut.json --applyMassCut --run-era 2022 --runOnLxplus
```

The `--applyMassCut` flag above applies the cut 100 < m<sub>γγ</sub> < 180 GeV. The requirement for WP90 for both photons is applied with the following script:
```
python preprocessing/photon_MVACut.py -i input.parquet --applyWP90Cuts
```

Currently, the model evaluation on the full set of HiggsDNA is unfeasible on lxplus due to memory constraints. We are figuring this out.

The full HiggsDNA preselection file with the SnT DNN score attached to it can be found on:
```
/eos/user/a/atalierc/bbgg_pq_files/SnT/output_optimized_Full2022_CRUW_1p0_PhoWP90_phoMVApreCut90_keepallfeatures.parquet
```
