import argparse
import common
from preprocessing.compositeVars import compositeVariables
import json
import numpy as np
import os
import pandas as pd
import pyarrow as pa
from pyarrow.parquet import ParquetFile
import sys
import yaml

BATCH_SIZE = 1000

def reduceMemory(df):
  for column in df.columns:
    if df[column].dtype == "float64":
      print("%s float64 -> float32"%column.ljust(50))
      df.loc[:, column] = df[column].astype("float32")
    elif df[column].dtype == "int64":
      print("%s  int64 -> uint8"%column.ljust(50))
      df.loc[:, column] = df[column].astype("uint8")
    else:
      pass
      #print("%s %s -> %s"%(column.ljust(50), df[column].dtype, df[column].dtype))

def checkNans(df):
  for column in df.columns:
    try:
      if np.isnan(df[column]).any():
        print(df.loc[np.isnan(df[column]), column], ", replacing it with ", common.dummy_val)
        df.loc[:, column].replace(np.nan, common.dummy_val, inplace=True)
    except:
      pass

def checkInfs(df):
  for column in df.columns:
    try:
      if np.isinf(df[column]).any():
        print(df.loc[np.isinf(df[column]), column], ", dropping it.")
        df.drop(df.loc[np.isinf(df[column])].index, inplace=True)    
    except:
      pass

def skimAndAdd(parquet_input, parquet_output, variables_json, keep_all_features, sig_procs=None):
  pf = ParquetFile(parquet_input) 
  pf_columns = pf.schema.names
  # Needed for DD sample as the pre-imputed Max and Min mvaID are still there, should remove and recalculate
  columns_to_remove = ["Max_mvaID", "Min_mvaID"]
  pf_columns = [col for col in pf_columns if col not in columns_to_remove]

  print()
  print("Loading input variables...")
  with open(variables_json, "r") as f_var:
    input_variables = json.load(f_var)["input_variables"]
  # We need to recalculate min and max mvaID because some of the inputs files may not be pre-selected and we need to apply the mvaID preselection here
  input_variables += [var for var in ["Max_mvaID", "Min_mvaID", "lead_isScEtaEB", "lead_isScEtaEE", "sublead_isScEtaEB", "sublead_isScEtaEE"] if var not in input_variables]
  skimmedVariables = []
  if not keep_all_features: skimmedVariables = compositeVariables(input_variables, pf_columns)
  print(skimmedVariables)

  print()
  print("Constructing the dataframe...")
  df = pd.DataFrame()
  for batch in pf.iter_batches(batch_size = BATCH_SIZE, columns = None if keep_all_features else skimmedVariables + ["mass", "weight"]):
    df_batch = pa.Table.from_batches([batch]).to_pandas() 
    df_batch = compositeVariables(input_variables, pf_columns, df = df_batch)
    df = pd.concat([df, df_batch], ignore_index=True)

  print()
  print("Reducing memory...")
  reduceMemory(df)

  print()
  print("Checking for NaNs...")
  checkNans(df)

  print()
  print("Checking for infs...")
  checkInfs(df)

  print()
  print("Final number of rows:")
  print(len(df.index))
  print()
  print("Final columns:")
  print(df.columns)
  print()

  os.makedirs("/".join(parquet_output.split("/")[:-1]), exist_ok=True)
  df.to_parquet(parquet_output)
  return df

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--run-era', type=str, required=True, choices = ['Run3', '2022', 'Run3_2022postEE', 'Run3_2022preEE','Run3_2022postEE_NW'])#FIXME Add other run eras
  parser.add_argument('--parquet-output', '-o', type=str, required=True)
  parser.add_argument('--variables-json', '-v', type=str, required=True)
  parser.add_argument('--applyMassCut', action="store_true", default=False)
  parser.add_argument('--keep-all-features', '-f', action="store_true", default=False)
  parser.add_argument('--processOnly', action="store_true", default=False)
  parser.add_argument('--mergeOnly', action="store_true", default=False)
  parser.add_argument('--runOnLxplus', action="store_true", default=False)

  args = parser.parse_args()
  
  conf_file = 'configs/run_eras.json'
  if args.runOnLxplus:
    conf_file = 'configs/run_eras_lxplus.json'

  with open(conf_file, 'r') as refile:
    remap = json.load(refile)
    if args.run_era == 'Run3':
      run_eras = ['Run3_2022postEE', 'Run3_2022preEE']
    if args.run_era == '2022':
      run_eras = ['Run3_2022postEE', 'Run3_2022preEE']
    else:
      run_eras = [args.run_era]
    sampleInfo = {}
    for re in run_eras:
      with open(remap[re], 'r') as f_parq:
        temp = yaml.load(f_parq, Loader = yaml.Loader)
        for s in temp:
          if s in sampleInfo:
            sampleInfo[s][re] = {f:temp[s][f] for f in temp[s]}
          else:
            sampleInfo[s] = {re:{f:temp[s][f] for f in temp[s]}}
  
  sampleNames = list(sampleInfo.keys())
  if not args.mergeOnly:
    for name in sampleNames:
      for re in sampleInfo[name]:
        print("###############")
        print()
        print()
        print("Processing "+name)
        print("---------------")
        skimAndAdd(sampleInfo[name][re]["path"], args.parquet_output+"/"+name+'_'+re+".parquet", args.variables_json, args.keep_all_features)

  if not args.processOnly:
    print("###############")
    print()
    print()

    totalLumi = 0.0
    for name in sampleNames:
      for re in sampleInfo[name]:
        if "Data" in name:
          totalLumi += sampleInfo[name][re]["lumi"]
    print("Total Data Luminosity = "+str(totalLumi)+" pb^{-1}")
    print()

    df = pd.DataFrame()
    for name in sampleNames:
      for re in sampleInfo[name]:
        pf = ParquetFile(args.parquet_output+"/"+name+'_'+re+".parquet") 

        # Grouping of processes
        procName = name
        if "Data" in name:
          procName = "Data"
        elif "G-4Jets" in name:
          procName = "GJets"
        elif "GJetPt" in name:
          procName = "GJets"

        # Getting the xs
        xs = 1.0

        if "Data" not in name and "DDQCDGJets" not in name:
          xs = sampleInfo[name][re]["xs"]
          print("Merging "+name+" from " + re + " with process name "+procName+" and xs = "+str(xs)+" pb...")
        else:
          print("Merging "+name+" from " + re + " with process name "+procName+"...")
        for batch in pf.iter_batches(batch_size = BATCH_SIZE):
          df_batch = pa.Table.from_batches([batch]).to_pandas()
          
          df_batch = df_batch[(df_batch["Max_mvaID"] > -0.7) & (df_batch["Min_mvaID"] > -0.7)]

          if args.applyMassCut:
            df_batch = df_batch[(df_batch["mass"] > 100)&(df_batch["mass"] < 180)]
            
          df_batch["proc"] = procName
          df_batch["year"] = int(re.split('_')[1][:4])
          if "Data" not in name and "DDQCDGJets" not in name:
            df_batch["weight"] = df_batch["weight"]*xs*totalLumi

          df = pd.concat([df, df_batch], ignore_index=True)
        print()

    print()
    print("Final number of rows:")
    print(len(df.index))
    print()
    print("Final columns:")
    print(df.columns)
    print()

    df.to_parquet(args.parquet_output+"/"+"merged.parquet")

