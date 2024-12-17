import argparse
import numpy as np
import os
import pandas as pd
import pyarrow as pa
from pyarrow.parquet import ParquetFile

# To be run on merged parquet files coming out from process_inputs.py

if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--input', '-i', type=str, required=True)
  parser.add_argument('--applyWP90Cuts', action="store_true", default=False)
  parser.add_argument('--applyWP80Cuts', action="store_true", default=False)

  args = parser.parse_args()
  
  df = pd.read_parquet(args.input)
  
  postfix = ""
  
  if args.applyWP80Cuts:
    lead_mvaID_condition = ((df["lead_isScEtaEB"] == True) & (df['lead_mvaID'] > 0.420473)) | ((df["lead_isScEtaEE"] == True) & (df['lead_mvaID'] > 0.203451))
    sublead_mvaID_condition = ((df["sublead_isScEtaEB"] == True) & (df['sublead_mvaID'] > 0.420473)) | ((df["sublead_isScEtaEE"] == True) & (df['sublead_mvaID'] > 0.203451))
    df = df[(lead_mvaID_condition) & (sublead_mvaID_condition)]
    print("Select mva WP80")
    postfix = "WP80"
    
  if args.applyWP90Cuts:
    lead_mvaID_condition = ((df["lead_isScEtaEB"] == True) & (df['lead_mvaID'] > 0.0439603)) | ((df["lead_isScEtaEE"] == True) & (df['lead_mvaID'] > -0.249526))
    sublead_mvaID_condition = ((df["sublead_isScEtaEB"] == True) & (df['sublead_mvaID'] > 0.0439603)) | ((df["sublead_isScEtaEE"] == True) & (df['sublead_mvaID'] > -0.249526))
    df = df[(lead_mvaID_condition) & (sublead_mvaID_condition)]
    print("Select mva WP90")
    postfix = "WP90"
   
  filename = os.path.splitext(os.path.basename(args.input))[0]
  new_filename = f"{filename}_{postfix}.parquet"

  df.to_parquet(new_filename)

