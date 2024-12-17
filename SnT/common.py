import numpy as np

dummy_val = -999.0 # value used for NaN values
HIGGS_MASS = 125.35

bkg_groups = {
  'GGJets': ['GGJets'],
  'QCDGJets':['DDQCDGJets'],
  #'GJets': ['GJetPt20To40', 'GJetPt40'],
  #'GJets': ['G-4Jets_HT_40to70', 'G-4Jets_HT_70to100', 'G-4Jets_HT_100to200', 'G-4Jets_HT_200to400', 'G-4Jets_HT_400to600', 'G-4Jets_HT_600'],
  'H': ['ttHToGG', 'VBFHToGG', 'VHToGG', 'GluGluHToGG'],
}
bkg_groups["all"] = [proc for key in bkg_groups.keys() for proc in bkg_groups[key]]

def deltaPhi(phi1, phi2):
  return abs(np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2)))

def deltaR(eta1, phi1, eta2, phi2):
  delta_eta = eta1 - eta2
  delta_phi = deltaPhi(phi1, phi2)
  return np.sqrt(delta_eta**2 + delta_phi**2)

def txtToList(txtList):
  procList = []

  for proc in txtList:
    procList.append(proc)

  return procList
  
def traced_print(*args, **kwargs):
    # Use builtins.print to call the original print function
    builtins.print(*args, **kwargs)
    stack = traceback.extract_stack()[-2]
    builtins.print(f"Printed from: {stack.filename}, line {stack.lineno}")

