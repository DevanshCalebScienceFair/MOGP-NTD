import os, sys, warnings
os.environ.setdefault("KMP_DUPLICATE_LIB_OK","TRUE"); warnings.filterwarnings("ignore")
sys.path.insert(0,"/Users/devansh/mogp-main-vscode/MOGP-NTD")
import numpy as np
from data import load_library
S="/Users/devansh/mogp-main-vscode/MOGP-NTD/analysis_scripts"
FP=np.asarray(load_library("data/library")["fingerprints"])
np.save(f"{S}/dim_bits_per_mol.npy", FP.sum(axis=1))
sub=FP[np.random.default_rng(0).choice(len(FP),1200,replace=False)].astype(float)
inter=sub@sub.T; n=sub.sum(1)
t=inter/(n[:,None]+n[None,:]-inter); iu=np.triu_indices(len(sub),1)
np.save(f"{S}/dim_tanimoto_sample.npy", t[iu].astype(np.float32))
print("saved bits + tanimoto distributions")
