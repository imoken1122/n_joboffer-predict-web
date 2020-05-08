import umap
import pandas as pd
import numpy as np
import pickle as pk
import re



def load_model(name):
    with open(f'./model/pickle/{name}_umap.pickle', mode='rb') as fp:
        clf = pk.load(fp)
    return clf

def umap_transfromer(te,name):
    umap_model = load_model(name)
    te_umap = umap_model.transform(te)

    c = [f"{name}_UMAP{i}" for i in range(te_umap.shape[1])]
    df_te = pd.DataFrame(te_umap,columns=c)
    return df_te

