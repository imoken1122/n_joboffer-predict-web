import pandas as pd
import numpy as np
from model.preprocess import prerpocess_run
import pickle as pk

RS = 1103

def load_model(idx):
    with open(f'./model/pickle/model_fold{idx}.pickle', mode='rb') as fp:
        clf = pk.load(fp)
    return clf

def predictor(filename):

    test,no = prerpocess_run(filename)
    preds = []
    for i in range(3):
        print(f"lgb_model{i} is learning")
        model = load_model(i)
        te_pred = model.predict(test)
        preds.append(te_pred)
    submit = np.mean(preds,axis = 0)
    df_pred = pd.DataFrame(np.c_[no,submit], columns = ('お仕事No.', '応募数 合計'))

    return df_pred
