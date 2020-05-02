import pandas as pd
import numpy as np
from model.preprocess import get_data
import pickle as pk
RS = 1103

def load_model():
    with open('./model/pickle/model_fold0.pickle', mode='rb') as fp:
        clf = pk.load(fp)
    return clf

def predictor(filename):

    test,no = get_data(filename)
    model = load_model()
    te_pred = model.predict(test)
    df_pred = pd.DataFrame(np.c_[no,te_pred], columns = ('お仕事No.', '応募数 合計'))
    return df_pred
