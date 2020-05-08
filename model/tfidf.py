from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


def tfidf_transform_and_save(tr_,te_,name):
    from sklearn.feature_extraction.text import TfidfVectorizer

    vectorizer = TfidfVectorizer(max_df=0.9)
    tf1 = vectorizer.fit_transform(tr_).toarray()
    tf2 = vectorizer.transform(te_).toarray()
    print(tf1.shape)
    df1 = pd.DataFrame(tf1, columns = [f"{name}_{i}" for i in  range(tf1.shape[1])])
    df2 = pd.DataFrame(tf2, columns = [f"{name}_{i}" for i in range(tf1.shape[1])])
    return df1,df2

    