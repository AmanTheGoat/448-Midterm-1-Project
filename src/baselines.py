"""
A cheap bag-of-words reference point for the CNN/LSTM, and its weights show
which words/symbols point to which family.
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

import config
from src.text_utils import tokenize


def _text(df, field):
    if field == "input":
        return df["llm_input"].tolist()
    if field == "output":
        return df["llm_output"].tolist()
    return (df["llm_input"] + " <sep> " + df["llm_output"]).tolist()


def tfidf_lr(train_df, test_df, field, labels=config.FAMILIES, return_model=False):
    vec = TfidfVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                          ngram_range=(1, 2), min_df=2, max_features=100000, sublinear_tf=True)
    Xtr, Xte = vec.fit_transform(_text(train_df, field)), vec.transform(_text(test_df, field))
    clf = LogisticRegression(max_iter=2000, C=4.0)
    clf.fit(Xtr, train_df["family"])
    pred = clf.predict(Xte)
    res = {"model": "TFIDF-LR", "field": field,
           "test_acc": accuracy_score(test_df["family"], pred),
           "test_macro_f1": f1_score(test_df["family"], pred, average="macro", labels=labels)}
    return (res, vec, clf) if return_model else res


def top_ngrams(vec, clf, k=15):
    names = np.array(vec.get_feature_names_out())
    return {c: names[np.argsort(clf.coef_[i])[::-1][:k]].tolist() for i, c in enumerate(clf.classes_)}
