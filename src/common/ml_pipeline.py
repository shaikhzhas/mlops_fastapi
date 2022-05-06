"""
Function for ml pipeline
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier


def get_inference_pipeline(rf_config):
    """
    inference pipeline
    """
    categorical = [
        "workclass",
        "education",
        "native-country",
        "education-num",
        "marital-status",
        'occupation',
        'relationship',
        'race',
        'sex'
    ]
    categorical_preproc = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown='ignore')
    )
    numeric_features = [
        'age',
        'fnlgt',
        'capital-gain',
        'capital-loss',
        'hours-per-week'
    ]
    numeric_transformer = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler()
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_preproc, categorical)
        ],
        remainder="drop",  # This drops the columns that we do not transform
    )
    used_columns = list(itertools.chain.from_iterable(
        [x[2] for x in preprocessor.transformers]))
    sk_pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(**rf_config)),
        ]
    )
    return sk_pipe, used_columns


def plot_feature_importance(pipe, feat_names):
    """
    plotting feature importance
    """
    feat_names = np.array(
        pipe["preprocessor"].transformers[0][-1] + pipe["preprocessor"].transformers[1][-1]
    )
    feat_imp = pipe["classifier"].feature_importances_[: len(feat_names)]
    fig_feat_imp, sub_feat_imp = plt.subplots(figsize=(10, 10))
    idx = np.argsort(feat_imp)[::-1]
    sub_feat_imp.bar(
        range(
            feat_imp.shape[0]),
        feat_imp[idx],
        color="r",
        align="center")
    _ = sub_feat_imp.set_xticks(range(feat_imp.shape[0]))
    _ = sub_feat_imp.set_xticklabels(feat_names[idx], rotation=90)
    fig_feat_imp.tight_layout()
    return fig_feat_imp
