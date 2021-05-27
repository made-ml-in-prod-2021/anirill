import pandas as pd
from sklearn.preprocessing import scale

from ml_project.src.classes.classes import FeatureParams


def build_features(df: pd.DataFrame):
    data_features = df.drop('target', axis=1)
    data_labels = df.target
    return scale(data_features), data_labels.to_numpy()
