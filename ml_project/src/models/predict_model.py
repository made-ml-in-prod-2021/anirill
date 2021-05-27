import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from typing import Union, Dict
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

SklearnClassifierModel = Union[KNeighborsClassifier, LogisticRegression]


def predict_model(model: SklearnClassifierModel, features: pd.DataFrame) -> np.ndarray:
    predicts = model.predict(features)
    return predicts


def evaluate_model(predicts: np.ndarray, target: pd.Series) -> Dict[str, float]:
    return {
        "roc_auc_score": roc_auc_score(target, predicts),
        "accuracy_score": accuracy_score(target, predicts),
        "f1_score": f1_score(target, predicts),
    }
