import pickle
import pandas as pd
import logging
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from typing import Union, NoReturn

from ..classes.classes import LRParams, KNNParams

APPLICATION_NAME = "train_pipeline"

SklearnClassifierModel = Union[KNeighborsClassifier, LogisticRegression]
logger = logging.getLogger(APPLICATION_NAME)


def train_model(
    features: pd.DataFrame,
    target: pd.Series,
    train_params: Union[LRParams, KNNParams],
) -> SklearnClassifierModel:
    if train_params.model_type == "RandomForestClassifier":
        model = KNeighborsClassifier(
            n_neighbors=train_params.n_neighbors,
        )
    elif train_params.model_type == "LogisticRegression":
        model = LogisticRegression(
            penalty=train_params.penalty,
            tol=train_params.tol,
            C=train_params.C,
            solver=train_params.solver,
            max_iter=train_params.max_iter,
        )
    else:
        raise NotImplementedError()
    model.fit(features, target)
    return model


def dump_model(model: SklearnClassifierModel, output: str) -> NoReturn:
    with open(output, "wb") as f:
        pickle.dump(model, f)
        logger.info(f"Model dumped at: {output}")


def load_model(input_: str) -> SklearnClassifierModel:
    with open(input_, "rb") as f:
        model = pickle.load(f)
        logger.info(f"Model loaded from: {input_}")
    return model
