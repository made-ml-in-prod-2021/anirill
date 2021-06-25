import logging
import os
import pickle
import time
from typing import List, Union, Optional

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, conlist
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from datetime import datetime

logger = logging.getLogger(__name__)


def load_object(path: str) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)


class HousePricesModel(BaseModel):
    data: List[conlist(Union[float, str, None], min_items=80, max_items=80)]
    features: List[str]


class PriceResponse(BaseModel):
    id: str
    price: float


model: Optional[Pipeline] = None
start_time: Optional[datetime] = None
WORK_TIME = 100


def make_predict(
    data: List, features: List[str], model: Pipeline,
) -> List[PriceResponse]:
    data = pd.DataFrame(data, columns=features)
    ids = [int(x) for x in data["Id"]]
    predicts = np.exp(model.predict(data))

    return [
        PriceResponse(id=id_, price=float(price)) for id_, price in zip(ids, predicts)
    ]


app = FastAPI()


@app.get("/")
def main():
    return "it is entry point of our predictor"


@app.on_event("startup")
def load_model():
    global model
    global start_time
    time.sleep(30)
    model_path = os.getenv("PATH_TO_MODEL")
    if model_path is None:
        err = f"PATH_TO_MODEL {model_path} is None"
        logger.error(err)
        raise RuntimeError(err)
    model = load_object(model_path)
    # time.sleep(90)
    # raise OSError("Application stop")
    start_time = datetime.now()


@app.get("/healthz")
def health() -> bool:
    if start_time is None:
        raise Exception("Early")
    if (datetime.now() - start_time).seconds > WORK_TIME:
        raise Exception("Late")
    return not (model is None)


@app.get("/predict/", response_model=List[PriceResponse])
def predict(request: HousePricesModel):
    return make_predict(request.data, request.features, model)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
