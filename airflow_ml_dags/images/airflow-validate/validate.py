import os
import pickle
import pandas as pd
import click
import json
from sklearn.naive_bayes import GaussianNB


@click.command("train")
@click.option("--input-dir")
@click.option("--models-dir")
def validate(input_dir: str, models_dir: str):
    data = pd.read_csv(os.path.join(input_dir, "data_val.csv"))
    target = pd.read_csv(os.path.join(input_dir, "target_val.csv"))

    with open(os.path.join(models_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)

    score = dict()
    score['score'] = model.score(data, target)

    with open(os.path.join(models_dir, "score.json"), "w") as f:
        json.dump(score, f)


if __name__ == '__main__':
    validate()
