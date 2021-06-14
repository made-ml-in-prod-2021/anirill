import os
import pickle
import pandas as pd
import click
import json

@click.command("predict")
@click.option("--input-dir")
@click.option("--models-dir")
def predict(input_dir: str, models_dir):
    data = pd.read_csv(os.path.join(input_dir, "data_val.csv"))
    target = pd.read_csv(os.path.join(input_dir, "target_val.csv"))

    with open(os.path.join(models_dir, "model.pkl"), "r") as f:
        model = pickle.load(f)

    score = dict()
    score['score'] = model.score(data, target)

    with open(os.path.join(models_dir, "score.json"), "wb") as f:
        json.dump(score, f)


    os.makedirs(output_dir, exist_ok=True)
    data.to_csv(os.path.join(output_dir, "data.csv"))


if __name__ == '__main__':
    predict()
