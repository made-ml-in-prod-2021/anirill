import os
import pickle
import pandas as pd
import click
import json

# PREDICTIONS_DIR = ""


@click.command("predict")
@click.option("--input-dir")
@click.option("--output-dir")
@click.option("--models-dir")
def predict(input_dir: str, output_dir, models_dir):
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    # target = pd.read_csv(os.path.join(input_dir, "target_val.csv"))

    with open(os.path.join(models_dir, "model.pkl"), "rb") as f:
        model = pickle.load(f)

    predictions = pd.DataFrame(model.predict(data))

    os.makedirs(output_dir, exist_ok=True)
    predictions.to_csv(os.path.join(output_dir, "predictions.csv"))


if __name__ == '__main__':
    predict()
