import os
import pickle
import pandas as pd
import click
from sklearn.naive_bayes import GaussianNB


@click.command("train")
@click.option("--input-dir")
@click.option("--output-dir")
def train(input_dir: str, output_dir: str):
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    target = pd.read_csv(os.path.join(input_dir, "target.csv"))

    model = GaussianNB()

    model.fit(data, target)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "target.csv"), "wb") as f:
        pickle.dump(model, f)


if __name__ == '__main__':
    train()
