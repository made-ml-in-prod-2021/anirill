import os
import pandas as pd
import click
from sklearn.model_selection import train_test_split


@click.command("split")
@click.option("--input-dir")
@click.option("--output-dir")
def split(input_dir: str, output_dir):
    data = pd.read_csv(os.path.join(input_dir, "data.csv"))
    target = pd.read_csv(os.path.join(input_dir, "target.csv"))

    X_train, X_val, Y_train, Y_val = train_test_split(data, target, test_size=0.3, random_state=1)

    os.makedirs(output_dir, exist_ok=True)
    X_train.to_csv(os.path.join(output_dir, "data.csv"))
    Y_train.to_csv(os.path.join(output_dir, "target.csv"))
    X_val.to_csv(os.path.join(output_dir, "data_val.csv"))
    Y_val.to_csv(os.path.join(output_dir, "target_val.csv"))


if __name__ == '__main__':
    split()
