import os
import pandas as pd
import click


@click.command("preprocess")
@click.option("--input-dir")
@click.option("--output-dir")
def preprocess(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    data = pd.read_csv(os.path.join(input_dir, "data.csv"), index_col=None)
    # print(data[:5])
    # data = data.drop(columns=2)  # drop bad feats
    data.fillna(0)
    data.to_csv(os.path.join(output_dir, "data.csv"), index=False)

    target_path = os.path.join(input_dir, "target.csv")

    if os.path.exists(target_path):
        target = pd.read_csv(target_path, index_col=None)
        target.fillna(0)
        target.to_csv(os.path.join(output_dir, "target.csv"), index=False)


if __name__ == '__main__':
    preprocess()
