import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from ml_project.src.classes.classes import SplittingParams
# from dotenv import find_dotenv, load_dotenv


def load_data(path: str) -> pd.DataFrame:
    with open(path) as f:
        return pd.read_csv(f)
