import logging
import logging.config
import yaml
import json
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from ml_project.src.classes.classes import TrainingPipelineParams, SplittingParams, read_training_pipeline_params
from ml_project.src.data.make_dataset import load_data
from ml_project.src.features.build_features import build_features
from ml_project.src.models.train_model import train_model, dump_model, load_model
from ml_project.src.models.predict_model import predict_model, evaluate_model


APPLICATION_NAME = "train_pipeline"
LOGGING_CONF_FILE = "../configs/logging_conf.yml"
DEFAULT_CONFIG_PATH = "../configs/train_config.yaml"
DEFAULT_MODE = "train predict"
TRAIN_MODE = "train"
PREDICT_MODE = "predict"
ALLOWED_MODES = [DEFAULT_MODE, TRAIN_MODE, PREDICT_MODE]

logger = logging.getLogger(APPLICATION_NAME)


def setup_logging():
    with open(LOGGING_CONF_FILE) as config_fin:
        logging.config.dictConfig(yaml.safe_load(config_fin))


def setup_parser(parser: ArgumentParser):
    subparsers = parser.add_subparsers(help="choose command")
    build_parser = subparsers.add_parser(
        "train-model", help="train a model on dataset",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    build_parser.add_argument(
        "-c", "--config",
        dest="config_path",
        default=DEFAULT_CONFIG_PATH,
        # metavar="config_path",
        help="path to config",
    )
    build_parser.add_argument(
        "-s", "--select-mode",
        dest="mode",
        default=DEFAULT_MODE,
        # metavar="index_path",
        help="train or predict",
    )


def train_pipeline(params: TrainingPipelineParams, mode: str):
    logger.debug("Starting training pipeline")
    logger.debug("Loading data")
    data = load_data(params.input_data_path)
    train, test = train_test_split(
        data,
        test_size=SplittingParams.val_size,
        random_state=SplittingParams.random_state
    )
    logger.debug("Extracting features")
    X_train, y_train = build_features(train)
    X_test, y_test = build_features(test)

    if mode in [TRAIN_MODE, DEFAULT_MODE]:
        logger.debug("Training model")
        model = train_model(X_train, y_train, params.train_params)
        # with open(params.model_path, 'w') as f:
        # logger.info(f"Model dump: {params.model_path}")
        dump_model(model, params.model_path)
    else:
        logger.debug("Loading model")
        # with open(params.model_path, 'r') as f:
        # logger.info(f"Model uploaded with: {params.model_path}")
        model = load_model(params.model_path)

    if mode in [PREDICT_MODE, DEFAULT_MODE] and model is not None:
        logger.debug("Predicting")
        predictions = predict_model(model, X_test)
        metrics = evaluate_model(predictions, y_test)
        logger.debug(f"Metrics: {metrics}")
        with open(params.metric_path, 'w') as f:
            json.dump(metrics, f)
        return model, metrics
    else:
        return model, None


def start_train_pipeline(config_path: str, mode: str):
    if mode not in ALLOWED_MODES:
        logger.warning("Incorrect mode, try again.")
        return
    params = read_training_pipeline_params(config_path)
    train_pipeline(params=params, mode=mode)


def main():
    setup_logging()
    parser = ArgumentParser(
        prog="model train or predict",
        description="homework1",
        # formatter_class=ArgumentDefaultsHelpFormatter,
    )
    # parser.set_defaults(func=lambda args: parser.print_help())
    setup_parser(parser)
    arguments = parser.parse_args()
    logger.info(arguments)


if __name__ == "__main__":
    main()
