import yaml
from dataclasses import dataclass, field
from marshmallow_dataclass import class_schema
from typing import List, Optional


@dataclass()
class LRParams:
    penalty: str = field(default="l1")
    tol: float = field(default=0.0001)
    C: float = field(default=0.2)
    solver: str = field(default="saga")
    max_iter: int = field(default=100)
# penalty='l1', tol=0.0001, C=.2, solver='saga', max_iter=100)


@dataclass()
class KNNParams:
    n_neighbors: int = field(default=9)


@dataclass()
class FeatureParams:
    target_col: Optional[str]


@dataclass()
class SplittingParams:
    val_size: float = field(default=0.2)
    random_state: int = field(default=42)


@dataclass()
class TrainingParams:
    model_type: str = field(default="LogisticRegression")
    model_param: dict = field(default=None)


@dataclass()
class MetricParams:
    metric_name: str = field(default="roc_auc_score")


@dataclass()
class TrainingPipelineParams:
    input_data_path: str
    model_path: str
    metric_path: str
    splitting_params: SplittingParams
    feature_params: FeatureParams
    train_params: TrainingParams
    metric_params: MetricParams


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    with open(path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))
