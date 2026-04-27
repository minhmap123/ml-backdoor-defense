import inspect

from .base import BaseTabularModel
from .ft_transformer import FTTransformer, FTTransformerClassifier
from .mlp import MLP, MLPClassifier
from .resnet import ResNet, ResNetClassifier
from .saint import SAINT, SAINTClassifier
from .tabnet import TabNet, TabNetClassifier
from .train import train_torch_model

MODEL_REGISTRY = {}


def register_model(name: str, cls) -> None:
    MODEL_REGISTRY[str(name).lower()] = cls


def get_model(model_cfg):
    name = str(model_cfg["name"]).lower()
    model_cls = MODEL_REGISTRY[name]
    raw_kwargs = {k: v for k, v in dict(model_cfg).items() if k != "name"}
    signature = inspect.signature(model_cls.__init__)
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_kwargs:
        kwargs = raw_kwargs
    else:
        allowed = {
            parameter_name
            for parameter_name, parameter in signature.parameters.items()
            if parameter_name != "self"
            and parameter.kind in {
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            }
        }
        kwargs = {k: v for k, v in raw_kwargs.items() if k in allowed}
    return model_cls(**kwargs)


register_model("mlp", MLP)
register_model("mlpclassifier", MLPClassifier)
register_model("resnet", ResNet)
register_model("resnetclassifier", ResNetClassifier)
register_model("ft_transformer", FTTransformer)
register_model("fttransformer", FTTransformer)
register_model("fttransformerclassifier", FTTransformerClassifier)
register_model("saint", SAINT)
register_model("saintclassifier", SAINTClassifier)
register_model("tabnet", TabNet)
register_model("tabnetclassifier", TabNetClassifier)


__all__ = [
    "BaseTabularModel",
    "FTTransformer",
    "FTTransformerClassifier",
    "MLP",
    "MLPClassifier",
    "ResNet",
    "ResNetClassifier",
    "SAINT",
    "SAINTClassifier",
    "TabNet",
    "TabNetClassifier",
    "MODEL_REGISTRY",
    "register_model",
    "get_model",
    "train_torch_model",
]
