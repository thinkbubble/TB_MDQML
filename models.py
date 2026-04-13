from dotenv import load_dotenv
import os
from tensorflow.keras import Sequential, layers

load_dotenv()

GPU_ASSIGNED = os.getenv("GPU_ASSIGNED")


LAYER_FACTORY = {
    "Conv2D": layers.Conv2D,
    "MaxPooling2D": layers.MaxPooling2D,
    "Flatten": layers.Flatten,
    "Dense": layers.Dense,
    "Dropout": layers.Dropout,
    "BatchNormalization": layers.BatchNormalization,
}


def build_tf_model(input_shape, num_classes, model_config):
    model = Sequential()
    model.add(layers.Input(shape=input_shape))

    for layer_config in model_config["layers"]:
        layer_type = layer_config["type"]
        params = dict(layer_config.get("params", {}))

        if "units" in params and params["units"] == "NUM_CLASSES":
            params["units"] = num_classes

        if layer_type not in LAYER_FACTORY:
            raise ValueError(f"Unsupported layer type: {layer_type}")

        layer_class = LAYER_FACTORY[layer_type]
        model.add(layer_class(**params))

    compile_config = model_config.get("compile", {})
    model.compile(
        optimizer=compile_config.get("optimizer", "adam"),
        loss=compile_config.get("loss", "categorical_crossentropy"),
        metrics=compile_config.get("metrics", ["accuracy"])
    )

    return model


def get_baseline_cnn_config():
    return {
        "name": "baseline_cnn_v1",
        "layers": [
            {
                "type": "Conv2D",
                "params": {
                    "filters": 32,
                    "kernel_size": (3, 3),
                    "activation": "relu",
                    "padding": "same"
                }
            },
            {
                "type": "MaxPooling2D",
                "params": {
                    "pool_size": (2, 2)
                }
            },
            {
                "type": "Conv2D",
                "params": {
                    "filters": 64,
                    "kernel_size": (3, 3),
                    "activation": "relu",
                    "padding": "same"
                }
            },
            {
                "type": "MaxPooling2D",
                "params": {
                    "pool_size": (2, 2)
                }
            },
            {
                "type": "Conv2D",
                "params": {
                    "filters": 128,
                    "kernel_size": (3, 3),
                    "activation": "relu",
                    "padding": "same"
                }
            },
            {
                "type": "MaxPooling2D",
                "params": {
                    "pool_size": (2, 2)
                }
            },
            {
                "type": "Flatten",
                "params": {}
            },
            {
                "type": "Dense",
                "params": {
                    "units": 128,
                    "activation": "relu"
                }
            },
            {
                "type": "Dropout",
                "params": {
                    "rate": 0.3
                }
            },
            {
                "type": "Dense",
                "params": {
                    "units": "NUM_CLASSES",
                    "activation": "softmax"
                }
            }
        ],
        "compile": {
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "metrics": ["accuracy"]
        }
    }