from dotenv import load_dotenv
import os
from tensorflow.keras import Sequential, layers

load_dotenv()

GPU_ASSIGNED = os.getenv("GPU_ASSIGNED")


LAYER_FACTORY = {
    "Dense": layers.Dense,
    "Dropout": layers.Dropout,
    "Flatten": layers.Flatten,
    "Reshape": layers.Reshape,
    "Conv1D": layers.Conv1D,
    "Conv2D": layers.Conv2D,
    "Conv3D": layers.Conv3D,
    "SeparableConv2D": layers.SeparableConv2D,
    "DepthwiseConv2D": layers.DepthwiseConv2D,
    "MaxPooling1D": layers.MaxPooling1D,
    "MaxPooling2D": layers.MaxPooling2D,
    "MaxPooling3D": layers.MaxPooling3D,
    "AveragePooling1D": layers.AveragePooling1D,
    "AveragePooling2D": layers.AveragePooling2D,
    "AveragePooling3D": layers.AveragePooling3D,
    "GlobalAveragePooling1D": layers.GlobalAveragePooling1D,
    "GlobalAveragePooling2D": layers.GlobalAveragePooling2D,
    "GlobalAveragePooling3D": layers.GlobalAveragePooling3D,
    "BatchNormalization": layers.BatchNormalization,
    "LayerNormalization": layers.LayerNormalization,
    "Embedding": layers.Embedding,
    "SimpleRNN": layers.SimpleRNN,
    "LSTM": layers.LSTM,
    "GRU": layers.GRU,
    "Activation": layers.Activation,
    "ReLU": layers.ReLU,
    "LeakyReLU": layers.LeakyReLU,
    "GaussianNoise": layers.GaussianNoise,
}


CNN_FACTORY = {
    "baseline_cnn_v1": {
        "model_type": "sequential",
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
    },
    "regularized_cnn_v1": {
        "model_type": "sequential",
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
                "type": "BatchNormalization",
                "params": {}
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
                "type": "BatchNormalization",
                "params": {}
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
                "type": "BatchNormalization",
                "params": {}
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
                    "units": 256,
                    "activation": "relu"
                }
            },
            {
                "type": "Dropout",
                "params": {
                    "rate": 0.4
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
}


DNN_FACTORY = {
    "baseline_dnn_v1": {
        "model_type": "sequential",
        "layers": [
            {
                "type": "Flatten",
                "params": {}
            },
            {
                "type": "Dense",
                "params": {
                    "units": 512,
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
                    "units": 256,
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
}


TRANSFORMER_FACTORY = {
    "baseline_transformer_v1": {
        "model_type": "transformer",
        "embed_dim": 64,
        "num_heads": 4,
        "ff_dim": 128,
        "dropout_rate": 0.1,
        "mlp_units": 128,
        "compile": {
            "optimizer": "adam",
            "loss": "categorical_crossentropy",
            "metrics": ["accuracy"]
        }
    }
}


def get_model_config(factory_name, config_name):
    factories = {
        "cnn": CNN_FACTORY,
        "dnn": DNN_FACTORY,
        "transformer": TRANSFORMER_FACTORY,
    }

    if factory_name not in factories:
        raise ValueError(f"Unsupported factory: {factory_name}")

    factory = factories[factory_name]

    if config_name not in factory:
        raise ValueError(
            f"Config '{config_name}' not found in {factory_name.upper()}_FACTORY"
        )

    return factory[config_name]


def build_tf_model(input_shape, num_classes, model_config):
    model_type = model_config.get("model_type", "sequential")

    if model_type != "sequential":
        raise ValueError(
            f"build_tf_model only supports sequential configs. "
            f"Received model_type='{model_type}'"
        )

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


def build_transformer_model(input_shape, num_classes, model_config):
    raise NotImplementedError(
        "Transformer builder is planned but not yet implemented."
    )