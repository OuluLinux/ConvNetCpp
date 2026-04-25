#!/usr/bin/env python3
"""Generate NeuralEditor template projects from ConvNetCpp examples.

Output layout:
  nn/<Example>/<Example>.nnsln
  nn/<Example>/<Example>.nnprj
  nn/<Example>/main.nngrf
  nn/manifest.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def graph_common_defs() -> str:
    return """node convnet.module.layer.input:
\tout layer_stack : LAYER_STACK
\tinput_width : int = 28 (EditIntSpin)
\tinput_height : int = 28 (EditIntSpin)
\tinput_depth : int = 1 (EditIntSpin)

node convnet.module.layer.fc:
\tin layer_stack : LAYER_STACK
\tout layer_stack : LAYER_STACK
\tneuron_count : int = 64 (EditIntSpin)

node convnet.module.layer.conv:
\tin layer_stack : LAYER_STACK
\tout layer_stack : LAYER_STACK
\tkernel_size : int = 5 (EditIntSpin)
\tout_channels : int = 16 (EditIntSpin)
\tstride : int = 1 (EditIntSpin)
\tpadding : int = 2 (EditIntSpin)

node convnet.module.layer.pool:
\tin layer_stack : LAYER_STACK
\tout layer_stack : LAYER_STACK
\tkernel_size : int = 2 (EditIntSpin)
\tstride : int = 2 (EditIntSpin)
\tpadding : int = 0 (EditIntSpin)

node convnet.module.layer.relu:
\tin layer_stack : LAYER_STACK
\tout layer_stack : LAYER_STACK

node convnet.module.layer.sigmoid:
\tin layer_stack : LAYER_STACK
\tout layer_stack : LAYER_STACK

node convnet.module.layer.regression:
\tin layer_stack : LAYER_STACK
\tout layer_stack : LAYER_STACK

node convnet.module.layer.softmax:
\tin layer_stack : LAYER_STACK
\tout layer_stack : LAYER_STACK
\tclass_count : int = 10 (EditIntSpin)

node convnet.module.layer.vit_patch_embed:
\tin layer_stack : LAYER_STACK
\tout layer_stack : LAYER_STACK
\tpatch_size : int = 4 (EditIntSpin)
\tembed_dim : int = 128 (EditIntSpin)
\tnum_patches : int = 64 (EditIntSpin)

node convnet.module.layer.vit_encoder:
\tin layer_stack : LAYER_STACK
\tout layer_stack : LAYER_STACK
\tembed_dim : int = 128 (EditIntSpin)
\tnum_heads : int = 8 (EditIntSpin)
\tff_dim : int = 512 (EditIntSpin)
\tnum_layers : int = 3 (EditIntSpin)
\tdropout_rate : float = 0.1 (EditDoubleSpin)

node convnet.module.layer.vit_classifier:
\tin layer_stack : LAYER_STACK
\tout layer_stack : LAYER_STACK
\tnum_classes : int = 10 (EditIntSpin)
\tembed_dim : int = 128 (EditIntSpin)

node convnet.compile:
\tin layer_stack : LAYER_STACK
\tout model : MODEL
\tout report : REPORT
\tmode : string = "compile" (DropList)

node convnet.module.trainer.adadelta:
\tin model : MODEL
\tout model : MODEL
\tlearning_rate : float = 0.01 (EditDoubleSpin)
\tbatch_size : int = 1 (EditIntSpin)
\tl2_decay : float = 0.001 (EditDoubleSpin)

node convnet.module.trainer.adam:
\tin model : MODEL
\tout model : MODEL
\tlearning_rate : float = 0.001 (EditDoubleSpin)
\tbatch_size : int = 32 (EditIntSpin)
\tl2_decay : float = 0.0001 (EditDoubleSpin)

node convnet.module.trainer.sgd:
\tin model : MODEL
\tout model : MODEL
\tlearning_rate : float = 0.01 (EditDoubleSpin)
\tmomentum : float = 0.9 (EditDoubleSpin)
\tbatch_size : int = 8 (EditIntSpin)
\tl2_decay : float = 0.001 (EditDoubleSpin)

node convnet.module.trainer.adagrad:
\tin model : MODEL
\tout model : MODEL
\tlearning_rate : float = 0.01 (EditDoubleSpin)
\teps : float = 0.000001 (EditDoubleSpin)
\tbatch_size : int = 8 (EditIntSpin)
\tl2_decay : float = 0.001 (EditDoubleSpin)

node convnet.module.trainer.windowgrad:
\tin model : MODEL
\tout model : MODEL
\tlearning_rate : float = 0.01 (EditDoubleSpin)
\teps : float = 0.000001 (EditDoubleSpin)
\tro : float = 0.95 (EditDoubleSpin)
\tbatch_size : int = 8 (EditIntSpin)
\tl2_decay : float = 0.001 (EditDoubleSpin)

node convnet.train:
\tin model : MODEL
\tout metrics : METRICS
\tepochs : int = 20 (EditIntSpin)
\tlearning_rate : float = 0.001 (EditDoubleSpin)
"""


def graph_regression1d() -> str:
    return f"""// Template conversion from examples/Regression1D

{graph_common_defs()}

net regression1d:
\tn_input: convnet.module.layer.input
\t\tinput_width = 1
\t\tinput_height = 1
\t\tinput_depth = 1

\tn_fc1: convnet.module.layer.fc
\t\tneuron_count = 20

\tn_relu1: convnet.module.layer.relu

\tn_fc2: convnet.module.layer.fc
\t\tneuron_count = 20

\tn_sigmoid1: convnet.module.layer.sigmoid

\tn_regression: convnet.module.layer.regression

\tn_compile: convnet.compile

\tn_adadelta: convnet.module.trainer.adadelta
\t\tlearning_rate = 0.01
\t\tbatch_size = 1
\t\tl2_decay = 0.001

\tn_train: convnet.train
\t\tepochs = 200
\t\tlearning_rate = 0.01

\tn_input.layer_stack -> n_fc1.layer_stack
\tn_fc1.layer_stack -> n_relu1.layer_stack
\tn_relu1.layer_stack -> n_fc2.layer_stack
\tn_fc2.layer_stack -> n_sigmoid1.layer_stack
\tn_sigmoid1.layer_stack -> n_regression.layer_stack
\tn_regression.layer_stack -> n_compile.layer_stack
\tn_compile.model -> n_adadelta.model
\tn_adadelta.model -> n_train.model
"""


def graph_trainer_benchmark() -> str:
    return f"""// Template conversion from examples/TrainerBenchmark

{graph_common_defs()}

net trainer_benchmark:
\tn_input: convnet.module.layer.input
\t\tinput_width = 24
\t\tinput_height = 24
\t\tinput_depth = 1

\tn_fc1: convnet.module.layer.fc
\t\tneuron_count = 20
\tn_relu1: convnet.module.layer.relu
\tn_fc2: convnet.module.layer.fc
\t\tneuron_count = 20
\tn_relu2: convnet.module.layer.relu
\tn_softmax: convnet.module.layer.softmax
\t\tclass_count = 10
\tn_compile: convnet.compile

\tn_sgd: convnet.module.trainer.sgd
\t\tlearning_rate = 0.1
\t\tmomentum = 0.0
\t\tbatch_size = 8
\t\tl2_decay = 0.001
\tn_adagrad: convnet.module.trainer.adagrad
\tn_windowgrad: convnet.module.trainer.windowgrad

\tn_train_sgd: convnet.train
\t\tepochs = 30
\t\tlearning_rate = 0.1
\tn_train_adagrad: convnet.train
\t\tepochs = 30
\tn_train_windowgrad: convnet.train
\t\tepochs = 30

\tn_input.layer_stack -> n_fc1.layer_stack
\tn_fc1.layer_stack -> n_relu1.layer_stack
\tn_relu1.layer_stack -> n_fc2.layer_stack
\tn_fc2.layer_stack -> n_relu2.layer_stack
\tn_relu2.layer_stack -> n_softmax.layer_stack
\tn_softmax.layer_stack -> n_compile.layer_stack
\tn_compile.model -> n_sgd.model
\tn_compile.model -> n_adagrad.model
\tn_compile.model -> n_windowgrad.model
\tn_sgd.model -> n_train_sgd.model
\tn_adagrad.model -> n_train_adagrad.model
\tn_windowgrad.model -> n_train_windowgrad.model
"""


def graph_regression_painter() -> str:
    return f"""// Template conversion from examples/RegressionPainter

{graph_common_defs()}

net regression_painter:
\tn_input: convnet.module.layer.input
\t\tinput_width = 1
\t\tinput_height = 1
\t\tinput_depth = 2

\tn_fc1: convnet.module.layer.fc
\t\tneuron_count = 64
\tn_relu1: convnet.module.layer.relu
\tn_fc2: convnet.module.layer.fc
\t\tneuron_count = 64
\tn_relu2: convnet.module.layer.relu
\tn_fc3: convnet.module.layer.fc
\t\tneuron_count = 32
\tn_relu3: convnet.module.layer.relu
\tn_regression: convnet.module.layer.regression
\tn_compile: convnet.compile

\tn_sgd: convnet.module.trainer.sgd
\t\tlearning_rate = 0.01
\t\tmomentum = 0.9
\t\tbatch_size = 5
\t\tl2_decay = 0.0

\tn_train: convnet.train
\t\tepochs = 100
\t\tlearning_rate = 0.01

\tn_input.layer_stack -> n_fc1.layer_stack
\tn_fc1.layer_stack -> n_relu1.layer_stack
\tn_relu1.layer_stack -> n_fc2.layer_stack
\tn_fc2.layer_stack -> n_relu2.layer_stack
\tn_relu2.layer_stack -> n_fc3.layer_stack
\tn_fc3.layer_stack -> n_relu3.layer_stack
\tn_relu3.layer_stack -> n_regression.layer_stack
\tn_regression.layer_stack -> n_compile.layer_stack
\tn_compile.model -> n_sgd.model
\tn_sgd.model -> n_train.model
"""


def graph_network_optimization() -> str:
    return f"""// Template conversion from examples/NetworkOptimization

{graph_common_defs()}

net network_optimization:
\tn_input: convnet.module.layer.input
\t\tinput_width = 1
\t\tinput_height = 1
\t\tinput_depth = 16

\tn_fc1: convnet.module.layer.fc
\t\tneuron_count = 48
\tn_relu1: convnet.module.layer.relu
\tn_fc2: convnet.module.layer.fc
\t\tneuron_count = 24
\tn_relu2: convnet.module.layer.relu
\tn_softmax: convnet.module.layer.softmax
\t\tclass_count = 3
\tn_compile: convnet.compile

\tn_adam: convnet.module.trainer.adam
\t\tlearning_rate = 0.001
\t\tbatch_size = 32
\t\tl2_decay = 0.0001

\tn_train: convnet.train
\t\tepochs = 40
\t\tlearning_rate = 0.001

\tn_input.layer_stack -> n_fc1.layer_stack
\tn_fc1.layer_stack -> n_relu1.layer_stack
\tn_relu1.layer_stack -> n_fc2.layer_stack
\tn_fc2.layer_stack -> n_relu2.layer_stack
\tn_relu2.layer_stack -> n_softmax.layer_stack
\tn_softmax.layer_stack -> n_compile.layer_stack
\tn_compile.model -> n_adam.model
\tn_adam.model -> n_train.model
"""


def graph_vision_transformer() -> str:
    return f"""// Template conversion from examples/VisionTransformer

{graph_common_defs()}

net vision_transformer:
\tn_input: convnet.module.layer.input
\t\tinput_width = 32
\t\tinput_height = 32
\t\tinput_depth = 3

\tn_patch: convnet.module.layer.vit_patch_embed
\t\tpatch_size = 4
\t\tembed_dim = 128
\t\tnum_patches = 64

\tn_encoder: convnet.module.layer.vit_encoder
\t\tembed_dim = 128
\t\tnum_heads = 8
\t\tff_dim = 512
\t\tnum_layers = 3
\t\tdropout_rate = 0.1

\tn_classifier: convnet.module.layer.vit_classifier
\t\tnum_classes = 10
\t\tembed_dim = 128

\tn_compile: convnet.compile
\tn_adam: convnet.module.trainer.adam
\t\tlearning_rate = 0.001
\t\tbatch_size = 64
\t\tl2_decay = 0.0001
\tn_train: convnet.train
\t\tepochs = 60
\t\tlearning_rate = 0.001

\tn_input.layer_stack -> n_patch.layer_stack
\tn_patch.layer_stack -> n_encoder.layer_stack
\tn_encoder.layer_stack -> n_classifier.layer_stack
\tn_classifier.layer_stack -> n_compile.layer_stack
\tn_compile.model -> n_adam.model
\tn_adam.model -> n_train.model
"""


def graph_classify2d() -> str:
    return f"""// Template conversion from examples/Classify2D

{graph_common_defs()}

net classify2d:
\tn_input: convnet.module.layer.input
\t\tinput_width = 1
\t\tinput_height = 1
\t\tinput_depth = 2

\tn_fc1: convnet.module.layer.fc
\t\tneuron_count = 6
\tn_relu1: convnet.module.layer.relu
\tn_fc2: convnet.module.layer.fc
\t\tneuron_count = 2
\tn_relu2: convnet.module.layer.relu
\tn_softmax: convnet.module.layer.softmax
\t\tclass_count = 2
\tn_compile: convnet.compile
\tn_sgd: convnet.module.trainer.sgd
\t\tlearning_rate = 0.01
\t\tmomentum = 0.1
\t\tbatch_size = 10
\t\tl2_decay = 0.001
\tn_train: convnet.train
\t\tepochs = 40
\t\tlearning_rate = 0.01

\tn_input.layer_stack -> n_fc1.layer_stack
\tn_fc1.layer_stack -> n_relu1.layer_stack
\tn_relu1.layer_stack -> n_fc2.layer_stack
\tn_fc2.layer_stack -> n_relu2.layer_stack
\tn_relu2.layer_stack -> n_softmax.layer_stack
\tn_softmax.layer_stack -> n_compile.layer_stack
\tn_compile.model -> n_sgd.model
\tn_sgd.model -> n_train.model
"""


def graph_classify_images_mnist() -> str:
    return f"""// Template conversion from examples/ClassifyImages (MNIST learner)

{graph_common_defs()}

net classify_images_mnist:
\tn_input: convnet.module.layer.input
\t\tinput_width = 24
\t\tinput_height = 24
\t\tinput_depth = 1

\tn_conv1: convnet.module.layer.conv
\t\tkernel_size = 5
\t\tout_channels = 8
\t\tstride = 1
\t\tpadding = 2
\tn_pool1: convnet.module.layer.pool
\t\tkernel_size = 2
\t\tstride = 2
\t\tpadding = 0
\tn_conv2: convnet.module.layer.conv
\t\tkernel_size = 5
\t\tout_channels = 16
\t\tstride = 1
\t\tpadding = 2
\tn_pool2: convnet.module.layer.pool
\t\tkernel_size = 3
\t\tstride = 3
\t\tpadding = 0
\tn_softmax: convnet.module.layer.softmax
\t\tclass_count = 10
\tn_compile: convnet.compile
\tn_adadelta: convnet.module.trainer.adadelta
\t\tlearning_rate = 0.01
\t\tbatch_size = 20
\t\tl2_decay = 0.001
\tn_train: convnet.train
\t\tepochs = 30
\t\tlearning_rate = 0.01

\tn_input.layer_stack -> n_conv1.layer_stack
\tn_conv1.layer_stack -> n_pool1.layer_stack
\tn_pool1.layer_stack -> n_conv2.layer_stack
\tn_conv2.layer_stack -> n_pool2.layer_stack
\tn_pool2.layer_stack -> n_softmax.layer_stack
\tn_softmax.layer_stack -> n_compile.layer_stack
\tn_compile.model -> n_adadelta.model
\tn_adadelta.model -> n_train.model
"""


def graph_classify_images_cifar() -> str:
    return f"""// Template conversion from examples/ClassifyImages (CIFAR-10 learner)

{graph_common_defs()}

net classify_images_cifar:
\tn_input: convnet.module.layer.input
\t\tinput_width = 30
\t\tinput_height = 30
\t\tinput_depth = 3

\tn_conv1: convnet.module.layer.conv
\t\tkernel_size = 5
\t\tout_channels = 16
\t\tstride = 1
\t\tpadding = 2
\tn_pool1: convnet.module.layer.pool
\t\tkernel_size = 2
\t\tstride = 2
\t\tpadding = 0
\tn_conv2: convnet.module.layer.conv
\t\tkernel_size = 5
\t\tout_channels = 20
\t\tstride = 1
\t\tpadding = 2
\tn_pool2: convnet.module.layer.pool
\t\tkernel_size = 3
\t\tstride = 2
\t\tpadding = 0
\tn_conv3: convnet.module.layer.conv
\t\tkernel_size = 5
\t\tout_channels = 20
\t\tstride = 1
\t\tpadding = 2
\tn_pool3: convnet.module.layer.pool
\t\tkernel_size = 3
\t\tstride = 2
\t\tpadding = 0
\tn_softmax: convnet.module.layer.softmax
\t\tclass_count = 10
\tn_compile: convnet.compile
\tn_adadelta: convnet.module.trainer.adadelta
\t\tlearning_rate = 0.01
\t\tbatch_size = 20
\t\tl2_decay = 0.001
\tn_train: convnet.train
\t\tepochs = 40
\t\tlearning_rate = 0.01

\tn_input.layer_stack -> n_conv1.layer_stack
\tn_conv1.layer_stack -> n_pool1.layer_stack
\tn_pool1.layer_stack -> n_conv2.layer_stack
\tn_conv2.layer_stack -> n_pool2.layer_stack
\tn_pool2.layer_stack -> n_conv3.layer_stack
\tn_conv3.layer_stack -> n_pool3.layer_stack
\tn_pool3.layer_stack -> n_softmax.layer_stack
\tn_softmax.layer_stack -> n_compile.layer_stack
\tn_compile.model -> n_adadelta.model
\tn_adadelta.model -> n_train.model
"""


def graph_efficientnet() -> str:
    return f"""// Template conversion from examples/EfficientNet (simplified B0-style flow)

{graph_common_defs()}

net efficientnet:
\tn_input: convnet.module.layer.input
\t\tinput_width = 224
\t\tinput_height = 224
\t\tinput_depth = 3

\tn_conv1: convnet.module.layer.conv
\t\tkernel_size = 3
\t\tout_channels = 32
\t\tstride = 2
\t\tpadding = 1
\tn_relu1: convnet.module.layer.relu
\tn_pool1: convnet.module.layer.pool
\t\tkernel_size = 2
\t\tstride = 2
\t\tpadding = 0
\tn_conv2: convnet.module.layer.conv
\t\tkernel_size = 3
\t\tout_channels = 80
\t\tstride = 2
\t\tpadding = 1
\tn_relu2: convnet.module.layer.relu
\tn_pool2: convnet.module.layer.pool
\t\tkernel_size = 2
\t\tstride = 2
\t\tpadding = 0
\tn_conv3: convnet.module.layer.conv
\t\tkernel_size = 1
\t\tout_channels = 320
\t\tstride = 1
\t\tpadding = 0
\tn_relu3: convnet.module.layer.relu
\tn_fc: convnet.module.layer.fc
\t\tneuron_count = 1000
\tn_softmax: convnet.module.layer.softmax
\t\tclass_count = 1000
\tn_compile: convnet.compile
\tn_adam: convnet.module.trainer.adam
\t\tlearning_rate = 0.001
\t\tbatch_size = 64
\t\tl2_decay = 0.00001
\tn_train: convnet.train
\t\tepochs = 60
\t\tlearning_rate = 0.001

\tn_input.layer_stack -> n_conv1.layer_stack
\tn_conv1.layer_stack -> n_relu1.layer_stack
\tn_relu1.layer_stack -> n_pool1.layer_stack
\tn_pool1.layer_stack -> n_conv2.layer_stack
\tn_conv2.layer_stack -> n_relu2.layer_stack
\tn_relu2.layer_stack -> n_pool2.layer_stack
\tn_pool2.layer_stack -> n_conv3.layer_stack
\tn_conv3.layer_stack -> n_relu3.layer_stack
\tn_relu3.layer_stack -> n_fc.layer_stack
\tn_fc.layer_stack -> n_softmax.layer_stack
\tn_softmax.layer_stack -> n_compile.layer_stack
\tn_compile.model -> n_adam.model
\tn_adam.model -> n_train.model
"""


def graph_swin_transformer() -> str:
    return f"""// Template conversion from examples/SwinTransformer (ViT-compatible approximation)

{graph_common_defs()}

net swin_transformer:
\tn_input: convnet.module.layer.input
\t\tinput_width = 32
\t\tinput_height = 32
\t\tinput_depth = 3

\tn_patch: convnet.module.layer.vit_patch_embed
\t\tpatch_size = 2
\t\tembed_dim = 96
\t\tnum_patches = 256
\tn_encoder: convnet.module.layer.vit_encoder
\t\tembed_dim = 96
\t\tnum_heads = 6
\t\tff_dim = 384
\t\tnum_layers = 2
\t\tdropout_rate = 0.1
\tn_classifier: convnet.module.layer.vit_classifier
\t\tnum_classes = 10
\t\tembed_dim = 96
\tn_compile: convnet.compile
\tn_adam: convnet.module.trainer.adam
\t\tlearning_rate = 0.001
\t\tbatch_size = 32
\t\tl2_decay = 0.0001
\tn_train: convnet.train
\t\tepochs = 80
\t\tlearning_rate = 0.001

\tn_input.layer_stack -> n_patch.layer_stack
\tn_patch.layer_stack -> n_encoder.layer_stack
\tn_encoder.layer_stack -> n_classifier.layer_stack
\tn_classifier.layer_stack -> n_compile.layer_stack
\tn_compile.model -> n_adam.model
\tn_adam.model -> n_train.model
"""


EXAMPLE_GRAPHS = {
    "Regression1D": graph_regression1d,
    "TrainerBenchmark": graph_trainer_benchmark,
    "RegressionPainter": graph_regression_painter,
    "NetworkOptimization": graph_network_optimization,
    "VisionTransformer": graph_vision_transformer,
    "Classify2D": graph_classify2d,
    "ClassifyImagesMNIST": graph_classify_images_mnist,
    "ClassifyImagesCIFAR": graph_classify_images_cifar,
    "EfficientNet": graph_efficientnet,
    "SwinTransformer": graph_swin_transformer,
}


def normalize_name(name: str) -> str:
    name = Path(name).name.strip()
    if name.lower().endswith(".nnsln"):
        name = Path(name).stem
    return name


def write_example(base_dir: Path, name: str, graph_text: str, source_example: str) -> dict:
    out_dir = base_dir / "nn" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    prj_name = f"{name}.nnprj"
    sln_name = f"{name}.nnsln"

    sln_data = {
        "version": 1,
        "name": name,
        "projects": [prj_name],
        "active_project": prj_name,
        "metadata": {"source_example": f"examples/{source_example}", "generator": "convert_examples_to_nn.py"},
    }
    prj_data = {
        "version": 1,
        "name": name,
        "graphs": ["main.nngrf"],
        "startup_graph": "main.nngrf",
        "metadata": {"source_example": f"examples/{source_example}", "generator": "convert_examples_to_nn.py"},
    }

    (out_dir / sln_name).write_text(json.dumps(sln_data, indent=2) + "\n", encoding="utf-8")
    (out_dir / prj_name).write_text(json.dumps(prj_data, indent=2) + "\n", encoding="utf-8")
    (out_dir / "main.nngrf").write_text(graph_text.strip() + "\n", encoding="utf-8")
    solutions = [f"nn/{name}/{sln_name}"]
    projects = [f"nn/{name}/{prj_name}"]
    graphs = [f"nn/{name}/main.nngrf"]
    return {
        "name": name,
        "source_example": source_example,
        "solution": solutions[0],
        "project": projects[0],
        "graph": graphs[0],
        "solutions": solutions,
        "projects": projects,
        "graphs": graphs,
    }


def write_classify_images_suite(base_dir: Path, name: str) -> dict:
    out_dir = base_dir / "nn" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    mnist_prj = "MnistLearner.nnprj"
    cifar_prj = "CifarLearner.nnprj"
    mnist_graph = "mnist_learner.nngrf"
    cifar_graph = "cifar_learner.nngrf"
    suite_sln = f"{name}.nnsln"
    cifar_sln = f"{name}_CIFAR.nnsln"

    suite_sln_data = {
        "version": 1,
        "name": name,
        "projects": [mnist_prj, cifar_prj],
        "active_project": mnist_prj,
        "metadata": {"source_example": "examples/ClassifyImages", "generator": "convert_examples_to_nn.py"},
    }
    cifar_sln_data = {
        "version": 1,
        "name": f"{name}_CIFAR",
        "projects": [cifar_prj],
        "active_project": cifar_prj,
        "metadata": {"source_example": "examples/ClassifyImages", "generator": "convert_examples_to_nn.py"},
    }
    mnist_prj_data = {
        "version": 1,
        "name": "MnistLearner",
        "graphs": [mnist_graph],
        "startup_graph": mnist_graph,
        "metadata": {"source_example": "examples/ClassifyImages", "variant": "mnist_learner", "generator": "convert_examples_to_nn.py"},
    }
    cifar_prj_data = {
        "version": 1,
        "name": "CifarLearner",
        "graphs": [cifar_graph],
        "startup_graph": cifar_graph,
        "metadata": {"source_example": "examples/ClassifyImages", "variant": "cifar_learner", "generator": "convert_examples_to_nn.py"},
    }

    (out_dir / suite_sln).write_text(json.dumps(suite_sln_data, indent=2) + "\n", encoding="utf-8")
    (out_dir / cifar_sln).write_text(json.dumps(cifar_sln_data, indent=2) + "\n", encoding="utf-8")
    (out_dir / mnist_prj).write_text(json.dumps(mnist_prj_data, indent=2) + "\n", encoding="utf-8")
    (out_dir / cifar_prj).write_text(json.dumps(cifar_prj_data, indent=2) + "\n", encoding="utf-8")
    (out_dir / mnist_graph).write_text(graph_classify_images_mnist().strip() + "\n", encoding="utf-8")
    (out_dir / cifar_graph).write_text(graph_classify_images_cifar().strip() + "\n", encoding="utf-8")

    solutions = [f"nn/{name}/{suite_sln}", f"nn/{name}/{cifar_sln}"]
    projects = [f"nn/{name}/{mnist_prj}", f"nn/{name}/{cifar_prj}"]
    graphs = [f"nn/{name}/{mnist_graph}", f"nn/{name}/{cifar_graph}"]
    return {
        "name": name,
        "source_example": "ClassifyImages",
        "solution": solutions[0],
        "project": projects[0],
        "graph": graphs[0],
        "solutions": solutions,
        "projects": projects,
        "graphs": graphs,
    }


SPECIAL_TEMPLATE_BUILDERS = {
    "ClassifyImagesSuite": write_classify_images_suite,
}


def write_manifest(base_dir: Path, generated: list[dict]) -> None:
    nn_dir = base_dir / "nn"
    nn_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": 1,
        "generator": "convert_examples_to_nn.py",
        "available_templates": sorted(list(EXAMPLE_GRAPHS.keys()) + list(SPECIAL_TEMPLATE_BUILDERS.keys())),
        "generated": generated,
    }
    (nn_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=".", help="Path to ConvNetCpp repository root")
    parser.add_argument("--only", nargs="*", default=[], help="Optional subset of example names")
    parser.add_argument("--new-template", default="", help="Generate one template project by source example name")
    parser.add_argument("--dest-name", default="", help="Optional destination project name for --new-template")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    generated: list[dict] = []

    known_templates = sorted(list(EXAMPLE_GRAPHS.keys()) + list(SPECIAL_TEMPLATE_BUILDERS.keys()))

    if args.new_template:
        src_name = normalize_name(args.new_template)
        if src_name not in known_templates:
            raise SystemExit(f"Unknown template requested: {src_name}")
        dst_name = normalize_name(args.dest_name) if args.dest_name else src_name
        if not dst_name:
            raise SystemExit("Destination template name is empty")
        if src_name in EXAMPLE_GRAPHS:
            generated.append(write_example(repo, dst_name, EXAMPLE_GRAPHS[src_name](), src_name))
        else:
            generated.append(SPECIAL_TEMPLATE_BUILDERS[src_name](repo, dst_name))
        print(f"generated nn/{dst_name}/* from {src_name}")
        write_manifest(repo, generated)
        return 0

    requested = set(args.only) if args.only else set(known_templates)
    unknown = sorted(requested - set(known_templates))
    if unknown:
        raise SystemExit(f"Unknown examples requested: {', '.join(unknown)}")

    for name, fn in EXAMPLE_GRAPHS.items():
        if name not in requested:
            continue
        generated.append(write_example(repo, name, fn(), name))
        print(f"generated nn/{name}/*")
    for name, fn in SPECIAL_TEMPLATE_BUILDERS.items():
        if name not in requested:
            continue
        generated.append(fn(repo, name))
        print(f"generated nn/{name}/*")
    write_manifest(repo, generated)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
