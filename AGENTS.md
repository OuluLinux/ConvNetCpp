# ConvNetCpp Agent Guide

## Repo Role

ConvNetCpp is a high-performance scientific machine learning engine, GUI, and CLI repository for neural networks, transformers, and vision recognition tasks with custom CUDA acceleration.

Build with `UppBuilder`:
```bash
uppbuilder src/ConvNetTest --bazaarcore <path-to-BazaarCore> -v
./bin/ConvNetTest
```

## Scientific Terminology Mandate
To maintain architectural purity and domain-agnostic scalability, always use generic, scientific terminology in code, commit messages, and documentation:
- **Element**: Primary recognized object unit.
- **Category**: Classification grouping for elements.
- **Level**: Scalar or ordinal value associated with an element.
- **GameEngine**: The host pipeline or simulation environment.
- **ProviderA**: The specific data source or upstream provider.
- **DS Repo**: Reference to the dataset repository.

## Architecture
- `src/ConvNet`: Primary neural network and deep learning engine (dense, conv, pool, transformers, ViT, Swin, GPT, DQN, custom CUDA kernels).
- `src/ConvNetTest`: Headless automated unit test verifying XOR forward/backward training and prediction.
- `src/ConvNetCtrl`: Visualization widgets, heatmap displays, and training loss curves.
- `src/ConvNetCLI`: Headless console runner with script execution support.
- `src/ConvNetPlugin`: ScriptIDE plugin exposing neural network training and inference.
- `src/AnnLayCore`: Core logic for annotation layers and generic recognition pipelines.
- `src/AnnotationEditor`: Main GUI tool for dataset management and model training.
- `src/OCR`: Optical Character Recognition engine and utilities.

## CI Matrix
Continuous integration tests 4 platforms on every push:
- Ubuntu Linux (GCC)
- Ubuntu Linux (Clang)
- macOS (Apple Clang)
- Windows (MSVC 2022 x64)
- Ubuntu Linux CUDA Backend & Evaluator Suite (`cuda-evaluator-ci`)
