# ConvNetCpp Layer Architecture

## Overview

The ConvNetCpp library implements neural network layers using a unified architecture based on a single `LayerBase` class that handles multiple layer types through a switch-based implementation.

## Current Architecture

### LayerBase Class

The core of the implementation is the `LayerBase` class which:

- Contains all possible member variables for all layer types (convolutional, fully-connected, pooling, etc.)
- Uses a `layer_type` enum to identify which layer implementation to execute
- Implements a single `Forward()` method that switches based on `layer_type`
- Implements a single `Backward()` method that switches based on `layer_type`
- Stores all layer-specific parameters and hyperparameters as member variables

### Layer Types

The system supports the following layer types:

- `NULL_LAYER` (0)
- `FULLYCONN_LAYER` (1)
- `LRN_LAYER` (2) - Local Response Normalization
- `DROPOUT_LAYER` (3)
- `INPUT_LAYER` (4)
- `SOFTMAX_LAYER` (5)
- `REGRESSION_LAYER` (6)
- `CONV_LAYER` (7) - Convolutional
- `DECONV_LAYER` (8) - Deconvolutional
- `POOL_LAYER` (9)
- `UNPOOL_LAYER` (10)
- `RELU_LAYER` (11)
- `SIGMOID_LAYER` (12)
- `TANH_LAYER` (13)
- `MAXOUT_LAYER` (14)
- `SVM_LAYER` (15)
- `HETEROSCEDASTICREGRESSION_LAYER` (16)

### Implementation Pattern

Each layer type has corresponding methods in the `LayerBase` class:
- `Forward*()` methods for forward pass
- `Backward*()` methods for backward pass  
- `Init*()` methods for initialization
- `ToString*()` methods for string representation

## Relationships Between Layer Types

### Inheritance Hierarchy

```
LayerBase (monolithic class)
├── Specific implementations via switch statements in methods
├── IDotProductLayer interface (for layers with weights/biases)
└── IClassificationLayer interface (for classification layers)
```

### Data Flow

1. Each layer takes a `Volume` as input (3D: width, height, depth)
2. Each layer produces a `Volume` as output
3. Layers connect sequentially in a `Net` object
4. Forward pass: input → layer0 → layer1 → ... → layerN → output
5. Backward pass: error signal flows backward through same chain

## Key Components

### Volume Class
- 3D tensor structure (width, height, depth)
- Stores both values and gradients
- Used for all data passing between layers

### Net Class
- Contains a vector of `LayerBase` objects
- Manages the sequence of layers
- Handles forward and backward propagation

### Session Class
- Higher-level wrapper for training
- Contains a `Net` and a `TrainerBase`
- Manages training loops and data

## Current Limitations

1. **Monolithic Design**: All layer types share the same class with unused member variables
2. **Switch-Based Logic**: Complex switch statements for every operation
3. **No Composition**: Difficult to create complex layer combinations
4. **Limited Extensibility**: Adding new layer types requires modifying the base class
5. **No Advanced Features**: No support for skip connections, multi-input layers, etc.

## File Structure

- `LayerBase.h` / `LayerBase.cpp` - Core implementation
- `Layers.h` - Class declarations for specific layer types
- Individual layer implementations like `ConvLayer.cpp`, `FullyConnLayer.cpp`, etc.