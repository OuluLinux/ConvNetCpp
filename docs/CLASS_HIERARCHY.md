# ConvNetCpp Class Hierarchy and Relationships

## Major Class Categories

### Core Classes
- **Net** - Main neural network container
- **LayerBase** - Base class for all layer types (monolithic implementation)
- **Volume** - 3D tensor storage for data and gradients
- **Session** - High-level training interface

### Layer Classes
All layers are implemented within the single `LayerBase` class via switch statements, but conceptually they include:
- InputLayer
- ConvLayer
- FullyConnLayer
- PoolLayer
- DeconvLayer
- UnpoolLayer
- Activation layers (Relu, Sigmoid, Tanh)
- Normalization layers (LrnLayer)
- Regularization layers (DropOutLayer)
- Output layers (Softmax, Regression, SVM)

### Training Classes
- **TrainerBase** - Abstract base for training algorithms
- **SgdTrainer**, **AdagradTrainer**, **AdamTrainer**, etc. - Specific optimizers
- **Session** - Manages training loop and data

## Key Class Relationships

### Net and LayerBase Relationship
```
Net contains Vector<LayerBase> layers
- Layers are connected sequentially
- Forward pass: Net.Forward() → LayerBase.Forward() for each layer
- Backward pass: Net.Backward() → LayerBase.Backward() in reverse order
```

### LayerBase Internal Structure
The LayerBase class contains ALL possible member variables for ALL layer types:

- Common: output_activation, input_activation, dimensions
- Conv/FC specific: filters, biases, neuron_count
- Dropout specific: dropped, drop_prob  
- Pool specific: switchx, switchy
- Softmax specific: es (exponential sums)
- And many more...

### Volume and Layer Relationship
- Each layer has one input_activation and one output_activation
- Layers access input data via input_activation pointer
- Gradients flow backward through connected Volume objects
- Volume manages both data and gradient information

### Session Integration
- Session contains Net and TrainerBase
- Session manages data flow from SessionData to Net
- Session orchestrates training process
- Session provides high-level API for users

## Inheritance Hierarchy

### Current Implementation
```
LayerBase (monolithic class with all possible variables)
├── Layer-specific methods implemented via switch statements
├── IDotProductLayer interface (mixed in via member variables)
└── IClassificationLayer interface (mixed in via member variables)
```

### Proposed New Implementation
```
LayerBase (abstract interface)
├── IWeightedLayer
│   ├── ConvLayer
│   ├── FullyConnLayer
│   └── DeconvLayer
├── IActivationLayer
│   ├── ReluLayer
│   ├── SigmoidLayer
│   └── TanhLayer
├── INormalizationLayer
│   └── LayerNormLayer (new for transformers)
├── IAttentionLayer (new for transformers)
│   ├── ScaledDotProductAttentionLayer
│   └── MultiHeadAttentionLayer
└── Others...
```

## Data Flow and Dependencies

### Forward Pass Data Flow
Input Volume → Layer0 → Layer1 → ... → LayerN → Output Volume

### Backward Pass Data Flow  
Output Gradients → LayerN → ... → Layer1 → Layer0 → Input Gradients

### Parameter Management
- LayerBase::GetParametersAndGradients() returns parameters for training
- Trainer uses these parameters to update weights
- Each layer may have filters and biases that get trained

## Key Member Variables and Their Relationships

### LayerBase Member Variables
- `Volume output_activation` - Output of the layer
- `int output_depth, output_width, output_height` - Output dimensions
- `int input_depth, input_width, input_height` - Input dimensions  
- `int layer_type` - Enum identifying which layer implementation to use
- `Volume biases` - Bias parameters (not used by all layers)
- `Vector<Volume> filters` - Filter/weight parameters (not used by all layers)
- `Volume* input_activation` - Pointer to input from previous layer
- `int neuron_count` - Number of neurons (for FC layers)
- `int filter_count` - Number of filters (for conv layers)
- `int width, height, stride, pad` - Convolution parameters
- `double drop_prob` - Dropout probability
- `Vector<bool> dropped` - Dropout mask
- And many more layer-specific variables

### Interdependencies
- Dimensions must match between consecutive layers
- Input/output activations connect layers in sequence
- Parameters are updated during backpropagation
- Some layers (pooling, activations) don't have learnable parameters

## Design Issues Identified

1. **Memory Inefficiency**: All layers have all possible member variables
2. **Code Complexity**: Large switch statements in Forward/Backward methods
3. **Maintenance Difficulty**: Adding new layer types requires modifying core base class
4. **Extensibility Problems**: Hard to implement complex layers like attention
5. **Modularity Issues**: Layer types are not properly separated

## Proposed Improvements

The new architecture should:
1. Use proper inheritance with abstract base classes
2. Separate layer-specific data and methods
3. Enable composition for complex layer types
4. Support multi-input layers for attention mechanisms
5. Enable skip connections and other modern architectures