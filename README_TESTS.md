# ConvNetCpp Unit Tests

This directory contains headless unit tests for ConvNetCpp examples following the U++ upptst pattern.

## Directory Structure

Located in `upptst/` directory, each test follows the U++ convention:
- `upptst/CharGenTest/` - Tests for the CharGen example functionality
- `upptst/Regression1DTest/` - Tests for the Regression1D example functionality
- `upptst/Classify2DTest/` - Tests for the Classify2D example functionality
- `upptst/SimpleGANTest/` - Tests for the SimpleGAN example functionality
- `upptst/GANTest/` - Tests for GAN-related functionality
- `upptst/RegressionPainterTest/` - Tests for regression painter functionality
- `upptst/ClassifyImagesTest/` - Tests for image classification functionality
- `upptst/GridWorldTest/` - Tests for grid world reinforcement learning
- `upptst/HeteroscedasticUncertaintyTest/` - Tests for uncertainty estimation
- `upptst/MartingaleTest/` - Tests for financial prediction
- `upptst/NetworkOptimizationTest/` - Tests for network optimization algorithms
- `upptst/PuckWorldTest/` - Tests for continuous control environments
- `upptst/ReinforcedLearningTest/` - Tests for reinforcement learning concepts
- `upptst/TemporalDifferenceTest/` - Tests for temporal difference learning
- `upptst/TrainerBenchmarkTest/` - Tests for different training algorithms
- `upptst/WaterWorldTest/` - Tests for multi-agent reinforcement learning

Each test directory contains:
- `TestName.cpp` - The test implementation
- `TestName.upp` - The U++ project file

## Running Tests

### Build tests:

1. Build all tests:
   ```bash
   ./build-tests.sh
   ```

2. Build a single test:
   ```bash
   ./build-tests.sh Classify2DTest
   ```

3. Build multiple specific tests:
   ```bash
   ./build-tests.sh Classify2DTest Regression1DTest
   ```

4. Build with clean option (force rebuild):
   ```bash
   ./build-tests.sh --clean
   ```
   
   Or for a single test:
   ```bash
   ./build-tests.sh --clean Classify2DTest
   ```

### Run tests:

1. Run all tests:
   ```bash
   ./run-tests.sh
   ```

2. Run a single test:
   ```bash
   ./run-test.sh Classify2DTest
   ```

### Available Test Names:
`CharGenTest`, `Regression1DTest`, `Classify2DTest`, `SimpleGANTest`, `GANTest`, `RegressionPainterTest`, `ClassifyImagesTest`, `GridWorldTest`, `HeteroscedasticUncertaintyTest`, `MartingaleTest`, `NetworkOptimizationTest`, `PuckWorldTest`, `ReinforcedLearningTest`, `TemporalDifferenceTest`, `TrainerBenchmarkTest`, `WaterWorldTest`

## Framework

Tests use the U++ Core library and ConvNetCpp functionality, with assertions to verify correct behavior. Each test is a standalone console application that follows the `CONSOLE_APP_MAIN` pattern from U++.

All tests are headless (no GUI components) and focus on the core neural network functionality demonstrated in the examples.