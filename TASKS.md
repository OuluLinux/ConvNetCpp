# TASKS.md - ConvNetCpp Project

## TODO

### 1. Fix GAN example that has never worked
- **Task ID**: 1
- **Priority**: High
- **Description**: The GAN example in examples/GAN has never worked. Need to debug and fix the implementation so it can properly train and generate samples.
- **Status**: Completed

### 2. Add Vision Transformer (ViT) implementation
- **Task ID**: 2
- **Priority**: Medium
- **Description**: Create a Vision Transformer example for image classification tasks using the existing ClassifyImages framework.
- **Status**: Not Started

### 3. Add Diffusion Model implementation
- **Task ID**: 3
- **Priority**: Medium
- **Description**: Implement a Diffusion model for generative tasks, extending the existing GAN example.
- **Status**: Not Started

### 4. Add Swin Transformer implementation
- **Task ID**: 4
- **Priority**: Medium
- **Description**: Create a Swin Transformer example for vision tasks with hierarchical representations.
- **Status**: Not Started

### 5. Add EfficientNet implementation
- **Task ID**: 5
- **Priority**: Medium
- **Description**: Implement an EfficientNet example with compound scaling.
- **Status**: Not Started

### 6. Add BERT/RoBERTa implementation
- **Task ID**: 6
- **Priority**: Medium
- **Description**: Create a BERTTester example similar to the existing GptTester.
- **Status**: Not Started

### 7. Add T5 implementation
- **Task ID**: 7
- **Priority**: Medium
- **Description**: Develop a T5Tester example for text-to-text generation tasks.
- **Status**: Not Started

### 8. Add Vision-Language models (CLIP)
- **Task ID**: 8
- **Priority**: Low
- **Description**: Create a CLIPTester example that combines vision and language transformers.
- **Status**: Not Started

### 9. Add DCGAN implementation
- **Task ID**: 9
- **Priority**: Medium
- **Description**: Implement Deep Convolutional GAN with transposed convolutions, batch normalization, and specific activation functions. Should fork from existing GAN example and create a new DCGAN directory with its own GUI app main function, using a common neural network GUI framework.
- **Status**: Completed

### 10. Add WGAN implementation
- **Task ID**: 10
- **Priority**: Medium
- **Description**: Implement Wasserstein GAN using Earth Mover's distance for more stable training. Should fork from existing GAN example and create a new WGAN directory with its own GUI app main function, using a common neural network GUI framework.
- **Status**: Completed

### 11. Add Progressive Growing GAN implementation
- **Task ID**: 11
- **Priority**: High
- **Description**: Implement Progressive Growing GAN that starts at low resolution and gradually increases resolution during training. Should fork from existing GAN example and create a new ProgressiveGAN directory with its own GUI app main function, using a common neural network GUI framework.
- **Status**: Completed

### 12. Add StyleGAN implementation
- **Task ID**: 12
- **Priority**: High
- **Description**: Implement StyleGAN with adaptive instance normalization and style control at different scales. Should fork from existing GAN example and create a new StyleGAN directory with its own GUI app main function, using a common neural network GUI framework.
- **Status**: Completed

### 13. Add BigGAN implementation
- **Task ID**: 13
- **Priority**: Medium
- **Description**: Implement BigGAN with large batch sizes and self-attention mechanisms for high-quality image generation. Should fork from existing GAN example and create a new BigGAN directory with its own GUI app main function, using a common neural network GUI framework.
- **Status**: Not Started

### 14. Add CycleGAN implementation
- **Task ID**: 14
- **Priority**: High
- **Description**: Implement CycleGAN for image-to-image translation without requiring paired training data. Should fork from existing GAN example and create a new CycleGAN directory with its own GUI app main function, using a common neural network GUI framework.
- **Status**: Not Started

### 15. Add Conditional GAN implementation
- **Task ID**: 15
- **Priority**: Medium
- **Description**: Implement Conditional GAN that allows conditioning generation on additional information like class labels. Should fork from existing GAN example and create a new ConditionalGAN directory with its own GUI app main function, using a common neural network GUI framework.
- **Status**: Not Started

### 16. Add Self-Attention GAN implementation
- **Task ID**: 16
- **Priority**: High
- **Description**: Implement Self-Attention GAN (SAGAN) using self-attention mechanisms to capture long-range dependencies in images. Should fork from existing GAN example and create a new SAGAN directory with its own GUI app main function, using a common neural network GUI framework.
- **Status**: Completed

### 17. Fix character-level tokenization in CharGenTest
- **Task ID**: 17
- **Priority**: High
- **Description**: The CharGenTest was failing due to improper character-level tokenization in the SubwordTokenizer. Fix the tokenizer implementation so that detokenize properly reconstructs the original text.
- **Status**: Completed

## IN_PROGRESS

## DONE

### 9. Add DCGAN implementation
- **Task ID**: 9
- **Priority**: Medium
- **Description**: Implement Deep Convolutional GAN with transposed convolutions, batch normalization, and specific activation functions. Should fork from existing GAN example and create a new DCGAN directory with its own GUI app main function, using a common neural network GUI framework.
- **Status**: Completed

### 10. Add WGAN implementation
- **Task ID**: 10
- **Priority**: Medium
- **Description**: Implement Wasserstein GAN using Earth Mover's distance for more stable training. Should fork from existing GAN example and create a new WGAN directory with its own GUI app main function, using a common neural network GUI framework.
- **Status**: Completed

### 11. Add Progressive Growing GAN implementation
- **Task ID**: 11
- **Priority**: High
- **Description**: Implement Progressive Growing GAN that starts at low resolution and gradually increases resolution during training. Should fork from existing GAN example and create a new ProgressiveGAN directory with its own GUI app main function, using a common neural network GUI framework.
- **Status**: Completed

### 12. Add StyleGAN implementation
- **Task ID**: 12
- **Priority**: High
- **Description**: Implement StyleGAN with adaptive instance normalization and style control at different scales. Should fork from existing GAN example and create a new StyleGAN directory with its own GUI app main function, using a common neural network GUI framework.
- **Status**: Completed

### 16. Add Self-Attention GAN implementation
- **Task ID**: 16
- **Priority**: High
- **Description**: Implement Self-Attention GAN (SAGAN) using self-attention mechanisms to capture long-range dependencies in images. Should fork from existing GAN example and create a new SAGAN directory with its own GUI app main function, using a common neural network GUI framework.
- **Status**: Completed
