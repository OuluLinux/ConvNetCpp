# ConvNetCpp

ConvNetCpp is a C++ implementation of convolutional neural networks with integration of llama.cpp for AI model processing capabilities.

## Prerequisites

- C++17 compatible compiler (clang++ or g++)
- Git
- CMake (required for building dependencies)
- Make or Ninja build system

## Getting Started

### Clone the Repository with Submodules

Since this project uses git submodules, you need to either clone with submodules:

```bash
git clone --recurse-submodules https://github.com/username/ConvNetCpp.git
```

Or if you've already cloned the repository, initialize and update submodules separately:

```bash
git submodule update --init --recursive
```

### Build the Project

The project requires building its dependencies. Follow the build instructions in the respective directories:

1. Build llama.cpp submodule:
```bash
cd src/llama.cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

The above commands will:
- Create a build directory inside llama.cpp
- Configure the build using CMake with Release flags for optimized performance
- Build all necessary binaries and libraries with parallel compilation using all CPU cores
- Place the resulting executables and libraries in `src/llama.cpp/build/bin`

After building, the llama.cpp directory will contain:
- Essential libraries (libggml.so, libllama.so, etc.)
- Executables (llama-cli, llama-server, llama-bench, etc.)
- Test utilities and example programs

2. Build the main ConvNetCpp project:
Instructions for building the main project are available in the build documentation.

## Configuration Notes

The project uses a CLANG.bm configuration file that references the built llama.cpp binaries and libraries. Ensure that the path in CLANG.bm matches where you built the project. By default, it expects the binaries in:
- Path: `/home/sblo/Dev/ConvNetCpp/src/llama.cpp/build/bin`
- Include: `/home/sblo/Dev/ConvNetCpp/src/llama.cpp/ggml/include;/home/sblo/Dev/ConvNetCpp/src/llama.cpp/include`
- Library: `/home/sblo/Dev/ConvNetCpp/src/llama.cpp/build/bin`

## Project Structure

- `src/ConvNet`: Main ConvNet implementation
- `src/ConvNetCtrl`: ConvNet control components
- `src/llama.cpp`: AI model processing submodule
- `examples/`: Example implementations
- `docs/`: Documentation
- `tutorial/`: Tutorial materials
- `upptst/`: Unit/performance tests

## Git Submodules

This project uses the following git submodules:

- `src/llama.cpp`: A C/C++ library for AI models, specifically optimized for LLM inference

### Working with Submodules

After cloning or when updating the repository, always ensure submodules are up to date:

```bash
# Update all submodules to their latest commits
git submodule update --remote --merge

# If you want to update submodules to the latest commits and also pull latest changes
git pull --recurse-submodules

# To initialize and update submodules (if they are not initialized)
git submodule update --init --recursive
```

### Contributing

1. Fork the repository
2. Create your feature branch
3. Ensure submodules are properly updated if needed
4. Commit your changes
5. Push to the branch
6. Create a new Pull Request

## License

This project is licensed under the terms found in the LICENSE file.