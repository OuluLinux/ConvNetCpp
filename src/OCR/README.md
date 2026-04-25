# OCR System Documentation

## Overview

This OCR (Optical Character Recognition) system provides a complete solution for recognizing text in images, from dataset creation to model training and deployment. It is built using the U++ framework and the ConvNet library.

## Components

1. **OCR Core Library** (`src/OCR`) - Embeddable OCR engine with no GUI dependencies.
2. **Dataset Editor** (`examples/OCRDatasetEditor`) - GUI tool for creating and annotating labeled datasets.
3. **Training Tool** (`examples/OCRTraining`) - GUI tool for training text splitting and character classification models.
4. **OCR Workbench** (`examples/OCRWorkbench`) - Unified application integrating the editor and trainer.

## Quick Start

### Using the OCR Engine

```cpp
#include <OCR/OCR.h>
using namespace Upp;
using namespace OCR;

void RunOCR() {
    OCRConfig config;
    config.splitter_model_path = "models/splitter.ocrmodel";
    config.classifier_model_path = "models/classifier.ocrmodel";

    OCREngine engine;
    if (engine.Initialize(config)) {
        Image img = StreamRaster::LoadFileAsImage("text.png");
        OCRResult result = engine.RecognizePage(img);
        LOG(result.full_text);
    }
}
```

### Training a Model

1. Create and annotate a dataset using **OCR Dataset Editor**.
2. Save the dataset as `.ocrdata`.
3. Open **OCR Training Tool**, load your dataset.
4. Train the **Splitter** model and export it.
5. Train the **Classifier** model and export it.
6. Use the exported `.ocrmodel` files with `OCREngine`.

## Building

Use the provided build script:
```bash
python3 scripts/build.py OCRWorkbench
```

## Testing

Run the unified test suite:
```bash
python3 scripts/build.py OCRTests && ./bin/OCRTests
```
