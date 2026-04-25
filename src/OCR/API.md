# OCR API Reference

## OCREngine

Main class for OCR operations.

### Methods

#### Initialize
```cpp
bool Initialize(const OCRConfig& config)
```
Initializes the OCR engine with the given configuration.
- **Parameters:** `config` specifies model paths and thresholds.
- **Returns:** `true` on success.

#### RecognizePage
```cpp
OCRResult RecognizePage(const Image& page_img)
```
Recognizes all text in a page image.
- **Returns:** `OCRResult` containing lines, full text, and confidence.

#### RecognizeLine
```cpp
OCRLine RecognizeLine(const Image& line_img)
```
Recognizes text in a single line image.

## OCRModel

Wrapper for ConvNet models with OCR metadata.

### Methods

#### Load / Save
```cpp
bool Load(const String& path)
bool Save(const String& path)
```
Loads or saves the model including architecture, weights, and metadata.

#### PredictClass
```cpp
int PredictClass(const Image& img)
```
Predicts the character class for a normalized 28x28 image.

## OCRDatasetProject

Extends `OCRDataset` with JSON persistence.

### Methods

#### Load / Save
```cpp
bool Load(const String& path)
bool Save(const String& path)
```
Handles `.ocrdata` files.
