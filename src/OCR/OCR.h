#ifndef _OCR_OCR_h_
#define _OCR_OCR_h_

#include <ConvNet/ConvNet.h>
#include <Draw/Draw.h>
#include "OCRTypes.h"
#include "OCRModel.h"
#include "TextSplitter.h"
#include "OCRPipeline.h"
#include "ImageUtils.h"
#include "Segmentation.h"
#include "Normalization.h"
#include "Preprocessing.h"
#include "Tesseract.h"

namespace Upp {

namespace OCR {

enum OCRBackend {
	OCR_AUTO = 0,
	OCR_CONVNET,
	OCR_TESSERACT
};

// Configuration
struct OCRConfig {
	OCRBackend backend = OCR_TESSERACT; // Default to Tesseract as requested
	OCRPreprocessMode mode = OCR_PREPROCESS_GRAYSCALE;
	Size input_size = Size(28, 28);
	String splitter_model_path;
	String classifier_model_path;
	double splitter_threshold = 0.5;
	double classifier_threshold = 0.5;
	bool use_splitter = true;
	bool normalize_images = true;
	int tesseract_psm = 7;
	String ocr_whitelist;
	String ocr_blacklist;
};

// Line recognition result
struct OCRLine : Moveable<OCRLine> {
	Rect bounds;
	String text;
	Vector<OCRChar> characters;
	double avg_confidence;
	
	OCRLine() : avg_confidence(0.0) {}
};

// Full page recognition result
struct OCRResult : Moveable<OCRResult> {
	Vector<OCRLine> lines;
	String full_text;
	double avg_confidence;
	int total_characters;
	double processing_time_ms;
	
	OCRResult() : avg_confidence(0.0), total_characters(0), processing_time_ms(0.0) {}
};

// Main OCR Engine class
class OCREngine {
public:
	OCREngine();
	~OCREngine();

	// Configuration
	bool Initialize(const OCRConfig& config);
	bool LoadSplitter(const String& model_path);
	bool LoadClassifier(const String& model_path);

	OCRPipeline& GetPipeline() { return pipeline; }

	// Recognition
	OCRResult RecognizePage(const Image& page_img);
	OCRLine RecognizeLine(const Image& line_img);
	OCRChar RecognizeChar(const Image& char_img);

	// Batch processing
	Vector<OCRResult> RecognizePages(const Vector<Image>& pages,
	                                 Callback1<int> progress = Callback1<int>());

	// Utilities
	Vector<Rect> SegmentLines(const Image& page_img);
	Vector<Rect> SegmentChars(const Image& line_img);

	// Status
	bool IsInitialized() const { return initialized; }
	String GetLastError() const { return last_error; }
	const OCRConfig& GetConfig() const { return config; }

private:
	OCRConfig config;
	OCRPipeline pipeline;
	bool initialized;
	String last_error;

	void SetError(const String& error) { last_error = error; }
};

// Version info
String GetOCRVersion();
String GetOCRBuildDate();

} // namespace OCR

} // namespace Upp

#endif
