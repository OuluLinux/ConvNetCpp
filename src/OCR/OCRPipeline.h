#ifndef _OCR_OCRPipeline_h_
#define _OCR_OCRPipeline_h_

#include "TextSplitter.h"
#include "OCRModel.h"

namespace Upp {

namespace OCR {

class OCRPipeline {
public:
	OCRPipeline();

	bool LoadSplitter(const String& model_path);
	bool LoadClassifier(const String& model_path);

	String RecognizeText(const Image& text_line_img);
	Vector<OCRChar> RecognizeChars(const Image& text_line_img);
	OCRChar PredictChar(const Image& char_img);

	OCRModel& GetClassifier() { return classifier; }

private:
	TextSplitter splitter;
	OCRModel classifier;
};

} // namespace OCR

} // namespace Upp

#endif
