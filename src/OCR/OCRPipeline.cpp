#include "OCRPipeline.h"
#include "Segmentation.h"

namespace Upp {

namespace OCR {

OCRPipeline::OCRPipeline() {
}

namespace {

void ClassifyCharWithConfidence(OCRModel& classifier, const Image& char_img, OCRChar& ch)
{
	Vector<double> probs = classifier.Predict(char_img);
	if(probs.IsEmpty()) {
		ch.label = classifier.PredictClass(char_img);
		ch.confidence = 0.0;
		if(ch.label >= 0)
			ch.text = classifier.GetClassLabel(ch.label);
		return;
	}

	int best = 0;
	for(int i = 1; i < probs.GetCount(); i++)
		if(probs[i] > probs[best])
			best = i;

	ch.label = best;
	ch.confidence = probs[best];
	ch.text = classifier.GetClassLabel(best);
}

}

bool OCRPipeline::LoadSplitter(const String& model_path) {
	return splitter.Load(model_path);
}

bool OCRPipeline::LoadClassifier(const String& model_path) {
	return classifier.Load(model_path);
}

String OCRPipeline::RecognizeText(const Image& text_line_img) {
	Vector<OCRChar> chars = RecognizeChars(text_line_img);
	String result;
	for (const OCRChar& ch : chars) {
		if (ch.label >= 0) {
			char c = ClassToChar(ch.label);
			if (c != '?') result.Cat(c);
		}
	}
	return result;
}

Vector<OCRChar> OCRPipeline::RecognizeChars(const Image& text_line_img) {
	if (text_line_img.IsEmpty()) return Vector<OCRChar>();

	// 1. Split text into character boundaries
	Vector<int> boundaries = splitter.PredictBoundaries(text_line_img);

	if (boundaries.IsEmpty()) {
		// Fallback to simple segmentation if model fails or isn't loaded
		Vector<Rect> rects = SegmentCharacters(text_line_img);
		Vector<OCRChar> result;
		for (const Rect& r : rects) {
			OCRChar& ch = result.Add();
			ch.bounds = r;
			
			Image char_img = CropImage(text_line_img, r);
			ClassifyCharWithConfidence(classifier, char_img, ch);
		}
		return result;
	}

	// 2. Convert boundaries to character rects and classify
	Vector<OCRChar> result;
	for (int i = 0; i < boundaries.GetCount() - 1; i++) {
		Rect r(
			boundaries[i],
			0,
			boundaries[i + 1],
			text_line_img.GetHeight()
		);
		
		// Refine bounds to actual ink
		Rect tight = FindTightBounds(text_line_img, r);
		if (tight.IsEmpty()) continue;

		OCRChar& ch = result.Add();
		ch.bounds = tight;
		
		Image char_img = CropImage(text_line_img, tight);
		ClassifyCharWithConfidence(classifier, char_img, ch);
	}

	return result;
}

OCRChar OCRPipeline::PredictChar(const Image& char_img) {
	OCRChar res;
	res.bounds = char_img.GetSize();

	Vector<double> probs = classifier.Predict(char_img);
	
	if (!probs.IsEmpty()) {
		int best = 0;
		for(int i = 1; i < probs.GetCount(); i++)
			if(probs[i] > probs[best]) best = i;
		res.label = best;
		res.confidence = probs[best];
		res.text = classifier.GetClassLabel(best);
	}
	res.img = char_img;
	return res;
}

} // namespace OCR

} // namespace Upp
