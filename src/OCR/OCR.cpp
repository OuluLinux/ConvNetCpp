#include "OCR.h"

namespace Upp {

namespace OCR {

OCREngine::OCREngine() : initialized(false) {
}

OCREngine::~OCREngine() {
}

bool OCREngine::Initialize(const OCRConfig& cfg) {
	this->config = cfg;

	if(config.backend == OCR_CONVNET) {
		if (!config.splitter_model_path.IsEmpty() && FileExists(config.splitter_model_path)) {
			if (!LoadSplitter(config.splitter_model_path)) {
				return false;
			}
		}

		if (!config.classifier_model_path.IsEmpty() && FileExists(config.classifier_model_path)) {
			if (!LoadClassifier(config.classifier_model_path)) {
				return false;
			}
		}
	}

	initialized = true;
	return true;
}

bool OCREngine::LoadSplitter(const String& model_path) {
	if (pipeline.LoadSplitter(model_path)) {
		return true;
	}
	SetError("Failed to load splitter model: " + model_path);
	return false;
}

bool OCREngine::LoadClassifier(const String& model_path) {
	if (pipeline.LoadClassifier(model_path)) {
		return true;
	}
	SetError("Failed to load classifier model: " + model_path);
	return false;
}

OCRLine OCREngine::RecognizeLine(const Image& line_img) {
	OCRLine res;
	res.bounds = line_img.GetSize();
	
	Vector<OCRChar> chars = pipeline.RecognizeChars(line_img);
	res.characters = pick(chars);
	
	double sum_conf = 0;
	int conf_count = 0;
	for (const auto& ch : res.characters) {
		if (ch.label < 0)
			continue;
		res.text.Cat(ClassToChar(ch.label));
		sum_conf += ch.confidence;
		conf_count++;
	}
	
	if (conf_count > 0)
		res.avg_confidence = sum_conf / conf_count;
	
	return res;
}

OCRChar OCREngine::RecognizeChar(const Image& char_img) {
	return pipeline.PredictChar(char_img);
}

Vector<OCRResult> OCREngine::RecognizePages(const Vector<Image>& pages, Callback1<int> progress) {
	Vector<OCRResult> res;
	for (int i = 0; i < pages.GetCount(); i++) {
		res.Add(RecognizePage(pages[i]));
		if (progress) progress(i + 1);
	}
	return res;
}

OCRResult OCREngine::RecognizePage(const Image& page_img) {
	OCRResult res;
	int64 start = GetTickCount();
	
	if(config.backend == OCR_TESSERACT) {
		Image prep = page_img;
		if(config.normalize_images) {
			PreprocessingOptions opts;
			opts.mode = config.mode;
			// Respect the mode; only binarize if it's a thresholding mode or explicitly requested
			opts.binarize = (config.mode >= OCR_PREPROCESS_ADAPTIVE_THRESHOLD);
			prep = Preprocess(page_img, opts);
		}
		
		String text;
		double confidence = 0;
		if(RecognizeWithTesseract(prep, config.tesseract_psm, text, confidence, config.ocr_whitelist, config.ocr_blacklist)) {
			res.full_text = text;
			res.avg_confidence = confidence;
			// For simplicity in page mode with tesseract stdout, we might not have lines here
			// unless we use hocr or tsv output.
			if(!text.IsEmpty()) {
				OCRLine& line = res.lines.Add();
				line.text = text;
				line.avg_confidence = confidence;
				line.bounds = page_img.GetSize();
				res.total_characters = text.GetCount();
			}
		}
		res.processing_time_ms = (double)(GetTickCount() - start);
		return res;
	}

	Vector<Rect> lines = SegmentLines(page_img);
	double sum_conf = 0;
	
	for (const Rect& r : lines) {
		Image line_img = CropImage(page_img, r);
		OCRLine line = RecognizeLine(line_img);
		line.bounds = r;
		
		res.lines.Add(pick(line));
		res.full_text << res.lines.Top().text << "\n";
		sum_conf += res.lines.Top().avg_confidence;
		res.total_characters += res.lines.Top().characters.GetCount();
	}
	
	if (!res.lines.IsEmpty())
		res.avg_confidence = sum_conf / res.lines.GetCount();
	
	res.processing_time_ms = (double)(GetTickCount() - start);
	
	return res;
}

Vector<Rect> OCREngine::SegmentLines(const Image& page_img) {
	return OCR::SegmentLines(page_img);
}

Vector<Rect> OCREngine::SegmentChars(const Image& line_img) {
	return OCR::SegmentCharacters(line_img);
}

String GetOCRVersion() {
	return "1.0.0";
}

String GetOCRBuildDate() {
	return __DATE__ " " __TIME__;
}

} // namespace OCR

} // namespace Upp
