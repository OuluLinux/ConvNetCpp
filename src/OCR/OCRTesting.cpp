#include "OCRTesting.h"
#include <Painter/Painter.h>

namespace Upp {

namespace OCR {

BatchTester::BatchTester() {
	model = nullptr;
	dataset = nullptr;
}

TestResults BatchTester::Run() {
	TestResults res;
	if (!model || !dataset) return res;

	int n = model->GetSession().Data().GetClassCount();
	if (n == 0) n = 62; // Fallback
	res.confusion_matrix.SetCount(n);
	for(int i = 0; i < n; i++) res.confusion_matrix[i].SetCount(n, 0);

	int64 total_time = 0;
	int time_count = 0;

	for (int i = 0; i < dataset->GetCount(); i++) {
		const Image& img = dataset->GetImage(i);
		const Vector<OCRSegment>& segs = dataset->GetSegments(i);

		for (const auto& seg : segs) {
			for (const auto& ch : seg.chars) {
				Image char_img = CropImage(img, ch.bounds);
				
				int64 start = msecs();
				int pred = model->PredictClass(char_img);
				int64 end = msecs();
				
				total_time += (end - start);
				time_count++;
				
				int actual = ch.label;

				if (actual >= 0 && actual < n && pred >= 0 && pred < n) {
					res.confusion_matrix[actual][pred]++;
					if (pred == actual) res.correct_predictions++;
					else res.incorrect_predictions++;
					res.total_samples++;
				}
			}
		}
	}

	if (res.total_samples > 0)
		res.accuracy = (double)res.correct_predictions / res.total_samples;
	
	if (time_count > 0)
		res.avg_inference_time = (double)total_time / time_count;

	return res;
}

bool OCRIntegrationTests::TestModelSaveLoad() {
	OCRModel m1;
	m1.SetType(OCR_MODEL_CLASSIFIER);
	m1.SetArchitecture("[]");
	
	String tmp = ConfigFile("tmp.ocrmodel");
	if (!m1.Save(tmp)) return false;
	
	OCRModel m2;
	if (!m2.Load(tmp)) {
		FileDelete(tmp);
		return false;
	}
	
	FileDelete(tmp);
	return m2.GetType() == OCR_MODEL_CLASSIFIER;
}

bool OCRIntegrationTests::TestEndToEndPipeline() {
	return true; // Stub
}

bool OCRIntegrationTests::TestSplitterAccuracy() {
	return true; // Stub
}

bool OCRIntegrationTests::TestClassifierAccuracy() {
	return true; // Stub
}

void OCRIntegrationTests::RunAllTests() {
	LOG("Running OCR Integration Tests...");
	LOG("1. Model Save/Load: " << (TestModelSaveLoad() ? "PASSED" : "FAILED"));
}

SyntheticText CreateTextCharByChar(const String& text, Font font) {
	SyntheticText result;

	int x = 10;  // Start position
	int y = 10;

	for (int i = 0; i < text.GetCount(); i++) {
		char c = text[i];
		String ch(c, 1);
		Size ch_sz = GetTextSize(ch, font);

		CharInImage& ci = result.characters.Add();
		ci.character = c;
		ci.bounds = Rect(x, y, x + ch_sz.cx, y + ch_sz.cy);
		result.boundaries.Add(x);

		x += ch_sz.cx + 1;  // 1-pixel spacing
	}
	result.boundaries.Add(x);  // Final boundary

	// Create image
	Size img_sz(x + 10, y + font.GetHeight() + 10);
	ImageBuffer ib(img_sz);
	Fill(~ib, White(), ib.GetLength());

	BufferPainter bp(ib);
	for (const CharInImage& ci : result.characters) {
		bp.DrawText(ci.bounds.left, ci.bounds.top, String(ci.character, 1), font, Black());
	}

	result.image = ib;
	return result;
}

Image CreateTestText(const String& text, Font font) {
	Size text_sz = GetTextSize(text, font);
	int padding = 10;
	Size img_sz(text_sz.cx + padding * 2, text_sz.cy + padding * 2);

	ImageBuffer ib(img_sz);
	Fill(~ib, White(), ib.GetLength());

	BufferPainter bp(ib);
	bp.DrawText(padding, padding, text, font, Black());

	return ib;
}

} // namespace OCR

} // namespace Upp
