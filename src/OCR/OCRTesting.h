#ifndef _OCR_OCRTesting_h_
#define _OCR_OCRTesting_h_

#include "OCRModel.h"
#include "OCRDatasetProject.h"

namespace Upp {

namespace OCR {

struct TestResults {
	int total_samples;
	int correct_predictions;
	int incorrect_predictions;
	double accuracy;
	Vector<Vector<int>> confusion_matrix;
	double avg_inference_time;
	
	TestResults() : total_samples(0), correct_predictions(0), incorrect_predictions(0), accuracy(0.0), avg_inference_time(0.0) {}
};

class BatchTester {
public:
	BatchTester();

	void SetModel(OCRModel* m) { model = m; }
	void SetDataset(OCRDatasetProject* d) { dataset = d; }

	TestResults Run();

private:
	OCRModel* model;
	OCRDatasetProject* dataset;
};

// Integration testing suite
class OCRIntegrationTests {
public:
	static bool TestEndToEndPipeline();
	static bool TestSplitterAccuracy();
	static bool TestClassifierAccuracy();
	static bool TestModelSaveLoad();
	
	static void RunAllTests();
};

// Synthetic text generation for testing
struct CharInImage : Moveable<CharInImage> {
	Rect bounds;
	char character;
};

struct SyntheticText : Moveable<SyntheticText> {
	Image image;
	Vector<CharInImage> characters;
	Vector<int> boundaries;  // X-coordinates
};

SyntheticText CreateTextCharByChar(const String& text, Font font = StdFont(20));
Image CreateTestText(const String& text, Font font = StdFont(20));

} // namespace OCR

} // namespace Upp

#endif
