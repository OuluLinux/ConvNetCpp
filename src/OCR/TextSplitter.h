#ifndef _OCR_TextSplitter_h_
#define _OCR_TextSplitter_h_

#include <ConvNet/ConvNet.h>
#include "OCRTypes.h"
#include "OCRModel.h"
#include "ImageUtils.h"
#include "OCRDatasetProject.h"

namespace Upp {

namespace OCR {

struct SplitterTrainingData {
	Vector<Image> windows;      // 3x28 image strips
	Vector<bool> is_boundary;   // Ground truth labels
};

class TextSplitter {
public:
	TextSplitter();

	bool Load(const String& model_path);
	Vector<int> PredictBoundaries(const Image& line_img);

	// Training data generation
	static SplitterTrainingData GenerateTrainingData(const OCRDatasetProject& proj);
	static Image ExtractWindow(const Image& src, int center_x, int width = 3, int height = 28);

	void SetThreshold(double t) { threshold = t; }
	double GetThreshold() const { return threshold; }

private:
	OCRModel model;
	double threshold;
};

} // namespace OCR

} // namespace Upp

#endif
