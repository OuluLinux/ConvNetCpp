#include "TextSplitter.h"

namespace Upp {

namespace OCR {

TextSplitter::TextSplitter() {
	threshold = 0.5;
}

bool TextSplitter::Load(const String& model_path) {
	return model.Load(model_path);
}

Vector<int> TextSplitter::PredictBoundaries(const Image& line_img) {
	if (line_img.IsEmpty() || model.GetSession().GetLayerCount() == 0) return Vector<int>();
	
	Image normalized = ResizeHeight(line_img, 28);
	Vector<double> scores;
	scores.SetCount(normalized.GetWidth(), 0.0);

	// Slide window across image
	for (int x = 0; x < normalized.GetWidth(); x++) {
		Image window = ExtractWindow(normalized, x, 3, 28);
		
		// Convert Image to Vector<double> for model input
		Vector<double> input;
		for (int wy = 0; wy < 28; wy++) {
			for (int wx = 0; wx < 3; wx++) {
				input.Add(GetGrayValue(window[wy][wx]) / 255.0);
			}
		}
		
		Vector<double> output = model.GetSession().Predict(input);
		if (output.GetCount() >= 2) {
			scores[x] = output[1];  // Probability of boundary
		}
	}

	// Non-maximum suppression
	Vector<int> boundaries;
	for (int x = 1; x < scores.GetCount() - 1; x++) {
		if (scores[x] > threshold &&
		    scores[x] >= scores[x-1] &&
		    scores[x] >= scores[x+1]) {
			
			// Map back to original image scale
			double scale = (double)line_img.GetHeight() / 28.0;
			boundaries.Add((int)(x * scale));
		}
	}

	return boundaries;
}

SplitterTrainingData TextSplitter::GenerateTrainingData(const OCRDatasetProject& proj) {
	SplitterTrainingData data;

	for (int i = 0; i < proj.GetCount(); i++) {
		const Image& img = proj.GetImage(i);
		const Vector<OCRSegment>& segs = proj.GetSegments(i);

		for (const OCRSegment& seg : segs) {
			Image line_img = CropImage(img, seg.bounds);
			line_img = ResizeHeight(line_img, 28);

			// Positive samples: at character boundaries
			for (int j = 0; j < seg.chars.GetCount(); j++) {
				// Normalize boundary x coordinate
				double scale = 28.0 / seg.bounds.Height();
				int boundary_x = (int)((seg.chars[j].bounds.left - seg.bounds.left) * scale);

				Image window = ExtractWindow(line_img, boundary_x, 3, 28);
				data.windows.Add(window);
				data.is_boundary.Add(true);
			}

			// Negative samples
			int num_negatives = seg.chars.GetCount() * 2;
			for (int j = 0; j < num_negatives; j++) {
				int x = Random(line_img.GetWidth());

				bool near_boundary = false;
				double scale = 28.0 / seg.bounds.Height();
				for (const OCRChar& ch : seg.chars) {
					int bx = (int)((ch.bounds.left - seg.bounds.left) * scale);
					if (abs(x - bx) < 3) {
						near_boundary = true;
						break;
					}
				}

				if (!near_boundary) {
					Image window = ExtractWindow(line_img, x, 3, 28);
					data.windows.Add(window);
					data.is_boundary.Add(false);
				}
			}

			// Add partial character examples (last char cropped)
			if (seg.chars.GetCount() > 0) {
				const Rect& last_char = seg.chars.Top().bounds;
				Vector<double> crop_ratios = {0.25, 0.5, 0.75};

				for (double ratio : crop_ratios) {
					int crop_x = (int)(last_char.left + last_char.Width() * ratio);
					if (crop_x > seg.bounds.left) {
						Image cropped = CropImage(img, Rect(seg.bounds.left, seg.bounds.top, crop_x, seg.bounds.bottom));
						cropped = ResizeHeight(cropped, 28);

						int edge_x = cropped.GetWidth() - 2;
						Image window = ExtractWindow(cropped, edge_x, 3, 28);
						data.windows.Add(window);
						data.is_boundary.Add(ratio > 0.3);
					}
				}
			}
		}
	}

	return data;
}

Image TextSplitter::ExtractWindow(const Image& src, int center_x, int width, int height) {
	ImageBuffer ib(width, height);
	Fill(~ib, White(), width * height);

	int start_x = center_x - width / 2;
	
	for (int y = 0; y < height; y++) {
		for (int dx = 0; dx < width; dx++) {
			int sx = start_x + dx;
			if (sx >= 0 && sx < src.GetWidth() && y < src.GetHeight()) {
				ib[y][dx] = src[y][sx];
			}
		}
	}

	return ib;
}

} // namespace OCR

} // namespace Upp
