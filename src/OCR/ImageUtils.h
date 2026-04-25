#ifndef _OCR_ImageUtils_h_
#define _OCR_ImageUtils_h_

#include "OCRTypes.h"

namespace Upp {

namespace OCR {

// Grayscale conversion (renamed to avoid conflict with Draw package)
Image GrayscaleOCR(const Image& src);
double GetGrayValue(const RGBA& pixel);

// Inversion
Image Invert(const Image& src);
Image InvertGrayscale(const Image& src);

// Binarization
Image Binarize(const Image& src, int threshold = 128);
Image AdaptiveBinarize(const Image& src, int block_size = 15, int c = 10);
int OtsuThreshold(const Image& src);

// Projection
Vector<int> HorizontalProjection(const Image& img);
Vector<int> VerticalProjection(const Image& img);

// Morphological operations
Image Dilate(const Image& src, int kernel_size = 3);
Image Erode(const Image& src, int kernel_size = 3);
Image MorphologicalOpen(const Image& src, int kernel_size = 3);
Image MorphologicalClose(const Image& src, int kernel_size = 3);

// Noise reduction
Image Denoise(const Image& src, int kernel_size = 3);
Image MedianFilter(const Image& src, int kernel_size = 3);

// Histogram
void GetHistogram(const Image& img, Vector<int>& histogram);
Image EqualizeHistogram(const Image& src);

// Scaling
Image ScaleImage(const Image& src, Size target_size);
Image ResizeWidth(const Image& src, int target_width);
Image ResizeHeight(const Image& src, int target_height);

// Statistics
struct ImageStats {
	double mean;
	double std_dev;
	int min_value;
	int max_value;
	Vector<int> histogram;

	ImageStats() : mean(0), std_dev(0), min_value(0), max_value(0) {
		histogram.SetCount(256, 0);
	}

	String ToString() const {
		return Format("ImageStats(mean=%.2f, std=%.2f, min=%d, max=%d)",
		              mean, std_dev, min_value, max_value);
	}
};

ImageStats ComputeImageStats(const Image& img);

// Connected components
struct ConnectedComponent : Moveable<ConnectedComponent> {
	Rect bounds;
	int area;
	Vector<Point> points;

	ConnectedComponent() : area(0) {}

	Point GetCenter() const {
		if (points.IsEmpty()) return Point(0, 0);
		int sum_x = 0, sum_y = 0;
		for (const Point& p : points) {
			sum_x += p.x;
			sum_y += p.y;
		}
		return Point(sum_x / points.GetCount(), sum_y / points.GetCount());
	}
};

Vector<ConnectedComponent> FindConnectedComponents(const Image& img, int min_area = 1);

// Image quality
bool IsImageTooDark(const Image& img, double threshold = 0.3);
bool IsImageTooLight(const Image& img, double threshold = 0.7);
bool IsImageTooBlurry(const Image& img, double threshold = 100.0);

} // namespace OCR

} // namespace Upp

#endif
