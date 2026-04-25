#include "Preprocessing.h"
#include <cmath>

namespace Upp {

namespace OCR {

// Apply full preprocessing pipeline
Image Preprocess(const Image& src, const PreprocessingOptions& opts) {
	if (src.IsEmpty()) return Image();

	Image result = src;

	// Initial mode-based processing
	switch(opts.mode) {
		case OCR_PREPROCESS_GRAYSCALE:
			result = GrayscaleOCR(result);
			break;
		case OCR_PREPROCESS_GRAYSCALE_INVERSE:
			result = InvertGrayscale(result);
			break;
		case OCR_PREPROCESS_GRAYSCALE_INVERSE_CONTRAST_STRETCH:
			result = LinearContrastStretch(InvertGrayscale(result));
			break;
		case OCR_PREPROCESS_COLOR_INVERSE:
			result = Invert(result);
			break;
		case OCR_PREPROCESS_ADAPTIVE_THRESHOLD:
			result = AdaptiveBinarize(result);
			break;
		case OCR_PREPROCESS_ADAPTIVE_THRESHOLD_INVERSE:
			result = Invert(AdaptiveBinarize(result));
			break;
		case OCR_PREPROCESS_OTSU:
			result = Binarize(result, OtsuThreshold(result));
			break;
		case OCR_PREPROCESS_OTSU_INVERSE:
			result = Invert(Binarize(result, OtsuThreshold(result)));
			break;
		case OCR_PREPROCESS_NONE:
		default:
			break;
	}

	// Skip standard binarization if we already did it in the switch
	bool already_binarized = (opts.mode >= OCR_PREPROCESS_ADAPTIVE_THRESHOLD);

	// Denoise before binarization (if not already binarized)
	if (opts.denoise) {
		result = Denoise(result, opts.denoise_kernel);
	}

	// Contrast normalization
	if (opts.normalize_contrast && !already_binarized) {
		result = EqualizeHistogram(result);
	}

	// Binarization
	if (opts.binarize && !already_binarized) {
		if (opts.binarize_threshold < 0) {
			// Use Otsu's method
			int threshold = OtsuThreshold(result);
			result = Binarize(result, threshold);
		}
		else {
			result = Binarize(result, opts.binarize_threshold);
		}
	}

	// Remove borders
	if (opts.remove_borders) {
		result = RemoveBorders(result, opts.border_margin);
	}

	return result;
}

// Remove borders
Image RemoveBorders(const Image& src, int margin) {
	if (src.IsEmpty()) return Image();

	Size sz = src.GetSize();
	if (margin * 2 >= sz.cx || margin * 2 >= sz.cy) {
		return src;  // Margin too large
	}

	Rect inner(margin, margin, sz.cx - margin, sz.cy - margin);
	return CropImage(src, inner);
}

// Linear contrast stretch: (v - min) / (max - min) * 255
Image LinearContrastStretch(const Image& src) {
	if(src.IsEmpty()) return Image();

	Size sz = src.GetSize();
	int lo = 255, hi = 0;
	for(int y = 0; y < sz.cy; y++)
		for(int x = 0; x < sz.cx; x++) {
			int v = src[y][x].r;
			if(v < lo) lo = v;
			if(v > hi) hi = v;
		}
	if(hi <= lo) return src;

	ImageBuffer ib(sz);
	for(int y = 0; y < sz.cy; y++)
		for(int x = 0; x < sz.cx; x++) {
			int v = (src[y][x].r - lo) * 255 / (hi - lo);
			RGBA& px = ib[y][x];
			px.r = px.g = px.b = (byte)v;
			px.a = 255;
		}
	return ib;
}

// Deskew image
Image DeskewImage(const Image& src) {
	if (src.IsEmpty()) return Image();

	double skew_angle = DetectSkew(src);

	if (fabs(skew_angle) > 0.5) {  // Only correct if skew > 0.5 degrees
		return RotateImage(src, -skew_angle);
	}

	return src;
}

// Remove noise
Image RemoveNoise(const Image& src, int kernel_size) {
	if (src.IsEmpty()) return Image();

	// Use median filter for noise removal
	return MedianFilter(src, kernel_size);
}

// Enhance text
Image EnhanceText(const Image& src) {
	if (src.IsEmpty()) return Image();

	Image result = src;

	// Convert to grayscale if needed
	if (!IsGrayscale(result)) {
		result = GrayscaleOCR(result);
	}

	// Enhance contrast
	result = EqualizeHistogram(result);

	// Apply morphological closing to fill gaps
	result = MorphologicalClose(result, 2);

	// Sharpen
	// (Simple sharpening using unsharp mask concept)
	Image blurred = Denoise(result, 3);
	Size sz = result.GetSize();
	ImageBuffer ib(sz);

	for (int y = 0; y < sz.cy; y++) {
		for (int x = 0; x < sz.cx; x++) {
			int original = result[y][x].r;
			int blur = blurred[y][x].r;
			int enhanced = original + (original - blur);
			enhanced = minmax(enhanced, 0, 255);

			RGBA& px = ib[y][x];
			px.r = px.g = px.b = (byte)enhanced;
			px.a = 255;
		}
	}

	return ib;
}

// Remove background
Image RemoveBackground(const Image& src) {
	if (src.IsEmpty()) return Image();

	Image gray = GrayscaleOCR(src);

	// Estimate background using large-scale blur
	Image background = gray;
	for (int i = 0; i < 3; i++) {
		background = Denoise(background, 15);
	}

	// Subtract background
	Size sz = gray.GetSize();
	ImageBuffer ib(sz);

	for (int y = 0; y < sz.cy; y++) {
		for (int x = 0; x < sz.cx; x++) {
			int fg = gray[y][x].r;
			int bg = background[y][x].r;
			int diff = fg - bg + 128;
			diff = minmax(diff, 0, 255);

			RGBA& px = ib[y][x];
			px.r = px.g = px.b = (byte)diff;
			px.a = 255;
		}
	}

	return ib;
}

// Correct illumination
Image CorrectIllumination(const Image& src) {
	if (src.IsEmpty()) return Image();

	// Similar to background removal but preserves intensity
	Image gray = GrayscaleOCR(src);

	// Estimate illumination using large-scale blur
	Image illumination = gray;
	for (int i = 0; i < 5; i++) {
		illumination = Denoise(illumination, 21);
	}

	// Normalize by illumination
	Size sz = gray.GetSize();
	ImageBuffer ib(sz);

	for (int y = 0; y < sz.cy; y++) {
		for (int x = 0; x < sz.cx; x++) {
			int original = gray[y][x].r;
			int illum = illumination[y][x].r;

			int corrected = 128;
			if (illum > 10) {
				corrected = (original * 128) / illum;
				corrected = minmax(corrected, 0, 255);
			}

			RGBA& px = ib[y][x];
			px.r = px.g = px.b = (byte)corrected;
			px.a = 255;
		}
	}

	return ib;
}

// Detect skew using Hough transform (simplified)
double DetectSkew(const Image& img) {
	if (img.IsEmpty()) return 0.0;

	// Binarize
	Image bw = AdaptiveBinarize(img);

	// Find edges (simple sobel-like)
	Size sz = bw.GetSize();
	Vector<Point> edge_points;

	for (int y = 1; y < sz.cy - 1; y++) {
		for (int x = 1; x < sz.cx - 1; x++) {
			if (bw[y][x].r < 128) {
				// Check if this is an edge point
				int neighbors = 0;
				for (int dy = -1; dy <= 1; dy++) {
					for (int dx = -1; dx <= 1; dx++) {
						if (dx == 0 && dy == 0) continue;
						if (bw[y+dy][x+dx].r < 128) neighbors++;
					}
				}
				if (neighbors > 0 && neighbors < 8) {
					edge_points.Add(Point(x, y));
				}
			}
		}
	}

	if (edge_points.IsEmpty()) return 0.0;

	// Simplified Hough transform for angles between -15 and +15 degrees
	Vector<int> votes(31, 0);  // -15 to +15 degrees
	int sample_count = min(1000, edge_points.GetCount());

	for (int i = 0; i < sample_count; i++) {
		Point p = edge_points[Random(edge_points.GetCount())];

		for (int angle_idx = 0; angle_idx < 31; angle_idx++) {
			double angle = (angle_idx - 15) * M_PI / 180.0;
			double rho = p.x * cos(angle) + p.y * sin(angle);

			// Vote for this angle
			votes[angle_idx]++;
		}
	}

	// Find peak
	int max_votes = 0;
	int best_angle_idx = 15;  // 0 degrees

	for (int i = 0; i < votes.GetCount(); i++) {
		if (votes[i] > max_votes) {
			max_votes = votes[i];
			best_angle_idx = i;
		}
	}

	return (best_angle_idx - 15);  // Return angle in degrees
}

// Rotate image
Image RotateImage(const Image& src, double angle_degrees) {
	if (src.IsEmpty()) return Image();

	// For now, only support small rotations
	if (fabs(angle_degrees) > 45) return src;

	double angle_rad = angle_degrees * M_PI / 180.0;
	double cos_a = cos(angle_rad);
	double sin_a = sin(angle_rad);

	Size sz = src.GetSize();

	// Calculate new size
	int new_w = (int)(fabs(sz.cx * cos_a) + fabs(sz.cy * sin_a));
	int new_h = (int)(fabs(sz.cx * sin_a) + fabs(sz.cy * cos_a));

	ImageBuffer ib(new_w, new_h);
	Fill(~ib, White(), new_w * new_h);

	int cx = sz.cx / 2;
	int cy = sz.cy / 2;
	int new_cx = new_w / 2;
	int new_cy = new_h / 2;

	// Inverse rotation mapping
	for (int y = 0; y < new_h; y++) {
		for (int x = 0; x < new_w; x++) {
			int dx = x - new_cx;
			int dy = y - new_cy;

			int src_x = (int)(dx * cos_a + dy * sin_a) + cx;
			int src_y = (int)(-dx * sin_a + dy * cos_a) + cy;

			if (src_x >= 0 && src_x < sz.cx && src_y >= 0 && src_y < sz.cy) {
				ib[y][x] = src[src_y][src_x];
			}
		}
	}

	return ib;
}

} // namespace OCR

} // namespace Upp
