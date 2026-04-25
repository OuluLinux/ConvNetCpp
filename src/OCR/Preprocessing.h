#ifndef _OCR_Preprocessing_h_
#define _OCR_Preprocessing_h_

#include "OCRTypes.h"
#include "ImageUtils.h"

namespace Upp {

namespace OCR {

enum OCRPreprocessMode {
	OCR_PREPROCESS_NONE = 0,
	OCR_PREPROCESS_GRAYSCALE,
	OCR_PREPROCESS_GRAYSCALE_INVERSE,
	OCR_PREPROCESS_GRAYSCALE_INVERSE_CONTRAST_STRETCH,
	OCR_PREPROCESS_COLOR_INVERSE,
	OCR_PREPROCESS_ADAPTIVE_THRESHOLD,
	OCR_PREPROCESS_ADAPTIVE_THRESHOLD_INVERSE,
	OCR_PREPROCESS_OTSU,
	OCR_PREPROCESS_OTSU_INVERSE,
};

inline const char* OCRPreprocessModeToString(OCRPreprocessMode m) {
	switch(m) {
		case OCR_PREPROCESS_GRAYSCALE: return "grayscale";
		case OCR_PREPROCESS_GRAYSCALE_INVERSE: return "grayscale_inverse";
		case OCR_PREPROCESS_GRAYSCALE_INVERSE_CONTRAST_STRETCH: return "grayscale_inverse_contrast_stretch";
		case OCR_PREPROCESS_COLOR_INVERSE: return "color_inverse";
		case OCR_PREPROCESS_ADAPTIVE_THRESHOLD: return "adaptive_threshold";
		case OCR_PREPROCESS_ADAPTIVE_THRESHOLD_INVERSE: return "adaptive_threshold_inverse";
		case OCR_PREPROCESS_OTSU: return "otsu";
		case OCR_PREPROCESS_OTSU_INVERSE: return "otsu_inverse";
		case OCR_PREPROCESS_NONE:
		default: return "none";
	}
}

inline OCRPreprocessMode StringToOCRPreprocessMode(const String& s) {
	if(s == "grayscale") return OCR_PREPROCESS_GRAYSCALE;
	if(s == "grayscale_inverse") return OCR_PREPROCESS_GRAYSCALE_INVERSE;
	if(s == "grayscale_inverse_contrast_stretch") return OCR_PREPROCESS_GRAYSCALE_INVERSE_CONTRAST_STRETCH;
	if(s == "color_inverse") return OCR_PREPROCESS_COLOR_INVERSE;
	if(s == "adaptive_threshold") return OCR_PREPROCESS_ADAPTIVE_THRESHOLD;
	if(s == "adaptive_threshold_inverse") return OCR_PREPROCESS_ADAPTIVE_THRESHOLD_INVERSE;
	if(s == "otsu") return OCR_PREPROCESS_OTSU;
	if(s == "otsu_inverse") return OCR_PREPROCESS_OTSU_INVERSE;
	return OCR_PREPROCESS_NONE;
}

// Preprocessing pipeline
struct PreprocessingOptions {
	OCRPreprocessMode mode = OCR_PREPROCESS_GRAYSCALE;
	bool binarize = false;
	bool denoise = false;
	bool normalize_contrast = false;
	bool remove_borders = false;
	int binarize_threshold = 128;  // -1 for Otsu
	int denoise_kernel = 3;
	int border_margin = 5;

	PreprocessingOptions() {}
};

// Apply full preprocessing pipeline
Image Preprocess(const Image& src, const PreprocessingOptions& opts = PreprocessingOptions());

// Individual preprocessing steps
Image RemoveBorders(const Image& src, int margin = 5);
Image LinearContrastStretch(const Image& src);
Image DeskewImage(const Image& src);
Image RemoveNoise(const Image& src, int kernel_size = 3);

// Advanced preprocessing
Image EnhanceText(const Image& src);  // Enhance text clarity
Image RemoveBackground(const Image& src);  // Remove background patterns
Image CorrectIllumination(const Image& src);  // Correct uneven lighting

// Skew detection and correction
double DetectSkew(const Image& img);
Image RotateImage(const Image& src, double angle_degrees);

} // namespace OCR

} // namespace Upp

#endif
