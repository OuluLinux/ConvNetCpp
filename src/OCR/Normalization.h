#ifndef _OCR_Normalization_h_
#define _OCR_Normalization_h_

#include "OCRTypes.h"
#include "ImageUtils.h"

namespace Upp {

namespace OCR {

enum NormalizationStrategy {
	NORM_ASPECT_FIT = 0,    // Preserve aspect ratio, pad to fit
	NORM_STRETCH = 1,       // Stretch to fill target size
	NORM_WIDTH_FIT = 2,     // Fit to width, may crop or pad height
	NORM_HEIGHT_FIT = 3,    // Fit to height, may crop or pad width
};

// Normalize character image to fixed size
Image NormalizeChar(const Image& src, Size target = Size(28, 28), int strategy = NORM_ASPECT_FIT);

// Center character in frame
Image CenterInFrame(const Image& src, Size frame, bool use_mass_center = false);

// Pad image to specific aspect ratio
Image PadToAspect(const Image& src, double aspect);

// Batch processing
Vector<Image> NormalizeBatch(const Vector<Image>& chars, Size target = Size(28, 28), int strategy = NORM_ASPECT_FIT);

// Elastic deformation for data augmentation
Image ElasticDeform(const Image& src, double sigma, double alpha);

// Random data augmentation for training
Image AugmentChar(const Image& src);

} // namespace OCR

} // namespace Upp

#endif
