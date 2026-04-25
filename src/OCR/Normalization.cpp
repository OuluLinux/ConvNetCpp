#include "Normalization.h"
#include "Segmentation.h"
#include "Preprocessing.h"
#include <cmath>

namespace Upp {

namespace OCR {

Image NormalizeChar(const Image& src, Size target, int strategy) {
	if (src.IsEmpty()) return Image();

	// First find tight bounds of the character
	Rect tight = FindTightBounds(src, src.GetSize());
	Image cropped = CropImage(src, tight);

	if (strategy == NORM_STRETCH) {
		return ScaleImage(cropped, target);
	}

	double src_aspect = (double)cropped.GetWidth() / cropped.GetHeight();
	double target_aspect = (double)target.cx / target.cy;

	if (strategy == NORM_ASPECT_FIT) {
		double scale;
		if (src_aspect > target_aspect) {
			scale = (double)target.cx / cropped.GetWidth();
		} else {
			scale = (double)target.cy / cropped.GetHeight();
		}

		int new_w = (int)(cropped.GetWidth() * scale);
		int new_h = (int)(cropped.GetHeight() * scale);
		
		// Ensure at least 1x1
		new_w = max(1, new_w);
		new_h = max(1, new_h);

		Image scaled = ScaleImage(cropped, Size(new_w, new_h));
		return CenterInFrame(scaled, target);
	}

	if (strategy == NORM_WIDTH_FIT) {
		Image scaled = ResizeWidth(cropped, target.cx);
		return CenterInFrame(scaled, target);
	}

	if (strategy == NORM_HEIGHT_FIT) {
		Image scaled = ResizeHeight(cropped, target.cy);
		return CenterInFrame(scaled, target);
	}

	return ScaleImage(cropped, target);
}

Image CenterInFrame(const Image& src, Size frame, bool use_mass_center) {
	if (src.IsEmpty()) return Image();

	Size sz = src.GetSize();
	ImageBuffer ib(frame);
	Fill(~ib, White(), frame.cx * frame.cy);

	int dx, dy;

	if (use_mass_center) {
		// Calculate mass center
		double sum_x = 0, sum_y = 0, count = 0;
		for (int y = 0; y < sz.cy; y++) {
			for (int x = 0; x < sz.cx; x++) {
				if (src[y][x].r < 128) {
					sum_x += x;
					sum_y += y;
					count++;
				}
			}
		}

		if (count > 0) {
			Point mc((int)(sum_x / count), (int)(sum_y / count));
			dx = frame.cx / 2 - mc.x;
			dy = frame.cy / 2 - mc.y;
		} else {
			dx = (frame.cx - sz.cx) / 2;
			dy = (frame.cy - sz.cy) / 2;
		}
	} else {
		// Geometric center
		dx = (frame.cx - sz.cx) / 2;
		dy = (frame.cy - sz.cy) / 2;
	}

	for (int y = 0; y < sz.cy; y++) {
		for (int x = 0; x < sz.cx; x++) {
			int tx = x + dx;
			int ty = y + dy;
			if (tx >= 0 && tx < frame.cx && ty >= 0 && ty < frame.cy) {
				ib[ty][tx] = src[y][x];
			}
		}
	}

	return ib;
}

Image PadToAspect(const Image& src, double aspect) {
	if (src.IsEmpty()) return Image();

	Size sz = src.GetSize();
	double cur_aspect = (double)sz.cx / sz.cy;

	if (cur_aspect > aspect) {
		// Too wide, add padding to height
		int new_h = (int)(sz.cx / aspect);
		return CenterInFrame(src, Size(sz.cx, new_h));
	} else {
		// Too tall, add padding to width
		int new_w = (int)(sz.cy * aspect);
		return CenterInFrame(src, Size(new_w, sz.cy));
	}
}

Vector<Image> NormalizeBatch(const Vector<Image>& chars, Size target, int strategy) {
	Vector<Image> result;
	for (const Image& img : chars) {
		result.Add(NormalizeChar(img, target, strategy));
	}
	return result;
}

// Simplified elastic deformation
Image ElasticDeform(const Image& src, double sigma, double alpha) {
	if (src.IsEmpty()) return Image();

	Size sz = src.GetSize();
	ImageBuffer ib(sz);
	Fill(~ib, White(), sz.cx * sz.cy);

	// In a real implementation, we would generate a random displacement field,
	// convolve it with a Gaussian (sigma), and scale it (alpha).
	// Here's a simplified version using a few sine waves.
	
	double freq_x = 2.0 * M_PI / sz.cx;
	double freq_y = 2.0 * M_PI / sz.cy;
	
	double phase_x = (double)Random(1000) / 100.0;
	double phase_y = (double)Random(1000) / 100.0;

	for (int y = 0; y < sz.cy; y++) {
		for (int x = 0; x < sz.cx; x++) {
			double dx = alpha * sin(freq_y * y + phase_x);
			double dy = alpha * sin(freq_x * x + phase_y);

			int sx = (int)(x + dx);
			int sy = (int)(y + dy);

			if (sx >= 0 && sx < sz.cx && sy >= 0 && sy < sz.cy) {
				ib[y][x] = src[sy][sx];
			}
		}
	}

	return ib;
}

Image AugmentChar(const Image& src) {
	if (src.IsEmpty()) return Image();
	Image img = src;

	// Random rotation (+/- 5 degrees)
	if (Random(100) > 50) {
		double angle = (Random(1000) / 100.0 - 5.0);
		img = RotateImage(img, angle);
	}

	// Random translation (+/- 2 pixels)
	if (Random(100) > 50) {
		int dx = Random(5) - 2;
		int dy = Random(5) - 2;
		
		Size sz = img.GetSize();
		ImageBuffer ib(sz);
		Fill(~ib, White(), sz.cx * sz.cy);
		
		for(int y = 0; y < sz.cy; y++) {
			for(int x = 0; x < sz.cx; x++) {
				int sx = x - dx;
				int sy = y - dy;
				if (sx >= 0 && sx < sz.cx && sy >= 0 && sy < sz.cy) {
					ib[y][x] = img[sy][sx];
				}
			}
		}
		img = ib;
	}

	return img;
}

} // namespace OCR

} // namespace Upp
