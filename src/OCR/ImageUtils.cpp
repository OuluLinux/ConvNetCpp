#include "ImageUtils.h"
#include <cmath>

namespace Upp {

namespace OCR {

// Grayscale conversion

Image GrayscaleOCR(const Image& src) {
	if (src.IsEmpty()) return Image();

	Size sz = src.GetSize();
	ImageBuffer ib(sz);

	for (int y = 0; y < sz.cy; y++) {
		const RGBA* src_line = src[y];
		RGBA* dst_line = ib[y];

		for (int x = 0; x < sz.cx; x++) {
			byte gray = (byte)GetGrayValue(src_line[x]);
			RGBA& px = dst_line[x];
			px.r = px.g = px.b = gray;
			px.a = 255;
		}
	}

	return ib;
}

double GetGrayValue(const RGBA& pixel) {
	// Standard grayscale conversion weights
	return 0.299 * pixel.r + 0.587 * pixel.g + 0.114 * pixel.b;
}

// Inversion

Image Invert(const Image& src) {
	if (src.IsEmpty()) return Image();

	Size sz = src.GetSize();
	ImageBuffer ib(sz);

	for (int y = 0; y < sz.cy; y++) {
		const RGBA* src_line = src[y];
		RGBA* dst_line = ib[y];

		for (int x = 0; x < sz.cx; x++) {
			RGBA& px = dst_line[x];
			px.r = 255 - src_line[x].r;
			px.g = 255 - src_line[x].g;
			px.b = 255 - src_line[x].b;
			px.a = src_line[x].a;
		}
	}

	return ib;
}

Image InvertGrayscale(const Image& src) {
	if (src.IsEmpty()) return Image();

	Size sz = src.GetSize();
	ImageBuffer ib(sz);

	for (int y = 0; y < sz.cy; y++) {
		const RGBA* src_line = src[y];
		RGBA* dst_line = ib[y];

		for (int x = 0; x < sz.cx; x++) {
			byte gray = (byte)GetGrayValue(src_line[x]);
			byte inv_gray = 255 - gray;
			RGBA& px = dst_line[x];
			px.r = px.g = px.b = inv_gray;
			px.a = 255;
		}
	}

	return ib;
}

// Binarization

Image Binarize(const Image& src, int threshold) {
	if (src.IsEmpty()) return Image();

	Size sz = src.GetSize();
	ImageBuffer ib(sz);

	for (int y = 0; y < sz.cy; y++) {
		const RGBA* src_line = src[y];
		RGBA* dst_line = ib[y];

		for (int x = 0; x < sz.cx; x++) {
			byte gray = (byte)GetGrayValue(src_line[x]);
			byte bw = (gray >= threshold) ? 255 : 0;
			RGBA& px = dst_line[x];
			px.r = px.g = px.b = bw;
			px.a = 255;
		}
	}

	return ib;
}

Image AdaptiveBinarize(const Image& src, int block_size, int c) {
	if (src.IsEmpty()) return Image();

	// Convert to grayscale first
	Image gray = GrayscaleOCR(src);
	Size sz = gray.GetSize();

	ImageBuffer ib(sz);
	int half_block = block_size / 2;

	for (int y = 0; y < sz.cy; y++) {
		for (int x = 0; x < sz.cx; x++) {
			// Calculate local threshold
			int sum = 0;
			int count = 0;

			for (int by = max(0, y - half_block); by < min(sz.cy, y + half_block + 1); by++) {
				for (int bx = max(0, x - half_block); bx < min(sz.cx, x + half_block + 1); bx++) {
					sum += gray[by][bx].r;
					count++;
				}
			}

			int local_threshold = sum / count - c;
			byte pixel_value = gray[y][x].r;
			byte bw = (pixel_value >= local_threshold) ? 255 : 0;

			RGBA& px = ib[y][x];
			px.r = px.g = px.b = bw;
			px.a = 255;
		}
	}

	return ib;
}

int OtsuThreshold(const Image& src) {
	Vector<int> histogram;
	GetHistogram(src, histogram);

	int total = 0;
	for (int i = 0; i < 256; i++) {
		total += histogram[i];
	}

	if (total == 0) return 128;

	double sum = 0;
	for (int i = 0; i < 256; i++) {
		sum += i * histogram[i];
	}

	double sum_bg = 0;
	int weight_bg = 0;
	double max_variance = 0;
	int threshold = 0;

	for (int t = 0; t < 256; t++) {
		weight_bg += histogram[t];
		if (weight_bg == 0) continue;

		int weight_fg = total - weight_bg;
		if (weight_fg == 0) break;

		sum_bg += t * histogram[t];

		double mean_bg = sum_bg / weight_bg;
		double mean_fg = (sum - sum_bg) / weight_fg;

		double variance = weight_bg * weight_fg * (mean_bg - mean_fg) * (mean_bg - mean_fg);

		if (variance > max_variance) {
			max_variance = variance;
			threshold = t;
		}
	}

	return threshold;
}

// Projection

Vector<int> HorizontalProjection(const Image& img) {
	if (img.IsEmpty()) return Vector<int>();

	Size sz = img.GetSize();
	Vector<int> proj(sz.cy, 0);

	for (int y = 0; y < sz.cy; y++) {
		const RGBA* line = img[y];
		for (int x = 0; x < sz.cx; x++) {
			// Count black pixels (assume binary image)
			if (line[x].r < 128) {
				proj[y]++;
			}
		}
	}

	return proj;
}

Vector<int> VerticalProjection(const Image& img) {
	if (img.IsEmpty()) return Vector<int>();

	Size sz = img.GetSize();
	Vector<int> proj(sz.cx, 0);

	for (int y = 0; y < sz.cy; y++) {
		const RGBA* line = img[y];
		for (int x = 0; x < sz.cx; x++) {
			// Count black pixels (assume binary image)
			if (line[x].r < 128) {
				proj[x]++;
			}
		}
	}

	return proj;
}

// Morphological operations

Image Dilate(const Image& src, int kernel_size) {
	if (src.IsEmpty()) return Image();

	Size sz = src.GetSize();
	ImageBuffer ib(sz);
	int radius = kernel_size / 2;

	for (int y = 0; y < sz.cy; y++) {
		for (int x = 0; x < sz.cx; x++) {
			int max_val = 0;

			for (int ky = -radius; ky <= radius; ky++) {
				for (int kx = -radius; kx <= radius; kx++) {
					int ny = y + ky;
					int nx = x + kx;

					if (ny >= 0 && ny < sz.cy && nx >= 0 && nx < sz.cx) {
						max_val = max(max_val, (int)src[ny][nx].r);
					}
				}
			}

			RGBA& px = ib[y][x];
			px.r = px.g = px.b = (byte)max_val;
			px.a = 255;
		}
	}

	return ib;
}

Image Erode(const Image& src, int kernel_size) {
	if (src.IsEmpty()) return Image();

	Size sz = src.GetSize();
	ImageBuffer ib(sz);
	int radius = kernel_size / 2;

	for (int y = 0; y < sz.cy; y++) {
		for (int x = 0; x < sz.cx; x++) {
			int min_val = 255;

			for (int ky = -radius; ky <= radius; ky++) {
				for (int kx = -radius; kx <= radius; kx++) {
					int ny = y + ky;
					int nx = x + kx;

					if (ny >= 0 && ny < sz.cy && nx >= 0 && nx < sz.cx) {
						min_val = min(min_val, (int)src[ny][nx].r);
					}
				}
			}

			RGBA& px = ib[y][x];
			px.r = px.g = px.b = (byte)min_val;
			px.a = 255;
		}
	}

	return ib;
}

Image MorphologicalOpen(const Image& src, int kernel_size) {
	return Dilate(Erode(src, kernel_size), kernel_size);
}

Image MorphologicalClose(const Image& src, int kernel_size) {
	return Erode(Dilate(src, kernel_size), kernel_size);
}

// Noise reduction

Image Denoise(const Image& src, int kernel_size) {
	// Simple averaging filter
	if (src.IsEmpty()) return Image();

	Size sz = src.GetSize();
	ImageBuffer ib(sz);
	int radius = kernel_size / 2;

	for (int y = 0; y < sz.cy; y++) {
		for (int x = 0; x < sz.cx; x++) {
			int sum_r = 0, sum_g = 0, sum_b = 0;
			int count = 0;

			for (int ky = -radius; ky <= radius; ky++) {
				for (int kx = -radius; kx <= radius; kx++) {
					int ny = y + ky;
					int nx = x + kx;

					if (ny >= 0 && ny < sz.cy && nx >= 0 && nx < sz.cx) {
						const RGBA& px = src[ny][nx];
						sum_r += px.r;
						sum_g += px.g;
						sum_b += px.b;
						count++;
					}
				}
			}

			RGBA& px = ib[y][x];
			px.r = sum_r / count;
			px.g = sum_g / count;
			px.b = sum_b / count;
			px.a = 255;
		}
	}

	return ib;
}

Image MedianFilter(const Image& src, int kernel_size) {
	if (src.IsEmpty()) return Image();

	Size sz = src.GetSize();
	ImageBuffer ib(sz);
	int radius = kernel_size / 2;

	Vector<byte> values;

	for (int y = 0; y < sz.cy; y++) {
		for (int x = 0; x < sz.cx; x++) {
			values.Clear();

			for (int ky = -radius; ky <= radius; ky++) {
				for (int kx = -radius; kx <= radius; kx++) {
					int ny = y + ky;
					int nx = x + kx;

					if (ny >= 0 && ny < sz.cy && nx >= 0 && nx < sz.cx) {
						values.Add(src[ny][nx].r);
					}
				}
			}

			Sort(values);
			byte median = values[values.GetCount() / 2];

			RGBA& px = ib[y][x];
			px.r = px.g = px.b = median;
			px.a = 255;
		}
	}

	return ib;
}

// Histogram

void GetHistogram(const Image& img, Vector<int>& histogram) {
	histogram.SetCount(256, 0);

	if (img.IsEmpty()) return;

	Size sz = img.GetSize();
	for (int y = 0; y < sz.cy; y++) {
		const RGBA* line = img[y];
		for (int x = 0; x < sz.cx; x++) {
			byte gray = (byte)GetGrayValue(line[x]);
			histogram[gray]++;
		}
	}
}

Image EqualizeHistogram(const Image& src) {
	if (src.IsEmpty()) return Image();

	Vector<int> histogram;
	GetHistogram(src, histogram);

	Size sz = src.GetSize();
	int total_pixels = sz.cx * sz.cy;

	// Compute CDF
	Vector<int> cdf(256, 0);
	cdf[0] = histogram[0];
	for (int i = 1; i < 256; i++) {
		cdf[i] = cdf[i-1] + histogram[i];
	}

	// Normalize CDF
	Vector<byte> lut(256);
	int cdf_min = 0;
	for (int i = 0; i < 256; i++) {
		if (cdf[i] > 0) {
			cdf_min = cdf[i];
			break;
		}
	}

	for (int i = 0; i < 256; i++) {
		lut[i] = (byte)(((cdf[i] - cdf_min) * 255) / (total_pixels - cdf_min));
	}

	// Apply LUT
	ImageBuffer ib(sz);
	for (int y = 0; y < sz.cy; y++) {
		const RGBA* src_line = src[y];
		RGBA* dst_line = ib[y];

		for (int x = 0; x < sz.cx; x++) {
			byte gray = (byte)GetGrayValue(src_line[x]);
			byte new_gray = lut[gray];
			RGBA& px = dst_line[x];
			px.r = px.g = px.b = new_gray;
			px.a = 255;
		}
	}

	return ib;
}

// Scaling

Image ScaleImage(const Image& src, Size target_size) {
	if (src.IsEmpty() || target_size.cx <= 0 || target_size.cy <= 0) {
		return Image();
	}

	return Rescale(src, target_size);
}

Image ResizeWidth(const Image& src, int target_width) {
	if (src.IsEmpty() || target_width <= 0) return Image();

	Size src_sz = src.GetSize();
	double scale = (double)target_width / src_sz.cx;
	int target_height = (int)(src_sz.cy * scale);

	return ScaleImage(src, Size(target_width, target_height));
}

Image ResizeHeight(const Image& src, int target_height) {
	if (src.IsEmpty() || target_height <= 0) return Image();

	Size src_sz = src.GetSize();
	double scale = (double)target_height / src_sz.cy;
	int target_width = (int)(src_sz.cx * scale);

	return ScaleImage(src, Size(target_width, target_height));
}

// Statistics

ImageStats ComputeImageStats(const Image& img) {
	ImageStats stats;

	if (img.IsEmpty()) return stats;

	Size sz = img.GetSize();
	int total_pixels = sz.cx * sz.cy;

	// Compute histogram and mean
	double sum = 0;
	stats.min_value = 255;
	stats.max_value = 0;

	for (int y = 0; y < sz.cy; y++) {
		const RGBA* line = img[y];
		for (int x = 0; x < sz.cx; x++) {
			byte gray = (byte)GetGrayValue(line[x]);
			stats.histogram[gray]++;
			sum += gray;
			stats.min_value = min(stats.min_value, (int)gray);
			stats.max_value = max(stats.max_value, (int)gray);
		}
	}

	stats.mean = sum / total_pixels;

	// Compute standard deviation
	double variance_sum = 0;
	for (int y = 0; y < sz.cy; y++) {
		const RGBA* line = img[y];
		for (int x = 0; x < sz.cx; x++) {
			byte gray = (byte)GetGrayValue(line[x]);
			double diff = gray - stats.mean;
			variance_sum += diff * diff;
		}
	}

	stats.std_dev = sqrt(variance_sum / total_pixels);

	return stats;
}

// Connected components

Vector<ConnectedComponent> FindConnectedComponents(const Image& img, int min_area) {
	if (img.IsEmpty()) return Vector<ConnectedComponent>();

	Size sz = img.GetSize();
	Vector<Vector<int>> labels;
	labels.SetCount(sz.cy);
	for (int y = 0; y < sz.cy; y++) {
		labels[y].SetCount(sz.cx, 0);
	}

	int next_label = 1;

	// Simple flood-fill labeling
	for (int y = 0; y < sz.cy; y++) {
		for (int x = 0; x < sz.cx; x++) {
			if (img[y][x].r < 128 && labels[y][x] == 0) {
				// Start new component
				Vector<Point> queue;
				queue.Add(Point(x, y));
				labels[y][x] = next_label;

				while (!queue.IsEmpty()) {
					Point p = queue.Pop();

					// Check 4-connected neighbors
					Point neighbors[] = {
						Point(p.x-1, p.y), Point(p.x+1, p.y),
						Point(p.x, p.y-1), Point(p.x, p.y+1)
					};

					for (const Point& np : neighbors) {
						if (np.x >= 0 && np.x < sz.cx && np.y >= 0 && np.y < sz.cy &&
						    img[np.y][np.x].r < 128 && labels[np.y][np.x] == 0) {
							labels[np.y][np.x] = next_label;
							queue.Add(np);
						}
					}
				}

				next_label++;
			}
		}
	}

	// Build components
	Vector<ConnectedComponent> components(next_label);

	for (int y = 0; y < sz.cy; y++) {
		for (int x = 0; x < sz.cx; x++) {
			int label = labels[y][x];
			if (label > 0) {
				ConnectedComponent& comp = components[label];
				comp.points.Add(Point(x, y));
				comp.area++;

				if (comp.bounds.IsEmpty()) {
					comp.bounds = Rect(x, y, x+1, y+1);
				} else {
					comp.bounds = UnionRect(comp.bounds, Rect(x, y, x+1, y+1));
				}
			}
		}
	}

	// Filter by min area and remove empty
	Vector<ConnectedComponent> result;
	for (int i = 1; i < components.GetCount(); i++) {
		if (components[i].area >= min_area) {
			result.Add(pick(components[i]));
		}
	}

	return result;
}

// Image quality

bool IsImageTooDark(const Image& img, double threshold) {
	ImageStats stats = ComputeImageStats(img);
	return stats.mean / 255.0 < threshold;
}

bool IsImageTooLight(const Image& img, double threshold) {
	ImageStats stats = ComputeImageStats(img);
	return stats.mean / 255.0 > threshold;
}

bool IsImageTooBlurry(const Image& img, double threshold) {
	// Use Laplacian variance as blur metric
	if (img.IsEmpty()) return true;

	Image gray = GrayscaleOCR(img);
	Size sz = gray.GetSize();

	double variance_sum = 0;
	int count = 0;

	// Laplacian kernel
	for (int y = 1; y < sz.cy - 1; y++) {
		for (int x = 1; x < sz.cx - 1; x++) {
			int laplacian =
				-1 * gray[y-1][x-1].r + -1 * gray[y-1][x].r + -1 * gray[y-1][x+1].r +
				-1 * gray[y][x-1].r   +  8 * gray[y][x].r   + -1 * gray[y][x+1].r +
				-1 * gray[y+1][x-1].r + -1 * gray[y+1][x].r + -1 * gray[y+1][x+1].r;

			variance_sum += laplacian * laplacian;
			count++;
		}
	}

	double variance = variance_sum / count;
	return variance < threshold;
}

} // namespace OCR

} // namespace Upp
