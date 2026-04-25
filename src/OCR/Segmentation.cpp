#include "Segmentation.h"

namespace Upp {

namespace OCR {

// Character segmentation from text line
Vector<Rect> SegmentCharacters(const Image& line_img, int min_width, int min_height) {
	if (line_img.IsEmpty()) return Vector<Rect>();

	// Binarize image
	Image bw = AdaptiveBinarize(line_img);

	// Get vertical projection
	Vector<int> v_proj = VerticalProjection(bw);

	// Find character boundaries
	Vector<int> boundaries;
	bool in_char = false;
	int start_x = 0;

	for (int x = 0; x < v_proj.GetCount(); x++) {
		if (!in_char && v_proj[x] > 0) {
			// Start of character
			in_char = true;
			start_x = x;
		}
		else if (in_char && v_proj[x] == 0) {
			// End of character
			in_char = false;
			int width = x - start_x;
			if (width >= min_width) {
				boundaries.Add(start_x);
				boundaries.Add(x);
			}
		}
	}

	// Handle last character if line ends with one
	if (in_char) {
		boundaries.Add(start_x);
		boundaries.Add(v_proj.GetCount());
	}

	// Convert boundaries to rectangles
	Vector<Rect> chars;
	for (int i = 0; i < boundaries.GetCount(); i += 2) {
		int x1 = boundaries[i];
		int x2 = boundaries[i + 1];

		// Find vertical bounds within this horizontal range
		int min_y = bw.GetHeight();
		int max_y = 0;

		for (int x = x1; x < x2; x++) {
			for (int y = 0; y < bw.GetHeight(); y++) {
				if (bw[y][x].r < 128) {
					min_y = min(min_y, y);
					max_y = max(max_y, y);
				}
			}
		}

		if (max_y >= min_y && (max_y - min_y + 1) >= min_height) {
			chars.Add(Rect(x1, min_y, x2, max_y + 1));
		}
	}

	return chars;
}

// Line segmentation from page
Vector<Rect> SegmentLines(const Image& page_img, int min_height) {
	if (page_img.IsEmpty()) return Vector<Rect>();

	// Binarize image
	Image bw = AdaptiveBinarize(page_img);

	// Get horizontal projection
	Vector<int> h_proj = HorizontalProjection(bw);

	// Find line boundaries
	Vector<int> boundaries;
	bool in_line = false;
	int start_y = 0;

	for (int y = 0; y < h_proj.GetCount(); y++) {
		if (!in_line && h_proj[y] > 0) {
			// Start of line
			in_line = true;
			start_y = y;
		}
		else if (in_line && h_proj[y] == 0) {
			// End of line
			in_line = false;
			int height = y - start_y;
			if (height >= min_height) {
				boundaries.Add(start_y);
				boundaries.Add(y);
			}
		}
	}

	// Handle last line if page ends with one
	if (in_line) {
		boundaries.Add(start_y);
		boundaries.Add(h_proj.GetCount());
	}

	// Convert boundaries to rectangles
	Vector<Rect> lines;
	for (int i = 0; i < boundaries.GetCount(); i += 2) {
		int y1 = boundaries[i];
		int y2 = boundaries[i + 1];

		// Find horizontal extent
		int min_x = bw.GetWidth();
		int max_x = 0;

		for (int y = y1; y < y2; y++) {
			for (int x = 0; x < bw.GetWidth(); x++) {
				if (bw[y][x].r < 128) {
					min_x = min(min_x, x);
					max_x = max(max_x, x);
				}
			}
		}

		if (max_x >= min_x) {
			lines.Add(Rect(min_x, y1, max_x + 1, y2));
		}
	}

	return lines;
}

// Find character boundaries using projection
Vector<int> FindCharacterBoundaries(const Image& line_img, int min_width) {
	if (line_img.IsEmpty()) return Vector<int>();

	Image bw = AdaptiveBinarize(line_img);
	Vector<int> v_proj = VerticalProjection(bw);

	Vector<int> boundaries;
	boundaries.Add(0);  // Always start at 0

	bool in_char = false;

	for (int x = 0; x < v_proj.GetCount(); x++) {
		if (!in_char && v_proj[x] > 0) {
			in_char = true;
			if (x > 0 && boundaries.Top() < x - 1) {
				boundaries.Add(x);
			}
		}
		else if (in_char && v_proj[x] == 0) {
			in_char = false;
			boundaries.Add(x);
		}
	}

	// Always end at width
	if (boundaries.Top() < v_proj.GetCount()) {
		boundaries.Add(v_proj.GetCount());
	}

	return boundaries;
}

// Find line boundaries using projection
Vector<int> FindLineBoundaries(const Image& page_img, int min_gap) {
	if (page_img.IsEmpty()) return Vector<int>();

	Image bw = AdaptiveBinarize(page_img);
	Vector<int> h_proj = HorizontalProjection(bw);

	Vector<int> boundaries;
	boundaries.Add(0);  // Always start at 0

	bool in_line = false;
	int gap_count = 0;

	for (int y = 0; y < h_proj.GetCount(); y++) {
		if (h_proj[y] > 0) {
			if (!in_line && gap_count >= min_gap) {
				boundaries.Add(y);
			}
			in_line = true;
			gap_count = 0;
		}
		else {
			if (in_line) {
				gap_count++;
				if (gap_count >= min_gap) {
					boundaries.Add(y - gap_count);
					in_line = false;
				}
			}
			else {
				gap_count++;
			}
		}
	}

	// Always end at height
	if (boundaries.Top() < h_proj.GetCount()) {
		boundaries.Add(h_proj.GetCount());
	}

	return boundaries;
}

// Refine boundaries to tight bounding boxes
Rect FindTightBounds(const Image& img, const Rect& initial_bounds) {
	if (img.IsEmpty() || initial_bounds.IsEmpty()) return Rect();

	Image cropped = CropImage(img, initial_bounds);
	Image bw = AdaptiveBinarize(cropped);

	int min_x = bw.GetWidth();
	int max_x = 0;
	int min_y = bw.GetHeight();
	int max_y = 0;

	for (int y = 0; y < bw.GetHeight(); y++) {
		for (int x = 0; x < bw.GetWidth(); x++) {
			if (bw[y][x].r < 128) {
				min_x = min(min_x, x);
				max_x = max(max_x, x);
				min_y = min(min_y, y);
				max_y = max(max_y, y);
			}
		}
	}

	if (max_x >= min_x && max_y >= min_y) {
		return Rect(
			initial_bounds.left + min_x,
			initial_bounds.top + min_y,
			initial_bounds.left + max_x + 1,
			initial_bounds.top + max_y + 1
		);
	}

	return initial_bounds;
}

// Merge nearby rectangles
Vector<Rect> MergeNearbyRects(const Vector<Rect>& rects, int max_gap) {
	if (rects.IsEmpty()) return Vector<Rect>();

	Vector<Rect> sorted = clone(rects);
	Sort(sorted, [](const Rect& a, const Rect& b) { return a.left < b.left; });

	Vector<Rect> merged;
	Rect current = sorted[0];

	for (int i = 1; i < sorted.GetCount(); i++) {
		int gap = sorted[i].left - current.right;

		if (gap <= max_gap) {
			// Merge
			current = UnionRect(current, sorted[i]);
		}
		else {
			// Start new rectangle
			merged.Add(current);
			current = sorted[i];
		}
	}

	merged.Add(current);

	return merged;
}

// Split wide rectangles
Vector<Rect> SplitWideRects(const Vector<Rect>& rects, const Image& img, double max_aspect_ratio) {
	Vector<Rect> result;

	for (const Rect& r : rects) {
		double aspect = (double)r.Width() / r.Height();

		if (aspect > max_aspect_ratio) {
			// This rectangle is too wide, try to split it
			Image cropped = CropImage(img, r);
			Vector<Rect> sub_chars = SegmentCharacters(cropped);

			if (sub_chars.GetCount() > 1) {
				// Successfully split
				for (const Rect& sub : sub_chars) {
					result.Add(Rect(
						r.left + sub.left,
						r.top + sub.top,
						r.left + sub.right,
						r.top + sub.bottom
					));
				}
			}
			else {
				// Couldn't split, keep original
				result.Add(r);
			}
		}
		else {
			result.Add(r);
		}
	}

	return result;
}

// Advanced segmentation using connected components
Vector<Rect> SegmentCharactersConnectedComponents(const Image& line_img, int min_area) {
	if (line_img.IsEmpty()) return Vector<Rect>();

	Image bw = AdaptiveBinarize(line_img);
	Vector<ConnectedComponent> components = FindConnectedComponents(bw, min_area);

	Vector<Rect> chars;
	for (const ConnectedComponent& comp : components) {
		chars.Add(comp.bounds);
	}

	// Sort left to right
	Sort(chars, [](const Rect& a, const Rect& b) { return a.left < b.left; });

	return chars;
}

// Watershed-based segmentation for touching characters
Vector<Rect> SegmentTouchingCharacters(const Image& line_img) {
	if (line_img.IsEmpty()) return Vector<Rect>();

	// First try connected components
	Vector<Rect> chars = SegmentCharactersConnectedComponents(line_img);

	// Check for overly wide components (likely touching characters)
	Vector<Rect> refined;
	Image bw = AdaptiveBinarize(line_img);

	for (const Rect& r : chars) {
		double aspect = (double)r.Width() / r.Height();

		if (aspect > 2.5) {
			// Likely touching characters - try to split
			Image cropped = CropImage(bw, r);

			// Use vertical projection to find split points
			Vector<int> v_proj = VerticalProjection(cropped);

			// Find local minima in projection
			Vector<int> split_points;
			for (int x = 2; x < v_proj.GetCount() - 2; x++) {
				// More robust local minima (handles small plateaus)
				if (v_proj[x] <= v_proj[x-1] && v_proj[x] <= v_proj[x+1] && 
				    (v_proj[x] < v_proj[x-2] || v_proj[x] < v_proj[x+2]) &&
				    v_proj[x] < r.Height() * 0.5) {
					split_points.Add(x);
					x += 3; // Avoid multiple splits close to each other
				}
			}

			if (!split_points.IsEmpty()) {
				// Split at the minima
				int prev_x = 0;
				for (int split_x : split_points) {
					if (split_x - prev_x > 3) {  // Min width
						refined.Add(Rect(
							r.left + prev_x,
							r.top,
							r.left + split_x,
							r.bottom
						));
					}
					prev_x = split_x;
				}
				// Last segment
				if (r.Width() - prev_x > 3) {
					refined.Add(Rect(
						r.left + prev_x,
						r.top,
						r.right,
						r.bottom
					));
				}
			}
			else {
				refined.Add(r);
			}
		}
		else {
			refined.Add(r);
		}
	}

	return refined;
}

} // namespace OCR

} // namespace Upp
