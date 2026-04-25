#include "OCRTypes.h"

namespace Upp {

namespace OCR {

// OCRDataset implementation

OCRDataset::OCRDataset() {
}

OCRDataset::~OCRDataset() {
}

void OCRDataset::Clear() {
	images.Clear();
	image_paths.Clear();
	segments.Clear();
}

void OCRDataset::AddImage(const Image& img, const String& path) {
	images.Add(img);
	image_paths.Add(path);
	segments.Add();  // Empty segment list
}

void OCRDataset::AddSegment(int image_idx, const OCRSegment& seg) {
	CheckIndex(image_idx);
	OCRSegment& copy = segments[image_idx].Add();
	copy.DeepCopyFrom(seg);
}

void OCRDataset::SetSegments(int image_idx, const Vector<OCRSegment>& segs) {
	CheckIndex(image_idx);
	segments[image_idx].Clear();
	for (const OCRSegment& seg : segs) {
		OCRSegment& copy = segments[image_idx].Add();
		copy.DeepCopyFrom(seg);
	}
}

const Image& OCRDataset::GetImage(int idx) const {
	CheckIndex(idx);
	return images[idx];
}

Image& OCRDataset::GetImage(int idx) {
	CheckIndex(idx);
	return images[idx];
}

const String& OCRDataset::GetImagePath(int idx) const {
	CheckIndex(idx);
	return image_paths[idx];
}

const Vector<OCRSegment>& OCRDataset::GetSegments(int idx) const {
	CheckIndex(idx);
	return segments[idx];
}

Vector<OCRSegment>& OCRDataset::GetSegments(int idx) {
	CheckIndex(idx);
	return segments[idx];
}

void OCRDataset::RemoveImage(int idx) {
	CheckIndex(idx);
	images.Remove(idx);
	image_paths.Remove(idx);
	segments.Remove(idx);
}

int OCRDataset::GetTotalCharacters() const {
	int count = 0;
	for (const auto& segs : segments) {
		for (const auto& seg : segs) {
			count += seg.chars.GetCount();
		}
	}
	return count;
}

int OCRDataset::GetTotalSegments() const {
	int count = 0;
	for (const auto& segs : segments) {
		count += segs.GetCount();
	}
	return count;
}

void OCRDataset::GetClassDistribution(Vector<int>& counts) const {
	counts.SetCount(256, 0);

	for (const auto& segs : segments) {
		for (const auto& seg : segs) {
			for (const auto& ch : seg.chars) {
				if (ch.label >= 0 && ch.label < 256) {
					counts[ch.label]++;
				}
			}
		}
	}
}

bool OCRDataset::Validate(String* error) const {
	if (images.GetCount() != image_paths.GetCount() ||
	    images.GetCount() != segments.GetCount()) {
		if (error) *error = "Size mismatch between images, paths, and segments";
		return false;
	}

	for (int i = 0; i < images.GetCount(); i++) {
		if (images[i].IsEmpty()) {
			if (error) *error = Format("Image %d is empty", i);
			return false;
		}

		Size img_sz = images[i].GetSize();

		for (const auto& seg : segments[i]) {
			// Check segment bounds within image
			if (seg.bounds.left < 0 || seg.bounds.top < 0 ||
			    seg.bounds.right > img_sz.cx || seg.bounds.bottom > img_sz.cy) {
				if (error) *error = Format("Image %d: segment bounds outside image", i);
				return false;
			}

			// Check character bounds within segment
			for (const auto& ch : seg.chars) {
				if (ch.bounds.left < seg.bounds.left ||
				    ch.bounds.right > seg.bounds.right ||
				    ch.bounds.top < seg.bounds.top ||
				    ch.bounds.bottom > seg.bounds.bottom) {
					if (error) *error = Format("Image %d: character bounds outside segment", i);
					return false;
				}
			}
		}
	}

	return true;
}

void OCRDataset::Serialize(Stream& s) {
	int version = 1;
	s % version;

	s % images % image_paths % segments;
}

void OCRDataset::CheckIndex(int idx) const {
	if (idx < 0 || idx >= images.GetCount()) {
		throw Exc(Format("OCRDataset: index %d out of range [0, %d)", idx, images.GetCount()));
	}
}

// Geometry utilities

Rect UnionRect(const Rect& a, const Rect& b) {
	if (a.IsEmpty()) return b;
	if (b.IsEmpty()) return a;

	return Rect(
		min(a.left, b.left),
		min(a.top, b.top),
		max(a.right, b.right),
		max(a.bottom, b.bottom)
	);
}

Rect IntersectRect(const Rect& a, const Rect& b) {
	Rect r(
		max(a.left, b.left),
		max(a.top, b.top),
		min(a.right, b.right),
		min(a.bottom, b.bottom)
	);

	if (r.left >= r.right || r.top >= r.bottom) {
		return Rect(0, 0, 0, 0);
	}

	return r;
}

bool RectsOverlap(const Rect& a, const Rect& b) {
	return !IntersectRect(a, b).IsEmpty();
}

double RectIoU(const Rect& a, const Rect& b) {
	Rect intersection = IntersectRect(a, b);
	if (intersection.IsEmpty()) {
		return 0.0;
	}

	int intersection_area = intersection.Width() * intersection.Height();
	int a_area = a.Width() * a.Height();
	int b_area = b.Width() * b.Height();
	int union_area = a_area + b_area - intersection_area;

	if (union_area == 0) return 0.0;

	return (double)intersection_area / union_area;
}

Point RectCenter(const Rect& r) {
	return Point(
		(r.left + r.right) / 2,
		(r.top + r.bottom) / 2
	);
}

Rect ExpandRect(const Rect& r, int margin) {
	return Rect(
		r.left - margin,
		r.top - margin,
		r.right + margin,
		r.bottom + margin
	);
}

Rect ExpandRect(const Rect& r, int margin_x, int margin_y) {
	return Rect(
		r.left - margin_x,
		r.top - margin_y,
		r.right + margin_x,
		r.bottom + margin_y
	);
}

// Image utilities

bool IsBlackWhite(const Image& img) {
	if (img.IsEmpty()) return false;

	Size sz = img.GetSize();
	for (int y = 0; y < sz.cy; y++) {
		const RGBA* line = img[y];
		for (int x = 0; x < sz.cx; x++) {
			const RGBA& px = line[x];
			if (px.r != 0 && px.r != 255) return false;
			if (px.r != px.g || px.g != px.b) return false;
		}
	}

	return true;
}

bool IsGrayscale(const Image& img) {
	if (img.IsEmpty()) return false;

	Size sz = img.GetSize();
	for (int y = 0; y < sz.cy; y++) {
		const RGBA* line = img[y];
		for (int x = 0; x < sz.cx; x++) {
			const RGBA& px = line[x];
			if (px.r != px.g || px.g != px.b) return false;
		}
	}

	return true;
}

Image CropImage(const Image& src, const Rect& bounds) {
	if (src.IsEmpty() || bounds.IsEmpty()) {
		return Image();
	}

	Size src_sz = src.GetSize();

	// Clip bounds to image
	Rect clipped(
		max(0, bounds.left),
		max(0, bounds.top),
		min(src_sz.cx, bounds.right),
		min(src_sz.cy, bounds.bottom)
	);

	if (clipped.IsEmpty()) {
		return Image();
	}

	int w = clipped.Width();
	int h = clipped.Height();

	ImageBuffer ib(w, h);

	for (int y = 0; y < h; y++) {
		const RGBA* src_line = src[clipped.top + y] + clipped.left;
		RGBA* dst_line = ib[y];
		memcpy(dst_line, src_line, w * sizeof(RGBA));
	}

	return ib;
}

Image PadImage(const Image& src, int padding) {
	return PadImage(src, padding, padding);
}

Image PadImage(const Image& src, int pad_x, int pad_y) {
	if (src.IsEmpty()) return Image();

	Size src_sz = src.GetSize();
	Size dst_sz(src_sz.cx + pad_x * 2, src_sz.cy + pad_y * 2);

	ImageBuffer ib(dst_sz);
	Fill(~ib, White(), dst_sz.cx * dst_sz.cy);

	for (int y = 0; y < src_sz.cy; y++) {
		const RGBA* src_line = src[y];
		RGBA* dst_line = ib[y + pad_y] + pad_x;
		memcpy(dst_line, src_line, src_sz.cx * sizeof(RGBA));
	}

	return ib;
}

Size GetFitSize(Size src, Size target) {
	if (src.cx == 0 || src.cy == 0) return Size(0, 0);

	double scale = min(
		(double)target.cx / src.cx,
		(double)target.cy / src.cy
	);

	return Size(
		(int)(src.cx * scale),
		(int)(src.cy * scale)
	);
}

// Conversion utilities

int CharToClass(char c) {
	if (c >= '0' && c <= '9') return c - '0';           // 0-9
	if (c >= 'A' && c <= 'Z') return 10 + (c - 'A');    // 10-35
	if (c >= 'a' && c <= 'z') return 36 + (c - 'a');    // 36-61
	return -1;  // Unknown
}

char ClassToChar(int cls) {
	if (cls >= 0 && cls <= 9) return '0' + cls;
	if (cls >= 10 && cls <= 35) return 'A' + (cls - 10);
	if (cls >= 36 && cls <= 61) return 'a' + (cls - 36);
	return '?';
}

String GetClassName(int cls) {
	char c = ClassToChar(cls);
	if (c >= '0' && c <= '9') return Format("Digit '%c'", c);
	if (c >= 'A' && c <= 'Z') return Format("Uppercase '%c'", c);
	if (c >= 'a' && c <= 'z') return Format("Lowercase '%c'", c);
	return "Unknown";
}

} // namespace OCR

} // namespace Upp
