#ifndef _OCR_OCRTypes_h_
#define _OCR_OCRTypes_h_

#include <Core/Core.h>
#include <Draw/Draw.h>

namespace Upp {

namespace OCR {

// Forward declarations
struct OCRChar;
struct OCRSegment;
class OCRDataset;

// Individual character representation
struct OCRChar : Moveable<OCRChar> {
	Rect bounds;          // Bounding box in image
	int label;            // Character class (ASCII code or class index)
	String text;          // Text label
	double confidence;    // Classification confidence [0.0, 1.0]
	Image img;            // Cropped character image (optional)

	OCRChar() : label(0), confidence(0.0) {}

	void Serialize(Stream& s) {
		s % bounds % label % text % confidence % img;
	}

	String ToString() const {
		return Format("OCRChar(label=%d, conf=%.2f, bounds=%s)",
		              label, confidence, AsString(bounds));
	}

	// Provide deep copy for U++
	void DeepCopyFrom(const OCRChar& src) {
		bounds = src.bounds;
		label = src.label;
		confidence = src.confidence;
		img = src.img;
	}
};

// Text line segmentation
struct OCRSegment : Moveable<OCRSegment> {
	Rect bounds;              // Line bounding box in image
	Vector<OCRChar> chars;    // Character bounding boxes and labels
	String text;              // Ground truth or recognized text
	double avg_confidence;    // Average confidence across characters

	OCRSegment() : avg_confidence(0.0) {}

	void Serialize(Stream& s) {
		s % bounds % chars % text % avg_confidence;
	}

	void UpdateText() {
		text.Clear();
		for (const OCRChar& ch : chars) {
			if (ch.label > 0 && ch.label < 128) {
				text.Cat(ch.label);
			}
		}
	}

	// Provide deep copy for U++
	void DeepCopyFrom(const OCRSegment& src) {
		bounds = src.bounds;
		text = src.text;
		avg_confidence = src.avg_confidence;
		chars.Clear();
		for (const OCRChar& ch : src.chars) {
			chars.Add().DeepCopyFrom(ch);
		}
	}

	void UpdateConfidence() {
		if (chars.IsEmpty()) {
			avg_confidence = 0.0;
			return;
		}
		double sum = 0.0;
		for (const OCRChar& ch : chars) {
			sum += ch.confidence;
		}
		avg_confidence = sum / chars.GetCount();
	}

	String ToString() const {
		return Format("OCRSegment(text='%s', chars=%d, bounds=%s)",
		              text, chars.GetCount(), AsString(bounds));
	}
};

// Dataset for training/testing
class OCRDataset {
public:
	OCRDataset();
	~OCRDataset();

	// Dataset management
	void Clear();
	bool IsEmpty() const { return images.IsEmpty(); }
	int GetCount() const { return images.GetCount(); }

	// Add data
	void AddImage(const Image& img, const String& path = String());
	void AddSegment(int image_idx, const OCRSegment& seg);
	void SetSegments(int image_idx, const Vector<OCRSegment>& segs);

	// Access data
	const Image& GetImage(int idx) const;
	Image& GetImage(int idx);
	const String& GetImagePath(int idx) const;

	const Vector<OCRSegment>& GetSegments(int idx) const;
	Vector<OCRSegment>& GetSegments(int idx);

	// Statistics
	int GetTotalCharacters() const;
	int GetTotalSegments() const;
	void GetClassDistribution(Vector<int>& counts) const;

	// Validation
	bool Validate(String* error = nullptr) const;

	// Serialization
	void Serialize(Stream& s);

	void RemoveImage(int idx);

	String ToString() const {
		return Format("OCRDataset(images=%d, total_chars=%d)",
		              GetCount(), GetTotalCharacters());
	}

protected:
	Vector<Image> images;
	Vector<String> image_paths;
	Vector<Vector<OCRSegment>> segments;

	void CheckIndex(int idx) const;
};

// Geometry utilities
Rect UnionRect(const Rect& a, const Rect& b);
Rect IntersectRect(const Rect& a, const Rect& b);
bool RectsOverlap(const Rect& a, const Rect& b);
double RectIoU(const Rect& a, const Rect& b);  // Intersection over Union
Point RectCenter(const Rect& r);
Rect ExpandRect(const Rect& r, int margin);
Rect ExpandRect(const Rect& r, int margin_x, int margin_y);

// Image utilities
bool IsBlackWhite(const Image& img);
bool IsGrayscale(const Image& img);
Image CropImage(const Image& src, const Rect& bounds);
Image PadImage(const Image& src, int padding);
Image PadImage(const Image& src, int pad_x, int pad_y);
Size GetFitSize(Size src, Size target);

// Conversion utilities
int CharToClass(char c);
char ClassToChar(int cls);
String GetClassName(int cls);

} // namespace OCR

} // namespace Upp

#endif
