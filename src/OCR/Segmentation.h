#ifndef _OCR_Segmentation_h_
#define _OCR_Segmentation_h_

#include "OCRTypes.h"
#include "ImageUtils.h"

namespace Upp {

namespace OCR {

// Character segmentation from text line
Vector<Rect> SegmentCharacters(const Image& line_img, int min_width = 2, int min_height = 5);

// Line segmentation from page
Vector<Rect> SegmentLines(const Image& page_img, int min_height = 5);

// Find character boundaries using projection
Vector<int> FindCharacterBoundaries(const Image& line_img, int min_width = 2);

// Find line boundaries using projection
Vector<int> FindLineBoundaries(const Image& page_img, int min_gap = 3);

// Refine boundaries to tight bounding boxes
Rect FindTightBounds(const Image& img, const Rect& initial_bounds);

// Merge nearby rectangles (for fixing over-segmentation)
Vector<Rect> MergeNearbyRects(const Vector<Rect>& rects, int max_gap = 5);

// Split wide rectangles (for fixing under-segmentation)
Vector<Rect> SplitWideRects(const Vector<Rect>& rects, const Image& img, double max_aspect_ratio = 3.0);

// Advanced segmentation using connected components
Vector<Rect> SegmentCharactersConnectedComponents(const Image& line_img, int min_area = 10);

// Watershed-based segmentation for touching characters
Vector<Rect> SegmentTouchingCharacters(const Image& line_img);

} // namespace OCR

} // namespace Upp

#endif
