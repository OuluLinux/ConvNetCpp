#include "FrameDiff.h"

NAMESPACE_UPP

namespace {

bool IsBlockChanged(const Image& prev, const Image& curr, const Rect& block, int threshold)
{
	for(int y = block.top; y < block.bottom; y++) {
		const RGBA* prow = prev[y];
		const RGBA* crow = curr[y];
		for(int x = block.left; x < block.right; x++) {
			const RGBA& a = prow[x];
			const RGBA& b = crow[x];
			if(abs((int)a.r - (int)b.r) > threshold
			|| abs((int)a.g - (int)b.g) > threshold
			|| abs((int)a.b - (int)b.b) > threshold
			|| abs((int)a.a - (int)b.a) > threshold)
				return true;
		}
	}
	return false;
}

Vector<Rect> ComputeChangedBlockList(const Image& prev, const Image& curr, int threshold, int block_size)
{
	Vector<Rect> changed;
	if(IsNull(prev) || IsNull(curr))
		return changed;
	if(prev.GetSize() != curr.GetSize())
		return changed;

	threshold = max(0, threshold);
	block_size = max(1, block_size);

	int w = curr.GetWidth();
	int h = curr.GetHeight();
	for(int top = 0; top < h; top += block_size) {
		for(int left = 0; left < w; left += block_size) {
			Rect block(left, top,
			           min(w, left + block_size),
			           min(h, top + block_size));
			if(IsBlockChanged(prev, curr, block, threshold))
				changed.Add(block);
		}
	}
	return changed;
}

}

Image ComputeChangedAreaMask(const Image& prev, const Image& curr, int threshold, int block_size)
{
	if(IsNull(curr))
		return Null;

	ImageBuffer out(curr.GetSize());
	for(int y = 0; y < out.GetHeight(); y++) {
		RGBA* row = out[y];
		for(int x = 0; x < out.GetWidth(); x++) {
			row[x].r = row[x].g = row[x].b = 0;
			row[x].a = 255;
		}
	}
	out.SetKind(IMAGE_OPAQUE);

	Vector<Rect> changed = ComputeChangedBlockList(prev, curr, threshold, block_size);
	for(int i = 0; i < changed.GetCount(); i++) {
		const Rect& r = changed[i];
		for(int y = r.top; y < r.bottom; y++) {
			RGBA* row = out[y];
			for(int x = r.left; x < r.right; x++) {
				row[x].r = row[x].g = row[x].b = 255;
				row[x].a = 255;
			}
		}
	}
	return out;
}

Vector<Rect> ComputeChangedBlocks(const Image& prev, const Image& curr, int threshold, int block_size)
{
	return ComputeChangedBlockList(prev, curr, threshold, block_size);
}

END_UPP_NAMESPACE
