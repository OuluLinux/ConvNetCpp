#include "TableOverviewCtrl.h"

#include <algorithm>

NAMESPACE_UPP

namespace {

const int kPackMargin = 4;
const int kPackGap = 6;
const int kLabelH = 16;
const int kMaxScale = 8;

Rect NormalizeAndClamp(const Rect& r, Size bounds)
{
	Rect rr = r;
	if(rr.left > rr.right)
		Swap(rr.left, rr.right);
	if(rr.top > rr.bottom)
		Swap(rr.top, rr.bottom);
	rr &= Rect(bounds);
	return rr;
}

bool TryPackShelf(const Vector<Size>& sizes, Size area, Vector<Point>& out_positions)
{
	int n = sizes.GetCount();
	out_positions.SetCount(n);
	if(n == 0)
		return true;
	if(area.cx <= 0 || area.cy <= 0)
		return false;

	Vector<int> order;
	order.SetCount(n);
	for(int i = 0; i < n; i++)
		order[i] = i;

	std::sort(order.Begin(), order.End(), [&](int a, int b) {
		if(sizes[a].cy != sizes[b].cy)
			return sizes[a].cy > sizes[b].cy;
		return sizes[a].cx > sizes[b].cx;
	});

	int x = 0;
	int y = 0;
	int row_h = 0;
	for(int idx : order) {
		const Size& item = sizes[idx];
		if(item.cx <= 0 || item.cy <= 0)
			return false;
		if(item.cx > area.cx || item.cy > area.cy)
			return false;

		if(x > 0 && x + item.cx > area.cx) {
			y += row_h;
			x = 0;
			row_h = 0;
		}
		if(y + item.cy > area.cy)
			return false;

		out_positions[idx] = Point(x, y);
		x += item.cx + kPackGap;
		row_h = max(row_h, item.cy + kPackGap);
	}
	return true;
}

}

void TableOverviewCtrl::SetFrame(const Image& frame, const Vector<Rect>& table_rects, const Vector<String>& table_names)
{
	frame_ = frame;
	rects_ <<= table_rects;
	names_ <<= table_names;
	Repack(GetSize());
	Refresh();
}

void TableOverviewCtrl::Layout()
{
	Repack(GetSize());
}

void TableOverviewCtrl::Repack(Size widget_sz)
{
	packed_.Clear();
	pack_scale_ = 0;

	if(frame_.IsEmpty() || rects_.IsEmpty())
		return;

	Size area(max(0, widget_sz.cx - 2 * kPackMargin), max(0, widget_sz.cy - 2 * kPackMargin));
	if(area.cx <= 0 || area.cy <= 0)
		return;

	for(int s = kMaxScale; s >= 1; --s) {
		Vector<Size> sizes;
		sizes.Reserve(rects_.GetCount());
		bool invalid = false;
		for(const Rect& r : rects_) {
			Rect clamped = NormalizeAndClamp(r, frame_.GetSize());
			if(clamped.GetWidth() <= 0 || clamped.GetHeight() <= 0) {
				invalid = true;
				break;
			}
			int sw = clamped.GetWidth() * s;
			int sh = clamped.GetHeight() * s + kLabelH;
			sizes.Add(Size(sw, sh));
		}
		if(invalid)
			continue;

		Vector<Point> positions;
		if(TryPackShelf(sizes, area, positions)) {
			pack_scale_ = s;
			for(int i = 0; i < rects_.GetCount(); i++) {
				PackedItem& item = packed_.Add();
				item.rect_idx = i;
				item.pos = positions[i] + Point(kPackMargin, kPackMargin);
			}
			return;
		}
	}

	// Fallback: scale 1, simple row layout.
	pack_scale_ = 1;
	int x = kPackMargin;
	int y = kPackMargin;
	int row_h = 0;
	for(int i = 0; i < rects_.GetCount(); i++) {
		Rect clamped = NormalizeAndClamp(rects_[i], frame_.GetSize());
		int w = max(1, clamped.GetWidth());
		int h = max(1, clamped.GetHeight()) + kLabelH;
		if(x > kPackMargin && x + w > widget_sz.cx - kPackMargin) {
			x = kPackMargin;
			y += row_h + kPackGap;
			row_h = 0;
		}
		PackedItem& item = packed_.Add();
		item.rect_idx = i;
		item.pos = Point(x, y);
		x += w + kPackGap;
		row_h = max(row_h, h);
	}
}

void TableOverviewCtrl::Paint(Draw& w)
{
	Size sz = GetSize();
	w.DrawRect(sz, SColorFace());

	if(frame_.IsEmpty() || rects_.IsEmpty() || pack_scale_ <= 0) {
		w.DrawText(4, 4, "No tables defined", StdFont(), SColorDisabled());
		return;
	}

	for(const PackedItem& item : packed_) {
		if(item.rect_idx < 0 || item.rect_idx >= rects_.GetCount())
			continue;

		Rect src = NormalizeAndClamp(rects_[item.rect_idx], frame_.GetSize());
		if(src.GetWidth() <= 0 || src.GetHeight() <= 0)
			continue;

		int sw = src.GetWidth() * pack_scale_;
		int sh = src.GetHeight() * pack_scale_;
		Rect dst(item.pos, Size(sw, sh));

		Image crop = Crop(frame_, src);
		if(!crop.IsEmpty()) {
			Image scaled = pack_scale_ == 1 ? crop : Rescale(crop, sw, sh);
			w.DrawImage(dst.left, dst.top, scaled);
		}

		w.DrawRect(dst.left, dst.top, sw, 1, SBlack());
		w.DrawRect(dst.left, dst.top, 1, sh, SBlack());
		w.DrawRect(dst.left, dst.bottom - 1, sw, 1, SBlack());
		w.DrawRect(dst.right - 1, dst.top, 1, sh, SBlack());

		String name = item.rect_idx < names_.GetCount() && !names_[item.rect_idx].IsEmpty()
		                ? names_[item.rect_idx]
		                : Format("Table %d", item.rect_idx + 1);
		w.DrawText(dst.left + 2, dst.bottom + 1, name, StdFont(), SBlack());
	}
}

END_UPP_NAMESPACE
