#include "ScreenCaptureSource.h"

NAMESPACE_UPP

bool ScreenCaptureSource::ParseRect(const String& param, Rect& out) const
{
	Vector<String> parts = Split(param, ',');
	if(parts.GetCount() != 4)
		return false;

	int x = ScanInt(TrimBoth(parts[0]));
	int y = ScanInt(TrimBoth(parts[1]));
	int w = ScanInt(TrimBoth(parts[2]));
	int h = ScanInt(TrimBoth(parts[3]));
	if(w <= 0 || h <= 0)
		return false;

	out = RectC(x, y, w, h);
	return true;
}

bool ScreenCaptureSource::Open(const String& param)
{
	Rect rect;
	String mode = TrimBoth(param);
	if(mode.IsEmpty() || ToLower(mode) == "full") {
		rect = Ctrl::GetPrimaryScreenArea();
		if(rect.IsEmpty())
			rect = Ctrl::GetVirtualScreenArea();
	}
	else if(!ParseRect(mode, rect)) {
		return false;
	}

	if(rect.IsEmpty())
		return false;

	capture_rect = rect;
	open = true;
	return true;
}

void ScreenCaptureSource::Close()
{
	open = false;
	capture_rect = Rect();
}

Image ScreenCaptureSource::GetFrame()
{
	if(!open)
		return Null;

	Rect screen = Ctrl::GetVirtualScreenArea();
	Rect r = capture_rect & screen;
	if(r.IsEmpty())
		return Null;

	ImageBuffer ib(r.GetSize());
	RGBA* dst = ib.Begin();
	for(int y = r.top; y < r.bottom; y++) {
		for(int x = r.left; x < r.right; x++) {
			Color c = GuiPlatformGetScreenPixel(x, y);
			dst->r = c.GetR();
			dst->g = c.GetG();
			dst->b = c.GetB();
			dst->a = 255;
			dst++;
		}
	}
	ib.SetKind(IMAGE_OPAQUE);
	return Image(ib);
}

END_UPP_NAMESPACE
