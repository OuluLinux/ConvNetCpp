#include "AnnotationEditorCommon.h"

NAMESPACE_UPP

template<> void Jsonize(JsonIO& jio, Pointf& p) {
	jio("x", p.x)("y", p.y);
}

bool IsPointInPolygon(Pointf pt, const Vector<Pointf>& poly) {
	if(poly.GetCount() < 3) return false;
	bool inside = false;
	for(int i = 0, j = poly.GetCount() - 1; i < poly.GetCount(); j = i++) {
		if(((poly[i].y > pt.y) != (poly[j].y > pt.y)) &&
		   (pt.x < (poly[j].x - poly[i].x) * (pt.y - poly[i].y) / (poly[j].y - poly[i].y) + poly[i].x)) {
			inside = !inside;
		}
	}
	return inside;
}

bool IsRectangle(const Vector<Pointf>& poly, double tolerance) {
	if(poly.GetCount() != 4) return false;
	bool x_match = (abs(poly[0].x - poly[3].x) < tolerance && abs(poly[1].x - poly[2].x) < tolerance) ||
	               (abs(poly[0].x - poly[1].x) < tolerance && abs(poly[2].x - poly[3].x) < tolerance);
	bool y_match = (abs(poly[0].y - poly[1].y) < tolerance && abs(poly[2].y - poly[3].y) < tolerance) ||
	               (abs(poly[0].y - poly[3].y) < tolerance && abs(poly[1].y - poly[2].y) < tolerance);
	return x_match && y_match;
}

String MakeIsoTimestamp() {
	Time t = GetSysTime();
	return Format("%04d-%02d-%02d", t.year, t.month, t.day) + "T" +
	       Format("%02d:%02d:%02d", t.hour, t.minute, t.second);
}

static bool IsPngSignature(const byte* sig, int n) {
	static const byte kPngSig[8] = {137, 80, 78, 71, 13, 10, 26, 10};
	if(n < 8) return false;
	for(int i = 0; i < 8; i++)
		if(sig[i] != kPngSig[i])
			return false;
	return true;
}

static bool IsJpegSignature(const byte* sig, int n) {
	return n >= 3 && sig[0] == 0xFF && sig[1] == 0xD8 && sig[2] == 0xFF;
}

Image LoadImageByExtension(const String& path) {
	if(path.IsEmpty() || !FileExists(path))
		return Image();

	byte sig[8] = {0};
	int sig_n = 0;
	{
		FileIn in(path);
		if(in.IsOpen())
			sig_n = (int)in.Get(sig, sizeof(sig));
	}

	bool sig_png = IsPngSignature(sig, sig_n);
	bool sig_jpg = IsJpegSignature(sig, sig_n);
	String ext = ToLower(GetFileExt(path));

	if(ext == ".png") {
		if(sig_png) {
			Image img = PNGRaster().LoadFile(path);
			if(!img.IsEmpty()) return img;
		}
		if(sig_jpg) {
			Image img = JPGRaster().LoadFile(path);
			if(!img.IsEmpty()) return img;
		}
		if(!sig_png && !sig_jpg)
			return StreamRaster::LoadFileAny(path);
		return Image();
	}

	if(ext == ".jpg" || ext == ".jpeg") {
		if(sig_jpg) {
			Image img = JPGRaster().LoadFile(path);
			if(!img.IsEmpty()) return img;
		}
		if(sig_png) {
			Image img = PNGRaster().LoadFile(path);
			if(!img.IsEmpty()) return img;
		}
		if(!sig_png && !sig_jpg)
			return StreamRaster::LoadFileAny(path);
		return Image();
	}

	return StreamRaster::LoadFileAny(path);
}

END_UPP_NAMESPACE
