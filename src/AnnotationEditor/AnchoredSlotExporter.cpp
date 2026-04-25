#include "AnchoredSlotExporter.h"
#include <AnnLayCore/AnchoredSlotClassifier.h>
#include "AnnotationEditorCommon.h"

#include <Draw/Draw.h>
#include <plugin/jpg/jpg.h>
#include <plugin/png/png.h>
#include <Painter/Painter.h>


NAMESPACE_UPP

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

namespace {

int ClampI(int v, int lo, int hi) {
	return v < lo ? lo : (v > hi ? hi : v);
}

String GetSubLayoutJsonOverride(const Value& ann) {
	if(IsNull(ann))
		return String();
	String j = ann["metadata"]["sub_layout_json"];
	if(j.IsEmpty())
		j = ann["metadata"]["annlay_sub_layout_json"];
	return TrimBoth(j);
}

Rect ResolveRegionRectWithOverride(const AnnLaySlot& slot,
                                   Size base_size,
                                   const String& region_name,
                                   const String& sub_layout_json_override) {
	if(sub_layout_json_override.IsEmpty())
		return AnnLayResolveRegionRect(slot, base_size, region_name);
	AnnLaySlot tmp;
	tmp.sub_layout_json = sub_layout_json_override;
	return AnnLayResolveRegionRect(tmp, base_size, region_name);
}

double ResolveSlotRotationWithOverride(const AnnLaySlot& slot,
                                       const String& slot_id,
                                       const String& sub_layout_json_override) {
	if(!sub_layout_json_override.IsEmpty()) {
		AnnLaySlot tmp;
		tmp.sub_layout_json = sub_layout_json_override;
		double rot = 0;
		if(AnnLayTryGetSubLayoutRotationDeg(tmp, rot))
			return rot;
	}
	return AnnLayGetSlotRotationDeg(slot, slot_id);
}

RGBA BilerpRGBA(const Image& src, double sx, double sy) {
	int w = src.GetWidth();
	int h = src.GetHeight();
	if(w <= 1 || h <= 1) {
		RGBA z = {0, 0, 0, 255};
		return z;
	}
	if(sx < 0 || sy < 0 || sx >= w - 1 || sy >= h - 1) {
		RGBA z = {0, 0, 0, 255};
		return z;
	}
	int ix = (int)sx;
	int iy = (int)sy;
	double fx = sx - ix;
	double fy = sy - iy;

	RGBA a = src[iy][ix];
	RGBA b = src[iy][ix + 1];
	RGBA c = src[iy + 1][ix];
	RGBA d = src[iy + 1][ix + 1];

	double ar = a.r * (1 - fx) + b.r * fx;
	double ag = a.g * (1 - fx) + b.g * fx;
	double ab = a.b * (1 - fx) + b.b * fx;

	double br = c.r * (1 - fx) + d.r * fx;
	double bg = c.g * (1 - fx) + d.g * fx;
	double bb = c.b * (1 - fx) + d.b * fx;

	RGBA out;
	out.r = (byte)ClampI((int)((ar * (1 - fy) + br * fy) + 0.5), 0, 255);
	out.g = (byte)ClampI((int)((ag * (1 - fy) + bg * fy) + 0.5), 0, 255);
	out.b = (byte)ClampI((int)((ab * (1 - fy) + bb * fy) + 0.5), 0, 255);
	out.a = 255;
	return out;
}

}

Image AnchoredSlotExporter::CropAnchor(const Image& img, const AnnLayAnchor& anchor,
                                       double bbox_expand, Size crop_size) const {
	if(img.IsEmpty() || crop_size.cx <= 0 || crop_size.cy <= 0) return Image();
	Size isz = img.GetSize();

	double exp_w = anchor.w * (1.0 + 2.0 * bbox_expand);
	double exp_h = anchor.h * (1.0 + 2.0 * bbox_expand);

	int w = (int)round(exp_w * isz.cx);
	int h = (int)round(exp_h * isz.cy);
	if(w <= 0 || h <= 0) return Image();

	int x = (int)round(anchor.cx * isz.cx - w * 0.5);
	int y = (int)round(anchor.cy * isz.cy - h * 0.5);

	ImageDraw id(w, h);
	id.DrawRect(0, 0, w, h, Black());
	id.DrawImage(-x, -y, img);

	return Rescale(id, crop_size);
}


Image AnchoredSlotExporter::CropAndRotate(const Image& img, const AnnLayAnchor& anchor,
                                          double bbox_expand, Size crop_size, double angle) const {
	if(img.IsEmpty() || crop_size.cx <= 0 || crop_size.cy <= 0) return Image();
	if(fabs(angle) < 0.01) return CropAnchor(img, anchor, bbox_expand, crop_size);

	Size isz = img.GetSize();
	// Crop a significantly larger area to allow rotation without edge artifacts
	double exp = (1.0 + 2.0 * bbox_expand) * 1.5; 
	double exp_w = anchor.w * exp;
	double exp_h = anchor.h * exp;
	
	int pw = (int)round(exp_w * isz.cx);
	int ph = (int)round(exp_h * isz.cy);
	if (pw <= 0 || ph <= 0) return Image();

	int px = (int)round(anchor.cx * isz.cx - pw * 0.5);
	int py = (int)round(anchor.cy * isz.cy - ph * 0.5);

	ImageDraw id(pw, ph);
	id.DrawRect(0, 0, pw, ph, Black());
	id.DrawImage(-px, -py, img);
	
	Image patch = id;
	Image rotated = RotateBilinear(patch, -angle);
	
	// Now crop the final target area from the center of the rotated patch
	Size rsz = rotated.GetSize();
	double final_exp_w = anchor.w * (1.0 + 2.0 * bbox_expand) * isz.cx;
	double final_exp_h = anchor.h * (1.0 + 2.0 * bbox_expand) * isz.cy;
	
	int cx = rsz.cx / 2;
	int cy = rsz.cy / 2;
	int fx1 = cx - (int)(final_exp_w * 0.5);
	int fy1 = cy - (int)(final_exp_h * 0.5);
	int fx2 = fx1 + (int)final_exp_w;
	int fy2 = fy1 + (int)final_exp_h;
	
	Image sub = Crop(rotated, Rect(fx1, fy1, fx2, fy2));
	if(sub.IsEmpty()) return Image();
	return Rescale(sub, crop_size);
}

Image AnchoredSlotExporter::CropPolygon(const Image& img,
                                         const Value& polygon_points,
                                         int img_w, int img_h,
                                         double bbox_expand,
                                         Size crop_size) const {
	if(img.IsEmpty() || !IsValueArray(polygon_points)) return Image();
	double min_x = 1e18, min_y = 1e18, max_x = -1e18, max_y = -1e18;
	bool ok = false;
	for(int i = 0; i < polygon_points.GetCount(); i++) {
		Value p = polygon_points[i];
		Value xv = p["x"], yv = p["y"];
		if(IsNull(xv) || IsNull(yv)) continue;
		double x = xv, y = yv;
		min_x = min(min_x, x); max_x = max(max_x, x);
		min_y = min(min_y, y); max_y = max(max_y, y);
		ok = true;
	}
	if(!ok || max_x <= min_x || max_y <= min_y) return Image();

	double cx = (min_x + max_x) * 0.5;
	double cy = (min_y + max_y) * 0.5;
	double w  = (max_x - min_x) * (1.0 + 2.0 * bbox_expand);
	double h  = (max_y - min_y) * (1.0 + 2.0 * bbox_expand);
	int x1 = max(0, (int)(cx - w * 0.5));
	int y1 = max(0, (int)(cy - h * 0.5));
	int x2 = min(img_w, (int)(cx + w * 0.5));
	int y2 = min(img_h, (int)(cy + h * 0.5));
	if(x2 <= x1 || y2 <= y1) return Image();
	Image sub = Crop(img, Rect(x1, y1, x2, y2));
	if(sub.IsEmpty()) return Image();
	return Rescale(sub, crop_size);
}

Image AnchoredSlotExporter::CropPolygonAspect(const Image& img,
                                              const Value& polygon_points,
                                              int img_w, int img_h,
                                              double bbox_expand,
                                              Size crop_size,
                                              double angle) const {
	if(img.IsEmpty() || !IsValueArray(polygon_points)) return Image();
	double min_x = 1e18, min_y = 1e18, max_x = -1e18, max_y = -1e18;
	bool ok = false;
	for(int i = 0; i < polygon_points.GetCount(); i++) {
		Value p = polygon_points[i];
		Value xv = p["x"], yv = p["y"];
		if(IsNull(xv) || IsNull(yv)) continue;
		double x = xv, y = yv;
		min_x = min(min_x, x); max_x = max(max_x, x);
		min_y = min(min_y, y); max_y = max(max_y, y);
		ok = true;
	}
	if(!ok || max_x <= min_x || max_y <= min_y) return Image();

	double cx = (min_x + max_x) * 0.5;
	double cy = (min_y + max_y) * 0.5;
	double w  = (max_x - min_x) * (1.0 + 2.0 * bbox_expand);
	double h  = (max_y - min_y) * (1.0 + 2.0 * bbox_expand);

	if(crop_size.cx > 0 && crop_size.cy > 0) {
		double target_ratio = (double)crop_size.cy / crop_size.cx;
		double actual_ratio = h / w;
		if(actual_ratio < target_ratio)
			h = w * target_ratio;
		else
			w = h / target_ratio;
	}

	if(fabs(angle) < 0.01) {
		int x1 = max(0, (int)(cx - w * 0.5));
		int y1 = max(0, (int)(cy - h * 0.5));
		int x2 = min(img_w, (int)(cx + w * 0.5));
		int y2 = min(img_h, (int)(cy + h * 0.5));
		if(x2 <= x1 || y2 <= y1) return Image();
		Image sub = Crop(img, Rect(x1, y1, x2, y2));
		if(sub.IsEmpty()) return Image();
		return Rescale(sub, crop_size);
	}

	// For rotation, crop a larger area first
	double exp = 1.5; // expansion factor for rotation
	int x1 = max(0, (int)(cx - w * exp * 0.5));
	int y1 = max(0, (int)(cy - h * exp * 0.5));
	int x2 = min(img_w, (int)(cx + w * exp * 0.5));
	int y2 = min(img_h, (int)(cy + h * exp * 0.5));
	if(x2 <= x1 || y2 <= y1) return Image();
	
	Image patch = Crop(img, Rect(x1, y1, x2, y2));
	if(patch.IsEmpty()) return Image();
	
	Image rotated = RotateBilinear(patch, -angle);
	
	Size rsz = rotated.GetSize();
	int rcx = rsz.cx / 2;
	int rcy = rsz.cy / 2;
	int fx1 = rcx - (int)(w * 0.5);
	int fy1 = rcy - (int)(h * 0.5);
	int fx2 = fx1 + (int)w;
	int fy2 = fy1 + (int)h;
	
	Image sub = Crop(rotated, Rect(fx1, fy1, fx2, fy2));
	if(sub.IsEmpty()) return Image();
	return Rescale(sub, crop_size);
}

static Image ToGrayscale(const Image& src) {
	int w = src.GetWidth(), h = src.GetHeight();
	ImageBuffer dst(w, h);
	for(int y = 0; y < h; y++) {
		const RGBA* s = src[y];
		RGBA* d = dst[y];
		for(int x = 0; x < w; x++) {
			byte g = (byte)(s[x].r * 299 / 1000 + s[x].g * 587 / 1000 + s[x].b * 114 / 1000);
			d[x] = {g, g, g, 255};
		}
	}
	return dst;
}


bool AnchoredSlotExporter::SaveCrop(const Image& img, const String& path, bool grayscale, bool equalize) const {
	if(img.IsEmpty()) return false;
	Image out = equalize ? AnchoredSlotClassifier::LinearContrastStretching(img) : img;
	String dir = GetFileDirectory(path);
	if(!DirectoryExists(dir))
		RealizeDirectory(dir);
	return PNGEncoder().SaveFile(path, grayscale ? ToGrayscale(out) : out);
}


END_UPP_NAMESPACE
