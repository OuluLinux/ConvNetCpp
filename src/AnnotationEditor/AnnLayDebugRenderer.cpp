#include "AnnLayDebugRenderer.h"

#include <CtrlLib/CtrlLib.h> // ImageDraw requires this

NAMESPACE_UPP

namespace {

Color MethodColor(const String& method) {
	if(method == "ocr_text")
		return Color(0, 200, 0);
	if(method == "classifier_bool")
		return Color(0, 100, 255);
	if(method == "classifier_label")
		return Color(0, 200, 200);
	if(method == "orb_match")
		return Color(220, 200, 0);
	return Color(120, 120, 120);
}

Rect AnchorToRect(const AnnLayAnchor& anchor, const AnnLay& lay, Size img_size) {
	double exp = lay.bbox_expand;
	double exp_w = anchor.w * (1.0 + 2.0 * exp);
	double exp_h = anchor.h * (1.0 + 2.0 * exp);
	int x1 = max(0, (int)((anchor.cx - exp_w * 0.5) * img_size.cx));
	int y1 = max(0, (int)((anchor.cy - exp_h * 0.5) * img_size.cy));
	int x2 = min(img_size.cx, (int)((anchor.cx + exp_w * 0.5) * img_size.cx));
	int y2 = min(img_size.cy, (int)((anchor.cy + exp_h * 0.5) * img_size.cy));
	if(x2 <= x1 || y2 <= y1)
		return Rect();
	return Rect(x1, y1, x2, y2);
}

void DrawRectOutline(Draw& w, Rect r, Color c) {
	if(r.IsEmpty() || r.right <= r.left || r.bottom <= r.top)
		return;
	w.DrawRect(r.left, r.top, r.GetWidth(), 1, c);
	w.DrawRect(r.left, r.bottom - 1, r.GetWidth(), 1, c);
	w.DrawRect(r.left, r.top, 1, r.GetHeight(), c);
	w.DrawRect(r.right - 1, r.top, 1, r.GetHeight(), c);

	Rect r2(r.left + 1, r.top + 1, r.right - 1, r.bottom - 1);
	if(!r2.IsEmpty() && r2.right > r2.left && r2.bottom > r2.top) {
		w.DrawRect(r2.left, r2.top, r2.GetWidth(), 1, c);
		w.DrawRect(r2.left, r2.bottom - 1, r2.GetWidth(), 1, c);
		w.DrawRect(r2.left, r2.top, 1, r2.GetHeight(), c);
		w.DrawRect(r2.right - 1, r2.top, 1, r2.GetHeight(), c);
	}
}

String BuildLabel(const SlotResult& r, const AnnLay& lay) {
	if(r.method == "ocr_text") {
		return r.slot_id + ": " + r.raw_text;
	}
	if(r.method == "classifier_bool") {
		String b = r.class_index == 1 ? "true" : "false";
		String lbl = Format("%s: %s %d%%", r.slot_id, b, (int)floor(r.confidence * 100.0 + 0.5));
		if(r.confidence < 0.5)
			lbl = "! " + lbl;
		return lbl;
	}
	if(r.method == "classifier_label") {
		String cls = "-";
		const AnnLaySlot* slot = lay.FindSlot(r.slot_id);
		if(slot && r.class_index >= 0 && r.class_index < slot->classes.GetCount())
			cls = slot->classes[r.class_index];
		String lbl = Format("%s: %s %d%%", r.slot_id, cls, (int)floor(r.confidence * 100.0 + 0.5));
		if(r.confidence < 0.5)
			lbl = "! " + lbl;
		return lbl;
	}
	if(r.method == "orb_match") {
		String core = r.class_index >= 0
			? Format("%s: cand=%d", r.slot_id, r.class_index)
			: (r.slot_id + ": -");
		if(r.confidence < 0.5)
			core = "! " + core;
		return core;
	}
	return r.slot_id;
}

void DrawSlotLabel(Draw& w, Rect bbox, const String& label, Size img_size) {
	Font fnt = StdFont();
	Size tsz = GetTextSize(label, fnt);
	int tx = max(0, min(bbox.left, img_size.cx - (tsz.cx + 4)));
	int ty = bbox.top - tsz.cy - 2;
	if(ty < 0)
		ty = min(max(0, bbox.top + 2), max(0, img_size.cy - (tsz.cy + 2)));

	int bw = min(tsz.cx + 4, img_size.cx - tx);
	int bh = min(tsz.cy + 2, img_size.cy - ty);
	if(bw <= 0 || bh <= 0)
		return;

	w.DrawRect(tx, ty, bw, bh, Color(30, 30, 30));
	w.DrawText(tx + 2, ty + 1, label, fnt, Color(255, 255, 255));
}

}

Image AnnLayDebugRenderer::Render(const Image& src, const Vector<SlotResult>& results,
                                  const AnnLay& lay) const {
	if(src.IsEmpty())
		return Image();

	ImageDraw iw(src.GetWidth(), src.GetHeight());
	iw.DrawImage(0, 0, src);
	Size img_sz = src.GetSize();

	for(int i = 0; i < results.GetCount(); i++) {
		const SlotResult& r = results[i];
		if(r.method == "ignored")
			continue;

		Color c = MethodColor(r.method);
		if(r.method != "ocr_text" && r.confidence < 0.5)
			c = Color(220, 0, 0);

		Rect bbox = r.pixel_bbox;
		if(bbox.IsEmpty()) {
			const AnnLaySlot* slot = lay.FindSlot(r.slot_id);
			if(slot && !slot->anchor.IsEmpty()) {
				bbox = AnchorToRect(slot->anchor, lay, img_sz);
				c = Color(120, 120, 120); // anchor fallback visualization
			}
		}
		if(bbox.IsEmpty())
			continue;

		DrawRectOutline(iw, bbox, c);
		DrawSlotLabel(iw, bbox, BuildLabel(r, lay), img_sz);
	}

	return iw;
}

bool AnnLayDebugRenderer::RenderToFile(const Image& src, const Vector<SlotResult>& results,
                                       const AnnLay& lay, const String& out_path) const {
	Image out = Render(src, results, lay);
	if(out.IsEmpty())
		return false;
	return PNGEncoder().SaveFile(out_path, out);
}

END_UPP_NAMESPACE
