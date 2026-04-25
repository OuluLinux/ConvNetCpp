#include "FrameRecognizer.h"
#include <AnnLayCore/AnchoredSlotClassifier.h>
#include <plugin/jpg/jpg.h>
#include <plugin/png/png.h>
#include <OCR/Preprocessing.h>
#include <cstdio>
#include <cstdlib>

NAMESPACE_UPP

static String SafeFormatInt(int v) {
	if(IsNull(v)) return "N/A";
	return Format("%+d", v);
}

bool FrameRecVerbose() {
	static bool verbose = [] {
		const char* e = getenv("FR_VERBOSE");
		if(!e || !*e)
			return false;
		return atoi(e) != 0;
	}();
	return verbose;
}

void FrameRecTrace(const String& msg) {
	if(!FrameRecVerbose())
		return;
	fprintf(stdout, "[FrameRecognizer] %s\n", ~msg);
	fflush(stdout);
}

String ResolveFromSlnDir(const String& sln_dir, const String& path) {
	String p = TrimBoth(path);
	if(p.IsEmpty())
		return String();
	if(IsFullPath(p))
		return NormalizePath(p);
	return NormalizePath(AppendFileName(sln_dir, p));
}

String ResultDisplayValue(const SlotResult& r) {
	if(!r.raw_text.IsEmpty())
		return r.raw_text;
	if(!r.top_class.IsEmpty())
		return r.top_class;
	if(r.class_index >= 0)
		return AsString(r.class_index);
	return String();
}

Color SlotColor(const String& slot_id);

void DrawRectOutline(ImageDraw& w, const Rect& r, Color c) {
	if(r.IsEmpty())
		return;
	w.DrawRect(r.left, r.top, r.GetWidth(), 1, c);
	w.DrawRect(r.left, r.bottom - 1, r.GetWidth(), 1, c);
	w.DrawRect(r.left, r.top, 1, r.GetHeight(), c);
	w.DrawRect(r.right - 1, r.top, 1, r.GetHeight(), c);
}

static Image HighLuminanceThresholdBinarizationPreview(const Image& src) {
	if(src.IsEmpty())
		return src;
	int w = src.GetWidth(), h = src.GetHeight();
	byte thr = (byte)minmax((int)round(ANNLAY_HIGH_LUMINANCE_THRESHOLD * 255.0), 0, 255);
	ImageBuffer out(w, h);
	for(int y = 0; y < h; y++) {
		const RGBA* s = src[y];
		RGBA* d = out[y];
		for(int x = 0; x < w; x++) {
			int lum = ((int)s[x].r * 299 + (int)s[x].g * 587 + (int)s[x].b * 114) / 1000;
			byte v = lum >= thr ? 255 : 0;
			d[x] = {v, v, v, 255};
		}
	}
	return out;
}

String ResolveImagePath(const String& sln_dir, const String& images_dir, const Value& img_rec) {
	String base_dir = ResolveFromSlnDir(sln_dir, images_dir);
	String fname = img_rec["file_name"];
	if(fname.IsEmpty())
		fname = img_rec["file_path"];

	if(IsFullPath(fname))
		return NormalizePath(fname);

	String p = NormalizePath(AppendFileName(base_dir, GetFileName(fname)));
	if(FileExists(p))
		return p;
	return NormalizePath(AppendFileName(sln_dir, fname));
}

Image RenderOverlayImage(const Image& src, const Vector<SlotResult>& results, bool show_offsets) {
	if(src.IsEmpty())
		return src;

	ImageDraw id(src.GetSize());
	id.DrawImage(0, 0, src);
	for(int i = 0; i < results.GetCount(); i++) {
		const SlotResult& r = results[i];
		if(r.pixel_bbox.IsEmpty())
			continue;
		Color c = SlotColor(r.slot_id);
		DrawRectOutline(id, r.pixel_bbox, c);
		if(show_offsets && (fabs(r.offset_dx) >= 0.5 || fabs(r.offset_dy) >= 0.5)) {
			Point c0 = r.pixel_bbox.CenterPoint();
			Point c1(c0.x + (int)round(r.offset_dx), c0.y + (int)round(r.offset_dy));
			id.DrawLine(c0.x, c0.y, c1.x, c1.y, 1, c);
		}
	}

	Image res = id;
	
	// Explicit channel-order rule: if we are on X11, ImageDraw might produce BGRA.
	// We check for this once and apply the swap if needed.
	static bool needs_swap = [] {
		Size sz(10, 10);
		ImageBuffer diag_ib(sz);
		for(RGBA *p = diag_ib.Begin(), *e = diag_ib.End(); p < e; p++) {
			p->r = 255; p->g = 0; p->b = 0; p->a = 255;
		}
		Image red_src = diag_ib;
		ImageDraw id(sz);
		id.DrawImage(0, 0, red_src);
		Image img = id;
		Color c = img[5][5];
		return (c.GetR() == 0 && c.GetB() == 255);
	}();

	if(needs_swap) {
		ImageBuffer ib(res);
		for(RGBA *p = ib.Begin(), *e = ib.End(); p < e; p++)
			Swap(p->r, p->b);
		return ib;
	}

	return res;
}

void CvTemplateStepInspector::Clear() {
	preview_.Clear();
	title_.Clear();
	note_.Clear();
	Refresh();
}

void CvTemplateStepInspector::SetData(const Image& img, const String& title, const String& note) {
	preview_ = img;
	title_ = title;
	note_ = note;
	Refresh();
}

void CvTemplateStepInspector::Paint(Draw& w) {
	Size sz = GetSize();
	w.DrawRect(sz, SColorPaper());
	w.DrawRect(0, 0, sz.cx, 1, SColorShadow());
	if(sz.cx <= 0 || sz.cy <= 0)
		return;

	String t = title_.IsEmpty() ? "TemplateMatch" : title_;
	w.DrawText(8, 6, t, StdFont().Bold());

	int top = 28;
	int bottom = max(top + 1, sz.cy - 64);
	Rect img_area(8, top, max(8, sz.cx - 8), bottom);
	w.DrawRect(img_area, Color(245, 245, 245));
	w.DrawRect(img_area.left, img_area.top, img_area.GetWidth(), 1, SColorShadow());
	w.DrawRect(img_area.left, img_area.bottom - 1, img_area.GetWidth(), 1, SColorShadow());
	w.DrawRect(img_area.left, img_area.top, 1, img_area.GetHeight(), SColorShadow());
	w.DrawRect(img_area.right - 1, img_area.top, 1, img_area.GetHeight(), SColorShadow());

	if(!preview_.IsEmpty() && img_area.GetWidth() > 4 && img_area.GetHeight() > 4) {
		Size isz = preview_.GetSize();
		double sx = (double)(img_area.GetWidth() - 4) / max(1, isz.cx);
		double sy = (double)(img_area.GetHeight() - 4) / max(1, isz.cy);
		double sc = min(sx, sy);
		int dw = max(1, (int)floor(isz.cx * sc));
		int dh = max(1, (int)floor(isz.cy * sc));
		Image dimg = (dw == isz.cx && dh == isz.cy) ? preview_ : Rescale(preview_, Size(dw, dh));
		int dx = img_area.left + (img_area.GetWidth() - dw) / 2;
		int dy = img_area.top + (img_area.GetHeight() - dh) / 2;
		w.DrawImage(dx, dy, dimg);
	}
	else {
		w.DrawText(img_area.left + 8, img_area.top + 8, "No TemplateMatch preview crop");
	}

	if(!note_.IsEmpty()) {
		String n = note_;
		n.Replace("\n", " ");
		w.DrawText(8, sz.cy - 44, n, StdFont());
	}
}

OcrStepInspector::OcrStepInspector() {
}

void OcrStepInspector::Clear() {
	original_.Clear();
	preprocessed_.Clear();
	title_.Clear();
	note_.Clear();
	tesseract_psm_ = 7;
	ocr_whitelist_.Clear();
	ocr_blacklist_.Clear();
	Refresh();
}

void OcrStepInspector::SetData(const Image& original, const Image& preprocessed, const String& title, const String& note,
                               int tesseract_psm, const String& ocr_whitelist, const String& ocr_blacklist) {
	original_ = original;
	preprocessed_ = preprocessed;
	title_ = title;
	note_ = note;
	tesseract_psm_ = tesseract_psm;
	ocr_whitelist_ = ocr_whitelist;
	ocr_blacklist_ = ocr_blacklist;
	Refresh();
}

void OcrStepInspector::Paint(Draw& w) {
	Size sz = GetSize();
	w.DrawRect(sz, SColorPaper());
	w.DrawRect(0, 0, sz.cx, 1, SColorShadow());
	if(sz.cx <= 0 || sz.cy <= 0)
		return;

	String t = title_.IsEmpty() ? "OCR Inspector" : title_;
	w.DrawText(8, 6, t, StdFont().Bold());

	int top = 28;
	int margin = 8;
	int avail_w = sz.cx - 2 * margin;
	int avail_h = sz.cy - top - margin - 40;

	if(avail_w > 0 && avail_h > 0) {
		int pane_h = (avail_h - margin) / 2;
		
		auto DrawImagePane = [&](const Image& img, int y, const char* label) {
			w.DrawText(margin, y, label, StdFont().Italic(), SColorDisabled());
			int img_top = y + 18;
			int img_h = pane_h - 18;
			
			Rect img_area(margin, img_top, margin + avail_w, img_top + img_h);
			w.DrawRect(img_area, Color(245, 245, 245));
			w.DrawRect(img_area.left, img_area.top, img_area.GetWidth(), 1, SColorShadow());
			w.DrawRect(img_area.left, img_area.bottom - 1, img_area.GetWidth(), 1, SColorShadow());
			w.DrawRect(img_area.left, img_area.top, 1, img_area.GetHeight(), SColorShadow());
			w.DrawRect(img_area.right - 1, img_area.top, 1, img_area.GetHeight(), SColorShadow());

			if(!img.IsEmpty() && avail_w > 4 && img_h > 4) {
				Size isz = img.GetSize();
				double sx = (double)(avail_w - 4) / isz.cx;
				double sy = (double)(img_h - 4) / isz.cy;
				double sc = min(sx, sy);
				if(sc > 4.0) sc = 4.0;
				
				int dw = (int)(isz.cx * sc);
				int dh = (int)(isz.cy * sc);
				Image dimg = sc == 1.0 ? img : Rescale(img, Size(dw, dh));
				
				w.DrawImage(img_area.left + (img_area.GetWidth() - dw) / 2, 
				            img_area.top + (img_area.GetHeight() - dh) / 2, dimg);
			}
		};

		DrawImagePane(original_, top, "Original Crop");
		DrawImagePane(preprocessed_, top + pane_h + margin, "Preprocessed");
	}

	if(!note_.IsEmpty()) {
		String n = note_;
		n.Replace("\n", " ");
		w.DrawText(margin, sz.cy - 44, n, StdFont(), SColorText());
	}
}

static String ShellQuoteDouble(const String& s) {
	String out = "\"";
	for(int i = 0; i < s.GetCount(); i++) {
		int c = (byte)s[i];
		if(c == '\\' || c == '"')
			out.Cat('\\');
		out.Cat(c);
	}
	out.Cat('"');
	return out;
}

Image OcrStepInspector::BuildTesseractInputImage(const Image& src) const {
	if(src.IsEmpty())
		return src;
	Image img = src;
	if(img.GetWidth() < 400 || img.GetHeight() < 120) {
		double factor = max(400.0 / img.GetWidth(), 120.0 / img.GetHeight());
		img = Rescale(img, (int)(img.GetWidth() * factor), (int)(img.GetHeight() * factor));
	}
	return img;
}

String OcrStepInspector::BuildTesseractCommand(const String& image_path) const {
	String cmd = "tesseract " + ShellQuoteDouble(image_path) + " stdout --psm " + AsString(max(0, tesseract_psm_));
	if(!ocr_whitelist_.IsEmpty())
		cmd << " -c tessedit_char_whitelist=" << ShellQuoteDouble(ocr_whitelist_);
	if(!ocr_blacklist_.IsEmpty())
		cmd << " -c tessedit_char_blacklist=" << ShellQuoteDouble(ocr_blacklist_);
	cmd << " -l eng 2>/dev/null";
	return cmd;
}

void OcrStepInspector::CopyTesseractCommand() {
	if(preprocessed_.IsEmpty()) {
		Exclamation("No OCR preprocessed image available.");
		return;
	}

	Image tess_input = BuildTesseractInputImage(preprocessed_);
	String png_path;
	for(int attempt = 0; attempt < 100; attempt++) {
		png_path = AppendFileName(GetTempDirectory(), Format("fr_ocr_inspector_%d_%d.png", (int)GetTickCount(), attempt));
		if(!FileExists(png_path))
			break;
	}

	if(png_path.IsEmpty() || !PNGEncoder().SaveFile(png_path, tess_input)) {
		Exclamation("Failed to save temporary OCR image.");
		return;
	}

	String cmd = BuildTesseractCommand(png_path);
	WriteClipboardText(cmd);
	PromptOK("Copied Tesseract command to clipboard.\nImage: " + png_path);
}

void OcrStepInspector::RightDown(Point, dword) {
	MenuBar::Execute([&](Bar& bar) {
		bar.Add("Copy Tesseract Command", [=] { CopyTesseractCommand(); })
		   .Enable(!preprocessed_.IsEmpty());
	});
}

// ── LabelAMatchVisualizer ────────────────────────────────────────────────────

String LabelAMatchVisualizer::MethodName(TemplateMatchMethod m) {
	switch(m) {
	case TM_CCOEFF:        return "TM_CCOEFF";
	case TM_CCOEFF_NORMED: return "TM_CCOEFF_NORMED";
	case TM_CCORR:         return "TM_CCORR";
	case TM_CCORR_NORMED:  return "TM_CCORR_NORMED";
	case TM_SQDIFF:        return "TM_SQDIFF";
	case TM_SQDIFF_NORMED: return "TM_SQDIFF_NORMED";
	default:               return "TM_CCOEFF_NORMED";
	}
}

TemplateMatchMethod LabelAMatchVisualizer::MethodFromName(const String& s) {
	if(s == "TM_CCOEFF")        return TM_CCOEFF;
	if(s == "TM_CCORR")         return TM_CCORR;
	if(s == "TM_CCORR_NORMED")  return TM_CCORR_NORMED;
	if(s == "TM_SQDIFF")        return TM_SQDIFF;
	if(s == "TM_SQDIFF_NORMED") return TM_SQDIFF_NORMED;
	return TM_CCOEFF_NORMED;
}

TemplateMatchMethod LabelAMatchVisualizer::GetEffectiveMethod() const {
	if(has_override_)
		return method_override_;
	return MethodFromName(pipeline_method_);
}

void LabelAMatchVisualizer::Clear() {
	response_map_.Clear();
	pipeline_method_.Clear();
	slot_id_.Clear();
	has_override_ = false;
	Refresh();
}

void LabelAMatchVisualizer::SetStep(const ProcessingStepRecord* ps, const Image& response_map_override) {
	if(!ps || ps->stage != "LABEL_A") {
		Clear();
		return;
	}
	slot_id_ = ps->slot_id;
	pipeline_method_ = ps->cv_match_method;
	response_map_ = response_map_override.IsEmpty() ? ps->cv_response_map : response_map_override;
	Refresh();
}

void LabelAMatchVisualizer::SetMethodOverride(TemplateMatchMethod m) {
	has_override_ = true;
	method_override_ = m;
	// Response map rebuild requires raw data we don't store here —
	// caller must call SetStep again after setting override.
	Refresh();
}

void LabelAMatchVisualizer::ClearMethodOverride() {
	has_override_ = false;
	Refresh();
}


void LabelAMatchVisualizer::Paint(Draw& w) {
	Size sz = GetSize();
	bool dark = IsDarkColorFace();
	Color bg = dark ? Color(40, 40, 40) : Color(245, 245, 245);
	Color fg = dark ? White() : Black();
	Color border = dark ? Color(80, 80, 80) : SColorShadow();

	w.DrawRect(sz, bg);
	w.DrawRect(0, 0, sz.cx, 1, border);

	String method_str = has_override_ ? MethodName(method_override_) : pipeline_method_;
	String title = "Match Response Map";
	if(!slot_id_.IsEmpty())
		title << " — " << slot_id_;
	if(!method_str.IsEmpty())
		title << " [" << method_str << "]";

	w.DrawText(8, 6, title, StdFont().Bold(), fg);

	int top = 28;
	Rect img_area(8, top, max(8, sz.cx - 8), max(top + 1, sz.cy - 4));
	w.DrawRect(img_area.left, img_area.top, img_area.GetWidth(), 1, border);
	w.DrawRect(img_area.left, img_area.bottom - 1, img_area.GetWidth(), 1, border);
	w.DrawRect(img_area.left, img_area.top, 1, img_area.GetHeight(), border);
	w.DrawRect(img_area.right - 1, img_area.top, 1, img_area.GetHeight(), border);

	if(!response_map_.IsEmpty() && img_area.GetWidth() > 4 && img_area.GetHeight() > 4) {
		Size isz = response_map_.GetSize();
		double sx = (double)(img_area.GetWidth() - 4) / max(1, isz.cx);
		double sy = (double)(img_area.GetHeight() - 4) / max(1, isz.cy);
		double sc = min(sx, sy);
		int dw = max(1, (int)(isz.cx * sc));
		int dh = max(1, (int)(isz.cy * sc));
		int dx = img_area.left + 2 + (img_area.GetWidth() - 4 - dw) / 2;
		int dy = img_area.top + 2 + (img_area.GetHeight() - 4 - dh) / 2;
		Image disp = (dw == isz.cx && dh == isz.cy) ? response_map_ : Rescale(response_map_, Size(dw, dh));
		w.DrawImage(dx, dy, disp);
	} else {
		String msg = response_map_.IsEmpty() ? "No response map (select a LABEL_A step)" : "";
		if(!msg.IsEmpty())
			w.DrawText(img_area.left + 8, img_area.top + 8, msg, StdFont(), dark ? LtGray() : Gray());
	}
}

void LabelAMatchVisualizer::RightDown(Point, dword) {
	MenuBar::Execute([&](Bar& bar) {
		const char* methods[] = {
			"TM_CCOEFF", "TM_CCOEFF_NORMED",
			"TM_CCORR", "TM_CCORR_NORMED",
			"TM_SQDIFF", "TM_SQDIFF_NORMED"
		};
		TemplateMatchMethod meths[] = {
			TM_CCOEFF, TM_CCOEFF_NORMED,
			TM_CCORR, TM_CCORR_NORMED,
			TM_SQDIFF, TM_SQDIFF_NORMED
		};
		TemplateMatchMethod cur = GetEffectiveMethod();
		for(int i = 0; i < 6; i++) {
			TemplateMatchMethod m = meths[i];
			bar.Add(methods[i], [=] {
				WhenMethodOverride(m);
			}).Check(cur == m);
		}
		if(has_override_) {
			bar.Separator();
			bar.Add("Reset to pipeline default", [=] { WhenMethodOverride(-1); });
		}
	});
}

// ── End LabelAMatchVisualizer ────────────────────────────────────────────────

static Rect AnchorToRectNoOffset(const AnnLayAnchor& anchor, Size img_size) {
	int w = (int)round(anchor.w * img_size.cx);
	int h = (int)round(anchor.h * img_size.cy);
	int cx = (int)round(anchor.cx * img_size.cx);
	int cy = (int)round(anchor.cy * img_size.cy);
	return RectC(cx - w / 2, cy - h / 2, w, h);
}

static Image BuildCvPreviewCrop(const ProcessingStepRecord& ps, const AnchoredSlotRecognizer& recognizer, const Image& src) {
	if(src.IsEmpty())
		return Image();

	Rect r = ps.candidate_bbox;
	if(r.IsEmpty() && !ps.slot_id.IsEmpty()) {
		String stem = ps.slot_id;
		int h = stem.Find('#');
		if(h >= 0)
			stem = stem.Left(h);
		const AnnLay& lay = recognizer.GetLayout();
		const AnnLaySlot* slot = lay.FindSlot(stem + "#label_a");
		if(!slot)
			slot = lay.FindSlot(stem);
		if(slot)
			r = AnchorToRectNoOffset(slot->anchor, src.GetSize());
	}

	Rect ib(0, 0, src.GetWidth(), src.GetHeight());
	r = r & ib;
	if(r.IsEmpty())
		return Image();
	Image img = Crop(src, r);
	if(!img.IsEmpty() && fabs(ps.angle) > 0.01) {
		// Use same inverse rotation as runtime to straighten
		img = RotateBilinear(img, -ps.angle);
	}
	return img;
}

FrameRecognizerWindow::FrameRecognizerWindow() {
	Title("Frame Recognizer");
	Sizeable().Zoomable();
	Maximize();

	Add(tabs.SizePos());
	tabs.Add(video_feed_.SizePos(), "Video feed");

	steps_inspector_host_.Add(steps_nn_inspector_.SizePos());
	steps_inspector_host_.Add(steps_cv_inspector_.SizePos());
	steps_inspector_host_.Add(steps_ocr_inspector_.SizePos());
	steps_inspector_host_.Add(steps_script_inspector_.SizePos());
	steps_nn_inspector_.Hide();
	steps_cv_inspector_.Hide();
	steps_ocr_inspector_.Hide();
	steps_script_inspector_.Hide();

	steps_label_a_candidates_.AddColumn("Template").SetDisplay(FittedImageDisplay());
	steps_label_a_candidates_.AddColumn("Class");
	steps_label_a_candidates_.AddColumn("Score / Info");
	steps_label_a_candidates_.ColumnWidths("48 80 200");
	steps_label_a_candidates_.EvenRowColor();

	steps_label_a_candidates_.WhenSel = [=] {
		if(steps_label_a_candidates_.IsCursor()) {
			int r = steps_label_a_candidates_.GetCursor();
			Image img = steps_label_a_candidates_.Get(r, 0);
			String name = steps_label_a_candidates_.Get(r, 1);
			String info = steps_label_a_candidates_.Get(r, 2);
			steps_label_a_template_view_.SetData(img, name, info);
		} else {
			steps_label_a_template_view_.Clear();
		}
	};

	steps_label_a_splitter_.Vert() << steps_label_a_vis_ << steps_label_a_candidates_ << steps_label_a_template_view_;
	steps_label_a_splitter_.SetPos(0, 4000);
	steps_label_a_splitter_.SetPos(1, 7000);
	steps_label_a_splitter_.Hide();

	steps_label_a_vis_.WhenMethodOverride = [=](int raw_method) {
		if(raw_method < 0) {
			steps_label_a_vis_.ClearMethodOverride();
		} else {
			steps_label_a_vis_.SetMethodOverride((TemplateMatchMethod)raw_method);
		}
		if(label_a_step_)
			PopulateLabelAInspector(*label_a_step_, steps_label_a_vis_.GetEffectiveMethod());
	};

	steps_inspector_splitter_.Horz(steps_inspector_host_, steps_label_a_splitter_);
	steps_details_splitter_.Vert(steps_details_, steps_inspector_splitter_);
	steps_details_splitter_.SetPos(4000);

	steps_splitter_.Horz(steps_list_, steps_details_splitter_);
	tabs.Add(steps_splitter_.SizePos(), "Processing steps");
	
	steps_list_.AddColumn("Step / Slot ID");
	steps_list_.AddColumn("Seq");
	steps_list_.AddColumn("Stage");
	steps_list_.AddColumn("Timing (ms)");
	steps_list_.AddColumn("Status");
	steps_list_.AddColumn("Note");
	steps_list_.AddColumn("StepID"); // Hidden column for ID
	steps_list_.ColumnWidths("200 60 100 80 100 200 0");
	steps_list_.EvenRowColor();
	steps_list_.Header(true); // Ensure header is visible
	steps_list_.WhenSel = [=] {
		if(steps_list_.IsCursor()) {
			int r = steps_list_.GetCursor();
			
			// We need to find the matching step in displayed_log_.steps
			String step_name = steps_list_.GetRowValue(r, 0);
			String stage_name = steps_list_.GetRowValue(r, 2);
			String note_text = steps_list_.GetRowValue(r, 5);
			Value step_id_val = steps_list_.GetRowValue(r, 6);
			int step_id = -1;
			if(!IsNull(step_id_val)) step_id = (int)step_id_val;
			
			const ProcessingStepRecord* ps = nullptr;
			for(const auto& s : displayed_log_.steps) {
				if(s.step_id == step_id && step_id != -1) {
					ps = &s;
					break;
				}
				// Fallback for TOTAL/GROUP/PIPELINE which might not have step_ids
				if(step_id == -1 && s.stage == stage_name && (s.slot_id == step_name || (s.slot_id.IsEmpty() && (step_name == s.stage || (s.stage == "TOTAL" && step_name == "Total"))))) {
					if(stage_name == "CANDIDATE" && s.note != note_text)
						continue;

					ps = &s;
					break;
				}
			}
			
			if(ps) {
				bool is_label_a = (ps->stage == "LABEL_A" || ps->stage == "LEVEL");
				if(is_label_a) {
					label_a_step_ = const_cast<ProcessingStepRecord*>(ps);
					PopulateLabelAInspector(*label_a_step_, steps_label_a_vis_.GetEffectiveMethod());
					steps_label_a_splitter_.Show();
					steps_inspector_splitter_.SetPos(4000); // Show right pane (approx 40% width)
					
					Image img = video_feed_.GetCurrentImage();
					if(img.IsEmpty() && !displayed_log_.image_path.IsEmpty())
						img = StreamRaster::LoadFileAny(displayed_log_.image_path);
					Image cv_crop = BuildCvPreviewCrop(*ps, recognizer_, img);
					cv_crop = AnchoredSlotClassifier::LinearContrastStretching(cv_crop);
					steps_cv_inspector_.SetData(cv_crop, "TemplateMatch Input: " + ps->slot_id, ps->note);
					steps_cv_inspector_.Show();
					steps_nn_inspector_.Hide();
					steps_ocr_inspector_.Hide();
					steps_script_inspector_.Hide();
				}
				else if(ps->stage == "OCR") {
					label_a_step_ = nullptr;
					steps_label_a_splitter_.Hide();
					steps_inspector_splitter_.SetPos(10000); // Collapse right pane
					
					Image img = video_feed_.GetCurrentImage();
					if(img.IsEmpty() && !displayed_log_.image_path.IsEmpty())
						img = StreamRaster::LoadFileAny(displayed_log_.image_path);
					
					if(!img.IsEmpty() && !ps->candidate_bbox.IsEmpty()) {
						Image original = Crop(img, ps->candidate_bbox);
						Image preprocessed = original;
						if(ps->ocr_mode >= 0) {
							OCR::PreprocessingOptions opts;
							opts.mode = (OCR::OCRPreprocessMode)ps->ocr_mode;
							preprocessed = OCR::Preprocess(original, opts);
						}
						int psm = 7;
						String whitelist;
						String blacklist;
						{
							Mutex::Lock __(recognizer_lock_);
							const AnnLaySlot* ocr_slot = recognizer_.GetLayout().FindSlot(ps->slot_id);
							if(ocr_slot) {
								if(ocr_slot->ocr_psm >= 0)
									psm = ocr_slot->ocr_psm;
								whitelist = ocr_slot->ocr_whitelist;
								blacklist = ocr_slot->ocr_blacklist;
							}
						}
						steps_ocr_inspector_.SetData(original, preprocessed, "OCR Input: " + ps->slot_id, ps->note,
						                             psm, whitelist, blacklist);
						steps_ocr_inspector_.Show();
						steps_nn_inspector_.Hide();
						steps_cv_inspector_.Hide();
						steps_script_inspector_.Hide();
					}
				}
				else if(ps->is_nn_step) {
					label_a_step_ = nullptr;
					steps_label_a_splitter_.Hide();
					steps_inspector_splitter_.SetPos(10000); // Collapse right pane
					Mutex::Lock __(recognizer_lock_);
					
					String effective_head_id = ps->head_id;
					// If head_id was not explicitly stored or is generic, resolve from stage
					if(effective_head_id.IsEmpty() || effective_head_id == ps->slot_id) {
						if(ps->stage == "LEVEL") effective_head_id = ps->slot_id + "#level";
						else if(ps->stage == "CATEGORY") effective_head_id = ps->slot_id + "#category";
						else if(ps->stage == "X_OFFSET" || ps->stage == "Y_OFFSET") {
							// Try to see if it's a composite element's offset
							const AnnLaySlot* slot = recognizer_.GetLayout().FindSlot(ps->slot_id);
							if(slot && slot->composite_type == ANNLAY_COMPOSITE_ELEMENT)
								effective_head_id = ps->slot_id + "#offset";
							else
								effective_head_id = ps->slot_id;
						}
						else if(ps->stage == "VISIBLE" || ps->stage == "CANDIDATE") {
							effective_head_id = ps->slot_id;
							// If it's a composite element, the presence head suffix is #presence
							const AnnLaySlot* slot = recognizer_.GetLayout().FindSlot(ps->slot_id);
							if(slot && slot->composite_type == ANNLAY_COMPOSITE_ELEMENT)
								effective_head_id = ps->slot_id + "#presence";
						}
					}
					
					// Re-map LEVEL/CATEGORY/OFFSET/PRESENCE too if needed
					String gkey = AnchoredSlotClassifier::BoolSlotGroupKey(effective_head_id, &recognizer_.GetLayout());
					::ConvNet::Session* ses = nullptr;
					if(!gkey.IsEmpty()) ses = recognizer_.GetSession(gkey);
					if(!ses) ses = recognizer_.GetSession(effective_head_id);
					
					if(ses) {
						// Attempt to re-predict on the current image if we have a bbox
						if(!ps->candidate_bbox.IsEmpty()) {
							Image img = video_feed_.GetCurrentImage();
							if(img.IsEmpty() && !displayed_log_.image_path.IsEmpty()) {
								img = StreamRaster::LoadFileAny(displayed_log_.image_path);
							}

							if(!img.IsEmpty()) {
								recognizer_.PredictCrop(effective_head_id, img, ps->candidate_bbox, ps->crop_size, ps->is_grayscale, ps->angle, ps->is_equalized);
							}
						}
						
						steps_nn_inspector_.SetColor(ses->GetInput()->input_depth == 3);
						steps_nn_inspector_.SetSession(*ses);
						steps_nn_inspector_.Show();
						steps_cv_inspector_.Hide();
						steps_ocr_inspector_.Hide();
						steps_script_inspector_.Hide();
						WhenSelNN();
					} else {
						// If session not found, still show empty inspector or hide it?
						// For now, hide if session is totally missing.
						steps_nn_inspector_.Hide();
						steps_cv_inspector_.Hide();
						steps_ocr_inspector_.Hide();
						steps_script_inspector_.Hide();
						steps_label_a_splitter_.Hide();
					}
				} else if(ps->stage == "SCRIPT" || ps->stage == "RecognizeSummary") {
					label_a_step_ = nullptr;
					steps_label_a_splitter_.Hide();
					steps_nn_inspector_.Hide();
					steps_cv_inspector_.Hide();
					steps_ocr_inspector_.Hide();
					steps_inspector_splitter_.SetPos(10000);
					String out;
					if(!displayed_log_.script_output.IsEmpty())
						out << displayed_log_.script_output;
					if(!displayed_log_.script_error.IsEmpty()) {
						if(!out.IsEmpty()) out << "\n";
						out << "--- Error ---\n" << displayed_log_.script_error;
					}
					steps_script_inspector_.Set(out);
					steps_script_inspector_.Show();
				} else {
					label_a_step_ = nullptr;
					steps_nn_inspector_.Hide();
					steps_cv_inspector_.Hide();
					steps_ocr_inspector_.Hide();
					steps_script_inspector_.Hide();
					steps_label_a_splitter_.Hide();
					steps_inspector_splitter_.SetPos(10000); // Collapse right pane
				}

				String info;
				info << "Step:      " << ps->slot_id << "\n";
				info << "Stage:     " << ps->stage << "\n";
				info << "Timing:    " << Format("%.2f", ps->duration_ms) << " ms\n";
				info << "Status:    " << ps->status << "\n";
				info << "Note:      " << ps->note << "\n";
				
				if(ps->stage == "SCRIPT") {
					info << "\n--- Python Script ---\n";
					String script_path = ResolveFromSlnDir(sln_dir_, sln_recognition_script_);
					info << "Path:      " << script_path << "\n";
					if(FileExists(script_path)) {
						info << "\nSource:\n" << LoadFile(script_path) << "\n";
					}
					if(!displayed_log_.script_output.IsEmpty()) {
						info << "\n--- Script Execution Output ---\n";
						info << displayed_log_.script_output << "\n";
					}
					if(!displayed_log_.script_error.IsEmpty()) {
						info << "\n--- Script Execution Error ---\n";
						info << displayed_log_.script_error << "\n";
					}
				}

				if(!ps->detailed_note.IsEmpty()) {
					info << "\nDetails:\n" << ps->detailed_note << "\n";
				}
				if(ps->counters.GetCount() > 0) {
					info << "\nCounters:\n";
					for(int i = 0; i < ps->counters.GetCount(); i++)
						info << "  " << ps->counters.GetKey(i) << ": " << ps->counters[i] << "\n";
				}
				steps_details_.Set(info);
			} else {
				label_a_step_ = nullptr;
				steps_nn_inspector_.Hide();
				steps_cv_inspector_.Hide();
				steps_ocr_inspector_.Hide();
				steps_script_inspector_.Hide();
				steps_label_a_splitter_.Hide();
				steps_inspector_splitter_.SetPos(10000); // Collapse right pane
				steps_details_.Set(String("Step details not found in log."));
			}
		} else {
			label_a_step_ = nullptr;
			steps_nn_inspector_.Hide();
			steps_cv_inspector_.Hide();
			steps_ocr_inspector_.Hide();
			steps_script_inspector_.Hide();
			steps_label_a_splitter_.Hide();
			steps_inspector_splitter_.SetPos(10000); // Collapse right pane
			steps_details_.Clear();
		}
	};

	AddFrame(menu_);
	AddFrame(toolbar_);
	menu_.Set(THISBACK(MainMenu));
	toolbar_.Set(THISBACK(BuildToolbar));

	model_set_drop_.WhenAction = THISBACK(OnModelSetChanged);
	model_set_drop_.Add("pass1");
	model_set_drop_.Add("pass2");
	model_set_drop_.SetIndex(0);

	offset_mode_drop_.WhenAction = THISBACK(OnOffsetModeChanged);
	offset_mode_drop_.Add("auto");
	offset_mode_drop_.Add("none");
	offset_mode_drop_.Add("combined");
	offset_mode_drop_.Add("split");
	offset_mode_drop_.SetData("auto");

	bool_policy_drop_.WhenAction = THISBACK(OnBoolPolicyChanged);
	bool_policy_drop_.Add("permissive");
	bool_policy_drop_.Add("strict");
	bool_policy_drop_.SetData("strict");

	lbl_offset_mode_.SetLabel("Offset:");
	lbl_bool_policy_.SetLabel("Gate Policy:");

	// Initialize image list with empty data
	images_list_.Clear();
}

FrameRecognizerWindow::~FrameRecognizerWindow() {
	closing_ = true;
	StopRecognitionThread();
	if(image_proc_thread_.IsOpen())
		image_proc_thread_.Wait();
	KillTimeCallback(TIMEID_SLIDESHOW);
}

void FrameRecognizerWindow::DockInit() {
	if(detections_list_.GetColumnCount() == 0) {
		detections_list_.AddColumn("Slot");
		detections_list_.AddColumn("Value");
		detections_list_.AddColumn("Conf%");
		detections_list_.AddColumn("Offset");
		detections_list_.AddColumn("Gate");
		detections_list_.AddColumn("Status");
		detections_list_.ColumnWidths("220 180 70 100 150 120");
		detections_list_.EvenRowColor();
	}
	if(!dock_detections_) {
		dock_detections_ = &Dockable(detections_list_, "Detections").SizeHint(Size(380, 280));
		DockRight(*dock_detections_);
	}

	if(images_list_.GetColumnCount() == 0) {
		images_list_.AddColumn("Number");
		images_list_.AddColumn("Frame Name");
		images_list_.AddColumn("Processing Time");
		images_list_.ColumnWidths("60 200 100");
		images_list_.EvenRowColor();
	}
	images_list_.WhenBar = THISBACK(OnImageListContextMenu);
	images_list_.WhenCursor = THISBACK(OnImageListCursor);
	if(!dock_images_) {
		dock_images_ = &Dockable(images_list_, "Images").SizeHint(Size(380, 280));
		DockLeft(*dock_images_);
	}

	if(log_list_.GetColumnCount() == 0) {
		log_list_.AddColumn("Seq", 40);
		log_list_.AddColumn("Index", 40);
		log_list_.AddColumn("File", 150);
		log_list_.AddColumn("Status", 60);
		log_list_.AddColumn("Pipeline (ms)", 80);
		log_list_.AddColumn("Recognize (ms)", 80);
		log_list_.AddColumn("Script (ms)", 80);
		log_list_.AddColumn("Detections", 80);
		log_list_.AddColumn("Warnings", 200);
		log_list_.AddColumn("ScriptOutput");
		log_list_.AddColumn("ScriptError");
		log_list_.HeaderTab(9).Hide();
		log_list_.HeaderTab(10).Hide();
		log_list_.EvenRowColor();

		log_list_.WhenBar = [=](Bar& bar) {
			bar.Add("Clear Log", THISBACK(OnClearLog));
		};

		log_list_.WhenSel = THISBACK(UpdateLogDetail);

		log_list_.WhenLeftDouble = [=] {
			if(log_list_.IsCursor()) {
				int r = log_list_.GetCursor();
				String info;
				info << "Frame Seq: " << log_list_.Get(r, 0) << "\n";
				info << "Flat Index: " << log_list_.Get(r, 1) << "\n";
				info << "File: " << log_list_.Get(r, 2) << "\n";
				info << "Status: " << log_list_.Get(r, 3) << "\n";
				info << "Pipeline: " << log_list_.Get(r, 4) << " ms\n";
				info << "Recognize: " << log_list_.Get(r, 5) << " ms\n";
				info << "Script: " << log_list_.Get(r, 6) << " ms\n";
				info << "Detections: " << log_list_.Get(r, 7) << "\n";
				info << "Warnings: " << log_list_.Get(r, 8) << "\n";

				String sout = log_list_.Get(r, 9);
				if(!sout.IsEmpty()) {
					info << "\n--- Script Output ---\n" << sout;
				}
				String serr = log_list_.Get(r, 10);
				if(!serr.IsEmpty()) {
					info << "\n--- Script Error ---\n" << serr;
				}
				PromptOK(DeQtf(info));
			}
		};
	}
	if(!dock_log_) {
		dock_log_ = &Dockable(log_list_, "Processing Log").SizeHint(Size(600, 200));
		DockBottom(*dock_log_);
	}

	if(!dock_log_details_) {
		log_details_.SetReadOnly();
		log_details_.SetFont(Monospace());
		dock_log_details_ = &Dockable(log_details_, "Frame Log Detail").SizeHint(Size(600, 300));
		DockBottom(*dock_log_details_);
	}
}

void FrameRecognizerWindow::UpdateImageList() {
	if(!slideshow_mode_ || slideshow_images_.IsEmpty())
		return;
	
	// Set the number of rows efficiently
	int count = slideshow_images_.GetCount();
	images_list_.SetCount(count);
	
	// Update each row with image information
	for(int i = 0; i < count; i++) {
		Value img_rec = slideshow_images_[i];
		String file_name = img_rec["file_name"].ToString();
		if(file_name.IsEmpty())
			file_name = img_rec["file_path"].ToString();
		
		// Get the filename without path
		file_name = GetFileName(file_name);
		
		// Set values for each column
		images_list_.Set(i, 0, AsString(i + 1));  // Number
		images_list_.Set(i, 1, file_name);        // Frame Name
		images_list_.Set(i, 2, String());         // Processing Time (empty initially)
	}
	
	// Highlight the current image if it exists
	if(current_flat_index_ >= 0 && current_flat_index_ < count) {
		images_list_.SetCursor(current_flat_index_);
	}
}

void FrameRecognizerWindow::UpdateLogDetail() {
	if(log_list_.IsCursor()) {
		int r = log_list_.GetCursor();
		if(r >= 0 && r < log_history_.GetCount()) {
			log_details_.Set(log_history_[r].FormatVerbose());
		}
	} else {
		log_details_.Clear();
	}
}

void FrameRecognizerWindow::OnSaveLog() {
	if(displayed_log_.frame_seq < 0) {
		PromptOK("No processing log currently displayed to save.");
		return;
	}

	String fn = SelectFileSaveAs("JSON file (*.json)\t*.json\n");
	if(fn.IsEmpty()) return;
	
	if(StoreAsJsonFile(displayed_log_, fn, true)) {
		PromptOK("Processing log saved successfully.");
	} else {
		PromptOK("Failed to save processing log.");
	}
}

void FrameRecognizerWindow::OnLoadLog() {
	String fn = SelectFileOpen("JSON file (*.json)\t*.json\n");
	if(fn.IsEmpty()) return;

	ProcessingLogRecord log;
	if(LoadFromJsonFile(log, fn)) {
		displayed_log_ <<= log;
		PopulateStepsTree(displayed_log_);
		
		// If image path exists, try to load and show it
		if(FileExists(displayed_log_.image_path)) {
			Image source = StreamRaster::LoadFileAny(displayed_log_.image_path);
			if(!source.IsEmpty()) {
				video_feed_.SetCurrentImage(source);
			}
		}
		
		UpdateOverlay(displayed_log_.results);
		UpdateDetectionsDock(displayed_log_.results);
		
		PromptOK("Processing log loaded successfully.");
	} else {
		PromptOK("Failed to load processing log.");
	}
}

void FrameRecognizerWindow::OnClearLog() {
	log_list_.Clear();
	log_history_.Clear();
	log_details_.Clear();
	stats_processed_ = 0;
	stats_failed_ = 0;
	stats_avg_total_ms_ = 0;
	stats_avg_recog_ms_ = 0;
	stats_max_total_ms_ = 0;
	stats_last_status_.Clear();
	if(dock_log_)
		dock_log_->Title("Processing Log");

	// Clear the image list as well
	images_list_.Clear();

	// Clear cached results
	{
		Mutex::Lock __(recog_lock_);
		results_cache_.Clear();
		slot_results_cache_.Clear();
	}
}

void FrameRecognizerWindow::Close() {
	slideshow_running_ = false;
	KillTimeCallback(TIMEID_SLIDESHOW);
	closing_ = true;
	StopRecognitionThread();
	TopWindow::Close();
}

bool FrameRecognizerWindow::LoadProject(const String& path) {
	annprj_path_ = NormalizePath(path);
	DumpFrameRecognizerMemoryEvent("FrameRecognizer.LoadProject.begin", "annprj='" + annprj_path_ + "'");
	String json = LoadFile(annprj_path_);
	if(json.IsEmpty()) return false;
	
	annprj_root_ = ParseJSON(json);
	if(annprj_root_.IsError()) return false;

	slideshow_images_.Clear();
	slideshow_dataset_idx_.Clear();
	slideshow_image_idx_.Clear();

	Value datasets = annprj_root_["datasets"];
	if(IsValueArray(datasets)) {
		for(int d = 0; d < datasets.GetCount(); d++) {
			Value imgs = datasets[d]["images"];
			if(!IsValueArray(imgs))
				continue;
			for(int i = 0; i < imgs.GetCount(); i++) {
				slideshow_images_.Add(imgs[i]);
				slideshow_dataset_idx_.Add(d);
				slideshow_image_idx_.Add(i);
			}
		}
	}
	
	UpdateImageList();
	DumpFrameRecognizerMemoryEvent("FrameRecognizer.LoadProject.ready",
		Format("images=%d", slideshow_images_.GetCount()));
	return true;
}

bool FrameRecognizerWindow::LoadRecognizer(const String& annmdl_path, const String& annlay_path) {
	DumpFrameRecognizerMemoryEvent("FrameRecognizer.LoadRecognizer.begin",
		Format("annlay='%s' annmdl='%s'", annlay_path, annmdl_path));
	if(recog_thread_.IsOpen())
		recog_thread_.Wait();
	{
		Mutex::Lock __(recog_lock_);
		recog_busy_ = false;
		has_pending_job_ = false;
		has_completed_job_ = false;
		completed_source_image_.Clear();
		completed_display_image_.Clear();
		completed_results_.Clear();
		completed_meta_.Clear();
		completed_flat_index_ = -1;
		completed_frame_seq_ = -1;
	}
	rec_loaded_ = false;

	if(annlay_path.IsEmpty() || !FileExists(annlay_path))
		return false;

	{
		Mutex::Lock __(recognizer_lock_);
		steps_nn_inspector_.Clear();
		steps_cv_inspector_.Clear();
		steps_cv_inspector_.Hide();
		label_a_step_ = nullptr;
		steps_label_a_vis_.Clear();
		steps_label_a_splitter_.Hide();
		if(has_sln_cfg_)
			recognizer_.SetSln(sln_cfg_);
		if(!recognizer_.Load(annlay_path, annmdl_path))
			return false;

		recognizer_.SetOffsetMode(StringToOffsetMode(offset_mode_drop_.GetData()));
		recognizer_.SetBoolGatePolicy(StringToBoolGatePolicy(bool_policy_drop_.GetData()));

		// Note: script loading might need more context if not using OpenSlideshow
	}

	rec_loaded_ = true;
	DumpFrameRecognizerMemoryEvent("FrameRecognizer.LoadRecognizer.ready",
		Format("layout_slots=%d", recognizer_.GetLayout().slots.GetCount()));
	return true;
}

bool FrameRecognizerWindow::LoadRecognizer(const String& model_set) {
	DumpFrameRecognizerMemoryEvent("FrameRecognizer.LoadRecognizerSet.begin", "model_set='" + model_set + "'");
	if(recog_thread_.IsOpen())
		recog_thread_.Wait();
	{
		Mutex::Lock __(recog_lock_);
		recog_busy_ = false;
		has_pending_job_ = false;
		has_completed_job_ = false;
		completed_source_image_.Clear();
		completed_display_image_.Clear();
		completed_results_.Clear();
		completed_meta_.Clear();
		completed_flat_index_ = -1;
		completed_frame_seq_ = -1;
	}
	rec_loaded_ = false;

	String annlay_path = ResolveFromSlnDir(sln_dir_, sln_annlay_);
	if(annlay_path.IsEmpty() || !FileExists(annlay_path))
		return false;

	String annmdl_override;
	int ms = sln_model_sets_.Find(model_set);
	if(ms >= 0)
		annmdl_override = ResolveFromSlnDir(sln_dir_, sln_model_sets_[ms]);

	{
		Mutex::Lock __(recognizer_lock_);
		steps_nn_inspector_.Clear();
		steps_cv_inspector_.Clear();
		steps_cv_inspector_.Hide();
		label_a_step_ = nullptr;
		steps_label_a_vis_.Clear();
		steps_label_a_splitter_.Hide();
		if(has_sln_cfg_)
			recognizer_.SetSln(sln_cfg_);
		if(!recognizer_.Load(annlay_path, annmdl_override))
			return false;

		recognizer_.SetOffsetMode(StringToOffsetMode(offset_mode_drop_.GetData()));
		recognizer_.SetBoolGatePolicy(StringToBoolGatePolicy(bool_policy_drop_.GetData()));

		if(!sln_recognition_script_.IsEmpty()) {
			String script_path = ResolveFromSlnDir(sln_dir_, sln_recognition_script_);
			if(FileExists(script_path))
				script_.Load(script_path);
		}
	}

	rec_loaded_ = true;
	DumpFrameRecognizerMemoryEvent("FrameRecognizer.LoadRecognizerSet.ready",
		Format("model_set='%s' layout_slots=%d", model_set, recognizer_.GetLayout().slots.GetCount()));
	return true;
}

void FrameRecognizerWindow::OpenSlideshow(const AnnSln& sln,
                                          const String& sln_dir,
                                          const String& annprj_path,
                                          const String& model_set) {
	closing_ = false;
	sln_dir_ = NormalizePath(sln_dir);
	sln_cfg_ = sln;
	has_sln_cfg_ = true;
	annprj_path_ = NormalizePath(annprj_path);
	sln_annlay_ = sln.annlay;
	sln_recognition_script_ = sln.recognition_script;
	sln_images_dir_ = sln.images_dir;
	sln_model_sets_ <<= sln.model_sets;
	slideshow_mode_ = true;
	slideshow_running_ = false;
	slideshow_idx_ = -1;
	current_flat_index_ = -1;
	current_frame_seq_ = 0;
	results_cache_.Clear();

	model_set_drop_.Clear();
	int selected = -1;
	for(int i = 0; i < sln_model_sets_.GetCount(); i++) {
		String name = sln_model_sets_.GetKey(i);
		model_set_drop_.Add(name);
		if(name == model_set)
			selected = i;
	}
	if(model_set_drop_.GetCount() == 0) {
		model_set_drop_.Add("pass1");
		model_set_drop_.Add("pass2");
	}
	if(selected >= 0)
		model_set_drop_.SetIndex(selected);
	else
		model_set_drop_.SetIndex(0);

	String active_model_set = model_set_drop_.GetValue();
	if(active_model_set.IsEmpty())
		active_model_set = model_set;
	LoadRecognizer(active_model_set);

	String json = LoadFile(annprj_path_);
	annprj_root_ = ParseJSON(json);
	slideshow_images_.Clear();
	slideshow_dataset_idx_.Clear();
	slideshow_image_idx_.Clear();

	Value datasets = annprj_root_["datasets"];
	if(IsValueArray(datasets)) {
		for(int d = 0; d < datasets.GetCount(); d++) {
			Value imgs = datasets[d]["images"];
			if(!IsValueArray(imgs))
				continue;
			for(int i = 0; i < imgs.GetCount(); i++) {
				slideshow_images_.Add(imgs[i]);
				slideshow_dataset_idx_.Add(d);
				slideshow_image_idx_.Add(i);
			}
		}
	}

	// Update the image list with new data
	UpdateImageList();
	DumpFrameRecognizerMemoryEvent("FrameRecognizer.OpenSlideshow.ready",
		Format("model_set='%s' images=%d", active_model_set, slideshow_images_.GetCount()));

	#if 0
	slideshow_running_ = !slideshow_images_.IsEmpty();
	KillTimeCallback(TIMEID_SLIDESHOW);
	if(slideshow_running_)
		SetTimeCallback(-80, THISBACK(OnSlideshowTimer), TIMEID_SLIDESHOW);
	#endif

	Title(Format("Frame Recognizer - Slideshow (%d images)", slideshow_images_.GetCount()));
	if(!IsOpen())
		Open();
}

void FrameRecognizerWindow::OpenRealtime(const AnnSln& sln,
                                         const String& sln_dir,
                                         const String& model_set) {
	closing_ = false;
	sln_dir_ = NormalizePath(sln_dir);
	sln_cfg_ = sln;
	has_sln_cfg_ = true;
	annprj_path_.Clear();
	sln_annlay_ = sln.annlay;
	sln_recognition_script_ = sln.recognition_script;
	sln_images_dir_ = sln.images_dir;
	sln_model_sets_ <<= sln.model_sets;
	slideshow_mode_ = false;
	slideshow_running_ = false;
	slideshow_images_.Clear();
	slideshow_dataset_idx_.Clear();
	slideshow_image_idx_.Clear();
	results_cache_.Clear();
	current_flat_index_ = -1;
	current_frame_seq_ = 0;
	KillTimeCallback(TIMEID_SLIDESHOW);

	model_set_drop_.Clear();
	int selected = -1;
	for(int i = 0; i < sln_model_sets_.GetCount(); i++) {
		String name = sln_model_sets_.GetKey(i);
		model_set_drop_.Add(name);
		if(name == model_set)
			selected = i;
	}
	if(model_set_drop_.GetCount() == 0) {
		model_set_drop_.Add("pass1");
		model_set_drop_.Add("pass2");
	}
	if(selected >= 0)
		model_set_drop_.SetIndex(selected);
	else
		model_set_drop_.SetIndex(0);

	String active_model_set = model_set_drop_.GetValue();
	if(active_model_set.IsEmpty())
		active_model_set = model_set;
	LoadRecognizer(active_model_set);

	Title("Frame Recognizer - Realtime");
	if(!IsOpen())
		Open();
	
	// Clear the image list for realtime mode
	images_list_.Clear();
}

void FrameRecognizerWindow::RunOnCurrentFrame(const Image& img) {
	if(!rec_loaded_ || img.IsEmpty())
		return;
	RecognitionJob job;
	job.img = img;
	job.flat_index = current_flat_index_;
	job.frame_seq = current_frame_seq_;
	job.show_offsets = video_feed_.IsShowingOffsets();
	QueueRecognition(job);
}

void FrameRecognizerWindow::QueueRecognition(const RecognitionJob& job) {
	if(!rec_loaded_)
		return;
	if(job.img.IsEmpty() && job.image_path.IsEmpty())
		return;

	RecognitionJob j = job;
	j.start_time = GetTickCount();

	{
		Mutex::Lock __(recog_lock_);
		if(recog_busy_) {
			pending_job_ = j;
			has_pending_job_ = true;
			FrameRecTrace(Format("queue: busy -> pending frame_seq=%d flat=%d path='%s'",
			                     j.frame_seq, j.flat_index, j.image_path));
			return;
		}
		recog_busy_ = true;
	}
	FrameRecTrace(Format("queue: start frame_seq=%d flat=%d path='%s'",
	                     j.frame_seq, j.flat_index, j.image_path));
	StartRecognitionThread(j);
}

void FrameRecognizerWindow::StartRecognitionThread(const RecognitionJob& job) {
	recog_thread_.Run([=] {
		uint64 t0 = GetTickCount();
		ProcessingLogRecord log;
		log.frame_seq = job.frame_seq;
		log.flat_index = job.flat_index;
		log.image_name = GetFileName(job.image_path);
		log.image_path = job.image_path;
		log.t_queue_ms = (double)(t0 - job.start_time);

		uint64 t_load_start = GetTickCount();
		Image source = job.img;
		String source_info = source.IsEmpty() ? job.image_path : "Memory (Video Feed)";
		if(source.IsEmpty() && !job.image_path.IsEmpty()) {
			DumpFrameRecognizerMemoryEvent("FrameRecognizer.ImageLoad.begin",
				Format("frame_seq=%d flat=%d path='%s'", job.frame_seq, job.flat_index, job.image_path));
			source = StreamRaster::LoadFileAny(job.image_path);
		}
		else if(!source.IsEmpty()) {
			DumpFrameRecognizerMemoryEvent("FrameRecognizer.ImageLoad.memory",
				Format("frame_seq=%d flat=%d size=%dx%d",
				       job.frame_seq, job.flat_index, source.GetWidth(), source.GetHeight()));
		}
		log.t_load_ms = (double)(GetTickCount() - t_load_start);
		if(!source.IsEmpty()) {
			double mb = (double)source.GetWidth() * (double)source.GetHeight() * 4.0 / (1024.0 * 1024.0);
			DumpFrameRecognizerMemoryEvent("FrameRecognizer.ImageLoad.ready",
				Format("frame_seq=%d flat=%d size=%dx%d approx_rgba=%.2fMB load_ms=%.1f",
				       job.frame_seq, job.flat_index, source.GetWidth(), source.GetHeight(), mb, log.t_load_ms));
		}
		else {
			DumpFrameRecognizerMemoryEvent("FrameRecognizer.ImageLoad.fail",
				Format("frame_seq=%d flat=%d path='%s' load_ms=%.1f",
				       job.frame_seq, job.flat_index, job.image_path, log.t_load_ms));
		}
		
		if(source.IsEmpty()) {
			log.AddStep("LOAD", log.t_load_ms, "ERROR", "Failed to load/decode image", String(), String(), false, "Source: " + source_info);
		} else {
			log.AddStep("LOAD", log.t_load_ms, "OK", Format("%d x %d", source.GetWidth(), source.GetHeight()), String(), String(), false, "Source: " + source_info);
		}

		Vector<SlotResult> results;
		VectorMap<String, String> meta;
		Image rendered;

		FrameRecTrace(Format("worker: begin frame_seq=%d flat=%d path='%s' loaded=%s",
		                     job.frame_seq, job.flat_index, job.image_path,
		                     source.IsEmpty() ? "no" : "yes"));

		if(!closing_) {
			Mutex::Lock __(recognizer_lock_);
			if(rec_loaded_ && !source.IsEmpty()) {
				uint64 t_rec_start = GetTickCount();
				results = recognizer_.Recognize(source, &log);
				log.t_recognize_ms = (double)(GetTickCount() - t_rec_start);

				if(job.flat_index >= 0) {
					uint64 t_script_start = GetTickCount();
					meta = script_.Run(results);
					log.t_script_ms = (double)(GetTickCount() - t_script_start);
					log.script_output = script_.GetLastOutput();
					log.script_error = script_.GetLastError();
					
					if(!log.script_error.IsEmpty())
						log.AddStep("SCRIPT", log.t_script_ms, "ERROR", "Script execution failed", String(), String(), false, log.script_error);
					else
						log.AddStep("SCRIPT", log.t_script_ms, "OK", Format("%d meta keys", meta.GetCount()));
				}
			}
		}

		if(closing_)
			return;
		
		log.t_total_ms = (double)(GetTickCount() - job.start_time);
		log.AddStep("TOTAL", log.t_total_ms, "FINISHED", "Pipeline completed");
		
		int result_count = results.GetCount();
		for(int i = 0; i < results.GetCount(); i++) {
			if(results[i].confidence > 0.5)
				log.detections_good++;
			else
				log.detections_missing++;
		}
		
		if(source.IsEmpty()) {
			log.status = "ERROR";
			log.warnings = "Failed to load image";
		}
		else if(result_count == 0) {
			log.status = "WARN";
			log.warnings = "No slots recognized";
		}
		else if(!log.script_error.IsEmpty()) {
			log.status = "ERROR";
			log.warnings = "Script error: " + log.script_error;
		}
		else if(log.detections_missing > 0) {
			log.status = "WARN";
			log.warnings = Format("%d detections with low confidence", log.detections_missing);
		}
		else {
			log.status = "OK";
		}

		{
			Mutex::Lock __(recog_lock_);
			completed_source_image_ = source;
			
			log.results <<= results;
			log.meta <<= meta;
			
			completed_results_ = pick(results);
			completed_meta_ = pick(meta);
			completed_flat_index_ = job.flat_index;
			completed_frame_seq_ = job.frame_seq;
			completed_log_record_ = pick(log);
			has_completed_job_ = true;
		}
		FrameRecTrace(Format("worker: done frame_seq=%d flat=%d results=%d",
		                     job.frame_seq, job.flat_index, result_count));
		PostCallback(THISBACK(OnRecognitionReady));
	});
}


void FrameRecognizerWindow::OnRecognitionReady() {
	if(closing_)
		return;

	Vector<SlotResult> results;
	VectorMap<String, String> meta;
	Image source;
	Image rendered;
	int flat_index = -1;
	int frame_seq = -1;
	ProcessingLogRecord log;

	RecognitionJob next_job;
	bool has_next = false;

	{
		Mutex::Lock __(recog_lock_);
		recog_busy_ = false;
		if(has_completed_job_) {
			source = completed_source_image_;
			// rendered will be created below
			results = pick(completed_results_);
			meta = pick(completed_meta_);
			flat_index = completed_flat_index_;
			frame_seq = completed_frame_seq_;
			log = pick(completed_log_record_);
			completed_source_image_.Clear();
			completed_display_image_.Clear();
			has_completed_job_ = false;
			completed_flat_index_ = -1;
			completed_frame_seq_ = -1;
		}
		if(has_pending_job_) {
			next_job = pending_job_;
			has_pending_job_ = false;
			has_next = true;
			recog_busy_ = true;
		}
	}
	
	if(flat_index < 0) return; // Nothing to process

	// Create overlay in main thread to avoid GuiLock deadlock
	uint64 t_overlay_start = GetTickCount();
	if(!source.IsEmpty())
		rendered = RenderOverlayImage(source, results, video_feed_.IsShowingOffsets());
	log.t_overlay_ms = (double)(GetTickCount() - t_overlay_start);
	log.AddStep("OVERLAY", log.t_overlay_ms, rendered.IsEmpty() ? "ERROR" : "OK", rendered.IsEmpty() ? "Render failed" : "Overlay rendered successfully");

	if(flat_index >= 0) {
		if(!meta.IsEmpty())
			results_cache_.GetAdd(flat_index) = clone(meta);
		slot_results_cache_.GetAdd(flat_index) = clone(results);
	}

	// Update Log UI
	if(log.frame_seq >= 0) {
		log_history_.Add(pick(log));
		const ProcessingLogRecord& h = log_history_.Top();

		log_list_.Add(h.frame_seq,
		              h.flat_index >= 0 ? AsString(h.flat_index) : String(),
		              h.image_name,
		              h.status,
		              Format("%.1f", h.t_total_ms),
		              Format("%.1f", h.t_recognize_ms),
		              Format("%.1f", h.t_script_ms),
		              Format("%d/%d", h.detections_good, h.detections_good + h.detections_missing),
		              h.warnings,
		              h.script_output,
		              h.script_error);

		if(log_list_.GetCount() > 1000) {
			log_list_.Remove(0);
			log_history_.Remove(0);
		}
		log_list_.ScrollEnd();

		// Update Stats
		stats_processed_++;
		if(h.status == "ERROR")
			stats_failed_++;

		stats_avg_total_ms_ = (stats_avg_total_ms_ * (stats_processed_ - 1) + h.t_total_ms) / stats_processed_;
		stats_avg_recog_ms_ = (stats_avg_recog_ms_ * (stats_processed_ - 1) + h.t_recognize_ms) / stats_processed_;
		stats_max_total_ms_ = max(stats_max_total_ms_, h.t_total_ms);
		stats_last_status_ = h.status;

		if(dock_log_) {
			dock_log_->Title(Format("Processing Log - Frames: %d, Failed: %d, Avg: %.1fms, Last: %s",
			                        stats_processed_, stats_failed_, stats_avg_total_ms_, stats_last_status_));
		}
	}

	// Only update visible UI overlays for the currently shown frame.
	bool is_current = (frame_seq == current_frame_seq_);
	if(slideshow_mode_ && flat_index >= 0 && flat_index == current_flat_index_)
		is_current = true;

	if(is_current) {
		if(log_history_.GetCount() > 0)
			displayed_log_ <<= log_history_.Top(); // Keep for steps tab selection

		if(!source.IsEmpty() && !rendered.IsEmpty())
			video_feed_.SetRenderedImage(source, rendered);
		else if(!source.IsEmpty())
			video_feed_.SetCurrentImage(source);
		UpdateOverlay(results);
		UpdateDetectionsDock(results);

		PopulateStepsTree(displayed_log_);
	}
	FrameRecTrace(Format("ui: ready frame_seq=%d current_seq=%d flat=%d shown=%s results=%d",
	                     frame_seq, current_frame_seq_, flat_index,
	                     frame_seq == current_frame_seq_ ? "yes" : "no",
	                     results.GetCount()));

	// Update image list with processing time
	if(flat_index >= 0 && flat_index < slideshow_images_.GetCount()) {
		// Convert processing time to string for display
		String processing_time = Format("%.1f ms", log.t_total_ms);
		images_list_.Set(flat_index, 2, processing_time);
	}

	// Update image list highlighting
	UpdateImageList();

	if(has_next && !closing_)
		StartRecognitionThread(next_job);
}

void FrameRecognizerWindow::PopulateStepsTree(const ProcessingLogRecord& log) {
	steps_list_.Clear();

	AnnSln sln_cfg;
	if(has_sln_cfg_)
		sln_cfg = sln_cfg_;

	Vector<ProcessingTreeNode> tree_nodes;
	BuildProcessingStepsTree(log, sln_cfg, tree_nodes);
	if(tree_nodes.IsEmpty())
		return;

	steps_list_.SetRoot(Null, tree_nodes[0].label);
	int root = 0;
	steps_list_.SetRowValue(root, 1, "0");
	steps_list_.SetRowValue(root, 2, tree_nodes[0].stage);
	steps_list_.SetRowValue(root, 3, Format("%.1f", tree_nodes[0].duration_ms));
	steps_list_.SetRowValue(root, 4, tree_nodes[0].status);
	steps_list_.SetRowValue(root, 5, tree_nodes[0].note);
	steps_list_.SetRowValue(root, 6, tree_nodes[0].step_id);

	VectorMap<int, int> step_idx_by_id;
	for(int i = 0; i < log.steps.GetCount(); i++) {
		int sid = log.steps[i].step_id;
		if(sid >= 0)
			step_idx_by_id.GetAdd(sid) = i;
	}

	VectorMap<String, String> slot_to_seq;
	Function<void(int, int, const String&)> AddChildren = [&](int tree_parent, int ui_parent, const String& parent_seq) {
		const Vector<int>& ch = tree_nodes[tree_parent].children;
		for(int ci = 0; ci < ch.GetCount(); ci++) {
			const ProcessingTreeNode& tn = tree_nodes[ch[ci]];
			int row = steps_list_.Add(ui_parent, Null, Null, tn.label);
			String seq = parent_seq + "." + AsString(ci);

			steps_list_.SetRowValue(row, 1, seq);
			steps_list_.SetRowValue(row, 2, tn.stage);
			steps_list_.SetRowValue(row, 3, Format("%.1f", tn.duration_ms));
			steps_list_.SetRowValue(row, 4, tn.status);
			steps_list_.SetRowValue(row, 5, tn.note);
			steps_list_.SetRowValue(row, 6, tn.step_id);

			if(tn.step_id >= 0) {
				int q = step_idx_by_id.Find(tn.step_id);
				if(q >= 0) {
					const ProcessingStepRecord& ps = log.steps[step_idx_by_id[q]];
					if(!ps.slot_id.IsEmpty() &&
					   (ps.stage == "RECOGNIZE" || ps.stage == "LOAD" || ps.stage == "OCR_PROCESS" ||
					    ps.stage == "PRESENT" || ps.stage == "VISIBLE")) {
						slot_to_seq.GetAdd(ps.slot_id) = seq;
					}
				}
			}

			if(!tn.is_leaf)
				AddChildren(ch[ci], row, seq);
		}
	};

	AddChildren(0, root, "0");

	for(int i = 0; i < steps_list_.GetCount(); i++) {
		Value v = steps_list_.GetRowValue(i, 6);
		if(IsNull(v))
			continue;
		int step_id = (int)v;
		if(step_id < 0)
			continue;

		int q = step_idx_by_id.Find(step_id);
		if(q < 0)
			continue;
		const ProcessingStepRecord& ps = log.steps[step_idx_by_id[q]];
		if(ps.gate_slot_id.IsEmpty() || !ps.gate_status.StartsWith("blocked"))
			continue;

		String gate_seq = slot_to_seq.Get(ps.gate_slot_id, "");
		if(gate_seq.IsEmpty())
			continue;
		String seq = steps_list_.GetRowValue(i, 1);
		seq << " (blocked by " << gate_seq << ")";
		steps_list_.SetRowValue(i, 1, seq);
	}

	steps_list_.OpenDeep(0);
	if(steps_list_.GetCount() > 0)
		steps_list_.SetCursor(0);
}

void FrameRecognizerWindow::StopRecognitionThread() {
	{
		Mutex::Lock __(recog_lock_);
		has_pending_job_ = false;
	}
	if(recog_thread_.IsOpen())
		recog_thread_.Wait();
	{
		Mutex::Lock __(recog_lock_);
		recog_busy_ = false;
		has_completed_job_ = false;
		completed_source_image_.Clear();
		completed_display_image_.Clear();
		completed_results_.Clear();
		completed_meta_.Clear();
		completed_flat_index_ = -1;
		completed_frame_seq_ = -1;
	}
}

void FrameRecognizerWindow::UpdateOverlay(const Vector<SlotResult>& results) {
	video_feed_.SetSlotResults(results);
}

void FrameRecognizerWindow::UpdateDetectionsDock(const Vector<SlotResult>& results) {
	detections_list_.Clear();
	for(int i = 0; i < results.GetCount(); i++) {
		const SlotResult& r = results[i];
		String val = ResultDisplayValue(r);
		String conf = r.confidence > 0.0 ? Format("%.0f", r.confidence * 100.0) : String();
		String off;
		if(fabs(r.offset_dx) >= 0.5 || fabs(r.offset_dy) >= 0.5)
			off = SafeFormatInt((int)round(r.offset_dx)) + "," + SafeFormatInt((int)round(r.offset_dy));
		detections_list_.Add(r.slot_id, val, conf, off, r.gate_slot_id, r.gate_status);
	}
}

void FrameRecognizerWindow::ShowCurrentImage() {
	if(slideshow_images_.IsEmpty())
		return;
	if(slideshow_idx_ < 0 || slideshow_idx_ >= slideshow_images_.GetCount())
		slideshow_idx_ = 0;

	current_flat_index_ = slideshow_idx_;
	Value img_rec = slideshow_images_[slideshow_idx_];
	String img_path = ResolveImagePath(sln_dir_, sln_images_dir_, img_rec);

	if(rec_loaded_) {
		current_frame_seq_++;
		RecognitionJob job;
		job.image_path = img_path;
		job.flat_index = current_flat_index_;
		job.frame_seq = current_frame_seq_;
		job.show_offsets = video_feed_.IsShowingOffsets();
		FrameRecTrace(Format("slideshow: queue idx=%d frame_seq=%d path='%s'",
		                     slideshow_idx_, job.frame_seq, job.image_path));
		QueueRecognition(job);
	}

	Title(Format("Frame Recognizer - %d / %d",
	             slideshow_idx_ + 1,
	             slideshow_images_.GetCount()));
	
	// Update the image list to highlight the current image
	UpdateImageList();
}

void FrameRecognizerWindow::OnSlideshowTimer() {
	if(!slideshow_running_ || slideshow_images_.IsEmpty())
		return;
	{
		Mutex::Lock __(recog_lock_);
		if(recog_busy_) {
			SetTimeCallback(-80, THISBACK(OnSlideshowTimer), TIMEID_SLIDESHOW);
			return;
		}
	}
	if(slideshow_idx_ < 0 || slideshow_idx_ >= slideshow_images_.GetCount())
		slideshow_idx_ = 0;
	else
		slideshow_idx_ = (slideshow_idx_ + 1) % slideshow_images_.GetCount();
	ShowCurrentImage();
	
	// Update the image list to highlight the current image
	UpdateImageList();
	
	SetTimeCallback(-80, THISBACK(OnSlideshowTimer), TIMEID_SLIDESHOW);
}

void FrameRecognizerWindow::OnPlayPause() {
	if(!slideshow_mode_)
		return;
	slideshow_running_ = !slideshow_running_;
	KillTimeCallback(TIMEID_SLIDESHOW);
	if(slideshow_running_)
		SetTimeCallback(-80, THISBACK(OnSlideshowTimer), TIMEID_SLIDESHOW);
	toolbar_.Refresh();
}

void FrameRecognizerWindow::OnSaveResults() {
	if(!slideshow_mode_) {
		PromptOK("Save Results is only available in slideshow mode.");
		return;
	}
	if(annprj_path_.IsEmpty()) {
		PromptOK("annprj path is empty.");
		return;
	}
	if(IsNull(annprj_root_) || !IsValueMap(annprj_root_)) {
		PromptOK("Invalid annprj data in memory.");
		return;
	}

	Value datasets = annprj_root_["datasets"];
	if(!IsValueArray(datasets)) {
		PromptOK("No datasets in annprj.");
		return;
	}

	ValueArray new_datasets;
	int flat_index = 0;
	for(int di = 0; di < datasets.GetCount(); di++) {
		Value ds = datasets[di];
		Value images = ds["images"];
		ValueArray new_images;
		int img_count = IsValueArray(images) ? images.GetCount() : 0;
		for(int ii = 0; ii < img_count; ii++) {
			Value img_rec = images[ii];
			ValueMap new_rec;
			if(IsValueMap(img_rec)) {
				ValueMap orig = img_rec;
				for(int fi = 0; fi < orig.GetCount(); fi++)
					new_rec.Add(orig.GetKey(fi), orig[fi]);
			}

			VectorMap<String, String> image_metadata;
			Value meta_keys = img_rec["image_meta_keys"];
			Value meta_vals = img_rec["image_meta_values"];
			if(IsValueArray(meta_keys) && IsValueArray(meta_vals)) {
				for(int mi = 0; mi < min(meta_keys.GetCount(), meta_vals.GetCount()); mi++) {
					String key = meta_keys[mi].ToString();
					if(!key.IsEmpty())
						image_metadata.GetAdd(key) = meta_vals[mi].ToString();
				}
			}

			int q = results_cache_.Find(flat_index);
			if(q >= 0) {
				const VectorMap<String, String>& cached = results_cache_[q];
				for(int mi = 0; mi < cached.GetCount(); mi++)
					image_metadata.GetAdd(cached.GetKey(mi)) = cached[mi];
			}

			ValueArray out_keys;
			ValueArray out_vals;
			for(int mi = 0; mi < image_metadata.GetCount(); mi++) {
				out_keys.Add(image_metadata.GetKey(mi));
				out_vals.Add(image_metadata[mi]);
			}
			new_rec.GetAdd("image_meta_keys") = out_keys;
			new_rec.GetAdd("image_meta_values") = out_vals;

			new_images.Add(new_rec);
			flat_index++;
		}

		ValueMap new_ds;
		if(IsValueMap(ds)) {
			ValueMap orig = ds;
			for(int fi = 0; fi < orig.GetCount(); fi++)
				new_ds.Add(orig.GetKey(fi), orig[fi]);
		}
		new_ds.GetAdd("images") = new_images;
		new_datasets.Add(new_ds);
	}

	ValueMap new_root;
	if(IsValueMap(annprj_root_)) {
		ValueMap orig = annprj_root_;
		for(int fi = 0; fi < orig.GetCount(); fi++)
			new_root.Add(orig.GetKey(fi), orig[fi]);
	}
	new_root.GetAdd("datasets") = new_datasets;

	if(!SaveFile(annprj_path_, StoreAsJson(new_root, true))) {
		PromptOK("Failed to save annprj.");
		return;
	}

	annprj_root_ = new_root;
	PromptOK(Format("Saved results for %d images.", results_cache_.GetCount()));
	
	// Update the image list to reflect saved results
	UpdateImageList();
}

void FrameRecognizerWindow::OnModelSetChanged() {
	String model_set = model_set_drop_.GetValue();
	if(model_set.IsEmpty())
		return;
	if(!LoadRecognizer(model_set)) {
		PromptOK("Failed to load recognizer/model set: " + model_set);
		return;
	}
	Image img = video_feed_.GetLatestFrame();
	if(!img.IsEmpty())
		RunOnCurrentFrame(img);
	
	// Refresh the image list when model set changes
	UpdateImageList();
}

void FrameRecognizerWindow::OnOffsetModeChanged() {
	{
		Mutex::Lock __(recognizer_lock_);
		recognizer_.SetOffsetMode(StringToOffsetMode(offset_mode_drop_.GetData()));
	}
	Image img = video_feed_.GetLatestFrame();
	if(!img.IsEmpty())
		RunOnCurrentFrame(img);
}

void FrameRecognizerWindow::OnBoolPolicyChanged() {
	{
		Mutex::Lock __(recognizer_lock_);
		recognizer_.SetBoolGatePolicy(StringToBoolGatePolicy(bool_policy_drop_.GetData()));
	}
	Image img = video_feed_.GetLatestFrame();
	if(!img.IsEmpty())
		RunOnCurrentFrame(img);
}

void FrameRecognizerWindow::OnToggleOffsets() {
	video_feed_.ShowOffsets(!video_feed_.IsShowingOffsets());
	toolbar_.Set(THISBACK(BuildToolbar));
	Image img = video_feed_.GetLatestFrame();
	if(!img.IsEmpty())
		RunOnCurrentFrame(img);
}

void FrameRecognizerWindow::OnImageListContextMenu(Bar& bar) {
	if(!slideshow_mode_ || images_list_.GetCursor() < 0)
		return;
	bar.Add("Process Selected Image", THISBACK(OnProcessSelectedImage));
}

void FrameRecognizerWindow::OnImageListCursor() {
	if(!slideshow_mode_)
		return;
	int row = images_list_.GetCursor();
	if(row < 0 || row >= slideshow_images_.GetCount())
		return;

	Value img_rec = slideshow_images_[row];
	String img_path = ResolveImagePath(sln_dir_, sln_images_dir_, img_rec);

	Image source = StreamRaster::LoadFileAny(img_path);
	if(source.IsEmpty())
		return;

	// Check if we have cached slot results for this image
	Vector<SlotResult> results;
	bool has_results = false;
	{
		Mutex::Lock __(recog_lock_);
		int q = slot_results_cache_.Find(row);
		if(q >= 0) {
			results <<= clone(slot_results_cache_[q]);
			has_results = true;
		}
	}

	if(has_results) {
		Image rendered = RenderOverlayImage(source, results, video_feed_.IsShowingOffsets());
		video_feed_.SetRenderedImage(source, rendered);
		UpdateOverlay(results);
		UpdateDetectionsDock(results);
	} else {
		video_feed_.SetCurrentImage(source);
		UpdateOverlay(results);
		UpdateDetectionsDock(results);
	}
}

void FrameRecognizerWindow::OnProcessSelectedImage() {
	if(!slideshow_mode_ || !rec_loaded_)
		return;
	int row = images_list_.GetCursor();
	if(row < 0 || row >= slideshow_images_.GetCount())
		return;

	{
		Mutex::Lock __(image_proc_lock_);
		if(image_proc_busy_) {
			PromptOK("An image is already being processed.");
			return;
		}
		image_proc_busy_ = true;
		image_proc_row_ = row;
	}

	Value img_rec = slideshow_images_[row];
	String img_path = ResolveImagePath(sln_dir_, sln_images_dir_, img_rec);
	int frame_seq = current_frame_seq_;
	bool show_offsets = video_feed_.IsShowingOffsets();

	FrameRecTrace(Format("manual process: row=%d path='%s'", row, img_path));

	image_proc_thread_.Run([=] {
		uint64 t0 = GetTickCount();
		ProcessingLogRecord log;
		log.frame_seq = frame_seq;
		log.flat_index = row;
		log.image_name = GetFileName(img_path);
		log.image_path = img_path;

		uint64 t_load_start = GetTickCount();
		Image source = StreamRaster::LoadFileAny(img_path);
		log.t_load_ms = (double)(GetTickCount() - t_load_start);
		
		if(source.IsEmpty()) {
			log.AddStep("LOAD", log.t_load_ms, "ERROR", "Failed to load/decode image", String(), String(), false, "Path: " + img_path);
			log.status = "ERROR";
		} else {
			log.AddStep("LOAD", log.t_load_ms, "OK", Format("%d x %d", source.GetWidth(), source.GetHeight()), String(), String(), false, "Path: " + img_path);
		}

		Vector<SlotResult> results;
		VectorMap<String, String> meta;
		Image rendered;

		if(!closing_ && !source.IsEmpty()) {
			Mutex::Lock __(recognizer_lock_);
			if(rec_loaded_) {
				uint64 t_rec_start = GetTickCount();
				results = recognizer_.Recognize(source, &log);
				log.t_recognize_ms = (double)(GetTickCount() - t_rec_start);
				log.AddStep("RecognizeSummary", log.t_recognize_ms, "OK", Format("%d results", results.GetCount()));

				uint64 t_script_start = GetTickCount();
				meta = script_.Run(results);
				log.t_script_ms = (double)(GetTickCount() - t_script_start);
				log.script_output = script_.GetLastOutput();
				log.script_error = script_.GetLastError();
				
				if(!log.script_error.IsEmpty()) {
					log.AddStep("SCRIPT", log.t_script_ms, "ERROR", "Script execution failed", String(), String(), false, log.script_error);
				} else {
					log.AddStep("SCRIPT", log.t_script_ms, "OK", Format("%d meta keys", meta.GetCount()));
				}
			} else {
				log.AddStep("RECOGNIZE", 0, "SKIPPED", "Recognizer not loaded");
			}
		}

		if(!closing_ && !source.IsEmpty()) {
			uint64 t_overlay_start = GetTickCount();
			rendered = RenderOverlayImage(source, results, show_offsets);
			log.t_overlay_ms = (double)(GetTickCount() - t_overlay_start);
			log.AddStep("OVERLAY", log.t_overlay_ms, rendered.IsEmpty() ? "ERROR" : "OK", rendered.IsEmpty() ? "Render failed" : "Overlay rendered successfully");
		}

		log.t_total_ms = (double)(GetTickCount() - t0);
		log.AddStep("TOTAL", log.t_total_ms, "FINISHED", "Pipeline completed");
		log.detections_good = 0;
		log.detections_missing = 0;
		for(int i = 0; i < results.GetCount(); i++) {
			if(results[i].confidence > 0.5)
				log.detections_good++;
			else
				log.detections_missing++;
		}

		if(source.IsEmpty()) {
			log.status = "ERROR";
			log.warnings = "Failed to load image";
		}
		else if(results.GetCount() == 0) {
			log.status = "WARN";
			log.warnings = "No slots recognized";
		}
		else if(!log.script_error.IsEmpty()) {
			log.status = "ERROR";
			log.warnings = "Script error: " + log.script_error;
		}
		else if(log.detections_missing > 0) {
			log.status = "WARN";
			log.warnings = Format("%d detections with low confidence", log.detections_missing);
		}
		else {
			log.status = "OK";
		}

		// Update the image list cell for processing time and UI — only if row still matches
		{
			Mutex::Lock __(image_proc_lock_);
			if(!closing_ && image_proc_row_ == row) {
				String pt = Format("%.1f ms", log.t_total_ms);

				// Cache slot results
				if(!results.IsEmpty()) {
					Mutex::Lock lock2(recog_lock_);
					slot_results_cache_.GetAdd(row) = clone(results);
					if(!meta.IsEmpty())
						results_cache_.GetAdd(row) = clone(meta);
				}

				// Capture images for UI update
				Image src = source;
				Image disp = rendered;
				ProcessingLogRecord* lp = new ProcessingLogRecord();
				*lp <<= log;

				PostCallback([=]() {
					images_list_.Set(row, 2, pt);
					if(!src.IsEmpty()) {
						if(!disp.IsEmpty())
							video_feed_.SetRenderedImage(src, disp);
						else
							video_feed_.SetCurrentImage(src);
						UpdateOverlay(lp->results);
						UpdateDetectionsDock(lp->results);
						
						// Update steps tab
						displayed_log_ <<= *lp;
						PopulateStepsTree(displayed_log_);
						
						delete lp;
					} else {
						delete lp;
					}
				});
			}
			image_proc_busy_ = false;
			image_proc_row_ = -1;
		}
	});
}

void FrameRecognizerWindow::MainMenu(Bar& bar) {
	bar.Sub("File", [=](Bar& b) {
		b.Add("Save Detections to annprj", THISBACK(OnSaveResults));
		b.Separator();
		b.Add("Save Diagnostic Log (JSON)...", THISBACK(OnSaveLog));
		b.Add("Load Diagnostic Log (JSON)...", THISBACK(OnLoadLog));
		b.Separator();
		b.Add("Exit", [=] { Break(); });
	});
	bar.Sub("Playback", [=](Bar& b) {
		b.Add(slideshow_running_ ? "Pause" : "Play", THISBACK(OnPlayPause));
	});
	bar.Sub("Image", [=](Bar& b) {
		b.Add("Process Selected Image", THISBACK(OnProcessSelectedImage))
		 .Enable(slideshow_mode_ && images_list_.GetCursor() >= 0);
	});
	bar.Sub("Model", [=](Bar& b) {
		b.Add("Reload Selected", THISBACK(OnModelSetChanged));
	});
	bar.Sub("Windows", [=](Bar& sub) { DockWindowMenu(sub); });
}

void FrameRecognizerWindow::BuildToolbar(Bar& bar) {
	bar.Add(slideshow_running_ ? "Pause" : "Play", THISBACK(OnPlayPause));
	bar.Add("Save", CtrlImg::save(), THISBACK(OnSaveResults)).Help("Save detections to .annprj");
	bar.Add("Save Log", THISBACK(OnSaveLog)).Help("Save current frame diagnostic log to JSON");
	bar.Add("Load Log", THISBACK(OnLoadLog)).Help("Load diagnostic log from JSON");
	bar.Separator();
	bar.Add("Show Offsets", CtrlImg::cross(), THISBACK(OnToggleOffsets)).Check(video_feed_.IsShowingOffsets());
	bar.Separator();
	bar.Add(model_set_drop_, 120);
	bar.Separator();
	bar.Add(lbl_offset_mode_, 50);
	bar.Add(offset_mode_drop_, 100);
	bar.Separator();
	bar.Add(lbl_bool_policy_, 80);
	bar.Add(bool_policy_drop_, 100);
}

void ImageLoadStep::Run(const Image& input, Vector<SlotResult>& results, VectorMap<String, String>& meta, ProcessingLogRecord& log) {
	uint64 t0 = GetTickCount();
	if(!input.IsEmpty()) {
		image = input;
		status = "OK";
		note = "Input image provided directly";
	} else if(!image_path.IsEmpty()) {
		image = StreamRaster::LoadFileAny(image_path);
		if(image.IsEmpty()) {
			status = "ERROR";
			note = "Failed to load/decode: " + image_path;
		} else {
			status = "OK";
			note = Format("%dx%d", image.GetWidth(), image.GetHeight());
		}
	} else {
		status = "ERROR";
		note = "No image source provided";
	}
	duration_ms = (double)(GetTickCount() - t0);
	log.t_load_ms = duration_ms;
}

void RecognizerStep::Run(const Image& input, Vector<SlotResult>& results, VectorMap<String, String>& meta, ProcessingLogRecord& log) {
	if(!recognizer || input.IsEmpty()) {
		status = "SKIPPED";
		note = !recognizer ? "Recognizer not loaded" : "Empty input image";
		return;
	}
	uint64 t0 = GetTickCount();
	input_image = input;
	results = recognizer->Recognize(input, &log);
	this->results <<= results;
	duration_ms = (double)(GetTickCount() - t0);
	status = "OK";
	note = Format("%d detections", results.GetCount());
	log.t_recognize_ms = duration_ms;
}

void OverlayRenderStep::Run(const Image& input, Vector<SlotResult>& results, VectorMap<String, String>& meta, ProcessingLogRecord& log) {
	if(input.IsEmpty()) {
		status = "SKIPPED";
		note = "Empty input image";
		return;
	}
	uint64 t0 = GetTickCount();
	input_image = input;
	this->results <<= results;
	output_image = RenderOverlayImage(input, results, false);
	duration_ms = (double)(GetTickCount() - t0);
	if(output_image.IsEmpty()) {
		status = "ERROR";
		note = "RenderOverlayImage returned empty";
	} else {
		status = "OK";
		note = Format("Rendered %d detections", results.GetCount());
	}
	log.t_overlay_ms = duration_ms;
}

void SummaryStep::Run(const Image& input, Vector<SlotResult>& results, VectorMap<String, String>& meta, ProcessingLogRecord& log) {
	uint64 t0 = GetTickCount();
	this->results <<= results;
	this->meta <<= meta;
	
	int good = 0;
	int missing = 0;
	for(const auto& r : results) {
		if(r.confidence > 0.5) good++;
		else missing++;
	}
	duration_ms = (double)(GetTickCount() - t0);
	status = "OK";
	note = Format("Good: %d, Missing: %d", good, missing);
	
	log.detections_good = good;
	log.detections_missing = missing;
}

void FrameRecognizerWindow::PopulateLabelAInspector(ProcessingStepRecord& ps,
                                                    TemplateMatchMethod method) {
	TemplateMatchMethod pipeline_method = LabelAMatchVisualizer::MethodFromName(ps.cv_match_method);
	Image response_map;
	if(method == pipeline_method || ps.cv_input_crop.IsEmpty()) {
		// Use stored runtime response map
		response_map = ps.cv_response_map;
	} else {
		// Re-run matching with the override method on the stored input crop
		ByteMat input_mat;
		{
			const Image& crop = ps.cv_input_crop;
			Size sz = crop.GetSize();
			input_mat.SetSize(sz.cx, sz.cy, 1);
			for(int y = 0; y < sz.cy; y++) {
				const RGBA* s = crop[y];
				for(int x = 0; x < sz.cx; x++)
					input_mat.data[y * sz.cx + x] = s[x].r;
			}
		}
		bool sqdiff = (method == TM_SQDIFF || method == TM_SQDIFF_NORMED);
		FloatMat best_res;
		double best_score = sqdiff ? 1e18 : -1e18;
		for(int k = 0; k < ps.cv_candidates.GetCount(); k++) {
			const CvCandidateRecord& cr = ps.cv_candidates[k];
			if(cr.crop_image.IsEmpty()) continue;
			const Image& ti = cr.crop_image;
			Size tsz = ti.GetSize();
			ByteMat tmpl;
			tmpl.SetSize(tsz.cx, tsz.cy, 1);
			for(int y = 0; y < tsz.cy; y++) {
				const RGBA* s = ti[y];
				for(int x = 0; x < tsz.cx; x++)
					tmpl.data[y * tsz.cx + x] = s[x].r;
			}
			if(input_mat.cols < tmpl.cols || input_mat.rows < tmpl.rows) continue;
			FloatMat res;
			MatchTemplate(input_mat, tmpl, res, method);
			double mn, mx;
			MinMaxLoc(res, &mn, &mx, nullptr, nullptr);
			double s = sqdiff ? mn : mx;
			if(sqdiff ? (s < best_score) : (s > best_score)) {
				best_score = s;
				best_res = pick(res);
			}
		}
		if(!best_res.IsEmpty() && best_res.channels == 1) {
			double min_v = 0, max_v = 0;
			MinMaxLoc(best_res, &min_v, &max_v, nullptr, nullptr);
			double span = max_v - min_v;
			if(span < 1e-20) span = 1.0;
			ImageBuffer ib(best_res.cols, best_res.rows);
			for(int y = 0; y < best_res.rows; y++) {
				RGBA* d = ib[y];
				for(int x = 0; x < best_res.cols; x++) {
					double v = best_res.data[y * best_res.cols + x];
					double t = (v - min_v) / span;
					if(sqdiff) t = 1.0 - t;
					int g = (int)clamp(t * 255.0, 0.0, 255.0);
					d[x] = {(byte)g, (byte)g, (byte)g, 255};
				}
			}
			response_map = ib;
		}
	}
	steps_label_a_vis_.SetStep(&ps, response_map);

	// Populate candidates ArrayCtrl
	steps_label_a_candidates_.Clear();
	for(int i = 0; i < ps.cv_candidates.GetCount(); i++) {
		const CvCandidateRecord& cr = ps.cv_candidates[i];
		String info = Format("score=%.4f zoom=%.2f rot=%.1f", cr.score, cr.zoom, cr.rot);
		steps_label_a_candidates_.Add(cr.crop_image, cr.class_name, info);
	}
}

END_UPP_NAMESPACE

void Upp::FrameRecognizerWindow::TestDumpXOffsetConvLayers(bool verbose) {
	if(slideshow_images_.GetCount() == 0) {
		Cout() << "No images found for testing. Checked annprj: " << annprj_path_ << "\n";
		return;
	}
	
	SeedRandom(GetSysTime().Get());
	slideshow_idx_ = Random(slideshow_images_.GetCount());
	Cout() << "TestDumpXOffsetConvLayers: Selected index " << slideshow_idx_ << " of " << slideshow_images_.GetCount() << "\n";
	ShowCurrentImage();
	
	int wait_ms = 0;
	while(true) {
		{
			Mutex::Lock __(recog_lock_);
			if(!recog_busy_ && has_completed_job_) break;
			if(!recog_busy_ && displayed_log_.steps.GetCount() > 0) break;
			if(wait_ms > 30000) { // 30 seconds
				Cout() << "Timeout waiting for recognition. recog_busy_=" << (int)recog_busy_ << " has_completed_job_=" << (int)has_completed_job_ << " displayed_log.steps=" << displayed_log_.steps.GetCount() << "\n";
				break;
			}
		}
		ProcessEvents();
		Sleep(100);
		wait_ms += 100;
	}
	
	if(has_completed_job_) {
		Cout() << "TestDumpXOffsetConvLayers: Manually calling OnRecognitionReady\n";
		OnRecognitionReady();
	}
	
	tabs.Set(1); // "Processing steps"
	ProcessEvents();
	
	int r_to_sel = -1;
	for(int i = 0; i < steps_list_.GetCount(); i++) {
		String name = steps_list_.GetRowValue(i, 0).ToString();
		if(name == "x_offset") {
			int parent_node = steps_list_.GetParent(i);
			if(parent_node >= 0 && steps_list_.GetRowValue(parent_node, 0).ToString().StartsWith("card1")) {
				int grandparent_node = steps_list_.GetParent(parent_node);
				if(grandparent_node >= 0 && steps_list_.GetRowValue(grandparent_node, 0).ToString().StartsWith("seat1")) {
					r_to_sel = i;
					break;
				}
			}
		}
	}
	
	if(r_to_sel >= 0) {
		steps_list_.SetCursor(r_to_sel);
		for(int i = 0; i < 20; i++) { // Increase wait for UI to catch up
			ProcessEvents();
			Sleep(50);
		}
		
		steps_nn_inspector_.Dump(verbose);
	} else {
		Cout() << "TestDumpXOffsetConvLayers: Target node 'Total/seat1/card1/x_offset' not found.\n";
		Cout() << "Available nodes:\n";
		for(int i = 0; i < steps_list_.GetCount(); i++) {
			int p = steps_list_.GetParent(i);
			Cout() << "  [" << i << "] parent=" << p << " label='" << steps_list_.GetRowValue(i, 0).ToString() << "' stage='" << steps_list_.GetRowValue(i, 2).ToString() << "'\n";
		}
	}
	
	StopRecognitionThread();
}

#ifdef flagMAIN
GUI_APP_MAIN {
	using namespace Upp;
	CommandLineArguments cl;
	cl.AddPositional("sln_path", "Path to .annsln solution file", STRING_V);
	if(!cl.Parse(CommandLine())) {
		cl.PrintHelp();
		return;
	}
	if(cl.GetPositionalCount() == 1) {
		String sln_path = cl.GetPositional(0);
		if(ToLower(GetFileExt(sln_path)) == ".annsln") {
			AnnSln sln;
			if(sln.Load(sln_path)) {
				String sln_dir = GetFileDirectory(sln_path);
				String annprj = sln.annprj;
				if(!IsFullPath(annprj))
					annprj = NormalizePath(AppendFileName(sln_dir, annprj));
				FrameRecognizerWindow win;
				win.OpenSlideshow(sln, sln_dir, annprj, "pass1");
				win.Run();
				return;
			} else {
				PromptOK("Failed to load .annsln file: " + sln_path);
			}
		} else {
			PromptOK("Expected a .annsln file, got: " + sln_path);
		}
	}
	FrameRecognizerWindow win;
	win.Run();
}
#endif
