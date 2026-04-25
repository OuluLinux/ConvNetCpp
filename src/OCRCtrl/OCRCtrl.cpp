#include "OCRCtrl.h"
#include <OCR/Segmentation.h>

namespace Upp {

namespace OCR {

// --- ImagePreviewCtrl Implementation ---

ImagePreviewCtrl::ImagePreviewCtrl() {
	zoom = 1.0;
	offset = Point(0, 0);
	show_bounds = true;
	show_labels = true;
	show_grid = false;
	show_projection = false;
	selected_segment = -1;
	selected_char = -1;
	dragging = false;
	panning = false;
	segments = nullptr;
	
	BackPaint();
}

void ImagePreviewCtrl::SetImage(const Image& img) {
	image = img;
	Refresh();
}

void ImagePreviewCtrl::SetSegments(Vector<OCRSegment>* segs) {
	segments = segs;
	Refresh();
}

void ImagePreviewCtrl::ZoomIn() {
	SetZoom(zoom * 1.25);
}

void ImagePreviewCtrl::ZoomOut() {
	SetZoom(zoom * 0.8);
}

void ImagePreviewCtrl::ZoomFit() {
	if (image.IsEmpty()) return;
	Size sz = GetSize();
	Size img_sz = image.GetSize();
	
	double zx = (double)sz.cx / img_sz.cx;
	double zy = (double)sz.cy / img_sz.cy;
	
	SetZoom(min(zx, zy) * 0.95);
	offset = Point(sz.cx / 2, sz.cy / 2);
	Refresh();
}

void ImagePreviewCtrl::SetZoom(double z) {
	zoom = minmax(z, 0.05, 50.0);
	Refresh();
}

void ImagePreviewCtrl::SelectSegment(int index) {
	selected_segment = index;
	Refresh();
}

void ImagePreviewCtrl::SelectChar(int char_index) {
	selected_char = char_index;
	Refresh();
}

void ImagePreviewCtrl::Paint(Draw& w) {
	Size sz = GetSize();
	w.DrawRect(sz, SColorPaper());
	
	if (image.IsEmpty()) {
		w.DrawText(sz.cx / 2 - 50, sz.cy / 2, "No image selected", StdFont().Italic());
		return;
	}
	
	if (show_grid) DrawGrid(w);
	
	Size img_sz = image.GetSize();
	Point p = ImageToScreen(Point(0, 0));
	
	w.DrawImage(p.x, p.y, (int)(img_sz.cx * zoom), (int)(img_sz.cy * zoom), image);
	
	if (show_projection) DrawProjection(w);
	
	DrawOverlays(w);
	
	if (dragging) {
		w.DrawRect(drag_rect, LtRed());
		DrawFrame(w, drag_rect, Red());
	}
}

Point ImagePreviewCtrl::ImageToScreen(Point p) const {
	Size sz = GetSize();
	Size img_sz = image.GetSize();
	
	int x = (int)((p.x - img_sz.cx / 2.0) * zoom) + offset.x;
	int y = (int)((p.y - img_sz.cy / 2.0) * zoom) + offset.y;
	
	return Point(x, y);
}

Point ImagePreviewCtrl::ScreenToImage(Point p) const {
	Size img_sz = image.GetSize();
	
	int x = (int)((p.x - offset.x) / zoom + img_sz.cx / 2.0);
	int y = (int)((p.y - offset.y) / zoom + img_sz.cy / 2.0);
	
	return Point(x, y);
}

void ImagePreviewCtrl::DrawGrid(Draw& w) {
	Size sz = GetSize();
	int step = (int)(10 * zoom);
	if (step < 5) return;
	
	for (int x = offset.x % step; x < sz.cx; x += step)
		w.DrawText(x, 0, "|", StdFont(10), SColorDisabled());
	for (int y = offset.y % step; y < sz.cy; y += step)
		w.DrawText(0, y, "-", StdFont(10), SColorDisabled());
}

void ImagePreviewCtrl::DrawOverlays(Draw& w) {
	if (!segments || !show_bounds) return;
	
	for (int i = 0; i < segments->GetCount(); i++) {
		const OCRSegment& seg = (*segments)[i];
		Point p1 = ImageToScreen(seg.bounds.TopLeft());
		Point p2 = ImageToScreen(seg.bounds.BottomRight());
		Rect r(p1, p2);
		
		Color col = (i == selected_segment) ? Blue() : LtBlue();
		DrawFrame(w, r, col);
		
		if (show_labels) {
			w.DrawText(r.left, r.top - 15, Format("Line %d: %s", i, seg.text), StdFont(), col);
		}
		
		for (int j = 0; j < seg.chars.GetCount(); j++) {
			const OCRChar& ch = seg.chars[j];
			Point cp1 = ImageToScreen(ch.bounds.TopLeft());
			Point cp2 = ImageToScreen(ch.bounds.BottomRight());
			Rect cr(cp1, cp2);
			
			Color ccol = (i == selected_segment && j == selected_char) ? Red() : Green();
			DrawFrame(w, cr, ccol);
			
			if (show_labels && zoom > 0.5) {
				String label = GetClassName(ch.label);
				if (label.GetCount() > 7) label = Format("%c", ClassToChar(ch.label));
				w.DrawText(cr.left, cr.bottom, label, StdFont(10), ccol);
			}
		}
	}
}

void ImagePreviewCtrl::DrawProjection(Draw& w) {
	if (image.IsEmpty()) return;
	
	Image bw = OCR::GrayscaleOCR(image);
	Vector<int> horiz = OCR::HorizontalProjection(bw);
	
	int max_v = 1;
	for (int v : horiz) max_v = max(max_v, v);
	
	for (int y = 0; y < horiz.GetCount(); y++) {
		Point p1 = ImageToScreen(Point(0, y));
		Point p2 = ImageToScreen(Point((horiz[y] * 50) / max_v, y));
		w.DrawLine(p1, p2, 1, Gray());
	}
}

void ImagePreviewCtrl::MouseWheel(Point p, int zdelta, dword keyflags) {
	if (zdelta > 0) ZoomIn();
	else ZoomOut();
}

void ImagePreviewCtrl::LeftDown(Point p, dword keyflags) {
	if (keyflags & K_SHIFT) {
		panning = true;
		last_mouse_pos = p;
		SetCapture();
		return;
	}
	
	// Check for character selection first
	Point img_p = ScreenToImage(p);
	if (segments) {
		for (int i = 0; i < segments->GetCount(); i++) {
			for (int j = 0; j < (*segments)[i].chars.GetCount(); j++) {
				if ((*segments)[i].chars[j].bounds.Contains(img_p)) {
					SelectSegment(i);
					SelectChar(j);
					WhenCharSelected(i, j);
					return;
				}
			}
			if ((*segments)[i].bounds.Contains(img_p)) {
				SelectSegment(i);
				SelectChar(-1);
				WhenSegmentSelected(i);
				return;
			}
		}
	}
	
	dragging = true;
	drag_rect = Rect(p, p);
	last_mouse_pos = p;
	SetCapture();
}

void ImagePreviewCtrl::LeftUp(Point p, dword keyflags) {
	if (panning) {
		panning = false;
		ReleaseCapture();
		return;
	}
	
	if (dragging) {
		dragging = false;
		ReleaseCapture();
		
		Rect r = Rect(ScreenToImage(last_mouse_pos), ScreenToImage(p)).Normalized();
		if (r.Width() > 2 && r.Height() > 2) {
			if (segments) {
				bool inside = false;
				for (int i = 0; i < segments->GetCount(); i++) {
					if ((*segments)[i].bounds.Contains(r.CenterPoint())) {
						OCRChar& ch = (*segments)[i].chars.Add();
						ch.bounds = r;
						// Sort chars by x position
						Sort((*segments)[i].chars, [](const OCRChar& a, const OCRChar& b) {
							return a.bounds.left < b.bounds.left;
						});
						// Re-find the added char after sort to select it
						for (int j = 0; j < (*segments)[i].chars.GetCount(); j++) {
							if ((*segments)[i].chars[j].bounds == r) {
								SelectSegment(i);
								SelectChar(j);
								WhenCharSelected(i, j);
								break;
							}
						}
						inside = true;
						break;
					}
				}
				if (!inside) {
					OCRSegment& seg = segments->Add();
					seg.bounds = r;
					SelectSegment(segments->GetCount() - 1);
					SelectChar(-1);
					WhenSegmentSelected(segments->GetCount() - 1);
				}
			}
		}
		
		Refresh();
	}
}

void ImagePreviewCtrl::MiddleDown(Point p, dword keyflags) {
	panning = true;
	last_mouse_pos = p;
	SetCapture();
}

void ImagePreviewCtrl::MiddleUp(Point p, dword keyflags) {
	panning = false;
	ReleaseCapture();
}

void ImagePreviewCtrl::MouseMove(Point p, dword keyflags) {
	if (panning) {
		offset += p - last_mouse_pos;
		last_mouse_pos = p;
		Refresh();
	}
	else if (dragging) {
		drag_rect = Rect(last_mouse_pos, p).Normalized();
		Refresh();
	}
}

bool ImagePreviewCtrl::Key(dword key, int count) {
	if (key == K_DELETE) {
		if (segments) {
			if (selected_segment >= 0 && selected_segment < segments->GetCount()) {
				if (selected_char >= 0 && selected_char < (*segments)[selected_segment].chars.GetCount()) {
					(*segments)[selected_segment].chars.Remove(selected_char);
					selected_char = -1;
					Refresh();
					return true;
				} else {
					segments->Remove(selected_segment);
					selected_segment = -1;
					selected_char = -1;
					Refresh();
					return true;
				}
			}
		}
	}
	if (key == K_PLUS || key == K_ADD) { ZoomIn(); return true; }
	if (key == K_MINUS || key == K_SUBTRACT) { ZoomOut(); return true; }
	if (key == '0') { ZoomFit(); return true; }
	
	return false;
}

// --- SplitToolPanel Implementation ---

SplitToolPanel::SplitToolPanel() {
	project = nullptr;
	image_index = -1;
	segment_index = -1;
	
	Add(btn_auto.LeftPos(5, 120).TopPos(5, 25));
	btn_auto.SetLabel("Auto Segment Line");
	btn_auto << [this] { OnAuto(); };
	
	Add(btn_auto_all.LeftPos(130, 120).TopPos(5, 25));
	btn_auto_all.SetLabel("Auto Segment All");
	btn_auto_all << [this] { OnAutoAll(); };
	
	Add(btn_clear.LeftPos(255, 100).TopPos(5, 25));
	btn_clear.SetLabel("Clear Splits");
	btn_clear << [this] { OnClear(); };
	
	Add(lbl_min_w.LeftPos(5, 80).TopPos(40, 20));
	lbl_min_w.SetText("Min Width:");
	Add(min_w.LeftPos(90, 50).TopPos(40, 20));
	min_w = 2;
	
	Add(lbl_min_h.LeftPos(150, 80).TopPos(40, 20));
	lbl_min_h.SetText("Min Height:");
	Add(min_h.LeftPos(235, 50).TopPos(40, 20));
	min_h = 5;
}

void SplitToolPanel::AutoSegment() {
	if (!project || image_index < 0 || segment_index < 0) return;
	
	const Image& img = project->GetImage(image_index);
	OCRSegment& seg = project->GetSegments(image_index)[segment_index];
	
	Image line_img = CropImage(img, seg.bounds);
	Vector<Rect> char_rects = SegmentCharacters(line_img, ~min_w, ~min_h);
	
	seg.chars.Clear();
	for (const Rect& r : char_rects) {
		OCRChar& ch = seg.chars.Add();
		ch.bounds = Rect(seg.bounds.left + r.left, 
		                 seg.bounds.top + r.top, 
		                 seg.bounds.left + r.right, 
		                 seg.bounds.top + r.bottom);
	}
	
	project->SetModified();
	WhenChanged();
}

void SplitToolPanel::AutoSegmentAll() {
	if (!project) return;
	
	for (int i = 0; i < project->GetCount(); i++) {
		const Image& img = project->GetImage(i);
		Vector<OCRSegment>& segments = project->GetSegments(i);
		
		for (OCRSegment& seg : segments) {
			Image line_img = CropImage(img, seg.bounds);
			Vector<Rect> char_rects = SegmentCharacters(line_img, ~min_w, ~min_h);
			
			seg.chars.Clear();
			for (const Rect& r : char_rects) {
				OCRChar& ch = seg.chars.Add();
				ch.bounds = Rect(seg.bounds.left + r.left, 
				                 seg.bounds.top + r.top, 
				                 seg.bounds.left + r.right, 
				                 seg.bounds.top + r.bottom);
			}
		}
	}
	
	project->SetModified();
	WhenChanged();
}

void SplitToolPanel::ClearSplits() {
	if (!project || image_index < 0 || segment_index < 0) return;
	
	project->GetSegments(image_index)[segment_index].chars.Clear();
	project->SetModified();
	WhenChanged();
}

// --- CharPaletteCtrl Implementation ---

CharPaletteCtrl::CharPaletteCtrl() {
	for(int i = '0'; i <= '9'; i++) chars.Add(i);
	for(int i = 'A'; i <= 'Z'; i++) chars.Add(i);
	for(int i = 'a'; i <= 'z'; i++) chars.Add(i);
}

void CharPaletteCtrl::Paint(Draw& w) {
	Size sz = GetSize();
	w.DrawRect(sz, SColorPaper());
	
	int x = 5, y = 5;
	for (int c : chars) {
		Rect r(x, y, x + 20, y + 20);
		w.DrawRect(r, SColorFace());
		DrawFrame(w, r, SColorText());
		w.DrawText(x + 5, y + 2, Format("%c", c), StdFont());
		
		x += 25;
		if (x + 20 > sz.cx) {
			x = 5;
			y += 25;
		}
	}
}

void CharPaletteCtrl::LeftDown(Point p, dword keyflags) {
	int c = HitTest(p);
	if (c != -1) WhenCharSelected(c);
}

int CharPaletteCtrl::HitTest(Point p) {
	int x = 5, y = 5;
	Size sz = GetSize();
	for (int c : chars) {
		Rect r(x, y, x + 20, y + 20);
		if (r.Contains(p)) return c;
		
		x += 25;
		if (x + 20 > sz.cx) {
			x = 5;
			y += 25;
		}
	}
	return -1;
}

// --- CharLabelCtrl Implementation ---

CharLabelCtrl::CharLabelCtrl() {
	project = nullptr;
	image_index = -1;
	segment_index = -1;
	char_index = -1;
	
	Add(char_preview.LeftPos(5, 128).TopPos(5, 128));
	
	Add(lbl_label.LeftPos(140, 50).TopPos(5, 20));
	lbl_label.SetText("Label:");
	Add(label_edit.LeftPos(140, 50).TopPos(30, 25));
	label_edit.WhenAction = THISBACK(OnLabelEdit);
	
	Add(btn_prev.LeftPos(140, 60).TopPos(65, 25));
	btn_prev.SetLabel("Prev");
	btn_prev << [this] { OnPrev(); };
	
	Add(btn_next.LeftPos(210, 60).TopPos(65, 25));
	btn_next.SetLabel("Next");
	btn_next << [this] { OnNext(); };
	
	Add(btn_skip.LeftPos(140, 60).TopPos(95, 25));
	btn_skip.SetLabel("Skip");
	btn_skip << [this] { OnSkip(); };
	
	Add(btn_delete.LeftPos(210, 60).TopPos(95, 25));
	btn_delete.SetLabel("Delete");
	btn_delete.SetImage(CtrlImg::remove());
	btn_delete << [this] { OnDelete(); };
	
	Add(progress.LeftPos(5, 300).BottomPos(30, 15));
	Add(lbl_progress.LeftPos(5, 300).BottomPos(5, 20));
	
	Add(palette.LeftPos(330, 250).TopPos(5, 350));
	palette.WhenCharSelected << [this](int c) { SetLabel(c); };
}

void CharLabelCtrl::UpdateChar() {
	if (!project || image_index < 0 || segment_index < 0 || char_index < 0) {
		char_preview.SetImage(Image());
		label_edit.SetData("");
		UpdateProgress();
		return;
	}
	
	const Image& img = project->GetImage(image_index);
	const OCRSegment& seg = project->GetSegments(image_index)[segment_index];
	if (char_index >= seg.chars.GetCount()) {
		char_preview.SetImage(Image());
		label_edit.SetData("");
		UpdateProgress();
		return;
	}
	
	const OCRChar& ch = seg.chars[char_index];
	
	Image char_img = CropImage(img, ch.bounds);
	char_preview.SetImage(Rescale(char_img, 128, 128));
	
	if (ch.label > 0)
		label_edit.SetData(Format("%c", ClassToChar(ch.label)));
	else
		label_edit.SetData("");
	
	UpdateProgress();
}

void CharLabelCtrl::SetLabel(int code) {
	if (!project || image_index < 0 || segment_index < 0 || char_index < 0) return;
	
	OCRSegment& seg = project->GetSegments(image_index)[segment_index];
	if (char_index < seg.chars.GetCount()) {
		seg.chars[char_index].label = CharToClass(code);
		project->SetModified();
		OnNext();
	}
}

void CharLabelCtrl::SkipChar() {
	OnNext();
}

void CharLabelCtrl::DeleteChar() {
	if (!project || image_index < 0 || segment_index < 0 || char_index < 0) return;
	
	OCRSegment& seg = project->GetSegments(image_index)[segment_index];
	if (char_index < seg.chars.GetCount()) {
		seg.chars.Remove(char_index);
		project->SetModified();
		if (char_index >= seg.chars.GetCount()) char_index = seg.chars.GetCount() - 1;
		UpdateChar();
		WhenChanged();
	}
}

void CharLabelCtrl::LabelSequence(const String& text) {
	if (!project || image_index < 0 || segment_index < 0 || char_index < 0) return;
	
	OCRSegment& seg = project->GetSegments(image_index)[segment_index];
	int n = min((int)text.GetCount(), seg.chars.GetCount() - char_index);
	
	for (int i = 0; i < n; i++) {
		seg.chars[char_index + i].label = CharToClass(text[i]);
	}
	
	char_index += n;
	if (char_index >= seg.chars.GetCount()) char_index = seg.chars.GetCount() - 1;
	
	project->SetModified();
	UpdateChar();
	WhenChanged();
}

void CharLabelCtrl::OnPrev() {
	if (char_index > 0) {
		char_index--;
		UpdateChar();
		WhenChanged();
	}
}

void CharLabelCtrl::OnNext() {
	if (project && image_index >= 0 && segment_index >= 0) {
		OCRSegment& seg = project->GetSegments(image_index)[segment_index];
		if (char_index < seg.chars.GetCount() - 1) {
			char_index++;
			UpdateChar();
			WhenChanged();
		}
	}
}

void CharLabelCtrl::OnLabelEdit() {
	String s = label_edit.GetData();
	if (s.GetCount() > 0) {
		SetLabel(s[0]);
	}
}

void CharLabelCtrl::UpdateProgress() {
	if (!project) {
		progress.Set(0, 1);
		lbl_progress.SetText("");
		return;
	}
	
	int total = project->GetTotalCharacters();
	int labeled = 0;
	for (int i = 0; i < project->GetCount(); i++) {
		for (const OCRSegment& seg : project->GetSegments(i)) {
			for (const OCRChar& ch : seg.chars) {
				if (ch.label > 0) labeled++;
			}
		}
	}
	
	progress.Set(labeled, total);
	lbl_progress.SetText(Format("Progress: %d / %d (%d%%)", labeled, total, total > 0 ? labeled * 100 / total : 0));
}

bool CharLabelCtrl::Key(dword key, int count) {
	if (key >= 'A' && key <= 'Z') { SetLabel(key); return true; }
	if (key >= 'a' && key <= 'z') { SetLabel(key); return true; }
	if (key >= '0' && key <= '9') { SetLabel(key); return true; }
	if (key == K_SPACE || key == K_RIGHT) { OnNext(); return true; }
	if (key == K_BACKSPACE || key == K_LEFT) { OnPrev(); return true; }
	if (key == K_DELETE) { OnDelete(); return true; }
	
	return false;
}

// --- ImageListCtrl Implementation ---

ImageListCtrl::ImageListCtrl() {
	project = nullptr;
	
	AddFrame(toolbar);
	toolbar.Set(THISBACK(MainToolbar));
	
	list.AddColumn("Thumbnail").FixedWidth(64);
	list.AddColumn("Filename");
	list.AddColumn("Size");
	list.AddColumn("Segments");
	
	list.SetLineCy(64);
	list.WhenLeftClick = THISBACK(OnImageClicked);
	
	Add(list.SizePos());
}

void ImageListCtrl::SetProject(OCRDatasetProject* proj) {
	project = proj;
	UpdateList();
}

void ImageListCtrl::UpdateList() {
	int cursor = list.GetCursor();
	list.Clear();
	if (!project) return;
	
	for (int i = 0; i < project->GetCount(); i++) {
		const Image& img = project->GetImage(i);
		String path = project->GetImagePath(i);
		
		Image thumb = CreateThumbnail(img, Size(64, 64));
		
		list.Add(thumb, 
		         GetFileName(path), 
		         Format("%d x %d", img.GetWidth(), img.GetHeight()),
		         project->GetSegments(i).GetCount());
	}
	
	if (cursor >= 0 && cursor < list.GetCount())
		list.SetCursor(cursor);
	else if (list.GetCount() > 0) {
		list.SetCursor(0);
		OnImageClicked();
	}
}

void ImageListCtrl::MainToolbar(Bar& bar) {
	bar.Add(CtrlImg::new_doc(), THISBACK(OnAddImage)).Help("Add Image");
	bar.Add(CtrlImg::copy(), THISBACK(OnPasteImage)).Help("Paste Image from Clipboard");
	bar.Add(CtrlImg::remove(), THISBACK(OnRemoveImage)).Help("Remove Selected Image");
}

void ImageListCtrl::OnImageClicked() {
	if (list.IsCursor()) {
		WhenImageSelected(list.GetCursor());
	}
}

void ImageListCtrl::AddImage() {
	if (!project) return;
	
	FileSel fs;
	fs.Type("Images", "*.png *.jpg *.jpeg *.bmp");
	fs.Multi();
	if (fs.ExecuteOpen("Add Images")) {
		for (int i = 0; i < fs.GetCount(); i++) {
			AddImageFromPath(fs[i]);
		}
		WhenListChanged();
		UpdateList();
	}
}

void ImageListCtrl::AddImageFromPath(const String& path) {
	if (!project) return;
	
	FileIn in(path);
	if (in.IsOpen()) {
		Image img = StreamRaster::LoadAny(in);
		if (!img.IsEmpty()) {
			project->AddImage(img, path);
			project->SetModified();
		}
	}
}

void ImageListCtrl::AddImageFromClipboard() {
	if (!project) return;
	
	Image img = ReadClipboardImage();
	if (!img.IsEmpty()) {
		project->AddImage(img, "clipboard_" + Format(GetSysTime()));
		project->SetModified();
		WhenListChanged();
		UpdateList();
	}
}

void ImageListCtrl::RemoveSelectedImage() {
	if (!project || !list.IsCursor()) return;
	
	int idx = list.GetCursor();
	if (PromptYesNo("Are you sure you want to remove the selected image?")) {
		project->RemoveImage(idx);
		project->SetModified();
		WhenListChanged();
		UpdateList();
	}
}

void ImageListCtrl::DragAndDrop(Point p, PasteClip& d) {
	if (IsAvailableFiles(d)) {
		Vector<String> files = GetFiles(d);
		bool changed = false;
		for (const String& file : files) {
			String ext = ToLower(GetFileExt(file));
			if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp") {
				AddImageFromPath(file);
				changed = true;
			}
		}
		if (changed) {
			WhenListChanged();
			UpdateList();
		}
	}
}

Image ImageListCtrl::CreateThumbnail(const Image& src, Size target_size) {
	if (src.IsEmpty()) return Image();
	
	Size thumb_size = OCR::GetFitSize(src.GetSize(), target_size);
	
	// Create a background-filled image
	ImageBuffer ib(target_size);
	Fill(~ib, White(), target_size.cx * target_size.cy);

	Point offset(
		(target_size.cx - thumb_size.cx) / 2,
		(target_size.cy - thumb_size.cy) / 2
	);

	if (thumb_size.cx > 0 && thumb_size.cy > 0) {
		Image scaled = Rescale(src, thumb_size);
		for(int y = 0; y < thumb_size.cy; y++) {
			for(int x = 0; x < thumb_size.cx; x++) {
				ib[y + offset.y][x + offset.x] = scaled[y][x];
			}
		}
	}

	return ib;
}

} // namespace OCR

} // namespace Upp
