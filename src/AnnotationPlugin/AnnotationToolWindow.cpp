#include "VideoCaptureModal.h"
#include "AnnotationToolWindow.h"

NAMESPACE_UPP

namespace {

Rect NormalizeRect(Point a, Point b)
{
	int left = min(a.x, b.x);
	int top = min(a.y, b.y);
	int right = max(a.x, b.x);
	int bottom = max(a.y, b.y);
	if(right <= left)
		right = left + 1;
	if(bottom <= top)
		bottom = top + 1;
	return Rect(left, top, right, bottom);
}

AnnBBox NormalizeBBox(const Rect& r)
{
	AnnBBox bbox;
	bbox.x = r.left;
	bbox.y = r.top;
	bbox.w = max(1, r.GetWidth());
	bbox.h = max(1, r.GetHeight());
	return bbox;
}

}

void AnnotationCanvas::SetModel(AnnLib* lib)
{
	model = lib;
	Refresh();
}

void AnnotationCanvas::SetCurrentImageId(int id)
{
	current_image_id = id;
	Refresh();
}

void AnnotationCanvas::SetSelectedObjectId(int id)
{
	selected_object_id = id;
	Refresh();
}

void AnnotationCanvas::SetDrawMode(bool b)
{
	draw_mode = b;
}

void AnnotationCanvas::SetCurrentCategory(const String& label)
{
	current_category = label;
}

const AnnImage* AnnotationCanvas::GetCurrentImage() const
{
	if(!model)
		return nullptr;
	const AnnDataset& ds = model->GetDataset();
	for(const AnnImage& img : ds.images)
		if(img.id == current_image_id)
			return &img;
	return nullptr;
}

Image AnnotationCanvas::LoadCurrentImage() const
{
	const AnnImage* current = GetCurrentImage();
	if(!current || !model)
		return Null;
	String img_path = AppendFileName(model->GetResolvedDataDir(), current->filename);
	return StreamRaster::LoadFileAny(img_path);
}

bool AnnotationCanvas::HasImageGeometry() const
{
	return image_scale > 0 && image_size.cx > 0 && image_size.cy > 0;
}

void AnnotationCanvas::UpdateImageGeometry(const Size& image_sz)
{
	image_size = image_sz;
	Size canvas = GetSize();
	if(image_sz.cx <= 0 || image_sz.cy <= 0 || canvas.cx <= 0 || canvas.cy <= 0) {
		image_scale = 1.0;
		image_x0 = 0;
		image_y0 = 0;
		return;
	}

	double sx = (double)canvas.cx / (double)image_sz.cx;
	double sy = (double)canvas.cy / (double)image_sz.cy;
	image_scale = min(sx, sy);
	if(image_scale <= 0)
		image_scale = 1.0;

	int draw_w = max(1, (int)floor(image_sz.cx * image_scale + 0.5));
	int draw_h = max(1, (int)floor(image_sz.cy * image_scale + 0.5));
	image_x0 = (canvas.cx - draw_w) / 2;
	image_y0 = (canvas.cy - draw_h) / 2;
}

Point AnnotationCanvas::CanvasToImage(Point p) const
{
	if(!HasImageGeometry())
		return Point(0, 0);
	int x = (int)floor((p.x - image_x0) / image_scale);
	int y = (int)floor((p.y - image_y0) / image_scale);
	x = minmax(x, 0, max(0, image_size.cx - 1));
	y = minmax(y, 0, max(0, image_size.cy - 1));
	return Point(x, y);
}

Point AnnotationCanvas::ImageToCanvas(Point p) const
{
	if(!HasImageGeometry())
		return Point(0, 0);
	int x = image_x0 + (int)floor(p.x * image_scale + 0.5);
	int y = image_y0 + (int)floor(p.y * image_scale + 0.5);
	return Point(x, y);
}

Rect AnnotationCanvas::CanvasToImage(const Rect& r) const
{
	Point tl = CanvasToImage(r.TopLeft());
	Point br = CanvasToImage(Point(max(r.left, r.right - 1), max(r.top, r.bottom - 1)));
	Rect out = NormalizeRect(tl, br);
	out.right = min(image_size.cx, out.right + 1);
	out.bottom = min(image_size.cy, out.bottom + 1);
	return out;
}

Rect AnnotationCanvas::ImageToCanvas(const Rect& r) const
{
	Point tl = ImageToCanvas(r.TopLeft());
	Point br = ImageToCanvas(Point(r.right, r.bottom));
	return NormalizeRect(tl, br);
}

Rect AnnotationCanvas::ImageToCanvas(const AnnBBox& bbox) const
{
	return ImageToCanvas(bbox.ToRect());
}

Rect AnnotationCanvas::GetObjectRect(const AnnObject& obj) const
{
	return obj.bbox.ToRect();
}

Color AnnotationCanvas::GetCategoryColor(const String& label) const
{
	if(label == "seat_name") return Color(46, 139, 87);
	if(label == "seat_stack") return Color(33, 150, 243);
	if(label == "seat_action") return Color(255, 152, 0);
	if(label == "pot_label") return Color(220, 50, 47);
	if(label == "board_card") return Color(121, 85, 72);
	if(label == "board_area") return Color(103, 58, 183);
	if(label == "dealer_button") return Color(255, 193, 7);
	if(label == "hero_indicator") return Color(0, 150, 136);
	return Color(96, 125, 139);
}

void AnnotationCanvas::DrawHandle(Draw& w, Point center, Color c) const
{
	w.DrawRect(center.x - 3, center.y - 3, 6, 6, White());
	w.DrawRect(center.x - 2, center.y - 2, 4, 4, c);
}

void AnnotationCanvas::DrawAnnotation(Draw& w, const AnnObject& obj, const Rect& r, bool selected)
{
	Color c = GetCategoryColor(obj.label);
	int thickness = selected ? 2 : 1;

	for(int i = 0; i < thickness; i++) {
		Rect rr = r.Deflated(i, i);
		w.DrawRect(rr.left, rr.top, rr.GetWidth(), 1, c);
		w.DrawRect(rr.left, rr.bottom - 1, rr.GetWidth(), 1, c);
		w.DrawRect(rr.left, rr.top, 1, rr.GetHeight(), c);
		w.DrawRect(rr.right - 1, rr.top, 1, rr.GetHeight(), c);
	}

	String txt = obj.label.IsEmpty() ? Format("category_%d", obj.category_id) : obj.label;
	w.DrawRect(r.left, max(0, r.top - 20), min(160, max(50, GetTextSize(txt, StdFont()).cx + 10)), 18, Blend(c, White(), 90));
	w.DrawText(r.left + 4, max(0, r.top - 18), txt, StdFont().Bold(), Black());

	if(selected) {
		Point tl = r.TopLeft();
		Point tr(r.right, r.top);
		Point bl(r.left, r.bottom);
		Point br = r.BottomRight();
		Point tm((r.left + r.right) / 2, r.top);
		Point bm((r.left + r.right) / 2, r.bottom);
		Point lm(r.left, (r.top + r.bottom) / 2);
		Point rm(r.right, (r.top + r.bottom) / 2);
		DrawHandle(w, tl, c);
		DrawHandle(w, tr, c);
		DrawHandle(w, bl, c);
		DrawHandle(w, br, c);
		DrawHandle(w, tm, c);
		DrawHandle(w, bm, c);
		DrawHandle(w, lm, c);
		DrawHandle(w, rm, c);
	}
}

bool AnnotationCanvas::HitTestObject(Point image_pt, int& object_id) const
{
	object_id = -1;
	if(!model)
		return false;
	const AnnDataset& ds = model->GetDataset();
	for(int i = ds.annotations.GetCount() - 1; i >= 0; i--) {
		const AnnObject& obj = ds.annotations[i];
		if(obj.image_id != current_image_id)
			continue;
		if(GetObjectRect(obj).Contains(image_pt)) {
			object_id = obj.id;
			return true;
		}
	}
	return false;
}

AnnObject* AnnotationCanvas::FindSelectedObject() const
{
	if(!model || selected_object_id <= 0)
		return nullptr;
	for(AnnObject& obj : model->GetDataset().annotations)
		if(obj.id == selected_object_id)
			return &obj;
	return nullptr;
}

Vector<Point> AnnotationCanvas::GetHandleCenters(const Rect& r) const
{
	Vector<Point> out;
	out << r.TopLeft()
	    << Point((r.left + r.right) / 2, r.top)
	    << Point(r.right, r.top)
	    << Point(r.right, (r.top + r.bottom) / 2)
	    << r.BottomRight()
	    << Point((r.left + r.right) / 2, r.bottom)
	    << Point(r.left, r.bottom)
	    << Point(r.left, (r.top + r.bottom) / 2);
	return out;
}

AnnotationCanvas::DragHandle AnnotationCanvas::HitTestHandle(Point canvas_pt) const
{
	const AnnObject* obj = FindSelectedObject();
	if(!obj)
		return HANDLE_NONE;
	Vector<Point> handles = GetHandleCenters(ImageToCanvas(obj->bbox));
	for(int i = 0; i < handles.GetCount(); i++) {
		Rect hr(handles[i].x - 3, handles[i].y - 3, handles[i].x + 3, handles[i].y + 3);
		if(hr.Contains(canvas_pt))
			return (DragHandle)i;
	}
	return HANDLE_NONE;
}

bool AnnotationCanvas::HitTestSelectedInterior(Point image_pt) const
{
	const AnnObject* obj = FindSelectedObject();
	return obj && GetObjectRect(*obj).Contains(image_pt);
}

Image AnnotationCanvas::GetCursorForHandle(DragHandle handle) const
{
	switch(handle) {
	case HANDLE_TOP:
	case HANDLE_BOTTOM:
		return Image::SizeVert();
	case HANDLE_LEFT:
	case HANDLE_RIGHT:
		return Image::SizeHorz();
	case HANDLE_TOP_LEFT:
	case HANDLE_BOTTOM_RIGHT:
		return Image::SizeTopLeft();
	case HANDLE_TOP_RIGHT:
	case HANDLE_BOTTOM_LEFT:
		return Image::SizeTopRight();
	default:
		return Image::Arrow();
	}
}

Image AnnotationCanvas::GetMoveCursor() const
{
	return Image::Cross();
}

void AnnotationCanvas::UpdateHoverCursor(Point p)
{
	if(draw_mode) {
		hover_cursor = Image::Arrow();
		return;
	}
	DragHandle handle = HitTestHandle(p);
	if(handle != HANDLE_NONE) {
		hover_cursor = GetCursorForHandle(handle);
		return;
	}
	if(HitTestSelectedInterior(CanvasToImage(p))) {
		hover_cursor = GetMoveCursor();
		return;
	}
	hover_cursor = Image::Arrow();
}

void AnnotationCanvas::ApplyResizeDrag(Point image_pt)
{
	AnnObject* obj = FindSelectedObject();
	if(!obj)
		return;
	int left = drag_origin_bbox.x;
	int top = drag_origin_bbox.y;
	int right = drag_origin_bbox.x + drag_origin_bbox.w;
	int bottom = drag_origin_bbox.y + drag_origin_bbox.h;
	int dx = image_pt.x - drag_anchor_image.x;
	int dy = image_pt.y - drag_anchor_image.y;

	switch(active_handle) {
	case HANDLE_TOP_LEFT:
		left += dx;
		top += dy;
		break;
	case HANDLE_TOP:
		top += dy;
		break;
	case HANDLE_TOP_RIGHT:
		right += dx;
		top += dy;
		break;
	case HANDLE_RIGHT:
		right += dx;
		break;
	case HANDLE_BOTTOM_RIGHT:
		right += dx;
		bottom += dy;
		break;
	case HANDLE_BOTTOM:
		bottom += dy;
		break;
	case HANDLE_BOTTOM_LEFT:
		left += dx;
		bottom += dy;
		break;
	case HANDLE_LEFT:
		left += dx;
		break;
	default:
		break;
	}

	left = minmax(left, 0, max(0, image_size.cx - 4));
	top = minmax(top, 0, max(0, image_size.cy - 4));
	right = minmax(right, left + 4, image_size.cx);
	bottom = minmax(bottom, top + 4, image_size.cy);
	obj->bbox = NormalizeBBox(Rect(left, top, right, bottom));
}

void AnnotationCanvas::ApplyMoveDrag(Point image_pt)
{
	AnnObject* obj = FindSelectedObject();
	if(!obj)
		return;
	int nx = image_pt.x - move_offset_image.x;
	int ny = image_pt.y - move_offset_image.y;
	nx = minmax(nx, 0, max(0, image_size.cx - obj->bbox.w));
	ny = minmax(ny, 0, max(0, image_size.cy - obj->bbox.h));
	obj->bbox.x = nx;
	obj->bbox.y = ny;
}

void AnnotationCanvas::Paint(Draw& w)
{
	Size sz = GetSize();
	w.DrawRect(sz, Color(24, 30, 36));

	if(!model)
		return;

	const AnnImage* current = GetCurrentImage();
	if(!current) {
		w.DrawText(16, 16, "No screenshot selected", StdFont().Bold(), White());
		return;
	}

	Image img = LoadCurrentImage();
	if(img.IsEmpty()) {
		w.DrawText(16, 16, "Image not found: " + current->filename, StdFont().Bold(), Color(255, 200, 200));
		return;
	}

	UpdateImageGeometry(img.GetSize());
	Image scaled = Rescale(img, Size((int)floor(img.GetWidth() * image_scale + 0.5),
	                                 (int)floor(img.GetHeight() * image_scale + 0.5)));
	w.DrawImage(image_x0, image_y0, scaled);

	const AnnDataset& ds = model->GetDataset();
	for(const AnnObject& obj : ds.annotations)
		if(obj.image_id == current_image_id)
			DrawAnnotation(w, obj, ImageToCanvas(obj.bbox), obj.id == selected_object_id);

	if(dragging) {
		Rect canvas_rect = NormalizeRect(drag_start, drag_end);
		AnnBBox image_bbox;
		Rect image_rect = CanvasToImage(canvas_rect);
		image_bbox.x = image_rect.left;
		image_bbox.y = image_rect.top;
		image_bbox.w = image_rect.GetWidth();
		image_bbox.h = image_rect.GetHeight();
		AnnObject preview;
		preview.image_id = current_image_id;
		preview.label = current_category;
		preview.bbox = image_bbox;
		DrawAnnotation(w, preview, ImageToCanvas(image_bbox), true);
	}
}

void AnnotationCanvas::LeftDown(Point p, dword)
{
	if(!model)
		return;

	drag_start = drag_end = p;
	if(draw_mode) {
		dragging = true;
		SetCapture();
		Refresh();
		return;
	}

	DragHandle handle = HitTestHandle(p);
	if(handle != HANDLE_NONE) {
		resize_drag = true;
		active_handle = handle;
		drag_anchor_image = CanvasToImage(p);
		if(AnnObject* obj = FindSelectedObject())
			drag_origin_bbox = obj->bbox;
		SetCapture();
		UpdateHoverCursor(p);
		return;
	}

	Point image_pt = CanvasToImage(p);
	if(HitTestSelectedInterior(image_pt)) {
		move_drag = true;
		if(AnnObject* obj = FindSelectedObject()) {
			move_offset_image = Point(image_pt.x - obj->bbox.x, image_pt.y - obj->bbox.y);
			drag_origin_bbox = obj->bbox;
		}
		SetCapture();
		UpdateHoverCursor(p);
		return;
	}

	int object_id = -1;
	if(HitTestObject(image_pt, object_id)) {
		selected_object_id = object_id;
		WhenObjectSelected(object_id);
		Refresh();
		UpdateHoverCursor(p);
		return;
	}

	selected_object_id = -1;
	WhenObjectSelected(-1);
	Refresh();
	UpdateHoverCursor(p);
}

void AnnotationCanvas::LeftUp(Point p, dword)
{
	if(resize_drag || move_drag) {
		bool changed = resize_drag || move_drag;
		resize_drag = false;
		move_drag = false;
		active_handle = HANDLE_NONE;
		ReleaseCapture();
		if(changed && WhenObjectChanged)
			WhenObjectChanged();
		WhenObjectSelected(selected_object_id);
		Refresh();
		UpdateHoverCursor(p);
		return;
	}

	if(!dragging)
		return;

	drag_end = p;
	ReleaseCapture();
	dragging = false;
	Refresh();

	Rect canvas_rect = NormalizeRect(drag_start, drag_end);
	if(canvas_rect.GetWidth() <= 5 && canvas_rect.GetHeight() <= 5)
		return;

	if(WhenCreateRect) {
		Rect image_rect = CanvasToImage(canvas_rect);
		AnnBBox bbox;
		bbox.x = image_rect.left;
		bbox.y = image_rect.top;
		bbox.w = max(1, image_rect.GetWidth());
		bbox.h = max(1, image_rect.GetHeight());
		WhenCreateRect(bbox);
	}
}

void AnnotationCanvas::MouseMove(Point p, dword)
{
	if(resize_drag) {
		ApplyResizeDrag(CanvasToImage(p));
		WhenObjectSelected(selected_object_id);
		Refresh();
		UpdateHoverCursor(p);
		return;
	}
	if(move_drag) {
		ApplyMoveDrag(CanvasToImage(p));
		WhenObjectSelected(selected_object_id);
		Refresh();
		UpdateHoverCursor(p);
		return;
	}
	if(dragging) {
		drag_end = p;
		Refresh();
		return;
	}
	UpdateHoverCursor(p);
}

bool AnnotationCanvas::Key(dword key, int)
{
	if(key == K_DELETE && WhenRequestDelete) {
		WhenRequestDelete();
		return true;
	}
	return false;
}

void AnnotationCanvas::MouseLeave()
{
	hover_cursor = Image::Arrow();
}

Image AnnotationCanvas::CursorImage(Point, dword)
{
	return hover_cursor;
}

AnnotationToolWindow::AnnotationToolWindow()
{
	Add(split.Horz().SizePos());
	split << left_panel << center_panel << right_panel;
	split.SetPos(2000, 0);
	split.SetPos(7600, 1);

	left_panel.Add(screenshot_list.HSizePosZ(6, 6).VSizePosZ(6, 90));
	left_panel.Add(add_screenshot.SetLabel("Add Screenshot").LeftPosZ(6, 186).BottomPosZ(62, 24));
	left_panel.Add(capture_video.SetLabel("Capture From Video").LeftPosZ(6, 186).BottomPosZ(34, 24));
	left_panel.Add(remove_screenshot.SetLabel("Remove").LeftPosZ(6, 186).BottomPosZ(6, 24));
	screenshot_list.AddColumn("Screenshot");
	screenshot_list.AddColumn("Objects");
	screenshot_list.ColumnWidths("4 1");
	screenshot_list.WhenSel = [this] { OnImageSelection(); };

	center_panel.Add(canvas.SizePos());
	canvas.WhenObjectSelected = [this](int id) { SelectObject(id); };
	canvas.WhenCreateRect = [this](const AnnBBox& bbox) { HandleCanvasCreateRect(Point(bbox.x, bbox.y), Point(bbox.x + bbox.w, bbox.y + bbox.h)); };
	canvas.WhenRequestDelete = [this] { DeleteSelectedObject(); };
	canvas.WhenObjectChanged = [this] {
		if(WhenModified)
			WhenModified();
		RefreshFromModel();
		SelectObject(selected_object_id);
	};

	right_panel.Add(object_list.HSizePosZ(6, 6).TopPosZ(6, 140));
	right_panel.Add(inspector_frame.HSizePosZ(6, 6).VSizePosZ(150, 40));
	right_panel.Add(add_object.SetLabel("Add Object").LeftPosZ(6, 100).BottomPosZ(8, 24));
	right_panel.Add(delete_object.SetLabel("Delete Object").LeftPosZ(112, 100).BottomPosZ(8, 24));

	object_list.AddColumn("Label");
	object_list.AddColumn("BBox");
	object_list.ColumnWidths("3 2");
	object_list.WhenSel = [this] { OnObjectSelection(); };

	inspector_frame.Color(Color(248, 248, 248));
	inspector_frame.Add(bbox_x_label.SetLabel("x").LeftPosZ(8, 10).TopPosZ(8, 20));
	inspector_frame.Add(bbox_x.LeftPosZ(24, 74).TopPosZ(6, 24));
	inspector_frame.Add(bbox_y_label.SetLabel("y").LeftPosZ(108, 10).TopPosZ(8, 20));
	inspector_frame.Add(bbox_y.LeftPosZ(124, 74).TopPosZ(6, 24));
	inspector_frame.Add(bbox_w_label.SetLabel("w").LeftPosZ(8, 10).TopPosZ(38, 20));
	inspector_frame.Add(bbox_w.LeftPosZ(24, 74).TopPosZ(36, 24));
	inspector_frame.Add(bbox_h_label.SetLabel("h").LeftPosZ(108, 10).TopPosZ(38, 20));
	inspector_frame.Add(bbox_h.LeftPosZ(124, 74).TopPosZ(36, 24));
	inspector_frame.Add(category.HSizePosZ(8, 8).TopPosZ(66, 24));
	inspector_frame.Add(notes.HSizePosZ(8, 8).VSizePosZ(96, 8));
	notes.WhenAction = [this] { CommitInspector(); };
	category.WhenAction = [this] {
		canvas.SetCurrentCategory(~category);
		CommitInspector();
	};
	bbox_x.WhenAction = [this] { CommitInspector(); };
	bbox_y.WhenAction = [this] { CommitInspector(); };
	bbox_w.WhenAction = [this] { CommitInspector(); };
	bbox_h.WhenAction = [this] { CommitInspector(); };

	add_screenshot << [this] { PromptAddScreenshot(); };
	capture_video << [this] { PromptCaptureFromVideo(); };
	remove_screenshot << [this] { RemoveSelectedScreenshot(); };
	add_object << [this] { AddObjectForSelection(); };
	delete_object << [this] { DeleteSelectedObject(); };

	label_filter.Add("show_all");
	for(const String& label : GetInitialAnnotationTaxonomy()) {
		category.Add(label);
		label_filter.Add(label);
	}
	category.SetIndex(0);
	label_filter.SetIndex(0);
	canvas.SetCurrentCategory(~category);
	canvas.SetFrame(BlackFrame());
	canvas.WantFocus();
}

void AnnotationToolWindow::SetModel(AnnLib* lib)
{
	model = lib;
	canvas.SetModel(model);
	RefreshFromModel();
}

AnnImage* AnnotationToolWindow::FindImage(int image_id)
{
	if(!model)
		return nullptr;
	for(AnnImage& img : model->GetDataset().images)
		if(img.id == image_id)
			return &img;
	return nullptr;
}

AnnObject* AnnotationToolWindow::FindObject(int object_id)
{
	if(!model)
		return nullptr;
	for(AnnObject& obj : model->GetDataset().annotations)
		if(obj.id == object_id)
			return &obj;
	return nullptr;
}

void AnnotationToolWindow::SelectImage(int image_id)
{
	selected_image_id = image_id;
	canvas.SetCurrentImageId(image_id);
	SelectObject(-1);
}

void AnnotationToolWindow::SelectObject(int object_id)
{
	selected_object_id = object_id;
	canvas.SetSelectedObjectId(object_id);
	for(int i = 0; i < object_ids.GetCount(); i++) {
		if(object_ids[i] == object_id) {
			object_list.SetCursor(i);
			break;
		}
	}
	SyncObjectInspector();
	canvas.SetFocus();
}

void AnnotationToolWindow::RefreshFromModel()
{
	screenshot_list.Clear();
	object_list.Clear();
	screenshot_ids.Clear();
	object_ids.Clear();
	if(!model)
		return;

	const AnnDataset& ds = model->GetDataset();
	for(const AnnImage& img : ds.images) {
		int count = 0;
		for(const AnnObject& obj : ds.annotations)
			if(obj.image_id == img.id)
				count++;
		screenshot_list.Add(img.filename, count);
		screenshot_ids.Add(img.id);
	}

	if(selected_image_id <= 0 && ds.images.GetCount())
		selected_image_id = ds.images[0].id;

	for(int i = 0; i < screenshot_ids.GetCount(); i++) {
		if(screenshot_ids[i] == selected_image_id) {
			screenshot_list.SetCursor(i);
			break;
		}
	}

	if(selected_image_id > 0) {
		String active_filter = ~label_filter;
		for(const AnnObject& obj : ds.annotations) {
			if(obj.image_id != selected_image_id)
				continue;
			if(active_filter != "show_all" && obj.label != active_filter)
				continue;
			object_list.Add(obj.label, Format("%d,%d %dx%d", obj.bbox.x, obj.bbox.y, obj.bbox.w, obj.bbox.h));
			object_ids.Add(obj.id);
		}
	}

	canvas.SetCurrentImageId(selected_image_id);
	canvas.SetCurrentCategory(~category);
	canvas.Refresh();
	SyncObjectInspector();
}

void AnnotationToolWindow::OnImageSelection()
{
	int row = screenshot_list.GetCursor();
	if(row < 0 || row >= screenshot_ids.GetCount())
		return;
	SelectImage(screenshot_ids[row]);
	RefreshFromModel();
}

void AnnotationToolWindow::OnObjectSelection()
{
	int row = object_list.GetCursor();
	if(row < 0 || row >= object_ids.GetCount()) {
		SelectObject(-1);
		return;
	}
	SelectObject(object_ids[row]);
}

void AnnotationToolWindow::SyncObjectInspector()
{
	AnnObject* obj = FindObject(selected_object_id);
	if(!obj) {
		bbox_x.SetData(0);
		bbox_y.SetData(0);
		bbox_w.SetData(0);
		bbox_h.SetData(0);
		notes.SetData(String());
		return;
	}

	bbox_x.SetData(obj->bbox.x);
	bbox_y.SetData(obj->bbox.y);
	bbox_w.SetData(obj->bbox.w);
	bbox_h.SetData(obj->bbox.h);
	category.SetData(obj->label);
	notes.SetData(obj->notes);
	canvas.SetCurrentCategory(~category);
}

void AnnotationToolWindow::CommitInspector()
{
	AnnObject* obj = FindObject(selected_object_id);
	if(!obj)
		return;

	obj->bbox.x = (int)bbox_x.GetData();
	obj->bbox.y = (int)bbox_y.GetData();
	obj->bbox.w = max(4, (int)bbox_w.GetData());
	obj->bbox.h = max(4, (int)bbox_h.GetData());
	obj->label = ~category;
	obj->notes = notes.GetData().ToString();
	canvas.SetCurrentCategory(~category);
	if(WhenModified)
		WhenModified();
	RefreshFromModel();
}

void AnnotationToolWindow::SetDrawMode(bool draw)
{
	draw_mode = draw;
	canvas.SetDrawMode(draw);
}

void AnnotationToolWindow::HandleCanvasCreateRect(Point a, Point b)
{
	if(!model || selected_image_id <= 0)
		return;

	Rect r = NormalizeRect(a, b);
	if(r.GetWidth() < 4 || r.GetHeight() < 4)
		return;

	AnnObject& obj = model->GetDataset().annotations.Add();
	int next_id = 1;
	for(const AnnObject& it : model->GetDataset().annotations)
		next_id = max(next_id, it.id + 1);
	obj.id = next_id;
	obj.image_id = selected_image_id;
	obj.category_id = 1;
	String selected_label = ~category;
	obj.label = selected_label.IsEmpty() ? "seat_name" : selected_label;
	obj.bbox.x = r.left;
	obj.bbox.y = r.top;
	obj.bbox.w = r.GetWidth();
	obj.bbox.h = r.GetHeight();

	if(WhenModified)
		WhenModified();
	RefreshFromModel();
	SelectObject(obj.id);
}

void AnnotationToolWindow::PromptAddScreenshot()
{
	if(!model)
		return;
	FileSel fs;
	fs.Type("Images", "*.png *.jpg *.jpeg");
	if(!fs.ExecuteOpen())
		return;
	if(model->AddImageFile(~fs)) {
		if(selected_image_id <= 0 && model->GetDataset().images.GetCount())
			selected_image_id = model->GetDataset().images.Top().id;
		if(WhenModified)
			WhenModified();
		RefreshFromModel();
	}
}

void AnnotationToolWindow::PromptCaptureFromVideo()
{
	if(!model)
		return;
	VideoCaptureModal modal;
	if(!modal.ExecuteCapture())
		return;
	if(model->AddImage(modal.GetCapturedImage())) {
		if(WhenModified)
			WhenModified();
		RefreshFromModel();
	}
}

void AnnotationToolWindow::RemoveSelectedScreenshot()
{
	if(!model || selected_image_id <= 0)
		return;

	for(int i = model->GetDataset().annotations.GetCount() - 1; i >= 0; i--)
		if(model->GetDataset().annotations[i].image_id == selected_image_id)
			model->GetDataset().annotations.Remove(i);

	for(int i = 0; i < model->GetDataset().images.GetCount(); i++) {
		if(model->GetDataset().images[i].id == selected_image_id) {
			model->GetDataset().images.Remove(i);
			break;
		}
	}
	selected_image_id = model->GetDataset().images.IsEmpty() ? -1 : model->GetDataset().images[0].id;
	selected_object_id = -1;
	if(WhenModified)
		WhenModified();
	RefreshFromModel();
}

void AnnotationToolWindow::AddObjectForSelection()
{
	if(!model || selected_image_id <= 0)
		return;

	AnnObject& obj = model->GetDataset().annotations.Add();
	int next_id = 1;
	for(const AnnObject& it : model->GetDataset().annotations)
		next_id = max(next_id, it.id + 1);
	obj.id = next_id;
	obj.image_id = selected_image_id;
	obj.category_id = 1;
	String selected_label = ~category;
	obj.label = selected_label.IsEmpty() ? "seat_name" : selected_label;
	obj.bbox.x = 40;
	obj.bbox.y = 40;
	obj.bbox.w = 120;
	obj.bbox.h = 32;

	if(WhenModified)
		WhenModified();
	RefreshFromModel();
	SelectObject(obj.id);
}

void AnnotationToolWindow::DeleteSelectedObject()
{
	if(!model || selected_object_id <= 0)
		return;

	for(int i = 0; i < model->GetDataset().annotations.GetCount(); i++) {
		if(model->GetDataset().annotations[i].id == selected_object_id) {
			model->GetDataset().annotations.Remove(i);
			break;
		}
	}
	selected_object_id = -1;
	if(WhenModified)
		WhenModified();
	RefreshFromModel();
	canvas.SetFocus();
}

END_UPP_NAMESPACE
