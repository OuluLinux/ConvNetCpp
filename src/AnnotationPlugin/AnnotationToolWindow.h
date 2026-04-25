#ifndef _AnnotationTool_AnnotationToolWindow_h_
#define _AnnotationTool_AnnotationToolWindow_h_

#include <CtrlLib/CtrlLib.h>

#include "AnnLib.h"

NAMESPACE_UPP

class AnnotationCanvas : public Ctrl {
public:
	typedef AnnotationCanvas CLASSNAME;
	enum DragHandle {
		HANDLE_NONE = -1,
		HANDLE_TOP_LEFT,
		HANDLE_TOP,
		HANDLE_TOP_RIGHT,
		HANDLE_RIGHT,
		HANDLE_BOTTOM_RIGHT,
		HANDLE_BOTTOM,
		HANDLE_BOTTOM_LEFT,
		HANDLE_LEFT,
	};

	void SetModel(AnnLib* lib);
	void SetCurrentImageId(int id);
	void SetSelectedObjectId(int id);
	void SetDrawMode(bool b);
	void SetCurrentCategory(const String& label);
	int  GetSelectedObjectId() const { return selected_object_id; }

	Event<int>  WhenObjectSelected;
	Event<AnnBBox> WhenCreateRect;
	Event<>     WhenRequestDelete;
	Event<>     WhenObjectChanged;

	virtual void Paint(Draw& w) override;
	virtual void LeftDown(Point p, dword keyflags) override;
	virtual void LeftUp(Point p, dword keyflags) override;
	virtual void MouseMove(Point p, dword keyflags) override;
	virtual bool Key(dword key, int count) override;
	virtual void MouseLeave() override;
	virtual Image CursorImage(Point p, dword keyflags) override;

private:
	const AnnImage* GetCurrentImage() const;
	Image  LoadCurrentImage() const;
	Rect  GetObjectRect(const AnnObject& obj) const;
	Rect  ImageToCanvas(const AnnBBox& bbox) const;
	Rect  ImageToCanvas(const Rect& r) const;
	Rect  CanvasToImage(const Rect& r) const;
	Point ImageToCanvas(Point p) const;
	Point CanvasToImage(Point p) const;
	bool  HasImageGeometry() const;
	void  UpdateImageGeometry(const Size& image_sz);
	Color GetCategoryColor(const String& label) const;
	void  DrawAnnotation(Draw& w, const AnnObject& obj, const Rect& r, bool selected);
	void  DrawHandle(Draw& w, Point center, Color c) const;
	bool  HitTestObject(Point image_pt, int& object_id) const;
	AnnObject* FindSelectedObject() const;
	Vector<Point> GetHandleCenters(const Rect& r) const;
	DragHandle HitTestHandle(Point canvas_pt) const;
	bool  HitTestSelectedInterior(Point image_pt) const;
	void  UpdateHoverCursor(Point p);
	void  ApplyResizeDrag(Point image_pt);
	void  ApplyMoveDrag(Point image_pt);
	Image GetCursorForHandle(DragHandle handle) const;
	Image GetMoveCursor() const;

	AnnLib* model = nullptr;
	int     current_image_id = -1;
	int     selected_object_id = -1;
	bool    draw_mode = true;
	bool    dragging = false;
	bool    move_drag = false;
	bool    resize_drag = false;
	String  current_category = "seat_name";
	Point   drag_start;
	Point   drag_end;
	Point   drag_anchor_image;
	Point   move_offset_image;
	AnnBBox drag_origin_bbox;
	DragHandle active_handle = HANDLE_NONE;
	Image   hover_cursor = Image::Arrow();
	double  image_scale = 1.0;
	int     image_x0 = 0;
	int     image_y0 = 0;
	Size    image_size;
};

class AnnotationToolWindow : public ParentCtrl {
public:
	typedef AnnotationToolWindow CLASSNAME;

	AnnotationToolWindow();

	void SetModel(AnnLib* lib);
	void RefreshFromModel();
	void SetDrawMode(bool draw);
	bool IsDrawMode() const                     { return draw_mode; }
	bool HasSelection() const                   { return selected_image_id > 0; }
	int  GetSelectedImageId() const             { return selected_image_id; }

	Event<> WhenModified;
	Event<> WhenExportCOCO;

	void PromptAddScreenshot();
	void PromptCaptureFromVideo();
	void RemoveSelectedScreenshot();
	void AddObjectForSelection();
	void DeleteSelectedObject();

private:
	void OnImageSelection();
	void OnObjectSelection();
	void SyncObjectInspector();
	void CommitInspector();
	void HandleCanvasCreateRect(Point a, Point b);
	void SelectImage(int image_id);
	void SelectObject(int object_id);
	AnnImage* FindImage(int image_id);
	AnnObject* FindObject(int object_id);

	Splitter   split;
	ParentCtrl left_panel;
	ParentCtrl center_panel;
	ParentCtrl right_panel;

	ArrayCtrl       screenshot_list;
	Button          add_screenshot;
	Button          capture_video;
	Button          remove_screenshot;
	AnnotationCanvas canvas;

	ArrayCtrl       object_list;
	StaticRect      inspector_frame;
	DropList        category;
	EditIntSpin     bbox_x;
	EditIntSpin     bbox_y;
	EditIntSpin     bbox_w;
	EditIntSpin     bbox_h;
	Label           bbox_x_label;
	Label           bbox_y_label;
	Label           bbox_w_label;
	Label           bbox_h_label;
	DocEdit         notes;
	Button          add_object;
	Button          delete_object;
	DropList        label_filter;

	AnnLib*         model = nullptr;
	int             selected_image_id = -1;
	int             selected_object_id = -1;
	bool            draw_mode = true;
	Vector<int>     screenshot_ids;
	Vector<int>     object_ids;
};

END_UPP_NAMESPACE

#endif
