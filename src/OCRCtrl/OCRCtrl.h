#ifndef _OCRCtrl_OCRCtrl_h_
#define _OCRCtrl_OCRCtrl_h_

#include <CtrlLib/CtrlLib.h>
#include <OCR/OCR.h>
#include <OCR/OCRDatasetProject.h>

namespace Upp {

namespace OCR {

class ImagePreviewCtrl : public Ctrl {
public:
	typedef ImagePreviewCtrl CLASSNAME;
	ImagePreviewCtrl();

	void SetImage(const Image& img);
	void SetSegments(Vector<OCRSegment>* segs);

	// View control
	void ZoomIn();
	void ZoomOut();
	void ZoomFit();
	void SetZoom(double z);

	// Overlay control
	void ShowBounds(bool show = true) { show_bounds = show; Refresh(); }
	void ShowLabels(bool show = true) { show_labels = show; Refresh(); }
	void ShowGrid(bool show = true) { show_grid = show; Refresh(); }
	void ShowProjection(bool show = true) { show_projection = show; Refresh(); }
	
	bool IsShowBounds() const { return show_bounds; }
	bool IsShowLabels() const { return show_labels; }
	bool IsShowGrid() const { return show_grid; }
	bool IsShowProjection() const { return show_projection; }

	// Selection
	void SelectSegment(int index);
	void SelectChar(int char_index);
	void ClearSelection() { selected_segment = -1; selected_char = -1; Refresh(); }
	void DeleteSelection() { Key(K_DELETE, 1); }

	// Events
	Callback1<int> WhenSegmentSelected;
	Callback2<int, int> WhenCharSelected;

protected:
	virtual void Paint(Draw& w);
	virtual void MouseWheel(Point p, int zdelta, dword keyflags);
	virtual void LeftDown(Point p, dword keyflags);
	virtual void LeftUp(Point p, dword keyflags);
	virtual void MouseMove(Point p, dword keyflags);
	virtual void MiddleDown(Point p, dword keyflags);
	virtual void MiddleUp(Point p, dword keyflags);
	virtual bool Key(dword key, int count);

private:
	Image image;
	Vector<OCRSegment>* segments;
	double zoom;
	Point offset;
	bool show_bounds, show_labels, show_grid, show_projection;
	int selected_segment, selected_char;
	bool dragging;
	bool panning;
	Point last_mouse_pos;
	Rect drag_rect;

	Point ImageToScreen(Point p) const;
	Point ScreenToImage(Point p) const;
	void DrawOverlays(Draw& w);
	void DrawGrid(Draw& w);
	void DrawProjection(Draw& w);
};

class ImageListCtrl : public ParentCtrl {
public:
	typedef ImageListCtrl CLASSNAME;
	ImageListCtrl();

	void SetProject(OCRDatasetProject* proj);
	void UpdateList();

	void AddImage();
	void AddImageFromClipboard();
	void AddImageFromPath(const String& path);
	void RemoveSelectedImage();

	virtual void DragAndDrop(Point p, PasteClip& d);

	Callback1<int> WhenImageSelected;
	Callback WhenListChanged;

private:
	OCRDatasetProject* project;
	ArrayCtrl list;
	ToolBar toolbar;

	void OnImageClicked();
	void OnAddImage() { AddImage(); }
	void OnPasteImage() { AddImageFromClipboard(); }
	void OnRemoveImage() { RemoveSelectedImage(); }
	void MainToolbar(Bar& bar);
	
	Image CreateThumbnail(const Image& src, Size target_size);
};

class SplitToolPanel : public ParentCtrl {
public:
	typedef SplitToolPanel CLASSNAME;
	SplitToolPanel();

	void SetProject(OCRDatasetProject* proj) { project = proj; }
	void SetImageIndex(int index) { image_index = index; }
	void SetSegmentIndex(int index) { segment_index = index; }
	
	void AutoSegment();
	void AutoSegmentAll();
	void ClearSplits();

	Callback WhenChanged;

private:
	OCRDatasetProject* project;
	int image_index;
	int segment_index;

	Button btn_auto;
	Button btn_auto_all;
	Button btn_clear;
	
	Label lbl_min_w, lbl_min_h;
	EditInt min_w, min_h;
	
	void OnAuto() { AutoSegment(); }
	void OnAutoAll() { AutoSegmentAll(); }
	void OnClear() { ClearSplits(); }
};

class CharPaletteCtrl : public Ctrl {
public:
	typedef CharPaletteCtrl CLASSNAME;
	CharPaletteCtrl();
	
	Callback1<int> WhenCharSelected;
	
protected:
	virtual void Paint(Draw& w);
	virtual void LeftDown(Point p, dword keyflags);
	
private:
	Vector<int> chars;
	int HitTest(Point p);
};

class CharLabelCtrl : public ParentCtrl {
public:
	typedef CharLabelCtrl CLASSNAME;
	CharLabelCtrl();

	void SetProject(OCRDatasetProject* proj) { project = proj; }
	void SetImageIndex(int index) { image_index = index; }
	void SetSegmentIndex(int index) { segment_index = index; UpdateChar(); }
	void SetCharIndex(int index) { char_index = index; UpdateChar(); }

	void UpdateChar();
	void SetLabel(int code);
	void SkipChar();
	void DeleteChar();
	void LabelSequence(const String& text);
	
	Callback WhenChanged;

private:
	OCRDatasetProject* project;
	int image_index;
	int segment_index;
	int char_index;

	ImageCtrl char_preview;
	Label lbl_label;
	EditString label_edit;
	CharPaletteCtrl palette;
	
	Button btn_prev, btn_next, btn_skip, btn_delete;
	ProgressIndicator progress;
	Label lbl_progress;
	
	void OnPrev();
	void OnNext();
	void OnSkip() { SkipChar(); }
	void OnDelete() { DeleteChar(); }
	void OnLabelEdit();
	void UpdateProgress();
	
	virtual bool Key(dword key, int count);
};

} // namespace OCR

} // namespace Upp

#endif
