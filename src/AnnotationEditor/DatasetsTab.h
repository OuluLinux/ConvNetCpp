#ifndef _AnnotationEditor_DatasetsTab_h_
#define _AnnotationEditor_DatasetsTab_h_

#include "AnnotationEditorCommon.h"

NAMESPACE_UPP
struct DatasetCard : public ParentCtrl {
	Dataset& dataset; int index; Label lbl_name, lbl_images, lbl_creator; Button btn_more;
	DatasetCard(Dataset& d, int i);
	virtual void Paint(Draw& w) override;
	virtual void LeftDown(Point p, dword flags) override;
	Event<int> WhenOpen;
	Event<int> WhenModified;
	Event<int> WhenDelete;
};

class DatasetsTab : public ParentCtrl {
public:
	typedef DatasetsTab CLASSNAME;
	DatasetsTab();
	void SetDatasets(Vector<Dataset>& d);
	void RefreshView();
	Callback WhenCreate, WhenRefresh, WhenChanged; Event<int> WhenOpenDataset, WhenDeleteDataset;
private:
	Label title, subtitle; Button btn_create, btn_import, btn_refresh; ParentCtrl scroll_view, grid_area; ScrollBar sb; Array<DatasetCard> cards; Vector<Dataset>* datasets = nullptr;
	void OnCreate();
	void OnRefresh();
	virtual void Layout() override;
};

struct ImageCell : public ParentCtrl {
	ImageEntry& entry; Image thumbnail; Label lbl_name, lbl_ann; Button btn_select; bool selected = false;
	ImageCell(ImageEntry& e, Image thumb);
	void SetThumbnail(Image img) { thumbnail = img; Refresh(); }
	void SetSelected(bool b);
	void UpdateStatus();
	virtual void Paint(Draw& w) override;
	virtual void LeftDown(Point p, dword flags) override;
	Event<ImageEntry&> WhenClick;
	Event<bool> WhenSelectChanged;
};

class DatasetContentView : public ParentCtrl {
public:
	typedef DatasetContentView CLASSNAME;
	DatasetContentView();
	void SetDataset(Dataset& ds, const Vector<Category>& cats, const Vector<Dataset>& all_ds);
	void ClearDataset();
	Vector<int> GetSelectedImageIndices() const;
	double GetCategoryRejectionRate(int cat_id);
	void UpdatePriorityScores();
	void UpdateStats();
	void OnScroll();
	void RefreshGrid();
	void RefreshThumbnails();
	Image LoadImageByExt(const String& path);
	Image GetThumbnail(const String& path) { if(thumbnails.Find(path) >= 0) return thumbnails.Get(path); Image img = LoadImageByExt(path); if(!img) return Image(); Size sz = img.GetSize(); if(sz.cx > 200 || sz.cy > 200) { double f = 200.0 / max(sz.cx, sz.cy); img = Rescale(img, Size(int(f * sz.cx), int(f * sz.cy))); } thumbnails.Add(path, img); if(thumbnails.GetCount() > 200) thumbnails.Remove(0); return img; }
	void ClearCache() { thumbnails.Clear(); }
	void UpdateSelectionInfo();
	void OnSelectAllVisible();
	void SelectAllImages() { OnSelectAllVisible(); }
	void OnClearSelection();
	void OnDeleteSelected();
	void OnAddImages();
	void OnImportPredictions();

	void SetBaseDir(const String& d) { base_dir_ = d; }

	Callback WhenBack, WhenScan, WhenImportCoco, WhenExportCoco, WhenDatasetChanged; Event<String> WhenImageLoaded; Event<ImageEntry&> WhenImageClick;
	Function<String(const String&)> WhenResolveImagePath;
private:
	String ResolveImagePath(const String& path) const;
	Dataset* dataset = nullptr; const Vector<Category>* categories = nullptr; const Vector<Dataset>* all_datasets = nullptr;
	String base_dir_;
	Label title, lbl_stats, lbl_filter, lbl_sort, lbl_selected; DropList dl_filter, dl_sort; Button btn_back, btn_scan, btn_add_images, btn_import_coco, btn_export_coco, btn_import_ai, btn_select_all, btn_clear_selection, btn_delete_selected; ParentCtrl scroll_view, grid_area; ScrollBar sb; Array<ImageCell> cells; Vector<int> cell_image_indices, visible_image_indices; Index<int> selected_image_indices; VectorMap<String, Image> thumbnails;
	virtual bool Key(dword key, int count) override;
	virtual void Layout() override;
};

END_UPP_NAMESPACE

#endif
