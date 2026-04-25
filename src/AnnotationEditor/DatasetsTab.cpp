#include "DatasetsTab.h"

NAMESPACE_UPP

DatasetCard::DatasetCard(Dataset& d, int i) : dataset(d), index(i) {
	SetFrame(FieldFrame());
	Add(lbl_name.SetLabel(dataset.name).HSizePos(5, 30).TopPos(5, 25));
	lbl_name.SetFont(StdFont().Bold());
	String img_text = dataset.images.GetCount() > 0 ? Format("%d images", dataset.images.GetCount()) : "No images in dataset";
	Add(lbl_images.SetLabel(img_text).HSizePos(5, 5).TopPos(30, 20));
	lbl_images.SetFont(StdFont().Italic());
	Add(lbl_creator.SetLabel("Created by " + dataset.created_by).HSizePos(5, 5).BottomPos(5, 20));
	lbl_creator.SetFont(StdFont(10));
	Add(btn_more.SetLabel("...").RightPos(5, 20).TopPos(5, 20));
	btn_more << [=] {
		MenuBar menu;
		menu.Add("Rename...", [=] {
			String name = dataset.name;
			if(EditText(name, "Rename Dataset", "New name:")) {
				dataset.name = name;
				WhenModified(index);
			}
		});
		menu.Add("Set Folder...", [=] {
			FileSel fs;
			fs.ActiveDir(dataset.folder_path);
			if(fs.ExecuteSelectDir("Select dataset folder")) {
				dataset.folder_path = fs.Get();
				WhenModified(index);
			}
		});
		menu.Separator();
		menu.Add("Delete Dataset", [=] {
			if(PromptOKCancel("Delete dataset '" + dataset.name + "'?"))
				WhenDelete(index);
		});
		menu.Execute();
	};
}

void DatasetCard::Paint(Draw& w) {
	w.DrawRect(GetSize(), SColorPaper());
}

void DatasetCard::LeftDown(Point p, dword flags) {
	WhenOpen(index);
}

DatasetsTab::DatasetsTab() {
	Add(title.SetLabel("Datasets").LeftPos(10, 200).TopPos(10, 30));
	title.SetFont(StdFont(20).Bold());
	Add(subtitle.SetLabel("loaded datasets").LeftPos(10, 200).TopPos(40, 20));
	subtitle.SetFont(StdFont().Italic());
	Add(btn_create.SetLabel("Create").LeftPos(10, 80).TopPos(70, 25));
	Add(btn_import.SetLabel("Import").LeftPos(100, 80).TopPos(70, 25));
	Add(btn_refresh.SetLabel("Refresh").LeftPos(190, 80).TopPos(70, 25));
	Add(scroll_view.HSizePos(10, 10).VSizePos(110, 10));
	scroll_view.SetFrame(FieldFrame());
	scroll_view.Add(grid_area.HSizePos());
	scroll_view.AddFrame(sb);
	sb.WhenScroll = [=] { grid_area.TopPos(-sb, grid_area.GetSize().cy); };
	btn_create << THISBACK(OnCreate);
	btn_import << [=] { PromptOK("Import Dataset not implemented"); };
	btn_refresh << THISBACK(OnRefresh);
}

void DatasetsTab::SetDatasets(Vector<Dataset>& d) {
	datasets = &d;
	RefreshView();
}

void DatasetsTab::RefreshView() {
	for(int i = 0; i < cards.GetCount(); i++)
		cards[i].Remove();
	cards.Clear();
	if(!datasets)
		return;
	int margin = 8;
	int card_h = 100;
	int view_cx = max(100, scroll_view.GetSize().cx - sb.GetSize().cx);
	int card_w = view_cx - 2 * margin;
	int x = margin;
	int y = margin;
	for(int i = 0; i < datasets->GetCount(); i++) {
		auto& element = cards.Add(new DatasetCard((*datasets)[i], i));
		element.WhenOpen = WhenOpenDataset;
		element.WhenModified = [=](int) { RefreshView(); WhenChanged(); };
		element.WhenDelete = WhenDeleteDataset;
		grid_area.Add(element.LeftPos(x, card_w).TopPos(y, card_h));
		y += card_h + margin;
	}
	int total_h = y + margin;
	grid_area.SetRect(0, 0, view_cx, total_h);
	sb.SetTotal(total_h);
	sb.SetPage(scroll_view.GetSize().cy);
}

void DatasetsTab::OnCreate() {
	WhenCreate();
}

void DatasetsTab::OnRefresh() {
	WhenRefresh();
}

void DatasetsTab::Layout() {
	RefreshView();
}

ImageCell::ImageCell(ImageEntry& e, Image thumb) : entry(e), thumbnail(thumb) {
	SetFrame(FieldFrame());
	Add(lbl_name.SetLabel(entry.file_name).HSizePos(5, 5).BottomPos(20, 20));
	lbl_name.SetFont(StdFont(9));
	lbl_name.AlignCenter();
	Add(btn_select.SetLabel("Select").LeftPos(5, 60).TopPos(5, 20));
	btn_select << [=] {
		SetSelected(!selected);
		if(WhenSelectChanged)
			WhenSelectChanged(selected);
	};
	UpdateStatus();
}

void ImageCell::SetSelected(bool b) {
	selected = b;
	btn_select.SetLabel(selected ? "Selected" : "Select");
	Refresh();
}

void ImageCell::UpdateStatus() {
	String status;
	if(entry.reviewed)
		status = "Reviewed";
	else if(!entry.annotations.IsEmpty())
		status = "Annotated";
	else
		status = "Not Annotated";
	lbl_ann.SetLabel(status).HSizePos(5, 5).BottomPos(5, 15);
	lbl_ann.SetFont(StdFont(8).Italic());
	if(entry.reviewed)
		lbl_ann.SetInk(LtBlue());
	else
		lbl_ann.SetInk(entry.annotations.IsEmpty() ? Gray() : Green());
	lbl_ann.AlignCenter();
	if(lbl_ann.GetParent() != this)
		Add(lbl_ann);
}

void ImageCell::Paint(Draw& w) {
	Size sz = GetSize();
	w.DrawRect(sz, SColorPaper());
	if(thumbnail) {
		Size tsz = thumbnail.GetSize();
		w.DrawImage((sz.cx - tsz.cx) / 2, (sz.cy - 40 - tsz.cy) / 2, thumbnail);
	}
	if(entry.priority_score > 1.0 && !entry.reviewed) {
		w.DrawRect(sz.cx - 25, 5, 20, 20, Color(255, 165, 0));
		w.DrawText(sz.cx - 22, 7, "P", StdFont().Bold(), White());
	}
	if(selected) {
		Color c = Color(65, 130, 255);
		w.DrawRect(0, 0, sz.cx, 2, c);
		w.DrawRect(0, sz.cy - 2, sz.cx, 2, c);
		w.DrawRect(0, 0, 2, sz.cy, c);
		w.DrawRect(sz.cx - 2, 0, 2, sz.cy, c);
	}
}

void ImageCell::LeftDown(Point p, dword flags) {
	if(flags & K_ALT) {
		SetSelected(!selected);
		if(WhenSelectChanged)
			WhenSelectChanged(selected);
		return;
	}
	WhenClick(entry);
}

DatasetContentView::DatasetContentView() {
	Add(btn_back.SetLabel("< Back").LeftPos(10, 80).TopPos(10, 25));
	Add(title.LeftPos(100, 400).TopPos(10, 30));
	title.SetFont(StdFont(20).Bold());
	Add(lbl_stats.LeftPos(100, 400).TopPos(40, 20));
	lbl_stats.SetFont(StdFont().Italic());

	Add(lbl_filter.SetLabel("Filter:").RightPos(560, 55).TopPos(10, 25));
	Add(dl_filter.RightPos(450, 105).TopPos(10, 25));
	dl_filter.Add(0, "All");
	dl_filter.Add(1, "Annotated");
	dl_filter.Add(2, "Not Annotated");
	dl_filter.SetData(0);
	dl_filter << THISBACK(RefreshGrid);

	Add(lbl_sort.SetLabel("Sort:").RightPos(390, 55).TopPos(10, 25));
	Add(dl_sort.RightPos(280, 105).TopPos(10, 25));
	dl_sort.Add(0, "Name");
	dl_sort.Add(1, "Priority (AI)");
	dl_sort.Add(2, "Low Confidence");
	dl_sort.Add(3, "Needs Review");
	dl_sort.SetData(0);
	dl_sort << THISBACK(RefreshGrid);

	Add(btn_scan.SetLabel("Scan Folder").LeftPos(10, 100).TopPos(40, 25));
	Add(btn_add_images.SetLabel("Add Images...").LeftPos(115, 110).TopPos(40, 25));
	Add(btn_import_coco.SetLabel("Import COCO").LeftPos(230, 100).TopPos(40, 25));
	Add(btn_export_coco.SetLabel("Export COCO").LeftPos(335, 100).TopPos(40, 25));
	Add(btn_import_ai.SetLabel("Import AI Predictions").LeftPos(440, 150).TopPos(40, 25));
	Add(btn_select_all.SetLabel("Select All").LeftPos(595, 90).TopPos(40, 25));
	Add(btn_clear_selection.SetLabel("Clear Sel").LeftPos(690, 90).TopPos(40, 25));
	Add(btn_delete_selected.SetLabel("Delete Sel").LeftPos(785, 95).TopPos(40, 25));
	Add(lbl_selected.SetLabel("Selected: 0").LeftPos(885, 110).TopPos(44, 20));
	lbl_selected.SetFont(StdFont().Italic());

	Add(scroll_view.HSizePos(10, 10).VSizePos(70, 10));
	scroll_view.SetFrame(FieldFrame());
	scroll_view.Add(grid_area.HSizePos());
	scroll_view.AddFrame(sb);
	sb.WhenScroll = THISBACK(OnScroll);

	btn_back << [=] { WhenBack(); };
	btn_scan << [=] { WhenScan(); };
	btn_add_images << THISBACK(OnAddImages);
	btn_import_coco << [=] { WhenImportCoco(); };
	btn_export_coco << [=] { WhenExportCoco(); };
	btn_import_ai << THISBACK(OnImportPredictions);
	btn_select_all << THISBACK(OnSelectAllVisible);
	btn_clear_selection << THISBACK(OnClearSelection);
	btn_delete_selected << THISBACK(OnDeleteSelected);
}

void DatasetContentView::SetDataset(Dataset& ds, const Vector<Category>& cats, const Vector<Dataset>& all_ds) {
	dataset = &ds;
	categories = &cats;
	all_datasets = &all_ds;
	selected_image_indices.Clear();
	title.SetLabel(dataset->name);
	UpdatePriorityScores();
	UpdateStats();
	RefreshGrid();
}

void DatasetContentView::ClearDataset() {
	dataset = nullptr;
	categories = nullptr;
	all_datasets = nullptr;
	selected_image_indices.Clear();
	title.SetLabel("Dataset");
	lbl_stats.SetLabel("");
	RefreshGrid();
}

String DatasetContentView::ResolveImagePath(const String& path) const {
	if(WhenResolveImagePath) {
		String p = WhenResolveImagePath(path);
		if(!p.IsEmpty())
			return p;
	}
	return path;
}

bool DatasetContentView::Key(dword key, int count) {
	if(key == K_CTRL_A || ((key == 'A' || key == 'a') && GetCtrl())) {
		OnSelectAllVisible();
		return true;
	}
	return ParentCtrl::Key(key, count);
}

void DatasetContentView::Layout() {
	RefreshGrid();
}

Vector<int> DatasetContentView::GetSelectedImageIndices() const {
	Vector<int> out;
	for(int i = 0; i < selected_image_indices.GetCount(); i++)
		out.Add(selected_image_indices[i]);
	return out;
}

double DatasetContentView::GetCategoryRejectionRate(int cat_id) {
	if(!all_datasets)
		return 0.2;
	int total = 0;
	int rejected = 0;
	for(const auto& ds : *all_datasets) {
		for(const auto& ie : ds.images) {
			for(const auto& obj : ie.annotations)
				if(obj.category_id == cat_id && obj.accepted)
					total++;
			for(const auto& obj : ie.rejected_suggestions)
				if(obj.category_id == cat_id) {
					total++;
					rejected++;
				}
		}
	}
	return total > 0 ? (double)rejected / total : 0.2;
}

void DatasetContentView::UpdatePriorityScores() {
	if(!dataset)
		return;
	for(auto& ie : dataset->images) {
		if(ie.reviewed) {
			ie.priority_score = -1.0;
			continue;
		}
		if(!ie.annotations.IsEmpty()) {
			ie.priority_score = 0.0;
			continue;
		}
		if(ie.suggestions.IsEmpty()) {
			ie.priority_score = 0.1;
			continue;
		}
		double score = 1.0;
		score += ie.suggestions.GetCount() * 0.3;
		for(const auto& sug : ie.suggestions) {
			double rej_rate = GetCategoryRejectionRate(sug.category_id);
			score += (1.0 - sug.score) * 2.0 * (1.0 + rej_rate);
		}
		ie.priority_score = score;
	}
}

void DatasetContentView::UpdateStats() {
	if(!dataset)
		return;
	int total = dataset->images.GetCount();
	int annotated = 0;
	int reviewed = 0;
	int issues = 0;
	for(const auto& ie : dataset->images) {
		if(!ie.annotations.IsEmpty())
			annotated++;
		if(ie.reviewed)
			reviewed++;
		for(const auto& obj : ie.annotations) {
			bool has_issue = obj.polygons.IsEmpty() || obj.category_id == -1;
			if(!has_issue) {
				for(const auto& p : obj.polygons) {
					if(p.GetCount() < 3) {
						has_issue = true;
						break;
					}
				}
			}
			if(has_issue)
				issues++;
		}
	}
	double pct = total > 0 ? (double)annotated * 100.0 / total : 0;
	String stats = Format("%d images - %d annotated (%.1f%%)", total, annotated, pct);
	if(reviewed > 0)
		stats << ", " << reviewed << " reviewed";
	if(issues > 0)
		stats << ", " << issues << " issues found";
	lbl_stats.SetLabel(stats);
}

void DatasetContentView::OnScroll() {
	grid_area.TopPos(-sb, grid_area.GetSize().cy);
	RefreshThumbnails();
}

void DatasetContentView::RefreshGrid() {
	for(int i = 0; i < cells.GetCount(); i++)
		cells[i].Remove();
	cells.Clear();
	cell_image_indices.Clear();
	visible_image_indices.Clear();
	if(!dataset)
		return;

	Vector<int> indices;
	for(int i = 0; i < dataset->images.GetCount(); i++)
		indices.Add(i);

	int sort_mode = dl_sort.GetData();
	if(sort_mode == 1) {
		Sort(indices, [&](int a, int b) { return dataset->images[a].priority_score > dataset->images[b].priority_score; });
	}
	else if(sort_mode == 2) {
		Sort(indices, [&](int a, int b) {
			auto GetMinScore = [&](int idx) {
				double ms = 1.0;
				if(dataset->images[idx].suggestions.IsEmpty())
					return 1.1;
				for(const auto& s : dataset->images[idx].suggestions)
					ms = min(ms, s.score);
				return ms;
			};
			return GetMinScore(a) < GetMinScore(b);
		});
	}
	else if(sort_mode == 3) {
		Sort(indices, [&](int a, int b) {
			auto GetReviewScore = [&](int idx) {
				const auto& ie = dataset->images[idx];
				if(ie.reviewed) return -1;
				if(!ie.annotations.IsEmpty()) return 100;
				if(!ie.suggestions.IsEmpty()) return 50;
				return 0;
			};
			int sa = GetReviewScore(a);
			int sb = GetReviewScore(b);
			if(sa != sb)
				return sa > sb;
			return dataset->images[a].file_name < dataset->images[b].file_name;
		});
	}
	else {
		Sort(indices, [&](int a, int b) { return dataset->images[a].file_name < dataset->images[b].file_name; });
	}

	int margin = 8;
	int cell_h = 240;
	int view_cx = max(100, scroll_view.GetSize().cx - sb.GetSize().cx);
	int min_cell_w = 150;
	int cols = max(1, (view_cx - margin) / (min_cell_w + margin));
	int cell_w = (view_cx - margin * (cols + 1)) / cols;
	int x = margin;
	int y = margin;
	int filter = dl_filter.GetData();
	for(int i : indices) {
		bool annotated = !dataset->images[i].annotations.IsEmpty();
		if(filter == 1 && !annotated) continue;
		if(filter == 2 && annotated) continue;
		visible_image_indices.Add(i);

		auto& cell = cells.Add(new ImageCell(dataset->images[i], Image()));
		cell.WhenClick = WhenImageClick;
		cell.WhenSelectChanged = [=](bool is_selected) {
			int fi = selected_image_indices.Find(i);
			if(is_selected) {
				if(fi < 0) selected_image_indices.Add(i);
			}
			else if(fi >= 0) {
				selected_image_indices.Remove(fi);
			}
			UpdateSelectionInfo();
		};
		cell.SetSelected(selected_image_indices.Find(i) >= 0);
		cell_image_indices.Add(i);
		grid_area.Add(cell.LeftPos(x, cell_w).TopPos(y, cell_h));
		x += cell_w + margin;
		if(x + cell_w > view_cx) {
			x = margin;
			y += cell_h + margin;
		}
	}
	int total_h = y + cell_h + margin;
	grid_area.SetRect(0, 0, view_cx, total_h);
	sb.SetTotal(total_h);
	sb.SetPage(scroll_view.GetSize().cy);
	UpdateSelectionInfo();
	RefreshThumbnails();
}

void DatasetContentView::RefreshThumbnails() {
	if(cells.IsEmpty())
		return;
	Rect view(0, sb, scroll_view.GetSize().cx, sb + scroll_view.GetSize().cy);
	for(int i = 0; i < cells.GetCount(); i++) {
		Rect r = cells[i].GetRect();
		if(!r.Intersects(view))
			continue;
		if(!cells[i].thumbnail) {
			int img_idx = (i >= 0 && i < cell_image_indices.GetCount()) ? cell_image_indices[i] : i;
			if(img_idx >= 0 && img_idx < dataset->images.GetCount())
				cells[i].SetThumbnail(GetThumbnail(ResolveImagePath(dataset->images[img_idx].file_path)));
		}
	}
}

Image DatasetContentView::LoadImageByExt(const String& path) {
	String ext = ToLower(GetFileExt(path));
	if(ext == ".jpg" || ext == ".jpeg")
		return JPGRaster().LoadFile(path);
	if(ext == ".png")
		return PNGRaster().LoadFile(path);
	return StreamRaster::LoadFileAny(path);
}

void DatasetContentView::UpdateSelectionInfo() {
	lbl_selected.SetLabel(Format("Selected: %d", selected_image_indices.GetCount()));
}

void DatasetContentView::OnSelectAllVisible() {
	if(!dataset)
		return;
	for(int i = 0; i < visible_image_indices.GetCount(); i++) {
		int idx = visible_image_indices[i];
		if(idx >= 0 && idx < dataset->images.GetCount() && selected_image_indices.Find(idx) < 0)
			selected_image_indices.Add(idx);
	}
	for(int i = 0; i < cells.GetCount() && i < cell_image_indices.GetCount(); i++)
		cells[i].SetSelected(selected_image_indices.Find(cell_image_indices[i]) >= 0);
	UpdateSelectionInfo();
}

void DatasetContentView::OnClearSelection() {
	selected_image_indices.Clear();
	for(int i = 0; i < cells.GetCount(); i++)
		cells[i].SetSelected(false);
	UpdateSelectionInfo();
}

void DatasetContentView::OnDeleteSelected() {
	if(!dataset || selected_image_indices.IsEmpty())
		return;
	Vector<int> idxs;
	for(int i = 0; i < selected_image_indices.GetCount(); i++) {
		int idx = selected_image_indices[i];
		if(idx >= 0 && idx < dataset->images.GetCount())
			idxs.Add(idx);
	}
	if(idxs.IsEmpty())
		return;
	Sort(idxs);
	if(!PromptOKCancel(Format("Delete %d selected image(s) from dataset?", idxs.GetCount())))
		return;
	for(int i = idxs.GetCount() - 1; i >= 0; i--)
		dataset->images.Remove(idxs[i]);
	selected_image_indices.Clear();
	UpdateStats();
	RefreshGrid();
	WhenDatasetChanged();
}

void DatasetContentView::OnAddImages() {
	if(!dataset)
		return;
	FileSel fs;
	fs.Type("Image files", "*.png *.jpg *.jpeg *.bmp");
	fs.Multi();
	fs.ActiveDir(base_dir_.IsEmpty() ? GetHomeDirectory() : base_dir_);
	if(!fs.ExecuteOpen("Add Images"))
		return;
	int added = 0;
	int duplicate = 0;
	int failed = 0;
	for(int i = 0; i < fs.GetCount(); i++) {
		String path = NormalizePath(fs[i]);
		if(path.Find('*') >= 0 || path.Find('?') >= 0) {
			failed++;
			continue;
		}
		bool exists = false;
		for(const auto& e : dataset->images) {
			if(NormalizePath(ResolveImagePath(e.file_path)) == path) {
				exists = true;
				break;
			}
		}
		if(exists) {
			duplicate++;
			continue;
		}
		Image img = LoadImageByExt(path);
		if(!img) {
			failed++;
			continue;
		}
		String stored = path;
		if(!base_dir_.IsEmpty()) {
			String base = NormalizePath(base_dir_);
			if(!base.IsEmpty() && base[base.GetCount() - 1] != '/' && base[base.GetCount() - 1] != '\\')
				base << '/';
			if(path.StartsWith(base))
				stored = path.Mid(base.GetCount());
		}
		ImageEntry& e = dataset->images.Add();
		e.file_path = stored;
		e.file_name = GetFileName(stored);
		if(e.file_name.IsEmpty())
			e.file_name = GetFileName(path);
		e.width = img.GetWidth();
		e.height = img.GetHeight();
		WhenImageLoaded(path);
		added++;
	}
	if(added > 0) {
		UpdateStats();
		RefreshGrid();
		WhenDatasetChanged();
	}
	if(duplicate > 0 || failed > 0)
		PromptOK(Format("Add Images completed.\nAdded: %d\nDuplicates skipped: %d\nFailed to decode: %d", added, duplicate, failed));
}

void DatasetContentView::OnImportPredictions() {
	if(!dataset)
		return;
	FileSel fs;
	fs.Type("Prediction JSON", "*.json");
	fs.ActiveDir(base_dir_.IsEmpty() ? GetHomeDirectory() : base_dir_);
	if(!fs.ExecuteOpen("Import AI Predictions"))
		return;
	String json = LoadFile(fs.Get());
	Value v = ParseJSON(json);
	if(IsNull(v) || !v.Is<ValueArray>()) {
		PromptOK("Invalid prediction format.");
		return;
	}
	int count = 0;
	for(int i = 0; i < v.GetCount(); i++) {
		Value obj = v[i];
		String fname = obj["file_name"];
		for(auto& ie : dataset->images) {
			if(ie.file_name != fname)
				continue;
			AnnotationObject& sug = ie.suggestions.Add();
			static int next_id = 8000;
			sug.id = next_id++;
			sug.category_id = (int)obj["category_id"];
			sug.score = (double)obj["score"];
			sug.name = "AI Suggestion " + AsString(ie.suggestions.GetCount());
			Value seg = obj["segmentation"];
			if(seg.Is<ValueArray>()) {
				Vector<Pointf>& poly = sug.polygons.Add();
				for(int j = 0; j < seg.GetCount(); j += 2)
					poly.Add(Pointf((double)seg[j], (double)seg[j + 1]));
			}
			sug.UpdateBBox();
			count++;
			break;
		}
	}
	UpdatePriorityScores();
	RefreshGrid();
	PromptOK(Format("Imported %d suggestions.", count));
}

END_UPP_NAMESPACE
