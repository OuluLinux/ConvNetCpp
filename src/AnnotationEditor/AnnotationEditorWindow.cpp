#include "AnnotationEditorWindow.h"

NAMESPACE_UPP

namespace {

struct CardMetaDialog : TopWindow {
	Label      lbl_hero;
	Label      lbl_board;
	EditString edit_hero;
	EditString edit_board;
	Button     btn_ok;
	Button     btn_cancel;

	CardMetaDialog() {
		Title("Element Data");
		SetRect(0, 0, 320, 130);

		Add(lbl_hero.SetLabel("Hero cards:").LeftPos(10, 90).TopPos(10, 22));
		Add(edit_hero.RightPos(10, 200).TopPos(10, 22));
		Add(lbl_board.SetLabel("Board cards:").LeftPos(10, 90).TopPos(40, 22));
		Add(edit_board.RightPos(10, 200).TopPos(40, 22));
		Add(btn_ok.SetLabel("OK").RightPos(94, 80).BottomPos(10, 24));
		Add(btn_cancel.SetLabel("Cancel").RightPos(10, 80).BottomPos(10, 24));

		edit_hero.WhenEnter = [=] { edit_board.SetFocus(); };
		edit_board.WhenEnter = [=] { AcceptBreak(IDOK); };
		Acceptor(btn_ok, IDOK);
		Rejector(btn_cancel, IDCANCEL);
	}
};

}

AnnotationEditorWindow::AnnotationEditorWindow() {
	Title("AnnotationEditor");
	Sizeable().Zoomable();
	SetRect(0, 0, 1000, 700);
	AddFrame(menu);
	menu.Set(THISBACK(MainMenu));
	datasets_tab.SetDatasets(datasets);
	datasets_tab.WhenCreate = THISBACK(OnCreateDataset);
	datasets_tab.WhenRefresh = THISBACK(RefreshDatasetsView);
	datasets_tab.WhenOpenDataset = THISBACK(OpenDataset);
	datasets_tab.WhenChanged = THISBACK(MarkDirty);
	datasets_tab.WhenDeleteDataset = [=](int idx) {
		if(idx >= 0 && idx < datasets.GetCount()) {
			datasets.Remove(idx);
			RefreshDatasetsView();
			MarkDirty();
		}
	};
	content_view.WhenBack = THISBACK(CloseDataset);
	content_view.WhenScan = THISBACK(OnScanDataset);
	content_view.WhenImageLoaded = [=](String path) { undo_tab.Log("Image loaded: " + GetFileName(path)); };
	content_view.WhenImageClick = THISBACK(OpenAnnotateView);
	content_view.WhenDatasetChanged = THISBACK(MarkDirty);
	content_view.WhenImportCoco = THISBACK(OnImportCoco);
	content_view.WhenExportCoco = THISBACK(OnExportCoco);
	content_view.WhenResolveImagePath = [=](const String& path) { return ResolveProjectImagePath(path); };
	annotate_view.SetCategories(categories);
	annotate_view.WhenBack = THISBACK(CloseAnnotateView);
	annotate_view.WhenLog = [=](String s) { undo_tab.Log(s); };
	annotate_view.WhenResolveImagePath = [=](const String& path) { return ResolveProjectImagePath(path); };
	annotate_view.WhenCategoriesChanged = [=] { categories_tab.RefreshView(); MarkDirty(); };
	annotate_view.WhenCommandExecuted = [=] { undo_tab.RefreshHistory(); MarkDirty(); };
	annotate_view.WhenOpenImage = THISBACK(OpenAnnotateViewAt);
	annotate_view.WhenDirty = THISBACK(MarkDirty);
	annotate_view.WhenCopyHintToLastMluiScript = [=](const AnnotationObject& obj) { CopyObjectHintToLastActiveMluiScript(obj); };
	annotate_view.WhenBBoxAdded = [=](int object_id) {
		if(!mlui_wizard.IsEnabled()) return;
		if(current_dataset_index < 0 || current_dataset_index >= datasets.GetCount()) return;
		int img_idx = annotate_view.GetCurrentImageIndex();
		if(img_idx < 0 || img_idx >= datasets[current_dataset_index].images.GetCount()) return;
		ImageEntry& ie = datasets[current_dataset_index].images[img_idx];

		int obj_idx = -1;
		for(int i = 0; i < ie.annotations.GetCount(); i++) {
			if(ie.annotations[i].id == object_id) { obj_idx = i; break; }
		}
		if(obj_idx < 0) return;

		MluiScriptSlot slot;
		if(!mlui_wizard.GetCurrentSlot(slot)) return;

		if(!mlui_wizard.IsOverwriteEnabled()) {
			int guard = 0;
			while(guard < current_script_.slots.GetCount() && mlui_wizard.IsCurrentSlotFilled()) {
				if(!mlui_wizard.SkipCurrent()) break;
				guard++;
			}
			if(!mlui_wizard.GetCurrentSlot(slot)) return;
			if(mlui_wizard.IsCurrentSlotFilled()) return;
		}

		int cat_id = EnsureMluiSlotCategory(slot.category);
		AnnotationObject& obj = ie.annotations[obj_idx];
		if(cat_id >= 0) {
			obj.category_id = cat_id;
			annotate_view.SetActiveCategoryById(cat_id);
			for(const auto& c : categories)
				if(c.id == cat_id) { obj.color = c.color; break; }
		}
		obj.name = slot.label;
		obj.slot_id = slot.slot_id;
		int mdi = obj.metadata.Find(MluiSlotIdKey());
		if(mdi >= 0) obj.metadata[mdi] = slot.slot_id;
		else obj.metadata.Add(MluiSlotIdKey(), slot.slot_id);

		if(mlui_wizard.IsOverwriteEnabled()) {
			for(int i = ie.annotations.GetCount() - 1; i >= 0; i--) {
				if(i == obj_idx) continue;
				if(ie.annotations[i].metadata.Get(MluiSlotIdKey(), "") == slot.slot_id) {
					ie.annotations.Remove(i);
					if(i < obj_idx) obj_idx--;
				}
			}
			ie.has_annotations = !ie.annotations.IsEmpty();
		}

		annotate_view.SelectById(obj.id);
		annotate_view.RefreshObjectTree();
		annotate_view.Refresh();
		MarkDirty();
		mlui_wizard.RefreshFillState();
		mlui_wizard.SkipCurrent();
	};
	annotate_view.SetDatasetsPtr(datasets);
	annotate_view.SetCommandManager(cmdmgr);
	undo_tab.SetCommandManager(cmdmgr);
	mlui_wizard.WhenCheckFill = [=](const String& slot_id) -> bool {
		if(current_dataset_index < 0 || current_dataset_index >= datasets.GetCount()) return false;
		if(annotate_view.GetImagePath().IsEmpty()) return false;
		int img_idx = annotate_view.GetCurrentImageIndex();
		if(img_idx < 0 || img_idx >= datasets[current_dataset_index].images.GetCount()) return false;
		const ImageEntry& ie = datasets[current_dataset_index].images[img_idx];
		for(const auto& obj : ie.annotations)
			if(obj.metadata.Get(MluiSlotIdKey(), "") == slot_id) return true;
		return false;
	};
	mlui_wizard.WhenApply = [=](const MluiScriptSlot& slot) {
		int cat_id = EnsureMluiSlotCategory(slot.category);
		if(cat_id >= 0)
			annotate_view.SetActiveCategoryById(cat_id);
		if(mlui_wizard.IsOverwriteEnabled() &&
		   current_dataset_index >= 0 && current_dataset_index < datasets.GetCount()) {
			int img_idx = annotate_view.GetCurrentImageIndex();
			if(img_idx >= 0 && img_idx < datasets[current_dataset_index].images.GetCount()) {
				ImageEntry& ie = datasets[current_dataset_index].images[img_idx];
				for(int i = ie.annotations.GetCount() - 1; i >= 0; i--) {
					if(ie.annotations[i].metadata.Get(MluiSlotIdKey(), "") == slot.slot_id)
						ie.annotations.Remove(i);
				}
				ie.has_annotations = !ie.annotations.IsEmpty();
			}
		}
		MluiScript single;
		single.name = current_script_.name;
		single.slots.Add(slot);
		int added = annotate_view.ApplyMluiScript(single);
		if(added > 0) {
			MarkDirty();
			mlui_wizard.RefreshFillState();
		}
	};
	mlui_wizard.WhenCopyHints = [=] {
		OnCopyMluiHintsFromImage();
		ReloadCurrentScript();
		mlui_wizard.RefreshFillState();
	};
	mlui_wizard.WhenEnable = [=](bool enabled) {
		if(!enabled) return;
		MluiScriptSlot slot;
		if(!mlui_wizard.GetCurrentSlot(slot)) return;
		int cat_id = EnsureMluiSlotCategory(slot.category);
		if(cat_id >= 0) annotate_view.SetActiveCategoryById(cat_id);
	};
	mlui_wizard.WhenSlotFocus = [=](String slot_id) {
		if(mlui_wizard.IsEnabled()) {
			int si = current_script_.FindSlot(slot_id);
			if(si >= 0 && si < current_script_.slots.GetCount()) {
				int cat_id = EnsureMluiSlotCategory(current_script_.slots[si].category);
				if(cat_id >= 0) annotate_view.SetActiveCategoryById(cat_id);
			}
		}
		if(current_dataset_index < 0 || current_dataset_index >= datasets.GetCount()) return;
		int img_idx = annotate_view.GetCurrentImageIndex();
		if(img_idx < 0 || img_idx >= datasets[current_dataset_index].images.GetCount()) return;
		const ImageEntry& ie = datasets[current_dataset_index].images[img_idx];
		for(const auto& obj : ie.annotations) {
			if(obj.metadata.Get(MluiSlotIdKey(), "") == slot_id) {
				annotate_view.SelectById(obj.id);
				return;
			}
		}
	};
	dock_ann_objects.Title("Object List").SizeHint(Size(420, 260)).SetGroup("ann_edit_panels");
	dock_ann_categories.Title("Image Categories").SizeHint(Size(320, 250)).SetGroup("ann_edit_panels");
	dock_ann_settings.Title("Image Tool Settings").SizeHint(Size(360, 560)).SetGroup("ann_edit_panels");
	dock_ann_metadata.Title("Image Metadata").SizeHint(Size(380, 420)).SetGroup("ann_edit_panels");
	dock_mlui_wizard.Title("MLUI Wizard").SizeHint(Size(360, 280)).SetGroup("ann_edit_panels");
	dock_annlay_train.Title("Slot Trainer").SizeHint(Size(800, 400));
	Register(dock_ann_metadata);
	dock_ann_objects.Add(annotate_view.GetObjectsPanel().SizePos());
	dock_ann_categories.Add(annotate_view.GetCategoriesPanel().SizePos());
	dock_ann_settings.Add(annotate_view.GetSettingsPanel().SizePos());
	dock_ann_metadata.Add(annotate_view.GetMetadataPanel().SizePos());
	dock_mlui_wizard.Add(mlui_wizard.SizePos());
	dock_annlay_train.Add(annlay_train_panel.SizePos());
	categories_tab.SetCategories(categories);
	categories_tab.SetDatasets(datasets);
	categories_tab.WhenChange << [=] { annotate_view.RefreshCategoryList(); MarkDirty(); };
	categories_tab.WhenLog << [=](String s) { undo_tab.Log(s); };
	Add(content_view.SizePos());
	SetTimeCallback(-30000, THISBACK(AutoSave), 1);
	PostCallback(THISBACK(CheckRecovery));
	LoadRecentProjects();
}

void AnnotationEditorWindow::DockInit() {
	ApplyDefaultDockLayout();
	CacheDefaultLayout();
	String layout_path = GetHomeDirectory() + "/.config/annotation_editor_layout.dat";
	FileIn in(layout_path);
	if(in.IsOpen())
		SerializeWindow(in);
	FileIn pin(GetPlacementPath());
	if(pin.IsOpen())
		SerializePlacement(pin);
}

void AnnotationEditorWindow::ApplyDefaultDockLayout() {
	DockLeft(Dockable(datasets_tab, "Datasets").SizeHint(Size(260, 400)));
	DockLeft(Dockable(categories_tab, "Categories").SizeHint(Size(260, 300)));
	DockBottom(Dockable(undo_tab, "Log").SizeHint(Size(600, 120)));
	DockBottom(Dockable(tasks_tab, "Tasks").SizeHint(Size(600, 120)));
	DockBottom(Dockable(dock_annlay_train, "Slot Trainer").SizeHint(Size(800, 400)));
	DockRight(Dockable(dock_ann_objects, "Object List").SizeHint(Size(420, 260)));
	DockRight(Dockable(dock_ann_categories, "Image Categories").SizeHint(Size(320, 250)));
	DockRight(Dockable(dock_ann_settings, "Image Tool Settings").SizeHint(Size(360, 560)));
	DockRight(dock_ann_metadata);
	DockRight(Dockable(dock_mlui_wizard, "MLUI Wizard").SizeHint(Size(360, 280)));
	DockRight(Dockable(quality_tab, "AI Quality").SizeHint(Size(260, 300)));
	TabDockGroup(DOCK_RIGHT, "ann_edit_panels");
}

void AnnotationEditorWindow::CacheDefaultLayout() {
	StringStream out;
	SerializeWindow(out);
	default_layout_data = out.GetResult();
}

void AnnotationEditorWindow::SetDefaultLayout() {
	if(default_layout_data.IsEmpty())
		return;
	StringStream in(default_layout_data);
	SerializeWindow(in);
}

bool AnnotationEditorWindow::CloseOpenMluiScriptEditors() {
	for(int i = 0; i < open_mlui_editors_.GetCount(); i++) {
		MluiScriptEditor* ed = open_mlui_editors_[i];
		if(!ed)
			continue;
		if(ed->IsOpen()) {
			ed->Close();
			if(ed->IsOpen())
				return false;
		}
	}
	return true;
}

void AnnotationEditorWindow::Close() {
	if(!CloseOpenMluiScriptEditors())
		return;
	String config_dir = GetHomeDirectory() + "/.config";
	if(!DirectoryExists(config_dir))
		RealizeDirectory(config_dir);
	String layout_path = config_dir + "/annotation_editor_layout.dat";
	FileOut out(layout_path);
	if(out.IsOpen())
		SerializeWindow(out);
	FileOut pout(GetPlacementPath());
	if(pout.IsOpen())
		SerializePlacement(pout);
	SaveProjectSessionState();
	TopWindow::Close();
}

void AnnotationEditorWindow::MarkDirty() {
	project_dirty = true;
	quality_tab.UpdateMetrics(datasets, categories);
}

void AnnotationEditorWindow::AutoSave() {
	if(!project_dirty)
		return;
	String path = GetAutoSavePath();
	Project p;
	p.categories <<= categories;
	p.datasets <<= datasets;
	p.last_saved = GetSysTime();
	p.is_autosave = true;
	p.autosave_source_project_path = project_path;
	p.last_dataset_index = GetSavedProjectDatasetIndex();
	p.last_image_index = GetSavedProjectImageIndex();
	if(!last_active_mlui_script_path.IsEmpty())
		p.last_mlui_script_path = StoreMluiScriptPath(last_active_mlui_script_path);
	else if(p.last_dataset_index >= 0 && p.last_dataset_index < datasets.GetCount())
		p.last_mlui_script_path = datasets[p.last_dataset_index].mlui_script_path;
	if(SaveAsJSON(p, path, false))
		undo_tab.Log("Autosaved to " + path);
}

String AnnotationEditorWindow::GetAutoSavePath() {
	return GetAutoSavePathForProject(project_path);
}

String AnnotationEditorWindow::GetAutoSavePathForProject(const String& prj_path) {
	String name = prj_path.IsEmpty() ? "untitled" : GetFileTitle(prj_path);
	return AppendFileName(GetTempInternalDirectory(), name + ".autosave.annprj");
}

String AnnotationEditorWindow::GetTempInternalDirectory() {
	String p = AppendFileName(GetHomeDirectory(), ".gemini/tmp/annotation_editor");
	RealizeDirectory(p);
	return p;
}

void AnnotationEditorWindow::CheckRecovery() {
	if(!CommandLine().IsEmpty()) {
		for(const String& s : CommandLine())
			if(s.StartsWith("--test-"))
				return;
	}
	String path = GetAutoSavePath();
	if(!FileExists(path))
		return;
	Project ap;
	if(!LoadFromJSON(ap, path) || !ap.is_autosave) {
		DeleteFile(path);
		return;
	}
	String source_project = ap.autosave_source_project_path;
	if(source_project.IsEmpty())
		source_project = project_path;
	if(!source_project.IsEmpty() && FileExists(source_project)) {
		Time auto_tm = FileGetTime(path);
		Time proj_tm = FileGetTime(source_project);
		if(!IsNull(auto_tm) && !IsNull(proj_tm) && auto_tm <= proj_tm) {
			DeleteFile(path);
			return;
		}
	}
	if(PromptYesNo("Recovered autosave found. Restore previous session?")) {
		LoadProject(path);
		project_dirty = true;
	}
	else {
		DeleteFile(path);
	}
}

void AnnotationEditorWindow::OnSaveSnapshot() {
	if(project_path.IsEmpty()) {
		PromptOK("Save project first before taking snapshots.");
		return;
	}
	Time t = GetSysTime();
	String ts = Format("%04d%02d%02d_%02d%02d%02d", t.year, t.month, t.day, t.hour, t.minute, t.second);
	String path = project_path + ".snapshot_" + ts + ".annprj";
	Project p;
	p.categories <<= categories;
	p.datasets <<= datasets;
	p.last_saved = GetSysTime();
	if(SaveAsJSON(p, path, true)) {
		undo_tab.Log("Snapshot saved to " + path);
		PromptOK("Snapshot saved to: " + path);
	}
	else {
		PromptOK("Failed to save snapshot");
	}
}

void AnnotationEditorWindow::CreateSampleDatasets() {
	Dataset d1;
	d1.name = "Sample Dataset 1";
	d1.folder_path = "/tmp";
	datasets.Add(pick(d1));
	RefreshDatasetsView();
}

void AnnotationEditorWindow::CreateSampleCategories() {
	if(categories.GetCount() > 0)
		return;
	Category& c1 = categories.Add();
	c1.id = 1;
	c1.name = "Person";
	c1.color = Red();
	c1.keypoint_labels.Add("nose");
	c1.keypoint_connects_to.Add("");
	c1.keypoint_labels.Add("left_eye");
	c1.keypoint_connects_to.Add("nose");
	c1.keypoint_labels.Add("right_eye");
	c1.keypoint_connects_to.Add("nose");
	Category& c2 = categories.Add();
	c2.id = 2;
	c2.name = "Car";
	c2.color = Blue();
}

void AnnotationEditorWindow::OpenDataset(int index) {
	if(index < 0 || index >= datasets.GetCount())
		return;
	current_dataset_index = index;
	content_view.ClearCache();
	content_view.SetBaseDir(ProjectDir());
	content_view.SetDataset(datasets[index], categories, datasets);
	ReloadCurrentScript();
	annotate_view.Remove();
	content_view.Remove();
	Add(content_view.SizePos());
}

void AnnotationEditorWindow::CloseDataset() {
	content_view.Remove();
	mlui_wizard.ClearScript();
	content_view.ClearDataset();
	datasets_tab.RefreshView();
}

void AnnotationEditorWindow::OnScanDataset() {
	if(current_dataset_index < 0)
		return;
	ScanDataset(datasets[current_dataset_index]);
	content_view.SetBaseDir(ProjectDir());
	content_view.SetDataset(datasets[current_dataset_index], categories, datasets);
}

void AnnotationEditorWindow::ScanDataset(Dataset& ds) {
	String scan_root = ds.folder_path;
	if(scan_root.IsEmpty())
		scan_root = ".";
	if(!IsFullPath(scan_root))
		scan_root = AppendFileName(ProjectDir(), scan_root);
	scan_root = NormalizePath(scan_root);
	undo_tab.Log("Scan started for " + ds.name + " @ " + scan_root);
	if(!DirectoryExists(scan_root)) {
		PromptOK("Invalid folder: " + scan_root);
		return;
	}

	ds.images.Clear();
	Index<String> seen;
	std::function<void(const String&)> scan_dir;
	scan_dir = [&](const String& dir) {
		FindFile ff;
		if(!ff.Search(AppendFileName(dir, "*")))
			return;
		do {
			String name = ff.GetName();
			if(name == "." || name == "..")
				continue;
			String path = NormalizePath(ff.GetPath());
			if(ff.IsFolder()) {
				scan_dir(path);
				continue;
			}
			if(!ff.IsFile())
				continue;
			String ext = ToLower(GetFileExt(name));
			if(ext != ".png" && ext != ".jpg" && ext != ".jpeg" && ext != ".bmp")
				continue;
			if(seen.Find(path) >= 0)
				continue;
			seen.Add(path);

			Image img;
			if(ext == ".jpg" || ext == ".jpeg")
				img = JPGRaster().LoadFile(path);
			else if(ext == ".png")
				img = PNGRaster().LoadFile(path);
			else
				img = StreamRaster::LoadFileAny(path);
			if(!img)
				continue;

			String stored = StoreProjectImagePath(path);
			ImageEntry& e = ds.images.Add();
			e.file_path = stored;
			e.file_name = GetFileName(stored);
			if(e.file_name.IsEmpty())
				e.file_name = GetFileName(path);
			e.width = img.GetWidth();
			e.height = img.GetHeight();
		}
		while(ff.Next());
	};
	scan_dir(scan_root);
	undo_tab.Log(Format("Scan finished: %d images", ds.images.GetCount()));
}

void AnnotationEditorWindow::OpenAnnotateView(ImageEntry& ie) {
	if(current_dataset_index < 0)
		return;
	Dataset& ds = datasets[current_dataset_index];
	for(int i = 0; i < ds.images.GetCount(); i++) {
		if(&ds.images[i] == &ie) {
			OpenAnnotateViewAt(i);
			return;
		}
	}
}

void AnnotationEditorWindow::OpenAnnotateViewAt(int img_idx) {
	if(current_dataset_index < 0)
		return;
	if(categories.IsEmpty())
		CreateSampleCategories();
	annotate_view.SetDataset(datasets[current_dataset_index], img_idx);
	ReloadCurrentScript();
	mlui_wizard.RefreshFillState();
	content_view.Remove();
	annotate_view.Remove();
	Add(annotate_view.SizePos());
}

void AnnotationEditorWindow::CloseAnnotateView() {
	annotate_view.Remove();
	content_view.Remove();
	if(current_dataset_index >= 0 && current_dataset_index < datasets.GetCount()) {
		content_view.SetBaseDir(ProjectDir());
		content_view.SetDataset(datasets[current_dataset_index], categories, datasets);
	}
	else {
		content_view.ClearDataset();
	}
	Add(content_view.SizePos());
}

void AnnotationEditorWindow::AutoOpenFirstImage() {
	if(current_dataset_index >= 0 && current_dataset_index < datasets.GetCount())
		if(datasets[current_dataset_index].images.GetCount() > 0)
			OpenAnnotateViewAt(0);
}

void AnnotationEditorWindow::CreateSampleObjects() {
	if(current_dataset_index == -1 || datasets[current_dataset_index].images.IsEmpty())
		return;
	ImageEntry& e = datasets[current_dataset_index].images[0];
	AnnotationObject obj1;
	obj1.id = 101;
	obj1.category_id = 1;
	obj1.name = "Person 101";
	obj1.color = Red();
	obj1.label_visibility_state = 2;
	Vector<Pointf>& p1 = obj1.polygons.Add();
	p1.Add(Pointf(10, 10));
	p1.Add(Pointf(100, 10));
	p1.Add(Pointf(100, 100));
	p1.Add(Pointf(10, 100));
	obj1.UpdateBBox();
	e.annotations.Add(pick(obj1));
	AnnotationObject obj2;
	obj2.id = 102;
	obj2.category_id = 2;
	obj2.name = "Car 102";
	obj2.color = Blue();
	obj2.label_visibility_state = 0;
	obj2.visible = false;
	Vector<Pointf>& p2 = obj2.polygons.Add();
	p2.Add(Pointf(150, 150));
	p2.Add(Pointf(250, 150));
	p2.Add(Pointf(250, 250));
	p2.Add(Pointf(150, 250));
	obj2.UpdateBBox();
	e.annotations.Add(pick(obj2));
	e.has_annotations = true;
}

void AnnotationEditorWindow::OnNewProject() {
	if(!PromptOKCancel("Start new project? All unsaved changes will be lost."))
		return;
	datasets.Clear();
	categories.Clear();
	mlui_category_create_declined_.Clear();
	cmdmgr.Clear();
	current_dataset_index = -1;
	current_script_ = MluiScript();
	mlui_wizard.ClearScript();
	project_path = "";
	last_active_mlui_script_path.Clear();
	RefreshAllUI();
	undo_tab.Log("Created new project");
}

void AnnotationEditorWindow::OnOpenProject() {
	FileSel fs;
	fs.Type("Annotation Project", "*.annprj");
	fs.ActiveDir(ProjectDir());
	if(fs.ExecuteOpen("Open Project"))
		LoadProject(fs.Get());
}

void AnnotationEditorWindow::OnSaveProject() {
	if(project_path.IsEmpty())
		OnSaveProjectAs();
	else
		SaveProject(project_path);
}

void AnnotationEditorWindow::OnSaveProjectAs() {
	FileSel fs;
	fs.Type("Annotation Project", "*.annprj");
	fs.ActiveDir(ProjectDir());
	if(fs.ExecuteSaveAs("Save Project As")) {
		project_path = fs.Get();
		SaveProject(project_path);
	}
}

void AnnotationEditorWindow::SaveProject(const String& path) {
	Project p;
	p.categories <<= categories;
	p.datasets <<= datasets;
	p.last_saved = GetSysTime();
	p.is_autosave = false;
	p.autosave_source_project_path.Clear();
	p.last_dataset_index = GetSavedProjectDatasetIndex();
	p.last_image_index = GetSavedProjectImageIndex();
	if(!last_active_mlui_script_path.IsEmpty())
		p.last_mlui_script_path = StoreMluiScriptPath(last_active_mlui_script_path);
	else if(p.last_dataset_index >= 0 && p.last_dataset_index < datasets.GetCount())
		p.last_mlui_script_path = datasets[p.last_dataset_index].mlui_script_path;
	if(SaveAsJSON(p, path, true)) {
		project_dirty = false;
		String as = GetAutoSavePathForProject(path);
		if(FileExists(as))
			DeleteFile(as);
		String untitled_as = GetAutoSavePathForProject(String());
		if(as != untitled_as && FileExists(untitled_as))
			DeleteFile(untitled_as);
		undo_tab.Log("Saved project to " + path);
	}
	else {
		PromptOK("Failed to save project");
	}
}

void AnnotationEditorWindow::LoadProject(const String& path) {
	Project p;
	if(!LoadFromJSON(p, path)) {
		PromptOK("Failed to load project");
		return;
	}
	categories <<= p.categories;
	mlui_category_create_declined_.Clear();
	datasets <<= p.datasets;
	cmdmgr.Clear();
	String effective_path = path;
	if(p.is_autosave && !p.autosave_source_project_path.IsEmpty())
		effective_path = NormalizePath(p.autosave_source_project_path);
	project_path = effective_path;
	last_active_mlui_script_path = p.last_mlui_script_path.IsEmpty() ? String() : ResolveMluiScriptPath(p.last_mlui_script_path);
	RefreshAllUI();

	int ds_idx = p.last_dataset_index;
	int img_idx = p.last_image_index;
	int s_ds_idx = -1;
	int s_img_idx = -1;
	String s_mlui_path;
	if(LoadProjectSessionState(effective_path, s_ds_idx, s_img_idx, &s_mlui_path)) {
		ds_idx = s_ds_idx;
		img_idx = s_img_idx;
		if(!s_mlui_path.IsEmpty())
			last_active_mlui_script_path = ResolveMluiScriptPath(s_mlui_path);
	}
	if(ds_idx < 0 || ds_idx >= datasets.GetCount())
		ds_idx = datasets.IsEmpty() ? -1 : 0;
	if(ds_idx >= 0 && ds_idx < datasets.GetCount()) {
		OpenDataset(ds_idx);
		if(datasets[ds_idx].mlui_script_path.IsEmpty() && !p.last_mlui_script_path.IsEmpty())
			datasets[ds_idx].mlui_script_path = p.last_mlui_script_path;
		if(last_active_mlui_script_path.IsEmpty() && !datasets[ds_idx].mlui_script_path.IsEmpty())
			RememberLastActiveMluiScript(datasets[ds_idx].mlui_script_path);
		if(img_idx >= 0 && img_idx < datasets[ds_idx].images.GetCount())
			OpenAnnotateViewAt(img_idx);
	}

	ReloadCurrentScript();
	undo_tab.Log("Loaded project from " + effective_path + (p.is_autosave ? " (autosave restore)" : ""));
	if(!effective_path.IsEmpty())
		AddRecentProject(effective_path);
	project_dirty = false;
}

void AnnotationEditorWindow::RefreshAllUI() {
	CloseAnnotateView();
	datasets_tab.RefreshView();
	categories_tab.RefreshView();
	annotate_view.RefreshCategoryList();
	quality_tab.UpdateMetrics(datasets, categories);
	undo_tab.RefreshHistory();
	ReloadCurrentScript();
}

void AnnotationEditorWindow::OnImportCoco() {
	if(current_dataset_index < 0)
		return;
	FileSel fs;
	fs.Type("COCO JSON", "*.json");
	fs.ActiveDir(ProjectDir());
	if(fs.ExecuteOpen("Import COCO")) {
		if(CocoImporter::Import(datasets[current_dataset_index], categories, fs.Get())) {
			undo_tab.Log("Imported COCO from " + fs.Get());
			RefreshAllUI();
			MarkDirty();
		}
		else {
			PromptOK("Failed to import COCO");
		}
	}
}

void AnnotationEditorWindow::OnExportCoco() {
	if(current_dataset_index < 0)
		return;
	FileSel fs;
	fs.Type("COCO JSON", "*.json");
	fs.ActiveDir(ProjectDir());
	if(fs.ExecuteSaveAs("Export COCO")) {
		if(CocoExporter::Export(datasets[current_dataset_index], categories, fs.Get()))
			undo_tab.Log("Exported COCO to " + fs.Get());
		else
			PromptOK("Failed to export COCO");
	}
}

String AnnotationEditorWindow::RecentProjectsPath() {
	return GetHomeDirectory() + "/.config/annotation_editor_recent.txt";
}

void AnnotationEditorWindow::SaveProjectSessionState() {
	if(project_path.IsEmpty())
		return;
	ProjectSessionState s;
	s.project_path = NormalizePath(project_path);
	s.last_dataset_index = GetSavedProjectDatasetIndex();
	s.last_image_index = GetSavedProjectImageIndex();
	if(!last_active_mlui_script_path.IsEmpty())
		s.last_mlui_script_path = StoreMluiScriptPath(last_active_mlui_script_path);
	SaveAsJSON(s, GetProjectSessionPath(project_path), false);
}

bool AnnotationEditorWindow::LoadProjectSessionState(const String& path, int& ds_idx, int& img_idx, String* last_mlui_path) {
	ProjectSessionState s;
	if(!LoadFromJSON(s, GetProjectSessionPath(path)))
		return false;
	if(!s.project_path.IsEmpty() && NormalizePath(s.project_path) != NormalizePath(path))
		return false;
	ds_idx = s.last_dataset_index;
	img_idx = s.last_image_index;
	if(last_mlui_path)
		*last_mlui_path = s.last_mlui_script_path;
	return true;
}

void AnnotationEditorWindow::LoadRecentProjects() {
	recent_projects_.Clear();
	FileIn f(RecentProjectsPath());
	if(!f.IsOpen())
		return;
	String line;
	while(!f.IsEof()) {
		line = f.GetLine();
		if(!line.IsEmpty() && FileExists(line))
			recent_projects_.Add(line);
		if(recent_projects_.GetCount() >= 10)
			break;
	}
}

void AnnotationEditorWindow::SaveRecentProjects() {
	String config_dir = GetHomeDirectory() + "/.config";
	if(!DirectoryExists(config_dir))
		RealizeDirectory(config_dir);
	FileOut f(RecentProjectsPath());
	for(const String& p : recent_projects_)
		f.PutLine(p);
}

void AnnotationEditorWindow::AddRecentProject(const String& path) {
	for(int i = 0; i < recent_projects_.GetCount(); i++) {
		if(recent_projects_[i] == path) {
			recent_projects_.Remove(i);
			break;
		}
	}
	recent_projects_.Insert(0, path);
	if(recent_projects_.GetCount() > 10)
		recent_projects_.SetCount(10);
	SaveRecentProjects();
}

int AnnotationEditorWindow::FindCategoryIdByName(const String& name) const {
	if(name.IsEmpty())
		return -1;
	for(int i = 0; i < categories.GetCount(); i++)
		if(categories[i].name == name)
			return categories[i].id;
	return -1;
}

int AnnotationEditorWindow::GetNextCategoryId() const {
	int max_id = 0;
	for(int i = 0; i < categories.GetCount(); i++)
		max_id = max(max_id, categories[i].id);
	return max_id + 1;
}

int AnnotationEditorWindow::EnsureMluiSlotCategory(const String& category_name) {
	String cname = TrimBoth(category_name);
	if(cname.IsEmpty())
		return -1;
	int found_id = FindCategoryIdByName(cname);
	if(found_id >= 0)
		return found_id;
	String key = ToLower(cname);
	if(mlui_category_create_declined_.Find(key) >= 0)
		return -1;
	if(!PromptYesNo(Format("MLUI slot category \"%s\" does not exist.\nCreate it now?", cname))) {
		mlui_category_create_declined_.FindAdd(key);
		return -1;
	}

	Category cat;
	cat.id = GetNextCategoryId();
	cat.name = cname;
	cat.color = Color(rand() % 200, rand() % 200, rand() % 200);
	categories.Add(pick(cat));
	categories_tab.RefreshView();
	annotate_view.RefreshCategoryList();
	annotate_view.SetActiveCategoryById(categories.Top().id);
	MarkDirty();
	undo_tab.Log("Created category from MLUI slot: " + categories.Top().name);
	return categories.Top().id;
}

void AnnotationEditorWindow::ReloadCurrentScript() {
	current_script_ = MluiScript();
	if(current_dataset_index < 0 || current_dataset_index >= datasets.GetCount()) {
		mlui_wizard.ClearScript();
		annotate_view.ClearMluiScript();
		return;
	}
	const String& sp = datasets[current_dataset_index].mlui_script_path;
	if(sp.IsEmpty()) {
		mlui_wizard.ClearScript();
		annotate_view.ClearMluiScript();
		return;
	}
	String resolved = ResolveMluiScriptPath(sp);
	if(!LoadMluiScript(current_script_, resolved)) {
		mlui_wizard.ClearScript();
		annotate_view.ClearMluiScript();
		return;
	}
	mlui_wizard.SetScript(current_script_);
	annotate_view.SetMluiScript(current_script_);
}

void AnnotationEditorWindow::MainMenu(Bar& bar) {
	bar.Add("File", THISBACK(MenuFile));
	bar.Add("Project", THISBACK(MenuProject));
	bar.Add("Edit", THISBACK(MenuEdit));
	bar.Add("Tools", THISBACK(MenuTools));
	bar.Add("View", THISBACK(MenuView));
	bar.Sub("Windows", [=](Bar& b) {
		DockWindowMenu(b);
		b.Separator();
		b.Add("Show Image Metadata", [=] {
			DockRight(dock_ann_metadata);
			ActivateDockable(dock_ann_metadata);
		});
		b.Add("Slot Trainer", [=] {
			DockBottom(dock_annlay_train);
			ActivateDockable(dock_annlay_train);
		});
		b.Add("Set Default layout", THISBACK(SetDefaultLayout));
	});
	bar.Add("Help", THISBACK(MenuHelp));
}

void AnnotationEditorWindow::MenuFile(Bar& bar) {
	bar.Add("New Project", THISBACK(OnNewProject)).Key(K_CTRL_N);
	bar.Add("Open Project...", THISBACK(OnOpenProject)).Key(K_CTRL_O);
	bar.Add("Save Project", THISBACK(OnSaveProject)).Key(K_CTRL_S);
	bar.Add("Save Project As...", THISBACK(OnSaveProjectAs));
	bar.Add("Save Snapshot...", THISBACK(OnSaveSnapshot));
	bar.Separator();
	bar.Add("New Dataset...", THISBACK(OnCreateDataset));
	bar.Sub("Recent Files", [=](Bar& sub) {
		if(recent_projects_.IsEmpty()) {
			sub.Add("No recent files", [] {}).Enable(false);
			return;
		}
		for(const String& p : recent_projects_)
			sub.Add(GetFileName(p), [=] { LoadProject(p); });
	});
	bar.Separator();
	bar.Add("Exit", [=] { Break(); });
}

void AnnotationEditorWindow::MenuProject(Bar& bar) {
	bar.Add("Previous Image", [=] { annotate_view.PrevImage(); }).Key(K_F2);
	bar.Add("Next Image", [=] { annotate_view.NextImage(); }).Key(K_F3);
	bar.Add("Edit Element Metadata", THISBACK(OnEditCardMetadata)).Key(K_F4);
	bar.Separator();
	bar.Add("Make project standalone", THISBACK(OnMakeProjectStandalone));
}

void AnnotationEditorWindow::MenuEdit(Bar& bar) {
	bar.Add("Undo", THISBACK(OnGlobalUndo)).Key(K_CTRL_Z);
	bar.Add("Redo", THISBACK(OnGlobalRedo)).Key(K_CTRL_Y);
	bar.Add("Select All", THISBACK(OnSelectAllCurrentView)).Key(K_CTRL_A);
	bar.Separator();
	bar.Add("Set Geometry...", THISBACK(OnSetGeometry));
	bar.Separator();
	bar.Add("MLUI Script...", THISBACK(OnOpenMluiScriptEditor));
	bar.Add("Skip", THISBACK(OnMluiWizardSkip)).Key(K_CTRL_1);
	bar.Add("Apply Last MLUI Script To Image", THISBACK(OnApplyLastMluiScriptToImage)).Key(K_F5);
	bar.Add("Copy hints from current image", THISBACK(OnCopyMluiHintsFromImage));
}

void AnnotationEditorWindow::MenuTools(Bar& bar) {
	String t = annotate_view.GetToolName();
	bar.Add("Select", [=] { annotate_view.SetToolByName("select"); }).Key('1').Check(t == "select");
	bar.Add("BBox", [=] { annotate_view.SetToolByName("bbox"); }).Key('2').Check(t == "bbox");
	bar.Add("Polygon", [=] { annotate_view.SetToolByName("polygon"); }).Key('3').Check(t == "polygon");
	bar.Add("Brush", [=] { annotate_view.SetToolByName("brush"); }).Key('4').Check(t == "brush");
	bar.Add("Eraser", [=] { annotate_view.SetToolByName("eraser"); }).Key('5').Check(t == "eraser");
	bar.Add("Wand", [=] { annotate_view.SetToolByName("wand"); }).Key('6').Check(t == "wand");
	bar.Add("Keypoint", [=] { annotate_view.SetToolByName("keypoint"); }).Key('7').Check(t == "keypoint");
	bar.Separator();
	bar.Add("Clipboard Image Collector...", THISBACK(OnClipboardImageCollector));
}

void AnnotationEditorWindow::OnClipboardImageCollector() {
	if(current_dataset_index < 0 || current_dataset_index >= datasets.GetCount()) {
		PromptOK("Open a dataset first.");
		return;
	}
	ClipboardImageCollectorDialog dlg;
	dlg.WhenCaptureImage = [=](const Image& img, const String& signature, String* out_path) {
		return AddClipboardImageToCurrentDataset(img, signature, out_path);
	};
	dlg.Run();
}

void AnnotationEditorWindow::OnGlobalUndo() {
	annotate_view.OnUndo();
}

void AnnotationEditorWindow::OnGlobalRedo() {
	annotate_view.OnRedo();
}

void AnnotationEditorWindow::OnSetGeometry() {
	annotate_view.OnSetGeometry();
}

void AnnotationEditorWindow::OnSelectAllCurrentView() {
	if(IsAnnotateViewActive()) {
		annotate_view.SelectAllObjects();
		return;
	}
	if(IsContentViewActive()) {
		content_view.SelectAllImages();
		return;
	}
}

void AnnotationEditorWindow::OnEditCardMetadata() {
	ImageEntry* entry = annotate_view.GetCurrentEntry();
	if(!entry)
		return;

	CardMetaDialog dlg;
	dlg.edit_hero.SetData(entry->hero_cards);
	dlg.edit_board.SetData(entry->board_cards);
	dlg.CenterScreen();
	dlg.edit_hero.SetFocus();
	if(dlg.Execute() != IDOK)
		return;

	entry->EnsureMetadataDefaults();
	entry->hero_cards = ~dlg.edit_hero;
	entry->board_cards = ~dlg.edit_board;
	entry->SyncLegacyToImageMetadata();
	annotate_view.RefreshImageMetadataUI();
	MarkDirty();
}

void AnnotationEditorWindow::OnMakeProjectStandalone() {
	if(project_path.IsEmpty()) {
		PromptOK("Save project first before making it standalone.");
		return;
	}
	int total_images = 0;
	for(const auto& ds : datasets)
		total_images += ds.images.GetCount();
	if(total_images <= 0) {
		PromptOK("Project has no images.");
		return;
	}
	String images_dir = AppendFileName(ProjectDir(), "images");
	if(!PromptOKCancel("Copy all project images to:\n" + images_dir +
	                   "\nas JPEG (quality 99) and rewrite project image paths?"))
		return;
	RealizeDirectory(images_dir);
	if(!DirectoryExists(images_dir)) {
		PromptOK("Failed to create images directory:\n" + images_dir);
		return;
	}

	VectorMap<String, String> source_to_rel;
	Index<String> used_rel;
	int converted = 0, reused = 0, failed = 0, updated = 0;
	Vector<String> errors;
	for(int di = 0; di < datasets.GetCount(); di++) {
		Dataset& ds = datasets[di];
		for(int ii = 0; ii < ds.images.GetCount(); ii++) {
			ImageEntry& ie = ds.images[ii];
			String src_abs = ResolveProjectImagePath(ie.file_path);
			if(src_abs.IsEmpty()) {
				failed++;
				errors.Add(Format("%s/%s: empty image path", ds.name, ie.file_name));
				continue;
			}
			String rel_out;
			int m = source_to_rel.Find(src_abs);
			if(m >= 0) {
				rel_out = source_to_rel[m];
				reused++;
			}
			else {
				Image img = StreamRaster::LoadFileAny(src_abs);
				if(!img) {
					failed++;
					errors.Add(Format("%s/%s: failed to load '%s'", ds.name, ie.file_name, src_abs));
					continue;
				}
				String seed = ie.file_name;
				if(seed.IsEmpty())
					seed = GetFileName(src_abs);
				String stem = SanitizeFileStem(GetFileTitle(seed));
				if(stem.IsEmpty())
					stem = Format("image_%d_%d", di + 1, ii + 1);
				String rel_candidate = AppendFileName("images", stem + ".jpg");
				String out_abs = ResolveProjectImagePath(rel_candidate);
				int seq = 2;
				while(used_rel.Find(rel_candidate) >= 0 ||
				      (FileExists(out_abs) && NormalizePath(out_abs) != NormalizePath(src_abs))) {
					rel_candidate = AppendFileName("images", stem + "_" + AsString(seq++) + ".jpg");
					out_abs = ResolveProjectImagePath(rel_candidate);
				}
				if(!JPGEncoder().Quality(99).SaveFile(out_abs, img)) {
					failed++;
					errors.Add(Format("%s/%s: failed to write '%s'", ds.name, ie.file_name, out_abs));
					continue;
				}
				rel_out = rel_candidate;
				source_to_rel.Add(src_abs, rel_out);
				used_rel.FindAdd(rel_out);
				converted++;
				ie.width = img.GetWidth();
				ie.height = img.GetHeight();
			}
			String stored = StoreProjectImagePath(rel_out);
			String fname = GetFileName(stored);
			if(ie.file_path != stored || ie.file_name != fname)
				updated++;
			ie.file_path = stored;
			ie.file_name = fname;
		}
	}

	content_view.ClearCache();
	if(current_dataset_index >= 0 && current_dataset_index < datasets.GetCount()) {
		int img_idx = annotate_view.GetCurrentImageIndex();
		if(IsAnnotateViewActive() && img_idx >= 0 && img_idx < datasets[current_dataset_index].images.GetCount())
			annotate_view.SetDataset(datasets[current_dataset_index], img_idx);
		else {
			content_view.SetBaseDir(ProjectDir());
			content_view.SetDataset(datasets[current_dataset_index], categories, datasets);
		}
	}
	if(updated > 0 || converted > 0)
		SaveProject(project_path);
	undo_tab.Log(Format("Standalone export: %d converted, %d reused, %d failed", converted, reused, failed));
	String msg = Format("Standalone export completed.\nConverted: %d\nReused: %d\nFailed: %d", converted, reused, failed);
	if(!errors.IsEmpty())
		msg << "\n\nFirst error:\n" << errors[0];
	PromptOK(msg);
}

void AnnotationEditorWindow::OnMluiWizardSkip() {
	mlui_wizard.SkipCurrent();
}

void AnnotationEditorWindow::OnApplyLastMluiScriptToImage() {
	if(!IsAnnotateViewActive() || annotate_view.GetImagePath().IsEmpty()) {
		PromptOK("Open an image in annotation view first.");
		return;
	}
	String resolved = GetLastActiveMluiScriptPath();
	if(resolved.IsEmpty()) {
		PromptOK("No active MLUI script. Open MLUI Script Editor and load a script first.");
		return;
	}
	MluiScript s;
	if(!LoadMluiScript(s, resolved)) {
		PromptOK("Failed to load MLUI script:\n" + resolved);
		return;
	}
	RememberLastActiveMluiScript(resolved);
	int added = annotate_view.ApplyMluiScript(s);
	if(added > 0) {
		current_script_ = s;
		mlui_wizard.SetScript(current_script_);
		annotate_view.SetMluiScript(current_script_);
		mlui_wizard.RefreshFillState();
		MarkDirty();
	}
	undo_tab.Log(Format("Applied MLUI script '%s' to image: %d object(s) added", GetFileName(resolved), added));
}

void AnnotationEditorWindow::OnOpenMluiScriptEditor() {
	MluiScriptEditor* ed = new MluiScriptEditor();
	open_mlui_editors_.Add(ed);
	ed->SetProjectDir(ProjectDir());
	Vector<String> cat_names;
	for(const auto& c : categories)
		cat_names.Add(c.name);
	ed->SetCategoryNames(cat_names);
	annotate_view.PopulateScriptEditorReference(*ed);
	ed->WhenApply = [=](const MluiScript& script, int iw, int ih) {
		annotate_view.ApplyMluiScript(script);
		current_script_ = script;
		mlui_wizard.SetScript(current_script_);
		annotate_view.SetMluiScript(current_script_);
		mlui_wizard.RefreshFillState();
		MarkDirty();
	};
	ed->WhenLoadScript = [=](String path) {
		RememberLastActiveMluiScript(path);
		if(current_dataset_index >= 0 && current_dataset_index < datasets.GetCount()) {
			datasets[current_dataset_index].mlui_script_path = StoreMluiScriptPath(path);
			ReloadCurrentScript();
			mlui_wizard.RefreshFillState();
			MarkDirty();
		}
	};
	int ds_idx = current_dataset_index;
	String load_script_path;
	if(ds_idx >= 0 && ds_idx < datasets.GetCount()) {
		const String& sp = datasets[ds_idx].mlui_script_path;
		if(!sp.IsEmpty())
			load_script_path = ResolveMluiScriptPath(sp);
	}
	if(load_script_path.IsEmpty() && !last_active_mlui_script_path.IsEmpty())
		load_script_path = ResolveMluiScriptPath(last_active_mlui_script_path);
	if(!load_script_path.IsEmpty() && FileExists(load_script_path))
		ed->LoadScript(load_script_path);
	ed->Open();
}

void AnnotationEditorWindow::OnCopyMluiHintsFromImage() {
	int ds_idx = current_dataset_index;
	if(ds_idx < 0 || ds_idx >= datasets.GetCount()) {
		PromptOK("No dataset open.");
		return;
	}
	String& sp = datasets[ds_idx].mlui_script_path;
	String resolved = ResolveMluiScriptPath(sp);
	if(sp.IsEmpty()) {
		FileSel fs;
		fs.Type("MLUI Script (*.mlui)", "*.mlui");
		fs.DefaultExt("mlui");
		if(!fs.ExecuteSaveAs())
			return;
		resolved = fs.Get();
		sp = StoreMluiScriptPath(resolved);
	}
	RememberLastActiveMluiScript(resolved);
	MluiScript s;
	LoadMluiScript(s, resolved);
	int updated = annotate_view.UpdateScriptHintsFromImage(s);
	if(SaveMluiScript(s, resolved)) {
		RememberLastActiveMluiScript(resolved);
		ReloadCurrentScript();
		mlui_wizard.RefreshFillState();
		MarkDirty();
		PromptOK(Format("Updated %d bbox hint(s) and saved to:\n%s", updated, resolved));
	}
	else {
		PromptOK("Save failed: " + resolved);
	}
}

void AnnotationEditorWindow::ShowDatasetsView() {}
void AnnotationEditorWindow::ShowAnnotateView() {}
void AnnotationEditorWindow::ShowCategoriesView() {}

void AnnotationEditorWindow::MenuView(Bar& bar) {
	bar.Add("Datasets", THISBACK(ShowDatasetsView));
	bar.Add("Annotations", THISBACK(ShowAnnotateView));
	bar.Add("Categories", THISBACK(ShowCategoriesView));
	bar.Separator();
	bar.Add("Refresh All", THISBACK(RefreshDatasetsView));
}

void AnnotationEditorWindow::MenuHelp(Bar& bar) {
	bar.Add("About", [=] { PromptOK("AnnotationEditor v0.1"); });
}

void AnnotationEditorWindow::OnCreateDataset() {
	CreateDatasetDialog dlg;
	if(!project_path.IsEmpty())
		dlg.SetProjectDir(GetFileDirectory(project_path));
	if(dlg.Run() == IDOK) {
		datasets.Add(pick(dlg.GetDataset()));
		undo_tab.Log("Created dataset: " + datasets.Top().name);
		RefreshDatasetsView();
	}
}

void AnnotationEditorWindow::RefreshDatasetsView() {
	datasets_tab.RefreshView();
	undo_tab.Log("Refreshed datasets view");
}

void AnnotationEditorWindow::RegisterFocusActions() {
		MLUI::GetFocusPage("annotation_canvas")
			.ActionHandler("select_tool", [this](const ValueMap& args) -> Value {
				String tool = args["tool"].ToString();
				annotate_view.SetToolByName(tool);
				ValueMap r; r.Add("ok", true); r.Add("tool", annotate_view.GetToolName()); return r;
			})
			.ActionHandler("select_object", [this](const ValueMap& args) -> Value {
				int id = (int)args["id"];
				annotate_view.SelectById(id);
				ValueMap r; r.Add("ok", true); r.Add("selected_id", id); return r;
			})
			.ActionHandler("delete_selected", [this](const ValueMap&) -> Value {
				int deleted_id = annotate_view.GetSelectedId();
				annotate_view.OnGeneralDelete();
				ValueMap r; r.Add("ok", true); r.Add("deleted_id", deleted_id); return r;
			})
			.ActionHandler("undo",  [this](const ValueMap&) -> Value {
				annotate_view.OnUndo();
				ValueMap r; r.Add("ok", true); return r;
			})
			.ActionHandler("redo",  [this](const ValueMap&) -> Value {
				annotate_view.OnRedo();
				ValueMap r; r.Add("ok", true); return r;
			})
			.ActionHandler("center_view", [this](const ValueMap&) -> Value {
				annotate_view.CenterImage();
				ValueMap r; r.Add("ok", true); return r;
			})
				.ActionHandler("apply_script_slot", [this](const ValueMap& args) -> Value {
					String slot_id = args["slot_id"].ToString();
					double x = args["x"], y = args["y"], w = args["w"], h = args["h"];
					// Find slot in active script and apply with given bbox
					if(current_dataset_index >= 0 && current_dataset_index < datasets.GetCount()) {
						MluiScript script;
						String sp = ResolveMluiScriptPath(datasets[current_dataset_index].mlui_script_path);
						if(!sp.IsEmpty() && LoadMluiScript(script, sp)) {
							// Temporarily override bbox_hint for this slot
							int si = script.FindSlot(slot_id);
						if(si >= 0) {
							int iw = max(1, annotate_view.GetImageWidth());
							int ih = max(1, annotate_view.GetImageHeight());
							script.slots[si].bbox_hint = Rectf(x/iw, y/ih, (x+w)/iw, (y+h)/ih);
						}
					}
					// Create single-slot script and apply
					MluiScript single; single.name = "slot_apply";
						if(script.FindSlot(slot_id) >= 0) {
							single.slots.Add(script.GetSlot(slot_id));
							int added = annotate_view.ApplyMluiScript(single);
							if(added > 0) {
								MarkDirty();
								mlui_wizard.RefreshFillState();
							}
						}
					}
					ValueMap r; r.Add("ok", true); r.Add("slot_id", slot_id); return r;
				});

		MLUI::GetFocusPage("image_navigation")
			.ActionHandler("prev_image",  [this](const ValueMap&) -> Value {
				annotate_view.PrevImage();
				ValueMap r; r.Add("ok", true); r.Add("image_index", annotate_view.GetCurrentImageIndex());
				r.Add("image_path", annotate_view.GetImagePath()); return r;
			})
			.ActionHandler("next_image",  [this](const ValueMap&) -> Value {
				annotate_view.NextImage();
				ValueMap r; r.Add("ok", true); r.Add("image_index", annotate_view.GetCurrentImageIndex());
				r.Add("image_path", annotate_view.GetImagePath()); return r;
			})
			.ActionHandler("jump_to_best",[this](const ValueMap&) -> Value {
				annotate_view.JumpToBestImage();
				ValueMap r; r.Add("ok", true); r.Add("image_index", annotate_view.GetCurrentImageIndex());
				r.Add("image_path", annotate_view.GetImagePath()); return r;
			})
			.ActionHandler("open_image",  [this](const ValueMap& args) -> Value {
				int idx = (int)args["index"];
				if(current_dataset_index >= 0) OpenAnnotateViewAt(idx);
				ValueMap r; r.Add("ok", true); r.Add("image_index", idx);
				r.Add("image_path", annotate_view.GetImagePath()); return r;
			})
			.ActionHandler("next_unannotated", [this](const ValueMap&) -> Value {
				if(current_dataset_index < 0 || current_dataset_index >= datasets.GetCount())
					{ ValueMap r; r.Add("ok", false); r.Add("reason", "no dataset open"); return r; }
				Dataset& ds = datasets[current_dataset_index];
				for(int i = 0; i < ds.images.GetCount(); i++) {
					if(!ds.images[i].has_annotations) {
						OpenAnnotateViewAt(i);
						ValueMap r; r.Add("ok", true);
						r.Add("image_index", i);
						r.Add("image_path", ds.images[i].file_path);
						return r;
					}
				}
				ValueMap r; r.Add("ok", false); r.Add("reason", "all images annotated"); return r;
			});

		MLUI::GetFocusPage("object_list")
			.ActionHandler("select_row", [this](const ValueMap& args) -> Value {
				int id = (int)args["id"];
				annotate_view.SelectById(id);
				ValueMap r; r.Add("ok", true); r.Add("selected_id", id); return r;
			})
			.ActionHandler("delete_object", [this](const ValueMap& args) -> Value {
				int id = (int)args["id"];
				annotate_view.DeleteObjectById(id);
				ValueMap r; r.Add("ok", true); r.Add("deleted_id", id); return r;
			})
			.ActionHandler("add_object", [this](const ValueMap& args) -> Value {
				annotate_view.SelectById(-1);
				annotate_view.OnSetGeometry();
				ValueMap r; r.Add("ok", true); return r;
			})
			.ActionHandler("add_bbox", [this](const ValueMap& args) -> Value {
				ValueMap r;
				if(annotate_view.GetImagePath().IsEmpty())
					{ r.Add("ok", false); r.Add("reason", "no image open"); return r; }
				double x = args["x"], y = args["y"], w = args["w"], h = args["h"];
				if(w <= 0 || h <= 0)
					{ r.Add("ok", false); r.Add("reason", "w and h must be > 0"); return r; }
				AnnotationObject obj;
				obj.id = annotate_view.GetNextObjectId();
				obj.name = args["name"].ToString();
				if(obj.name.IsEmpty()) obj.name = "object";
				obj.category_id = args.Find("category_id") >= 0 ? (int)args["category_id"] : -1;
				// If category_id not given, try to find by name in categories
				if(obj.category_id < 0 && !obj.name.IsEmpty()) {
					for(const auto& c : categories)
						if(c.name == obj.name) { obj.category_id = c.id; break; }
				}
				Vector<Pointf> pts;
				pts.Add(Pointf(x, y)); pts.Add(Pointf(x+w, y));
				pts.Add(Pointf(x+w, y+h)); pts.Add(Pointf(x, y+h));
				obj.polygons.Add(pick(pts));
				obj.UpdateBBox();
				int new_id = obj.id;
				annotate_view.AddAnnotationObject(pick(obj));
				r.Add("ok", true);
				r.Add("id", new_id);
				r.Add("name", args["name"].ToString().IsEmpty() ? String("object") : args["name"].ToString());
				return r;
			})
			.ActionHandler("get_annotations", [this](const ValueMap&) -> Value {
				if(!annotate_view.GetImagePath().IsEmpty() &&
				   current_dataset_index >= 0 && current_dataset_index < datasets.GetCount()) {
					int img_idx = annotate_view.GetCurrentImageIndex();
					Dataset& ds = datasets[current_dataset_index];
					if(img_idx >= 0 && img_idx < ds.images.GetCount()) {
						const ImageEntry& ie = ds.images[img_idx];
						ValueArray out;
						for(const auto& obj : ie.annotations) {
							ValueMap o;
							o.Add("id",          obj.id);
							o.Add("name",        obj.name);
							o.Add("category_id", obj.category_id);
							// bbox as [x,y,w,h]
							ValueArray bb;
							bb.Add(obj.bbox.left); bb.Add(obj.bbox.top);
							bb.Add(obj.bbox.Width()); bb.Add(obj.bbox.Height());
							o.Add("bbox", bb);
							// metadata as flat object
							ValueMap meta;
							for(int k = 0; k < obj.metadata.GetCount(); k++)
								meta.Add(obj.metadata.GetKey(k), obj.metadata[k]);
							o.Add("metadata", meta);
							out.Add(o);
						}
						return out;
					}
				}
				return ValueArray();
			});

			MLUI::GetFocusPage("mlui_script")
				.ActionHandler("apply_script", [this](const ValueMap&) -> Value {
					ValueMap r;
					if(current_dataset_index < 0) { r.Add("ok", false); r.Add("reason", "no dataset open"); return r; }
					String sp = ResolveMluiScriptPath(datasets[current_dataset_index].mlui_script_path);
					MluiScript s;
					if(!sp.IsEmpty() && LoadMluiScript(s, sp)) {
						int added = annotate_view.ApplyMluiScript(s);
						r.Add("ok", true); r.Add("slots_added", added);
						if(added > 0) {
							MarkDirty();
							mlui_wizard.RefreshFillState();
						}
					} else {
					r.Add("ok", false); r.Add("reason", "failed to load script");
				}
				return r;
			})
				.ActionHandler("list_slots", [this](const ValueMap&) -> Value {
					ValueArray out;
					if(current_dataset_index >= 0) {
						String sp = ResolveMluiScriptPath(datasets[current_dataset_index].mlui_script_path);
						MluiScript s;
						if(!sp.IsEmpty() && LoadMluiScript(s, sp)) {
							Vector<String> unfilled = annotate_view.GetUnfilledSlots(s);
						for(const auto& slot : s.slots) {
							ValueMap m;
							m.Add("slot_id", slot.slot_id);
							m.Add("label", slot.label);
							m.Add("category", slot.category);
							m.Add("hint", slot.hint);
							m.Add("required", slot.required);
							bool filled = true; for(const String& u : unfilled) if(u == slot.slot_id) { filled = false; break; }
						m.Add("filled", filled);
							out.Add(m);
						}
					}
				}
				return out;
			})
			.ActionHandler("copy_hints_from_image", [this](const ValueMap&) -> Value {
				OnCopyMluiHintsFromImage();
				ValueMap r; r.Add("ok", true); return r;
			})
			.ActionHandler("bulk_apply_script", [this](const ValueMap&) -> Value {
				ValueMap r;
				if(current_dataset_index < 0 || current_dataset_index >= datasets.GetCount())
					{ r.Add("ok", false); r.Add("reason", "no dataset open"); return r; }
					Dataset& ds = datasets[current_dataset_index];
					if(ds.mlui_script_path.IsEmpty())
						{ r.Add("ok", false); r.Add("reason", "no mlui_script_path set on dataset"); return r; }
					String resolved = ResolveMluiScriptPath(ds.mlui_script_path);
					MluiScript script;
					if(!LoadMluiScript(script, resolved))
						{ r.Add("ok", false); r.Add("reason", "failed to load script: " + resolved); return r; }

				int images_updated = 0;
				int slots_added = 0;
				// Remember current image so we can restore it
				int saved_idx = annotate_view.GetCurrentImageIndex();

				for(int i = 0; i < ds.images.GetCount(); i++) {
					// Temporarily point annotate_view at this image for ApplyMluiScript
					// We do it directly on the ImageEntry to avoid full UI reload overhead
					ImageEntry& entry = ds.images[i];
					int img_w = entry.width  > 0 ? entry.width  : 1371; // fallback
					int img_h = entry.height > 0 ? entry.height : 1014;
					int added = 0;
					int next_id = 1;
					for(const auto& obj : entry.annotations)    next_id = max(next_id, obj.id + 1);
					for(const auto& obj : entry.suggestions)    next_id = max(next_id, obj.id + 1);

					for(const auto& slot : script.slots) {
						bool found = false;
						for(const auto& obj : entry.annotations) {
							if(obj.metadata.Get(MluiSlotIdKey(), "") == slot.slot_id)
								{ found = true; break; }
						}
						if(found && !slot.allow_multiple) continue;

						AnnotationObject obj;
						obj.id          = next_id++;
						obj.name        = slot.label;
						obj.category_id = -1;
						for(const auto& c : categories)
							if(c.name == slot.category) { obj.category_id = c.id; break; }
						obj.metadata.Add(MluiSlotIdKey(), slot.slot_id);

						if(slot.bbox_hint.Width() > 0 && slot.bbox_hint.Height() > 0) {
							double x0 = slot.bbox_hint.left   * img_w;
							double y0 = slot.bbox_hint.top    * img_h;
							double x1 = slot.bbox_hint.right  * img_w;
							double y1 = slot.bbox_hint.bottom * img_h;
							Vector<Pointf> pts;
							pts.Add(Pointf(x0,y0)); pts.Add(Pointf(x1,y0));
							pts.Add(Pointf(x1,y1)); pts.Add(Pointf(x0,y1));
							obj.polygons.Add(pick(pts));
							obj.UpdateBBox();
						}
						entry.annotations.Add(pick(obj));
						entry.has_annotations = true;
						added++;
					}
					if(added > 0) { images_updated++; slots_added += added; }
				}

				MarkDirty();
				// Refresh the currently open image's object tree if it was affected
				if(saved_idx >= 0 && saved_idx < ds.images.GetCount())
					annotate_view.RefreshAfterBulkEdit(ds.images[saved_idx]);

				r.Add("ok", true);
				r.Add("images_updated", images_updated);
				r.Add("slots_added", slots_added);
				return r;
			});

		MLUI::GetFocusPage("project")
			.ActionHandler("load_project", [this](const ValueMap& args) -> Value {
				String path = args["path"].ToString();
				ValueMap r;
				if(path.IsEmpty() || !FileExists(path))
					{ r.Add("ok", false); r.Add("reason", "file not found: " + path); return r; }
				LoadProject(path);
				r.Add("ok", true);
				r.Add("project_path", project_path);
				r.Add("dataset_count", datasets.GetCount());
				return r;
			})
			.ActionHandler("save_project", [this](const ValueMap&) -> Value {
				ValueMap r;
				if(project_path.IsEmpty())
					{ r.Add("ok", false); r.Add("reason", "no project path; use save_project with path arg or load first"); return r; }
				OnSaveProject();
				r.Add("ok", true);
				r.Add("project_path", project_path);
				return r;
			})
			.ActionHandler("open_dataset", [this](const ValueMap& args) -> Value {
				int idx = (int)args["index"];
				ValueMap r;
				if(idx < 0 || idx >= datasets.GetCount())
					{ r.Add("ok", false); r.Add("reason", "index out of range"); return r; }
				OpenDataset(idx);
				AutoOpenFirstImage();
				r.Add("ok", true);
				r.Add("dataset_index", idx);
				r.Add("dataset_name", datasets[idx].name);
				r.Add("image_count", datasets[idx].images.GetCount());
				return r;
			})
				.ActionHandler("set_script_path", [this](const ValueMap& args) -> Value {
					String path = args["path"].ToString();
					ValueMap r;
					if(current_dataset_index < 0 || current_dataset_index >= datasets.GetCount())
						{ r.Add("ok", false); r.Add("reason", "no dataset open"); return r; }
					String resolved = ResolveMluiScriptPath(path);
					if(!FileExists(resolved))
						{ r.Add("ok", false); r.Add("reason", "file not found: " + resolved); return r; }
					datasets[current_dataset_index].mlui_script_path = StoreMluiScriptPath(resolved);
					RememberLastActiveMluiScript(resolved);
					ReloadCurrentScript();
					mlui_wizard.RefreshFillState();
					MarkDirty();
					// Count slots for confirmation
					MluiScript s; int slot_count = 0;
					if(LoadMluiScript(s, resolved)) slot_count = s.slots.GetCount();
					r.Add("ok", true);
					r.Add("script_path", datasets[current_dataset_index].mlui_script_path);
					r.Add("slot_count", slot_count);
					return r;
				});
	}

bool AnnotationEditorWindow::Key(dword key, int count) {
	if(key == K_CTRL_S || ((key == 'S' || key == 's') && GetCtrl())) { OnSaveProject(); return true; }
	if(key == K_CTRL_A || ((key == 'A' || key == 'a') && GetCtrl())) { OnSelectAllCurrentView(); return true; }
	if(key == K_F2) { annotate_view.PrevImage(); return true; }
	if(key == K_F3) { annotate_view.NextImage(); return true; }
	if(key == K_F4) { OnEditCardMetadata(); return true; }
	if(key == K_CTRL_1) { OnMluiWizardSkip(); return true; }
	if(key == K_F5) { OnApplyLastMluiScriptToImage(); return true; }
	if(IsAnnotateViewActive()) {
		if(key == '1') { annotate_view.SetToolByName("select"); return true; }
		if(key == '2') { annotate_view.SetToolByName("bbox"); return true; }
		if(key == '3') { annotate_view.SetToolByName("polygon"); return true; }
		if(key == '4') { annotate_view.SetToolByName("brush"); return true; }
		if(key == '5') { annotate_view.SetToolByName("eraser"); return true; }
		if(key == '6') { annotate_view.SetToolByName("wand"); return true; }
		if(key == '7') { annotate_view.SetToolByName("keypoint"); return true; }
	}

	if(key == K_CTRL_Z) { annotate_view.OnUndo(); return true; }
	if(key == K_CTRL_Y) { annotate_view.OnRedo(); return true; }
	if(key == K_LEFT) { annotate_view.PrevImage(); return true; }
	if(key == K_RIGHT) { annotate_view.NextImage(); return true; }
	return DockWindow::Key(key, count);
}

String AnnotationEditorWindow::GetPlacementPath() const {
	return GetHomeDirectory() + "/.config/annotation_editor_window_placement.dat";
}

String AnnotationEditorWindow::GetProjectSessionPath(const String& path) const {
	String config_dir = GetHomeDirectory() + "/.config";
	RealizeDirectory(config_dir);
	return AppendFileName(config_dir, Format("annotation_editor_session_%08x.json", (int)GetHashValue(NormalizePath(path))));
}

String AnnotationEditorWindow::ProjectDir() const {
	return project_path.IsEmpty() ? GetHomeDirectory() : GetFileDirectory(project_path);
}

String AnnotationEditorWindow::ResolveProjectImagePath(const String& path) const {
	if(path.IsEmpty()) return path;
	if(IsFullPath(path)) return NormalizePath(path);
	return NormalizePath(AppendFileName(ProjectDir(), path));
}

String AnnotationEditorWindow::StoreProjectImagePath(const String& path) const {
	if(path.IsEmpty()) return path;
	String full = ResolveProjectImagePath(path);
	String base = NormalizePath(ProjectDir());
	if(!base.IsEmpty() && base[base.GetCount() - 1] != '/' && base[base.GetCount() - 1] != '\\')
		base << '/';
	if(!base.IsEmpty() && full.StartsWith(base))
		return full.Mid(base.GetCount());
	return full;
}

bool AnnotationEditorWindow::AddClipboardImageToCurrentDataset(const Image& img, const String& signature, String* out_path) {
	if(current_dataset_index < 0 || current_dataset_index >= datasets.GetCount())
		return false;
	if(img.IsEmpty())
		return false;

	Dataset& ds = datasets[current_dataset_index];

	String target_dir = ds.folder_path;
	if(target_dir.IsEmpty())
		target_dir = AppendFileName(ProjectDir(), "images");
	else if(!IsFullPath(target_dir))
		target_dir = AppendFileName(ProjectDir(), target_dir);
	target_dir = NormalizePath(target_dir);

	if(!RealizeDirectory(target_dir) || !DirectoryExists(target_dir)) {
		undo_tab.Log("Clipboard collector: failed to create target directory: " + target_dir);
		return false;
	}

	String file_stem = SanitizeFileStem("clipboard_" + signature.Left(16));
	String abs_path = NormalizePath(AppendFileName(target_dir, file_stem + ".png"));
	String rel_or_abs = StoreProjectImagePath(abs_path);

	for(const auto& ie : ds.images) {
		String existing_abs = ResolveProjectImagePath(ie.file_path);
		if(!existing_abs.IsEmpty() && NormalizePath(existing_abs) == abs_path)
			return false;
	}

	if(!FileExists(abs_path) && !PNGEncoder().SaveFile(abs_path, img)) {
		undo_tab.Log("Clipboard collector: failed to write image: " + abs_path);
		return false;
	}

	ImageEntry& e = ds.images.Add();
	e.file_path = rel_or_abs;
	e.file_name = GetFileName(rel_or_abs);
	e.width = img.GetWidth();
	e.height = img.GetHeight();

	content_view.ClearCache();
	content_view.SetBaseDir(ProjectDir());
	content_view.SetDataset(ds, categories, datasets);
	datasets_tab.RefreshView();

	if(IsAnnotateViewActive()) {
		int img_idx = annotate_view.GetCurrentImageIndex();
		if(img_idx >= 0 && img_idx < ds.images.GetCount())
			annotate_view.SetDataset(ds, img_idx);
	}

	MarkDirty();
	undo_tab.Log("Clipboard collector added image: " + e.file_name);
	if(out_path) *out_path = e.file_path;
	return true;
}

String AnnotationEditorWindow::SanitizeFileStem(const String& in) const {
	String out;
	for(int i = 0; i < in.GetCount(); i++) {
		int c = (byte)in[i];
		if(IsAlNum(c) || c == '_' || c == '-')
			out.Cat(c);
		else if(c == ' ' || c == '.')
			out.Cat('_');
	}
	while(out.GetCount() > 1 && out[out.GetCount() - 1] == '_')
		out = out.Left(out.GetCount() - 1);
	if(out.IsEmpty()) out = "image";
	return out;
}

bool AnnotationEditorWindow::IsAnnotateViewActive() const {
	return annotate_view.GetParent() == this;
}

bool AnnotationEditorWindow::IsContentViewActive() const {
	return content_view.GetParent() == this;
}

int AnnotationEditorWindow::GetSavedProjectDatasetIndex() const {
	if(current_dataset_index < 0 || current_dataset_index >= datasets.GetCount()) return -1;
	if(!IsAnnotateViewActive() && !IsContentViewActive()) return -1;
	return current_dataset_index;
}

int AnnotationEditorWindow::GetSavedProjectImageIndex() const {
	if(!IsAnnotateViewActive()) return -1;
	return annotate_view.GetCurrentImageIndex();
}

String AnnotationEditorWindow::ResolveMluiScriptPath(const String& path) const {
	if(path.IsEmpty()) return path;
	if(IsFullPath(path)) return NormalizePath(path);
	return NormalizePath(AppendFileName(ProjectDir(), path));
}

String AnnotationEditorWindow::StoreMluiScriptPath(const String& path) const {
	if(path.IsEmpty()) return path;
	String full = ResolveMluiScriptPath(path);
	String base = NormalizePath(ProjectDir());
	if(!base.IsEmpty() && base[base.GetCount() - 1] != '/' && base[base.GetCount() - 1] != '\\')
		base << '/';
	if(!base.IsEmpty() && full.StartsWith(base))
		return full.Mid(base.GetCount());
	return full;
}

void AnnotationEditorWindow::RememberLastActiveMluiScript(const String& path) {
	if(path.IsEmpty()) return;
	last_active_mlui_script_path = ResolveMluiScriptPath(path);
}

String AnnotationEditorWindow::GetLastActiveMluiScriptPath() const {
	if(!last_active_mlui_script_path.IsEmpty())
		return ResolveMluiScriptPath(last_active_mlui_script_path);
	if(current_dataset_index >= 0 && current_dataset_index < datasets.GetCount()) {
		const String& sp = datasets[current_dataset_index].mlui_script_path;
		if(!sp.IsEmpty()) return ResolveMluiScriptPath(sp);
	}
	return String();
}

bool AnnotationEditorWindow::CopyObjectHintToLastActiveMluiScript(const AnnotationObject& obj) {
	int iw = annotate_view.GetImageWidth();
	int ih = annotate_view.GetImageHeight();
	if(iw <= 0 || ih <= 0) { PromptOK("No active image."); return false; }
	if(obj.bbox.Width() <= 0 || obj.bbox.Height() <= 0) { PromptOK("Selected object has no bbox."); return false; }

	String slot_id = obj.metadata.Get(MluiSlotIdKey(), "");
	if(slot_id.IsEmpty()) slot_id = obj.slot_id;
	if(slot_id.IsEmpty()) {
		PromptOK("Selected object has no MLUI slot id.");
		return false;
	}

	String resolved = GetLastActiveMluiScriptPath();
	if(resolved.IsEmpty()) {
		PromptOK("No active MLUI script. Open MLUI Script Editor and load a script first.");
		return false;
	}
	MluiScript s;
	if(!LoadMluiScript(s, resolved)) {
		PromptOK("Failed to load MLUI script:\n" + resolved);
		return false;
	}

	int si = s.FindSlot(slot_id);
	if(si < 0 || si >= s.slots.GetCount()) {
		PromptOK("Slot not found in active MLUI script: " + slot_id);
		return false;
	}

	s.slots[si].bbox_hint = Rectf(
		obj.bbox.left   / (double)iw,
		obj.bbox.top    / (double)ih,
		obj.bbox.right  / (double)iw,
		obj.bbox.bottom / (double)ih);
	String img_path = annotate_view.GetImagePath();
	if(!img_path.IsEmpty()) {
		s.reference_image.file_path = img_path;
		s.reference_image.width = iw;
		s.reference_image.height = ih;
	}

	if(!SaveMluiScript(s, resolved)) {
		PromptOK("Failed to save MLUI script:\n" + resolved);
		return false;
	}

	RememberLastActiveMluiScript(resolved);
	if(current_dataset_index >= 0 && current_dataset_index < datasets.GetCount()) {
		String active_ds_script = ResolveMluiScriptPath(datasets[current_dataset_index].mlui_script_path);
		if(!active_ds_script.IsEmpty() && active_ds_script == resolved) {
			ReloadCurrentScript();
			mlui_wizard.RefreshFillState();
		}
	}
	undo_tab.Log(Format("Copied hint for slot '%s' to %s", slot_id, GetFileName(resolved)));
	return true;
}

bool AnnotationEditorWindow::Access(Visitor& v) {
	bool base = DockWindow::Access(v);
	if(!dynamic_cast<AutomationVisitor*>(&v)) return base;

	{
		auto& page = MLUI::GetFocusPage("annotation_canvas");
		page.ClearRuntime();
		String tool = annotate_view.GetToolName();
		int sel_id   = annotate_view.GetSelectedId();
		String img_path = annotate_view.GetImagePath();
		double zoom  = annotate_view.GetZoom();
		int ann_count = annotate_view.GetAnnotationCount();
		int sug_count = annotate_view.GetSuggestionCount();
		MLUI_USE_STATE(page, "tool",             tool,      "Active annotation tool");
		MLUI_USE_STATE(page, "selected_id",      sel_id,    "Currently selected object id (-1 = none)");
		MLUI_USE_STATE(page, "image_path",       img_path,  "Currently open image");
		MLUI_USE_STATE(page, "zoom",             zoom,      "Current zoom level");
		MLUI_USE_STATE(page, "annotation_count", ann_count, "Number of annotations on current image");
		MLUI_USE_STATE(page, "suggestion_count", sug_count, "Number of AI suggestions on current image");
		MLUI_USE_ACTION(page, "select_tool",        true,         "Switch annotation tool");
		MLUI_USE_ACTION(page, "select_object",      true,         "Select object by id");
		MLUI_USE_ACTION(page, "delete_selected",    sel_id != -1, "Delete the selected object");
		MLUI_USE_ACTION(page, "undo",               true,         "Undo last action");
		MLUI_USE_ACTION(page, "redo",               true,         "Redo last undone action");
		MLUI_USE_ACTION(page, "center_view",        true,         "Reset view to fit image");
		MLUI_USE_ACTION(page, "apply_script_slot",  !img_path.IsEmpty(), "Create/update a script slot object");
	}

	{
		auto& page = MLUI::GetFocusPage("image_navigation");
		page.ClearRuntime();
		int cur   = annotate_view.GetCurrentImageIndex();
		int total = annotate_view.GetDatasetImageCount();
		bool can_prev = annotate_view.CanPrev();
		bool can_next = annotate_view.CanNext();
		String img_path = annotate_view.GetImagePath();
		MLUI_USE_STATE(page, "current_index", cur,      "Current image index (0-based)");
		MLUI_USE_STATE(page, "total_images",  total,    "Total images in dataset");
		MLUI_USE_STATE(page, "image_path",    img_path, "Current image path");
		MLUI_USE_ACTION(page, "prev_image",   can_prev, "Go to previous image");
		MLUI_USE_ACTION(page, "next_image",   can_next, "Go to next image");
		MLUI_USE_ACTION(page, "jump_to_best", total > 0, "Jump to highest-priority image");
		MLUI_USE_ACTION(page, "open_image",   total > 0, "Open image by index");
		bool has_unannotated = false;
		if(current_dataset_index >= 0 && current_dataset_index < datasets.GetCount())
			for(const auto& img : datasets[current_dataset_index].images)
				if(!img.has_annotations) { has_unannotated = true; break; }
		MLUI_USE_ACTION(page, "next_unannotated", has_unannotated, "Jump to next image with no annotations");
	}

	{
		auto& page = MLUI::GetFocusPage("object_list");
		page.ClearRuntime();
		int sel_id   = annotate_view.GetSelectedId();
		int ann_count = annotate_view.GetAnnotationCount();
		bool has_image = !annotate_view.GetImagePath().IsEmpty();
		MLUI_USE_STATE(page, "selected_id",      sel_id,    "Selected object id");
		MLUI_USE_STATE(page, "annotation_count", ann_count, "Total annotation objects");
		MLUI_USE_ACTION(page, "select_row",     ann_count > 0,  "Select object by id");
		MLUI_USE_ACTION(page, "delete_object",  sel_id != -1,   "Delete object by id");
		MLUI_USE_ACTION(page, "add_object",     has_image,      "Add new object with geometry");
		MLUI_USE_ACTION(page, "add_bbox",       has_image,      "Add annotation from pixel coordinates");
		MLUI_USE_ACTION(page, "get_annotations", has_image,     "Return list of all annotations on current image");
	}

	{
		auto& page = MLUI::GetFocusPage("mlui_script");
		page.ClearRuntime();
		String sp = (current_dataset_index >= 0 && current_dataset_index < datasets.GetCount())
		            ? datasets[current_dataset_index].mlui_script_path : String();
		String resolved = ResolveMluiScriptPath(sp);
		bool has_script = !sp.IsEmpty() && FileExists(resolved);
		int slot_count = 0;
		if(has_script) {
			MluiScript s;
			if(LoadMluiScript(s, resolved)) slot_count = s.slots.GetCount();
		}
		MLUI_USE_STATE(page, "script_path",  sp,         "Active .mlui script path");
		MLUI_USE_STATE(page, "slot_count",   slot_count, "Number of slots in script");
		MLUI_USE_ACTION(page, "apply_script",          has_script, "Create stubs for all unfilled slots");
		MLUI_USE_ACTION(page, "list_slots",            has_script, "List all slots with fill status");
		MLUI_USE_ACTION(page, "copy_hints_from_image", !annotate_view.GetImagePath().IsEmpty(), "Update bbox hints from current annotations");
		MLUI_USE_ACTION(page, "bulk_apply_script", has_script, "Apply script to all images in dataset");
	}

	{
		auto& page = MLUI::GetFocusPage("project");
		page.ClearRuntime();
		bool has_project = !project_path.IsEmpty();
		int ds_count = datasets.GetCount();
		MLUI_USE_STATE(page, "project_path",     project_path,         "Path of loaded project file");
		MLUI_USE_STATE(page, "dataset_count",    ds_count,             "Number of datasets in project");
		MLUI_USE_STATE(page, "current_dataset",  current_dataset_index,"Active dataset index (-1 = none)");
		MLUI_USE_ACTION(page, "load_project",    true,                 "Load a project file by path");
		MLUI_USE_ACTION(page, "save_project",    has_project,          "Save current project to disk");
		MLUI_USE_ACTION(page, "open_dataset",    ds_count > 0,         "Switch active dataset by index");
		MLUI_USE_ACTION(page, "set_script_path", current_dataset_index >= 0, "Set mlui_script_path on active dataset");
	}

	return true;
}

INITBLOCK {
	using namespace MLUI;

	RegisterFocusPage("annotation_canvas", "Annotation Canvas",
		"Main image editing surface - active tool, selected object, zoom. "
		"Tools: select bbox polygon brush eraser wand keypoint review")
		.Context("tool_values", "select|bbox|polygon|brush|eraser|wand|keypoint|review")
		.Action("select_tool",       "Select tool",       "Switch annotation mode; arg: tool=<name>")
		.Action("select_object",     "Select object",     "Highlight object by id; arg: id=<int>")
		.Action("delete_selected",   "Delete selected",   "Remove currently selected object")
		.Action("undo",              "Undo",              "Undo last annotation action")
		.Action("redo",              "Redo",              "Redo last undone action")
		.Action("center_view",       "Center view",       "Reset zoom and pan to fit image")
		.Action("apply_script_slot", "Apply script slot", "Create/update slot object; args: slot_id x y w h");

	RegisterFocusPage("image_navigation", "Image Navigation",
		"Navigate between images in the current dataset")
		.Action("prev_image",   "Previous image", "Go to previous image")
		.Action("next_image",   "Next image",     "Go to next image (or smart-next by priority)")
		.Action("jump_to_best", "Jump to best",   "Open highest-priority unannotated image")
		.Action("open_image",   "Open by index",  "Open image at 0-based index; arg: index=<int>")
		.Action("next_unannotated", "Next unannotated", "Open next image with has_annotations=false");

	RegisterFocusPage("object_list", "Object List",
		"All annotation objects, suggestions and keypoints on the current image")
		.Action("select_row",    "Select row",    "Highlight object on canvas; arg: id=<int>")
		.Action("delete_object", "Delete object", "Remove object by id; arg: id=<int>")
		.Action("add_object",    "Add object",    "Open Set Geometry dialog to create a new annotation")
		.Action("add_bbox", "Add bbox", "Create annotation from pixel coords; args: x y w h name category_id(optional)")
		.Action("get_annotations", "Get annotations", "Return all annotation objects on current image with ids, names, bboxes, metadata");

	RegisterFocusPage("mlui_script", "MLUI Script",
		"Active .mlui layout script - slot definitions, fill status, bbox hint management")
		.Action("apply_script",          "Apply script", "Create stubs for all unfilled slots")
		.Action("list_slots",            "List slots",   "Return slot definitions with fill status")
		.Action("copy_hints_from_image", "Copy hints",   "Derive bbox hints from current image annotations")
		.Action("bulk_apply_script", "Bulk apply script", "Apply script to all images in dataset; creates stubs for unfilled slots on every image");

	RegisterFocusPage("project", "Project",
		"Project file management - load, save, dataset selection")
		.Action("load_project",   "Load project",   "Load .annprj file; arg: path=<string>")
		.Action("save_project",   "Save project",   "Save current project to disk")
		.Action("open_dataset",   "Open dataset",   "Switch active dataset by 0-based index; arg: index=<int>")
		.Action("set_script_path","Set script path","Set mlui_script_path on current dataset; arg: path=<string>");
}

END_UPP_NAMESPACE
