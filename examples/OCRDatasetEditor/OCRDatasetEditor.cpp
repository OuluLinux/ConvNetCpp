#include "OCRDatasetEditor.h"

namespace Upp {

OCRDatasetEditor::OCRDatasetEditor() {
	Title("OCR Dataset Editor");
	Sizeable().Zoomable();
	
	SetRect(0, 0, 800, 600);
	
	selected_image_index = -1;
	selected_segment_index = -1;
	selected_char_index = -1;
	
	AddFrame(menu);
	AddFrame(toolbar);
	AddFrame(status);
	
	image_list.Create();
	image_list->SetProject(&project);
	image_list->WhenImageSelected = THISBACK(OnImageSelected);
	image_list->WhenListChanged << [this] { UpdateTitle(); };
	
	preview.Create();
	preview->WhenSegmentSelected = THISBACK(OnSegmentSelected);
	preview->WhenCharSelected = THISBACK(OnCharSelected);
	
	split_tool.Create();
	split_tool->SetProject(&project);
	split_tool->WhenChanged << [this] { preview->Refresh(); UpdateTitle(); };
	
	label_tool.Create();
	label_tool->SetProject(&project);
	label_tool->WhenChanged << [this] { preview->Refresh(); UpdateTitle(); };
	
	menu.Set(THISBACK(MainMenu));
	toolbar.Set(THISBACK(MainToolbar));
	
	PostCallback(THISBACK(DockInit));
	
	UpdateTitle();
}

OCRDatasetEditor::~OCRDatasetEditor() {
}

void OCRDatasetEditor::DockInit() {
	if (docking_initialized) return;
	
	Add(preview->SizePos());
	
	DockLeft(Dockable(*image_list, "Images").SizeHint(Size(200, 400)));
	DockRight(Dockable(*split_tool, "Segmentation").SizeHint(Size(200, 200)));
	DockBottom(Dockable(*label_tool, "Labeling").SizeHint(Size(600, 200)));
	
	docking_initialized = true;
	Layout();
	Refresh();
}

void OCRDatasetEditor::MainMenu(Bar& bar) {
	bar.Add("File", THISBACK(FileMenu));
	bar.Add("Edit", THISBACK(EditMenu));
	bar.Add("View", THISBACK(ViewMenu));
	bar.Add("Image", THISBACK(ImageMenu));
}

void OCRDatasetEditor::FileMenu(Bar& bar) {
	bar.Add("New Project", CtrlImg::new_doc(), THISBACK(FileNew))
	   .Key(K_CTRL_N);
	bar.Add("Open Project...", CtrlImg::open(), THISBACK(FileOpen))
	   .Key(K_CTRL_O);
	bar.Add("Save", CtrlImg::save(), THISBACK(FileSave))
	   .Key(K_CTRL_S)
	   .Enable(project.IsModified() || current_path.IsEmpty());
	bar.Add("Save As...", CtrlImg::save_as(), THISBACK(FileSaveAs))
	   .Key(K_SHIFT_CTRL_S);
}

void OCRDatasetEditor::EditMenu(Bar& bar) {
	bar.Add("Undo", CtrlImg::undo(), [] { /* TODO */ }).Key(K_CTRL_Z).Enable(false);
	bar.Add("Redo", CtrlImg::redo(), [] { /* TODO */ }).Key(K_CTRL_Y).Enable(false);
	bar.Separator();
	bar.Add("Delete", CtrlImg::remove(), [this] { preview->DeleteSelection(); }).Key(K_DELETE);
}

void OCRDatasetEditor::ViewMenu(Bar& bar) {
	bar.Add("Zoom In", [this] { preview->ZoomIn(); }).Key(K_PLUS);
	bar.Add("Zoom Out", [this] { preview->ZoomOut(); }).Key(K_MINUS);
	bar.Add("Fit to Window", [this] { preview->ZoomFit(); }).Key(K_0);
	bar.Separator();
	bar.Add("Show Grid", [this] { preview->ShowGrid(!preview->IsShowGrid()); }).Check(preview->IsShowGrid()).Key(K_G);
	bar.Add("Show Bounds", [this] { preview->ShowBounds(!preview->IsShowBounds()); }).Check(preview->IsShowBounds()).Key(K_B);
	bar.Add("Show Labels", [this] { preview->ShowLabels(!preview->IsShowLabels()); }).Check(preview->IsShowLabels()).Key(K_L);
}

void OCRDatasetEditor::ImageMenu(Bar& bar) {
	bar.Add("Add Image...", CtrlImg::open(), [this] { image_list->AddImage(); }).Key(K_CTRL_I);
	bar.Add("Paste from Clipboard", CtrlImg::copy(), [this] { image_list->AddImageFromClipboard(); }).Key(K_SHIFT_CTRL_V);
	bar.Separator();
	bar.Add("Auto-Segment Line", [this] { split_tool->AutoSegment(); }).Key(K_CTRL_A);
	bar.Add("Auto-Segment All", [this] { split_tool->AutoSegmentAll(); }).Key(K_SHIFT_CTRL_A);
}

void OCRDatasetEditor::MainToolbar(Bar& bar) {
	bar.Add(CtrlImg::new_doc(), THISBACK(FileNew))
	   .Help("New Project");
	bar.Add(CtrlImg::open(), THISBACK(FileOpen))
	   .Help("Open Project");
	bar.Add(CtrlImg::save(), THISBACK(FileSave))
	   .Enable(project.IsModified())
	   .Help("Save Project");
}

void OCRDatasetEditor::LoadProject(const String& path) {
	DockInit();
	if (project.Load(path)) {
		current_path = path;
		selected_image_index = -1;
		selected_segment_index = -1;
		UpdateTitle();
		image_list->UpdateList();
		WhenProjectLoaded(path);
	} else {
		Exclamation("Failed to load project!");
	}
}

void OCRDatasetEditor::SaveProject(const String& path) {
	if (project.Save(path)) {
		current_path = path;
		UpdateTitle();
	} else {
		Exclamation("Failed to save project!");
	}
}

void OCRDatasetEditor::FileNew() {
	if (!FileClose()) return;
	DockInit();
	project.Clear();
	current_path.Clear();
	selected_image_index = -1;
	selected_segment_index = -1;
	UpdateTitle();
	preview->SetImage(Image());
	image_list->UpdateList();
}

void OCRDatasetEditor::FileOpen() {
	if (!FileClose()) return;
	
	FileSel fs;
	fs.Type("OCR Dataset (.ocrdata)", "*.ocrdata");
	if (fs.ExecuteOpen("Open Project")) {
		LoadProject(fs.Get());
	}
}

void OCRDatasetEditor::FileSave() {
	if (current_path.IsEmpty()) {
		FileSaveAs();
	} else {
		SaveProject(current_path);
	}
}

void OCRDatasetEditor::FileSaveAs() {
	FileSel fs;
	fs.Type("OCR Dataset (.ocrdata)", "*.ocrdata");
	if (fs.ExecuteSaveAs("Save Project As")) {
		SaveProject(fs.Get());
	}
}

bool OCRDatasetEditor::FileClose() {
	if (project.IsModified()) {
		int res = PromptYesNoCancel("Do you want to save changes to the current project?");
		if (res == 1) { // Yes
			FileSave();
			return !project.IsModified(); // Only allow close if save succeeded
		} else if (res == 0) { // No
			return true;
		} else { // Cancel
			return false;
		}
	}
	return true;
}

void OCRDatasetEditor::UpdateTitle() {
	String title = "OCR Dataset Editor";
	if (!current_path.IsEmpty()) {
		title << " - " << GetFileName(current_path);
	} else {
		title << " - [New Project]";
	}
	
	if (project.IsModified()) {
		title << " *";
	}
	
	Title(title);
}

void OCRDatasetEditor::OnImageSelected(int index) {
	if (index >= 0 && index < project.GetCount()) {
		selected_image_index = index;
		selected_segment_index = -1;
		selected_char_index = -1;
		
		preview->SetImage(project.GetImage(index));
		preview->SetSegments(&project.GetSegments(index));
		preview->ZoomFit();
		
		split_tool->SetImageIndex(index);
		split_tool->SetSegmentIndex(-1);
		
		label_tool->SetImageIndex(index);
		label_tool->SetSegmentIndex(-1);
		label_tool->SetCharIndex(-1);
		
		status.Set(Format("Image: %s (%d x %d)", 
		           GetFileName(project.GetImagePath(index)),
		           project.GetImage(index).GetWidth(),
		           project.GetImage(index).GetHeight()));
	}
}

void OCRDatasetEditor::OnSegmentSelected(int index) {
	selected_segment_index = index;
	selected_char_index = -1;
	
	preview->SelectSegment(index);
	preview->SelectChar(-1);
	
	split_tool->SetSegmentIndex(index);
	
	label_tool->SetSegmentIndex(index);
	label_tool->SetCharIndex(-1);
	
	status.Set(Format("Selected segment %d", index));
}

void OCRDatasetEditor::OnCharSelected(int segment_idx, int char_idx) {
	selected_segment_index = segment_idx;
	selected_char_index = char_idx;
	
	preview->SelectSegment(segment_idx);
	preview->SelectChar(char_idx);
	
	split_tool->SetSegmentIndex(segment_idx);
	
	label_tool->SetSegmentIndex(segment_idx);
	label_tool->SetCharIndex(char_idx);
	
	status.Set(Format("Selected segment %d, char %d", segment_idx, char_idx));
}

void OCRDatasetEditor::SetEmbedded(bool b) {
	if (b) {
		menu.Hide();
		toolbar.Hide();
		status.Hide();
	} else {
		menu.Show();
		toolbar.Show();
		status.Show();
	}
	Layout();
}

} // namespace Upp
