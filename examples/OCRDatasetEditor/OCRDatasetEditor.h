#ifndef _OCRDatasetEditor_OCRDatasetEditor_h_
#define _OCRDatasetEditor_OCRDatasetEditor_h_

#include <CtrlLib/CtrlLib.h>
#include <Docking/Docking.h>
#include <OCR/OCRDatasetProject.h>
#include <OCRCtrl/OCRCtrl.h>

namespace Upp {

class OCRDatasetEditor : public DockWindow {
public:
	typedef OCRDatasetEditor CLASSNAME;
	OCRDatasetEditor();
	~OCRDatasetEditor();

	virtual void DockInit();

	void MainMenu(Bar& bar);
	void FileMenu(Bar& bar);
	void EditMenu(Bar& bar);
	void ViewMenu(Bar& bar);
	void ImageMenu(Bar& bar);
	void MainToolbar(Bar& bar);

	// Embeddable interface
	void LoadProject(const String& path);
	void SaveProject(const String& path);
	bool IsModified() const { return project.IsModified(); }

	// File operations
	void FileNew();
	void FileOpen();
	void FileSave();
	void FileSaveAs();
	bool FileClose();
	
	void UpdateTitle();
	void UpdateMenuBar();
	
	void OnImageSelected(int index);
	void OnSegmentSelected(int index);
	void OnCharSelected(int segment_idx, int char_idx);

	void SetEmbedded(bool b = true);

	Callback1<const String&> WhenProjectLoaded;

private:
	bool docking_initialized = false;
	OCR::OCRDatasetProject project;
	String current_path;
	int selected_image_index;
	int selected_segment_index;
	int selected_char_index;
	
	MenuBar menu;
	ToolBar toolbar;
	StatusBar status;
	
	One<OCR::ImageListCtrl> image_list;
	One<OCR::ImagePreviewCtrl> preview;
	One<OCR::SplitToolPanel> split_tool;
	One<OCR::CharLabelCtrl> label_tool;
};

} // namespace Upp

#endif
