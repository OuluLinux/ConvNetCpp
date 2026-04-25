#ifndef _OCRWorkbench_OCRWorkbench_h_
#define _OCRWorkbench_OCRWorkbench_h_

#include <CtrlLib/CtrlLib.h>
#include <Docking/Docking.h>
#include <OCR/OCR.h>
#include "../OCRDatasetEditor/OCRDatasetEditor.h"
#include "../OCRTraining/OCRTraining.h"

using namespace Upp;

class OCRWorkbench : public DockWindow {
public:
	typedef OCRWorkbench CLASSNAME;
	OCRWorkbench();
	~OCRWorkbench();

	virtual void DockInit();

	void MainMenu(Bar& bar);
	void FileMenu(Bar& bar);
	void WorkflowMenu(Bar& bar);
	void ToolsMenu(Bar& bar);
	void HelpMenu(Bar& bar);
	
	void DatasetEditMenu(Bar& bar) { dataset_editor.EditMenu(bar); }
	void DatasetViewMenu(Bar& bar) { dataset_editor.ViewMenu(bar); }
	void DatasetImageMenu(Bar& bar) { dataset_editor.ImageMenu(bar); }
	void TrainingMenu(Bar& bar) { trainer.TrainingMenu(bar); }

	void LoadProject(const String& path) { dataset_editor.LoadProject(path); }
	void StartTraining();

private:
	TabCtrl tabs;
	OCRDatasetEditor dataset_editor;
	OCRTraining trainer;

	MenuBar menu;
	ToolBar toolbar;
	StatusBar status;

	void OnProjectLoaded(const String& path);
};

#endif
