#include "OCRWorkbench.h"
#include <plugin/jpg/jpg.h>

using namespace Upp;

void GenerateTestProject() {
	String dir = "examples/OCRWorkbench/test_project";
	RealizeDirectory(dir);
	
	OCR::OCRDatasetProject project;
	
	Vector<String> texts = { "HELLO", "WORLD", "OCR123" };
	
	for(int i = 0; i < texts.GetCount(); i++) {
		String text = texts[i];
		Font font = Roman(40).Bold();
		Size sz = GetTextSize(text, font);
		sz.cx += 20;
		sz.cy += 20;
		
		Image img;
		{
			ImageDraw id(sz);
			id.DrawRect(sz, White());
			id.DrawText(10, 10, text, font, Black());
			img = id;
		}
		
		String img_path = AppendFileName(dir, Format("img%d.jpg", i + 1));
		JPGEncoder().SaveFile(img_path, img);
		
		// Add to project
		project.AddImage(img, img_path);
		
		// Create segment for the whole line
		OCR::OCRSegment& seg = project.GetSegments(i).Add();
		seg.bounds = Rect(10, 10, 10 + sz.cx - 20, 10 + sz.cy - 20);
		seg.text = text;
		
		// Create character segments
		int x = 10;
		for(int j = 0; j < text.GetCount(); j++) {
			String ch = text.Mid(j, 1);
			Size ch_sz = GetTextSize(ch, font);
			
			OCR::OCRChar& c = seg.chars.Add();
			c.bounds = Rect(x, 10, x + ch_sz.cx, 10 + ch_sz.cy);
			c.label = OCR::CharToClass(text[j]);
			
			x += ch_sz.cx;
		}
	}
	
	String project_path = AppendFileName(dir, "project.ocrdata");
	if(project.Save(project_path)) {
		Cout() << "Test project generated successfully at: " << project_path << "\n";
	} else {
		Cerr() << "Failed to save test project!\n";
	}
}

OCRWorkbench::OCRWorkbench() {
	Title("OCR Workbench - Unified OCR Suite");
	Sizeable().MaximizeBox().MinimizeBox();
	
	SetRect(0, 0, 1024, 768);
	
	AddFrame(menu);
	AddFrame(toolbar);
	AddFrame(status);
	
	dataset_editor.SetEmbedded();
	trainer.SetEmbedded();
	
	tabs.Add(dataset_editor.SizePos(), "1. Dataset Editor");
	tabs.Add(trainer.SizePos(), "2. Model Training");
	
	Add(tabs.SizePos());
	
	menu.Set(THISBACK(MainMenu));
	toolbar.Set([this](Bar& bar) {
		bar.Add("Annotate", CtrlImg::open(), [this] { tabs.Set(0); });
		bar.Add("Train", CtrlImg::plus(), [this] { tabs.Set(1); });
		bar.Separator();
		dataset_editor.MainToolbar(bar);
		bar.Separator();
		bar.Add("About", CtrlImg::help(), [this] { PromptOK("OCR Workbench 1.0\nBuilt with U++ and ConvNet"); });
	});
	
	dataset_editor.WhenProjectLoaded = THISBACK(OnProjectLoaded);
	
	tabs << [this] { menu.Set(THISBACK(MainMenu)); };
	tabs.Set(0);
	
	status.Set("Ready");
	Refresh();
}

OCRWorkbench::~OCRWorkbench() {
}

void OCRWorkbench::DockInit() {
}

void OCRWorkbench::MainMenu(Bar& bar) {
	bar.Add("File", THISBACK(FileMenu));
	
	if (tabs.Get() == 0) {
		bar.Add("Edit", THISBACK(DatasetEditMenu));
		bar.Add("View", THISBACK(DatasetViewMenu));
		bar.Add("Image", THISBACK(DatasetImageMenu));
	} else if (tabs.Get() == 1) {
		bar.Add("Training", THISBACK(TrainingMenu));
	}
	
	bar.Add("Workflow", THISBACK(WorkflowMenu));
	bar.Add("Tools", THISBACK(ToolsMenu));
	bar.Add("Help", THISBACK(HelpMenu));
}

void OCRWorkbench::ToolsMenu(Bar& bar) {
	bar.Add("Model Manager...", [] { PromptOK("Model Manager coming soon"); });
	bar.Add("Dataset Statistics...", [] { PromptOK("Dataset Statistics coming soon"); });
}

void OCRWorkbench::FileMenu(Bar& bar) {
	dataset_editor.FileMenu(bar); // Uses dataset editor's New/Open/Save logic
	bar.Separator();
	bar.Add("Exit", [this] { TopWindow::Close(); });
}

void OCRWorkbench::WorkflowMenu(Bar& bar) {
	bar.Add("1. Annotate Dataset", [this] { tabs.Set(0); }).Key(K_F5);
	bar.Add("2. Train Models", [this] { tabs.Set(1); }).Key(K_F6);
	bar.Separator();
	bar.Add("Quick Test OCR", [this] { tabs.Set(1); trainer.RunBatchTest(); }).Key(K_F7);
}

void OCRWorkbench::HelpMenu(Bar& bar) {
	bar.Add("About...", [] { PromptOK("OCR Workbench 1.0\nBuilt with U++ and ConvNet"); });
}

void OCRWorkbench::OnProjectLoaded(const String& path) {
	RLOG("OCRWorkbench::OnProjectLoaded: " << path);
	status.Set("Loaded project: " + path);
	
	// Sync project to trainer
	trainer.LoadProject(path);
	
	// Force refresh
	dataset_editor.Refresh();
	trainer.Refresh();
	Layout();
}

void OCRWorkbench::StartTraining() {
	tabs.Set(1);
	trainer.StartTraining();
}

// Main entry point
#ifdef flagMAIN
GUI_APP_MAIN {
	StdLogSetup(LOG_FILE|LOG_COUT);
	RLOG("OCRWorkbench starting...");

	CommandLineArguments cmd;
	cmd.AddArg("train", 't', "Start training automatically", false);
	cmd.AddArg("generate-test-project", 'g', "Generate test project", false);
	cmd.AddPositional("project", "Path to project file");
	
	if (!cmd.Parse()) {
		RLOG("Failed to parse command line");
		return;
	}
	
	if (cmd.IsArg("generate-test-project")) {
		GenerateTestProject();
		return;
	}

	OCRWorkbench app;
	app.Layout();
	
	if (cmd.GetPositionalCount() > 0) {
		String path = cmd.GetPositional(0);
		if (FileExists(path)) {
			RLOG("Loading project: " << path);
			app.LoadProject(path);
			app.Layout();
		}
	}

	if (cmd.IsArg("train")) {
		RLOG("Starting training from CLI...");
		app.StartTraining();
	}

	app.Run();
}
#endif
