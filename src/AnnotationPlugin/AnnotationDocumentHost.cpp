#include "AnnotationTool.h"

NAMESPACE_UPP

AnnotationDocumentHost::AnnotationDocumentHost()
{
	Add(editor.SizePos());
	editor.SetModel(&document);
	editor.WhenModified = [this] { modified = true; };
	editor.WhenExportCOCO = [this] { DoExportCOCO(); };
}

bool AnnotationDocumentHost::Load(const String& path_)
{
	path = path_;
	modified = false;
	document.Clear();
	if(FileExists(path) && !document.Load(path))
		return false;
	if(document.GetProjectPath().IsEmpty())
		document.SetProjectPath(path);
	editor.RefreshFromModel();
	return true;
}

bool AnnotationDocumentHost::Save()
{
	if(path.IsEmpty())
		return false;
	document.SetProjectPath(path);
	if(!document.Save(path))
		return false;
	modified = false;
	return true;
}

bool AnnotationDocumentHost::SaveAs(const String& path_)
{
	path = path_;
	return Save();
}

void AnnotationDocumentHost::SetFocus()
{
	editor.SetFocus();
}

void AnnotationDocumentHost::NewDocument()
{
	document.Clear();
	document.SetProjectPath(path.IsEmpty() ? AppendFileName(GetCurrentDirectory(), "annotation_project.annlib") : path);
	modified = true;
	editor.RefreshFromModel();
}

void AnnotationDocumentHost::DoExportCOCO()
{
	FileSel fs;
	fs.Type("JSON files", "*.json");
	if(!fs.ExecuteSaveAs())
		return;
	document.ExportCOCO(~fs);
}

void AnnotationDocumentHost::MainMenu(Bar& bar)
{
	bar.Sub("Annotation", [this](Bar& menu) {
		menu.Add("New", [this] { NewDocument(); });
		menu.Add("Save", [this] { Save(); }).Enable(!path.IsEmpty());
		menu.Add("Save As...", [this] {
			FileSel fs;
			fs.Type("Annotation library", "*.annlib");
			if(fs.ExecuteSaveAs())
				SaveAs(~fs);
		});
		menu.Separator();
		menu.Add(editor.IsDrawMode() ? "Switch To Select Mode" : "Switch To Draw Mode", [this] {
			editor.SetDrawMode(!editor.IsDrawMode());
		});
		menu.Add("Add Screenshot...", [this] { editor.PromptAddScreenshot(); });
		menu.Add("Capture From Video...", [this] { editor.PromptCaptureFromVideo(); });
		menu.Add("Export COCO JSON...", [this] { DoExportCOCO(); });
	});
}

void AnnotationDocumentHost::Toolbar(Bar& bar)
{
	bar.Add(Icons::OpenFile(), [this] { editor.PromptAddScreenshot(); }).Tip("Add screenshot");
	bar.Add(Icons::Save(), [this] { Save(); }).Tip("Save .annlib");
	bar.Add(Icons::Run(), [this] { editor.SetDrawMode(!editor.IsDrawMode()); })
	   .Tip(editor.IsDrawMode() ? "Draw mode enabled" : "Select mode enabled");
	bar.Add(Icons::Save(), [this] { DoExportCOCO(); }).Tip("Export COCO JSON");
}

void AnnotationDocumentHost::SelectAll()
{
}

END_UPP_NAMESPACE
