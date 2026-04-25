#ifndef _AnnotationEditor_AnnotationEditorWindow_h_
#define _AnnotationEditor_AnnotationEditorWindow_h_

#include "AnnotationEditorCommon.h"
#include "AnnotateView.h"
#include "ClipboardImageCollectorDialog.h"
#include "DatasetsTab.h"
#include "CategoriesTab.h"
#include "UndoTab.h"
#include "QualityTab.h"
#include "TasksTab.h"
#include "MluiWizardTab.h"

NAMESPACE_UPP
class AnnotationEditorWindow : public DockWindow {
public:
	typedef AnnotationEditorWindow CLASSNAME;
	AnnotationEditorWindow();
	virtual void DockInit() override;
	void ApplyDefaultDockLayout();
	void CacheDefaultLayout();
	void SetDefaultLayout();
	bool CloseOpenMluiScriptEditors();
	virtual void Close() override;
	void MarkDirty();
	void AutoSave();
	String GetAutoSavePath();
	String GetAutoSavePathForProject(const String& prj_path);
	String GetTempInternalDirectory();
	void CheckRecovery();

	void OnSaveSnapshot();
	void CreateSampleDatasets();
	void CreateSampleCategories();
	void OpenDataset(int index);
	void CloseDataset();
	void OnScanDataset();
	void ScanDataset(Dataset& ds);
	void OpenAnnotateView(ImageEntry& ie);
	void OpenAnnotateViewAt(int img_idx);
	void CloseAnnotateView();
	void AutoOpenFirstImage();
	void CreateSampleObjects();
	void OnNewProject();
	void OnOpenProject();
	void OnSaveProject();
	void OnSaveProjectAs();
	void SaveProject(const String& path);
	void LoadProject(const String& path);
	void RefreshAllUI();
	void OnImportCoco();
	void OnExportCoco();
	virtual bool Key(dword key, int count) override;
private:
	MenuBar menu; DatasetsTab datasets_tab; DatasetContentView content_view; AnnotateView annotate_view; CategoriesTab categories_tab; QualityTab quality_tab; UndoTab undo_tab; TasksTab tasks_tab; MluiWizardTab mlui_wizard; AnnLayTrainPanel annlay_train_panel;
	DockableCtrl dock_ann_objects, dock_ann_categories, dock_ann_settings, dock_ann_metadata, dock_mlui_wizard, dock_annlay_train;
	MluiScript current_script_;
	String default_layout_data;
	Vector<MluiScriptEditor*> open_mlui_editors_;
private:
	Vector<Dataset> datasets; Vector<Category> categories; int current_dataset_index = -1; String project_path; String last_active_mlui_script_path;
	CommandManager cmdmgr; bool project_dirty = false;
	Vector<String> recent_projects_;
	Index<String> mlui_category_create_declined_;

	String RecentProjectsPath();
	String GetPlacementPath() const;
	String GetProjectSessionPath(const String& path) const;
	void SaveProjectSessionState();
	bool LoadProjectSessionState(const String& path, int& ds_idx, int& img_idx, String* last_mlui_path = nullptr);
	void LoadRecentProjects();
	void SaveRecentProjects();
	void AddRecentProject(const String& path);
	int FindCategoryIdByName(const String& name) const;
	int GetNextCategoryId() const;
	int EnsureMluiSlotCategory(const String& category_name);
	String ProjectDir() const;
	String ResolveProjectImagePath(const String& path) const;
	String StoreProjectImagePath(const String& path) const;
	bool AddClipboardImageToCurrentDataset(const Image& img, const String& signature, String* out_path = nullptr);
	String SanitizeFileStem(const String& in) const;
	bool IsAnnotateViewActive() const;
	bool IsContentViewActive() const;
	int GetSavedProjectDatasetIndex() const;
	int GetSavedProjectImageIndex() const;
	String ResolveMluiScriptPath(const String& path) const;
	String StoreMluiScriptPath(const String& path) const;
	void RememberLastActiveMluiScript(const String& path);
	String GetLastActiveMluiScriptPath() const;
	bool CopyObjectHintToLastActiveMluiScript(const AnnotationObject& obj);
	void ReloadCurrentScript();
	void MainMenu(Bar& bar);
	void MenuFile(Bar& bar);
	void MenuProject(Bar& bar);
	void MenuEdit(Bar& bar);
	void MenuTools(Bar& bar);
	void OnClipboardImageCollector();
	void OnGlobalUndo();
	void OnGlobalRedo();
	void OnSetGeometry();
	void OnSelectAllCurrentView();
	void OnEditCardMetadata();
	void OnMakeProjectStandalone();
	void OnMluiWizardSkip();
	void OnApplyLastMluiScriptToImage();
	void OnOpenMluiScriptEditor();
	void OnCopyMluiHintsFromImage();
	void ShowDatasetsView();
	void ShowAnnotateView();
	void ShowCategoriesView();
	void MenuView(Bar& bar);
	void MenuHelp(Bar& bar);
	void OnCreateDataset();
	void RefreshDatasetsView();

	// ---------------------------------------------------------------
	// MLUI Focus — MCP exposure
	// ---------------------------------------------------------------
public:
	void RegisterFocusActions();

	virtual bool Access(Visitor& v) override;
};

END_UPP_NAMESPACE

#endif
