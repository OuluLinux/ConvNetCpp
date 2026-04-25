#include "CreateDatasetDialog.h"

CreateDatasetDialog::CreateDatasetDialog() {
	Title("Create Dataset");
	SetRect(0, 0, 400, 450);

	Add(lbl_name.SetLabel("Dataset Name:").LeftPos(10, 100).TopPos(10, 20));
	Add(edit_name.HSizePos(10, 10).TopPos(35, 20));

	Add(lbl_categories.SetLabel("Default Categories:").LeftPos(10, 150).TopPos(65, 20));
	Add(edit_category.LeftPos(10, 290).TopPos(90, 20));
	Add(btn_add_cat.SetLabel("Add").LeftPos(310, 80).TopPos(90, 20));
	Add(list_categories.HSizePos(10, 10).VSizePos(120, 100));
	Add(btn_remove_cat.SetLabel("Remove selected").LeftPos(10, 150).BottomPos(75, 20));

	Add(lbl_path.SetLabel("Folder Path:").LeftPos(10, 100).BottomPos(45, 20));
	Add(edit_path.HSizePos(10, 100).BottomPos(20, 20));
	Add(btn_browse.SetLabel("Browse...").RightPos(10, 80).BottomPos(20, 20));

	Add(btn_ok.SetLabel("OK").RightPos(100, 80).BottomPos(75, 20));
	Add(btn_cancel.SetLabel("Cancel").RightPos(10, 80).BottomPos(75, 20));

	btn_add_cat << THISBACK(AddCategory);
	btn_remove_cat << THISBACK(RemoveCategory);
	btn_browse << THISBACK(OnBrowse);
	btn_ok << THISBACK(OnOK);
	btn_cancel << THISBACK(OnCancel);

	edit_category.WhenEnter = THISBACK(AddCategory);
}

void CreateDatasetDialog::AddCategory() {
	String name = ~edit_category;
	if (name.IsEmpty()) return;
	list_categories.Add(name);
	edit_category.Clear();
}

void CreateDatasetDialog::RemoveCategory() {
	if (list_categories.IsCursor())
		list_categories.Remove(list_categories.GetCursor());
}

void CreateDatasetDialog::OnOK() {
	if (IsNull(edit_name.GetData())) {
		Exclamation("Dataset name cannot be empty");
		return;
	}
	Break(IDOK);
}

void CreateDatasetDialog::OnCancel() {
	Break(IDCANCEL);
}

void CreateDatasetDialog::SetProjectDir(const String& dir) {
	project_dir_ = dir;
	edit_path.SetData(dir);
}

void CreateDatasetDialog::OnBrowse() {
	FileSel f;
	f.ActiveDir(edit_path.GetData().ToString());
	if (f.ExecuteSelectDir("Select dataset folder")) {
		edit_path.SetData(f.Get());
	}
}

Dataset CreateDatasetDialog::GetDataset() const {
	Dataset d;
	d.name = ~edit_name;
	String full = ~edit_path;
	String rel  = full;
	if(!project_dir_.IsEmpty() && full.StartsWith(project_dir_))
		rel = "." + full.Mid(project_dir_.GetLength());
	d.folder_path = rel;
	for (int i = 0; i < list_categories.GetCount(); i++)
		d.default_categories.Add(list_categories.Get(i).ToString());
	return d;
}
