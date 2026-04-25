#ifndef _AnnotationEditor_CreateDatasetDialog_h_
#define _AnnotationEditor_CreateDatasetDialog_h_

#include <CtrlLib/CtrlLib.h>
#include "Dataset.h"

using namespace Upp;

class CreateDatasetDialog : public TopWindow {
public:
	typedef CreateDatasetDialog CLASSNAME;
	CreateDatasetDialog();

	Dataset GetDataset() const;
	void SetProjectDir(const String& dir);

private:
	String project_dir_;

	Label lbl_name, lbl_categories, lbl_path;
	EditString edit_name, edit_category, edit_path;
	ColumnList list_categories;
	Button btn_add_cat, btn_remove_cat, btn_browse;
	Button btn_ok, btn_cancel;

	void AddCategory();
	void RemoveCategory();
	void OnOK();
	void OnCancel();
	void OnBrowse();
};

#endif
