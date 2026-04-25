#ifndef _AnnotationEditor_ObjectSettingsDialog_h_
#define _AnnotationEditor_ObjectSettingsDialog_h_

#include <CtrlLib/CtrlLib.h>
#include "Dataset.h"

using namespace Upp;

class ObjectSettingsDialog : public TopWindow {
public:
	typedef ObjectSettingsDialog CLASSNAME;
	ObjectSettingsDialog(AnnotationObject& obj, const Vector<Category>& categories);

	void OnOK();
	void OnCancel();
	
	void AddMetadata();
	void RemoveMetadata();

private:
	AnnotationObject& object;
	const Vector<Category>& categories;

	Label lbl_name, lbl_category, lbl_color, lbl_state, lbl_metadata;
	EditString edit_name;
	DropList dl_category, dl_state;
	ColorPusher cp_color;
	ArrayCtrl list_metadata;
	Button btn_add_meta, btn_remove_meta;
	Button btn_ok, btn_cancel;
};

#endif
