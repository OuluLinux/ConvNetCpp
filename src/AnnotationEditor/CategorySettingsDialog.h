#ifndef _AnnotationEditor_CategorySettingsDialog_h_
#define _AnnotationEditor_CategorySettingsDialog_h_

#include <CtrlLib/CtrlLib.h>
#include "Dataset.h"

using namespace Upp;

class CategorySettingsDialog : public TopWindow {
public:
	typedef CategorySettingsDialog CLASSNAME;
	CategorySettingsDialog(Category& cat, bool is_new = false);

	void OnOK();
	void OnCancel();

private:
	Category& category;
	bool is_new;

	Label lbl_name, lbl_super, lbl_color, lbl_keypoints;
	EditString edit_name, edit_super;
	ColorPusher cp_color;
	ArrayCtrl list_keypoints;
	Button btn_add_kp, btn_remove_kp;
	Button btn_ok, btn_cancel;
	
	EditString edit_kp_label;    // For Label column inline editing
	EditString edit_kp_connects; // For Connects To column inline editing
};

#endif
