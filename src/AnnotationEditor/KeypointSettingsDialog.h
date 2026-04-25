#ifndef _AnnotationEditor_KeypointSettingsDialog_h_
#define _AnnotationEditor_KeypointSettingsDialog_h_

#include <CtrlLib/CtrlLib.h>
#include "Dataset.h"

using namespace Upp;

class KeypointSettingsDialog : public TopWindow {
public:
	typedef KeypointSettingsDialog CLASSNAME;
	KeypointSettingsDialog(KeypointInstance& kp);

	void OnOK();
	void OnCancel();

private:
	KeypointInstance& keypoint;

	Label lbl_label, lbl_state;
	DropList dl_state;
	Button btn_ok, btn_cancel;
};

#endif
