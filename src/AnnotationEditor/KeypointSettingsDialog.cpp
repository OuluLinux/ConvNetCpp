#include "KeypointSettingsDialog.h"

KeypointSettingsDialog::KeypointSettingsDialog(KeypointInstance& kp) : keypoint(kp) {
	Title("Keypoint Settings: " + kp.label);
	SetRect(0, 0, 300, 150);

	Add(lbl_label.SetLabel("Label: " + kp.label).LeftPos(10, 200).TopPos(10, 20));
	
	Add(lbl_state.SetLabel("Visibility State:").LeftPos(10, 200).TopPos(40, 20));
	Add(dl_state.HSizePos(10, 10).TopPos(60, 20));
	dl_state.Add(0, "Not labeled");
	dl_state.Add(1, "Labeled not visible");
	dl_state.Add(2, "Labeled visible");
	dl_state.SetData(keypoint.visibility_state);

	Add(btn_ok.SetLabel("OK").RightPos(100, 80).BottomPos(10, 25));
	Add(btn_cancel.SetLabel("Cancel").RightPos(10, 80).BottomPos(10, 25));

	btn_ok << THISBACK(OnOK);
	btn_cancel << THISBACK(OnCancel);
}

void KeypointSettingsDialog::OnOK() {
	keypoint.visibility_state = ~dl_state;
	Break(IDOK);
}

void KeypointSettingsDialog::OnCancel() {
	Break(IDCANCEL);
}
