#ifndef _AnnotationEditor_MluiWizardTab_h_
#define _AnnotationEditor_MluiWizardTab_h_

#include "AnnotationEditorCommon.h"

NAMESPACE_UPP
class MluiWizardTab : public ParentCtrl {
public:
	typedef MluiWizardTab CLASSNAME;

	MluiWizardTab();

	void ClearScript();
	void SetScript(const MluiScript& script);
	void RefreshFillState();

	bool IsEnabled() const { return (bool)opt_enable.Get(); }
	bool IsOverwriteEnabled() const { return (bool)chk_overwrite.Get(); }
	bool IsCurrentSlotFilled() const;
	bool GetCurrentSlot(MluiScriptSlot& out) const;
	bool SkipCurrent();

	Function<bool(const String&)> WhenCheckFill;
	Event<const MluiScriptSlot&> WhenApply;
	Event<> WhenCopyHints;
	Event<String> WhenSlotFocus;
	Event<bool> WhenEnable;

private:
	bool IsSlotFilled(const MluiScriptSlot& slot) const;
	void SelectSlot(int i);
	void SelectFirstUnfilled();
	int FindNextSlot(int start, int step) const;
	void RefreshSlots();
	void SyncButtons();
	void OnSlotSel();
	void OnPrevSlot();
	void OnNextSlot();
	void OnApplySlot();
	void OnSkipSlot();

	Label lbl_title, lbl_script;
	ArrayCtrl list_slots;
	Option opt_enable, chk_overwrite;
	Button btn_prev, btn_next, btn_skip, btn_apply, btn_copy_hints;
	MluiScript script_;
	int cursor_ = -1;
};

END_UPP_NAMESPACE

#endif
