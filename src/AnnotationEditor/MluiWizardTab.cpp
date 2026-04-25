#include "MluiWizardTab.h"

NAMESPACE_UPP

MluiWizardTab::MluiWizardTab() {
	Add(lbl_title.SetLabel("MLUI Wizard").HSizePos(6, 6).TopPos(6, 20));
	lbl_title.SetFont(StdFont().Bold());
	Add(lbl_script.SetLabel("No script loaded").HSizePos(6, 6).TopPos(28, 18));
	lbl_script.SetInk(SColorDisabled());
	Add(list_slots.HSizePos(6, 6).VSizePos(50, 62));
	list_slots.AddColumn("Slot", 110);
	list_slots.AddColumn("Label", 160);
	list_slots.AddColumn("Category", 100);
	list_slots.AddColumn("Status", 60);
	list_slots.WhenCursor = THISBACK(OnSlotSel);

	Add(opt_enable.SetLabel("Enable").LeftPos(6, 76).BottomPos(34, 24));
	Add(chk_overwrite.SetLabel("overwrite previous").LeftPos(88, 150).BottomPos(34, 24));
	Add(btn_prev.SetLabel("Prev").LeftPos(6, 60).BottomPos(6, 24));
	Add(btn_next.SetLabel("Next").LeftPos(72, 60).BottomPos(6, 24));
	Add(btn_skip.SetLabel("Skip").LeftPos(138, 60).BottomPos(6, 24));
	Add(btn_apply.SetLabel("Apply Slot").RightPos(132, 120).BottomPos(6, 24));
	Add(btn_copy_hints.SetLabel("Copy Hints").RightPos(6, 120).BottomPos(34, 24));

	opt_enable.Set(0);
	chk_overwrite.Set(0);
	btn_prev << THISBACK(OnPrevSlot);
	btn_next << THISBACK(OnNextSlot);
	btn_skip << THISBACK(OnSkipSlot);
	btn_apply << THISBACK(OnApplySlot);
	btn_copy_hints << [=] {
		if(WhenCopyHints)
			WhenCopyHints();
	};
	opt_enable << [=] {
		SyncButtons();
		if(WhenEnable)
			WhenEnable(IsEnabled());
	};
	chk_overwrite << [=] { SyncButtons(); };
	SyncButtons();
}

void MluiWizardTab::ClearScript() {
	script_ = MluiScript();
	cursor_ = -1;
	list_slots.Clear();
	lbl_script.SetLabel("No script loaded");
	SyncButtons();
}

void MluiWizardTab::SetScript(const MluiScript& script) {
	script_ = script;
	cursor_ = script_.slots.IsEmpty() ? -1 : 0;
	RefreshSlots();
	SelectFirstUnfilled();
	SyncButtons();
}

void MluiWizardTab::RefreshFillState() {
	RefreshSlots();
}

bool MluiWizardTab::IsCurrentSlotFilled() const {
	if(cursor_ < 0 || cursor_ >= script_.slots.GetCount())
		return false;
	return IsSlotFilled(script_.slots[cursor_]);
}

bool MluiWizardTab::GetCurrentSlot(MluiScriptSlot& out) const {
	if(cursor_ < 0 || cursor_ >= script_.slots.GetCount())
		return false;
	out = script_.slots[cursor_];
	return true;
}

bool MluiWizardTab::SkipCurrent() {
	if(script_.slots.IsEmpty())
		return false;
	int start = cursor_ >= 0 ? cursor_ : 0;
	int idx = FindNextSlot(start, 1);
	if(idx < 0)
		return false;
	SelectSlot(idx);
	return true;
}

bool MluiWizardTab::IsSlotFilled(const MluiScriptSlot& slot) const {
	if(!WhenCheckFill)
		return false;
	return (bool)WhenCheckFill(slot.slot_id);
}

void MluiWizardTab::SelectSlot(int i) {
	if(i < 0 || i >= script_.slots.GetCount())
		return;
	cursor_ = i;
	list_slots.SetCursor(i);
	SyncButtons();
	OnSlotSel();
}

void MluiWizardTab::SelectFirstUnfilled() {
	if(IsOverwriteEnabled()) {
		if(!script_.slots.IsEmpty())
			SelectSlot(0);
		return;
	}
	for(int i = 0; i < script_.slots.GetCount(); i++) {
		if(!IsSlotFilled(script_.slots[i])) {
			SelectSlot(i);
			return;
		}
	}
	if(!script_.slots.IsEmpty())
		SelectSlot(0);
}

int MluiWizardTab::FindNextSlot(int start, int step) const {
	if(script_.slots.IsEmpty())
		return -1;
	for(int pass = 1; pass <= script_.slots.GetCount(); pass++) {
		int idx = (start + pass * step + script_.slots.GetCount()) % script_.slots.GetCount();
		if(IsOverwriteEnabled() || !IsSlotFilled(script_.slots[idx]))
			return idx;
	}
	return -1;
}

void MluiWizardTab::RefreshSlots() {
	int keep_cursor = list_slots.GetCursor();
	list_slots.Clear();
	String name = script_.name.IsEmpty() ? "(unnamed script)" : script_.name;
	lbl_script.SetLabel(script_.slots.IsEmpty() ? String("No script loaded")
	                                            : name + " (" + AsString(script_.slots.GetCount()) + " slots)");
	for(const auto& slot : script_.slots) {
		String status = IsSlotFilled(slot) ? "Done" : "Open";
		list_slots.Add(slot.slot_id, slot.label, slot.category, status);
	}
	if(!script_.slots.IsEmpty()) {
		if(cursor_ < 0 || cursor_ >= script_.slots.GetCount())
			cursor_ = 0;
		if(keep_cursor >= 0 && keep_cursor < script_.slots.GetCount())
			cursor_ = keep_cursor;
		list_slots.SetCursor(cursor_);
	}
	SyncButtons();
}

void MluiWizardTab::SyncButtons() {
	bool has_slots = !script_.slots.IsEmpty();
	btn_prev.Enable(has_slots);
	btn_next.Enable(has_slots);
	btn_skip.Enable(has_slots);
	btn_apply.Enable(has_slots && cursor_ >= 0 && cursor_ < script_.slots.GetCount());
	btn_copy_hints.Enable(has_slots);
}

void MluiWizardTab::OnSlotSel() {
	if(!list_slots.IsCursor())
		return;
	cursor_ = list_slots.GetCursor();
	if(cursor_ >= 0 && cursor_ < script_.slots.GetCount())
		if(WhenSlotFocus)
			WhenSlotFocus(script_.slots[cursor_].slot_id);
}

void MluiWizardTab::OnPrevSlot() {
	if(script_.slots.IsEmpty())
		return;
	int start = cursor_ >= 0 ? cursor_ : 0;
	int idx = FindNextSlot(start, -1);
	if(idx >= 0)
		SelectSlot(idx);
}

void MluiWizardTab::OnNextSlot() {
	if(script_.slots.IsEmpty())
		return;
	int start = cursor_ >= 0 ? cursor_ : 0;
	int idx = FindNextSlot(start, 1);
	if(idx >= 0)
		SelectSlot(idx);
}

void MluiWizardTab::OnApplySlot() {
	if(cursor_ < 0 || cursor_ >= script_.slots.GetCount())
		return;
	if(WhenApply)
		WhenApply(script_.slots[cursor_]);
	RefreshSlots();
	int idx = FindNextSlot(cursor_, 1);
	if(idx >= 0)
		SelectSlot(idx);
}

void MluiWizardTab::OnSkipSlot() {
	SkipCurrent();
}

END_UPP_NAMESPACE
