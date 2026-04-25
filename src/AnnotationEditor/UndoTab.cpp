#include "UndoTab.h"

NAMESPACE_UPP

UndoTab::UndoTab() {
	Add(list.SizePos());
	list.Add("Application started");
}

void UndoTab::SetCommandManager(CommandManager& m) {
	cmdmgr = &m;
}

void UndoTab::RefreshHistory() {
	list.Clear();
	if(!cmdmgr)
		return;
	const auto& undo = cmdmgr->GetUndoStack();
	for(int i = 0; i < undo.GetCount(); i++)
		list.Add(Format("[%d] %s", i + 1, undo[i].GetName()));
	const auto& redo = cmdmgr->GetRedoStack();
	for(int i = redo.GetCount() - 1; i >= 0; i--)
		list.Add(Format("[%d] (Redo) %s", undo.GetCount() + (redo.GetCount() - i), redo[i].GetName()));
	list.SetCursor(undo.GetCount() - 1);
}

void UndoTab::Log(const String& s) {
	list.Add(s);
	list.SetCursor(list.GetCount() - 1);
}

END_UPP_NAMESPACE
