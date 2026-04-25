#ifndef _AnnotationEditor_UndoTab_h_
#define _AnnotationEditor_UndoTab_h_

#include "AnnotationEditorCommon.h"

NAMESPACE_UPP
class UndoTab : public ParentCtrl {
public:
	UndoTab();
	void SetCommandManager(CommandManager& m);
	void RefreshHistory();
	void Log(const String& s);
private:
	ColumnList list; CommandManager* cmdmgr = nullptr;
};

END_UPP_NAMESPACE

#endif
