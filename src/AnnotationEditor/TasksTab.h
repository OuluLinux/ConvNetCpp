#ifndef _AnnotationEditor_TasksTab_h_
#define _AnnotationEditor_TasksTab_h_

#include "AnnotationEditorCommon.h"

NAMESPACE_UPP
class TasksTab : public ParentCtrl {
public:
	TasksTab() { Add(log.SizePos()); log.SetReadOnly(); log.Set(String("Task scanner initialized.\nWaiting for tasks...\n")); }
	void Log(const String& s) { log.Set(log.Get() + s + "\n"); }
private:
	LineEdit log;
};

END_UPP_NAMESPACE

#endif
