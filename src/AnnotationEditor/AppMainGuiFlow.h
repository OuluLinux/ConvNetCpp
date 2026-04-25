#ifndef _AnnotationEditor_AppMainGuiFlow_h_
#define _AnnotationEditor_AppMainGuiFlow_h_

#include "AppMainArgs.h"

namespace Upp {

enum WindowKind {
	WIN_EXIT,
	WIN_PROJECT_MANAGER,
	WIN_ANNOTATION_EDITOR,
	WIN_SLOT_TRAINER,
};

void RunGuiFlow(const AppArguments& args);

}

#endif
