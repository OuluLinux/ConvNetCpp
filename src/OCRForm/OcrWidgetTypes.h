#ifndef _OCRForm_OcrWidgetTypes_h_
#define _OCRForm_OcrWidgetTypes_h_

#include <Core/Core.h>

NAMESPACE_UPP

namespace OcrWidgetTypes {
	static const char *const Button     = "button";
	static const char *const Label      = "label";
	static const char *const EditField  = "editfield";
	static const char *const MenuBar    = "menubar";
	static const char *const MenuItem   = "menu_item";
	static const char *const Splitter   = "splitter";
	static const char *const Icon       = "icon";
	static const char *const Separator  = "separator";
	static const char *const Custom     = "custom";
	static const char *const Unknown    = "";
}

END_UPP_NAMESPACE

namespace OCRForm {
namespace OcrWidgetTypes = Upp::OcrWidgetTypes;
}

#endif
