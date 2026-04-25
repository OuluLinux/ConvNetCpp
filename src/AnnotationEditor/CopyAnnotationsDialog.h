#ifndef _AnnotationEditor_CopyAnnotationsDialog_h_
#define _AnnotationEditor_CopyAnnotationsDialog_h_

#include <CtrlLib/CtrlLib.h>
#include "Dataset.h"

using namespace Upp;

class CopyAnnotationsDialog : public TopWindow {
public:
	typedef CopyAnnotationsDialog CLASSNAME;
	CopyAnnotationsDialog(const Dataset& ds, int current_img_idx, const Vector<Category>& categories);

	int GetTargetImageIndex() const;
	Vector<int> GetSelectedCategoryIds() const;
	bool IsFilterEnabled() const { return ~cb_filter; }

private:
	const Dataset& dataset;
	const Vector<Category>& categories;

	Label lbl_target;
	DropList dl_target;
	Option cb_filter;
	ArrayCtrl list_categories;
	Button btn_copy, btn_cancel;

	void OnFilterToggle();
};

#endif
