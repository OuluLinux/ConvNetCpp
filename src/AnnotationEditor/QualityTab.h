#ifndef _AnnotationEditor_QualityTab_h_
#define _AnnotationEditor_QualityTab_h_

#include "AnnotationEditorCommon.h"

NAMESPACE_UPP
class QualityTab : public ParentCtrl {
public:
	QualityTab();
	void UpdateMetrics(const Vector<Dataset>& datasets, const Vector<Category>& categories);
private:
	ArrayCtrl list;
};

END_UPP_NAMESPACE

#endif
