#ifndef _OCRForm_OcrAnnotation_h_
#define _OCRForm_OcrAnnotation_h_

#include <Core/Core.h>
#include <Form/Common/XMLConfig.h>

#include "OcrWidgetTypes.h"

NAMESPACE_UPP

class OcrAnnotation : public XMLConfig, Moveable<OcrAnnotation> {
public:
	Rect rect;
	String widget_type;
	String name;
	Array<OcrAnnotation> children;
	String analyser_id;
	double confidence;
	VectorMap<String, String> hints;
	bool selected;

	OcrAnnotation();
	OcrAnnotation(const OcrAnnotation& src);
	OcrAnnotation& operator=(const OcrAnnotation& src);

	void Xmlize(XmlIO xml);
};

END_UPP_NAMESPACE

namespace OCRForm {
using Upp::OcrAnnotation;
}

#endif
