#ifndef _OCRForm_OcrFormLayout_h_
#define _OCRForm_OcrFormLayout_h_

#include <Core/Core.h>
#include <Form/Common/XMLConfig.h>

#include "OcrAnnotation.h"

NAMESPACE_UPP

class OcrFormLayout : public XMLConfig, Moveable<OcrFormLayout> {
public:
	String name;
	Size canvas_size;
	Array<OcrAnnotation> roots;

	OcrFormLayout();
	OcrFormLayout(const OcrFormLayout& src);
	OcrFormLayout& operator=(const OcrFormLayout& src);

	void Xmlize(XmlIO xml);
};

END_UPP_NAMESPACE

namespace OCRForm {
using Upp::OcrFormLayout;
}

#endif
