#ifndef _OCRForm_AnnotationProposal_h_
#define _OCRForm_AnnotationProposal_h_

#include <Core/Core.h>

#include "OcrAnnotation.h"

NAMESPACE_UPP

struct AnnotationProposal : Moveable<AnnotationProposal> {
	String        analyser_id;
	OcrAnnotation proposed;
	double        confidence = 0.0;
};

END_UPP_NAMESPACE

namespace OCRForm {
using Upp::AnnotationProposal;
}

#endif
