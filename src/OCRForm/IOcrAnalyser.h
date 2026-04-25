#ifndef _OCRForm_IOcrAnalyser_h_
#define _OCRForm_IOcrAnalyser_h_

#include <Core/Core.h>
#include <Draw/Draw.h>

#include "OcrAnnotation.h"
#include "OcrFormLayout.h"

NAMESPACE_UPP

struct OcrAnalyserContext {
	const Image& frame;
	Rect region;
	const OcrAnnotation* parent;
	const OcrFormLayout& layout;
	const VectorMap<String, String>& global_hints;
};

class IOcrAnalyser {
public:
	virtual ~IOcrAnalyser() {}

	virtual String GetID() const = 0;
	virtual String GetName() const = 0;
	virtual Vector<String> GetSupportedTypes() const { return Vector<String>(); }
	virtual bool CanAnalyse(const OcrAnalyserContext& ctx) const { return true; }
	virtual OcrAnnotation Analyse(const OcrAnalyserContext& ctx) = 0;
	virtual OcrAnnotation Refine(const OcrAnnotation& existing, const OcrAnalyserContext& ctx) {
		return Analyse(ctx);
	}
};

END_UPP_NAMESPACE

namespace OCRForm {
using Upp::IOcrAnalyser;
using Upp::OcrAnalyserContext;
}

#endif
