#ifndef _OCRForm_OcrAnalyserRegistry_h_
#define _OCRForm_OcrAnalyserRegistry_h_

#include <Core/Core.h>

#include "IOcrAnalyser.h"

NAMESPACE_UPP

class OcrAnalyserRegistry {
public:
	static OcrAnalyserRegistry& Get();

	void Register(IOcrAnalyser& analyser);
	void Unregister(const String& id);

	IOcrAnalyser* Find(const String& id);
	const Vector<IOcrAnalyser*>& GetAll() const;

	Vector<OcrAnnotation> AnalyseAll(const OcrAnalyserContext& ctx);

private:
	OcrAnalyserRegistry() {}

	RWMutex lock;
	Vector<IOcrAnalyser*> analysers;
};

END_UPP_NAMESPACE

namespace OCRForm {
using Upp::OcrAnalyserRegistry;
}

#endif
