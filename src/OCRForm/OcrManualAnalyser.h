#ifndef _OCRForm_OcrManualAnalyser_h_
#define _OCRForm_OcrManualAnalyser_h_

#include "IOcrAnalyser.h"

NAMESPACE_UPP

class OcrManualAnalyser : public IOcrAnalyser {
public:
	virtual String GetID() const override { return "manual"; }
	virtual String GetName() const override { return "Manual"; }
	virtual Vector<String> GetSupportedTypes() const override { return Vector<String>(); }
	virtual bool CanAnalyse(const OcrAnalyserContext& ctx) const override { return true; }
	virtual OcrAnnotation Analyse(const OcrAnalyserContext& ctx) override;
	virtual OcrAnnotation Refine(const OcrAnnotation& existing, const OcrAnalyserContext& ctx) override;
};

END_UPP_NAMESPACE

namespace OCRForm {
using Upp::OcrManualAnalyser;
}

#endif
