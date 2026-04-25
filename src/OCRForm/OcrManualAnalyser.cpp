#include "OcrManualAnalyser.h"

NAMESPACE_UPP

OcrAnnotation OcrManualAnalyser::Analyse(const OcrAnalyserContext& ctx)
{
	OcrAnnotation result;
	result.rect = ctx.region;
	result.widget_type = OcrWidgetTypes::Custom;
	result.analyser_id = GetID();
	result.confidence = 1.0;
	return result;
}

OcrAnnotation OcrManualAnalyser::Refine(const OcrAnnotation& existing, const OcrAnalyserContext& ctx)
{
	OcrAnnotation result = existing;
	result.analyser_id = GetID();
	result.confidence = 1.0;
	if(result.rect.IsEmpty())
		result.rect = ctx.region;
	return result;
}

END_UPP_NAMESPACE
