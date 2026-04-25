#ifndef _OCRForm_OCRFormPlugin_h_
#define _OCRForm_OCRFormPlugin_h_

#include <ScriptIDE/ScriptIDE.h>

#include "OCRFormFileTypeHandler.h"
#include "OcrManualAnalyser.h"

NAMESPACE_UPP

class OCRFormPlugin : public IPlugin {
public:
	virtual String GetID() const override { return "ocr.form"; }
	virtual String GetName() const override { return "OCR Form Editor"; }
	virtual String GetDescription() const override { return "Provides OCR form annotation editing and runtime hosting."; }
	virtual void   Init(IPluginContext& ctx) override;
	virtual void   Shutdown() override;

private:
	IPluginContext*      context = nullptr;
	OCRFormFileTypeHandler file_type_handler;
	OcrManualAnalyser      manual_analyser;
};

END_UPP_NAMESPACE

#endif
