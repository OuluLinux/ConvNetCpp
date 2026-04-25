#ifndef _AnnotationTool_AnnotationPlugin_h_
#define _AnnotationTool_AnnotationPlugin_h_

#include <ScriptIDE/ScriptIDE.h>

#include "AnnotationFileTypeHandler.h"

NAMESPACE_UPP

class AnnotationPlugin : public IPlugin {
public:
	virtual String GetID() const override          { return "AnnotationTool"; }
	virtual String GetName() const override        { return "AnnotationTool"; }
	virtual String GetDescription() const override { return "Annotation library editor for detector training and OCR replay."; }
	virtual void   Init(IPluginContext& ctx) override;
	virtual void   Shutdown() override;

private:
	IPluginContext*          context = nullptr;
	AnnotationFileTypeHandler file_type_handler;
};

END_UPP_NAMESPACE

#endif
