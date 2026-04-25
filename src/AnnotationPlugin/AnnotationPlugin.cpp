#include "AnnotationTool.h"

NAMESPACE_UPP

void AnnotationPlugin::Init(IPluginContext& ctx)
{
	context = &ctx;
	context->RegisterFileTypeHandler(file_type_handler);
}

void AnnotationPlugin::Shutdown()
{
	context = nullptr;
}

END_UPP_NAMESPACE
