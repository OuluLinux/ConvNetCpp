#include "AnnotationTool.h"

NAMESPACE_UPP

IDocumentHost* AnnotationFileTypeHandler::CreateEditorHost()
{
	return new AnnotationDocumentHost();
}

END_UPP_NAMESPACE
