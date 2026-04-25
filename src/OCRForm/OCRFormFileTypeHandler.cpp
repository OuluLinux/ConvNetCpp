#include "OCRForm.h"

NAMESPACE_UPP

IDocumentHost* OCRFormFileTypeHandler::CreateEditorHost()
{
	return new OCRFormDocumentHost();
}

END_UPP_NAMESPACE
