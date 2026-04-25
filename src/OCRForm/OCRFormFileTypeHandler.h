#ifndef _OCRForm_OCRFormFileTypeHandler_h_
#define _OCRForm_OCRFormFileTypeHandler_h_

#include <ScriptCommon/ScriptCommon.h>

NAMESPACE_UPP

class OCRFormFileTypeHandler : public IFileTypeHandler {
public:
	virtual String         GetExtension() const override { return ".ocrform"; }
	virtual String         GetFileDescription() const override { return "OCR Form Annotation"; }
	virtual bool           SupportsHostRole(HostRole role) const override { return role == HOSTROLE_EDITOR; }
	virtual IDocumentHost* CreateEditorHost() override;
};

END_UPP_NAMESPACE

#endif
