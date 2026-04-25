#ifndef _AnnotationTool_AnnotationFileTypeHandler_h_
#define _AnnotationTool_AnnotationFileTypeHandler_h_

#include <ScriptIDE/ScriptIDE.h>

NAMESPACE_UPP

class AnnotationFileTypeHandler : public IFileTypeHandler {
public:
	virtual String GetExtension() const override       { return ".annlib"; }
	virtual String GetFileDescription() const override { return "AnnotationTool Library"; }
	virtual bool SupportsHostRole(HostRole role) const override { return role == HOSTROLE_EDITOR; }
	virtual IDocumentHost* CreateEditorHost() override;
};

END_UPP_NAMESPACE

#endif
