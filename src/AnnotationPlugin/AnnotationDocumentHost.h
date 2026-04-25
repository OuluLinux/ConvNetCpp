#ifndef _AnnotationTool_AnnotationDocumentHost_h_
#define _AnnotationTool_AnnotationDocumentHost_h_

#include <ScriptIDE/ScriptIDE.h>

#include "AnnotationToolWindow.h"

NAMESPACE_UPP

class AnnotationDocumentHost : public IDocumentHost, public ParentCtrl {
public:
	typedef AnnotationDocumentHost CLASSNAME;

	AnnotationDocumentHost();

	virtual Ctrl&  GetCtrl() override         { return *this; }
	virtual bool   Load(const String& path) override;
	virtual bool   Save() override;
	virtual bool   SaveAs(const String& path) override;
	virtual String GetPath() const override   { return path; }
	virtual bool   IsModified() const override{ return modified; }
	virtual void   SetFocus() override;
	virtual void   ActivateUI() override      {}
	virtual void   DeactivateUI() override    {}
	virtual void   MainMenu(Bar& bar) override;
	virtual void   Toolbar(Bar& bar) override;
	virtual bool   CanRun() const override    { return false; }
	virtual bool   IsRunning() const override { return false; }
	virtual RunMode GetRunMode() const override { return RUNMODE_NONE; }
	virtual void   Run() override             {}
	virtual void   Stop() override            {}
	virtual void   SelectAll() override;

private:
	void NewDocument();
	void DoExportCOCO();

	String               path;
	bool                 modified = false;
	AnnLib               document;
	AnnotationToolWindow editor;
};

END_UPP_NAMESPACE

#endif
