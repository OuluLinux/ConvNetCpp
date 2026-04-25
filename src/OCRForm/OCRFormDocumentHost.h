#ifndef _OCRForm_OCRFormDocumentHost_h_
#define _OCRForm_OCRFormDocumentHost_h_

#include <FormEditor/FormEditor.h>
#include <ScriptIDE/ScriptIDE.h>

#include "OcrFormDocument.h"
#include "AnnotationProposal.h"
#include "OcrAnalysisWorker.h"
#include "IVideoSource.h"

NAMESPACE_UPP

class OcrFormPreview : public FormView {
public:
	typedef OcrFormPreview CLASSNAME;

	OcrFormPreview();
	void SetChangedRegions(Vector<Rect> regions);
	void ClearChangedRegions();

	virtual void Paint(Draw& w) override;

private:
	Vector<Rect> changed_regions;
};

class OCRFormDocumentHost : public IDocumentHost, public ParentCtrl {
public:
	typedef OCRFormDocumentHost CLASSNAME;

	OCRFormDocumentHost();
	virtual ~OCRFormDocumentHost();

	virtual Ctrl&  GetCtrl() override { return *this; }
	virtual bool   Load(const String& path) override;
	virtual bool   Save() override;
	virtual bool   SaveAs(const String& path) override;
	virtual String GetPath() const override { return path; }
	virtual bool   IsModified() const override { return modified; }
	virtual void   SetFocus() override;
	virtual void   ActivateUI() override;
	virtual void   DeactivateUI() override;
	virtual void   MainMenu(Bar& bar) override;
	virtual void   Toolbar(Bar& bar) override;
	virtual bool   CanRun() const override { return true; }
	virtual bool   IsRunning() const override { return running; }
	virtual RunMode GetRunMode() const override { return running ? RUNMODE_RUN : RUNMODE_NONE; }
	virtual void   Run() override;
	virtual void   Stop() override;
	virtual void   SelectAll() override;

	OcrFormDocument&       GetDocument() { return document; }
	const OcrFormDocument& GetDocument() const { return document; }

private:
	void   RefreshViewFromDocument();
	void   AppendAnnotations(const Array<OcrAnnotation>& annotations, const String& path_prefix);
	void   AppendAnnotationRows(const Array<OcrAnnotation>& annotations, const String& path_prefix, int depth);
	void   RebuildAnnotationPane();
	void   RefreshAnnotationDetails(int row = -1);
	void   SyncSelectionFromPane();
	void   RefreshHostUI();
	void   OnLiveRefresh();
	void   RunAnalysersNow(bool interactive = true);
	void   AcceptSelectedProposal();
	void   RejectSelectedProposal();
	void   AcceptAllProposals();
	void   ClearPendingProposals();
	void   ExportToForm();
	bool   AcceptProposalByPath(const String& path);
	bool   RejectProposalByPath(const String& path);
	void   ClearPendingByPrefix(const String& path_prefix);
	String GetSelectedAnnotationPath() const;
	OcrAnnotation* FindAnnotationByPath(Array<OcrAnnotation>& annotations,
	                                    const String& path,
	                                    const String& path_prefix = String());
	void   RunAnalyserRecursive(const Image& frame,
	                           const OcrFormLayout& layout,
	                           const OcrAnnotation* parent,
	                           const Array<OcrAnnotation>& annotations,
	                           const String& path_prefix,
	                           Index<String>& visited_paths);
	void   OnAnalysisTick();
	void   OnAnalysisResult(const AnalysisResult& result);
	uint64 ComputeFrameHash(const Image& frame) const;
	bool   EnsureVideoSource();
	void   ResetVideoSource();
	Image  CaptureSourceFrame();
	String ResolveSourceParam(const String& type_id, const String& param) const;
	String SuggestFormExportPath() const;
	String BuildAnnotationLabel(const OcrAnnotation& annotation) const;

	String          path;
	bool            modified = false;
	bool            running = false;
	int             refresh_interval_ms = 250;
	ScrollContainer scroll;
	OcrFormPreview  view;
	ArrayCtrl       annotation_list;
	LineEdit        annotation_details;
	Vector<String>  annotation_paths;
	Vector<String>  annotation_descriptions;
	VectorMap<String, AnnotationProposal> pending_proposals;
	TimeCallback    live_refresh;
	TimeCallback    analysis_tick;
	OcrAnalysisWorker analysis_worker;
	bool            analysis_inflight = false;
	int             analysis_generation = 0;
	int             analysis_job_seq = 0;
	int             analysis_last_applied = 0;
	uint64          analysis_last_frame_hash = 0;
	Image           analysis_last_frame;
	Image           last_live_frame;
	int             changed_area_blocks = 0;
	One<IVideoSource> video_source;
	String          active_source_type;
	String          active_source_param;
	String          source_status;
	OcrFormDocument document;
};

END_UPP_NAMESPACE

#endif
