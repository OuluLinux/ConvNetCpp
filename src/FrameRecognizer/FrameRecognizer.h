#ifndef _FrameRecognizer_FrameRecognizer_h_
#define _FrameRecognizer_FrameRecognizer_h_

#include <CtrlLib/CtrlLib.h>
#include <Docking/Docking.h>
#include <atomic>

#include "VideoFeedView.h"
#include <AnnLayCore/AnnSln.h>
#include <AnnLayCore/AnchoredSlotRecognizer.h>
#include <AnnLayCore/RecognitionScript.h>
#include <ConvNetCtrl/ConvNetCtrl.h>

NAMESPACE_UPP

Image  RenderOverlayImage(const Image& src, const Vector<SlotResult>& results, bool show_offsets);
String ResolveImagePath(const String& sln_dir, const String& images_dir, const Value& img_rec);
String ResultDisplayValue(const SlotResult& r);
Color  SlotColor(const String& slot_id);

// Base class for modular pipeline steps
struct PipelineStep {
	virtual ~PipelineStep() = default;
	virtual String Name() const = 0;
	virtual void Run(const Image& input, Vector<SlotResult>& results, VectorMap<String, String>& meta, ProcessingLogRecord& log) = 0;
	virtual String GetStatus() const = 0;
	virtual double GetDurationMs() const = 0;
	virtual String GetNote() const = 0;
	virtual bool IsNNStep() const { return false; }
	virtual String GetHeadId() const { return ""; }
};

// Concrete step implementations
struct ImageLoadStep : public PipelineStep {
	Image image;
	String image_path;
	double duration_ms = 0.0;
	String status = "";
	String note = "";

	String Name() const override { return "Image Load/Decode"; }
	void Run(const Image& input, Vector<SlotResult>& results, VectorMap<String, String>& meta, ProcessingLogRecord& log) override;
	String GetStatus() const override { return status; }
	double GetDurationMs() const override { return duration_ms; }
	String GetNote() const override { return note; }
};

struct RecognizerStep : public PipelineStep {
	AnchoredSlotRecognizer* recognizer;
	Image input_image;
	Vector<SlotResult> results;
	double duration_ms = 0.0;
	String status = "";
	String note = "";

	String Name() const override { return "Recognizer Run"; }
	void Run(const Image& input, Vector<SlotResult>& results, VectorMap<String, String>& meta, ProcessingLogRecord& log) override;
	String GetStatus() const override { return status; }
	double GetDurationMs() const override { return duration_ms; }
	String GetNote() const override { return note; }
	bool IsNNStep() const override { return true; }
	String GetHeadId() const override { return "recognizer"; }
};

struct OverlayRenderStep : public PipelineStep {
	Image input_image;
	Vector<SlotResult> results;
	Image output_image;
	double duration_ms = 0.0;
	String status = "";
	String note = "";

	String Name() const override { return "Overlay Render"; }
	void Run(const Image& input, Vector<SlotResult>& results, VectorMap<String, String>& meta, ProcessingLogRecord& log) override;
	String GetStatus() const override { return status; }
	double GetDurationMs() const override { return duration_ms; }
	String GetNote() const override { return note; }
};

struct SummaryStep : public PipelineStep {
	Vector<SlotResult> results;
	VectorMap<String, String> meta;
	double duration_ms = 0.0;
	String status = "";
	String note = "";

	String Name() const override { return "Summary/Statistics"; }
	void Run(const Image& input, Vector<SlotResult>& results, VectorMap<String, String>& meta, ProcessingLogRecord& log) override;
	String GetStatus() const override { return status; }
	double GetDurationMs() const override { return duration_ms; }
	String GetNote() const override { return note; }
};

class CvTemplateStepInspector : public Ctrl {
public:
	typedef CvTemplateStepInspector CLASSNAME;

	void Clear();
	void SetData(const Image& img, const String& title, const String& note);

	virtual void Paint(Draw& w) override;

private:
	Image  preview_;
	String title_;
	String note_;
};

class LabelAMatchVisualizer : public Ctrl {
public:
	typedef LabelAMatchVisualizer CLASSNAME;

	void Clear();
	void SetStep(const ProcessingStepRecord* ps, const Image& response_map_override = Image());
	void SetMethodOverride(TemplateMatchMethod m);
	void ClearMethodOverride();
	TemplateMatchMethod GetEffectiveMethod() const;

	// Called when user picks a method from the right-click menu.
	// Arg is -1 to reset to pipeline default.
	Function<void(int)> WhenMethodOverride;

	virtual void Paint(Draw& w) override;
	virtual void RightDown(Point, dword) override;

	static String MethodName(TemplateMatchMethod m);
	static TemplateMatchMethod MethodFromName(const String& s);

private:
	Image  response_map_;
	String pipeline_method_;
	bool   has_override_ = false;
	TemplateMatchMethod method_override_ = TM_CCOEFF_NORMED;
	String slot_id_;

};


class OcrStepInspector : public Ctrl {
public:
	typedef OcrStepInspector CLASSNAME;

	OcrStepInspector();

	void Clear();
	void SetData(const Image& original, const Image& preprocessed, const String& title, const String& note,
	             int tesseract_psm = 7, const String& ocr_whitelist = String(), const String& ocr_blacklist = String());

	virtual void Paint(Draw& w) override;
	virtual void RightDown(Point, dword) override;

private:
	void CopyTesseractCommand();
	Image BuildTesseractInputImage(const Image& src) const;
	String BuildTesseractCommand(const String& image_path) const;

	Image  original_;
	Image  preprocessed_;
	String title_;
	String note_;
	int    tesseract_psm_ = 7;
	String ocr_whitelist_;
	String ocr_blacklist_;
};

class FrameRecognizerWindow : public DockWindow {

public:
	typedef FrameRecognizerWindow CLASSNAME;

	FrameRecognizerWindow();
	virtual ~FrameRecognizerWindow();

	void OpenSlideshow(const AnnSln& sln, const String& sln_dir,
	                   const String& annprj_path,
	                   const String& model_set = "pass1");

	void OpenRealtime(const AnnSln& sln, const String& sln_dir,
	                  const String& model_set = "pass2");

	void TestDumpXOffsetConvLayers(bool verbose);

	void SetOffsetMode(OffsetMode m) { offset_mode_drop_.SetData(OffsetModeToString(m)); }
	void SetBoolGatePolicy(BoolGatePolicy p) { bool_policy_drop_.SetData(BoolGatePolicyToString(p)); }

	virtual void DockInit() override;
	virtual void Close() override;

	bool LoadProject(const String& path);
	bool LoadRecognizer(const String& model_set);
	bool LoadRecognizer(const String& annmdl_path, const String& annlay_path);
	void PopulateStepsTree(const ProcessingLogRecord& log);

	Callback WhenSelNN;

private:
	MenuBar       menu_;
	ToolBar       toolbar_;
	TabCtrl       tabs;
	VideoFeedView video_feed_;
	ArrayCtrl     detections_list_;
	ArrayCtrl     images_list_;  // New ArrayCtrl for images
	DropList      model_set_drop_;
	DropList      offset_mode_drop_;
	DropList      bool_policy_drop_;
	Label         lbl_offset_mode_;
	Label         lbl_bool_policy_;

	DockableCtrl* dock_detections_ = nullptr;
	DockableCtrl* dock_images_ = nullptr;  // New dock for images list

	String                  sln_dir_;
	String                  annprj_path_;
	String                  sln_annlay_;
	String                  sln_recognition_script_;
	String                  sln_images_dir_;
	VectorMap<String, String> sln_model_sets_;
	AnnSln                    sln_cfg_;
	bool                      has_sln_cfg_ = false;
	bool                    slideshow_mode_ = false;

	AnchoredSlotRecognizer recognizer_;
	RecognitionScript      script_;
	bool                   rec_loaded_ = false;
	mutable Mutex          recognizer_lock_;

	Vector<Value> slideshow_images_;
	Vector<int>   slideshow_dataset_idx_;
	Vector<int>   slideshow_image_idx_;
	int           slideshow_idx_ = 0;
	bool          slideshow_running_ = false;
	Value         annprj_root_;
	VectorMap<int, VectorMap<String, String>> results_cache_;
	VectorMap<int, Vector<SlotResult>>        slot_results_cache_;
	int           current_flat_index_ = -1;
	int           current_frame_seq_ = 0;

	struct RecognitionJob {
		Image img;
		String image_path;
		bool show_offsets = true;
		int flat_index = -1;
		int frame_seq = -1;
		uint64 start_time = 0;
	};

	Thread                recog_thread_;
	mutable Mutex         recog_lock_;
	bool                  recog_busy_ = false;
	bool                  has_pending_job_ = false;
	RecognitionJob        pending_job_;
	bool                  has_completed_job_ = false;
	Image                 completed_source_image_;
	Image                 completed_display_image_;
	Vector<SlotResult>    completed_results_;
	VectorMap<String, String> completed_meta_;
	int                   completed_flat_index_ = -1;
	int                   completed_frame_seq_ = -1;
	ProcessingLogRecord   completed_log_record_;
	ProcessingLogRecord   displayed_log_;
	std::atomic<bool>     closing_{false};

	Thread                image_proc_thread_;
	mutable Mutex         image_proc_lock_;
	bool                  image_proc_busy_ = false;
	int                   image_proc_row_ = -1;
	Image                 image_proc_source_image_;
	Image                 image_proc_display_image_;
	Vector<SlotResult>    image_proc_results_;

	ArrayCtrl             log_list_;
	DocEdit               log_details_;
	DockableCtrl*         dock_log_ = nullptr;
	DockableCtrl*         dock_log_details_ = nullptr;
	Vector<ProcessingLogRecord> log_history_;

	Splitter              steps_splitter_;
	TreeArrayCtrl         steps_list_;
	Splitter              steps_details_splitter_;
	DocEdit               steps_details_;
	Splitter              steps_inspector_splitter_; // horz: inspector host + label a vis
	ParentCtrl            steps_inspector_host_;
	::ConvNet::SessionConvLayers steps_nn_inspector_;
	CvTemplateStepInspector       steps_cv_inspector_;
	OcrStepInspector              steps_ocr_inspector_;
	DocEdit                       steps_script_inspector_;
	Splitter              steps_label_a_splitter_;  // vert: response map + candidates + full template
	LabelAMatchVisualizer steps_label_a_vis_;
	ArrayCtrl             steps_label_a_candidates_;
	CvTemplateStepInspector       steps_label_a_template_view_;
	ProcessingStepRecord* label_a_step_ = nullptr;

	int                   stats_processed_ = 0;
	int                   stats_failed_ = 0;
	double                stats_avg_total_ms_ = 0;
	double                stats_avg_recog_ms_ = 0;
	double                stats_max_total_ms_ = 0;
	String                stats_last_status_;

	enum {
		TIMEID_SLIDESHOW = TopWindow::TIMEID_COUNT,
		TIMEID_COUNT
	};

	void RunOnCurrentFrame(const Image& img);
	void QueueRecognition(const RecognitionJob& job);
	void StartRecognitionThread(const RecognitionJob& job);
	void OnRecognitionReady();
	void StopRecognitionThread();
	void UpdateOverlay(const Vector<SlotResult>& results);
	void UpdateDetectionsDock(const Vector<SlotResult>& results);
	void UpdateImageList();
	void UpdateLogDetail();
	void PopulateLabelAInspector(ProcessingStepRecord& ps, TemplateMatchMethod method);
	void ShowCurrentImage();

	void OnSlideshowTimer();
	void OnPlayPause();
	void OnSaveResults();
	void OnSaveLog();
	void OnLoadLog();
	void OnClearLog();
	void OnModelSetChanged();
	void OnOffsetModeChanged();
	void OnBoolPolicyChanged();
	void OnToggleOffsets();
	void OnProcessSelectedImage();
	void OnImageListContextMenu(Bar& bar);
	void OnImageListCursor();

	void MainMenu(Bar& bar);
	void BuildToolbar(Bar& bar);
};

END_UPP_NAMESPACE
#endif
