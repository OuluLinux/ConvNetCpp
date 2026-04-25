#ifndef _AnnotationEditor_AnchoredSlotRecognizer_h_
#define _AnnotationEditor_AnchoredSlotRecognizer_h_

#include "AnnLay.h"
#include "AnnSln.h"
#include "GroupRegistry.h"
#include <ComputerVision/ComputerVision.h>
#include <ConvNet/ConvNet.h>
#include <OCR/OCR.h>

NAMESPACE_UPP

struct SlotResult : Moveable<SlotResult> {
	String slot_id;
	String method;
	String raw_text;
	String top_class;
	int    class_index = -1;
	double confidence = 0.0;
	Rect   pixel_bbox;
	double offset_dx = 0.0;
	double offset_dy = 0.0;
	double t_ms = 0.0; // Recognition time for this slot
	String details;    // Debug details for step-tracing

	// Slice 19: Gate debug info
	String gate_slot_id;
	String gate_status = "not_applicable"; // pass, blocked_false, blocked_missing, blocked_invalid, not_applicable

	// For variable-classifier (anchor_candidates): index of the winning candidate (-1 if N/A)
	int    winner_cand_index = -1;

	// Slice FR-Fix6: OCR Fallback info
	String ocr_primary_mode;
	String ocr_fallback_mode;
	String ocr_mode_used;
	bool   ocr_fallback_used = false;

	void Jsonize(JsonIO& jio) {
		jio("slot_id", slot_id)("method", method)("raw_text", raw_text)("top_class", top_class)
		   ("class_index", class_index)("confidence", confidence)("pixel_bbox", pixel_bbox)
		   ("offset_dx", offset_dx)("offset_dy", offset_dy)("t_ms", t_ms)("details", details)
		   ("gate_slot_id", gate_slot_id)("gate_status", gate_status)
		   ("ocr_primary_mode", ocr_primary_mode)("ocr_fallback_mode", ocr_fallback_mode)
		   ("ocr_mode_used", ocr_mode_used)("ocr_fallback_used", ocr_fallback_used);
	}
};

struct CvCandidateRecord : Moveable<CvCandidateRecord> {
	String class_name;
	double score = 0;
	double zoom = 1.0;
	double rot = 0;
	Image  crop_image;  // input patch used for this match (not serialized)
};

struct ProcessingStepRecord : Moveable<ProcessingStepRecord> {
	int    step_id = -1;
	String stage;
	String slot_id;
	String head_id;           // resolved session key (actual model used)
	String requested_head_id; // logical head before group resolution (empty = same as head_id)
	String top_class;
	double confidence = 0;
	String gate_slot_id;
	String gate_status;
	bool   is_nn_step = false;
	double duration_ms = 0;
	String status;
	String note;
	String detailed_note;
	VectorMap<String, int> counters;
	Rect   candidate_bbox;
	Size   crop_size;
	bool   is_grayscale = false;
	bool   is_equalized = false;
	double angle = 0;
	int    ocr_mode = -1; // OCR::OCRPreprocessMode

	// CV template match debug data (LABEL_A only, not serialized)
	Image                   cv_response_map;  // response map for best zoom/rot
	Image                   cv_input_crop;    // binarized level crop at best zoom/rot
	String                  cv_match_method;  // method used (e.g. "TM_CCOEFF_NORMED")
	Vector<CvCandidateRecord> cv_candidates;  // per-template scores, best first

	void operator<<=(const ProcessingStepRecord& s) {
		step_id = s.step_id; stage = s.stage; slot_id = s.slot_id; head_id = s.head_id;
		requested_head_id = s.requested_head_id;
		top_class = s.top_class; confidence = s.confidence;
		gate_slot_id = s.gate_slot_id; gate_status = s.gate_status;
		is_nn_step = s.is_nn_step; duration_ms = s.duration_ms;
		status = s.status; note = s.note;
		detailed_note = s.detailed_note;
		counters <<= s.counters;
		candidate_bbox = s.candidate_bbox;
		crop_size = s.crop_size;
		is_grayscale = s.is_grayscale;
		is_equalized = s.is_equalized;
		angle = s.angle;
		ocr_mode = s.ocr_mode;
		cv_response_map = s.cv_response_map;
		cv_input_crop = s.cv_input_crop;
		cv_match_method = s.cv_match_method;
		cv_candidates <<= s.cv_candidates;
	}

	void Jsonize(JsonIO& jio) {
		jio("step_id", step_id)("stage", stage)("slot_id", slot_id)("head_id", head_id)
		   ("requested_head_id", requested_head_id)
		   ("gate_slot_id", gate_slot_id)("gate_status", gate_status)
		   ("is_nn_step", is_nn_step)("duration_ms", duration_ms)("status", status)
		   ("note", note)("detailed_note", detailed_note)("counters", counters)
		   ("candidate_bbox", candidate_bbox)("crop_size", crop_size)("is_grayscale", is_grayscale)
		   ("is_equalized", is_equalized)
		   ("angle", angle)("ocr_mode", ocr_mode);
		// cv_response_map, cv_match_method, cv_candidates intentionally not serialized
	}
};

struct ProcessingLogRecord : Moveable<ProcessingLogRecord> {
	int    frame_seq = -1;
	int    flat_index = -1;
	int    next_step_id = 1;
	String image_name;
	String image_path;
	double t_queue_ms = 0;
	double t_load_ms = 0;
	double t_recognize_ms = 0;
	double t_script_ms = 0;
	double t_overlay_ms = 0;
	double t_total_ms = 0;
	int    detections_good = 0;
	int    detections_missing = 0;
	String status; // OK, WARN, ERROR
	String warnings;
	String script_output;
	String script_error;
	String watchlist;

	Vector<SlotResult> results;
	VectorMap<String, String> meta;
	Vector<ProcessingStepRecord> steps;

	void operator<<=(const ProcessingLogRecord& r) {
		frame_seq = r.frame_seq; flat_index = r.flat_index;
		image_name = r.image_name; image_path = r.image_path;
		t_queue_ms = r.t_queue_ms; t_load_ms = r.t_load_ms;
		t_recognize_ms = r.t_recognize_ms; t_script_ms = r.t_script_ms;
		t_overlay_ms = r.t_overlay_ms; t_total_ms = r.t_total_ms;
		detections_good = r.detections_good; detections_missing = r.detections_missing;
		status = r.status; warnings = r.warnings;
		script_output = r.script_output; script_error = r.script_error;
		watchlist = r.watchlist;
		results <<= r.results;
		meta <<= r.meta;
		
		steps.Clear();
		for(int i = 0; i < r.steps.GetCount(); i++)
			steps.Add() <<= r.steps[i];
	}

	void Jsonize(JsonIO& jio) {
		jio("frame_seq", frame_seq)("flat_index", flat_index)("next_step_id", next_step_id)
		   ("image_name", image_name)("image_path", image_path)
		   ("t_queue_ms", t_queue_ms)("t_load_ms", t_load_ms)
		   ("t_recognize_ms", t_recognize_ms)("t_script_ms", t_script_ms)
		   ("t_overlay_ms", t_overlay_ms)("t_total_ms", t_total_ms)
		   ("detections_good", detections_good)("detections_missing", detections_missing)
		   ("status", status)("warnings", warnings)
		   ("script_output", script_output)("script_error", script_error)
		   ("results", results)("meta", meta)("steps", steps);
	}

	String FormatVerbose() const;
	void AddStep(const String& stage, double duration_ms, const String& status,
	             const String& note = String(), const String& slot_id = String(),
	             const String& head_id = String(), bool is_nn = false,
	             const String& detailed_note = String(),
	             const String& gate_slot_id = String(), const String& gate_status = String());

};

struct ProcessingTreeNode : Moveable<ProcessingTreeNode> {
	String label;
	String path;
	String stage;
	double duration_ms = 0;
	String status;
	String note;
	bool   is_leaf = false;
	int    step_id = -1;
	int    source_index = -1;
	Vector<int> children;

	void operator<<=(const ProcessingTreeNode& s) {
		label = s.label;
		path = s.path;
		stage = s.stage;
		duration_ms = s.duration_ms;
		status = s.status;
		note = s.note;
		is_leaf = s.is_leaf;
		step_id = s.step_id;
		source_index = s.source_index;
		children <<= s.children;
	}
};

struct ProcessingInputRef : Moveable<ProcessingInputRef> {
	String id;
	Rect   bbox;

	void operator<<=(const ProcessingInputRef& s) {
		id = s.id;
		bbox = s.bbox;
	}
};

void BuildProcessingStepsTree(const ProcessingLogRecord& log, const AnnSln& sln, Vector<ProcessingTreeNode>& out_nodes,
                              Vector<ProcessingInputRef>* out_inputs = nullptr);


enum OffsetMode {
	OFFSET_AUTO = 0,
	OFFSET_NONE,
	OFFSET_COMBINED,
	OFFSET_SPLIT
};

inline const char* OffsetModeToString(OffsetMode m) {
	switch(m) {
		case OFFSET_NONE: return "none";
		case OFFSET_COMBINED: return "combined";
		case OFFSET_SPLIT: return "split";
		case OFFSET_AUTO: return "auto";
		default: return "auto";
	}
}

inline OffsetMode StringToOffsetMode(const String& s) {
	if(s == "none") return OFFSET_NONE;
	if(s == "combined") return OFFSET_COMBINED;
	if(s == "split") return OFFSET_SPLIT;
	return OFFSET_AUTO;
}

enum BoolGatePolicy {
	BOOL_GATE_PERMISSIVE = 0,
	BOOL_GATE_STRICT
};

inline const char* BoolGatePolicyToString(BoolGatePolicy p) {
	switch(p) {
		case BOOL_GATE_STRICT: return "strict";
		case BOOL_GATE_PERMISSIVE: return "permissive";
		default: return "permissive";
	}
}

inline BoolGatePolicy StringToBoolGatePolicy(const String& s) {
	if(s == "strict") return BOOL_GATE_STRICT;
	return BOOL_GATE_PERMISSIVE;
}

void SetFrameRecognizerMemoryDumpEnabled(bool enabled);
bool IsFrameRecognizerMemoryDumpEnabled();
void DumpFrameRecognizerMemoryEvent(const String& event, const String& details = String());

Image RotateBilinear(const Image& src, double angle_deg);

class AnchoredSlotRecognizer {
public:
	AnchoredSlotRecognizer();
	~AnchoredSlotRecognizer();

	bool Load(const String& annlay_path, const String& annmdl_path = String());

	// Optional: set a loaded AnnSln before Load() to enable cv_template_match groups.
	// If not set, Load() will attempt to find a sidecar .annsln in the same directory.
	void SetSln(const AnnSln& s) { sln = s; sln_set = true; }

	bool IsLoaded() const { return loaded; }

	Vector<SlotResult> Recognize(const Image& img, ProcessingLogRecord* log = nullptr);
	void Recognize(const Image& img, VectorMap<String, SlotResult>& out, ProcessingLogRecord* log = nullptr);

	::ConvNet::Session* GetSession(const String& head_id);
	Vector<String> GetLoadedSessionHeads();

	const AnnLay& GetLayout() const { return lay; }

	int ocr_upscale = 3;
	static bool IsIntSlot(const String& slot_id);

	void SetOffsetMode(OffsetMode m) { offset_mode = m; }
	OffsetMode GetOffsetMode() const { return offset_mode; }

	void SetBoolGatePolicy(BoolGatePolicy p) { bool_gate_policy = p; }
	BoolGatePolicy GetBoolGatePolicy() const { return bool_gate_policy; }

	void SetOcrBackend(OCR::OCRBackend b);
	OCR::OCRBackend GetOcrBackend() const;

	void SetSlotFilter(const Index<String>& filter) { slot_filter.Clear(); for(int i = 0; i < filter.GetCount(); i++) slot_filter.Add(filter[i]); }
	void ClearSlotFilter() { slot_filter.Clear(); }

	void SetHeadFilter(const Index<String>& filter) { head_filter.Clear(); for(int i = 0; i < filter.GetCount(); i++) head_filter.Add(filter[i]); }
	void ClearHeadFilter() { head_filter.Clear(); }

	Rect AnchorToRect(const AnnLayAnchor& anchor, Size img_size) const;
	Rect AnchorToRect(const AnnLayAnchor& anchor, Size img_size, double dx, double dy) const;
	Image CropAndScale(const Image& img, Rect r, Size target_size) const;
	Image CropRotateScale(const Image& img, const AnnLayAnchor& anchor, Size target_size, double dx, double dy, double angle) const;
	Vector<double> ImageToVec(const Image& crop, bool grayscale, Size sz, bool equalize = false) const;
	void PredictCrop(const String& head_id, const Image& img, Rect bbox, Size crop_size, bool grayscale, double angle = 0, bool equalize = true);

	String RunTesseract(const Image& crop, const String& slot_id, String& error_reason, double& confidence, Rect& out_bbox, OCR::OCRPreprocessMode mode = OCR::OCR_PREPROCESS_GRAYSCALE, int psm = -1, const String& whitelist = String(), const String& blacklist = String());

private:
	AnnLay lay;
	AnnSln sln;
	bool sln_set = false;
	bool loaded = false;
	bool offset_strategy_logged = false;
	OffsetMode offset_mode = OFFSET_AUTO;
	BoolGatePolicy bool_gate_policy = BOOL_GATE_PERMISSIVE;
	OCR::OCRBackend ocr_backend = OCR::OCR_TESSERACT;
	Index<String> slot_filter;
	Index<String> head_filter;
	Mutex recognize_lock;

	VectorMap<String, One< ::ConvNet::Session > > sessions;
	VectorMap<String, String> head_preprocess_mode_;
	GroupRegistry group_registry_;
	VectorMap<String, SlotResult> ocr_cache_;
	int64 cached_image_serial = -1;
	OCR::OCREngine ocr_engine;
	bool ocr_initialized = false;
	bool ocr_eager_ = false;    // if true, run all OCR slots upfront in ComputeOcrOffset

	// cv_template_match: pre-scaled templates per group × zoom level
	// Outer: group index; inner: zoom step; value: map from class name to grayscale ByteMat
	struct CvTemplateCache;
	Vector< One<CvTemplateCache> > tmpl_cache_;

	void ComputeOcrOffset(const Image& img, double& out_dx, double& out_dy, ProcessingLogRecord* log = nullptr);
	void ComputeLocalOffset(const AnnLaySlot& slot, const Image& img, double global_dx, double global_dy, double& out_dx, double& out_dy);
	void ApplyBoolGates(Vector<SlotResult>& results);

	void LoadCvTemplateGroups(const String& annlay_path);
	void RecognizeCvTemplateGroup(int group_idx, const String& stem,
	                              const Image& img, double dx, double dy,
	                              Vector<SlotResult>& out, ProcessingLogRecord* log);

	SlotResult RecognizeClassifierWithOffset(const AnnLaySlot& slot, const Image& img, double dx, double dy, ProcessingLogRecord* log = nullptr);
	SlotResult RecognizeVariableClassifier(const AnnLaySlot& slot, const Image& img, double dx, double dy, ProcessingLogRecord* log = nullptr);
	SlotResult RecognizeCompositeElement(const AnnLaySlot& slot, const Image& img, double dx, double dy, ProcessingLogRecord* log = nullptr);
	SlotResult RecognizeCompositeElementIterative(const AnnLaySlot& slot, const Image& img, double dx, double dy, ProcessingLogRecord* log = nullptr);
	SlotResult RecognizeOrb(const AnnLaySlot& slot, const Image& img);
	bool ShouldUseHighLumaHead(const String& head_id) const;
};

END_UPP_NAMESPACE

#endif
