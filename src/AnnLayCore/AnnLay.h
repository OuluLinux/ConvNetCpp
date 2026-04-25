#ifndef _AnnotationEditor_AnnLay_h_
#define _AnnotationEditor_AnnLay_h_

#include <Core/Core.h>

NAMESPACE_UPP

// Shared high-luminance binarization threshold used by recognizer/exporter preprocessing.
// 0.85 means pixels with luminance >= 85% are mapped to white; others to black.
#ifndef ANNLAY_HIGH_LUMINANCE_THRESHOLD
#define ANNLAY_HIGH_LUMINANCE_THRESHOLD 0.7
#endif

struct AnnLayAnchor : Moveable<AnnLayAnchor> {
	double cx = 0, cy = 0, w = 0, h = 0;
	double std_cx = 0, std_cy = 0;
	int    n_annotations = 0;

	void Jsonize(JsonIO& jio);
	bool IsEmpty() const { return n_annotations == 0; }
};

struct BoolGateMetrics : Moveable<BoolGateMetrics> {
	String slot_id;
	int tp = 0, fp = 0, tn = 0, fn = 0;
	int samples = 0;
	int gt_pos = 0;
	double precision = 0;
	double recall = 0;
	double f1 = 0;
	double accuracy = 0;
	double threshold = 0.5;

	void Jsonize(JsonIO& jio);
	void Calc();
};

enum AnnLayMethod {
	ANNLAY_OCR_TEXT = 0,
	ANNLAY_CLASSIFIER_BOOL,
	ANNLAY_CLASSIFIER_LABEL,
	ANNLAY_ORB_MATCH,
	ANNLAY_IGNORED,
	ANNLAY_CV_TEMPLATE_MATCH,
};

inline bool IsTrainableMethod(int m) { return m == ANNLAY_CLASSIFIER_BOOL || m == ANNLAY_CLASSIFIER_LABEL; }

String AnnLayMethodToString(AnnLayMethod m);
AnnLayMethod AnnLayMethodFromString(const String& s);

enum AnnLayCompositeType {
	ANNLAY_COMPOSITE_NONE = 0,
	ANNLAY_COMPOSITE_ELEMENT,
};

String AnnLayCompositeTypeToString(AnnLayCompositeType t);
AnnLayCompositeType AnnLayCompositeTypeFromString(const String& s);

enum AnnLaySplitPolicy {
	ANNLAY_SPLIT_NONE = 0,
	ANNLAY_SPLIT_VERTICAL_HALVES,
};

String AnnLaySplitPolicyToString(AnnLaySplitPolicy p);
AnnLaySplitPolicy AnnLaySplitPolicyFromString(const String& s);

struct AnnLaySlot : Moveable<AnnLaySlot> {
	String            id;
	AnnLayMethod      method = ANNLAY_IGNORED;
	String            color_mode = "grayscale";
	Size              crop_size = Size(32, 32);
	// Optional JSON splitter-tree layout (from annotation metadata key "sub_layout_json").
	// Leaf names typically: card_region, rank_region, suit_region.
	String            sub_layout_json;
	AnnLayAnchor      anchor;
	Vector<AnnLayAnchor> anchor_candidates;
	Vector<String>    classes;
	// Natural size of template images (e.g. 132x174 for element templates).
	// Used to map template_crop rect to the anchor crop from game frames.
	int               template_w = 0;
	int               template_h = 0;
	// Pixel crop rect for template images after rotation.
	// If all zero, full image is used.
	int               template_crop_x = 0;
	int               template_crop_y = 0;
	int               template_crop_w = 0;
	int               template_crop_h = 0;

	AnnLayCompositeType composite_type = ANNLAY_COMPOSITE_NONE;
	AnnLaySplitPolicy   split_policy = ANNLAY_SPLIT_NONE;
	String              group;
	String              head_id;
	String              gate;
	VectorMap<String, String> sub_groups; // sub-task name (e.g. "presence") -> group key
	VectorMap<String, String> sub_heads;  // sub-task name -> session head id
	double              presence_threshold = 0.5;
	double              fixed_rotation_deg = 0;
	String              ocr_preprocess_mode;
	int                 ocr_psm = -1;  // -1 means default
	String              ocr_whitelist;
	String              ocr_blacklist;

	void Jsonize(JsonIO& jio);
};

struct AnnLayGroup : Moveable<AnnLayGroup> {
	String         id;
	AnnLayMethod   method = ANNLAY_CLASSIFIER_BOOL;
	Size           crop_size = Size(32, 32);
	Vector<String> classes;
	String         color_mode = "grayscale";
	String         display_name;
	Vector<String> legacy_aliases;
	String         head_role;         // e.g. presence, level, category, offset_x
	String         preprocess;        // e.g. high_luma_bin, color_raw
	String         export_dataset_dir;// optional directory override for crop export

	void Jsonize(JsonIO& jio);
};

struct AnnLay {
	String           description;
	String           format = "annlay";
	String           version = "1";
	int              image_width_ref = 0;
	int              image_height_ref = 0;
	double           bbox_expand = 0.10;
	Vector<AnnLaySlot> slots;
	Vector<AnnLayGroup> groups;

	void Jsonize(JsonIO& jio);

	bool Load(const String& path);
	bool Save(const String& path) const;

	AnnLaySlot* FindSlot(const String& id);
	const AnnLaySlot* FindSlot(const String& id) const;

	void ComputeAnchorsFromAnnprj(const String& annprj_path,
	                              double variable_threshold = 0.05);
};

// Parse optional slot splitter layout. Returns false when slot has no valid JSON;
// in that case callers should use default regions.
bool AnnLayTryGetSubLayoutRegions(const AnnLaySlot& slot,
                                  VectorMap<String, Rectf>& out_regions);

// Resolve named region to pixel rect inside base rect or local size.
// Defaults when region not found:
//   card_region => full
//   rank_region => top half
//   suit_region => bottom half
Rect AnnLayResolveRegionRect(const AnnLaySlot& slot, const Rect& base, const String& region_name);
Rect AnnLayResolveRegionRect(const AnnLaySlot& slot, Size base_size, const String& region_name);

// Rotation from sub_layout_json (if present). Fallback preserves legacy hero defaults.
bool AnnLayTryGetSubLayoutRotationDeg(const AnnLaySlot& slot, double& out_deg);
double AnnLayGetSlotRotationDeg(const AnnLaySlot& slot, const String& slot_id);

END_UPP_NAMESPACE

#endif
