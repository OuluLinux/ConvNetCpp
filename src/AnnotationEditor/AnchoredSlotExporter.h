#ifndef _AnnotationEditor_AnchoredSlotExporter_h_
#define _AnnotationEditor_AnchoredSlotExporter_h_

#include <Draw/Draw.h>
#include <AnnLayCore/AnnLay.h>

NAMESPACE_UPP

struct AnchoredSlotExportResult {
	VectorMap<String, int> samples_per_group; // display_name → total sample count
	int   images_processed = 0;
	String error;
};

enum OffsetStyle {
	OFFSET_STYLE_COMBINED = 0,
	OFFSET_STYLE_SPLIT,
	OFFSET_STYLE_BOTH,
	OFFSET_STYLE_X,
	OFFSET_STYLE_Y,
};

// Exports crops from annprj + annlay to pass1/ or pass2/ subdirs.
// Mirrors train_annlay_classifiers.py logic. Generic — no domain names.
//
// Pass 1 label slots:
//   - Template images copied from templates_dir (0.png, 1.png ... N-1.png)
//     where N = classes.GetCount() - 1  (index 0 = absent class, no template)
//   - Absent-class anchor crops from images with NO bbox annotation for that slot
//
// Pass 1 bool slots:
//   - BBox crop where annotation IS present         → true/
//   - Anchor crop where annotation is absent        → false/
//
// Pass 2 adds, for each verified image (metadata_verified == true):
//   - Label slots: bbox crops labelled from image metadata
//   - Label slots: absent-class anchor crops from verified images where absent
//   - Bool slots : anchor crops labelled from annotation presence
//
class AnchoredSlotExporter {
public:
	// pass_index: 1 or 2
	AnchoredSlotExportResult Export(const AnnLay& lay,
	                                const String& annprj_path,
	                                const String& images_dir,
	                                const String& templates_dir,
	                                const String& crops_root,
	                                int pass_index,
	                                OffsetStyle offset_style = OFFSET_STYLE_COMBINED);

private:
	// Crop a slot anchor from img, expanded by bbox_expand, resized to crop_size.
	Image CropAnchor(const Image& img, const AnnLayAnchor& anchor,
	                 double bbox_expand, Size crop_size) const;

	// Like CropAnchor, but rotates the region by angle degrees before final crop/scale.
	Image CropAndRotate(const Image& img, const AnnLayAnchor& anchor,
	                    double bbox_expand, Size crop_size, double angle) const;

	// Crop a polygon bbox from img, expanded by bbox_expand, resized to crop_size.
	Image CropPolygon(const Image& img, const Value& polygon_points,
	                  int img_w, int img_h,
	                  double bbox_expand, Size crop_size) const;

	// Like CropPolygon, but expands the crop region to match crop_size aspect ratio.
	// Region is only expanded (never shrunk) before resize.
	Image CropPolygonAspect(const Image& img, const Value& polygon_points,
	                        int img_w, int img_h,
	                        double bbox_expand, Size crop_size, double angle = 0) const;

	// Save image to path, creating dirs as needed. Returns true on success.
	bool SaveCrop(const Image& img, const String& path, bool grayscale = false, bool equalize = true) const;

	// Check if an annotation has a non-empty polygon (bbox present).
	bool IsAnnotationPresent(const Value& ann) const;

	// Check if image metadata has metadata_verified == true.
	bool IsVerified(const Value& img_rec) const;

	// Get metadata value by key from image record.
	String GetMetaValue(const Value& img_rec, const String& key) const;

	// Returns class name for element slots from compact metadata strings.
	// slot_id examples: board_card_1, hero_card_2
	// returns empty if missing/invalid/out-of-range.
	String GetCardMetaClass(const Value& img_rec, const String& slot_id) const;

	// Load annprj image from disk.
	Image LoadImage(const Value& img_rec, const String& images_dir) const;

	// Export 3-head crops for composite element slots.
	void SaveCompositeCardCrops(const AnnLaySlot& slot,
	                            const Image& full_card,
	                            const String& presence_class,
	                            const String& rank_class,
	                            const String& suit_class,
	                            const String& presence_group,
	                            const String& rank_group,
	                            const String& suit_group,
	                            const String& sub_layout_json_override,
	                            const String& slot_id,
	                            const String& stem,
	                            const String& pass_dir,
	                            AnchoredSlotExportResult& result) const;

	void SaveRankJitterCrop(const Image& img,
	                        const AnnLaySlot& slot,
	                        const String& sub_layout_json_override,
	                        double ann_cx, double ann_cy,
	                        double anchor_w, double anchor_h,
	                        double angle,
	                        double bbox_expand,
	                        Size crop_size,
	                        int dx, int dy,
	                        const String& rank_class,
	                        const String& slot_id,
	                        const String& stem,
	                        const String& pass_dir,
	                        AnchoredSlotExportResult& result) const;

	// Export zoom crops for bbox scaling head.
	void SaveZoomCrops(const Image& img,
	                   const AnnLaySlot& slot,
	                   const String& sub_layout_json_override,
	                   double ann_cx, double ann_cy,
	                   double anchor_w, double anchor_h,
	                   double angle,
	                   double bbox_expand,
	                   Size crop_size,
	                   double zoom,
	                   const String& slot_id,
	                   const String& stem,
	                   const String& pass_dir,
	                   AnchoredSlotExportResult& result) const;

	// Export offset crops for bbox alignment heads.
	void SaveOffsetCrops(const Image& anchor_crop,
	                     int dx, int dy,
	                     const String& slot_id,
	                     const String& stem,
	                     const String& pass_dir,
	                     AnchoredSlotExportResult& result,
	                     OffsetStyle style) const;

	// Export templates for one label slot group (pass 1 label slots).
	// Returns number of template images written.
	int ExportTemplates(const AnnLay& lay,
	                    const Vector<String>& slot_ids,
	                    const String& group_disp,
	                    const String& templates_dir,
	                    const String& out_dir) const;
};

END_UPP_NAMESPACE

#endif
