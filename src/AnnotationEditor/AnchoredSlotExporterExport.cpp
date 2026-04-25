#include "AnchoredSlotExporter.h"
#include <AnnLayCore/AnchoredSlotClassifier.h>
#include <AnnLayCore/GroupRegistry.h>
#include "AnnotationEditorCommon.h"

#include <Draw/Draw.h>
#include <plugin/jpg/jpg.h>
#include <plugin/png/png.h>
#include <Painter/Painter.h>

NAMESPACE_UPP

namespace {

// ClampI, GetSubLayoutJsonOverride, ResolveRegionRectWithOverride,
// ResolveSlotRotationWithOverride are defined in AnchoredSlotExporter.cpp.
// In non-blitz builds they are redeclared locally; in blitz builds they
// share the same TU so the definitions below are guarded to avoid duplicates.
#ifndef BLITZ_INDEX__
int ClampI(int v, int lo, int hi) {
	return v < lo ? lo : (v > hi ? hi : v);
}

String GetSubLayoutJsonOverride(const Value& ann) {
	if(IsNull(ann))
		return String();
	String j = ann["metadata"]["sub_layout_json"];
	if(j.IsEmpty())
		j = ann["metadata"]["annlay_sub_layout_json"];
	return TrimBoth(j);
}

Rect ResolveRegionRectWithOverride(const AnnLaySlot& slot,
                                   Size base_size,
                                   const String& region_name,
                                   const String& sub_layout_json_override) {
	if(sub_layout_json_override.IsEmpty())
		return AnnLayResolveRegionRect(slot, base_size, region_name);
	AnnLaySlot tmp;
	tmp.sub_layout_json = sub_layout_json_override;
	return AnnLayResolveRegionRect(tmp, base_size, region_name);
}

double ResolveSlotRotationWithOverride(const AnnLaySlot& slot,
                                       const String& slot_id,
                                       const String& sub_layout_json_override) {
	if(!sub_layout_json_override.IsEmpty()) {
		AnnLaySlot tmp;
		tmp.sub_layout_json = sub_layout_json_override;
		double rot = 0;
		if(AnnLayTryGetSubLayoutRotationDeg(tmp, rot))
			return rot;
	}
	return AnnLayGetSlotRotationDeg(slot, slot_id);
}
#endif // BLITZ_INDEX__

bool ContainsString(const Vector<String>& values, const String& value) {
	for(const String& v : values)
		if(v == value)
			return true;
	return false;
}

}

bool AnchoredSlotExporter::IsAnnotationPresent(const Value& ann) const {
	Value polygons = ann["polygons"];
	if(!IsValueArray(polygons) || polygons.GetCount() == 0) return false;
	if(!IsValueArray(polygons[0]) || polygons[0].GetCount() == 0) return false;
	return true;
}

bool AnchoredSlotExporter::IsVerified(const Value& img_rec) const {
	Value meta_keys = img_rec["image_meta_keys"];
	Value meta_vals = img_rec["image_meta_values"];
	if(!IsValueArray(meta_keys) || !IsValueArray(meta_vals)) return false;
	for(int i = 0; i < meta_keys.GetCount(); i++) {
		if((String)meta_keys[i] == "metadata_verified") {
			String v = meta_vals[i].ToString();
			return v == "true" || v == "1";
		}
	}
	return false;
}

String AnchoredSlotExporter::GetMetaValue(const Value& img_rec, const String& key) const {
	Value meta_keys = img_rec["image_meta_keys"];
	Value meta_vals = img_rec["image_meta_values"];
	if(!IsValueArray(meta_keys) || !IsValueArray(meta_vals)) return String();
	for(int i = 0; i < meta_keys.GetCount(); i++) {
		if((String)meta_keys[i] == key)
			return meta_vals[i].ToString();
	}
	return String();
}

String AnchoredSlotExporter::GetCardMetaClass(const Value& img_rec, const String& slot_id) const {
	int pos = slot_id.GetCount() - 1;
	if(pos < 0 || !IsDigit((byte)slot_id[pos]))
		return String();
	while(pos >= 0 && IsDigit((byte)slot_id[pos]))
		pos--;
	if(pos < 0 || slot_id[pos] != '_')
		return String();
	String stem = slot_id.Left(pos);
	int n = StrInt(slot_id.Mid(pos + 1));
	int idx = n - 1;
	if(stem.IsEmpty() || idx < 0)
		return String();
	String compound_key = stem + "s";
	String val = TrimBoth(GetMetaValue(img_rec, compound_key));
	if(val.IsEmpty())
		val = TrimBoth(GetMetaValue(img_rec, "board_cards"));
	if(val.IsEmpty())
		val = TrimBoth(GetMetaValue(img_rec, "hero_cards"));
	if(val.IsEmpty())
		return String();

	Vector<String> cards;
	for(int i = 0; i + 1 < val.GetCount(); i++) {
		byte level = (byte)ToUpper((byte)val[i]);
		byte category = (byte)ToLower((byte)val[i + 1]);
		bool rank_ok = level == '2' || level == '3' || level == '4' || level == '5' ||
		               level == '6' || level == '7' || level == '8' || level == '9' ||
		               level == 'T' || level == 'J' || level == 'Q' || level == 'K' || level == 'A';
		bool suit_ok = category == 's' || category == 'h' || category == 'c' || category == 'd';
		if(rank_ok && suit_ok) {
			String cls;
			cls.Cat((char)level);
			cls.Cat((char)category);
			cards.Add(cls);
			i++;
			continue;
		}

		// Accept reversed two-char token ("h2" -> "2h") to tolerate
		// occasional metadata formatting inconsistencies.
		byte suit_first = (byte)ToLower((byte)val[i]);
		byte rank_second = (byte)ToUpper((byte)val[i + 1]);
		bool suit_first_ok = suit_first == 's' || suit_first == 'h' || suit_first == 'c' || suit_first == 'd';
		bool rank_second_ok = rank_second == '2' || rank_second == '3' || rank_second == '4' || rank_second == '5' ||
		                      rank_second == '6' || rank_second == '7' || rank_second == '8' || rank_second == '9' ||
		                      rank_second == 'T' || rank_second == 'J' || rank_second == 'Q' || rank_second == 'K' || rank_second == 'A';
		if(suit_first_ok && rank_second_ok) {
			String cls;
			cls.Cat((char)rank_second);
			cls.Cat((char)suit_first);
			cards.Add(cls);
			i++;
		}
	}
	if(idx >= cards.GetCount())
		return String();
	return cards[idx];
}

Image AnchoredSlotExporter::LoadImage(const Value& img_rec, const String& images_dir) const {
	auto TryLoad = [](const String& p) -> Image {
		if(!p.IsEmpty() && FileExists(p)) {
			Image img = LoadImageByExtension(p);
			if(!img.IsEmpty()) return img;
		}
		return Image();
	};

	String file_name = img_rec["file_name"];
	if(!file_name.IsEmpty()) {
		Image img = TryLoad(AppendFileName(images_dir, file_name));
		if(!img.IsEmpty()) return img;
		if(IsFullPath(file_name)) { img = TryLoad(file_name); if(!img.IsEmpty()) return img; }
	}
	String file_path = img_rec["file_path"];
	if(!file_path.IsEmpty()) {
		String base = GetFileName(file_path);
		Image img = TryLoad(AppendFileName(images_dir, base));
		if(!img.IsEmpty()) return img;
		img = TryLoad(file_path);
		if(!img.IsEmpty()) return img;
	}
	return Image();
}

void AnchoredSlotExporter::SaveCompositeCardCrops(const AnnLaySlot& slot,
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
                                                 AnchoredSlotExportResult& result) const
{
	if(full_card.IsEmpty()) return;

	Rect card_r = ResolveRegionRectWithOverride(slot, full_card.GetSize(), "card_region", sub_layout_json_override);
	Rect rank_r = ResolveRegionRectWithOverride(slot, full_card.GetSize(), "rank_region", sub_layout_json_override);
	Rect suit_r = ResolveRegionRectWithOverride(slot, full_card.GetSize(), "suit_region", sub_layout_json_override);
	Image card_crop = Crop(full_card, card_r);
	Image rank_crop = Crop(full_card, rank_r);
	Image suit_crop = Crop(full_card, suit_r);
	if(card_crop.IsEmpty()) card_crop = full_card;
	if(rank_crop.IsEmpty()) rank_crop = full_card;
	if(suit_crop.IsEmpty()) suit_crop = full_card;
	if(card_crop.GetSize() != full_card.GetSize())
		card_crop = Rescale(card_crop, full_card.GetSize());
	if(rank_crop.GetWidth() != 40 || rank_crop.GetHeight() != 40)
		rank_crop = Rescale(rank_crop, 40, 40);
	if(suit_crop.GetWidth() != 40 || suit_crop.GetHeight() != 40)
		suit_crop = Rescale(suit_crop, 40, 40);
	Image card_bin = HighLuminanceThresholdBinarization(card_crop);
	Image rank_bin = HighLuminanceThresholdBinarization(rank_crop);

	// Presence (grayscale, full crop)
	if(presence_class == "true" || presence_class == "false") {
		String group = TrimBoth(presence_group);
		if(group.IsEmpty())
			group = "presence";
		String out_dir = AppendFileName(pass_dir, group);
		String fname = Format("%s_%s.png", slot_id, stem);
		String true_path = AppendFileName(AppendFileName(out_dir, "true"), fname);
		String false_path = AppendFileName(AppendFileName(out_dir, "false"), fname);

		// Keep card_visibility_gate labels mutually exclusive when re-exporting.
		if(presence_class == "true" && FileExists(false_path))
			DeleteFile(false_path);
		if(presence_class == "false" && FileExists(true_path))
			DeleteFile(true_path);

		String path = AppendFileName(AppendFileName(out_dir, presence_class), fname);
		// card_visibility_gate uses high-luminance binary input; no linear contrast stretching.
		if(SaveCrop(card_bin, path, true, false)) {
			result.samples_per_group.GetAdd(group, 0)++;
		}
	}

	// Level (grayscale)
	if(!rank_class.IsEmpty()) {
		String group = TrimBoth(rank_group);
		if(group.IsEmpty())
			group = "level";
		String out_dir = AppendFileName(pass_dir, group);
		String fname = Format("%s_%s.png", slot_id, stem);
		String path = AppendFileName(AppendFileName(out_dir, rank_class), fname);
		// Level head uses high-luminance binary input; no linear contrast stretching.
		if(SaveCrop(rank_bin, path, true, false)) {
			result.samples_per_group.GetAdd(group, 0)++;
		}
	}

	// Category (color)
	if(!suit_class.IsEmpty()) {
		String group = TrimBoth(suit_group);
		if(group.IsEmpty())
			group = "category";
		String out_dir = AppendFileName(pass_dir, group);
		String fname = Format("%s_%s.png", slot_id, stem);
		String path = AppendFileName(AppendFileName(out_dir, suit_class), fname);
		if(SaveCrop(suit_crop, path, false, false)) { // Bypass for category
			result.samples_per_group.GetAdd(group, 0)++;
		}
	}
}

void AnchoredSlotExporter::SaveRankJitterCrop(const Image& img,
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
                                             AnchoredSlotExportResult& result) const
{
	if(img.IsEmpty() || rank_class.IsEmpty()) return;

	int img_w = img.GetWidth();
	int img_h = img.GetHeight();

	// Add random scale jitter for level robustness (0.8 - 1.2)
	double zoom = 0.8 + (double)Random(41) / 100.0;

	AnnLayAnchor jitter_anchor;
	jitter_anchor.cx = (ann_cx - dx) / img_w;
	jitter_anchor.cy = (ann_cy - dy) / img_h;
	jitter_anchor.w = anchor_w / zoom;
	jitter_anchor.h = anchor_h / zoom;

	Image full_card = CropAndRotate(img, jitter_anchor, bbox_expand, crop_size, angle);
	if(full_card.IsEmpty()) return;
	Rect rank_r = ResolveRegionRectWithOverride(slot, full_card.GetSize(), "rank_region", sub_layout_json_override);
	Image rank_crop = Crop(full_card, rank_r);
	if(rank_crop.IsEmpty()) return;
	if(rank_crop.GetWidth() != 40 || rank_crop.GetHeight() != 40)
		rank_crop = Rescale(rank_crop, 40, 40);

	String group = "card_rank";
	String out_dir = AppendFileName(pass_dir, group);
	String fname = Format("%s_%s_j%d_%d_z%.2f.png", slot_id, stem, dx, dy, zoom);
	String path = AppendFileName(AppendFileName(out_dir, rank_class), fname);

	if(SaveCrop(rank_crop, path, true)) {
		result.samples_per_group.GetAdd(group, 0)++;
	}
}

void AnchoredSlotExporter::SaveZoomCrops(const Image& img,
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
                                         AnchoredSlotExportResult& result) const
{
	if(img.IsEmpty()) return;

	int img_w = img.GetWidth();
	int img_h = img.GetHeight();

	AnnLayAnchor suit_anchor;
	suit_anchor.cx = ann_cx / img_w;
	suit_anchor.cy = ann_cy / img_h;
	suit_anchor.w = anchor_w / zoom;
	suit_anchor.h = anchor_h / zoom;

	Image full_card = CropAndRotate(img, suit_anchor, bbox_expand, crop_size, angle);
	if(full_card.IsEmpty()) return;
	Rect suit_r = ResolveRegionRectWithOverride(slot, full_card.GetSize(), "suit_region", sub_layout_json_override);
	Image zoom_suit_crop = Crop(full_card, suit_r);
	if(!zoom_suit_crop.IsEmpty() && (zoom_suit_crop.GetWidth() != 40 || zoom_suit_crop.GetHeight() != 40))
		zoom_suit_crop = Rescale(zoom_suit_crop, 40, 40);
	if(zoom_suit_crop.IsEmpty()) return;

	String group = "card_zoom";
	String out_dir = AppendFileName(pass_dir, group);
	String cls = Format("z%.2f", zoom);
	String fname = Format("%s_%s.png", slot_id, stem);
	String path = AppendFileName(AppendFileName(out_dir, cls), fname);

	if(SaveCrop(zoom_suit_crop, path, true)) {
		result.samples_per_group.GetAdd(group, 0)++;
	}
}

void AnchoredSlotExporter::SaveOffsetCrops(const Image& anchor_crop,
                                           int dx, int dy,
                                           const String& slot_id,
                                           const String& stem,
                                           const String& pass_dir,
                                           AnchoredSlotExportResult& result,
                                           OffsetStyle style) const
{
	if(anchor_crop.IsEmpty()) return;

	auto SaveHead = [&](const String& group, const String& class_name) {
		String out_dir = AppendFileName(pass_dir, group);
		String fname = Format("%s_%s.png", slot_id, stem);
		String path = AppendFileName(AppendFileName(out_dir, class_name), fname);
		// Offset heads use grayscale crops for robustness against table/theme color drift.
		if(SaveCrop(anchor_crop, path, true)) {
			result.samples_per_group.GetAdd(group, 0)++;
		}
	};

	if(style == OFFSET_STYLE_SPLIT || style == OFFSET_STYLE_BOTH || style == OFFSET_STYLE_X) {
		SaveHead("card_offset_x", Format("%+d", dx));
	}
	if(style == OFFSET_STYLE_SPLIT || style == OFFSET_STYLE_BOTH || style == OFFSET_STYLE_Y) {
		SaveHead("card_offset_y", Format("%+d", dy));
	}
	if(style == OFFSET_STYLE_COMBINED || style == OFFSET_STYLE_BOTH) {
		SaveHead("card_offset",   Format("dx=%+d,dy=%+d", dx, dy));
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Template export (label slots, pass 1)
// ─────────────────────────────────────────────────────────────────────────────

int AnchoredSlotExporter::ExportTemplates(const AnnLay& lay,
                                           const Vector<String>& slot_ids,
                                           const String& group_disp,
                                           const String& templates_dir,
                                           const String& out_dir) const {
	if(templates_dir.IsEmpty() || slot_ids.IsEmpty()) return 0;

	// Find a reference slot with classes
	const AnnLaySlot* ref = nullptr;
	for(const String& id : slot_ids) {
		const AnnLaySlot* s = lay.FindSlot(id);
		if(s && s->classes.GetCount() > 1) { ref = s; break; }
	}
	if(!ref) return 0;

	// classes[0] = absent class (no template), classes[1..N] = present classes
	int n_classes = ref->classes.GetCount();
	int written = 0;
	String tpl_dir = AppendFileName(templates_dir, group_disp);
	if(!DirectoryExists(tpl_dir))
		tpl_dir = templates_dir;

	static const int kAngles[] = {-4, -3, -2, -1, 0, 1, 2, 3, 4};
	bool has_crop = ref->template_crop_w > 0 && ref->template_crop_h > 0;

	for(int k = 0; k < n_classes - 1; k++) {
		String class_name = ref->classes[k + 1];
		String tpl_path = AppendFileName(tpl_dir, Format("%d.png", k));
		if(!FileExists(tpl_path)) {
			// Also try .jpg
			tpl_path = AppendFileName(tpl_dir, Format("%d.jpg", k));
			if(!FileExists(tpl_path)) continue;
		}

		Image tpl = LoadImageByExtension(tpl_path);
		if(tpl.IsEmpty()) continue;

		for(int ai = 0; ai < (int)(sizeof(kAngles) / sizeof(kAngles[0])); ai++) {
			int angle = kAngles[ai];
			Image rotated = RotateBilinear(tpl, (double)angle);
			if(rotated.IsEmpty())
				continue;
			int tw = rotated.GetWidth();
			int th = rotated.GetHeight();
			if(tw <= 0 || th <= 0)
				continue;
			Image patch;
			if(has_crop) {
				int x1 = ClampI(ref->template_crop_x, 0, max(0, tw - 1));
				int y1 = ClampI(ref->template_crop_y, 0, max(0, th - 1));
				int w = max(1, ref->template_crop_w);
				int h = max(1, ref->template_crop_h);
				int x2 = min(tw, x1 + w);
				int y2 = min(th, y1 + h);
				if(x2 > x1 && y2 > y1) {
					#if 0
					Rect r(x1, y1, x2, y2);
					ImagePainter id(r.GetSize());
					id.DrawImage(0,0,rotated,r);
					patch = id;
					#else
					patch = Crop(rotated, Rect(x1, y1, x2, y2));
					#endif
				}
			}
			else {
				patch = rotated;
			}
			if(patch.IsEmpty())
				continue;

			String out_path = AppendFileName(
			    AppendFileName(out_dir, class_name),
			    Format("tpl_%04d_r%+d.png", k, angle));
			// cv_template_match/label_a uses high-luminance binary template patches.
			bool is_label_a_group = ToLower(group_disp).Find("label_a") >= 0;
			Image out_patch = patch;
			if(is_label_a_group)
				out_patch = HighLuminanceThresholdBinarization(out_patch);
			if(SaveCrop(out_patch, out_path, ref->color_mode != "color", false))
				written++;
		}
	}
	return written;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main Export
// ─────────────────────────────────────────────────────────────────────────────

AnchoredSlotExportResult AnchoredSlotExporter::Export(
    const AnnLay& lay,
    const String& annprj_path,
    const String& images_dir,
    const String& templates_dir,
    const String& crops_root,
    int pass_index,
    OffsetStyle offset_style)
{
	AnchoredSlotExportResult result;

	if(annprj_path.IsEmpty() || !FileExists(annprj_path)) {
		result.error = "annprj file not found: " + annprj_path;
		return result;
	}
	if(crops_root.IsEmpty()) {
		result.error = "crops_root is empty";
		return result;
	}

	String pass_dir = AppendFileName(crops_root, pass_index == 1 ? "pass1" : "pass2");
	if(DirectoryExists(pass_dir))
		DeleteFolderDeep(pass_dir);
	RealizeDirectory(pass_dir);

	// Load annprj
	String prj_json = LoadFile(annprj_path);
	if(prj_json.IsEmpty()) { result.error = "Failed to read annprj"; return result; }
	Value root = ParseJSON(prj_json);
	if(IsNull(root) || !IsValueMap(root)) { result.error = "Invalid annprj JSON"; return result; }
	Value datasets = root["datasets"];
	if(!IsValueArray(datasets)) { result.error = "No datasets in annprj"; return result; }

	// Get slot groups
	VectorMap<String, Vector<String>> groups = AnchoredSlotClassifier::GetSlotGroups(lay);
	GroupRegistry group_registry;
	group_registry.Build(lay);
	bool has_rank_role_group = false;
	Vector<String> composite_cleanup_groups;
	for(int gi = 0; gi < groups.GetCount(); gi++) {
		String gkey = groups.GetKey(gi);
		String role = group_registry.HeadRole(gkey);
		if(role == "level") {
			has_rank_role_group = true;
		}
		if(role == "presence" || role == "level" || role == "category" ||
		   role == "zoom" || role == "offset" || role == "offset_x" || role == "offset_y") {
			String out = group_registry.ExportDatasetDir(gkey);
			if(out.IsEmpty())
				out = AnchoredSlotClassifier::GetSlotGroupDisplayName(lay, gkey);
			if(!out.IsEmpty() && !ContainsString(composite_cleanup_groups, out))
				composite_cleanup_groups.Add(out);
		}
	}
	// Legacy element aux heads (zoom/offset jitter datasets) are retired from this flow.
	const bool export_card_aux_datasets = false;

	auto CleanupDatasetDirectory = [&](const String& out_dir) {
		auto CleanupRecursive = [&](const String& dir, auto& self) -> void {
			FindFile ff;
			if(ff.Search(AppendFileName(dir, "*"))) {
				do {
					if(ff.IsFile()) {
						String name = ff.GetName();
						String ext = ToLower(GetFileExt(name));
						if((ext == ".png" || ext == ".jpg" || ext == ".jpeg") && !name.EndsWith("_fp.png"))
							DeleteFile(ff.GetPath());
					}
					else if(ff.IsFolder()) {
						String name = ff.GetName();
						if(name != "." && name != "..")
							self(ff.GetPath(), self);
					}
				} while(ff.Next());
			}
		};
		CleanupRecursive(out_dir, CleanupRecursive);
	};

	// ──────────────────────────────────────────────────────────────────────
	// Phase 6: Clean stale composite element datasets before export
	// ──────────────────────────────────────────────────────────────────────
	for(const String& g : composite_cleanup_groups)
		CleanupDatasetDirectory(AppendFileName(pass_dir, g));

	// Collect all images
	Vector<Value> all_images;
	for(int di = 0; di < datasets.GetCount(); di++) {
		Value imgs = datasets[di]["images"];
		if(IsValueArray(imgs))
			for(int ii = 0; ii < imgs.GetCount(); ii++)
				all_images.Add(imgs[ii]);
	}

	// ──────────────────────────────────────────────────────────────────────
	// Pass 1: templates + absent-class anchor crops (label)
	//         annotation-based bool crops
	// Pass 2: same as pass1 PLUS verified image crops
	// ──────────────────────────────────────────────────────────────────────

	// Step A: For label slots, export templates (pass 1 + pass 2 both include them)
	for(int gi = 0; gi < groups.GetCount(); gi++) {
		const String& gkey = groups.GetKey(gi);
		const Vector<String>& slot_ids = groups[gi];
		if(slot_ids.IsEmpty()) continue;

		const AnnLaySlot* ref = nullptr;
		for(const String& id : slot_ids) {
			const AnnLaySlot* s = lay.FindSlot(id);
			if(s) { ref = s; break; }
		}
		if(!ref) continue;

		if(ref->composite_type == ANNLAY_COMPOSITE_NONE && ref->method == ANNLAY_CLASSIFIER_LABEL) {
			String disp    = AnchoredSlotClassifier::GetSlotGroupDisplayName(lay, gkey);
			String out_dir = AppendFileName(pass_dir, disp);
			int n = ExportTemplates(lay, slot_ids, disp, templates_dir, out_dir);
			if(n > 0) {
				int& cnt = result.samples_per_group.GetAdd(disp, 0);
				cnt += n;
			}
		}
	}

	// Step B: Walk all images; for each image/slot, generate anchor/bbox crops
	VectorMap<String, int> bool_true_saved;
	VectorMap<String, int> bool_false_saved;
	Index<String> bool_groups_cleaned;

	VectorMap<int, int> rank_jitter_counts; // Combined (dx, dy) encoded as (dx+15)*100 + (dy+15)
	VectorMap<double, int> zoom_counts;
	VectorMap<int, VectorMap<String, int>> stats_x;

	VectorMap<int, VectorMap<String, int>> stats_y;

	auto AllowBoolFalseSample = [&](const String& group_name) {
		int t = bool_true_saved.Get(group_name, 0);
		int f = bool_false_saved.Get(group_name, 0);
		return f < 2 * t;
	};

	for(int ii = 0; ii < all_images.GetCount(); ii++) {
		Value img_rec = all_images[ii];
		bool verified = IsVerified(img_rec);

		// For pass 1 we only use ALL images for anchor crops (absent class for label,
		// both classes for bool). We do NOT use frame bbox crops for label slots in pass 1.
		// For pass 2 we also use verified images for bbox crops.

		Image img;  // loaded lazily

		int img_w = img_rec["width"];
		int img_h = img_rec["height"];

		Value annotations = img_rec["annotations"];

		// Build a fast lookup: slot_id → annotation (first present)
		VectorMap<String, Value> ann_by_slot;
		if(IsValueArray(annotations)) {
			for(int ai = 0; ai < annotations.GetCount(); ai++) {
				Value ann = annotations[ai];
				String sid = ann["metadata"]["mlui_slot_id"];
				if(sid.IsEmpty()) continue;
				if(ann_by_slot.Find(sid) < 0)
					ann_by_slot.Add(sid, ann);
			}
		}

		// Get image stem for crop filenames
		String file_name = img_rec["file_name"];
		if(file_name.IsEmpty()) file_name = img_rec["file_path"];
		String stem = GetFileTitle(file_name);
		if(stem.IsEmpty()) stem = Format("%04d", ii);

		for(int gi = 0; gi < groups.GetCount(); gi++) {
			const String& gkey    = groups.GetKey(gi);
			const Vector<String>& slot_ids = groups[gi];
			if(slot_ids.IsEmpty()) continue;

			const AnnLaySlot* ref_slot = nullptr;
			for(const String& id : slot_ids) {
				const AnnLaySlot* s = lay.FindSlot(id);
				if(s) { ref_slot = s; break; }
			}
			if(!ref_slot) continue;

			String disp    = AnchoredSlotClassifier::GetSlotGroupDisplayName(lay, gkey);
			String out_dir = AppendFileName(pass_dir, disp);
			bool force_high_luma = (group_registry.Preprocess(gkey) == "high_luma_bin");

			if(ref_slot->method == ANNLAY_CV_TEMPLATE_MATCH &&
			   ref_slot->composite_type != ANNLAY_COMPOSITE_ELEMENT)
				continue; // non-element cv_template_match slots don't use NN crop exports

			if(ref_slot->composite_type == ANNLAY_COMPOSITE_ELEMENT) {
				String role = group_registry.HeadRole(gkey);
				bool export_visibility = (role == "presence");
				bool export_suit = (role == "category");
				// Fallback: if slot grouping policy omits element#level, export level through
				// existing composite-element passes so card_rank dataset still gets generated.
				bool export_rank = (role == "level") || (!has_rank_role_group && (export_visibility || export_suit));
				if(!export_visibility && !export_suit && !export_rank)
					continue;

				for(const String& slot_id : slot_ids) {
					const AnnLaySlot* slot = lay.FindSlot(slot_id);
					if(!slot) continue;

					int ann_idx_slot = -1;
					for(int k = 0; k < ann_by_slot.GetCount(); k++) {
						if(AnchoredSlotClassifier::MatchSlotForBool(slot_id, ann_by_slot.GetKey(k))) {
							ann_idx_slot = k;
							break;
						}
					}
					bool has_ann = (ann_idx_slot >= 0) &&
					               IsAnnotationPresent(ann_by_slot[ann_idx_slot]);
					Value ann = has_ann ? ann_by_slot[ann_idx_slot] : Value();
					String sub_layout_override = GetSubLayoutJsonOverride(ann);

					// Get ground truth element class (e.g. "As")
					String card_meta = GetCardMetaClass(img_rec, slot_id);
					
					// If pass 2, only export verified images
					if(pass_index == 2 && !verified) continue;

					if(img.IsEmpty()) img = LoadImage(img_rec, images_dir);
					if(img.IsEmpty()) break;

					double angle = ResolveSlotRotationWithOverride(*slot, slot_id, sub_layout_override);

					Image card_crop;
					if(has_ann) {
						Value polygons = ann["polygons"];
						card_crop = CropPolygonAspect(img, polygons[0], img_w, img_h,
						                              lay.bbox_expand, slot->crop_size, angle);
					}
					else if(!slot->anchor.IsEmpty()) {
						card_crop = CropAndRotate(img, slot->anchor, lay.bbox_expand, slot->crop_size, angle);
					}

					if(!card_crop.IsEmpty()) {
						// Presence is driven by bbox/object annotation existence, not
						// metadata element-string formatting.
						String presence = has_ann ? "true" : "false";
						String level = card_meta.GetCount() >= 1 ? card_meta.Left(1) : "";
						String category = card_meta.GetCount() >= 2 ? card_meta.Mid(1, 1) : "";

						if(export_visibility) {
							SaveCompositeCardCrops(*slot, card_crop, presence, String(), String(),
							                       group_registry.ExportDatasetDir(gkey), String(), String(),
							                       sub_layout_override,
							                       slot_id, stem, pass_dir, result);
						}
						if(export_suit && has_ann) {
							SaveCompositeCardCrops(*slot, card_crop, String(), String(), category,
							                       String(), String(), group_registry.ExportDatasetDir(gkey),
							                       sub_layout_override,
							                       slot_id, stem, pass_dir, result);
						}
						if(export_rank && has_ann) {
							String rank_group = gkey;
							if(role != "level") {
								for(int ri = 0; ri < groups.GetCount(); ri++) {
									if(group_registry.HeadRole(groups.GetKey(ri)) == "level") {
										rank_group = groups.GetKey(ri);
										break;
									}
								}
							}
							SaveCompositeCardCrops(*slot, card_crop, String(), level, String(),
							                       String(), group_registry.ExportDatasetDir(rank_group), String(),
							                       sub_layout_override,
							                       slot_id, stem, pass_dir, result);
						}

						// Optional legacy aux datasets (bbox alignment/zoom jitter).
						// Keep disabled by default: level base dataset is exported above.
						if(export_card_aux_datasets && export_rank && has_ann && !card_meta.IsEmpty()) {
							Value polygons = ann["polygons"];
							Value pts = IsValueArray(polygons) && polygons.GetCount() > 0 ? polygons[0] : Value();
							
							double ann_cx = 0, ann_cy = 0;
							int npts = 0;
							if(IsValueArray(pts)) {
								for(int pi = 0; pi < pts.GetCount(); pi++) {
									ann_cx += (double)pts[pi]["x"];
									ann_cy += (double)pts[pi]["y"];
									npts++;
								}
							}
							if(npts > 0) {
								ann_cx /= npts;
								ann_cy /= npts;

								double anchor_cx = slot->anchor.cx * img_w;
								double anchor_cy = slot->anchor.cy * img_h;

								int dx = (int)round(ann_cx - anchor_cx);
								int dy = (int)round(ann_cy - anchor_cy);

								dx = ClampI(dx, -12, 12);
								dy = ClampI(dy, -12, 12);

								// Phase 6: Uniform Level Jittering (Combined dx, dy)
								// Target: ~20,000 total. 441 combinations -> ~45 per combination.
								if(!level.IsEmpty()) {
									rank_jitter_counts.GetAdd(1515, 0)++; // (0+15)*100 + (0+15)

									// Generate 10 random variations per element to fill the 2D distribution.
									for(int j = 0; j < 10; j++) {
										int rdx = Random(21) - 10;
										int rdy = Random(21) - 10;
										if(rdx == 0 && rdy == 0) continue;

										int rkey = (rdx + 15) * 100 + (rdy + 15);
										if(rank_jitter_counts.Get(rkey, 0) < 45) {
											SaveRankJitterCrop(img, *slot, sub_layout_override, ann_cx, ann_cy, slot->anchor.w, slot->anchor.h,
											                    angle, lay.bbox_expand, slot->crop_size,
											                    rdx, rdy, level, slot_id, stem, pass_dir, result);
											rank_jitter_counts.GetAdd(rkey, 0)++;
										}
									}
								}

								// Phase 6: Zoom Jittering (0.5 - 1.5)
								static const double kZoomLevels[] = {
									0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.10, 1.20, 1.30, 1.40, 1.50
								};
								if(!category.IsEmpty()) {
									for(double z : kZoomLevels) {
										if(zoom_counts.Get(z, 0) < 1000) {
											SaveZoomCrops(img, *slot, sub_layout_override, ann_cx, ann_cy, slot->anchor.w, slot->anchor.h,
											              angle, lay.bbox_expand, slot->crop_size, z,
											              slot_id, stem, pass_dir, result);
											zoom_counts.GetAdd(z, 0)++;
										}
									}
								}

								// Base anchor crop (what the recognizer sees before local refinement)
								// Phase 6: Jittered offset generation.
								// We generate multiple samples per element by jittering the crop position.
								// Label is the relative dx/dy from the ground truth center.
								
								// Jitter logic: for a target dx, we can vary dy from -10 to 10.
								// We limit to ~300 samples per offset value.
								
								auto JitterSave = [&](int target_dx, int target_dy, const String& group, VectorMap<int, VectorMap<String, int>>& stats, const String& current_suit) {
									if(current_suit.IsEmpty()) return;
									
									bool is_x = group.EndsWith("_x");
									bool is_y = group.EndsWith("_y");
									
									// To get a crop that is 'target_dx' pixels away from center,
									// we need to crop at 'ann_cx - target_dx'.
									AnnLayAnchor jitter_anchor;
									jitter_anchor.cx = (ann_cx - target_dx) / img_w;
									jitter_anchor.cy = (ann_cy - target_dy) / img_h;
									jitter_anchor.w = slot->anchor.w;
									jitter_anchor.h = slot->anchor.h;
									
									// 1. Get the straightened 40x80 element crop
									Image full_card = CropAndRotate(img, jitter_anchor, lay.bbox_expand, slot->crop_size, angle);
									if(!full_card.IsEmpty()) {
										// 2. Extract category region (anchor for offset regression)
										Rect suit_r = ResolveRegionRectWithOverride(*slot, full_card.GetSize(), "suit_region", sub_layout_override);
										Image suit_anchor = Crop(full_card, suit_r);
										if(!suit_anchor.IsEmpty() && (suit_anchor.GetWidth() != 40 || suit_anchor.GetHeight() != 40))
											suit_anchor = Rescale(suit_anchor, 40, 40);
										
										if(!suit_anchor.IsEmpty()) {
											SaveOffsetCrops(suit_anchor, target_dx, target_dy, slot_id, stem + Format("_j%d_%d", target_dx, target_dy), pass_dir, result, is_x ? OFFSET_STYLE_X : OFFSET_STYLE_Y);
											
											if(is_x) stats.GetAdd(target_dx).GetAdd(current_suit, 0)++;
											if(is_y) stats.GetAdd(target_dy).GetAdd(current_suit, 0)++;
										}
									}
								};

								// Phase 6: Diversity-focused jittering.
								// For each element, we generate 1 variation per possible offset value.
								// We use a random secondary axis jitter to avoid learning bias.
								for(int v = -10; v <= 10; v++) {
									int random_secondary = Random(21) - 10;
									JitterSave(v, random_secondary, "card_offset_x", stats_x, category);
									
									random_secondary = Random(21) - 10;
									JitterSave(random_secondary, v, "card_offset_y", stats_y, category);
								}
							}
						}
					}
				}
			}
			else if(ref_slot->method == ANNLAY_CLASSIFIER_BOOL) {
				// Clean stale regular bool crops once per group to avoid retaining
				// outdated artifacts from previous exports/rules.
				if(bool_groups_cleaned.Find(disp) < 0) {
					CleanupDatasetDirectory(out_dir);
					bool_groups_cleaned.Add(disp);
				}

				// Bool: one entry per slot_id in the group
				for(const String& slot_id : slot_ids) {
					const AnnLaySlot* slot = lay.FindSlot(slot_id);
					if(!slot) continue;

					if(pass_index == 2 && !verified) continue;

					int ann_idx_slot = -1;
					for(int k = 0; k < ann_by_slot.GetCount(); k++) {
						if(AnchoredSlotClassifier::MatchSlotForBool(slot_id, ann_by_slot.GetKey(k))) {
							ann_idx_slot = k;
							break;
						}
					}
					bool has_ann = (ann_idx_slot >= 0) &&
					               IsAnnotationPresent(ann_by_slot[ann_idx_slot]);

					if(img.IsEmpty()) img = LoadImage(img_rec, images_dir);
					if(img.IsEmpty()) break;

					Size isz = img.GetSize();

					if(!slot->anchor_candidates.IsEmpty() && has_ann) {
						// Multi-candidate bool: find which candidate matches the annotation,
						// crop it as "true", crop all others as "false".
						Value ann = ann_by_slot[ann_idx_slot];
						Value polygons = ann["polygons"];
						Value pts = IsValueArray(polygons) && polygons.GetCount() > 0 ? polygons[0] : Value();

						// Compute annotation bbox center
						double ann_cx = 0, ann_cy = 0;
						int npts = 0;
						if(IsValueArray(pts)) {
							for(int pi = 0; pi < pts.GetCount(); pi++) {
								ann_cx += (double)pts[pi]["x"];
								ann_cy += (double)pts[pi]["y"];
								npts++;
							}
						}
						if(npts > 0) { ann_cx /= npts; ann_cy /= npts; }

						// Find closest candidate
						int best = 0;
						double best_dist = 1e18;
						for(int ci = 0; ci < slot->anchor_candidates.GetCount(); ci++) {
							const AnnLayAnchor& cand = slot->anchor_candidates[ci];
							double dx = cand.cx * isz.cx - ann_cx;
							double dy = cand.cy * isz.cy - ann_cy;
							double d = dx*dx + dy*dy;
							if(d < best_dist) { best_dist = d; best = ci; }
						}

						// Save positive sample first to update false-cap baseline.
						{
							Image crop = CropAnchor(img, slot->anchor_candidates[best],
							                        lay.bbox_expand, slot->crop_size);
							if(!crop.IsEmpty()) {
								String fname = Format("%s_c%d_%s.png", slot_id, best, stem);
								String path  = AppendFileName(AppendFileName(out_dir, "true"), fname);
								bool gray = (slot->color_mode != "color") || force_high_luma;
								bool existed = FileExists(path);
								if(SaveCrop(crop, path, gray, false)) {
									if(!existed) {
										int& cnt = result.samples_per_group.GetAdd(disp, 0);
										cnt++;
										bool_true_saved.GetAdd(disp, 0)++;
									}
								}
							}
						}

						for(int ci = 0; ci < slot->anchor_candidates.GetCount(); ci++) {
							if(ci == best)
								continue;
							if(!AllowBoolFalseSample(disp))
								continue;
							Image crop = CropAnchor(img, slot->anchor_candidates[ci],
							                        lay.bbox_expand, slot->crop_size);
							if(crop.IsEmpty()) continue;
							String fname = Format("%s_c%d_%s.png", slot_id, ci, stem);
							String path  = AppendFileName(AppendFileName(out_dir, "false"), fname);
							bool gray = (slot->color_mode != "color") || force_high_luma;
							bool existed = FileExists(path);
							if(SaveCrop(crop, path, gray, false)) {
								if(!existed) {
									int& cnt = result.samples_per_group.GetAdd(disp, 0);
									cnt++;
									bool_false_saved.GetAdd(disp, 0)++;
								}
							}
						}
					}
					else {
						// Fixed anchor (or no annotation): single crop, label by annotation presence.
						// For positives, prefer annotation bbox crop so true samples follow the
						// actual object location. For negatives, use average anchor crop.
						const AnnLayAnchor* anchor = !slot->anchor.IsEmpty() ? &slot->anchor
						                           : (!slot->anchor_candidates.IsEmpty() ? &slot->anchor_candidates[0] : nullptr);
						if(!anchor) continue;

						String class_dir = has_ann ? "true" : "false";
						if(class_dir == "false" && !AllowBoolFalseSample(disp))
							continue;

						Image crop;
						if(has_ann && ann_idx_slot >= 0) {
							Value ann = ann_by_slot[ann_idx_slot];
							Value polygons = ann["polygons"];
							Value pts = IsValueArray(polygons) && polygons.GetCount() > 0 ? polygons[0] : Value();

							if(img_w <= 0) img_w = img.GetWidth();
							if(img_h <= 0) img_h = img.GetHeight();

							crop = CropPolygonAspect(img, pts, img_w, img_h,
							                         lay.bbox_expand, slot->crop_size);
							// Safety fallback for malformed polygon data.
							if(crop.IsEmpty())
								crop = CropAnchor(img, *anchor, lay.bbox_expand, slot->crop_size);
						}
						else {
							crop = CropAnchor(img, *anchor, lay.bbox_expand, slot->crop_size);
						}

						if(crop.IsEmpty()) continue;
						String fname = Format("%s_%s.png", slot_id, stem);
						String path  = AppendFileName(AppendFileName(out_dir, class_dir), fname);
						bool gray = (slot->color_mode != "color") || force_high_luma;
						bool existed = FileExists(path);
						if(SaveCrop(crop, path, gray, false)) {
							if(!existed) {
								int& cnt = result.samples_per_group.GetAdd(disp, 0);
								cnt++;
								if(class_dir == "true")
									bool_true_saved.GetAdd(disp, 0)++;
								else
									bool_false_saved.GetAdd(disp, 0)++;
							}
						}
					}
				}
			}
			else if(ref_slot->method == ANNLAY_CLASSIFIER_LABEL) {
				// Label: one model shared across all slot_ids in group
				// Absent class: anchor crop from slot with NO bbox annotation
				// Present class: bbox crop from annotation (pass 2 only, verified)

				for(const String& slot_id : slot_ids) {
					const AnnLaySlot* slot = lay.FindSlot(slot_id);
					if(!slot) continue;

					int ann_idx_slot = -1;
					for(int k = 0; k < ann_by_slot.GetCount(); k++) {
						if(AnchoredSlotClassifier::MatchSlotForBool(slot_id, ann_by_slot.GetKey(k))) {
							ann_idx_slot = k;
							break;
						}
					}
					bool has_ann = (ann_idx_slot >= 0) &&
					               IsAnnotationPresent(ann_by_slot[ann_idx_slot]);

					if(!has_ann) {
						// Absent class anchor crop — include in both pass 1 and pass 2
						if(slot->anchor.IsEmpty()) continue;
						if(img.IsEmpty()) img = LoadImage(img_rec, images_dir);
						if(img.IsEmpty()) break;

						Image crop;
						bool use_tpl_crop = slot->template_w > 0 && slot->template_h > 0 &&
						                    slot->template_crop_w > 0 && slot->template_crop_h > 0;
						if(use_tpl_crop) {
							// Crop anchor → rescale to template dims → sub-crop pip region
							Image element = CropAnchor(img, slot->anchor, lay.bbox_expand,
							                        Size(slot->template_w, slot->template_h));
							int x1 = ClampI(slot->template_crop_x, 0, max(0, slot->template_w - 1));
							int y1 = ClampI(slot->template_crop_y, 0, max(0, slot->template_h - 1));
							int x2 = min(slot->template_w, x1 + slot->template_crop_w);
							int y2 = min(slot->template_h, y1 + slot->template_crop_h);
							if(!element.IsEmpty() && x2 > x1 && y2 > y1)
								crop = Crop(element, Rect(x1, y1, x2, y2));
						}
						else {
							crop = CropAnchor(img, slot->anchor, lay.bbox_expand, slot->crop_size);
						}
						if(crop.IsEmpty()) continue;

						// absent class name = classes[0]
						String absent_class = slot->classes.GetCount() > 0
						                    ? slot->classes[0] : "absent";
						bool gray = slot->color_mode != "color";
						String fname = Format("%s_%s_absent.png", slot_id, stem);
						String path  = AppendFileName(AppendFileName(out_dir, absent_class), fname);
						if(SaveCrop(crop, path, gray)) {
							int& cnt = result.samples_per_group.GetAdd(disp, 0);
							cnt++;
						}
					}

					// Pass 1: use image metadata element strings as labels for anchor crops.
					if(pass_index == 1) {
						String class_name = GetCardMetaClass(img_rec, slot_id);
						if(!class_name.IsEmpty()) {
							if(FindIndex(slot->classes, class_name) < 0)
								continue;
							if(img.IsEmpty()) img = LoadImage(img_rec, images_dir);
							if(img.IsEmpty()) break;
							if(img_w <= 0) img_w = img.GetWidth();
							if(img_h <= 0) img_h = img.GetHeight();

							Image crop;
							bool gray = slot->color_mode != "color";
							Size crop_aspect = slot->crop_size;
							if(crop_aspect.cx <= 0 || crop_aspect.cy <= 0)
								crop_aspect = Size(max(1, slot->template_crop_w), max(1, slot->template_crop_h));

							bool use_bbox = has_ann && (ann_idx_slot >= 0);
							if(use_bbox) {
								Value ann = ann_by_slot[ann_idx_slot];
								Value polygons = ann["polygons"];
								use_bbox = IsValueArray(polygons) && polygons.GetCount() > 0;
								if(use_bbox) {
									Value pts = polygons[0];
									crop = CropPolygonAspect(img, pts, img_w, img_h,
									                         lay.bbox_expand, crop_aspect);
								}
							}

							if(crop.IsEmpty()) {
								bool use_tpl = slot->template_w > 0 && slot->template_h > 0 &&
								               slot->template_crop_w > 0 && slot->template_crop_h > 0;
								if(use_tpl && !slot->anchor.IsEmpty()) {
									Image element = CropAnchor(img, slot->anchor, lay.bbox_expand,
									                        Size(slot->template_w, slot->template_h));
									int x1 = ClampI(slot->template_crop_x, 0, max(0, slot->template_w - 1));
									int y1 = ClampI(slot->template_crop_y, 0, max(0, slot->template_h - 1));
									int x2 = min(slot->template_w, x1 + slot->template_crop_w);
									int y2 = min(slot->template_h, y1 + slot->template_crop_h);
									if(!element.IsEmpty() && x2 > x1 && y2 > y1)
										crop = Crop(element, Rect(x1, y1, x2, y2));
								}
								else if(!slot->anchor.IsEmpty()) {
									crop = CropAnchor(img, slot->anchor, lay.bbox_expand, crop_aspect);
								}
							}
							if(crop.IsEmpty()) continue;

							String fname = Format("%s_%s_meta.png", slot_id, stem);
							String path  = AppendFileName(AppendFileName(out_dir, class_name), fname);
							if(SaveCrop(crop, path, gray)) {
								int& cnt = result.samples_per_group.GetAdd(disp, 0);
								cnt++;
							}
						}
					}

					if(pass_index == 2 && verified && has_ann) {
						// Pass 2: bbox crop from verified image
						// Get class name from image metadata (slot_id as key)
						String class_name = GetMetaValue(img_rec, slot_id);
						if(class_name.IsEmpty()) continue;

						Value ann = ann_by_slot[ann_idx_slot];
						Value polygons = ann["polygons"];
						if(!IsValueArray(polygons) || polygons.GetCount() == 0) continue;
						Value pts = polygons[0];

						if(img.IsEmpty()) img = LoadImage(img_rec, images_dir);
						if(img.IsEmpty()) break;

						if(img_w <= 0) img_w = img.GetWidth();
						if(img_h <= 0) img_h = img.GetHeight();

						Image crop = CropPolygon(img, pts, img_w, img_h,
						                         lay.bbox_expand, slot->crop_size);
						if(crop.IsEmpty()) continue;

						bool gray = slot->color_mode != "color";
						String fname = Format("%s_%s_bbox.png", slot_id, stem);
						String path  = AppendFileName(AppendFileName(out_dir, class_name), fname);
						if(SaveCrop(crop, path, gray)) {
							int& cnt = result.samples_per_group.GetAdd(disp, 0);
							cnt++;
						}
					}
				}
			}
		}

		result.images_processed++;
	}

	// Write manifest.json
	{
		ValueMap manifest;
		manifest.Add("pass_index",    pass_index);
		{
			Time t = GetSysTime();
			manifest.Add("export_time", Format("%04d-%02d-%02dT%02d:%02d:%02d",
			             t.year, t.month, t.day, t.hour, t.minute, t.second));
		}
		manifest.Add("images_count",  result.images_processed);
		ValueMap samples_map;
		for(int i = 0; i < result.samples_per_group.GetCount(); i++)
			samples_map.Add(result.samples_per_group.GetKey(i), result.samples_per_group[i]);
		manifest.Add("samples_per_group", samples_map);
		String manifest_path = AppendFileName(pass_dir, "manifest.json");
		RealizeDirectory(pass_dir);
		SaveFile(manifest_path, StoreAsJson(manifest, true));
	}

	return result;
}

END_UPP_NAMESPACE
