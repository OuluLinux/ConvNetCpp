#define BLITZ_PROHIBIT
#include "AnchoredSlotClassifier.h"
#include "GroupRegistry.h"

#include <Draw/Draw.h>
#include <plugin/jpg/jpg.h>
#include <plugin/png/png.h>

NAMESPACE_UPP

namespace {

struct BBoxNorm {
	double cx = 0;
	double cy = 0;
	double w = 0;
	double h = 0;
	bool ok = false;
};

String TrimLower(const String& s) {
	return ToLower(TrimBoth(s));
}

String NormalizeCardCode(const String& element) {
	if(element.GetCount() != 2)
		return String();
	String out;
	out.Cat(ToUpper((int)element[0]));
	out.Cat(ToLower((int)element[1]));
	return out;
}

String StripSpaces(const String& s) {
	String out;
	out.Reserve(s.GetCount());
	for(int i = 0; i < s.GetCount(); i++) {
		if(!IsSpace((int)s[i]))
			out.Cat(s[i]);
	}
	return out;
}

BBoxNorm ComputeNormalizedBBox(const Value& polygon_points, int iw, int ih) {
	BBoxNorm out;
	if(!IsValueArray(polygon_points) || iw <= 0 || ih <= 0)
		return out;

	double min_x = 1e18, min_y = 1e18, max_x = -1e18, max_y = -1e18;
	bool has_point = false;
	for(int i = 0; i < polygon_points.GetCount(); i++) {
		Value p = polygon_points[i];
		Value xv = p["x"];
		Value yv = p["y"];
		if(IsNull(xv) || IsNull(yv))
			continue;
		double x = xv;
		double y = yv;
		min_x = min(min_x, x);
		max_x = max(max_x, x);
		min_y = min(min_y, y);
		max_y = max(max_y, y);
		has_point = true;
	}
	if(!has_point)
		return out;

	out.cx = ((min_x + max_x) * 0.5) / iw;
	out.cy = ((min_y + max_y) * 0.5) / ih;
	out.w = (max_x - min_x) / iw;
	out.h = (max_y - min_y) / ih;
	out.ok = out.w > 0 && out.h > 0;
	return out;
}

bool IsAnnotationPresent(const Value& ann) {
	Value polygons = ann["polygons"];
	if(!IsValueArray(polygons) || polygons.GetCount() == 0)
		return false;
	if(!IsValueArray(polygons[0]) || polygons[0].GetCount() == 0)
		return false;
	return true;
}

Image LoadAnnprjImage(const Value& imgv, const String& images_dir) {
	String file_name = imgv["file_name"];
	if(!IsNull(file_name) && !file_name.IsEmpty()) {
		String p = AppendFileName(images_dir, file_name);
		Image img = StreamRaster::LoadFileAny(p);
		if(!img.IsEmpty())
			return img;
	}

	String file_path = imgv["file_path"];
	if(!IsNull(file_path) && !file_path.IsEmpty()) {
		String base = GetFileName(file_path);
		String p = AppendFileName(images_dir, base);
		Image img = StreamRaster::LoadFileAny(p);
		if(!img.IsEmpty())
			return img;
		img = StreamRaster::LoadFileAny(file_path);
		if(!img.IsEmpty())
			return img;
	}
	return Image();
}

} // namespace

Image HighLuminanceThresholdBinarization(const Image& src, double white_thresh) {
	if(src.IsEmpty())
		return src;
	int w = src.GetWidth();
	int h = src.GetHeight();
	byte thr = (byte)minmax((int)round(white_thresh * 255.0), 0, 255);
	ImageBuffer out(w, h);
	for(int y = 0; y < h; y++) {
		const RGBA* s = src[y];
		RGBA* d = out[y];
		for(int x = 0; x < w; x++) {
			int lum = ((int)s[x].r * 299 + (int)s[x].g * 587 + (int)s[x].b * 114) / 1000;
			byte v = lum >= thr ? 255 : 0;
			d[x] = RGBA{v, v, v, 255};
		}
	}
	return out;
}

namespace {

void ShuffleSamples(Vector<Vector<double>>& v) {
	srand(42);
	for(int i = v.GetCount() - 1; i > 0; i--) {
		int j = rand() % (i + 1);
		if(i != j)
			Swap(v[i], v[j]);
	}
}

void ShufflePaired(Vector<Vector<double>>& samples, Vector<int>& labels) {
	ASSERT(samples.GetCount() == labels.GetCount());
	srand(42);
	for(int i = samples.GetCount() - 1; i > 0; i--) {
		int j = rand() % (i + 1);
		if(i != j) {
			Swap(samples[i], samples[j]);
			Swap(labels[i], labels[j]);
		}
	}
}

bool ParseTrailingIndex(const String& slot_id, String& stem, int& index) {
	int pos = slot_id.GetCount() - 1;
	if(pos < 0 || !IsDigit((int)slot_id[pos]))
		return false;
	while(pos >= 0 && IsDigit((int)slot_id[pos]))
		pos--;
	if(pos < 0 || slot_id[pos] != '_')
		return false;
	stem = slot_id.Left(pos);
	index = StrInt(slot_id.Mid(pos + 1));
	return !stem.IsEmpty() && index >= 1;
}

String GetIndexedClassFromImageMeta(const Value& img, const String& slot_id) {
	String stem;
	int idx = 0;
	if(!ParseTrailingIndex(slot_id, stem, idx))
		return String();
	String meta_key = stem + "s";
	String seq = img[meta_key];
	seq = StripSpaces(seq);
	int p = (idx - 1) * 2;
	if(p + 1 < seq.GetCount())
		return NormalizeCardCode(seq.Mid(p, 2));
	return String();
}

String ClassesGroupKey(const Vector<String>& classes) {
	String key;
	for(const String& s : classes) {
		key << s;
		key << "|";
	}
	return key;
}

bool ContainsString(const Vector<String>& values, const String& value) {
	for(const String& v : values)
		if(v == value)
			return true;
	return false;
}

void AddSlotToGroup(VectorMap<String, Vector<String>>& groups, const String& group_key, const String& slot_id) {
	if(group_key.IsEmpty() || slot_id.IsEmpty())
		return;
	Vector<String>& ids = groups.GetAdd(group_key);
	if(!ContainsString(ids, slot_id))
		ids.Add(slot_id);
}

bool SameClasses(const Vector<String>& a, const Vector<String>& b) {
	if(a.GetCount() != b.GetCount())
		return false;
	for(int i = 0; i < a.GetCount(); i++) {
		if(TrimLower(a[i]) != TrimLower(b[i]))
			return false;
	}
	return true;
}

String FindLabelGroupByClasses(const AnnLay& lay, const Vector<String>& classes) {
	for(const AnnLayGroup& g : lay.groups) {
		if(g.method != ANNLAY_CLASSIFIER_LABEL)
			continue;
		if(SameClasses(g.classes, classes))
			return g.id;
	}
	return String();
}

String FindRoleGroupInLayout(const AnnLay& lay, const String& role) {
	String target = TrimLower(role);
	for(const AnnLayGroup& g : lay.groups) {
		if(TrimLower(g.head_role) == target)
			return g.id;
	}
	return String();
}

String BuildNetJson(int w, int h, int d, int cls_count) {
	String s;
	s << "["
	  << "{\"type\":\"input\",\"input_width\":" << w
	  << ",\"input_height\":" << h
	  << ",\"input_depth\":" << d << "},"
	  << "{\"type\":\"conv\",\"width\":5,\"height\":5,\"filter_count\":8,\"stride\":1,\"pad\":2,\"activation\":\"relu\"},"
	  << "{\"type\":\"pool\",\"width\":2,\"height\":2,\"stride\":2},"
	  << "{\"type\":\"conv\",\"width\":5,\"height\":5,\"filter_count\":16,\"stride\":1,\"pad\":2,\"activation\":\"relu\"},"
	  << "{\"type\":\"pool\",\"width\":2,\"height\":2,\"stride\":2},"
	  << "{\"type\":\"softmax\",\"class_count\":" << cls_count << "},"
	  << "{\"type\":\"adadelta\",\"batch_size\":20,\"l2_decay\":0.001}"
	  << "]";
	return s;
}

}

String AnchoredSlotClassifier::BoolSlotGroupKey(const String& slot_id, const AnnLay* lay) {
	if(lay) {
		String base_id = slot_id;
		String suffix;
		int hash = slot_id.Find('#');
		if(hash >= 0) {
			base_id = slot_id.Left(hash);
			suffix = slot_id.Mid(hash + 1); // e.g. "presence", "level"
		}

		for(const auto& s : lay->slots) {
			if(s.id == base_id) {
				if(!suffix.IsEmpty()) {
					String g = s.sub_groups.Get(suffix, "");
					if(!g.IsEmpty()) return g;
				}
				if(!s.group.IsEmpty())
					return s.group;
				break;
			}
		}
	}

	return slot_id;
}

double AnchoredSlotClassifier::GetSlotRotation(const String& slot_id, const AnnLay* lay) {
	if(lay) {
		for(const auto& s : lay->slots) {
			if(s.id == slot_id)
				return s.fixed_rotation_deg;
		}
	}

	if(slot_id == "hero_card_1" || slot_id == "is_hero_card_1") return -4.0;
	if(slot_id == "hero_card_2" || slot_id == "is_hero_card_2") return 4.0;
	return 0.0;
}

bool AnchoredSlotClassifier::MatchSlotForBool(const String& target_slot, const String& candidate_slot) {
	if(target_slot == candidate_slot)
		return true;
	
	if(target_slot.EndsWith("_is_bet_chip") && candidate_slot.EndsWith("_bet")) {
		String prefix1 = target_slot.Left(target_slot.GetCount() - 12);
		String prefix2 = candidate_slot.Left(candidate_slot.GetCount() - 4);
		if(prefix1 == prefix2)
			return true;
	}
	
	if(target_slot.EndsWith("_is_bet_chip") && candidate_slot.EndsWith("_bet_chip")) {
		String prefix1 = target_slot.Left(target_slot.GetCount() - 12);
		String prefix2 = candidate_slot.Left(candidate_slot.GetCount() - 9);
		if(prefix1 == prefix2)
			return true;
	}

	if(target_slot == "is_previous_streets_pot" && candidate_slot == "previous_streets_pot")
		return true;
	if(target_slot == "is_side_pot" && candidate_slot == "side_pot")
		return true;
	if(target_slot == "is_previous_pot" && candidate_slot == "pot_previous")
		return true;
	if(target_slot == "is_total_pot" && candidate_slot == "total_pot")
		return true;
	
	if(target_slot == "is_action_call" && candidate_slot == "action_call")
		return true;
	if(target_slot == "is_action_fold" && candidate_slot == "action_fold")
		return true;
	if(target_slot == "is_action_raise" && candidate_slot == "action_raise")
		return true;

	if(target_slot.StartsWith("is_board_card_") && candidate_slot.StartsWith("board_card_")) {
		if(target_slot.Mid(14) == candidate_slot.Mid(11))
			return true;
	}
	if(target_slot.StartsWith("is_hero_card_") && candidate_slot.StartsWith("hero_card_")) {
		if(target_slot.Mid(13) == candidate_slot.Mid(10))
			return true;
	}

	if(target_slot == "active_timer" && candidate_slot.EndsWith("_active_timer"))

		return true;
	if(target_slot == "dealer_chip" && candidate_slot.EndsWith("_dealer_chip"))
		return true;
	
	return false;
}

VectorMap<String, Vector<String>> AnchoredSlotClassifier::GetSlotGroups(const AnnLay& lay) {
	VectorMap<String, Vector<String>> groups;
	
	// Pre-fill from explicit groups defined in layout
	for(const auto& g : lay.groups) {
		groups.GetAdd(ResolveCanonicalGroupKey(lay, g.id));
	}

	for(const AnnLaySlot& slot : lay.slots) {
		// 1. Check sub-groups mapping (e.g. composite element tasks)
		bool has_sub_groups = false;
		for(int i = 0; i < slot.sub_groups.GetCount(); i++) {
			String gkey = ResolveCanonicalGroupKey(lay, slot.sub_groups[i]);
			if(!gkey.IsEmpty()) {
				AddSlotToGroup(groups, gkey, slot.id);
				has_sub_groups = true;
			}
		}

		// Composite layouts may omit explicit per-slot sub_groups. Fall back to
		// role-defined groups from layout metadata.
		if(slot.composite_type == ANNLAY_COMPOSITE_ELEMENT) {
			auto AddRoleGroup = [&](const String& role) {
				String gkey = TrimBoth(slot.sub_groups.Get(role, ""));
				if(gkey.IsEmpty())
					gkey = FindRoleGroupInLayout(lay, role);
				gkey = ResolveCanonicalGroupKey(lay, gkey);
				if(!gkey.IsEmpty()) {
					AddSlotToGroup(groups, gkey, slot.id);
					has_sub_groups = true;
				}
			};
			AddRoleGroup("presence");
			AddRoleGroup("level");
			AddRoleGroup("category");
		}

		// 2. Check primary group assignment
		if(!slot.group.IsEmpty()) {
			AddSlotToGroup(groups, ResolveCanonicalGroupKey(lay, slot.group), slot.id);
			continue;
		}

		if(has_sub_groups) continue;

		// 3. Fallback heuristic for unconfigured slots
		if(slot.method == ANNLAY_CLASSIFIER_BOOL) {
			String key = BoolSlotGroupKey(slot.id, &lay);
			key = ResolveCanonicalGroupKey(lay, key);
			AddSlotToGroup(groups, key, slot.id);
		}
		else if(slot.method == ANNLAY_CLASSIFIER_LABEL) {
			String key;
			if(slot.composite_type == ANNLAY_COMPOSITE_ELEMENT) {
				String suit_group = TrimBoth(slot.sub_groups.Get("category", ""));
				if(!suit_group.IsEmpty())
					key = suit_group;
			}
			if(key.IsEmpty())
				key = FindLabelGroupByClasses(lay, slot.classes);
			if(key.IsEmpty())
				key = ClassesGroupKey(slot.classes);
			key = ResolveCanonicalGroupKey(lay, key);
			AddSlotToGroup(groups, key, slot.id);
		}
	}

	// Filter out any virtual IDs that accidentally got added as group keys
	for(int i = groups.GetCount() - 1; i >= 0; i--) {
		String k = groups.GetKey(i);
		if(k.Find('#') >= 0 || groups[i].IsEmpty())
			groups.Remove(i);
	}

	return groups;
}

String AnchoredSlotClassifier::ResolveCanonicalGroupKey(const AnnLay& lay, const String& group_key) {
	GroupRegistry reg;
	reg.Build(lay);
	return reg.Resolve(group_key);
}

String AnchoredSlotClassifier::GetGroupRole(const AnnLay& lay, const String& group_key) {
	GroupRegistry reg;
	reg.Build(lay);
	return reg.HeadRole(group_key);
}

String AnchoredSlotClassifier::GetSlotGroupDisplayName(const AnnLay& lay, const String& group_key) {
	GroupRegistry reg;
	reg.Build(lay);
	String canonical = reg.Resolve(group_key);
	const GroupRegistryEntry* ge = reg.Find(canonical);
	if(ge && !ge->display_name.IsEmpty())
		return ge->display_name;

	VectorMap<String, Vector<String>> groups = GetSlotGroups(lay);
	int g = groups.Find(canonical);
	if(g < 0)
		return canonical;
	const Vector<String>& ids = groups[g];
	if(ids.IsEmpty())
		return canonical;

	const AnnLaySlot* first = lay.FindSlot(ids[0]);
	if(!first)
		return canonical;

	if(first->method == ANNLAY_CLASSIFIER_LABEL) {
		if(!first->classes.IsEmpty() && !first->classes[0].IsEmpty())
			return first->classes[0];
	}
	return canonical;
}

void AnchoredSlotClassifier::TrainAll(AnnLay& lay, const String& annprj_path,
                                      const String& images_dir,
                                      const String& card_strip_path,
                                      AnnMdl* mdl) {
	VectorMap<String, Vector<int>> model_groups;
	VectorMap<String, Vector<String>> slot_groups = GetSlotGroups(lay);
	for(int g = 0; g < slot_groups.GetCount(); g++) {
		Vector<int>& indices = model_groups.GetAdd(slot_groups.GetKey(g));
		for(const String& id : slot_groups[g]) {
			for(int i = 0; i < lay.slots.GetCount(); i++) {
				if(lay.slots[i].id == id) {
					indices.Add(i);
					break;
				}
			}
		}
	}

	for(int i = 0; i < model_groups.GetCount(); i++) {
		String group_id = model_groups.GetKey(i);
		const Vector<int>& indices = model_groups[i];
		if(indices.IsEmpty()) continue;

		AnnLaySlot& ref_slot = lay.slots[indices[0]];

		// Configuration resolution (prioritize groups metadata)
		AnnLayMethod method = ref_slot.method;
		Size crop_size = ref_slot.crop_size;
		Vector<String> class_names;
		class_names <<= ref_slot.classes;
		String color_mode = ref_slot.color_mode;

		for(const auto& g : lay.groups) {
			if(g.id == group_id) {
				method = g.method;
				crop_size = g.crop_size;
				class_names <<= g.classes;
				color_mode = g.color_mode;
				break;
			}
		}

		if(!IsTrainableMethod(method))
			continue;

		int w = max(1, crop_size.cx);
		int h = max(1, crop_size.cy);
		int depth = (color_mode == "color" ? 3 : 1);
		int cls_count = max(2, class_names.GetCount());

		Vector<Vector<double>> samples;
		Vector<int> labels;

		if(method == ANNLAY_CLASSIFIER_BOOL) {
			Vector<Vector<double>> true_samples, false_samples;
			for(int idx : indices) {
				BuildBoolDataset(lay.slots[idx], lay, annprj_path, images_dir, true_samples, false_samples);
			}
			if(true_samples.IsEmpty() && false_samples.IsEmpty()) continue;

			for(auto& s : true_samples) { samples.Add(pick(s)); labels.Add(1); }
			for(auto& s : false_samples) { samples.Add(pick(s)); labels.Add(0); }
			ShufflePaired(samples, labels);

			String net_json = BuildNetJson(w, h, depth, 2);
			ConvNet::Session ses;
			Vector<String> bool_classes;
			bool_classes.Add("false");
			bool_classes.Add("true");
			TrainSession(ses, net_json, 2, w * h * depth, w, h, depth, samples, labels, group_id, &bool_classes);
			if(mdl) {
				EmbedWeights(group_id, *mdl, ses, net_json);
				for(int idx : indices)
					EmbedWeights(lay.slots[idx].id, *mdl, ses, net_json);
			}
		}
		else if(method == ANNLAY_CLASSIFIER_LABEL) {
			// Element slots (first class == "no_card") require element templates — skip if not provided.
			bool is_card_slot = class_names.GetCount() > 0 && class_names[0] == "no_card";
			if(is_card_slot && card_strip_path.IsEmpty()) continue;
			for(int idx : indices) {
				BuildLabelDataset(lay.slots[idx], lay, annprj_path, images_dir, card_strip_path, samples, labels);
			}
			if(samples.IsEmpty()) continue;

			ShufflePaired(samples, labels);
			String net_json = BuildNetJson(w, h, depth, cls_count);
			ConvNet::Session ses;
			TrainSession(ses, net_json, cls_count, w * h * depth, w, h, depth, samples, labels, group_id, &class_names);
			if(mdl) {
				EmbedWeights(group_id, *mdl, ses, net_json);
				for(int idx : indices)
					EmbedWeights(lay.slots[idx].id, *mdl, ses, net_json);
			}
		}
	}
}

void AnchoredSlotClassifier::TrainSlot(AnnLay& lay, const String& slot_id,
                                       const String& annprj_path,
                                       const String& images_dir,
                                       const String& card_strip_path,
                                       AnnMdl* mdl) {
	AnnLaySlot* slot = lay.FindSlot(slot_id);
	if(!slot)
		return;

	int w = max(1, slot->crop_size.cx);
	int h = max(1, slot->crop_size.cy);

	if(slot->method == ANNLAY_CLASSIFIER_BOOL) {
		Vector<Vector<double>> true_samples, false_samples;
		BuildBoolDataset(*slot, lay, annprj_path, images_dir, true_samples, false_samples);
		if(true_samples.IsEmpty() && false_samples.IsEmpty())
			return;

		Vector<Vector<double>> samples;
		Vector<int> labels;
		for(int i = 0; i < true_samples.GetCount(); i++) {
			samples.Add(pick(true_samples[i]));
			labels.Add(1);
		}
		for(int i = 0; i < false_samples.GetCount(); i++) {
			samples.Add(pick(false_samples[i]));
			labels.Add(0);
		}
		ShufflePaired(samples, labels);

		String net_json = BuildNetJson(w, h, 1, 2);
		ConvNet::Session ses;
		Vector<String> bool_classes;
		bool_classes.Add("false");
		bool_classes.Add("true");
		TrainSession(ses, net_json, 2, w * h, w, h, 1, samples, labels, slot->id, &bool_classes);
		if(mdl)
			EmbedWeights(slot->id, *mdl, ses, net_json);
	}
	else if(slot->method == ANNLAY_CLASSIFIER_LABEL) {
		// Element slots require element templates — skip if not provided.
		bool is_card_slot = slot->classes.GetCount() > 0 && slot->classes[0] == "no_card";
		if(is_card_slot && card_strip_path.IsEmpty())
			return;
		Vector<Vector<double>> samples;
		Vector<int> labels;
		BuildLabelDataset(*slot, lay, annprj_path, images_dir, card_strip_path, samples, labels);
		if(samples.IsEmpty() || slot->classes.GetCount() <= 1)
			return;

		ShufflePaired(samples, labels);
		int cls_count = max(2, slot->classes.GetCount());
		String net_json = BuildNetJson(w, h, 3, cls_count);
		ConvNet::Session ses;
		TrainSession(ses, net_json, cls_count, w * h * 3, w, h, 3, samples, labels, slot->id, &slot->classes);
		if(mdl)
			EmbedWeights(slot->id, *mdl, ses, net_json);
	}
}

void AnchoredSlotClassifier::BuildBoolDataset(const AnnLaySlot& slot,
                                               const AnnLay& lay,
                                               const String& annprj_path,
                                               const String& images_dir,
                                               Vector<Vector<double>>& true_samples,
                                               Vector<Vector<double>>& false_samples) {
	const bool is_composite_card_gate = (slot.composite_type == ANNLAY_COMPOSITE_ELEMENT);
	const bool is_trainable = (slot.method == ANNLAY_CLASSIFIER_BOOL || slot.method == ANNLAY_CV_TEMPLATE_MATCH);

	if(!is_trainable && !is_composite_card_gate)
		return;

	auto BuildBoolSample = [&](const Image& crop) -> Vector<double> {

		if(crop.IsEmpty())
			return Vector<double>();
		Image src = crop;
		bool equalize = false;
		if(is_composite_card_gate)
			src = HighLuminanceThresholdBinarization(src);
		return ImageToSample(src, true, slot.crop_size, equalize);
	};

	String data = LoadFile(annprj_path);
	if(data.IsEmpty())
		return;
	Value root = ParseJSON(data);
	if(IsNull(root) || !IsValueMap(root))
		return;

	Value datasets = root["datasets"];
	if(!IsValueArray(datasets))
		return;

	for(int di = 0; di < datasets.GetCount(); di++) {
		Value ds = datasets[di];
		Value images = ds["images"];
		if(!IsValueArray(images))
			continue;

		for(int ii = 0; ii < images.GetCount(); ii++) {
			Value imgv = images[ii];
			Image img = LoadAnnprjImage(imgv, images_dir);
			if(img.IsEmpty())
				continue;
			int iw = img.GetWidth();
			int ih = img.GetHeight();
			if(iw <= 0 || ih <= 0)
				continue;

			Value annotations = imgv["annotations"];
			if(!IsValueArray(annotations))
				continue;

			bool has_present = false;
			for(int ai = 0; ai < annotations.GetCount(); ai++) {
				Value ann = annotations[ai];
				String ann_slot = ann["metadata"]["mlui_slot_id"];
				if(IsNull(ann_slot) || ann_slot.IsEmpty())
					continue;
				if(!MatchSlotForBool(slot.id, ann_slot))
					continue;

				if(IsAnnotationPresent(ann)) {
					has_present = true;
					Value polygons = ann["polygons"];
					BBoxNorm bb = ComputeNormalizedBBox(polygons[0], iw, ih);
					if(!bb.ok)
						continue;
					AnnLayAnchor a;
					a.cx = bb.cx;
					a.cy = bb.cy;
					a.w = bb.w;
					a.h = bb.h;
					if(slot.anchor_candidates.IsEmpty()) {
						// Fixed-position bool slots train on the same averaged anchor used at inference.
						Image crop = CropSlot(img, is_composite_card_gate ? a : slot.anchor, lay.bbox_expand, slot.crop_size);
						if(!crop.IsEmpty())
							true_samples.Add(BuildBoolSample(crop));
					}
					else {
						// Variable-position slot: find closest candidate -> true,
						// all other candidates -> false
						int best = 0;
						double best_d = 1e18;
						for(int ci = 0; ci < slot.anchor_candidates.GetCount(); ci++) {
							const AnnLayAnchor& cand = slot.anchor_candidates[ci];
							double dx = cand.cx - a.cx;
							double dy = cand.cy - a.cy;
							double d = dx*dx + dy*dy;
							if(d < best_d) { best_d = d; best = ci; }
						}
						for(int ci = 0; ci < slot.anchor_candidates.GetCount(); ci++) {
							Image crop = CropSlot(img, slot.anchor_candidates[ci],
							                      lay.bbox_expand, slot.crop_size);
							if(crop.IsEmpty())
								continue;
							if(ci == best)
								true_samples.Add(BuildBoolSample(crop));
							else
								false_samples.Add(BuildBoolSample(crop));
						}
					}
				}
			}

			// No present annotation for this slot in this image => false sample(s).
			if(!has_present) {
				if(!slot.anchor.IsEmpty()) {
					// Fixed-position: false crop from average anchor.
					Image crop = CropSlot(img, slot.anchor, lay.bbox_expand, slot.crop_size);
					if(!crop.IsEmpty())
						false_samples.Add(BuildBoolSample(crop));
				}
				else {
					// Variable-position: all candidates are false.
					for(int ci = 0; ci < slot.anchor_candidates.GetCount(); ci++) {
						Image crop = CropSlot(img, slot.anchor_candidates[ci],
						                      lay.bbox_expand, slot.crop_size);
						if(!crop.IsEmpty())
							false_samples.Add(BuildBoolSample(crop));
					}
				}
			}
		}
	}

	ShuffleSamples(true_samples);
	ShuffleSamples(false_samples);
}

void AnchoredSlotClassifier::BuildLabelDataset(const AnnLaySlot& slot,
                                               const AnnLay& lay,
                                               const String& annprj_path,
                                               const String& images_dir,
                                               const String& card_strip_path,
                                               Vector<Vector<double>>& samples,
                                               Vector<int>& labels) {
	String resolved_key = slot.id;
	if(slot.composite_type == ANNLAY_COMPOSITE_ELEMENT) {
		String suit_group = TrimBoth(slot.sub_groups.Get("category", ""));
		if(!suit_group.IsEmpty())
			resolved_key = suit_group;
	}
	String resolved_role = AnchoredSlotClassifier::GetGroupRole(lay, resolved_key);

	Vector<String> effective_classes;
	effective_classes <<= slot.classes;

	VectorMap<String, int> class_map;
	for(int i = 0; i < effective_classes.GetCount(); i++) {
		class_map.GetAdd(TrimLower(effective_classes[i])) = i;
	}

	String data = LoadFile(annprj_path);
	if(!data.IsEmpty()) {
		Value root = ParseJSON(data);
		if(!IsNull(root) && IsValueMap(root)) {
			Value datasets = root["datasets"];
			if(IsValueArray(datasets)) {
				for(int di = 0; di < datasets.GetCount(); di++) {
					Value ds = datasets[di];
					Value images = ds["images"];
					if(!IsValueArray(images))
						continue;

					for(int ii = 0; ii < images.GetCount(); ii++) {
						Value imgv = images[ii];
						Image img = LoadAnnprjImage(imgv, images_dir);
						if(img.IsEmpty())
							continue;
						int iw = img.GetWidth();
						int ih = img.GetHeight();
						if(iw <= 0 || ih <= 0)
							continue;

						String code = GetIndexedClassFromImageMeta(imgv, slot.id);
						if(resolved_role == "category" && code.GetCount() >= 2) {
							code = code.Right(1); // Extract category from "As", "Kh", etc.
						}

						int label = -1;
						if(!code.IsEmpty()) {
							int q = class_map.Find(TrimLower(code));
							if(q >= 0)
								label = class_map[q];
						}
						
						if(label < 0) continue; // Skip unknown classes for this task

						Value annotations = imgv["annotations"];
						if(!IsValueArray(annotations))
							continue;

						// Check if this slot has a bbox annotation in this image.
						bool has_bbox = false;
						for(int ai = 0; ai < annotations.GetCount(); ai++) {
							Value ann = annotations[ai];
							String ann_slot = ann["metadata"]["mlui_slot_id"];
							if(ann_slot == slot.id && IsAnnotationPresent(ann)) {
								has_bbox = true;
								break;
							}
						}

						if(has_bbox) {
							// Phase 2: use bbox crops (skipped in phase 1 via element slot check above)
							for(int ai = 0; ai < annotations.GetCount(); ai++) {
								Value ann = annotations[ai];
								String ann_slot = ann["metadata"]["mlui_slot_id"];
								if(ann_slot != slot.id || !IsAnnotationPresent(ann))
									continue;
								Value polygons = ann["polygons"];
								BBoxNorm bb = ComputeNormalizedBBox(polygons[0], iw, ih);
								if(!bb.ok)
									continue;
								AnnLayAnchor a;
								a.cx = bb.cx;
								a.cy = bb.cy;
								a.w = bb.w;
								a.h = bb.h;
								Image crop = CropSlot(img, a, lay.bbox_expand, slot.crop_size);
								if(crop.IsEmpty())
									continue;
								samples.Add(ImageToSample(crop, false, slot.crop_size));
								labels.Add(label);
							}
						}
						else if(!slot.anchor.IsEmpty()) {
							// No bbox annotation → use anchor crop as no_card
							Image crop = CropSlot(img, slot.anchor, lay.bbox_expand, slot.crop_size);
							if(crop.IsEmpty())
								continue;
							samples.Add(ImageToSample(crop, false, slot.crop_size));
							labels.Add(0);
						}
					}
				}
			}
		}
	}

	if(!card_strip_path.IsEmpty()) {
		Image strip = StreamRaster::LoadFileAny(card_strip_path);
		if(!strip.IsEmpty() && strip.GetWidth() > 0 && strip.GetHeight() > 0) {
			int cell_w = max(1, strip.GetWidth() / 85);
			for(int cls = 1; cls <= 52 && cls < slot.classes.GetCount(); cls++) {
				int col = cls - 1;
				int x = col * cell_w + 4;
				int y = 6;
				int w = min(22, strip.GetWidth() - x);
				int h = min(22, strip.GetHeight() - y);
				if(x < 0 || y < 0 || w <= 0 || h <= 0)
					continue;
				Image cell = Crop(strip, RectC(x, y, w, h));
				if(cell.IsEmpty())
					continue;
				Image crop = Rescale(cell, slot.crop_size);
				samples.Add(ImageToSample(crop, false, slot.crop_size));
				labels.Add(cls);
			}
		}
	}
}

Image AnchoredSlotClassifier::CropSlot(const Image& img, const AnnLayAnchor& anchor,
                                       double bbox_expand, Size crop_size) {
	if(img.IsEmpty() || crop_size.cx <= 0 || crop_size.cy <= 0)
		return Image();

	int iw = img.GetWidth();
	int ih = img.GetHeight();
	if(iw <= 0 || ih <= 0)
		return Image();

	double exp_w = anchor.w * (1.0 + 2.0 * bbox_expand);
	double exp_h = anchor.h * (1.0 + 2.0 * bbox_expand);
	int x1 = max(0, (int)((anchor.cx - exp_w * 0.5) * iw));
	int y1 = max(0, (int)((anchor.cy - exp_h * 0.5) * ih));
	int x2 = min(iw, (int)((anchor.cx + exp_w * 0.5) * iw));
	int y2 = min(ih, (int)((anchor.cy + exp_h * 0.5) * ih));
	if(x2 <= x1 || y2 <= y1)
		return Image();

	Image sub = Crop(img, Rect(x1, y1, x2, y2));
	if(sub.IsEmpty())
		return Image();
	return Rescale(sub, crop_size);
}

Vector<double> AnchoredSlotClassifier::ImageToSample(const Image& crop, bool grayscale, Size crop_size, bool equalize) {
	Image out = equalize ? LinearContrastStretching(crop) : crop;
	int depth = grayscale ? 1 : 3;
	int len = crop_size.cx * crop_size.cy * depth;
	Vector<double> sample;
	sample.SetCount(len, 0.0);

	if(grayscale) {
		Image gray = Grayscale(out);
		for(int y = 0; y < crop_size.cy; y++) {
			const RGBA* row = gray[y];
			for(int x = 0; x < crop_size.cx; x++) {
				int pos = (y * crop_size.cx + x);
				sample[pos] = row[x].r / 255.0;
			}
		}
	}
	else {
		for(int y = 0; y < crop_size.cy; y++) {
			const RGBA* row = out[y];
			for(int x = 0; x < crop_size.cx; x++) {
				int base = (y * crop_size.cx + x) * 3;
				sample[base + 0] = row[x].r / 255.0;
				sample[base + 1] = row[x].g / 255.0;
				sample[base + 2] = row[x].b / 255.0;
			}
		}
	}
	return sample;
}

void AnchoredSlotClassifier::TrainSession(ConvNet::Session& ses,
                                          const String& net_json,
                                          int n_classes,
                                          int sample_len,
                                          int w, int h, int depth,
                                          const Vector<Vector<double>>& samples,
                                          const Vector<int>& labels,
                                          const String& slot_id,
                                          const Vector<String>* class_names) {
	if(samples.IsEmpty() || samples.GetCount() != labels.GetCount())
		return;
	if(!ses.MakeLayers(net_json))
		return;

	Vector<Vector<double>> valid_samples;
	Vector<int> valid_labels;
	for(int i = 0; i < samples.GetCount(); i++) {
		if(samples[i].GetCount() != sample_len)
			continue;
		int lbl = labels[i];
		if(lbl < 0 || lbl >= n_classes)
			continue;
		Vector<double>& dst = valid_samples.Add();
		dst <<= samples[i];
		valid_labels.Add(lbl);
	}
	if(valid_samples.IsEmpty())
		return;

	int n = valid_samples.GetCount();
	int val_n = n >= 2 ? max(1, n / 10) : 0;
	int train_n = n - val_n;
	if(train_n <= 0)
		return;

	ConvNet::SessionData& d = ses.Data();
	d.BeginData(n_classes, train_n, w, h, depth, val_n);
	for(int cls = 0; cls < n_classes; cls++) {
		String cls_name = Format("class_%d", cls);
		if(class_names && cls < class_names->GetCount() && !(*class_names)[cls].IsEmpty())
			cls_name = (*class_names)[cls];
		d.SetClass(cls, cls_name);
	}
	for(int i = 0; i < train_n; i++) {
		d.Get(i) <<= valid_samples[i];
		d.SetLabel(i, valid_labels[i]);
	}
	for(int i = 0; i < val_n; i++) {
		d.GetTest(i) <<= valid_samples[train_n + i];
		d.SetTestLabel(i, valid_labels[train_n + i]);
	}
	d.EndData();

	ses.SetTestPredict(true);
	ses.SetPredictInterval(0);
	ses.SetStepCallbackInterval(0);
	ses.SetIterationCallbackInterval(0);

	int total_iters = max(1, max_epochs);
	ses.SetMaxTrainIters(total_iters);
	ses.TrainBegin();
	int perfect_streak = 0;
	const int early_stop_patience = 5;
	for(int i = 0; i < total_iters; i++) {
		ses.TrainIteration();
		double val_acc = ses.GetValidationAccuracyAverage();
		if(WhenProgress)
			WhenProgress(slot_id, i + 1, ses.GetLossAverage(), val_acc);
		if(val_acc >= 1.0)
			perfect_streak++;
		else
			perfect_streak = 0;
		if(perfect_streak >= early_stop_patience)
			break;
	}
	ses.TrainEnd();
}

void AnchoredSlotClassifier::EmbedWeights(const String& slot_id, AnnMdl& mdl, ConvNet::Session& ses, const String& net_json) {
	AnnMdlEntry& e = mdl.GetOrAdd(slot_id);
	e.slot_id = slot_id;
	e.net_str = net_json;
	// Write weights-only blob into net_data so AnnMdl::SavePath (V3) picks it up.
	// The recognizer loads via SerializeWeights; full Serialize (session_data) is V1/V2 legacy.
	StringStream ss;
	ses.SerializeWeights(ss);
	e.net_data = ss.GetResult();
	// Clear any stale legacy fields so V3 save doesn't get confused.
	e.session_data.Clear();
	e.session_ref.Clear();
}

bool AnchoredSlotClassifier::TrainBoolGroup(AnnLay& lay,
                                           const String& group_key,
                                           const String& annprj_path,
                                           const String& images_dir,
                                           int max_epochs,
                                           Event<String,int,double,double> WhenProgress,
                                           ConvNet::Session* ext_ses,
                                           AnnMdl* mdl,
                                           bool balance_by_slot,
                                           int slot_cap,
                                           bool ignore_auto_stop,
                                           bool debug) {

	if(annprj_path.IsEmpty() || images_dir.IsEmpty())
		return false;
	if(!FileExists(annprj_path) || !DirectoryExists(images_dir))
		return false;

	VectorMap<String, Vector<String>> groups = GetSlotGroups(lay);
	GroupRegistry registry;
	registry.Build(lay);
	String resolved_group_key = group_key;
	if(resolved_group_key.IsEmpty()) {
		for(int g = 0; g < groups.GetCount() && resolved_group_key.IsEmpty(); g++) {
			for(const String& id : groups[g]) {
				const AnnLaySlot* slot = lay.FindSlot(id);
				if(slot && slot->method == ANNLAY_CLASSIFIER_BOOL) {
					resolved_group_key = groups.GetKey(g);
					break;
				}
			}
		}
	}

	Vector<String> target_slots;
	bool is_presence_group = (registry.HeadRole(resolved_group_key) == "presence");
	int g = groups.Find(resolved_group_key);
	if(g >= 0) {
		for(const String& id : groups[g]) {
			const AnnLaySlot* slot = lay.FindSlot(id);
			bool allow_composite_card = slot && slot->composite_type == ANNLAY_COMPOSITE_ELEMENT && is_presence_group;
			if(slot && (slot->method == ANNLAY_CLASSIFIER_BOOL || allow_composite_card))
				target_slots.Add(id);
		}
	}
	else {
		const AnnLaySlot* slot = lay.FindSlot(resolved_group_key);
		bool allow_composite_card = slot && slot->composite_type == ANNLAY_COMPOSITE_ELEMENT && is_presence_group;
		if(slot && (slot->method == ANNLAY_CLASSIFIER_BOOL || allow_composite_card))
			target_slots.Add(resolved_group_key);
		if(target_slots.IsEmpty() && is_presence_group) {
			for(const AnnLaySlot& s : lay.slots) {
				if(s.composite_type == ANNLAY_COMPOSITE_ELEMENT)
					target_slots.Add(s.id);
			}
		}
	}
	if(target_slots.IsEmpty()) {
		Cout() << "[TrainBoolGroup] no target slots for group '" << resolved_group_key << "'\n";
		return false;
	}

	const AnnLaySlot* ref_slot = lay.FindSlot(target_slots[0]);
	if(!ref_slot)
	        return false;

	Size crop_size = ref_slot->crop_size;
	Vector<String> class_names;
	class_names <<= ref_slot->classes;
	AnnLayMethod method = ref_slot->method;
	String color_mode = ref_slot->color_mode;

	for(const auto& g : lay.groups) {
		if(g.id == resolved_group_key) {
			method = g.method;
			crop_size = g.crop_size;
			class_names <<= g.classes;
			color_mode = g.color_mode;
			break;
		}
	}

	if(crop_size.cx <= 0 || crop_size.cy <= 0)
	        crop_size = Size(32, 32);
	int w = crop_size.cx;
	int h = crop_size.cy;

	// BuildBoolDataset always emits grayscale vectors.
	int depth = 1;

	VectorMap<String, Vector<Vector<double>>> true_by_slot;
	VectorMap<String, Vector<Vector<double>>> false_by_slot;
	
	AnchoredSlotClassifier helper;
	for(const String& slot_id : target_slots) {
		const AnnLaySlot* slot = lay.FindSlot(slot_id);
		if(slot)
			helper.BuildBoolDataset(*slot, lay, annprj_path, images_dir, true_by_slot.Add(slot_id), false_by_slot.Add(slot_id));
	}

	Vector<Vector<double>> true_samples;
	Vector<Vector<double>> false_samples;

	if(balance_by_slot || resolved_group_key == "chip_box_gate") {
		Cout() << "[TrainBoolGroup] Per-slot balancing active for group: " << resolved_group_key << "\n";
		int cap = slot_cap;
		if(cap <= 0) {
			// Auto-calculate cap: use a reasonable percentile or average of true samples?
			// The goal is to avoid seat1 (often many annotations) drowning out seat8 (few).
			// Let's use the average true count as a starting point if no cap provided.
			double avg = 0;
			int n = 0;
			for(int i = 0; i < true_by_slot.GetCount(); i++) {
				if(!true_by_slot[i].IsEmpty()) {
					avg += true_by_slot[i].GetCount();
					n++;
				}
			}
			cap = n > 0 ? (int)(avg / n * 1.5) : 100;
			if(cap < 20) cap = 20; // Minimum reasonable cap
		}
		Cout() << "[TrainBoolGroup] Using slot cap: " << cap << "\n";

		for(int i = 0; i < true_by_slot.GetCount(); i++) {
			const String& sid = true_by_slot.GetKey(i);
			Vector<Vector<double>>& ts = true_by_slot[i];
			Vector<Vector<double>>& fs = false_by_slot[i];
			
			Cout() << Format("  - %-25s: raw_true=%-4d raw_false=%-4d", sid, ts.GetCount(), fs.GetCount());
			
			if(ts.GetCount() > cap) {
				ShuffleSamples(ts);
				ts.SetCount(cap);
			}
			// False samples for this slot also capped to preserve slot weight
			if(fs.GetCount() > cap * 2) {
				ShuffleSamples(fs);
				fs.SetCount(cap * 2);
			}
			
			Cout() << Format(" -> balanced_true=%-4d balanced_false=%-4d\n", ts.GetCount(), fs.GetCount());

			for(auto& s : ts) true_samples.Add(pick(s));
			for(auto& s : fs) false_samples.Add(pick(s));
		}
	}
	else {
		// Legacy merge-all behavior
		for(int i = 0; i < true_by_slot.GetCount(); i++) {
			for(auto& s : true_by_slot[i]) true_samples.Add(pick(s));
			for(auto& s : false_by_slot[i]) false_samples.Add(pick(s));
		}
	}

	if(true_samples.IsEmpty() && false_samples.IsEmpty())
		return false;

	// Balancing logic for bool training (class balance)
	{
		int t_count = true_samples.GetCount();
		int f_count = false_samples.GetCount();
		int f_limit = 2 * t_count;
		if(f_count > f_limit && t_count > 0) {
			Cout() << "[TrainBoolGroup] Class balancing: capping false samples from " << f_count << " to " << f_limit << " (ratio 2:1)\n";
			ShuffleSamples(false_samples);
			false_samples.SetCount(f_limit);
		}
		Cout() << "[TrainBoolGroup] Final counts: true=" << true_samples.GetCount()
		       << " false=" << false_samples.GetCount()
		       << " ratio=" << (double)false_samples.GetCount() / max(1, true_samples.GetCount()) << "\n";
	}

	Vector<Vector<double>> samples;
	Vector<int> labels;
	for(int i = 0; i < true_samples.GetCount(); i++) {
		samples.Add(pick(true_samples[i]));
		labels.Add(1);
	}
	for(int i = 0; i < false_samples.GetCount(); i++) {
		samples.Add(pick(false_samples[i]));
		labels.Add(0);
	}
	if(samples.IsEmpty())
		return false;
	ShufflePaired(samples, labels);

	int cls_count = 2;
	int sample_len = w * h * depth;

	Vector<Vector<double>> valid_samples;
	Vector<int> valid_labels;
	for(int i = 0; i < samples.GetCount(); i++) {
		if(samples[i].GetCount() != sample_len)
			continue;
		int lbl = labels[i];
		if(lbl < 0 || lbl >= cls_count)
			continue;
		Vector<double>& dst = valid_samples.Add();
		dst <<= samples[i];
		valid_labels.Add(lbl);
	}
	if(valid_samples.IsEmpty())
		return false;

	int n = valid_samples.GetCount();
	int val_n = n >= 2 ? max(1, n / 10) : 0;
	int train_n = n - val_n;
	if(train_n <= 0)
		return false;

	String net_json = BuildNetJson(w, h, depth, cls_count);
	ConvNet::Session local_ses;
	ConvNet::Session& ses = ext_ses ? *ext_ses : local_ses;
	// Always rebuild the session for this training run so stale preview/session
	// architectures (e.g. old RGB net) cannot mismatch current sample depth.
	ses.Clear();
	if(!ses.MakeLayers(net_json))
		return false;

	if(class_names.GetCount() < cls_count) {
		class_names.SetCount(cls_count);
		if(class_names[0].IsEmpty()) class_names[0] = "false";
		if(class_names[1].IsEmpty()) class_names[1] = "true";
	}

	ConvNet::SessionData& sd = ses.Data();
	sd.BeginData(cls_count, train_n, w, h, depth, val_n);
	for(int cls = 0; cls < cls_count; cls++)
		sd.SetClass(cls, class_names[cls]);
	for(int i = 0; i < train_n; i++) {
		sd.Get(i) <<= valid_samples[i];
		sd.SetLabel(i, valid_labels[i]);
	}
	for(int i = 0; i < val_n; i++) {
		sd.GetTest(i) <<= valid_samples[train_n + i];
		sd.SetTestLabel(i, valid_labels[train_n + i]);
	}
	sd.EndData();

	int total_epochs = max(1, max_epochs);
	ses.SetTestPredict(true);
	ses.SetPredictInterval(0);
	ses.SetStepCallbackInterval(0);
	ses.SetIterationCallbackInterval(0);
	ses.SetMaxTrainIters(total_epochs);
	ses.TrainBegin();
	if(debug) {
		Cout() << "[TrainBoolGroup] starting training for " << resolved_group_key << " samples=" << n << " epochs=" << total_epochs << "\n";
		
		if(n > 0) {
			const Vector<double>& first_sample = sd.Get(0);
			double s_sum = 0;
			for(double v : first_sample) s_sum += v;
			Cout() << "  DEBUG: First sample avg value: " << (s_sum / first_sample.GetCount()) << "\n";
		}
	}

	int loss_streak = 0;
	const double kLossStop = 0.001;
	const int kLossPatience = 8;
	for(int i = 0; i < total_epochs; i++) {
		if(!ses.IsTraining()) break;
		ses.TrainIteration();
		double loss = ses.GetLossAverage();
		double acc = ses.GetValidationAccuracyAverage();
		
		if(debug && !IsFin(loss)) {
			Cout() << "  DEBUG: Epoch " << (i+1) << " NaN detected! Step=" << ses.GetStepNum() << " Loss=" << loss << " Acc=" << acc << "\n";
			// Try to inspect first layer weights
			if(ses.GetLayerCount() > 0) {
				::ConvNet::Volume& w = ses.GetLayer(1).output_activation;
				Cout() << "  DEBUG: Layer 1 Output[0]=" << w.Get(0) << "\n";
			}
		}
		
		if(WhenProgress)
			WhenProgress(resolved_group_key, i + 1, loss, acc);
		
		if(!ignore_auto_stop) {
			if(loss < kLossStop)
				loss_streak++;
			else
				loss_streak = 0;
			if(loss_streak >= kLossPatience) {
				Cout() << "[TrainBoolGroup] early-stop at epoch " << (i + 1)
				       << " (loss=" << loss << ")\n";
				break;
			}
		}
	}
	ses.TrainEnd();

	// Embed weights into all bool slots in group.
	if(mdl) {
		for(const String& slot_id : target_slots)
			helper.EmbedWeights(slot_id, *mdl, ses, net_json);
			
		// Phase 6: also embed into the group key itself
		if(!resolved_group_key.IsEmpty())
			helper.EmbedWeights(resolved_group_key, *mdl, ses, net_json);
	}
	return true;
}

bool AnchoredSlotClassifier::TrainFromCropsDir(AnnLay& lay,
                                               const String& crops_dir,
                                               const String& slot_key,
                                               int max_epochs,
                                               Event<String,int,double,double> WhenProgress,
                                               ConvNet::Session* ext_ses,
                                               AnnMdl* mdl,
                                               bool balance_by_slot,
                                               int slot_cap,
                                               bool ignore_auto_stop,
                                               bool debug) {
	Cout() << "[TrainFromCropsDir] begin crops='" << crops_dir
	       << "' slot_key='" << slot_key
	       << "' max_epochs=" << max_epochs << "\n";
	if(!DirectoryExists(crops_dir)) {
		Cout() << "[TrainFromCropsDir] fail: crops dir does not exist\n";
		return false;
	}

	// Resolve target slots
	Vector<String> target_slots;
	VectorMap<String, Vector<String>> groups = GetSlotGroups(lay);
	Cout() << "DEBUG: lay.groups.count=" << lay.groups.GetCount() << "\n";
	String resolved_slot_key = slot_key;
	if(resolved_slot_key.IsEmpty()) {
		for(int g = 0; g < groups.GetCount() && resolved_slot_key.IsEmpty(); g++) {
			resolved_slot_key = groups.GetKey(g);
		}
	}

	// Phase 6: Resolve crops path by checking if slot-specific subdirectory exists.
	// This allows passing the root pass1 crops dir.
	String crops_path = crops_dir;
	if(!resolved_slot_key.IsEmpty()) {
		String gdir = resolved_slot_key;
		gdir.Replace("#", "_");
		String sub = AppendFileName(crops_dir, gdir);
		if(DirectoryExists(sub)) {
			crops_path = sub;
			Cout() << "[TrainFromCropsDir] using resolved crops path: " << crops_path << "\n";
		} else {
			Cout() << "[TrainFromCropsDir] sub-directory NOT FOUND: " << sub << "\n";
		}
	}

	int qg = groups.Find(resolved_slot_key);
	if(qg >= 0) {
		target_slots <<= groups[qg];
	} else {
		if(lay.FindSlot(resolved_slot_key))
			target_slots.Add(resolved_slot_key);
	}

	if(target_slots.IsEmpty())
	{
		Cout() << "[TrainFromCropsDir] fail: no target slots for resolved key '" << resolved_slot_key << "'\n";
		return false;
	}
	Cout() << "[TrainFromCropsDir] resolved_slot_key='" << resolved_slot_key
	       << "' target_slots=" << target_slots.GetCount() << "\n";
	for(int i = 0; i < target_slots.GetCount(); i++)
		Cout() << "[TrainFromCropsDir]   target[" << i << "]=" << target_slots[i] << "\n";

	// Use the first target slot to define class map and crop size.
	const AnnLaySlot* ref_slot = lay.FindSlot(target_slots[0]);
	if(!ref_slot) {
		Cout() << "[TrainFromCropsDir] fail: ref slot not found for '" << target_slots[0] << "'\n";
		return false;
	}

	Size crop_size = ref_slot->crop_size;
	Vector<String> class_names;
	class_names <<= ref_slot->classes;
	AnnLayMethod method = ref_slot->method;
	String color_mode = ref_slot->color_mode;

	for(const auto& g : lay.groups) {
		if(g.id == resolved_slot_key) {
			method = g.method;
			crop_size = g.crop_size;
			class_names <<= g.classes;
			color_mode = g.color_mode;
			break;
		}
	}

	if(crop_size.cx <= 0 || crop_size.cy <= 0)
		crop_size = Size(32, 32);

	int w = crop_size.cx;
	int h = crop_size.cy;
	int depth = color_mode == "color" ? 3 : 1;
	bool is_bool = (method == ANNLAY_CLASSIFIER_BOOL);

	VectorMap<String, int> class_map;
	if(is_bool) {
		class_map.Add("false", 0);
		class_map.Add("true", 1);
		class_names.Clear();
		class_names.Add("false");
		class_names.Add("true");
	}
	else {
		for(int i = 0; i < class_names.GetCount(); i++)
			class_map.GetAdd(TrimLower(class_names[i])) = i;
	}

	if(resolved_slot_key.EndsWith("#offset")) {
		// Build class map dynamically from subdirectory names (e.g. dx=+2,dy=-1)
		struct OffItem : Moveable<OffItem> {
			String name;
			int dx = 0, dy = 0;
			bool operator<(const OffItem& b) const {
				if(dx != b.dx) return dx < b.dx;
				return dy < b.dy;
			}
		};
		Vector<OffItem> items;
		FindFile ff;
		if(ff.Search(AppendFileName(crops_path, "*"))) {
			do {
				if(ff.IsFolder()) {
					String n = ff.GetName();
					if(n != "." && n != "..") {
						OffItem& it = items.Add();
						it.name = n;
						// Parse "dx=%+d,dy=%+d" or "dx=%d,dy=%d"
						Cout() << "[TrainFromCropsDir] parsing offset dir: " << n << "\n";
						int dx_p = n.Find("dx=");
						int dy_p = n.Find("dy=");
						if(dx_p >= 0 && dy_p >= 0) {
							it.dx = StrInt(n.Mid(dx_p + 3));
							it.dy = StrInt(n.Mid(dy_p + 3));
						}
						else {
							// Fallback to lexical-ish if parsing fails
							it.dx = 999;
							it.dy = 999;
						}
					}
				}
			} while(ff.Next());
		}
		Sort(items);
		for(int i = 0; i < items.GetCount(); i++) {
			class_map.Add(items[i].name, i);
			class_names.Add(items[i].name);
		}
		// Offset export stores grayscale PNGs. Switch to grayscale input depth.
		depth = 1;
	}
	else if(resolved_slot_key.EndsWith("#offset_x") || resolved_slot_key.EndsWith("#offset_y")) {
		// Build class map dynamically from subdirectory names (signed integers)
		Vector<int> vals;
		FindFile ff;
		if(ff.Search(AppendFileName(crops_path, "*"))) {
			do {
				if(ff.IsFolder()) {
					String n = ff.GetName();
					if(n != "." && n != "..") {
						int v = StrInt(n);
						if(!IsNull(v))
							vals.Add(v);
					}
				}
			} while(ff.Next());
		}
		Sort(vals);
		for(int i = 0; i < vals.GetCount(); i++) {
			String cls = Format("%+d", vals[i]);
			class_map.Add(cls, i);
			class_names.Add(cls);
		}
		// Offset export stores grayscale PNGs. Switch to grayscale input depth.
		depth = 1;
	}
	else if(resolved_slot_key.EndsWith("#zoom")) {
		// Build class map dynamically from subdirectory names (z0.80, z0.85, etc)
		Vector<String> subdirs;
		FindFile ff;
		if(ff.Search(AppendFileName(crops_path, "*"))) {
			do {
				if(ff.IsFolder()) {
					String n = ff.GetName();
					if(n.StartsWith("z"))
						subdirs.Add(n);
				}
			} while(ff.Next());
		}
		Sort(subdirs);
		for(int i = 0; i < subdirs.GetCount(); i++) {
			class_map.Add(subdirs[i], i);
			class_names.Add(subdirs[i]);
		}
		// Zoom export stores grayscale PNGs. Switch to grayscale input depth.
		depth = 1;
	}
	else {
		// Default: use class_names already resolved from group/slot metadata
		for(int i = 0; i < class_names.GetCount(); i++)
			class_map.GetAdd(TrimLower(class_names[i])) = i;
	}

	int cls_count = max(2, class_names.GetCount());
	Cout() << "[TrainFromCropsDir] ref_slot='" << ref_slot->id
	       << "' method='" << AnnLayMethodToString(method)
	       << "' crop=" << w << "x" << h
	       << " depth=" << depth
	       << " classes=" << cls_count << "\n";

	// Load crops from subdirectories
	VectorMap<String, Vector<Vector<double>>> true_by_slot;
	VectorMap<String, Vector<Vector<double>>> false_by_slot;
	Vector<Vector<double>> generic_samples;
	Vector<int> generic_labels;

	int ignored_dirs = 0;
	Vector<int> samples_per_class;
	samples_per_class.SetCount(cls_count, 0);

	FindFile ff;
	if(ff.Search(AppendFileName(crops_path, "*"))) {
		do {
			if(!ff.IsFolder())
				continue;
			String dname = ff.GetName();
			if(dname == "." || dname == "..")
				continue;
			String cls_name = TrimLower(dname);
			int q = class_map.Find(cls_name);
			if(q < 0) {
				ignored_dirs++;
				continue;
			}
			int label = class_map[q];
			if(label >= 0 && label < class_names.GetCount())
				class_names[label] = dname;
			String subdir = AppendFileName(crops_path, dname);

			FindFile imgf;
			if(imgf.Search(AppendFileName(subdir, "*"))) {
				do {
					if(!imgf.IsFile())
						continue;
					String fname = imgf.GetName();
					String ext = ToLower(GetFileExt(fname));
					if(ext != ".png" && ext != ".jpg" && ext != ".jpeg")
						continue;
					
					Image img = StreamRaster::LoadFileAny(imgf.GetPath());
					if(img.IsEmpty())
						continue;
					Image crop = Rescale(img, crop_size);
					int len = w * h * depth;
					Vector<double> sample;
					sample.SetCount(len, 0.0);
					if(depth == 1) {
						Image gray = Grayscale(crop);
						for(int y = 0; y < h; y++) {
							const RGBA* row = gray[y];
							for(int x = 0; x < w; x++) {
								int base = (y * w + x);
								sample[base] = row[x].r / 255.0;
							}
						}
					}
					else {
						for(int y = 0; y < h; y++) {
							const RGBA* row = crop[y];
							for(int x = 0; x < w; x++) {
								int base = (y * w + x) * 3;
								sample[base + 0] = row[x].r / 255.0;
								sample[base + 1] = row[x].g / 255.0;
								sample[base + 2] = row[x].b / 255.0;
							}
						}
					}
					
					if(is_bool && cls_count == 2) {
						// Robustly extract slot_id from filename: "<slot_id>_<stem>.png"
						// Uses longest-prefix match against known target_slots.
						String matched_slot;
						for(const String& tsid : target_slots) {
							if(fname.StartsWith(tsid + "_")) {
								if(tsid.GetCount() > matched_slot.GetCount())
									matched_slot = tsid;
							}
						}

						if(!matched_slot.IsEmpty()) {
							if(label == 1) true_by_slot.GetAdd(matched_slot).Add(pick(sample));
							else false_by_slot.GetAdd(matched_slot).Add(pick(sample));
						}
						else {
							// Fallback to generic balancing if no prefix match (e.g. legacy files)
							generic_samples.Add(pick(sample));
							generic_labels.Add(label);
						}
					}
					else {
						generic_samples.Add(pick(sample));
						generic_labels.Add(label);
					}
					
					if(label >= 0 && label < samples_per_class.GetCount())
						samples_per_class[label]++;
				}
				while(imgf.Next());
			}
		}
		while(ff.Next());
	}

	Vector<Vector<double>> samples;
	Vector<int> labels;

	if(is_bool && cls_count == 2 && (balance_by_slot || resolved_slot_key == "chip_box_gate")) {
		Cout() << "[TrainFromCropsDir] Per-slot balancing active for group: " << resolved_slot_key << "\n";
		int cap = slot_cap;
		if(cap <= 0) {
			double avg = 0;
			int n = 0;
			for(int i = 0; i < true_by_slot.GetCount(); i++) {
				if(!true_by_slot[i].IsEmpty()) {
					avg += true_by_slot[i].GetCount();
					n++;
				}
			}
			cap = n > 0 ? (int)(avg / n * 1.5) : 100;
			if(cap < 20) cap = 20;
		}
		Cout() << "[TrainFromCropsDir] Using slot cap: " << cap << "\n";

		Index<String> all_slots;
		for(int i = 0; i < true_by_slot.GetCount(); i++) all_slots.Add(true_by_slot.GetKey(i));
		for(int i = 0; i < false_by_slot.GetCount(); i++) all_slots.Add(false_by_slot.GetKey(i));

		for(const String& sid : all_slots) {
			Vector<Vector<double>>& ts = true_by_slot.GetAdd(sid);
			Vector<Vector<double>>& fs = false_by_slot.GetAdd(sid);
			
			Cout() << Format("  - %-25s: raw_true=%-4d raw_false=%-4d", sid, ts.GetCount(), fs.GetCount());
			
			if(ts.GetCount() > cap) {
				ShuffleSamples(ts);
				ts.SetCount(cap);
			}
			if(fs.GetCount() > cap * 2) {
				ShuffleSamples(fs);
				fs.SetCount(cap * 2);
			}
			
			Cout() << Format(" -> balanced_true=%-4d balanced_false=%-4d\n", ts.GetCount(), fs.GetCount());

			for(auto& s : ts) { samples.Add(pick(s)); labels.Add(1); }
			for(auto& s : fs) { samples.Add(pick(s)); labels.Add(0); }
		}
	}
	else {
		// Non-balanced path
		if(is_bool && cls_count == 2) {
			for(int i = 0; i < true_by_slot.GetCount(); i++) {
				for(auto& s : true_by_slot[i]) { samples.Add(pick(s)); labels.Add(1); }
			}
			for(int i = 0; i < false_by_slot.GetCount(); i++) {
				for(auto& s : false_by_slot[i]) { samples.Add(pick(s)); labels.Add(0); }
			}
		}
		else {
			samples = pick(generic_samples);
			labels = pick(generic_labels);
		}
	}

	Cout() << "[TrainFromCropsDir] final samples=" << samples.GetCount() << "\n";

	if(samples.IsEmpty()) {
		Cout() << "[TrainFromCropsDir] fail: no samples\n";
		return false;
	}

	// Class balancing (true:false)
	if(is_bool && cls_count == 2) {
		Vector<Vector<double>> true_samples;
		Vector<Vector<double>> false_samples;
		for(int i = 0; i < samples.GetCount(); i++) {
			if(labels[i] == 1) true_samples.Add(pick(samples[i]));
			else false_samples.Add(pick(samples[i]));
		}
		int t_count = true_samples.GetCount();
		int f_count = false_samples.GetCount();
		int f_limit = 2 * t_count;
		if(f_count > f_limit && t_count > 0) {
			Cout() << "[TrainFromCropsDir] Class balancing: capping false samples from " << f_count << " to " << f_limit << " (ratio 2:1)\n";
			ShuffleSamples(false_samples);
			false_samples.SetCount(f_limit);
		}
		Cout() << "[TrainFromCropsDir] Final counts: true=" << true_samples.GetCount()
		       << " false=" << false_samples.GetCount()
		       << " ratio=" << (double)false_samples.GetCount() / max(1, true_samples.GetCount()) << "\n";
		
		samples.Clear();
		labels.Clear();
		for(int i = 0; i < true_samples.GetCount(); i++) {
			samples.Add(pick(true_samples[i]));
			labels.Add(1);
		}
		for(int i = 0; i < false_samples.GetCount(); i++) {
			samples.Add(pick(false_samples[i]));
			labels.Add(0);
		}
	}

	ShufflePaired(samples, labels);

	// Filter valid samples
	int sample_len = w * h * depth;
	Vector<Vector<double>> valid_samples;
	Vector<int> valid_labels;
	Vector<int> final_samples_per_class;
	final_samples_per_class.SetCount(cls_count, 0);

	for(int i = 0; i < samples.GetCount(); i++) {
		if(samples[i].GetCount() != sample_len)
			continue;
		int lbl = labels[i];
		if(lbl < 0 || lbl >= cls_count)
			continue;
		Vector<double>& dst = valid_samples.Add();
		dst <<= samples[i];
		valid_labels.Add(lbl);
		final_samples_per_class[lbl]++;
	}

	Cout() << "[TrainFromCropsDir] final class distribution:\n";
	int classes_with_samples = 0;
	for(int i = 0; i < cls_count; i++) {
		Cout() << "  - class " << i << " (" << class_names[i] << "): " << final_samples_per_class[i] << " samples\n";
		if(final_samples_per_class[i] > 0)
			classes_with_samples++;
	}

	if(valid_samples.IsEmpty()) {
		Cout() << "[TrainFromCropsDir] fail: no valid samples after shape/label filtering (sample_len expected=" << sample_len << ")\n";
		return false;
	}

	if(classes_with_samples < 2 && cls_count >= 2) {
		Cout() << "[TrainFromCropsDir] fail: less than 2 classes have samples. Neural network needs at least two classes to learn anything.\n";
		return false;
	}

	int n = valid_samples.GetCount();
	int val_n = n >= 2 ? max(1, n / 10) : 0;
	int train_n = n - val_n;
	if(train_n <= 0) {
		Cout() << "[TrainFromCropsDir] fail: train_n <= 0 (n=" << n << ", val_n=" << val_n << ")\n";
		return false;
	}
	Cout() << "[TrainFromCropsDir] split train_n=" << train_n << " val_n=" << val_n << "\n";

	String net_json = BuildNetJson(w, h, depth, cls_count);
	ConvNet::Session local_ses;
	ConvNet::Session& ses = ext_ses ? *ext_ses : local_ses;
	// Always rebuild the session for this training run so stale preview/session
	// architectures (e.g. old RGB net) cannot mismatch current sample depth.
	ses.Clear();
	if(!ses.MakeLayers(net_json)) {
		Cout() << "[TrainFromCropsDir] fail: ses.MakeLayers returned false (json length=" << net_json.GetCount() << ")\n";
		return false;
	}

	ConvNet::SessionData& sd = ses.Data();
	sd.BeginData(cls_count, train_n, w, h, depth, val_n);
	for(int cls = 0; cls < cls_count; cls++) {
		String cls_name = class_names[cls];
		if(cls_name.IsEmpty())
			cls_name = Format("class_%d", cls);
		sd.SetClass(cls, cls_name);
	}
	for(int i = 0; i < train_n; i++) {
		sd.Get(i) <<= valid_samples[i];
		sd.SetLabel(i, valid_labels[i]);
	}
	for(int i = 0; i < val_n; i++) {
		sd.GetTest(i) <<= valid_samples[train_n + i];
		sd.SetTestLabel(i, valid_labels[train_n + i]);
	}
	sd.EndData();

	// Enable validation pass to report accuracy.
	ses.SetTestPredict(true);
	ses.SetPredictInterval(0);
	ses.SetStepCallbackInterval(0);
	ses.SetIterationCallbackInterval(0);
	ses.SetMaxTrainIters(max(1, max_epochs));
	ses.TrainBegin();
	double last_val_acc = 0.0;
	int loss_streak = 0;
	int epochs_done = 0;
	const double kLossStop = is_bool ? 0.001 : 0.002;
	const int kLossPatience = is_bool ? 8 : 4;
	for(int i = 0; i < max_epochs; i++) {
		if(!ses.IsTraining()) break;
		ses.TrainIteration();
		double loss = ses.GetLossAverage();
		last_val_acc = ses.GetTrainingAccuracyAverage();
		
		if(debug && !IsFin(loss)) {
			Cout() << "  DEBUG: Epoch " << (i+1) << " NaN detected! Step=" << ses.GetStepNum() << " Loss=" << loss << " Acc=" << last_val_acc << "\n";
			if(ses.GetLayerCount() > 0) {
				::ConvNet::Volume& w = ses.GetLayer(1).output_activation;
				Cout() << "  DEBUG: Layer 1 Output[0]=" << w.Get(0) << "\n";
			}
		}

		epochs_done = i + 1;
		if(WhenProgress)
			WhenProgress(resolved_slot_key, epochs_done, loss, last_val_acc);
		if(!ignore_auto_stop) {
			if(loss < kLossStop)
				loss_streak++;
			else
				loss_streak = 0;
			if(loss_streak >= kLossPatience) {
				Cout() << "[TrainFromCropsDir] early-stop at epoch " << epochs_done
				       << " (loss=" << loss << ")\n";
				break;
			}
		}
	}
	ses.TrainEnd();

	if(WhenProgress)
		WhenProgress(target_slots[0], epochs_done, ses.GetLossAverage(), ses.GetTrainingAccuracyAverage());

	// Embed weights into all target slots
	if(mdl) {
		AnchoredSlotClassifier helper;
		String suffix;
		int hash = resolved_slot_key.Find('#');
		if(hash >= 0)
			suffix = resolved_slot_key.Mid(hash);

		for(const String& slot_id : target_slots)
			helper.EmbedWeights(slot_id + suffix, *mdl, ses, net_json);
			
		// Phase 6: also embed into the group key itself if it differs from slot IDs
		// This is required by AnchoredSlotRecognizer::Load which looks for group sessions.
		if(!resolved_slot_key.IsEmpty() && target_slots.GetCount() > 0) {
			bool key_is_slot = false;
			for(const auto& sid : target_slots) if(sid == resolved_slot_key) key_is_slot = true;
			if(!key_is_slot)
				helper.EmbedWeights(resolved_slot_key, *mdl, ses, net_json);
		}
	}
	Cout() << "[TrainFromCropsDir] success: final loss=" << ses.GetLossAverage()
	       << " val_acc=n/a\n";

	return true;
}

bool AnchoredSlotClassifier::RunTrainingJob(AnnLay& lay,
                                            const TrainingJobRequest& request,
                                            TrainingJobResult& result,
                                            Event<String,int,double,double> WhenProgress,
                                            ConvNet::Session* ses,
                                            AnnMdl* mdl) {
	result.groups.Clear();

	Vector<String> keys = clone(request.group_keys);
	if(keys.IsEmpty()) {
		VectorMap<String, Vector<String>> groups = GetSlotGroups(lay);
		for(int i = 0; i < groups.GetCount(); i++)
			keys.Add(groups.GetKey(i));
	}
	if(keys.IsEmpty()) {
		result.ok = false;
		return false;
	}

	Cout() << "[RunTrainingJob] source_mode="
	       << (request.source_mode == TRAIN_SOURCE_CROPS ? "crops" : "annprj_bool")
	       << " groups=" << keys.GetCount()
	       << " epochs=" << request.max_epochs << "\n";

	bool all_ok = true;
	for(int gi = 0; gi < keys.GetCount(); gi++) {
		const String& key = keys[gi];
		TrainingGroupResult& gr = result.groups.Add();
		gr.group_key = key;

		Cout() << "\n=== Training group " << (gi + 1) << "/" << keys.GetCount()
		       << ": " << key << " ===\n";

		auto TrackProgress = [&](String slot, int epoch, double loss, double acc) {
			double safe_loss = IsFin(loss) ? loss : 0.0;
			double safe_acc = IsFin(acc) ? acc : 0.0;
			gr.epochs_done = max(gr.epochs_done, epoch);
			int idx = max(0, epoch - 1);
			while(gr.loss_history.GetCount() <= idx) {
				gr.loss_history.Add(safe_loss);
				gr.val_acc_history.Add(safe_acc);
			}
			gr.loss_history[idx] = safe_loss;
			gr.val_acc_history[idx] = safe_acc;
			if(WhenProgress)
				WhenProgress(slot, epoch, safe_loss, safe_acc);
		};

		bool ok = false;
		if(request.source_mode == TRAIN_SOURCE_CROPS) {
			if(request.crops_dir.IsEmpty()) {
				gr.message = "missing crops_dir";
				ok = false;
			}
			else {
				ok = TrainFromCropsDir(lay, request.crops_dir, key, request.max_epochs,
				                       TrackProgress,
				                       ses, mdl,
				                       request.balance_by_slot,
				                       request.slot_cap,
				                       request.ignore_auto_stop,
				                       request.debug);
			}
		}
		else {
			if(request.annprj_path.IsEmpty() || request.images_dir.IsEmpty()) {
				gr.message = "missing annprj/images path";
				ok = false;
			}
			else {
				ok = TrainBoolGroup(lay, key, request.annprj_path, request.images_dir, request.max_epochs,
				                    TrackProgress,
				                    ses, mdl,
				                    request.balance_by_slot,
				                    request.slot_cap,
				                    request.ignore_auto_stop,
				                    request.debug);
			}
		}

		gr.ok = ok;
		if(gr.message.IsEmpty())
			gr.message = ok ? "ok" : "failed";

		if(!ok) {
			Cout() << "[RunTrainingJob] group failed: " << key << "\n";
			all_ok = false;
		}
	}

	result.ok = all_ok;
	return all_ok;
}


Image AnchoredSlotClassifier::EqualizeHistogram(const Image& src) {
	if(src.IsEmpty()) return src;
	
	int histogram[256];
	memset(histogram, 0, sizeof(histogram));
	
	Size sz = src.GetSize();
	for(int y = 0; y < sz.cy; y++) {
		const RGBA* s = src[y];
		for(int x = 0; x < sz.cx; x++) {
			byte g = (byte)(s[x].r * 299 / 1000 + s[x].g * 587 / 1000 + s[x].b * 114 / 1000);
			histogram[g]++;
		}
	}
	
	int total = sz.cx * sz.cy;
	int cdf[256];
	cdf[0] = histogram[0];
	for(int i = 1; i < 256; i++) cdf[i] = cdf[i - 1] + histogram[i];
	
	int cdf_min = 0;
	for(int i = 0; i < 256; i++) {
		if(cdf[i] > 0) {
			cdf_min = cdf[i];
			break;
		}
	}
	
	if(total == cdf_min) return src; // Single color image
	
	byte lut[256];
	for(int i = 0; i < 256; i++) {
		lut[i] = (byte)clamp((cdf[i] - cdf_min) * 255 / (total - cdf_min), 0, 255);
	}
	
	ImageBuffer ib(sz);
	for(int y = 0; y < sz.cy; y++) {
		const RGBA* s = src[y];
		RGBA* d = ib[y];
		for(int x = 0; x < sz.cx; x++) {
			byte g = (byte)(s[x].r * 299 / 1000 + s[x].g * 587 / 1000 + s[x].b * 114 / 1000);
			byte ng = lut[g];
			if(g > 0) {
				d[x].r = (byte)clamp(s[x].r * ng / g, 0, 255);
				d[x].g = (byte)clamp(s[x].g * ng / g, 0, 255);
				d[x].b = (byte)clamp(s[x].b * ng / g, 0, 255);
			}
			else {
				d[x].r = d[x].g = d[x].b = ng;
			}
			d[x].a = s[x].a;
		}
	}
	
	return ib;
}

Image AnchoredSlotClassifier::LinearContrastStretching(const Image& src) {
	if(src.IsEmpty()) return src;
	
	Size sz = src.GetSize();
	int min_g = 255;
	int max_g = 0;
	
	for(int y = 0; y < sz.cy; y++) {
		const RGBA* s = src[y];
		for(int x = 0; x < sz.cx; x++) {
			int g = (s[x].r * 299 + s[x].g * 587 + s[x].b * 114) / 1000;
			if(g < min_g) min_g = g;
			if(g > max_g) max_g = g;
		}
	}
	
	if(max_g <= min_g) return src;
	
	ImageBuffer ib(sz);
	double scale = 255.0 / (max_g - min_g);
	
	for(int y = 0; y < sz.cy; y++) {
		const RGBA* s = src[y];
		RGBA* d = ib[y];
		for(int x = 0; x < sz.cx; x++) {
			int g = (s[x].r * 299 + s[x].g * 587 + s[x].b * 114) / 1000;
			int ng = (int)clamp((g - min_g) * scale, 0.0, 255.0);
			
			if(g > 0) {
				d[x].r = (byte)clamp(s[x].r * ng / g, 0, 255);
				d[x].g = (byte)clamp(s[x].g * ng / g, 0, 255);
				d[x].b = (byte)clamp(s[x].b * ng / g, 0, 255);
			}
			else {
				d[x].r = d[x].g = d[x].b = (byte)ng;
			}
			d[x].a = s[x].a;
		}
	}
	
	return ib;
}

END_UPP_NAMESPACE
