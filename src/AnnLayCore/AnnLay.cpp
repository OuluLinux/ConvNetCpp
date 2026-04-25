#include "AnnLay.h"
#include "AnchoredSlotClassifier.h"

NAMESPACE_UPP

namespace {

template <class T>
bool SaveAsJSONLocal(const T& obj, const String& path, bool pretty = true) {
	try {
		String json = StoreAsJson(obj, pretty);
		return SaveFile(path, json);
	}
	catch(...) {
		return false;
	}
}

template <class T>
bool LoadFromJSONLocal(T& obj, const String& path) {
	try {
		String json = LoadFile(path);
		if(json.IsEmpty())
			return false;
		Value v = ParseJSON(json);
		if(IsNull(v))
			return false;
		LoadFromJsonValue(obj, v);
		return true;
	}
	catch(...) {
		return false;
	}
}

ValueMap NormalizeLegacySlotIndexMapField(const Value& src) {
	ValueMap out;
	if(src.GetType() == VALUEMAP_V) {
		ValueMap vm = src;
		for(int i = 0; i < vm.GetCount(); i++) {
			String k = vm.GetKey(i).ToString();
			if(k.IsEmpty())
				continue;
			out.Add(k, vm.GetValue(i));
		}
		return out;
	}
	if(src.GetType() != VALUEARRAY_V)
		return out;

	ValueArray arr = src;
	for(int i = 0; i < arr.GetCount(); i++) {
		Value item = arr[i];
		if(item.GetType() != VALUEMAP_V)
			continue;
		ValueMap kv = item;

		Value keyv;
		Value valv;
		for(int j = 0; j < kv.GetCount(); j++) {
			String kk = kv.GetKey(j);
			if(kk == "key") keyv = kv.GetValue(j);
			else if(kk == "value") valv = kv.GetValue(j);
		}

		String key;
		if(IsString(keyv)) key = keyv;
		else if(keyv.GetType() == VALUEMAP_V) {
			ValueMap km = keyv;
			for(int j = 0; j < km.GetCount(); j++) {
				if(km.GetKey(j) == "value") {
					key = km.GetValue(j).ToString();
					break;
				}
			}
		}
		if(key.IsEmpty())
			continue;

		String val;
		if(IsString(valv)) val = valv;
		else if(valv.GetType() == VALUEMAP_V) {
			ValueMap vm = valv;
			for(int j = 0; j < vm.GetCount(); j++) {
				if(vm.GetKey(j) == "value") {
					val = vm.GetValue(j).ToString();
					break;
				}
			}
		}
		out.Add(key, val);
	}
	return out;
}

Value NormalizeLegacyAnnLayValue(const Value& in) {
	if(!IsValueMap(in))
		return in;

	ValueMap root = in;
	ValueMap out_root;
	for(int i = 0; i < root.GetCount(); i++) {
		String k = root.GetKey(i);
		Value  v = root.GetValue(i);

		if(k != "slots" || !IsValueArray(v)) {
			out_root.Add(k, v);
			continue;
		}

		ValueArray slots = v;
		ValueArray out_slots;
		for(int si = 0; si < slots.GetCount(); si++) {
			Value sv = slots[si];
			if(!IsValueMap(sv)) {
				out_slots.Add(sv);
				continue;
			}

			ValueMap sm = sv;
			ValueMap out_sm;
			for(int sj = 0; sj < sm.GetCount(); sj++) {
				String sk = sm.GetKey(sj);
				Value  fv = sm.GetValue(sj);
				if(sk == "sub_groups" || sk == "sub_heads")
					out_sm.Add(sk, NormalizeLegacySlotIndexMapField(fv));
				else
					out_sm.Add(sk, fv);
			}
			out_slots.Add(out_sm);
		}
		out_root.Add(k, out_slots);
	}

	return out_root;
}

bool ReplaceArrayFieldWithObject(String& json, const String& field_name) {
	String needle = "\"" + field_name + "\"";
	bool changed = false;
	int pos = 0;
	while((pos = json.Find(needle, pos)) >= 0) {
		int i = pos + needle.GetCount();
		while(i < json.GetCount() && IsSpace((int)json[i])) i++;
		if(i >= json.GetCount() || json[i] != ':') { pos = i; continue; }
		i++;
		while(i < json.GetCount() && IsSpace((int)json[i])) i++;
		if(i >= json.GetCount() || json[i] != '[') { pos = i; continue; }

		int start = i;
		int depth = 0;
		bool in_str = false;
		bool esc = false;
		for(; i < json.GetCount(); i++) {
			char c = json[i];
			if(in_str) {
				if(esc) esc = false;
				else if(c == '\\') esc = true;
				else if(c == '"') in_str = false;
				continue;
			}
			if(c == '"') { in_str = true; continue; }
			if(c == '[') depth++;
			else if(c == ']') {
				depth--;
				if(depth == 0) {
					json.Remove(start, i - start + 1);
					json.Insert(start, "{}");
					changed = true;
					pos = start + 2;
					break;
				}
			}
		}
		if(i >= json.GetCount())
			break;
	}
	return changed;
}

struct AnchorSample {
	double cx = 0;
	double cy = 0;
	double w = 0;
	double h = 0;
};

AnnLayAnchor BuildAnchorFromSamples(const Vector<AnchorSample>& samples) {
	AnnLayAnchor out;
	int n = samples.GetCount();
	if(n <= 0)
		return out;

	double sum_cx = 0, sum_cy = 0, sum_w = 0, sum_h = 0;
	for(const AnchorSample& s : samples) {
		sum_cx += s.cx;
		sum_cy += s.cy;
		sum_w += s.w;
		sum_h += s.h;
	}
	out.cx = sum_cx / n;
	out.cy = sum_cy / n;
	out.w = sum_w / n;
	out.h = sum_h / n;
	out.n_annotations = n;

	double var_cx = 0, var_cy = 0;
	for(const AnchorSample& s : samples) {
		var_cx += (s.cx - out.cx) * (s.cx - out.cx);
		var_cy += (s.cy - out.cy) * (s.cy - out.cy);
	}
	out.std_cx = sqrt(var_cx / n);
	out.std_cy = sqrt(var_cy / n);
	return out;
}

double Dist(double x1, double y1, double x2, double y2) {
	double dx = x1 - x2;
	double dy = y1 - y2;
	return sqrt(dx * dx + dy * dy);
}

Vector<AnnLayAnchor> BuildGreedyCandidates(const Vector<AnchorSample>& samples, double threshold) {
	struct Cluster : Moveable<Cluster> {
		Vector<AnchorSample> members;
		double cx = 0;
		double cy = 0;
	};

	Vector<Cluster> clusters;
	for(const AnchorSample& s : samples) {
		int best = -1;
		double best_dist = threshold;
		for(int i = 0; i < clusters.GetCount(); i++) {
			double d = Dist(s.cx, s.cy, clusters[i].cx, clusters[i].cy);
			if(d <= best_dist) {
				best_dist = d;
				best = i;
			}
		}

		if(best < 0) {
			Cluster& c = clusters.Add();
			c.members.Add(s);
			c.cx = s.cx;
			c.cy = s.cy;
		}
		else {
			Cluster& c = clusters[best];
			c.members.Add(s);
			double inv = 1.0 / c.members.GetCount();
			c.cx = c.cx + (s.cx - c.cx) * inv;
			c.cy = c.cy + (s.cy - c.cy) * inv;
		}
	}

	Vector<AnnLayAnchor> out;
	for(const Cluster& c : clusters)
		out.Add(BuildAnchorFromSamples(c.members));
	return out;
}

}

void AnnLayAnchor::Jsonize(JsonIO& jio) {
	jio("cx", cx)("cy", cy)("w", w)("h", h)
	   ("std_cx", std_cx)("std_cy", std_cy)
	   ("n_annotations", n_annotations);
}

void BoolGateMetrics::Jsonize(JsonIO& jio) {
	jio("slot_id", slot_id)("tp", tp)("fp", fp)("tn", tn)("fn", fn)
	   ("samples", samples)("gt_pos", gt_pos)
	   ("precision", precision)("recall", recall)("f1", f1)
	   ("accuracy", accuracy)("threshold", threshold);
}

void BoolGateMetrics::Calc() {
	precision = (tp + fp) > 0 ? (double)tp / (tp + fp) : 0;
	recall = (tp + fn) > 0 ? (double)tp / (tp + fn) : 0;
	f1 = (precision + recall) > 0 ? 2 * (precision * recall) / (precision + recall) : 0;
	accuracy = samples > 0 ? (double)(tp + tn) / samples : 0;
}

String AnnLayMethodToString(AnnLayMethod m) {
	switch(m) {
	case ANNLAY_OCR_TEXT:           return "ocr_text";
	case ANNLAY_CLASSIFIER_BOOL:    return "classifier_bool";
	case ANNLAY_CLASSIFIER_LABEL:   return "classifier_label";
	case ANNLAY_ORB_MATCH:          return "orb_match";
	case ANNLAY_CV_TEMPLATE_MATCH:  return "cv_template_match";
	case ANNLAY_IGNORED:            return "ignored";
	default:                        return "ignored";
	}
}

AnnLayMethod AnnLayMethodFromString(const String& s) {
	String v = ToLower(TrimBoth(s));
	if(v == "ocr_text")           return ANNLAY_OCR_TEXT;
	if(v == "classifier_bool")    return ANNLAY_CLASSIFIER_BOOL;
	if(v == "classifier_label")   return ANNLAY_CLASSIFIER_LABEL;
	if(v == "orb_match")          return ANNLAY_ORB_MATCH;
	if(v == "cv_template_match")  return ANNLAY_CV_TEMPLATE_MATCH;
	return ANNLAY_IGNORED;
}

String AnnLayCompositeTypeToString(AnnLayCompositeType t) {
	switch(t) {
	case ANNLAY_COMPOSITE_ELEMENT: return "element";
	case ANNLAY_COMPOSITE_NONE: return "none";
	default:                    return "none";
	}
}

AnnLayCompositeType AnnLayCompositeTypeFromString(const String& s) {
	String v = ToLower(TrimBoth(s));
	if(v == "element" || v == "token") return ANNLAY_COMPOSITE_ELEMENT;
	return ANNLAY_COMPOSITE_NONE;
}

String AnnLaySplitPolicyToString(AnnLaySplitPolicy p) {
	switch(p) {
	case ANNLAY_SPLIT_VERTICAL_HALVES: return "vertical_halves";
	case ANNLAY_SPLIT_NONE:            return "none";
	default:                           return "none";
	}
}

AnnLaySplitPolicy AnnLaySplitPolicyFromString(const String& s) {
	String v = ToLower(TrimBoth(s));
	if(v == "vertical_halves") return ANNLAY_SPLIT_VERTICAL_HALVES;
	return ANNLAY_SPLIT_NONE;
}

void JsonizeIndexMap(JsonIO& jio, VectorMap<String, String>& map) {
	auto DecodeLegacyString = [](const Value& v) -> String {
		if(IsString(v))
			return v;
		if(IsValueMap(v)) {
			ValueMap vm = v;
			for(int i = 0; i < vm.GetCount(); i++) {
				if(vm.GetKey(i) == "value")
					return vm.GetValue(i).ToString();
			}
		}
		return v.ToString();
	};

	if(jio.IsLoading()) {
		map.Clear();
		Value v = jio.Get();
		if(IsValueMap(v)) {
			ValueMap vm = v;
			for(int i = 0; i < vm.GetCount(); i++)
				map.Add(vm.GetKey(i), vm.GetValue(i));
		}
		else if(IsValueArray(v)) {
			ValueArray va = v;
			for(int i = 0; i < va.GetCount(); i++) {
				Value item = va[i];
				if(!IsValueMap(item))
					continue;
				ValueMap kv = item;
				Value keyv, valv;
				for(int j = 0; j < kv.GetCount(); j++) {
					String k = kv.GetKey(j);
					if(k == "key") keyv = kv.GetValue(j);
					else if(k == "value") valv = kv.GetValue(j);
				}
				String key = DecodeLegacyString(keyv);
				if(key.IsEmpty())
					continue;
				map.Add(key, DecodeLegacyString(valv));
			}
		}
	}
	else {
		ValueMap vm;
		for(int i = 0; i < map.GetCount(); i++)
			vm.Add(map.GetKey(i), map[i]);
		jio.Set(vm);
	}
}

void AnnLaySlot::Jsonize(JsonIO& jio) {
	String method_s = AnnLayMethodToString(method);
	String composite_type_s = AnnLayCompositeTypeToString(composite_type);
	String split_policy_s = AnnLaySplitPolicyToString(split_policy);

	jio("id", id)
	   ("method", method_s)
	   ("color_mode", color_mode)
	   ("crop_size", crop_size)
	   ("sub_layout_json", sub_layout_json)
	   ("anchor", anchor)
	   ("anchor_candidates", anchor_candidates)
	   ("classes", classes)
	   ("template_w", template_w)
	   ("template_h", template_h)
	   ("template_crop_x", template_crop_x)
	   ("template_crop_y", template_crop_y)
	   ("template_crop_w", template_crop_w)
	   ("template_crop_h", template_crop_h)
	   ("composite_type", composite_type_s)
	   ("split_policy", split_policy_s)
	   ("group", group)
	   ("head_id", head_id)
	   ("gate", gate);

	auto LoadLegacyIndexMap = [](const Value& v, VectorMap<String, String>& out) {
		out.Clear();
		auto DecodeLegacyString = [](const Value& x) -> String {
			if(IsString(x))
				return x;
			if(x.GetType() == VALUEMAP_V) {
				ValueMap vm = x;
				for(int i = 0; i < vm.GetCount(); i++) {
					if(vm.GetKey(i) == "value")
						return vm.GetValue(i).ToString();
				}
			}
			return x.ToString();
		};

		if(IsNull(v) || v.IsVoid())
			return;

		if(v.GetType() == VALUEMAP_V) {
			ValueMap vm = v;
			for(int i = 0; i < vm.GetCount(); i++)
				out.Add(vm.GetKey(i).ToString(), vm.GetValue(i).ToString());
			return;
		}
		if(v.GetType() != VALUEARRAY_V)
			return;

		ValueArray va = v;
		for(int i = 0; i < va.GetCount(); i++) {
			Value item = va[i];
			if(item.GetType() != VALUEMAP_V)
				continue;
			ValueMap kv = item;
			Value keyv, valv;
			for(int j = 0; j < kv.GetCount(); j++) {
				String k = kv.GetKey(j);
				if(k == "key") keyv = kv.GetValue(j);
				else if(k == "value") valv = kv.GetValue(j);
			}
			String key = DecodeLegacyString(keyv);
			if(key.IsEmpty())
				continue;
			out.Add(key, DecodeLegacyString(valv));
		}
	};

	if(jio.IsLoading()) {
		LoadLegacyIndexMap(jio.Get("sub_groups"), sub_groups);
		LoadLegacyIndexMap(jio.Get("sub_heads"),  sub_heads);
	}
	else {
		ValueMap vmg, vmh;
		for(int i = 0; i < sub_groups.GetCount(); i++)
			vmg.Add(sub_groups.GetKey(i), sub_groups[i]);
		for(int i = 0; i < sub_heads.GetCount(); i++)
			vmh.Add(sub_heads.GetKey(i), sub_heads[i]);
		jio.Set("sub_groups", vmg);
		jio.Set("sub_heads",  vmh);
	}

	jio("presence_threshold", presence_threshold)
	   ("fixed_rotation_deg", fixed_rotation_deg)
	   ("ocr_preprocess_mode", ocr_preprocess_mode)
	   ("ocr_psm", ocr_psm)
	   ("ocr_whitelist", ocr_whitelist)
	   ("ocr_blacklist", ocr_blacklist);


	if(jio.IsLoading()) {
		const Value& mv = jio.Get("method");
		if(IsNumber(mv))
			method = (AnnLayMethod)minmax((int)mv, (int)ANNLAY_OCR_TEXT, (int)ANNLAY_CV_TEMPLATE_MATCH);
		else
			method = AnnLayMethodFromString(method_s);

		composite_type = AnnLayCompositeTypeFromString(composite_type_s);
		split_policy = AnnLaySplitPolicyFromString(split_policy_s);
	}
}

void AnnLayGroup::Jsonize(JsonIO& jio) {
	String method_s = AnnLayMethodToString(method);
	jio("id", id)
	   ("method", method_s)
	   ("crop_size", crop_size)
	   ("classes", classes)
	   ("color_mode", color_mode)
	   ("display_name", display_name)
	   ("legacy_aliases", legacy_aliases)
	   ("head_role", head_role)
	   ("preprocess", preprocess)
	   ("export_dataset_dir", export_dataset_dir);
	if(jio.IsLoading())
		method = AnnLayMethodFromString(method_s);
}

void AnnLay::Jsonize(JsonIO& jio) {
	jio("description", description)
	   ("format", format)
	   ("version", version)
	   ("image_width_ref", image_width_ref)
	   ("image_height_ref", image_height_ref)
	   ("bbox_expand", bbox_expand)
	   ("slots", slots)
	   ("groups", groups);
}

bool AnnLay::Load(const String& path) {
	try {
		String json = LoadFile(path);
		if(json.IsEmpty())
			return false;

		Value v = ParseJSON(json);
		if(IsNull(v))
			return false;

		try {
			LoadFromJsonValue(*this, v);
			return true;
		}
		catch(...) {}
		{
			Value normalized = NormalizeLegacyAnnLayValue(v);
			LoadFromJsonValue(*this, normalized);
			return true;
		}
	}
	catch(...) {
		return false;
	}
}

bool AnnLay::Save(const String& path) const {
	return SaveAsJSONLocal(*this, path, true);
}

AnnLaySlot* AnnLay::FindSlot(const String& id) {
	for(int i = 0; i < slots.GetCount(); i++)
		if(slots[i].id == id)
			return &slots[i];
	return nullptr;
}

const AnnLaySlot* AnnLay::FindSlot(const String& id) const {
	for(int i = 0; i < slots.GetCount(); i++)
		if(slots[i].id == id)
			return &slots[i];
	return nullptr;
}

void AnnLay::ComputeAnchorsFromAnnprj(const String& annprj_path, double variable_threshold) {
	String json = LoadFile(annprj_path);
	if(json.IsEmpty())
		return;

	Value root = ParseJSON(json);
	if(IsNull(root) || !IsValueMap(root))
		return;

	VectorMap<String, Vector<AnchorSample> > by_slot;
	VectorMap<String, int> absent_count;
	VectorMap<String, String> sublayout_by_slot;
	bool ref_set = false;

	Value datasets = root["datasets"];
	if(!IsValueArray(datasets))
		return;

	for(int di = 0; di < datasets.GetCount(); di++) {
		Value ds = datasets[di];
		Value images = ds["images"];
		if(!IsValueArray(images))
			continue;

		for(int ii = 0; ii < images.GetCount(); ii++) {
			Value img = images[ii];
			int iw = img["width"];
			int ih = img["height"];
			if(iw <= 0 || ih <= 0)
				continue;

			if(!ref_set) {
				image_width_ref = iw;
				image_height_ref = ih;
				ref_set = true;
			}

			Value annotations = img["annotations"];
			if(!IsValueArray(annotations))
				continue;

			for(int ai = 0; ai < annotations.GetCount(); ai++) {
				Value ann = annotations[ai];
				String ann_slot_id = ann["metadata"]["mlui_slot_id"];
				if(IsNull(ann_slot_id) || ann_slot_id.IsEmpty())
					continue;

				String slot_id = ann_slot_id;
				
				// Try to find if this annotation ID is an alias for an existing BOOL gate slot
				// We prioritize gate slots so they get anchors even if the legacy slot ID also exists in layout.
				for(int i = 0; i < slots.GetCount(); i++) {
					const AnnLaySlot& s = slots[i];
					if(s.method == ANNLAY_CLASSIFIER_BOOL && AnchoredSlotClassifier::MatchSlotForBool(s.id, ann_slot_id)) {
						if(s.id != ann_slot_id) {
							slot_id = s.id;
						}
						break;
					}
				}

				String sub_layout = ann["metadata"]["sub_layout_json"];
				if(sub_layout.IsEmpty())
					sub_layout = ann["metadata"]["annlay_sub_layout_json"];
				if(!sub_layout.IsEmpty() && sublayout_by_slot.Find(slot_id) < 0)
					sublayout_by_slot.Add(slot_id, sub_layout);

				Value polygons = ann["polygons"];
				if(!IsValueArray(polygons) || polygons.GetCount() == 0 ||
				   !IsValueArray(polygons[0]) || polygons[0].GetCount() == 0) {
					int q = absent_count.Find(slot_id);
					if(q < 0) absent_count.Add(slot_id, 1);
					else absent_count[q]++;
					continue;
				}

				Value pts = polygons[0];
				double min_x = 1e18, min_y = 1e18, max_x = -1e18, max_y = -1e18;
				bool has_point = false;
				for(int pi = 0; pi < pts.GetCount(); pi++) {
					Value p = pts[pi];
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
					continue;

				AnchorSample s;
				s.cx = ((min_x + max_x) * 0.5) / iw;
				s.cy = ((min_y + max_y) * 0.5) / ih;
				s.w = (max_x - min_x) / iw;
				s.h = (max_y - min_y) / ih;

				int k = by_slot.Find(slot_id);
				if(k < 0) {
					Vector<AnchorSample>& v = by_slot.Add(slot_id);
					v.Add(s);
				}
				else {
					by_slot[k].Add(s);
				}
			}
		}
	}

	for(int i = 0; i < absent_count.GetCount(); i++) {
		if(by_slot.Find(absent_count.GetKey(i)) < 0)
			by_slot.Add(absent_count.GetKey(i), Vector<AnchorSample>());
	}

	for(int i = 0; i < by_slot.GetCount(); i++) {
		const String& slot_id = by_slot.GetKey(i);
		const Vector<AnchorSample>& samples = by_slot[i];
		AnnLaySlot* slot = FindSlot(slot_id);
		if(!slot) {
			AnnLaySlot& ns = slots.Add();
			ns.id = slot_id;
			ns.method = ANNLAY_IGNORED;
			slot = &ns;
		}

		slot->anchor_candidates.Clear();
		slot->anchor = AnnLayAnchor();
		int lj = sublayout_by_slot.Find(slot_id);
		if(lj >= 0 && !sublayout_by_slot[lj].IsEmpty())
			slot->sub_layout_json = sublayout_by_slot[lj];
		if(samples.IsEmpty())
			continue;

		AnnLayAnchor mean = BuildAnchorFromSamples(samples);
		if(mean.std_cx > variable_threshold || mean.std_cy > variable_threshold) {
			slot->anchor_candidates = BuildGreedyCandidates(samples, 0.05);
		}
		else {
			slot->anchor = mean;
		}
	}
}

namespace {

static Rectf ClampNormRect(const Rectf& r) {
	double l = minmax(r.left, 0.0, 1.0);
	double t = minmax(r.top, 0.0, 1.0);
	double rr = minmax(r.right, 0.0, 1.0);
	double bb = minmax(r.bottom, 0.0, 1.0);
	if(rr < l) Swap(rr, l);
	if(bb < t) Swap(bb, t);
	if(rr <= l) rr = min(1.0, l + 0.001);
	if(bb <= t) bb = min(1.0, t + 0.001);
	return Rectf(l, t, rr, bb);
}

static bool ParseNode(const Value& node, const Rectf& box, VectorMap<String, Rectf>& out) {
	if(!IsValueMap(node))
		return false;

	ValueMap vm = node;
	String leaf = vm.Get("leaf", String());
	if(!leaf.IsEmpty()) {
		out.GetAdd(leaf) = ClampNormRect(box);
		return true;
	}

	String split = ToLower(TrimBoth(vm.Get("split", String())));
	double at = (double)vm.Get("at", 0.5);
	at = minmax(at, 0.01, 0.99);
	if(split.IsEmpty())
		return false;

	Value a = vm.Get("a", Value());
	Value b = vm.Get("b", Value());
	if(IsNull(a) || IsNull(b))
		return false;

	bool horizontal = (split == "h" || split == "horizontal" || split == "y" || split == "tb");
	bool vertical   = (split == "v" || split == "vertical" || split == "x" || split == "lr");
	if(!horizontal && !vertical)
		return false;

	Rectf ra = box, rb = box;
	if(horizontal) {
		double y = box.top + box.Height() * at;
		ra.bottom = y;
		rb.top = y;
	}
	else {
		double x = box.left + box.Width() * at;
		ra.right = x;
		rb.left = x;
	}
	return ParseNode(a, ra, out) && ParseNode(b, rb, out);
}

static Rectf DefaultRegion(const String& name) {
	if(name == "rank_region")
		return Rectf(0.0, 0.0, 1.0, 0.5);
	if(name == "suit_region")
		return Rectf(0.0, 0.5, 1.0, 1.0);
	return Rectf(0.0, 0.0, 1.0, 1.0);
}

static Rect RectFromNorm(const Rectf& nr, const Rect& base) {
	Rectf r = ClampNormRect(nr);
	int x1 = base.left + (int)floor(r.left * base.GetWidth());
	int y1 = base.top + (int)floor(r.top * base.GetHeight());
	int x2 = base.left + (int)ceil(r.right * base.GetWidth());
	int y2 = base.top + (int)ceil(r.bottom * base.GetHeight());
	x1 = minmax(x1, base.left, base.right);
	y1 = minmax(y1, base.top, base.bottom);
	x2 = minmax(x2, base.left, base.right);
	y2 = minmax(y2, base.top, base.bottom);
	if(x2 <= x1) x2 = min(base.right, x1 + 1);
	if(y2 <= y1) y2 = min(base.bottom, y1 + 1);
	return Rect(x1, y1, x2, y2);
}

}

bool AnnLayTryGetSubLayoutRegions(const AnnLaySlot& slot, VectorMap<String, Rectf>& out_regions) {
	out_regions.Clear();
	String json = TrimBoth(slot.sub_layout_json);
	if(json.IsEmpty())
		return false;

	Value root = ParseJSON(json);
	if(IsNull(root) || !IsValueMap(root))
		return false;

	ValueMap vm = root;
	Value node = vm.Get("root", Value());
	if(IsNull(node))
		node = vm.Get("tree", Value());
	if(IsNull(node))
		node = root; // also accept plain node as root

	VectorMap<String, Rectf> tmp;
	if(!ParseNode(node, Rectf(0.0, 0.0, 1.0, 1.0), tmp))
		return false;
	out_regions <<= tmp;
	return !out_regions.IsEmpty();
}

Rect AnnLayResolveRegionRect(const AnnLaySlot& slot, const Rect& base, const String& region_name) {
	VectorMap<String, Rectf> rr;
	Rectf nr = DefaultRegion(region_name);
	if(AnnLayTryGetSubLayoutRegions(slot, rr)) {
		int q = rr.Find(region_name);
		if(q >= 0)
			nr = rr[q];
		else if(region_name == "card_region") {
			int qc = rr.Find("is_region");
			if(qc >= 0)
				nr = rr[qc];
		}
	}
	return RectFromNorm(nr, base);
}

Rect AnnLayResolveRegionRect(const AnnLaySlot& slot, Size base_size, const String& region_name) {
	Rect base = RectC(0, 0, max(1, base_size.cx), max(1, base_size.cy));
	return AnnLayResolveRegionRect(slot, base, region_name);
}

bool AnnLayTryGetSubLayoutRotationDeg(const AnnLaySlot& slot, double& out_deg) {
	String json = TrimBoth(slot.sub_layout_json);
	if(json.IsEmpty())
		return false;
	Value root = ParseJSON(json);
	if(IsNull(root) || !IsValueMap(root))
		return false;
	ValueMap vm = root;
	Value rv = vm.Get("rotation_deg", Value());
	if(IsNull(rv))
		rv = vm.Get("rot_deg", Value());
	if(IsNull(rv))
		return false;
	out_deg = (double)rv;
	return true;
}

double AnnLayGetSlotRotationDeg(const AnnLaySlot& slot, const String& slot_id) {
	double rot = 0;
	if(AnnLayTryGetSubLayoutRotationDeg(slot, rot))
		return rot;
	if(slot.fixed_rotation_deg != 0.0)
		return slot.fixed_rotation_deg;
	return 0.0;
}

END_UPP_NAMESPACE
