#include "AnnSln.h"

NAMESPACE_UPP

CvTemplateGroup::CvTemplateGroup(const CvTemplateGroup& s) {
	name = s.name;
	slot_stems <<= s.slot_stems;
	templates_label_a_dir = s.templates_label_a_dir;
	templates_label_b_dir = s.templates_label_b_dir;
	label_a_crop = s.label_a_crop;
	label_b_crop = s.label_b_crop;
	label_a_display = s.label_a_display;
	label_b_display = s.label_b_display;
	present_display = s.present_display;
	match_method = s.match_method;
	match_threshold = s.match_threshold;
	zoom_min = s.zoom_min;
	zoom_max = s.zoom_max;
	zoom_steps = s.zoom_steps;
	rotation_slots <<= s.rotation_slots;
	slot_rotation_deg <<= s.slot_rotation_deg;
	rotation_min_deg = s.rotation_min_deg;
	rotation_max_deg = s.rotation_max_deg;
	rotation_steps = s.rotation_steps;
}

CvTemplateGroup& CvTemplateGroup::operator=(const CvTemplateGroup& s) {
	name = s.name;
	slot_stems <<= s.slot_stems;
	templates_label_a_dir = s.templates_label_a_dir;
	templates_label_b_dir = s.templates_label_b_dir;
	label_a_crop = s.label_a_crop;
	label_b_crop = s.label_b_crop;
	label_a_display = s.label_a_display;
	label_b_display = s.label_b_display;
	present_display = s.present_display;
	match_method = s.match_method;
	match_threshold = s.match_threshold;
	zoom_min = s.zoom_min;
	zoom_max = s.zoom_max;
	zoom_steps = s.zoom_steps;
	rotation_slots <<= s.rotation_slots;
	slot_rotation_deg <<= s.slot_rotation_deg;
	rotation_min_deg = s.rotation_min_deg;
	rotation_max_deg = s.rotation_max_deg;
	rotation_steps = s.rotation_steps;
	return *this;
}

void CvTemplateGroup::Jsonize(JsonIO& jio) {
	Vector<String> rotation_slots_json;
	if(jio.IsStoring()) {
		for(int i = 0; i < rotation_slots.GetCount(); i++)
			rotation_slots_json.Add(rotation_slots[i]);
	}
	jio("slot_stems", slot_stems)
	   ("templates_label_a_dir", templates_label_a_dir)
	   ("templates_label_b_dir", templates_label_b_dir)
	   ("label_a_crop", label_a_crop)
	   ("label_b_crop", label_b_crop)
	   ("label_a_display", label_a_display)
	   ("label_b_display", label_b_display)
	   ("present_display", present_display)
	   ("match_method", match_method)
	   ("match_threshold", match_threshold)
	   ("zoom_min", zoom_min)
	   ("zoom_max", zoom_max)
	   ("zoom_steps", zoom_steps)
	   ("rotation_slots", rotation_slots_json)
	   ("slot_rotation_deg", slot_rotation_deg)
	   ("rotation_min_deg", rotation_min_deg)
	   ("rotation_max_deg", rotation_max_deg)
	   ("rotation_steps", rotation_steps);
	if(jio.IsLoading()) {
		rotation_slots.Clear();
		for(int i = 0; i < rotation_slots_json.GetCount(); i++)
			rotation_slots.Add(rotation_slots_json[i]);
	}
}

AnnSln::AnnSln(const AnnSln& s) {
	version = s.version;
	name = s.name;
	annprj = s.annprj;
	annlay = s.annlay;
	mlui = s.mlui;
	crops_dir = s.crops_dir;
	templates_dir = s.templates_dir;
	images_dir = s.images_dir;
	recognition_script = s.recognition_script;
	model_sets <<= s.model_sets;
	train_epochs = s.train_epochs;
	pass2_min_verified = s.pass2_min_verified;
	cv_template_groups <<= s.cv_template_groups;
}

AnnSln& AnnSln::operator=(const AnnSln& s) {
	version = s.version;
	name = s.name;
	annprj = s.annprj;
	annlay = s.annlay;
	mlui = s.mlui;
	crops_dir = s.crops_dir;
	templates_dir = s.templates_dir;
	images_dir = s.images_dir;
	recognition_script = s.recognition_script;
	model_sets <<= s.model_sets;
	train_epochs = s.train_epochs;
	pass2_min_verified = s.pass2_min_verified;
	cv_template_groups <<= s.cv_template_groups;
	return *this;
}

void AnnSln::Jsonize(JsonIO& jio) {
	jio("version", version)
	   ("name", name)
	   ("annprj", annprj)
	   ("annlay", annlay)
	   ("mlui", mlui)
	   ("crops_dir", crops_dir)
	   ("templates_dir", templates_dir)
	   ("images_dir", images_dir)
	   ("recognition_script", recognition_script)
	   ("model_sets", model_sets)
	   ("train_epochs", train_epochs)
	   ("pass2_min_verified", pass2_min_verified);
	// cv_template_groups is handled manually in Load/Save due to named-object format
}

static String ResolvePath(const String& base_dir, const String& rel) {
	if(rel.IsEmpty() || IsFullPath(rel))
		return rel;
	return NormalizePath(AppendFileName(base_dir, rel));
}

bool AnnSln::Load(const String& path) {
	try {
		String json = LoadFile(path);
		if(json.IsEmpty())
			return false;

		Value root = ParseJSON(json);
		if(IsNull(root) || !IsValueMap(root))
			return false;

		AnnSln tmp;
		LoadFromJsonValue(tmp, root);

		// Manually load model_sets if it was in object format
		Value ms = root["model_sets"];
		if(IsValueMap(ms)) {
			tmp.model_sets.Clear();
			ValueMap vms = ms;
			for(int i = 0; i < vms.GetCount(); i++)
				tmp.model_sets.Add(vms.GetKey(i), vms.GetValue(i));
		}

		// Load cv_template_groups from named-object format
		String base_dir = GetFileFolder(path);
		Value cvtg = root["cv_template_groups"];
		if(IsValueMap(cvtg)) {
			tmp.cv_template_groups.Clear();
			ValueMap vmap = cvtg;
			for(int i = 0; i < vmap.GetCount(); i++) {
				CvTemplateGroup g;
				g.name = vmap.GetKey(i);
				Value gv = vmap.GetValue(i);
				LoadFromJsonValue(g, gv);
				if(IsValueMap(gv)) {
					Value zv = gv["zoom_range"];
					if(IsValueArray(zv) && zv.GetCount() == 2) {
						g.zoom_min = (double)zv[0];
						g.zoom_max = (double)zv[1];
					}
					Value rv = gv["rotation_range_deg"];
					if(IsValueArray(rv) && rv.GetCount() == 2) {
						g.rotation_min_deg = (double)rv[0];
						g.rotation_max_deg = (double)rv[1];
					}
					Value srd = gv["slot_rotation_deg"];
					if(IsValueMap(srd)) {
						g.slot_rotation_deg.Clear();
						ValueMap sm = srd;
						for(int si = 0; si < sm.GetCount(); si++)
							g.slot_rotation_deg.Add((String)sm.GetKey(si), (double)sm.GetValue(si));
					}
				}
				// Resolve relative paths
				g.templates_label_a_dir = ResolvePath(base_dir, g.templates_label_a_dir);
				g.templates_label_b_dir = ResolvePath(base_dir, g.templates_label_b_dir);
				tmp.cv_template_groups.Add(pick(g));
			}
		}

		int v = tmp.version;
		if(v <= 0) v = 1;
		if(v > 1)  return false;
		tmp.version = v;
		tmp.train_epochs = max(1, tmp.train_epochs);
		tmp.pass2_min_verified = max(1, tmp.pass2_min_verified);
		*this = pick(tmp);
		return true;
	}
	catch(...) { return false; }
}

bool AnnSln::Save(const String& path) const {
	try {
		AnnSln out = *this;
		out.version = 1;
		out.train_epochs = max(1, out.train_epochs);
		out.pass2_min_verified = max(1, out.pass2_min_verified);
		// Serialize base fields
		String json = StoreAsJson(out, true);

		// Inject cv_template_groups as named-object if any exist
		if(!cv_template_groups.IsEmpty()) {
			// Build as JSON object manually; paths stored absolute (relative paths are resolved at load)
			String groups_json = "{\n";
			for(int i = 0; i < cv_template_groups.GetCount(); i++) {
				const CvTemplateGroup& g = cv_template_groups[i];
				if(i > 0) groups_json += ",\n";
				groups_json += "\"" + g.name + "\": " + StoreAsJson(g, true);
			}
			groups_json += "\n}";
			// Insert before closing brace of the main JSON object
			int pos = json.ReverseFind('}');
			if(pos >= 0)
				json = json.Left(pos) + ",\n\"cv_template_groups\": " + groups_json + "\n}";
		}

		return SaveFile(path, json);
	}
	catch(...) { return false; }
}

END_UPP_NAMESPACE
