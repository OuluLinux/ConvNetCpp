#include "AppMainCommands.h"
#include "AnnotationEditorWindow.h"
#include "AnchoredSlotExporter.h"
#include "AnnotationEditorCommon.h"
#include "ProjectManager.h"
#include <AnnLayCore/AnnMdl.h>
#include <AnnLayCore/AnnSln.h>
#include <AnnLayCore/AiTrainState.h>
#include <AnnLayCore/AnchoredSlotRecognizer.h>
#include <AnnLayCore/RecognitionScript.h>
#include <OCR/Preprocessing.h>
#include <plugin/png/png.h>
#include <cfloat>

namespace Upp {

static String GetFallbackBaseAnnlayPathLocal(const String& annlay_path) {
	String p = TrimBoth(annlay_path);
	if(!p.EndsWith("_trained.annlay"))
		return String();
	String dir = GetFileDirectory(p);
	String title = GetFileTitle(p);
	if(!title.EndsWith("_trained"))
		return String();
	String base_title = title.Left(title.GetCount() - 8);
	return AppendFileName(dir, base_title + ".annlay");
}

static String ResolveFromSlnDir(const String& sln_dir, const String& path) {
	if(IsFullPath(path)) return path;
	return NormalizePath(AppendFileName(sln_dir, path));
}

namespace {

struct SplitData {
	Vector<String> train;
	Vector<String> holdout;
	int seed = 42;
	double ratio = 0.2;
	void Jsonize(JsonIO& jio) {
		jio("train", train)("holdout", holdout)("seed", seed)("ratio", ratio);
	}
};

Index<String> LoadSubsetIds(const String& path, const String& subset) {
	Index<String> out;
	if(path.IsEmpty() || subset == "all") return out;
	
	Value root = ParseJSON(LoadFile(path));
	if(IsNull(root) || !IsValueMap(root)) {
		Cout() << "LoadSubsetIds: failed to parse split file: " << path << "\n";
		return out;
	}
	
	Value ids = root[subset];
	if(IsValueArray(ids)) {
		for(int i = 0; i < ids.GetCount(); i++) {
			Value v = ids[i];
			if(IsString(v)) out.Add(v);
			else if(IsValueMap(v) && v["type"] == "String")
				out.Add(v["value"]);
		}
	}
	return out;
}

struct BoolBaseline : Moveable<BoolBaseline> {
	int schema_version = 2;
	String sln_path;
	String annlay_path;
	String annmdl_path;
	String model_set;
	String group;
	String split_file;
	String offset_mode;
	String timestamp;

	int split_seed = 0;
	double split_ratio = 0;
	int train_image_count = 0;
	int holdout_image_count = 0;

	VectorMap<String, BoolGateMetrics> per_slot;
	BoolGateMetrics train_micro;
	BoolGateMetrics train_macro;
	BoolGateMetrics holdout_micro;
	BoolGateMetrics holdout_macro;
	BoolGateMetrics overall_micro;

	void Jsonize(JsonIO& jio) {
		jio("schema_version", schema_version)
		   ("sln_path", sln_path)("annlay_path", annlay_path)
		   ("annmdl_path", annmdl_path)("model_set", model_set)
		   ("group", group)("split_file", split_file)
		   ("offset_mode", offset_mode)
		   ("split_seed", split_seed)("split_ratio", split_ratio)
		   ("train_image_count", train_image_count)("holdout_image_count", holdout_image_count)
		   ("timestamp", timestamp)("per_slot", per_slot)
		   ("train_micro", train_micro)("train_macro", train_macro)
		   ("holdout_micro", holdout_micro)("holdout_macro", holdout_macro)
		   ("overall_micro", overall_micro);
	}
};

struct AuditFinding : Moveable<AuditFinding> {
	String severity; // ERROR, WARNING
	String message;
	String slot_id;
	String related_slot_id;
	void Jsonize(JsonIO& jio) {
		jio("severity", severity)("message", message)("slot_id", slot_id)("related_slot_id", related_slot_id);
	}
};

struct BoolAuditReport {
	String annlay_path;
	String timestamp;
	int error_count = 0;
	int warning_count = 0;
	Vector<AuditFinding> findings;
	void Jsonize(JsonIO& jio) {
		jio("annlay_path", annlay_path)("timestamp", timestamp)
		   ("error_count", error_count)("warning_count", warning_count)
		   ("findings", findings);
	}
};

struct BoolCheckReport {
	bool passed = false;
	double max_f1_drop = 0;
	double max_recall_drop = 0;
	double min_holdout_f1 = 0;
	bool baseline_mismatch = false;
	
	int    min_bool_support = 0;
	bool   fail_on_low_support = false;
	int    low_support_count = 0;
	ValueArray low_support_slots;
	
	ValueMap metrics;
	ValueArray degraded_slots;
	ValueArray failure_reasons;
	ValueArray mismatch_reasons;

	void Jsonize(JsonIO& jio) {
		jio("passed", passed)
		   ("max_f1_drop", max_f1_drop)
		   ("max_recall_drop", max_recall_drop)
		   ("min_holdout_f1", min_holdout_f1)
		   ("baseline_mismatch", baseline_mismatch)
		   ("min_bool_support", min_bool_support)
		   ("fail_on_low_support", fail_on_low_support)
		   ("low_support_count", low_support_count)
		   ("low_support_slots", low_support_slots)
		   ("metrics", metrics)
		   ("degraded_slots", degraded_slots)
		   ("failure_reasons", failure_reasons)
		   ("mismatch_reasons", mismatch_reasons);
	}
};

struct OffsetMetrics : Moveable<OffsetMetrics> {
	String name;
	int samples = 0;
	double mae_x = 0;
	double mae_y = 0;
	double mean_l1 = 0;
	double p90_l1 = 0;

	void Jsonize(JsonIO& jio) {
		jio("name", name)("samples", samples)("mae_x", mae_x)("mae_y", mae_y)("mean_l1", mean_l1)("p90_l1", p90_l1);
	}
};

struct OffsetBaseline : Moveable<OffsetBaseline> {
	int schema_version = 2;
	String sln_path;
	String annlay_path;
	String annmdl_path;
	String model_set;
	String split_file;
	String subset;
	String timestamp;
	
	int train_image_count = 0;
	int holdout_image_count = 0;

	VectorMap<String, OffsetMetrics> modes;

	void Jsonize(JsonIO& jio) {
		jio("schema_version", schema_version)
		   ("sln_path", sln_path)("annlay_path", annlay_path)
		   ("annmdl_path", annmdl_path)("model_set", model_set)
		   ("split_file", split_file)("subset", subset)
		   ("timestamp", timestamp)
		   ("train_image_count", train_image_count)("holdout_image_count", holdout_image_count)
		   ("modes", modes);
	}
};

struct OffsetCheckReport {
	bool passed = false;
	double max_meanl1_rise = 0;
	double max_p90_rise = 0;
	bool baseline_mismatch = false;
	
	ValueMap metrics;
	ValueArray failure_reasons;
	ValueArray mismatch_reasons;

	void Jsonize(JsonIO& jio) {
		jio("passed", passed)
		   ("max_meanl1_rise", max_meanl1_rise)
		   ("max_p90_rise", max_p90_rise)
		   ("baseline_mismatch", baseline_mismatch)
		   ("metrics", metrics)
		   ("failure_reasons", failure_reasons)
		   ("mismatch_reasons", mismatch_reasons);
	}
};

struct SlotStats : Moveable<SlotStats> {
	String slot_id;
	int tp = 0, fp = 0, tn = 0, fn = 0;
	int samples = 0;
	int gt_pos = 0;

	double Precision() const { return (tp + fp) > 0 ? (double)tp / (tp + fp) : 0; }
	double Recall() const { return (tp + fn) > 0 ? (double)tp / (fn + tp) : 0; }
	double F1() const {
		double p = Precision(), r = Recall();
		return (p + r) > 0 ? 2 * (p * r) / (p + r) : 0;
	}
	double Accuracy() const { return samples > 0 ? (double)(tp + tn) / samples : 0; }
	
	void FillMetrics(BoolGateMetrics& m, double thr) const {
		m.slot_id = slot_id;
		m.tp = tp; m.fp = fp; m.tn = tn; m.fn = fn;
		m.samples = samples; m.gt_pos = gt_pos;
		m.precision = Precision();
		m.recall = Recall();
		m.f1 = F1();
		m.accuracy = Accuracy();
		m.threshold = thr;
	}
};

String ResolveFromBase(const String& base_dir, const String& path) {
	String p = TrimBoth(path);
	if(p.IsEmpty())
		return String();
	if(IsFullPath(p))
		return NormalizePath(p);
	return NormalizePath(AppendFileName(base_dir, p));
}

Image ToGray(const Image& src) {
	if(src.IsEmpty())
		return src;
	int w = src.GetWidth(), h = src.GetHeight();
	ImageBuffer out(w, h);
	for(int y = 0; y < h; y++) {
		const RGBA* s = src[y];
		RGBA* d = out[y];
		for(int x = 0; x < w; x++) {
			byte g = (byte)(s[x].r * 299 / 1000 + s[x].g * 587 / 1000 + s[x].b * 114 / 1000);
			d[x] = {g, g, g, 255};
		}
	}
	return out;
}

bool SavePngImage(const Image& img, const String& path) {
	if(img.IsEmpty())
		return false;
	String dir = GetFileDirectory(path);
	if(!dir.IsEmpty())
		RealizeDirectory(dir);
	return PNGEncoder().SaveFile(path, img);
}

void ExportCvTemplatesToCrops(const AnnSln& sln, const String& crops_root, int pass_index,
                              VectorMap<String, int>& samples_per_group) {
	if(sln.cv_template_groups.IsEmpty())
		return;
	String pass_dir = AppendFileName(crops_root, pass_index == 1 ? "pass1" : "pass2");
	String cv_dir = AppendFileName(pass_dir, "cv_template_match");
	RealizeDirectory(cv_dir);

	for(int gi = 0; gi < sln.cv_template_groups.GetCount(); gi++) {
		const CvTemplateGroup& g = sln.cv_template_groups[gi];
		String group_name = TrimBoth(g.name);
		if(group_name.IsEmpty())
			group_name = Format("group_%d", gi);
		String group_dir = AppendFileName(cv_dir, group_name);
		String out_a = AppendFileName(group_dir, "label_a");
		String out_b = AppendFileName(group_dir, "label_b");
		RealizeDirectory(out_a);
		RealizeDirectory(out_b);

		int count_a = 0;
		FindFile fa(AppendFileName(g.templates_label_a_dir, "*.png"));
		while(fa) {
			Image img = StreamRaster::LoadFileAny(fa.GetPath());
			if(!img.IsEmpty()) {
				Image gray = ToGray(img); // label_a is always B/W
				String dst = AppendFileName(out_a, fa.GetName());
				if(SavePngImage(gray, dst))
					count_a++;
			}
			fa.Next();
		}

		int count_b = 0;
		FindFile fb(AppendFileName(g.templates_label_b_dir, "*.png"));
		while(fb) {
			Image img = StreamRaster::LoadFileAny(fb.GetPath());
			if(!img.IsEmpty()) {
				String dst = AppendFileName(out_b, fb.GetName());
				if(SavePngImage(img, dst))
					count_b++;
			}
			fb.Next();
		}

		samples_per_group.GetAdd("cv_template_match/" + group_name + "/label_a", 0) += count_a;
		samples_per_group.GetAdd("cv_template_match/" + group_name + "/label_b", 0) += count_b;
	}
}

String JoinPairs(const VectorMap<String, String>& m) {
	String out;
	for(int i = 0; i < m.GetCount(); i++) {
		if(i)
			out << " ";
		out << m.GetKey(i) << "=" << m[i];
	}
	return out;
}

String BuildFallbackMetadata(const Vector<SlotResult>& results) {
	String out;
	int shown = 0;
	for(int i = 0; i < results.GetCount() && shown < 6; i++) {
		String val = results[i].raw_text;
		if(val.IsEmpty())
			val = results[i].top_class;
		if(val.IsEmpty() && results[i].class_index >= 0)
			val = AsString(results[i].class_index);
		if(val.IsEmpty())
			continue;
		if(!out.IsEmpty())
			out << " ";
		out << results[i].slot_id << "=" << val;
		shown++;
	}
	if(out.IsEmpty())
		out = "(no metadata)";
	return out;
}

String ResolveImagePathForSmoke(const String& sln_dir,
                                const String& annprj_path,
                                const String& images_dir,
                                const Value& img_rec) {
	String fname = img_rec["file_name"];
	if(fname.IsEmpty())
		fname = img_rec["file_path"];
	if(fname.IsEmpty())
		fname = img_rec["filename"];
	fname = TrimBoth(fname);
	if(fname.IsEmpty())
		return String();

	Vector<String> candidates;
	if(IsFullPath(fname))
		candidates.Add(NormalizePath(fname));
	if(!images_dir.IsEmpty()) {
		candidates.Add(NormalizePath(AppendFileName(images_dir, fname)));
		candidates.Add(NormalizePath(AppendFileName(images_dir, GetFileName(fname))));
	}
	candidates.Add(NormalizePath(AppendFileName(GetFileDirectory(annprj_path), fname)));
	candidates.Add(NormalizePath(AppendFileName(sln_dir, fname)));

	for(int i = 0; i < candidates.GetCount(); i++)
		if(FileExists(candidates[i]))
			return candidates[i];

	return candidates.IsEmpty() ? String() : candidates[0];
}

bool LoadSubsetIds(const String& path, const String& subset, Index<String>& out) {
	out.Clear();
	if(path.IsEmpty() || subset == "all") return true;
	
	if(!FileExists(path)) {
		Cout() << "Error: split file not found: " << path << "\n";
		return false;
	}

	Value root = ParseJSON(LoadFile(path));
	if(IsNull(root) || !IsValueMap(root)) {
		Cout() << "Error: failed to parse split JSON: " << path << "\n";
		return false;
	}
	
	Value ids = root[subset];
	if(IsValueArray(ids)) {
		for(int i = 0; i < ids.GetCount(); i++) {
			Value v = ids[i];
			if(IsValueMap(v)) out.Add((String)v["value"]);
			else out.Add((String)v);
		}
	}
	
	if(out.IsEmpty()) {
		Cout() << "Error: split subset '" << subset << "' is empty in: " << path << "\n";
		return false;
	}
	
	Cout() << "Resolved " << out.GetCount() << " image IDs for subset '" << subset << "' from split file.\n";
	return true;
}

bool ComputeBoolGateMetrics(const String& sln_path, 
                            const String& split_file,
                            const String& group_key,
                            int limit,
                            bool all,
                            const String& subset,
                            OffsetMode offset_mode,
                            BoolGatePolicy gate_policy,
                            BoolBaseline& out)
{
	AnnSln sln;
	if(!sln.Load(sln_path)) return false;

	String sln_dir = GetFileDirectory(sln_path);
	String annlay_path = ResolveFromBase(sln_dir, sln.annlay);
	String annprj_path = ResolveFromBase(sln_dir, sln.annprj);
	String images_dir = ResolveFromBase(sln_dir, sln.images_dir);
	if(images_dir.IsEmpty())
		images_dir = AppendFileName(GetFileDirectory(annprj_path), "images");

	String annmdl_path;
	int pass1 = sln.model_sets.Find("pass1");
	if(pass1 >= 0 && !TrimBoth(sln.model_sets[pass1]).IsEmpty())
		annmdl_path = ResolveFromBase(sln_dir, sln.model_sets[pass1]);
	if(annmdl_path.IsEmpty())
		annmdl_path = AnnMdl::PathFromAnnlay(annlay_path);

	if(!FileExists(annmdl_path)) return false;

	AnnLay lay;
	if(!lay.Load(annlay_path)) return false;

	uint64 load_start = GetTickCount();
	AnchoredSlotRecognizer rec;
	rec.SetOffsetMode(offset_mode);
	rec.SetBoolGatePolicy(gate_policy);

	// Pre-build filter to only load required heads
	Index<String> filter;
	for(int i = 0; i < lay.slots.GetCount(); i++) {
		if(AnchoredSlotClassifier::BoolSlotGroupKey(lay.slots[i].id, &lay) ==
 group_key)
			filter.Add(lay.slots[i].id);
	}
	if(filter.IsEmpty()) {
		Cout() << "Error: group '" << group_key << "' contains no slots in layout.\n";
		return false;
	}
	rec.SetHeadFilter(filter); // Optimization: only load these sessions
	rec.SetSlotFilter(filter); // Optimization: only process these in Recognize()

	if(!rec.Load(annlay_path, annmdl_path)) return false;
	uint64 load_end = GetTickCount();

	Value root = ParseJSON(LoadFile(annprj_path));
	if(IsNull(root) || !IsValueMap(root)) return false;
	Value datasets = root["datasets"];
	if(!IsValueArray(datasets)) return false;

	Index<String> subset_ids;
	if(!LoadSubsetIds(split_file, subset, subset_ids)) return false;
	
	Index<String> train_ids, holdout_ids;
	if(!LoadSubsetIds(split_file, "train", train_ids)) return false;
	if(!LoadSubsetIds(split_file, "holdout", holdout_ids)) return false;

	VectorMap<String, SlotStats> stats;
	VectorMap<String, SlotStats> t_stats, h_stats;

	auto GetStats = [&](VectorMap<String, SlotStats>& m, const String& sid) -> SlotStats& {
		int q = m.Find(sid);
		if(q >= 0) return m[q];
		SlotStats& s = m.Add(sid);
		s.slot_id = sid;
		return s;
	};

	int total_imgs = 0;
	for(int d = 0; d < datasets.GetCount(); d++)
		if(IsValueArray(datasets[d]["images"])) total_imgs += datasets[d]["images"].GetCount();
	
	int kLimit = all ? total_imgs : limit;
	int processed_imgs = 0;

	uint64 start_ts = GetTickCount();

	for(int di = 0; di < datasets.GetCount() && processed_imgs < kLimit; di++) {
		Value images = datasets[di]["images"];
		if(!IsValueArray(images)) continue;
		for(int ii = 0; ii < images.GetCount() && processed_imgs < kLimit; ii++) {
			Value img_rec = images[ii];
			String id = img_rec["file_name"];
			if(id.IsEmpty()) id = img_rec["file_path"];
			if(id.IsEmpty()) id = img_rec["filename"];
			
			if(!subset_ids.IsEmpty() && subset_ids.Find(id) < 0)
				continue;

			String img_path = ResolveImagePathForSmoke(sln_dir, annprj_path, images_dir, img_rec);
			if(img_path.IsEmpty() || !FileExists(img_path)) continue;
			Image img = LoadImageByExtension(img_path);
			if(img.IsEmpty()) continue;

			Vector<SlotResult> results = rec.Recognize(img);
			Value annotations = img_rec["annotations"];

			for(int ri = 0; ri < results.GetCount(); ri++) {
				const SlotResult& res = results[ri];
				
				// Filtering already handled by recognizer, but we double-check group here
				String gkey = AnchoredSlotClassifier::BoolSlotGroupKey(res.slot_id, &rec.GetLayout());
				if(gkey != group_key) continue;

				bool gt_present = false;
				if(IsValueArray(annotations)) {
					for(int ai = 0; ai < annotations.GetCount(); ai++) {
						String ann_slot = annotations[ai]["metadata"]["mlui_slot_id"];
						if(AnchoredSlotClassifier::MatchSlotForBool(res.slot_id, ann_slot)) {
							gt_present = true;
							break;
						}
					}
				}

				bool pred_present = (res.top_class == "true" || res.class_index == 1);
				
				auto UpdateStats = [&](VectorMap<String, SlotStats>& m) {
					SlotStats& s = GetStats(m, res.slot_id);
					s.samples++;
					if(gt_present) s.gt_pos++;
					if(gt_present && pred_present) s.tp++;
					else if(!gt_present && pred_present) s.fp++;
					else if(!gt_present && !pred_present) s.tn++;
					else if(gt_present && !pred_present) s.fn++;
				};

				UpdateStats(stats);
				if(train_ids.Find(id) >= 0) UpdateStats(t_stats);
				if(holdout_ids.Find(id) >= 0) UpdateStats(h_stats);
			}
			processed_imgs++;
		}
	}

	uint64 end_ts = GetTickCount();
	double load_time = (load_end - load_start) / 1000.0;
	double eval_time = (end_ts - start_ts) / 1000.0;
	double total_time = load_time + eval_time;

	Cout() << "\nEvaluation Throughput:\n";
	Cout() << "  Total images: " << processed_imgs << "\n";
	Cout() << "  Model load:   " << Format("%.2f", load_time) << " s\n";
	Cout() << "  Eval loop:    " << Format("%.2f", eval_time) << " s\n";
	Cout() << "  Total time:   " << Format("%.2f", total_time) << " s\n";
	if(eval_time > 0)
		Cout() << "  Throughput:   " << Format("%.2f", processed_imgs / eval_time) << " images/sec\n";

	out.sln_path = sln_path;
	out.annlay_path = annlay_path;
	out.annmdl_path = annmdl_path;
	out.model_set = pass1 >= 0 ? sln.model_sets[pass1] : String();
	out.group = group_key;
	out.split_file = split_file;
	out.offset_mode = OffsetModeToString(offset_mode);
	out.timestamp = MakeIsoTimestamp();

	out.train_image_count = train_ids.GetCount();
	out.holdout_image_count = holdout_ids.GetCount();
	if(!split_file.IsEmpty()) {
		SplitData sd;
		if(LoadFromJsonFile(sd, split_file)) {
			out.split_seed = sd.seed;
			out.split_ratio = sd.ratio;
		}
	}

	for(int i = 0; i < stats.GetCount(); i++) {
		const AnnLaySlot* slot = rec.GetLayout().FindSlot(stats.GetKey(i));
		stats[i].FillMetrics(out.per_slot.Add(stats.GetKey(i)), slot ? slot->presence_threshold : 0.5);
	}

	auto FillAgg = [&](const VectorMap<String, SlotStats>& m, BoolGateMetrics& micro, BoolGateMetrics& macro) {
		int tp = 0, fp = 0, tn = 0, fn = 0, samples = 0, gt_pos = 0;
		double sum_prec = 0, sum_rec = 0, sum_f1 = 0, sum_acc = 0;
		int n = 0;
		for(int i = 0; i < m.GetCount(); i++) {
			const auto& s = m[i];
			tp += s.tp; fp += s.fp; tn += s.tn; fn += s.fn;
			samples += s.samples; gt_pos += s.gt_pos;
			if(s.samples > 0) {
				sum_prec += s.Precision();
				sum_rec += s.Recall();
				sum_f1 += s.F1();
				sum_acc += s.Accuracy();
				n++;
			}
		}
		micro.tp = tp; micro.fp = fp; micro.tn = tn; micro.fn = fn;
		micro.samples = samples; micro.gt_pos = gt_pos;
		micro.precision = (tp + fp) > 0 ? (double)tp / (tp + fp) : 0;
		micro.recall = (tp + fn) > 0 ? (double)tp / (tp + fn) : 0;
		micro.f1 = (micro.precision + micro.recall) > 0 ? 2 * (micro.precision * micro.recall) / (micro.precision + micro.recall) : 0;
		micro.accuracy = samples > 0 ? (double)(tp + tn) / samples : 0;

		if(n > 0) {
			macro.precision = sum_prec / n;
			macro.recall = sum_rec / n;
			macro.f1 = sum_f1 / n;
			macro.accuracy = sum_acc / n;
		}
	};

	FillAgg(stats, out.overall_micro, out.overall_micro);
	FillAgg(t_stats, out.train_micro, out.train_macro);
	FillAgg(h_stats, out.holdout_micro, out.holdout_macro);

	return true;
}

bool ComputeOffsetMetrics(const String& sln_path, 
                          const String& split_file,
                          const String& subset,
                          int limit,
                          bool all,
                          OffsetBaseline& out)
{
	AnnSln sln;
	if(!sln.Load(sln_path)) return false;

	String sln_dir = GetFileDirectory(sln_path);
	String annlay_path = ResolveFromBase(sln_dir, sln.annlay);
	String annprj_path = ResolveFromBase(sln_dir, sln.annprj);
	String images_dir = ResolveFromBase(sln_dir, sln.images_dir);
	if(images_dir.IsEmpty())
		images_dir = AppendFileName(GetFileDirectory(annprj_path), "images");

	String annmdl_path;
	int pass1 = sln.model_sets.Find("pass1");
	if(pass1 >= 0 && !TrimBoth(sln.model_sets[pass1]).IsEmpty())
		annmdl_path = ResolveFromBase(sln_dir, sln.model_sets[pass1]);
	if(annmdl_path.IsEmpty())
		annmdl_path = AnnMdl::PathFromAnnlay(annlay_path);

	if(!FileExists(annmdl_path)) return false;

	AnnLay lay;
	if(!lay.Load(annlay_path)) return false;

	uint64 load_start = GetTickCount();
	AnchoredSlotRecognizer rec;

	// Pre-build filters
	Index<String> filter;
	Index<String> h_filter;
	for(int i = 0; i < lay.slots.GetCount(); i++) {
		const AnnLaySlot& s = lay.slots[i];
		if(s.composite_type == ANNLAY_COMPOSITE_ELEMENT ||
		   s.method == ANNLAY_CLASSIFIER_BOOL ||
		   s.method == ANNLAY_CLASSIFIER_LABEL)
		{
			filter.Add(s.id);
			if(s.composite_type == ANNLAY_COMPOSITE_ELEMENT) {
				h_filter.Add(s.id + "#presence");
				h_filter.Add(s.id + "#level");
				h_filter.Add(s.id + "#category");
			}
			else {
				h_filter.Add(s.id);
			}
			h_filter.Add(s.id + "#offset");
			h_filter.Add(s.id + "#offset_x");
			h_filter.Add(s.id + "#offset_y");
		}
	}
	rec.SetHeadFilter(h_filter);
	rec.SetSlotFilter(filter);

	if(!rec.Load(annlay_path, annmdl_path)) return false;
	uint64 load_end = GetTickCount();

	Value root = ParseJSON(LoadFile(annprj_path));
	if(IsNull(root) || !IsValueMap(root)) return false;
	Value datasets = root["datasets"];
	if(!IsValueArray(datasets)) return false;

	Index<String> subset_ids;
	if(!LoadSubsetIds(split_file, subset, subset_ids)) return false;
	Index<String> train_ids, holdout_ids;
	if(!LoadSubsetIds(split_file, "train", train_ids)) return false;
	if(!LoadSubsetIds(split_file, "holdout", holdout_ids)) return false;

	struct ModeStats : Moveable<ModeStats> {
		OffsetMode mode;
		String name;
		int samples = 0;
		double sum_dx = 0, sum_dy = 0, sum_l1 = 0;
		Vector<double> l1_errors;
	};

	Vector<ModeStats> stats;
	auto& s_none = stats.Add(); s_none.mode = OFFSET_NONE; s_none.name = "none";
	auto& s_split = stats.Add(); s_split.mode = OFFSET_SPLIT; s_split.name = "split";
	auto& s_comb = stats.Add(); s_comb.mode = OFFSET_COMBINED; s_comb.name = "combined";
	auto& s_auto = stats.Add(); s_auto.mode = OFFSET_AUTO; s_auto.name = "auto";

	int total_imgs = 0;
	for(int d = 0; d < datasets.GetCount(); d++)
		if(IsValueArray(datasets[d]["images"])) total_imgs += datasets[d]["images"].GetCount();
	
	int kLimit = all ? total_imgs : limit;
	int processed_imgs = 0;

	uint64 start_ts = GetTickCount();

	for(int di = 0; di < datasets.GetCount() && processed_imgs < kLimit; di++) {
		Value images = datasets[di]["images"];
		if(!IsValueArray(images)) continue;
		for(int ii = 0; ii < images.GetCount() && processed_imgs < kLimit; ii++) {
			Value img_rec = images[ii];
			String id = img_rec["file_name"];
			if(id.IsEmpty()) id = img_rec["file_path"];
			if(id.IsEmpty()) id = img_rec["filename"];
			
			if(!subset_ids.IsEmpty() && subset_ids.Find(id) < 0)
				continue;

			String img_path = ResolveImagePathForSmoke(sln_dir, annprj_path, images_dir, img_rec);
			if(img_path.IsEmpty() || !FileExists(img_path)) continue;
			Image img = LoadImageByExtension(img_path);
			if(img.IsEmpty()) continue;

			VectorMap<String, Pointf> ground_truth;
			Value annotations = img_rec["annotations"];
			if(IsValueArray(annotations)) {
				for(int ai = 0; ai < annotations.GetCount(); ai++) {
					Value ann = annotations[ai];
					String sid = ann["metadata"]["mlui_slot_id"];
					Value polygons = ann["polygons"];
					if(!sid.IsEmpty() && IsValueArray(polygons) && polygons.GetCount() > 0) {
						Value pts = polygons[0];
						if(IsValueArray(pts) && pts.GetCount() > 0) {
							double scx = 0, scy = 0;
							for(int pi = 0; pi < pts.GetCount(); pi++) {
								scx += (double)pts[pi]["x"];
								scy += (double)pts[pi]["y"];
							}
							ground_truth.Add(sid, Pointf(scx / pts.GetCount(), scy / pts.GetCount()));
						}
					}
				}
			}

			if(ground_truth.IsEmpty()) continue;

			for(int m = 0; m < stats.GetCount(); m++) {
				ModeStats& s = stats[m];
				rec.SetOffsetMode(s.mode);
				Vector<SlotResult> results = rec.Recognize(img);
				for(int ri = 0; ri < results.GetCount(); ri++) {
					const SlotResult& res = results[ri];
					int q = ground_truth.Find(res.slot_id);
					if(q >= 0) {
						Pointf gt = ground_truth[q];
						Pointf pred((res.pixel_bbox.left + res.pixel_bbox.right) * 0.5,
						            (res.pixel_bbox.top + res.pixel_bbox.bottom) * 0.5);
						double dx = abs(pred.x - gt.x);
						double dy = abs(pred.y - gt.y);
						double l1 = dx + dy;
						s.samples++;
						s.sum_dx += dx;
						s.sum_dy += dy;
						s.sum_l1 += l1;
						s.l1_errors.Add(l1);
					}
				}
			}
			processed_imgs++;
		}
	}

	uint64 end_ts = GetTickCount();
	double load_time = (load_end - load_start) / 1000.0;
	double eval_time = (end_ts - start_ts) / 1000.0;
	double total_time = load_time + eval_time;

	Cout() << "\nOffset Evaluation Throughput:\n";
	Cout() << "  Total images: " << processed_imgs << "\n";
	Cout() << "  Model load:   " << Format("%.2f", load_time) << " s\n";
	Cout() << "  Eval loop:    " << Format("%.2f", eval_time) << " s\n";
	Cout() << "  Total time:   " << Format("%.2f", total_time) << " s\n";
	if(eval_time > 0)
		Cout() << "  Throughput:   " << Format("%.2f", processed_imgs / eval_time) << " images/sec\n";

	out.sln_path = sln_path;
	out.annlay_path = annlay_path;
	out.annmdl_path = annmdl_path;
	out.model_set = pass1 >= 0 ? sln.model_sets[pass1] : String();
	out.split_file = split_file;
	out.subset = subset;
	out.timestamp = MakeIsoTimestamp();
	out.train_image_count = train_ids.GetCount();
	out.holdout_image_count = holdout_ids.GetCount();

	for(int m = 0; m < stats.GetCount(); m++) {
		ModeStats& s = stats[m];
		OffsetMetrics& om = out.modes.Add(s.name);
		om.name = s.name;
		om.samples = s.samples;
		if(s.samples > 0) {
			om.mae_x = s.sum_dx / s.samples;
			om.mae_y = s.sum_dy / s.samples;
			om.mean_l1 = s.sum_l1 / s.samples;
			if(!s.l1_errors.IsEmpty()) {
				Sort(s.l1_errors);
				om.p90_l1 = s.l1_errors[min((int)(s.l1_errors.GetCount() * 0.9), s.l1_errors.GetCount() - 1)];
			}
		}
	}

	return true;
}

bool CheckBaselineMismatch(const String& cmd_name, const String& base_annlay, const String& cur_annlay, 
                           const String& base_annmdl, const String& cur_annmdl, 
                           const String& base_model_set, const String& cur_model_set,
                           bool allow_mismatch, ValueArray& mismatch_reasons, bool& mismatch_flag) {
	mismatch_flag = false;
	
	if(!base_annlay.IsEmpty() && base_annlay != cur_annlay) {
		mismatch_flag = true;
		mismatch_reasons.Add("annlay_path mismatch: base=" + base_annlay + " cur=" + cur_annlay);
	}
	if(!base_annmdl.IsEmpty() && base_annmdl != cur_annmdl) {
		mismatch_flag = true;
		mismatch_reasons.Add("annmdl_path mismatch: base=" + base_annmdl + " cur=" + cur_annmdl);
	}
	if(!base_model_set.IsEmpty() && base_model_set != cur_model_set) {
		mismatch_flag = true;
		mismatch_reasons.Add("model_set mismatch: base=" + base_model_set + " cur=" + cur_model_set);
	}

	if(mismatch_flag) {
		Cout() << "\nWARNING: Baseline provenance mismatch detected!\n";
		for(int i = 0; i < mismatch_reasons.GetCount(); i++) {
			Cout() << "  - " << mismatch_reasons[i] << "\n";
		}
		if(!allow_mismatch) {
			Cout() << "FAIL: Mismatch is not allowed. Use --allow-baseline-mismatch to override.\n";
			return false; // Fail check
		}
		Cout() << "OVERRIDE: --allow-baseline-mismatch is set. Continuing check...\n";
	}
	return true;
}

} // namespace

bool RunDumpProjectManager(const AppArguments& args) {
	if(!args.annsln_path.IsEmpty() && !FileExists(args.annsln_path)) {
		Cout() << "Failed to open .annsln (file not found): " << args.annsln_path << "\n";
		return false;
	}

	ProjectManager pm;
	if(!args.annsln_path.IsEmpty()) {
		pm.OpenSln(args.annsln_path);
		if(pm.GetCurrentSlnPath().IsEmpty()) {
			Cout() << "Failed to open .annsln: " << args.annsln_path << "\n";
			return false;
		}
	}
	pm.DumpToStdout();
	return true;
}

bool RunCliMode(const AppArguments& args, int& exit_code, bool& handled_cli_mode) {
	handled_cli_mode = true;
	switch(args.mode) {
	case AppArguments::MODE_DUMP_PROJECT_MANAGER:
		exit_code = RunDumpProjectManager(args) ? 0 : 1;
		return true;
	case AppArguments::MODE_TRAIN_HEADLESS:
		exit_code = RunTrainHeadless(args) ? 0 : 1;
		return true;
	case AppArguments::MODE_TRAIN_SLOTS:
		exit_code = RunTrainSlots(args) ? 0 : 1;
		return true;
	case AppArguments::MODE_TRAIN_SLOTS_FROM_CROPS:
		exit_code = RunTrainSlotsFromCrops(args) ? 0 : 1;
		return true;
	case AppArguments::MODE_RECOGNIZE_SLOTS:
		exit_code = RunRecognizeSlots(args) ? 0 : 1;
		return true;
	case AppArguments::MODE_SMOKE_TEST:
		exit_code = RunSmokeTest(args) ? 0 : 1;
		return true;
	case AppArguments::MODE_EVAL_OFFSET_HEADS:
		exit_code = RunEvalOffsetHeads(args) ? 0 : 1;
		return true;
	case AppArguments::MODE_AUDIT_OFFSET_HEADS:
		exit_code = RunAuditOffsetHeads(args) ? 0 : 1;
		return true;
	case AppArguments::MODE_PRUNE_OFFSET_HEADS:
		exit_code = RunPruneOffsetHeads(args) ? 0 : 1;
		return true;
	case AppArguments::MODE_RECOMPUTE_ANCHORS:
		exit_code = RunRecomputeAnchors(args) ? 0 : 1;
		return true;
	case AppArguments::MODE_MAKE_EVAL_SPLIT:
		exit_code = RunMakeEvalSplit(args) ? 0 : 1;
		return true;
	case AppArguments::MODE_SAVE_BOOL_BASELINE:
		exit_code = RunSaveBoolBaseline(args) ? 0 : 1;
		return true;
	case AppArguments::MODE_CHECK_BOOL_BASELINE:
		exit_code = RunCheckBoolBaseline(args) ? 0 : 1;
		return true;
	case AppArguments::MODE_REPORT_BOOL_GATES:
		exit_code = RunReportBoolGates(args) ? 0 : 1;
		return true;
	case AppArguments::MODE_SAVE_OFFSET_BASELINE:
		exit_code = RunSaveOffsetBaseline(args) ? 0 : 1;
		return true;
	case AppArguments::MODE_CHECK_OFFSET_BASELINE:
		exit_code = RunCheckOffsetBaseline(args) ? 0 : 1;
		return true;
	case AppArguments::MODE_EVAL_BOOL_GATES:
		exit_code = RunEvalBoolGates(args) ? 0 : 1;
		return true;
	case AppArguments::MODE_AUDIT_BOOL_GATES:
		exit_code = RunAuditBoolGates(args) ? 0 : 1;
		return true;
	case AppArguments::MODE_FIX_BOOL_GATES:
		exit_code = RunFixBoolGates(args) ? 0 : 1;
		return true;
	case AppArguments::MODE_TUNE_BOOL_GATES:
		exit_code = RunTuneBoolGates(args) ? 0 : 1;
		return true;
	case AppArguments::MODE_MINE_BOOL_HARD_NEGATIVES:
		exit_code = RunMineBoolHardNegatives(args) ? 0 : 1;
		return true;
	case AppArguments::MODE_MERGE_HARDNEG:
		exit_code = RunMergeHardneg(args) ? 0 : 1;
		return true;
	case AppArguments::MODE_EXPORT_CROPS:
		exit_code = RunExportCrops(args) ? 0 : 1;
		return true;
	case AppArguments::MODE_RUN_PHASE6_PREFLIGHT:
		exit_code = RunRunPhase6Preflight(args) ? 0 : 1;
		return true;
	case AppArguments::MODE_REFRESH_PHASE6_BASELINES:
		exit_code = RunRefreshPhase6Baselines(args) ? 0 : 1;
		return true;
	case AppArguments::MODE_RUN_PHASE6_EXIT_GATE:
		exit_code = RunRunPhase6ExitGate(args) ? 0 : 1;
		return true;
	case AppArguments::MODE_TRACE_FRAME_LOG:
		exit_code = RunTraceFrameLog(args) ? 0 : 1;
		return true;
	case AppArguments::MODE_REPORT_CARD_VISIBILITY_CONSISTENCY:
		exit_code = RunReportCardVisibilityConsistency(args) ? 0 : 1;
		return true;
	case AppArguments::MODE_TUNE_CARD_VISIBILITY_BY_CONSISTENCY:
		exit_code = RunTuneCardVisibilityByConsistency(args) ? 0 : 1;
		return true;
	case AppArguments::MODE_TUNE_CARD_JOINT_PIPELINE:
		exit_code = RunTuneCardJointPipeline(args) ? 0 : 1;
		return true;
	case AppArguments::MODE_SENTINEL_TEST_RB:
		exit_code = RunSentinelTestRB(args) ? 0 : 1;
		return true;
	case AppArguments::MODE_PIPELINE_COMPLEATNESS_TEST:
		exit_code = RunPipelineCompletenessTest(args) ? 0 : 1;
		return true;
	case AppArguments::MODE_TEST_FR_INVARIANTS:
		exit_code = RunTestFRInvariants(args) ? 0 : 1;
		return true;
	case AppArguments::MODE_TEST_XOFFSET_CONVLAYERS:
		exit_code = RunTestXOffsetConvLayers(args) ? 0 : 1;
		return true;
	case AppArguments::MODE_EVAL_OCR_MODES:
		exit_code = RunEvalOcrModes(args) ? 0 : 1;
		return true;
	case AppArguments::MODE_DUMP_NN_STEPS:
		exit_code = RunDumpNNSteps(args) ? 0 : 1;
		return true;
	case AppArguments::MODE_FRAME_RECOGNIZER_DUMP_STEPS:
		exit_code = RunFrameRecognizerDumpSteps(args) ? 0 : 1;
		return true;
	case AppArguments::MODE_FRAME_RECOGNIZER_DUMP_MEMORY_ALLOC:
		exit_code = RunFrameRecognizerDumpMemoryAlloc(args) ? 0 : 1;
		return true;
	case AppArguments::MODE_MIGRATE_ANNMDL_SESSIONS:
		exit_code = RunMigrateAnnmdlSessions(args) ? 0 : 1;
		return true;
	case AppArguments::MODE_MIGRATE_ANNMDL_V3:
		exit_code = RunMigrateAnnmdlV3(args) ? 0 : 1;
		return true;
	case AppArguments::MODE_NORMALIZE_COMPOSITE_HEADS:
		exit_code = RunNormalizeCompositeHeads(args) ? 0 : 1;
		return true;
	case AppArguments::MODE_TRAIN_ALL:
		exit_code = RunTrainAll(args) ? 0 : 1;
		return true;
	case AppArguments::MODE_DEBUG_GUI_TRAIN:
		exit_code = RunDebugGuiTrain(args) ? 0 : 1;
		return true;
	case AppArguments::MODE_MERGE_MODES:
		exit_code = RunMergeModes(args) ? 0 : 1;
		return true;
	default:
		handled_cli_mode = false;
		return false;
	}
}

bool RunTrainSlots(const AppArguments& args) {
	AnnLay lay;
	if(FileExists(args.train_annlay_path)) {
		if(!lay.Load(args.train_annlay_path)) {
			Cout() << "Failed to load annlay: " << args.train_annlay_path << "\n";
			return false;
		}
	}

	lay.ComputeAnchorsFromAnnprj(args.train_annprj_path);

	for(int i = 0; i < lay.slots.GetCount(); i++) {
		AnnLaySlot& slot = lay.slots[i];
		if(slot.method != ANNLAY_IGNORED)
			continue;
		if(slot.id.Find("_name") >= 0 || slot.id.Find("_stack") >= 0 || slot.id.Find("_bet") >= 0 || slot.id == "total_pot") {
			slot.method = ANNLAY_OCR_TEXT;
		}
		else if(slot.id.Find("_panel") >= 0 || slot.id.Find("_in_game") >= 0 || slot.id == "dealer_chip" || slot.id == "active_timer") {
			slot.method = ANNLAY_CLASSIFIER_BOOL;
		}
		else if(slot.id.Find("_card_") >= 0) {
			slot.method = ANNLAY_CLASSIFIER_LABEL;
			slot.classes.Clear();
			static const char* ranks[] = {"2","3","4","5","6","7","8","9","T","J","Q","K","A"};
			static const char* suits[] = {"s","h","d","c"};
			slot.classes.Add("no_card");
			for(int r = 0; r < 13; r++)
				for(int s = 0; s < 4; s++)
					slot.classes.Add(String(ranks[r]) + suits[s]);
		}
	}

	if(!lay.Save(args.train_annlay_path)) {
		Cout() << "Failed to save initial annlay: " << args.train_annlay_path << "\n";
		return false;
	}

	AnchoredSlotClassifier trainer;
	trainer.max_epochs = args.train_epochs;
	trainer.WhenProgress = [](String slot, int epoch, double loss, double acc) {
		Cout() << "Slot: " << slot << " Epoch: " << epoch << " Loss: " << loss << " Acc: " << acc << "\n";
	};
	AnnMdl mdl;
	mdl.Load(args.train_annlay_path);
	trainer.TrainAll(lay, args.train_annprj_path, args.train_images_dir, args.train_card_strip, &mdl);
	if(!lay.Save(args.train_annlay_path)) {
		Cout() << "Failed to save trained annlay: " << args.train_annlay_path << "\n";
		return false;
	}
	if(!mdl.Save(args.train_annlay_path)) {
		Cout() << "Failed to save annmdl: " << AnnMdl::PathFromAnnlay(args.train_annlay_path) << "\n";
		return false;
	}
	Cout() << "Training complete.\n";
	return true;
}

static String GroupKeyToCropsSubdir(const String& group_key) {
	String s = group_key;
	s.Replace("#", "_");
	return s;
}


bool RunTrainAll(const AppArguments& args) {
	if(!FileExists(args.train_all_sln_path)) {
		Cout() << "Error: .annsln file not found: " << args.train_all_sln_path << "\n";
		return false;
	}

	AnnSln sln;
	if(!sln.Load(args.train_all_sln_path)) {
		Cout() << "Error: failed to load .annsln: " << args.train_all_sln_path << "\n";
		return false;
	}

	String annlay_path = sln.annlay;
	if(!IsFullPath(annlay_path))
		annlay_path = NormalizePath(AppendFileName(GetFileDirectory(args.train_all_sln_path), annlay_path));

	if(!FileExists(annlay_path)) {
		Cout() << "Error: .annlay file not found: " << annlay_path << "\n";
		return false;
	}

	String crops_dir = sln.crops_dir;
	if(!IsFullPath(crops_dir))
		crops_dir = NormalizePath(AppendFileName(GetFileDirectory(args.train_all_sln_path), crops_dir));

	String pass_sub = (args.train_all_pass == 1) ? "pass1" : "pass2";
	String pass_crops = AppendFileName(crops_dir, pass_sub);

	if(!DirectoryExists(pass_crops)) {
		Cout() << "Error: crops directory does not exist: " << pass_crops << "\n";
		return false;
	}

	AppArguments train_args;
	train_args.mode = AppArguments::MODE_TRAIN_SLOTS_FROM_CROPS;
	train_args.crops_annlay_path = annlay_path;
	train_args.crops_crops_dir = pass_crops;
	train_args.crops_epochs = args.train_all_epochs;
	train_args.crops_bool_balance_by_slot = true;
	train_args.crops_bool_slot_cap = 0;
	train_args.crops_ignore_auto_stop = false;

	return RunTrainSlotsFromCrops(train_args);
}

static void DumpDebugGuiTrainPredictions(ConvNet::Session& ses, int max_samples) {
	ConvNet::SessionData& sd = ses.Data();
	int n = min(max_samples, sd.GetTestCount());
	if(n <= 0)
		return;

	Cout() << "[debug-gui-train] --- test predictions (" << n << " samples) ---\n";
	for(int i = 0; i < n; i++) {
		Vector<double> pred = ses.Predict(sd.GetTest(i));
		int best = -1;
		double best_v = -DBL_MAX;
		String pred_s;
		pred_s << "[";
		for(int c = 0; c < pred.GetCount(); c++) {
			if(c)
				pred_s << ", ";
			String cls = c < sd.GetClassCount() ? sd.GetClass(c) : AsString(c);
			pred_s << cls << ":" << Format("%.2f", pred[c]);
			if(pred[c] > best_v) {
				best_v = pred[c];
				best = c;
			}
		}
		pred_s << "]";
		int label = sd.GetTestLabel(i);
		String label_s = label >= 0 && label < sd.GetClassCount() ? sd.GetClass(label) : AsString(label);
		Cout() << "[debug-gui-train]   #" << i
		       << "  pred=" << pred_s
		       << "  label=" << label_s
		       << "  " << (best == label ? "CORRECT" : "WRONG") << "\n";
	}
}

bool RunDebugGuiTrain(const AppArguments& args) {
	if(!FileExists(args.debug_gui_train_sln_path)) {
		Cout() << "Error: .annsln file not found: " << args.debug_gui_train_sln_path << "\n";
		return false;
	}

	AnnSln sln;
	if(!sln.Load(args.debug_gui_train_sln_path)) {
		Cout() << "Error: failed to load .annsln: " << args.debug_gui_train_sln_path << "\n";
		return false;
	}

	String sln_dir = GetFileDirectory(args.debug_gui_train_sln_path);
	String annlay_path = sln.annlay;
	if(!IsFullPath(annlay_path))
		annlay_path = NormalizePath(AppendFileName(sln_dir, annlay_path));
	String annprj_path = sln.annprj;
	if(!IsFullPath(annprj_path))
		annprj_path = NormalizePath(AppendFileName(sln_dir, annprj_path));
	String images_dir = sln.images_dir;
	if(!IsFullPath(images_dir))
		images_dir = NormalizePath(AppendFileName(sln_dir, images_dir));

	AnnLay lay;
	if(!lay.Load(annlay_path)) {
		Cout() << "Error: failed to load .annlay: " << annlay_path << "\n";
		return false;
	}
	if(!FileExists(annprj_path)) {
		Cout() << "Error: .annprj file not found: " << annprj_path << "\n";
		return false;
	}
	if(!DirectoryExists(images_dir)) {
		Cout() << "Error: images directory not found: " << images_dir << "\n";
		return false;
	}

	VectorMap<String, Vector<String>> groups = AnchoredSlotClassifier::GetSlotGroups(lay);
	Vector<String> slot_ids;
	int g = groups.Find(args.debug_gui_train_group_key);
	if(g >= 0)
		slot_ids <<= groups[g];
	else if(lay.FindSlot(args.debug_gui_train_group_key))
		slot_ids.Add(args.debug_gui_train_group_key);

	Cout() << "[debug-gui-train] group=" << args.debug_gui_train_group_key
	       << "  epochs=" << args.debug_gui_train_epochs
	       << "  timeout=" << args.debug_gui_train_timeout_sec << "s\n";
	Cout() << "[debug-gui-train] slots:";
	for(const String& sid : slot_ids)
		Cout() << " " << sid;
	Cout() << " (" << slot_ids.GetCount() << " slots)\n";

	AnnMdl mdl;
	mdl.Load(annlay_path);
	ConvNet::Session ses;
	uint64 t_start = GetTickCount();
	uint64 last_loss = 0;
	uint64 last_pred = 0;
	bool timed_out = false;

	auto Progress = [&](String group, int epoch, double loss, double acc) {
		uint64 now = GetTickCount();
		double elapsed = (double)(now - t_start) / 1000.0;
		bool print_loss = args.debug_gui_train_loss_interval <= 0 ||
		                  last_loss == 0 ||
		                  (double)(now - last_loss) / 1000.0 >= args.debug_gui_train_loss_interval ||
		                  epoch >= args.debug_gui_train_epochs;
		if(print_loss) {
			Cout() << "[debug-gui-train] epoch " << epoch
			       << "  iter " << ses.GetStepNum()
			       << "  loss=" << Format("%.4f", loss)
			       << "  acc=" << Format("%.2f", acc) << "\n";
			last_loss = now;
		}
		bool print_pred = args.debug_gui_train_pred_samples > 0 &&
		                  (args.debug_gui_train_pred_interval <= 0 ||
		                   last_pred == 0 ||
		                   (double)(now - last_pred) / 1000.0 >= args.debug_gui_train_pred_interval ||
		                   epoch >= args.debug_gui_train_epochs);
		if(print_pred) {
			DumpDebugGuiTrainPredictions(ses, args.debug_gui_train_pred_samples);
			last_pred = now;
		}
		if(elapsed >= args.debug_gui_train_timeout_sec) {
			timed_out = true;
			ses.StopTraining();
		}
	};

	bool ok = AnchoredSlotClassifier::TrainBoolGroup(lay,
	                                                args.debug_gui_train_group_key,
	                                                annprj_path,
	                                                images_dir,
	                                                args.debug_gui_train_epochs,
	                                                Progress,
	                                                &ses,
	                                                &mdl,
	                                                true,
	                                                0,
	                                                true,
	                                                true);
	double elapsed = (double)(GetTickCount() - t_start) / 1000.0;
	if(ok && !mdl.Save(annlay_path)) {
		Cout() << "[debug-gui-train] ERROR: failed to save annmdl: " << AnnMdl::PathFromAnnlay(annlay_path) << "\n";
		return false;
	}
	Cout() << "[debug-gui-train] " << (ok ? "DONE" : "FAILED")
	       << " (" << ses.GetIteration() << " epochs, "
	       << Format("%.1f", elapsed) << " sec";
	if(timed_out)
		Cout() << ", timeout";
	Cout() << ")\n";
	return ok;
}

bool RunTrainSlotsFromCrops(const AppArguments& args) {
	AnnLay lay;
	if(!lay.Load(args.crops_annlay_path)) {
		String fallback_annlay = GetFallbackBaseAnnlayPathLocal(args.crops_annlay_path);
		bool recovered = false;
		if(!fallback_annlay.IsEmpty() && FileExists(fallback_annlay)) {
			Cout() << "Warning: failed to load trained annlay, attempting recovery from base annlay: "
			       << fallback_annlay << "\n";
			String base_json = LoadFile(fallback_annlay);
			if(!base_json.IsEmpty() && SaveFile(args.crops_annlay_path, base_json) && lay.Load(args.crops_annlay_path)) {
				Cout() << "Recovered trained annlay from base layout: " << args.crops_annlay_path << "\n";
				recovered = true;
			}
		}
		if(!recovered) {
			Cout() << "Failed to load: " << args.crops_annlay_path << "\n";
			return false;
		}
	}

	// Build list of group keys to train.
	// If --slot was given, train only that group; otherwise train all groups
	// that have a corresponding crops subdirectory.
	VectorMap<String, Vector<String>> all_groups = AnchoredSlotClassifier::GetSlotGroups(lay);
	Vector<String> keys_to_train;

	if(!args.crops_slot_key.IsEmpty()) {
		keys_to_train.Add(args.crops_slot_key);
	}
	else {
		for(int g = 0; g < all_groups.GetCount(); g++) {
			String key = all_groups.GetKey(g);
			String subdir = GroupKeyToCropsSubdir(key);
			String full = AppendFileName(args.crops_crops_dir, subdir);
			if(DirectoryExists(full))
				keys_to_train.Add(key);
		}
		if(keys_to_train.IsEmpty()) {
			// Fall back: try crops_dir itself as a single group
			if(all_groups.GetCount() > 0)
				keys_to_train.Add(all_groups.GetKey(0));
		}
	}

	Cout() << "[RunTrainSlotsFromCrops] groups to train: " << keys_to_train.GetCount() << "\n";
	for(int i = 0; i < keys_to_train.GetCount(); i++)
		Cout() << "  [" << i << "] " << keys_to_train[i] << "\n";

	AnnMdl mdl;
	mdl.Load(args.crops_annlay_path);

	TrainingJobRequest req;
	req.source_mode = TRAIN_SOURCE_CROPS;
	req.group_keys = clone(keys_to_train);
	req.crops_dir = args.crops_crops_dir;
	req.max_epochs = args.crops_epochs;
	req.balance_by_slot = args.crops_bool_balance_by_slot;
	req.slot_cap = args.crops_bool_slot_cap;
	req.ignore_auto_stop = args.crops_ignore_auto_stop;

	TrainingJobResult job_res;
	bool any_failed = !AnchoredSlotClassifier::RunTrainingJob(
		lay, req, job_res,
		[](String slot, int epoch, double loss, double acc) {
			Cout() << "Slot: " << slot << " Epoch: " << epoch
			       << " Loss: " << loss << " Acc: " << acc << "\n";
		},
		nullptr, &mdl);

	for(int gi = 0; gi < job_res.groups.GetCount(); gi++) {
		const TrainingGroupResult& gr = job_res.groups[gi];
		if(!gr.ok)
			continue;

		// Collect slot ids for trainstate
		Vector<String> slot_ids;
		int g = all_groups.Find(gr.group_key);
		if(g >= 0) {
			for(const String& id : all_groups[g]) {
				const AnnLaySlot* slot = lay.FindSlot(id);
				if(slot && (slot->method == ANNLAY_CLASSIFIER_LABEL || slot->method == ANNLAY_CLASSIFIER_BOOL))
					slot_ids.Add(id);
			}
		}

		ConvNet::AiTrainState state;
		state.group_key = gr.group_key;
		state.slot_ids = pick(slot_ids);
		state.epochs_completed = gr.epochs_done;
		state.target_epochs = args.crops_epochs;
		state.timestamp = MakeIsoTimestamp();
		state.loss_history = clone(gr.loss_history);
		state.val_acc_history = clone(gr.val_acc_history);
		if(!state.Save(args.crops_annlay_path)) {
			Cout() << "Warning: failed to save trainstate sidecar for group: " << gr.group_key << "\n";
		}
	}

	if(!lay.Save(args.crops_annlay_path)) {
		Cout() << "Failed to save: " << args.crops_annlay_path << "\n";
		return false;
	}
	if(!mdl.Save(args.crops_annlay_path)) {
		Cout() << "Failed to save annmdl: " << AnnMdl::PathFromAnnlay(args.crops_annlay_path) << "\n";
		return false;
	}

	if(any_failed) {
		Cout() << "Some groups failed. Saved partial results: " << args.crops_annlay_path << "\n";
		return false;
	}
	Cout() << "Training complete. Saved: " << args.crops_annlay_path << "\n";
	return true;
}

bool RunRecognizeSlots(const AppArguments& args) {
	AnchoredSlotRecognizer rec;
	rec.SetOffsetMode(args.offset_mode_arg);
	rec.SetBoolGatePolicy(args.bool_gate_policy_arg);
	if(!rec.Load(args.recognize_annlay_path)) {
		Cout() << "Failed to load annlay: " << args.recognize_annlay_path << "\n";
		return false;
	}
	Image img = LoadImageByExtension(args.recognize_image_path);
	if(img.IsEmpty()) {
		Cout() << "Failed to load image: " << args.recognize_image_path << "\n";
		return false;
	}
	Vector<SlotResult> results = rec.Recognize(img);
	ValueArray ja;
	for(int i = 0; i < results.GetCount(); i++) {
		ValueMap jo;
		jo.Add("slot_id", results[i].slot_id);
		jo.Add("method", results[i].method);
		jo.Add("raw_text", results[i].raw_text);
		jo.Add("class_index", results[i].class_index);
		jo.Add("confidence", results[i].confidence);
		ValueMap bbox;
		bbox.Add("left", results[i].pixel_bbox.left);
		bbox.Add("top", results[i].pixel_bbox.top);
		bbox.Add("right", results[i].pixel_bbox.right);
		bbox.Add("bottom", results[i].pixel_bbox.bottom);
		jo.Add("pixel_bbox", bbox);
		jo.Add("top_class", results[i].top_class);
		jo.Add("offset_dx", results[i].offset_dx);
		jo.Add("offset_dy", results[i].offset_dy);
		jo.Add("gate_slot_id", results[i].gate_slot_id);
		jo.Add("gate_status", results[i].gate_status);
		ja.Add(jo);
	}
	Cout() << StoreAsJson(ja, true) << "\n";
	return true;
}

bool RunTrainHeadless(const AppArguments& args) {
	AnnSln sln;
	if(!sln.Load(args.train_headless_sln_path)) {
		Cout() << "Error: failed to load .annsln: " << args.train_headless_sln_path << "\n";
		return false;
	}

	String annlay_path = ResolveFromSlnDir(GetFileDirectory(args.train_headless_sln_path), sln.annlay);
	AnnLay lay;
	if(!lay.Load(annlay_path)) {
		Cout() << "Error: failed to load .annlay: " << annlay_path << "\n";
		return false;
	}

	AnnMdl mdl;
	if(!mdl.Load(annlay_path)) {
		Cout() << "Error: failed to load .annmdl for " << annlay_path << "\n";
		return false;
	}

	Cout() << "[TrainHeadless] Starting training...\n";
	Cout() << "  SLN   : " << args.train_headless_sln_path << "\n";
	Cout() << "  Crops : " << args.train_headless_crops_dir << "\n";
	Cout() << "  Epochs: " << args.train_headless_epochs << "\n";
	if(!args.train_headless_slot_key.IsEmpty())
		Cout() << "  Slot  : " << args.train_headless_slot_key << "\n";

	VectorMap<String, Vector<String>> model_groups = AnchoredSlotClassifier::GetSlotGroups(lay);
	Vector<String> keys_to_train;
	if(!args.train_headless_slot_key.IsEmpty()) {
	        keys_to_train.Add(args.train_headless_slot_key);
	} else {
	        for(int i = 0; i < model_groups.GetCount(); i++)
	                keys_to_train.Add(model_groups.GetKey(i));
	}

	TrainingJobRequest req;
	req.source_mode = TRAIN_SOURCE_CROPS;
	req.group_keys = clone(keys_to_train);
	req.crops_dir = args.train_headless_crops_dir;
	req.max_epochs = args.train_headless_epochs;
	req.balance_by_slot = true;
	req.slot_cap = 0;
	req.ignore_auto_stop = args.train_headless_ignore_auto_stop;
	req.debug = args.train_headless_debug;

	TrainingJobResult job_res;
	bool all_ok = AnchoredSlotClassifier::RunTrainingJob(
		lay, req, job_res,
		[](String slot, int epoch, double loss, double acc) {
			Cout() << Format("Slot: %-20s Epoch: %3d Loss: %10.6f Acc: %5.3f\n", slot, epoch, loss, acc);
		},
		nullptr, &mdl);

	for(int i = 0; i < job_res.groups.GetCount(); i++) {
		if(!job_res.groups[i].ok)
			Cout() << "[TrainHeadless] Group failed: " << job_res.groups[i].group_key << "\n";
	}

	if(all_ok) {

		if(!mdl.Save(annlay_path)) {
			Cout() << "Error: failed to save trained models to " << AnnMdl::PathFromAnnlay(annlay_path) << "\n";
			return false;
		}
		Cout() << "[TrainHeadless] Success. Models saved.\n";
	} else {
	        Cout() << "[TrainHeadless] Failed.\n";
	}
	return all_ok;
}


bool RunExportCrops(const AppArguments& args) {
	String sln_path = NormalizePath(args.export_sln_path);
	if(sln_path.IsEmpty()) {
		Cout() << "export-crops: missing .annsln path\n";
		return false;
	}

	AnnSln sln;
	if(!sln.Load(sln_path)) {
		Cout() << "export-crops: failed to load .annsln: " << sln_path << "\n";
		return false;
	}

	String sln_dir = GetFileDirectory(sln_path);
	String annlay_path = ResolveFromBase(sln_dir, sln.annlay);
	String annprj_path = ResolveFromBase(sln_dir, sln.annprj);
	String images_dir = ResolveFromBase(sln_dir, sln.images_dir);
	String templates_dir = ResolveFromBase(sln_dir, sln.templates_dir);
	String crops_root = ResolveFromBase(sln_dir, sln.crops_dir);
	if(images_dir.IsEmpty() && !annprj_path.IsEmpty()) {
		String def_images = AppendFileName(GetFileDirectory(annprj_path), "images");
		if(DirectoryExists(def_images))
			images_dir = def_images;
	}

	if(annlay_path.IsEmpty() || !FileExists(annlay_path)) {
		Cout() << "export-crops: annlay file does not exist: " << annlay_path << "\n";
		return false;
	}
	if(annprj_path.IsEmpty() || !FileExists(annprj_path)) {
		Cout() << "export-crops: annprj file does not exist: " << annprj_path << "\n";
		return false;
	}
	if(images_dir.IsEmpty() || !DirectoryExists(images_dir)) {
		Cout() << "export-crops: images dir does not exist: " << images_dir << "\n";
		return false;
	}
	if(crops_root.IsEmpty()) {
		Cout() << "export-crops: crops dir is not configured in .annsln\n";
		return false;
	}
	if(args.export_pass == 1 && (templates_dir.IsEmpty() || !DirectoryExists(templates_dir))) {
		Cout() << "export-crops: templates dir does not exist: " << templates_dir << "\n";
		return false;
	}

	AnnLay lay;
	if(!lay.Load(annlay_path)) {
		Cout() << "export-crops: failed to load annlay: " << annlay_path << "\n";
		return false;
	}

	AnchoredSlotExporter exporter;
	AnchoredSlotExportResult result = exporter.Export(lay, annprj_path, images_dir, templates_dir, crops_root, args.export_pass, args.offset_style_arg);
	if(!result.error.IsEmpty()) {
		Cout() << "export-crops: failed: " << result.error << "\n";
		return false;
	}
	// Keep cv_template_match template assets in the same export workflow.
	ExportCvTemplatesToCrops(sln, crops_root, args.export_pass, result.samples_per_group);

	Cout() << "export-crops: success images_processed=" << result.images_processed << "\n";
	for(int i = 0; i < result.samples_per_group.GetCount(); i++) {
		Cout() << "  group[" << result.samples_per_group.GetKey(i) << "]="
		       << result.samples_per_group[i] << "\n";
	}
	return true;
}

bool RunMineBoolHardNegatives(const AppArguments& args) {
	String sln_path = NormalizePath(args.mine_sln_path);
	if(sln_path.IsEmpty()) return false;

	AnnSln sln;
	if(!sln.Load(sln_path)) return false;

	String sln_dir = GetFileDirectory(sln_path);
	String annlay_path = ResolveFromBase(sln_dir, sln.annlay);
	String annprj_path = ResolveFromBase(sln_dir, sln.annprj);
	String images_dir = ResolveFromBase(sln_dir, sln.images_dir);
	if(images_dir.IsEmpty())
		images_dir = AppendFileName(GetFileDirectory(annprj_path), "images");

	String annmdl_path;
	int pass1 = sln.model_sets.Find("pass1");
	if(pass1 >= 0 && !TrimBoth(sln.model_sets[pass1]).IsEmpty())
		annmdl_path = ResolveFromBase(sln_dir, sln.model_sets[pass1]);
	if(annmdl_path.IsEmpty())
		annmdl_path = AnnMdl::PathFromAnnlay(annlay_path);

	if(!FileExists(annmdl_path)) return false;

	AnnLay lay;
	if(!lay.Load(annlay_path)) return false;

	uint64 load_start = GetTickCount();
	AnchoredSlotRecognizer rec;
	rec.SetOffsetMode(args.offset_mode_arg);

	Index<String> h_filter, s_filter;
	for(int i = 0; i < lay.slots.GetCount(); i++) {
		if(AnchoredSlotClassifier::BoolSlotGroupKey(lay.slots[i].id, &lay) ==
 args.mine_group) {
			s_filter.Add(lay.slots[i].id);
			h_filter.Add(lay.slots[i].id);
		}
	}
	if(s_filter.IsEmpty()) {
		Cout() << "Error: group '" << args.mine_group << "' contains no slots in layout.\n";
		return false;
	}
	h_filter.Add(args.mine_group); // CRITICAL: load shared session
	rec.SetHeadFilter(h_filter);
	rec.SetSlotFilter(s_filter);

	if(!rec.Load(annlay_path, annmdl_path)) return false;
	uint64 load_end = GetTickCount();

	Value root = ParseJSON(LoadFile(annprj_path));
	if(IsNull(root) || !IsValueMap(root)) return false;
	Value datasets = root["datasets"];
	if(!IsValueArray(datasets)) return false;

	String out_dir = args.mine_out_dir;
	String false_dir = AppendFileName(out_dir, "false");
	RealizeDirectory(false_dir);

	int total_imgs = 0;
	for(int d = 0; d < datasets.GetCount(); d++)
		if(IsValueArray(datasets[d]["images"])) total_imgs += datasets[d]["images"].GetCount();
	
	int kLimit = args.mine_all ? total_imgs : args.mine_count;
	int processed_imgs = 0;
	int total_fp = 0;

	Cout() << "Mining hard negatives from " << kLimit << " images for group '" << args.mine_group << "'...\n";

	uint64 start_ts = GetTickCount();

	for(int di = 0; di < datasets.GetCount() && processed_imgs < kLimit; di++) {
		Value images = datasets[di]["images"];
		if(!IsValueArray(images)) continue;
		for(int ii = 0; ii < images.GetCount() && processed_imgs < kLimit; ii++) {
			Value img_rec = images[ii];
			String img_path = ResolveImagePathForSmoke(sln_dir, annprj_path, images_dir, img_rec);
			if(img_path.IsEmpty() || !FileExists(img_path)) continue;
			Image img = LoadImageByExtension(img_path);
			if(img.IsEmpty()) continue;

			String stem = GetFileTitle(img_path);
			Vector<SlotResult> results = rec.Recognize(img);
			Value annotations = img_rec["annotations"];

			for(int ri = 0; ri < results.GetCount(); ri++) {
				const SlotResult& res = results[ri];
				
				String gkey = AnchoredSlotClassifier::BoolSlotGroupKey(res.slot_id, &rec.GetLayout());
				if(gkey != args.mine_group) continue;

				bool gt_present = false;
				if(IsValueArray(annotations)) {
					for(int ai = 0; ai < annotations.GetCount(); ai++) {
						String ann_slot = annotations[ai]["metadata"]["mlui_slot_id"];
						Value polygons = annotations[ai]["polygons"];
						bool has_bbox = IsValueArray(polygons) && polygons.GetCount() > 0 &&
						                IsValueArray(polygons[0]) && polygons[0].GetCount() > 0;
						if(has_bbox && AnchoredSlotClassifier::MatchSlotForBool(res.slot_id, ann_slot)) {
							gt_present = true;
							break;
						}
					}
				}

				const AnnLaySlot* slot = rec.GetLayout().FindSlot(res.slot_id);
				double threshold = args.mine_threshold >= 0 ? args.mine_threshold : (slot ? slot->presence_threshold : 0.5);
				bool pred_present = res.confidence >= threshold;

				if(pred_present && !gt_present) {
					Rect bbox = res.pixel_bbox;
					if(bbox.IsEmpty() && slot)
						bbox = rec.AnchorToRect(slot->anchor, img.GetSize(), res.offset_dx, res.offset_dy);

					Image crop = Crop(img, bbox);
					if(!crop.IsEmpty()) {
						if(slot && (crop.GetWidth() != slot->crop_size.cx || crop.GetHeight() != slot->crop_size.cy))
							crop = Rescale(crop, slot->crop_size);

						static int log_count = 0;
						if(log_count++ < 10)
							Cout() << "DEBUG: saving FP crop for " << res.slot_id << " from " << stem << " bbox=" << bbox << "\n";

						String fname = Format("%s_%s_fp.png", res.slot_id, stem);
						String bool_group = slot ? AnchoredSlotClassifier::BoolSlotGroupKey(slot->id, &lay) : String();
						bool gray = (slot && slot->color_mode != "color") ||
						            bool_group == "card_visibility_gate" ||
						            bool_group == "action_button_gate";
						PNGEncoder().SaveFile(AppendFileName(false_dir, fname), gray ? Grayscale(crop) : crop);

						total_fp++;
					}
				}
			}
			processed_imgs++;
			if(processed_imgs % 10 == 0) Cout() << "Processed " << processed_imgs << " images...\n";
		}
	}

	uint64 end_ts = GetTickCount();
	double load_time = (load_end - load_start) / 1000.0;
	double eval_time = (end_ts - start_ts) / 1000.0;
	double total_time = load_time + eval_time;

	Cout() << "Mining complete. Total False Positives saved: " << total_fp << " to " << false_dir << "\n";
	Cout() << "  Model load:   " << Format("%.2f", load_time) << " s\n";
	Cout() << "  Eval loop:    " << Format("%.2f", eval_time) << " s\n";
	Cout() << "  Total time:   " << Format("%.2f", total_time) << " s\n";
	if(eval_time > 0)
		Cout() << "  Throughput:   " << Format("%.2f", processed_imgs / eval_time) << " images/sec\n";
	return true;
}

bool RunMergeHardneg(const AppArguments& args) {
	String base_false = AppendFileName(args.merge_base_dir, "false");
	String hn_false = AppendFileName(args.merge_hardneg_dir, "false");

	if(!DirectoryExists(base_false)) {
		Cout() << "merge-hardneg: base 'false' directory not found: " << base_false << "\n";
		return false;
	}
	if(!DirectoryExists(hn_false)) {
		Cout() << "merge-hardneg: hardneg 'false' directory not found: " << hn_false << "\n";
		return false;
	}

	int merged = 0;
	int skipped = 0;

	FindFile ff;
	if(ff.Search(AppendFileName(hn_false, "*.png"))) {
		do {
			String name = ff.GetName();
			String src = ff.GetPath();
			String dst = AppendFileName(base_false, name);

			if(FileExists(dst)) {
				String title = GetFileTitle(name);
				String ext = GetFileExt(name);
				bool found_unique = false;
				for(int i = 1; i < 100; i++) {
					String new_name = title + "_" + AsString(i) + ext;
					dst = AppendFileName(base_false, new_name);
					if(!FileExists(dst)) {
						found_unique = true;
						break;
					}
				}
				if(!found_unique) {
					skipped++;
					continue;
				}
			}

			if(args.merge_dry_run) {
				Cout() << "[DRY-RUN] Would copy: " << name << "\n";
			}
			else {
				if(FileCopy(src, dst))
					merged++;
				else
					skipped++;
			}
		} while(ff.Next());
	}

	Cout() << (args.merge_dry_run ? "[DRY-RUN] " : "")
	       << "Merge complete. Merged: " << merged << ", Skipped: " << skipped << "\n";
	return true;
}

bool RunSaveOffsetBaseline(const AppArguments& args) {
	OffsetBaseline bl;
	if(!ComputeOffsetMetrics(args.save_baseline_sln_path, args.split_file, args.subset, 
	                         args.eval_count, args.eval_all, bl)) {
		Cout() << "save-offset-baseline: failed to compute metrics\n";
		return false;
	}

	if(StoreAsJsonFile(bl, args.save_offset_baseline_out_path, true)) {
		Cout() << "Successfully saved offset baseline to: " << args.save_offset_baseline_out_path << "\n";
		return true;
	}
	else {
		Cout() << "save-offset-baseline: failed to save output file\n";
		return false;
	}
}

bool RunCheckOffsetBaseline(const AppArguments& args) {
	OffsetBaseline baseline;
	if(!LoadFromJsonFile(baseline, args.check_offset_baseline_in_path)) {
		Cout() << "check-offset-baseline: failed to load baseline file: " << args.check_offset_baseline_in_path << "\n";
		return false;
	}

	String effective_split = args.split_file;
	if(effective_split.IsEmpty()) {
		effective_split = baseline.split_file;
		if(!effective_split.IsEmpty())
			Cout() << "Note: --split-file omitted, auto-using baseline split: " << effective_split << "\n";
	}

	OffsetBaseline current;
	if(!ComputeOffsetMetrics(args.check_baseline_sln_path, effective_split, args.subset,
	                         args.eval_count, args.eval_all, current)) {
		Cout() << "check-offset-baseline: failed to compute current metrics\n";
		return false;
	}

	OffsetCheckReport check_rep;
	check_rep.max_meanl1_rise = args.max_meanl1_rise;
	check_rep.max_p90_rise = args.max_p90_rise;

	bool pass = true;

	if(!CheckBaselineMismatch("check-offset-baseline", baseline.annlay_path, current.annlay_path,
	                          baseline.annmdl_path, current.annmdl_path,
	                          baseline.model_set, current.model_set,
	                          args.allow_baseline_mismatch, check_rep.mismatch_reasons, check_rep.baseline_mismatch)) {
		if(!args.check_offset_out_path.IsEmpty()) {
			check_rep.passed = false;
			StoreAsJsonFile(check_rep, args.check_offset_out_path, true);
		}
		return false;
	}

	String console_report;

	auto Compare = [&](const char* name, double cur, double base, double max_rise) {
		double delta = cur - base;
		bool fail = (delta > max_rise);
		
		String line = Format("  %-20s: cur=%7.2f base=%7.2f delta=%+7.2f (max_rise=%7.2f)", 
		                     name, cur, base, delta, max_rise);

		ValueMap m;
		m.Add("current", cur);
		m.Add("baseline", base);
		m.Add("delta", delta);
		m.Add("passed", !fail);
		check_rep.metrics.Add(name, m);

		if(fail) {
			line << " [FAIL]";
			pass = false;
			check_rep.failure_reasons.Add(Format("%s: delta %+7.2f exceeds max_rise %7.2f", name, delta, max_rise));
		}
		else {
			line << " [PASS]";
		}
		console_report << line << "\n";
	};

	console_report << "\nOffset Regression Check Summary (AUTO Mode):\n";
	console_report << "--------------------------------------------------------------------------------\n";
	
	OffsetMetrics *cur_auto = current.modes.FindPtr("auto");
	OffsetMetrics *base_auto = baseline.modes.FindPtr("auto");

	if(cur_auto && base_auto) {
		Compare("MeanL1", cur_auto->mean_l1, base_auto->mean_l1, args.max_meanl1_rise);
		Compare("P90_L1", cur_auto->p90_l1, base_auto->p90_l1, args.max_p90_rise);
	}
	else {
		pass = false;
		console_report << "ERROR: 'auto' mode metrics missing in either current or baseline run.\n";
		check_rep.failure_reasons.Add("Missing 'auto' mode metrics.");
	}
	console_report << "--------------------------------------------------------------------------------\n";

	Cout() << console_report;
	check_rep.passed = pass;

	if(!args.check_offset_out_path.IsEmpty()) {
		if(StoreAsJsonFile(check_rep, args.check_offset_out_path, true))
			Cout() << "Check report saved to: " << args.check_offset_out_path << "\n";
	}

	if(pass) {
		Cout() << "\nREGRESSION CHECK PASSED.\n";
	}
	else {
		Cout() << "\nREGRESSION CHECK FAILED!\n";
	}

	return pass;
}

bool RunEvalOffsetHeads(const AppArguments& args) {
	OffsetBaseline bl;
	if(!ComputeOffsetMetrics(args.eval_sln_path, args.split_file, args.subset,
	                         args.eval_count, args.eval_all, bl)) {
		Cout() << "eval-offset-heads: failed to compute metrics\n";
		return false;
	}

	int images = (bl.train_image_count + bl.holdout_image_count);
	if(args.subset == "train") images = bl.train_image_count;
	else if(args.subset == "holdout") images = bl.holdout_image_count;

	Cout() << "\nOffset Head Evaluation Results (" << images << " images, Subset: " << args.subset << "):\n";
	Cout() << "----------------------------------------------------------------------\n";
	Cout() << Format("%-10s | %-8s | %-7s | %-7s | %-7s | %-7s\n", 
	                 "Mode", "Samples", "MAE_X", "MAE_Y", "MeanL1", "P90_L1");
	Cout() << "----------------------------------------------------------------------\n";

	for(int i = 0; i < bl.modes.GetCount(); i++) {
		const OffsetMetrics& m = bl.modes[i];
		Cout() << Format("%-10s | %8d | %7.2f | %7.2f | %7.2f | %7.2f\n",
		                 m.name, m.samples, m.mae_x, m.mae_y, m.mean_l1, m.p90_l1);
	}
	Cout() << "----------------------------------------------------------------------\n";

	if(!args.report_bool_out_path.IsEmpty()) {
		if(StoreAsJsonFile(bl, args.report_bool_out_path, true))
			Cout() << "Report saved to: " << args.report_bool_out_path << "\n";
	}

	return true;
}

bool RunTuneBoolGates(const AppArguments& args) {
	String sln_path = NormalizePath(args.tune_sln_path);
	if(sln_path.IsEmpty()) return false;

	AnnSln sln;
	if(!sln.Load(sln_path)) return false;

	String sln_dir = GetFileDirectory(sln_path);
	String annlay_path = ResolveFromBase(sln_dir, sln.annlay);
	String annprj_path = ResolveFromBase(sln_dir, sln.annprj);
	String images_dir = ResolveFromBase(sln_dir, sln.images_dir);
	if(images_dir.IsEmpty())
		images_dir = AppendFileName(GetFileDirectory(annprj_path), "images");

	String annmdl_path;
	int pass1 = sln.model_sets.Find("pass1");
	if(pass1 >= 0 && !TrimBoth(sln.model_sets[pass1]).IsEmpty())
		annmdl_path = ResolveFromBase(sln_dir, sln.model_sets[pass1]);
	if(annmdl_path.IsEmpty())
		annmdl_path = AnnMdl::PathFromAnnlay(annlay_path);

	if(!FileExists(annmdl_path)) return false;

	AnnLay lay;
	if(!lay.Load(annlay_path)) return false;

	uint64 load_start = GetTickCount();
	AnchoredSlotRecognizer rec;

	Index<String> h_filter, s_filter;
	for(int i = 0; i < lay.slots.GetCount(); i++) {
		if(AnchoredSlotClassifier::BoolSlotGroupKey(lay.slots[i].id, &lay) ==
 args.tune_group) {
			s_filter.Add(lay.slots[i].id);
			h_filter.Add(lay.slots[i].id);
		}
	}
	if(s_filter.IsEmpty()) {
		Cout() << "Error: group '" << args.tune_group << "' contains no slots in layout.\n";
		return false;
	}
	h_filter.Add(args.tune_group); // CRITICAL: load shared session
	rec.SetHeadFilter(h_filter);
	rec.SetSlotFilter(s_filter);

	if(!rec.Load(annlay_path, annmdl_path)) return false;
	uint64 load_end = GetTickCount();

	Value root = ParseJSON(LoadFile(annprj_path));
	if(IsNull(root) || !IsValueMap(root)) return false;
	Value datasets = root["datasets"];
	if(!IsValueArray(datasets)) return false;

	Index<String> subset_ids;
	if(!LoadSubsetIds(args.split_file, args.subset, subset_ids)) return false;

	struct ScorePair : Moveable<ScorePair> {
		double score;
		bool gt;
	};
	VectorMap<String, Vector<ScorePair>> score_data;

	int total_imgs = 0;
	for(int d = 0; d < datasets.GetCount(); d++)
		if(IsValueArray(datasets[d]["images"])) total_imgs += datasets[d]["images"].GetCount();
	
	int kLimit = args.tune_all ? total_imgs : args.tune_count;
	int processed_imgs = 0;

	Cout() << "Collecting scores from " << kLimit << " images for group '" << args.tune_group << "' (Subset: " << args.subset << ")...\n";

	uint64 start_ts = GetTickCount();

	for(int di = 0; di < datasets.GetCount() && processed_imgs < kLimit; di++) {
		Value images = datasets[di]["images"];
		if(!IsValueArray(images)) continue;
		for(int ii = 0; ii < images.GetCount() && processed_imgs < kLimit; ii++) {
			Value img_rec = images[ii];
			String id = img_rec["file_name"];
			if(id.IsEmpty()) id = img_rec["file_path"];
			if(id.IsEmpty()) id = img_rec["filename"];
			
			if(!subset_ids.IsEmpty() && subset_ids.Find(id) < 0)
				continue;

			String img_path = ResolveImagePathForSmoke(sln_dir, annprj_path, images_dir, img_rec);
			if(img_path.IsEmpty() || !FileExists(img_path)) continue;
			Image img = LoadImageByExtension(img_path);
			if(img.IsEmpty()) continue;

			Vector<SlotResult> results = rec.Recognize(img);
			Value annotations = img_rec["annotations"];

			for(int ri = 0; ri < results.GetCount(); ri++) {
				const SlotResult& res = results[ri];
				
				String gkey = AnchoredSlotClassifier::BoolSlotGroupKey(res.slot_id, &rec.GetLayout());
				if(gkey != args.tune_group) continue;

				bool gt_present = false;
				if(IsValueArray(annotations)) {
					for(int ai = 0; ai < annotations.GetCount(); ai++) {
						String ann_slot = annotations[ai]["metadata"]["mlui_slot_id"];
						if(ann_slot.IsEmpty()) ann_slot = annotations[ai]["slot_id"];
						
						if(AnchoredSlotClassifier::MatchSlotForBool(res.slot_id, ann_slot)) {
							gt_present = true;
							break;
						}
					}
				}
				
				if(gt_present) {
					static int positive_count = 0;
					if(positive_count++ < 10)
						Cout() << "DEBUG: " << res.slot_id << " matched GT positive, conf=" << res.confidence << "\n";
				}

				ScorePair& p = score_data.GetAdd(res.slot_id).Add();
				p.score = res.confidence;
				p.gt = gt_present;
			}
			processed_imgs++;
			if(processed_imgs % 10 == 0) Cout() << "Processed " << processed_imgs << " images...\n";
		}
	}

	uint64 end_ts = GetTickCount();
	double load_time = (load_end - load_start) / 1000.0;
	double eval_time = (end_ts - start_ts) / 1000.0;
	double total_time = load_time + eval_time;

	Cout() << "Score collection complete.\n";
	Cout() << "  Model load:   " << Format("%.2f", load_time) << " s\n";
	Cout() << "  Eval loop:    " << Format("%.2f", eval_time) << " s\n";
	Cout() << "  Total time:   " << Format("%.2f", total_time) << " s\n";
	if(eval_time > 0)
		Cout() << "  Throughput:   " << Format("%.2f", processed_imgs / eval_time) << " images/sec\n";

	if(score_data.IsEmpty()) {
		Cout() << "No data collected for group '" << args.tune_group << "'.\n";
		return false;
	}

	auto CalcScore = [&](double prec, double recall) {
		if(args.tune_optimize == "precision") return prec;
		if(args.tune_optimize == "f05") {
			if(prec + recall == 0) return 0.0;
			return (1 + 0.5 * 0.5) * (prec * recall) / (0.5 * 0.5 * prec + recall);
		}
		if(prec + recall == 0) return 0.0;
		return 2 * (prec * recall) / (prec + recall);
	};

	auto FindBestThresh = [&](const Vector<ScorePair>& pairs, double& out_prec, double& out_recall, double& out_f1) {
		double best_metric = -1;
		double best_recall = -1;
		double best_thresh = 0.5;
		
		out_prec = 0; out_recall = 0; out_f1 = 0;

		double step = max(0.01, args.tune_max_step);
		for(double th = 0.05; th <= 0.95; th += step) {
			int tp = 0, fp = 0, tn = 0, fn = 0;
			for(const auto& p : pairs) {
				bool pred = p.score >= th;
				if(p.gt && pred) tp++;
				else if(!p.gt && pred) fp++;
				else if(!p.gt && !pred) tn++;
				else if(p.gt && !pred) fn++;
			}
			double prec = (tp + fp) > 0 ? (double)tp / (tp + fp) : 0;
			double recall = (tp + fn) > 0 ? (double)tp / (tp + fn) : 0;
			double f1 = (prec + recall) > 0 ? 2 * (prec * recall) / (prec + recall) : 0;
			
			if(recall < args.tune_min_recall) continue;

			double metric = CalcScore(prec, recall);
			if(metric > best_metric || (metric == best_metric && recall > best_recall)) {
				best_metric = metric;
				best_recall = recall;
				best_thresh = th;
				out_prec = prec;
				out_recall = recall;
				out_f1 = f1;
			}
		}
		return best_thresh;
	};

	VectorMap<String, double> best_thresholds;

	if(args.tune_per_slot) {
		Cout() << "\nPer-Slot Threshold Sweep Results (min_recall=" << args.tune_min_recall << ", opt=" << args.tune_optimize << "):\n";
		Cout() << "----------------------------------------------------------------------\n";
		Cout() << Format("%-25s | %-7s | %-7s | %-7s | %-7s\n", "Slot ID", "Thresh", "Prec", "Recall", "F1");
		Cout() << "----------------------------------------------------------------------\n";
		for(int i = 0; i < score_data.GetCount(); i++) {
			double p, r, f;
			double best_th = FindBestThresh(score_data[i], p, r, f);
			best_thresholds.Add(score_data.GetKey(i), best_th);
			Cout() << Format("%-25s | %7.2f | %7.2f | %7.2f | %7.2f\n", score_data.GetKey(i), best_th, p, r, f);
		}
		Cout() << "----------------------------------------------------------------------\n";
	}
	else {
		Vector<ScorePair> all_pairs;
		for(int i = 0; i < score_data.GetCount(); i++)
			for(const auto& p : score_data[i])
				all_pairs.Add(p);
		
		double p, r, f;
		double best_th = FindBestThresh(all_pairs, p, r, f);
		Cout() << "\nGlobal Threshold Sweep (min_recall=" << args.tune_min_recall << ", opt=" << args.tune_optimize << "):\n";
		Cout() << "Best Threshold: " << best_th << " (Prec=" << p << ", Recall=" << r << ", F1=" << f << ")\n";
		
		for(int i = 0; i < score_data.GetCount(); i++)
			best_thresholds.Add(score_data.GetKey(i), best_th);
	}

	if(args.tune_apply) {
		Cout() << "Applying thresholds to .annlay...\n";
		AnnLay lay;
		if(lay.Load(annlay_path)) {
			int count = 0;
			for(auto& slot : lay.slots) {
				int q = best_thresholds.Find(slot.id);
				if(q >= 0) {
					double old_th = slot.presence_threshold;
					double new_th = best_thresholds[q];
					if(abs(old_th - new_th) > 0.001) {
						Cout() << "  - " << slot.id << ": " << old_th << " -> " << new_th << "\n";
						slot.presence_threshold = new_th;
						count++;
					}
				}
			}
			if(count > 0) {
				if(lay.Save(annlay_path))
					Cout() << "Successfully updated " << count << " slots in " << annlay_path << "\n";
				else Cout() << "Failed to save updated .annlay!\n";
			}
			else Cout() << "No thresholds changed.\n";
		}
		else Cout() << "Failed to load .annlay for applying thresholds!\n";
	}

	return true;
}

bool RunSaveBoolBaseline(const AppArguments& args) {
	BoolBaseline bl;
	if(!ComputeBoolGateMetrics(args.save_baseline_sln_path, args.split_file, args.tune_group, 
	                           args.eval_count, args.eval_all, "all", args.offset_mode_arg, args.bool_gate_policy_arg, bl)) {
		Cout() << "save-bool-baseline: failed to compute metrics\n";
		return false;
	}

	if(StoreAsJsonFile(bl, args.save_baseline_out_path, true)) {
		Cout() << "Successfully saved baseline to: " << args.save_baseline_out_path << "\n";
		return true;
	}
	else {
		Cout() << "save-bool-baseline: failed to save output file\n";
		return false;
	}
}

bool RunCheckBoolBaseline(const AppArguments& args) {
	BoolBaseline baseline;
	if(!LoadFromJsonFile(baseline, args.check_baseline_in_path)) {
		Cout() << "check-bool-baseline: failed to load baseline file: " << args.check_baseline_in_path << "\n";
		return false;
	}

	String effective_split = args.split_file;
	if(effective_split.IsEmpty()) {
		effective_split = baseline.split_file;
		if(!effective_split.IsEmpty())
			Cout() << "Note: --split-file omitted, auto-using baseline split: " << effective_split << "\n";
	}

	if(!args.tune_group.IsEmpty() && baseline.group != args.tune_group) {
		Cout() << "WARNING: Group mismatch! CLI=" << args.tune_group << ", Baseline=" << baseline.group << "\n";
	}

	BoolBaseline current;
	if(!ComputeBoolGateMetrics(args.check_baseline_sln_path, effective_split, baseline.group,
	                           args.eval_count, args.eval_all, "all", args.offset_mode_arg, args.bool_gate_policy_arg, current)) {
		Cout() << "check-bool-baseline: failed to compute current metrics\n";
		return false;
	}

	BoolCheckReport check_rep;
	check_rep.max_f1_drop = args.max_f1_drop;
	check_rep.max_recall_drop = args.max_recall_drop;
	check_rep.min_holdout_f1 = args.min_holdout_f1;

	bool pass = true;

	if(!CheckBaselineMismatch("check-bool-baseline", baseline.annlay_path, current.annlay_path,
	                          baseline.annmdl_path, current.annmdl_path,
	                          baseline.model_set, current.model_set,
	                          args.allow_baseline_mismatch, check_rep.mismatch_reasons, check_rep.baseline_mismatch)) {
		if(!args.check_bool_out_path.IsEmpty()) {
			check_rep.passed = false;
			StoreAsJsonFile(check_rep, args.check_bool_out_path, true);
		}
		return false;
	}

	String console_report;

	// Check for low support
	if(args.min_bool_support > 0) {
		check_rep.min_bool_support = args.min_bool_support;
		check_rep.fail_on_low_support = args.fail_on_low_support_arg;
		
		for(int i = 0; i < current.per_slot.GetCount(); i++) {
			const BoolGateMetrics& m = current.per_slot[i];
			if(m.gt_pos < args.min_bool_support) {
				check_rep.low_support_count++;
				ValueMap ls;
				ls.Add("slot_id", m.slot_id);
				ls.Add("gt_pos", m.gt_pos);
				ls.Add("required_min", args.min_bool_support);
				check_rep.low_support_slots.Add(ls);
			}
		}
		
		if(check_rep.low_support_count > 0) {
			console_report << "\nWARNING: Low positive support detected in " << check_rep.low_support_count << " slots (min=" << args.min_bool_support << "):\n";
			for(int i = 0; i < check_rep.low_support_slots.GetCount(); i++) {
				ValueMap ls = check_rep.low_support_slots[i];
				console_report << Format("  - %-25s: gt_pos=%d\n", (String)ls["slot_id"], (int)ls["gt_pos"]);
			}
			
			if(args.fail_on_low_support_arg) {
				pass = false;
				check_rep.failure_reasons.Add(Format("Low support detected in %d slots", check_rep.low_support_count));
			}
		}
	}

	auto Compare = [&](const char* name, double cur, double base, double rule_val, bool is_abs_min = false) {
		double delta = cur - base;
		bool fail = is_abs_min ? (cur < rule_val) : (delta < -rule_val);
		
		String label = name;
		String line;
		if(is_abs_min) {
			line = Format("  %-20s: cur=%.4f min=%.4f", label, cur, rule_val);
		}
		else {
			line = Format("  %-20s: cur=%.4f base=%.4f delta=%+.4f", label, cur, base, delta);
		}

		ValueMap m;
		m.Add("current", cur);
		if(!is_abs_min) m.Add("baseline", base);
		m.Add("delta", delta);
		m.Add("passed", !fail);
		check_rep.metrics.Add(label, m);

		if(fail) {
			line << " [FAIL]";
			pass = false;
			check_rep.failure_reasons.Add(Format("%s: cur=%.4f vs %s=%.4f", 
				label, cur, is_abs_min ? "min" : "base", is_abs_min ? rule_val : base));
		}
		else {
			line << " [PASS]";
		}
		console_report << line << "\n";
	};

	console_report << "\nRegression Check Summary:\n";
	console_report << "--------------------------------------------------\n";
	Compare("Holdout Micro F1", current.holdout_micro.f1, baseline.holdout_micro.f1, args.max_f1_drop);
	Compare("Holdout Micro Rec", current.holdout_micro.recall, baseline.holdout_micro.recall, args.max_recall_drop);
	Compare("Holdout Micro F1 (Min)", current.holdout_micro.f1, 0, args.min_holdout_f1, true);
	Compare("Train Micro F1", current.train_micro.f1, baseline.train_micro.f1, args.max_f1_drop);
	console_report << "--------------------------------------------------\n";

	struct Degradation : Moveable<Degradation> {
		String id;
		double drop;
		bool operator<(const Degradation& b) const { return drop > b.drop; }
	};
	Vector<Degradation> degraded;

	for(int i = 0; i < current.per_slot.GetCount(); i++) {
		String sid = current.per_slot.GetKey(i);
		int q = baseline.per_slot.Find(sid);
		if(q >= 0) {
			double cur_f1 = current.per_slot[i].f1;
			double base_f1 = baseline.per_slot[q].f1;
			if(cur_f1 < base_f1 - 0.01) {
				Degradation& d = degraded.Add();
				d.id = sid;
				d.drop = base_f1 - cur_f1;
				
				ValueMap ds;
				ds.Add("slot_id", sid);
				ds.Add("drop", d.drop);
				check_rep.degraded_slots.Add(ds);
			}
		}
	}

	if(!degraded.IsEmpty()) {
		Sort(degraded);
		console_report << "\nTop Degraded Slots (by F1 drop):\n";
		for(int i = 0; i < min(5, degraded.GetCount()); i++) {
			console_report << Format("  - %-25s: drop=%.4f\n", degraded[i].id, degraded[i].drop);
		}
	}

	Cout() << console_report;
	check_rep.passed = pass;

	if(!args.check_bool_out_path.IsEmpty()) {
		if(StoreAsJsonFile(check_rep, args.check_bool_out_path, true))
			Cout() << "Check report saved to: " << args.check_bool_out_path << "\n";
	}

	if(pass) {
		Cout() << "\nREGRESSION CHECK PASSED.\n";
	}
	else {
		Cout() << "\nREGRESSION CHECK FAILED!\n";
	}

	return pass;
}

bool RunReportBoolGates(const AppArguments& args) {
	BoolBaseline bl;
	if(!ComputeBoolGateMetrics(args.save_baseline_sln_path, args.split_file, "chip_box_gate", 
	                           0, true, "all", args.offset_mode_arg, args.bool_gate_policy_arg, bl)) {
		return false;
	}
	return StoreAsJsonFile(bl, args.report_bool_out_path, true);
}

bool RunEvalBoolGates(const AppArguments& args) {
	String sln_path = NormalizePath(args.eval_sln_path);
	if(sln_path.IsEmpty()) {
		Cout() << "eval-bool-gates: missing .annsln path\n";
		return false;
	}

	AnnSln sln;
	if(!sln.Load(sln_path)) {
		Cout() << "eval-bool-gates: failed to load .annsln: " << sln_path << "\n";
		return false;
	}

	String sln_dir = GetFileDirectory(sln_path);
	String annlay_path = ResolveFromBase(sln_dir, sln.annlay);
	String annprj_path = ResolveFromBase(sln_dir, sln.annprj);
	String images_dir = ResolveFromBase(sln_dir, sln.images_dir);
	if(images_dir.IsEmpty())
		images_dir = AppendFileName(GetFileDirectory(annprj_path), "images");

	String annmdl_path;
	int pass1 = sln.model_sets.Find("pass1");
	if(pass1 >= 0 && !TrimBoth(sln.model_sets[pass1]).IsEmpty())
		annmdl_path = ResolveFromBase(sln_dir, sln.model_sets[pass1]);
	if(annmdl_path.IsEmpty())
		annmdl_path = AnnMdl::PathFromAnnlay(annlay_path);

	if(!FileExists(annmdl_path)) {
		Cout() << "eval-bool-gates: missing .annmdl: " << annmdl_path << "\n";
		return false;
	}

	AnchoredSlotRecognizer rec;
	rec.SetOffsetMode(args.offset_mode_arg);
	if(!rec.Load(annlay_path, annmdl_path)) {
		Cout() << "eval-bool-gates: failed to load recognizer\n";
		return false;
	}

	BoolBaseline bl;
	if(!ComputeBoolGateMetrics(args.eval_sln_path, args.split_file, args.tune_group,
	                           args.eval_count, args.eval_all, args.subset, args.offset_mode_arg, args.bool_gate_policy_arg, bl)) {
		Cout() << "eval-bool-gates: failed to compute metrics\n";
		return false;
	}

	Cout() << "\nBool Gate Evaluation Results (" << bl.overall_micro.samples / max(1, (int)bl.per_slot.GetCount()) << " images):\n";
	Cout() << "---------------------------------------------------------------------------------------------------------------------------\n";
	Cout() << Format("%-25s | %-4s | %-7s | %-5s | %-15s | %-7s | %-7s | %-7s | %-7s\n", 
	                 "Slot ID", "Thr", "Samples", "GT+", "TP/FP/TN/FN", "Prec", "Recall", "F1", "Acc");
	Cout() << "---------------------------------------------------------------------------------------------------------------------------\n";

	int tuned_count = 0;
	double macro_f1 = 0;
	int n_f1 = 0;

	for(int i = 0; i < bl.per_slot.GetCount(); i++) {
		const BoolGateMetrics& m = bl.per_slot[i];
		if(abs(m.threshold - 0.5) > 0.001) tuned_count++;

		String cm = Format("%d/%d/%d/%d", m.tp, m.fp, m.tn, m.fn);
		Cout() << Format("%-25s | %4.2f | %7d | %5d | %-15s | %7.2f | %7.2f | %7.2f | %7.2f",
		                 m.slot_id, m.threshold, m.samples, m.gt_pos, cm, m.precision, m.recall, m.f1, m.accuracy);
		
		if(args.eval_min_positive > 0 && m.gt_pos < args.eval_min_positive) {
			Cout() << " [INSUFFICIENT POSITIVES]";
		}
		Cout() << "\n";
		
		if(m.precision > 0 || m.recall > 0) {
			macro_f1 += m.f1;
			n_f1++;
		}
	}

	Cout() << "---------------------------------------------------------------------------------------------------------------------------\n";
	
	const BoolGateMetrics& micro = bl.overall_micro;
	Cout() << Format("%-25s | %4s | %7d | %5d | %-15s | %7.2f | %7.2f | %7.2f | %7.2f (Micro)\n",
	                 "OVERALL", "", micro.samples, micro.gt_pos, Format("%d/%d/%d/%d", micro.tp, micro.fp, micro.tn, micro.fn),
	                 micro.precision, micro.recall, micro.f1, micro.accuracy);
	
	if(n_f1 > 0) {
		Cout() << Format("%-25s | %4s | %7s | %5s | %-15s | %7s | %7s | %7.2f | %7s (Macro F1)\n",
		                 "", "", "", "", "", "", "", macro_f1 / n_f1, "");
	}
	Cout() << "---------------------------------------------------------------------------------------------------------------------------\n";
	Cout() << "Evaluation complete. Tuned slots (Thr != 0.50): " << tuned_count << "\n";

	if(bl.train_micro.samples > 0 && bl.holdout_micro.samples > 0) {
		Cout() << "\nSplit Summary:\n";
		Cout() << Format("  Train F1:   %.4f (%d images)\n", bl.train_micro.f1, bl.train_micro.samples / max(1, (int)bl.per_slot.GetCount()));
		Cout() << Format("  Holdout F1: %.4f (%d images)\n", bl.holdout_micro.f1, bl.holdout_micro.samples / max(1, (int)bl.per_slot.GetCount()));
		
		double gap = bl.train_micro.f1 - bl.holdout_micro.f1;
		if(gap > 0.10) {
			Cout() << "\nWARNING: SIGNIFICANT OVERFIT DETECTED! (Holdout F1 is " 
			       << Format("%.2f", gap * 100.0) << "% lower than Train F1)\n";
		}
		else if(gap > 0.05) {
			Cout() << "\nNote: Slight overfit detected (gap > 0.05).\n";
		}
		else {
			Cout() << "\nGeneralization looks good (gap <= 0.05).\n";
		}
	}

	return true;
}

bool RunAuditOffsetHeads(const AppArguments& args) {
	String annlay_path = NormalizePath(args.audit_annlay_path);
	String annmdl_path = AnnMdl::PathFromAnnlay(annlay_path);

	if(!FileExists(annmdl_path)) {
		Cout() << "audit-offset-heads: .annmdl file not found: " << annmdl_path << "\n";
		return false;
	}

	AnnMdl mdl;
	if(!mdl.LoadPath(annmdl_path)) {
		Cout() << "audit-offset-heads: failed to load .annmdl: " << annmdl_path << "\n";
		return false;
	}

	int total = mdl.entries.GetCount();
	int n_comb = 0, n_offx = 0, n_offy = 0;

	struct SlotAudit {
		bool has_comb = false;
		bool has_offx = false;
		bool has_offy = false;
	};
	VectorMap<String, SlotAudit> slots;

	for(const auto& e : mdl.entries) {
		String id = e.slot_id;
		String base = id;
		int hash = id.Find('#');
		String suffix;
		if(hash >= 0) {
			base = id.Left(hash);
			suffix = id.Mid(hash);
		}

		SlotAudit& sa = slots.GetAdd(base);
		if(suffix == "#offset") { n_comb++; sa.has_comb = true; }
		else if(suffix == "#offset_x") { n_offx++; sa.has_offx = true; }
		else if(suffix == "#offset_y") { n_offy++; sa.has_offy = true; }
	}

	Cout() << "Offset Head Audit for: " << GetFileName(annlay_path) << "\n";
	Cout() << "--------------------------------------------------\n";
	Cout() << "Total AnnMdl entries: " << total << "\n";
	Cout() << "Combined (#offset):   " << n_comb << "\n";
	Cout() << "Split X (#offset_x):  " << n_offx << "\n";
	Cout() << "Split Y (#offset_y):  " << n_offy << "\n";
	Cout() << "--------------------------------------------------\n";

	int card_slots_complete = 0;
	int card_slots_total = 0;

	for(int i = 0; i < slots.GetCount(); i++) {
		const String& name = slots.GetKey(i);
		const SlotAudit& sa = slots[i];
		if(sa.has_comb || sa.has_offx || sa.has_offy) {
			Cout() << Format("%-25s: combined=%-3s split=%s/%s\n",
			                 name,
			                 sa.has_comb ? "yes" : "no",
			                 sa.has_offx ? "yes" : "no",
			                 sa.has_offy ? "yes" : "no");
			
			if(name.Find("element") >= 0) {
				card_slots_total++;
				if(sa.has_comb) card_slots_complete++;
			}
		}
	}

	Cout() << "--------------------------------------------------\n";
	if(card_slots_total > 0 && card_slots_complete == card_slots_total) {
		Cout() << "MIGRATION HINT: All element slots have combined heads. Safe to prune split heads.\n";
	}
	else if(card_slots_total > 0) {
		Cout() << "MIGRATION HINT: " << (card_slots_total - card_slots_complete) 
		       << " element slots still missing combined heads. Do not prune yet.\n";
	}

	return true;
}

bool RunPruneOffsetHeads(const AppArguments& args) {
	String annlay_path = NormalizePath(args.prune_annlay_path);
	String annmdl_path = AnnMdl::PathFromAnnlay(annlay_path);

	if(!FileExists(annmdl_path)) {
		Cout() << "prune-offset-heads: .annmdl file not found: " << annmdl_path << "\n";
		return false;
	}

	AnnMdl mdl;
	if(!mdl.LoadPath(annmdl_path)) {
		Cout() << "prune-offset-heads: failed to load .annmdl: " << annmdl_path << "\n";
		return false;
	}

	Vector<int> to_remove;
	for(int i = 0; i < mdl.entries.GetCount(); i++) {
		String id = mdl.entries[i].slot_id;
		bool remove = false;
		if(args.prune_keep == OFFSET_STYLE_COMBINED) {
			if(id.EndsWith("#offset_x") || id.EndsWith("#offset_y")) remove = true;
		}
		else if(args.prune_keep == OFFSET_STYLE_SPLIT) {
			if(id.EndsWith("#offset")) remove = true;
		}
		
		if(remove) to_remove.Add(i);
	}

	if(to_remove.IsEmpty()) {
		Cout() << "No entries matching prune criteria. Nothing to do.\n";
		return true;
	}

	Cout() << (args.prune_dry_run ? "[DRY-RUN] " : "") 
	       << "Pruning " << to_remove.GetCount() << " entries from " << GetFileName(annmdl_path) << "...\n";

	for(int idx : to_remove) {
		Cout() << "  - remove: " << mdl.entries[idx].slot_id << "\n";
	}

	if(args.prune_dry_run) {
		Cout() << "[DRY-RUN] Finished. No changes written.\n";
		return true;
	}

	for(int i = to_remove.GetCount() - 1; i >= 0; i--) {
		mdl.entries.Remove(to_remove[i]);
	}

	if(!mdl.Save(annlay_path)) {
		Cout() << "prune-offset-heads: failed to save updated .annmdl\n";
		return false;
	}

	Cout() << "Successfully saved pruned .annmdl. Total entries now: " << mdl.entries.GetCount() << "\n";
	return true;
}

bool RunFixBoolGates(const AppArguments& args) {
	String annlay_path = NormalizePath(args.fix_bool_annlay_path);
	if(!FileExists(annlay_path)) {
		Cout() << "fix-bool-gates: .annlay file not found: " << annlay_path << "\n";
		return false;
	}

	AnnLay lay;
	if(!lay.Load(annlay_path)) {
		Cout() << "fix-bool-gates: failed to load .annlay: " << annlay_path << "\n";
		return false;
	}

	Cout() << (args.fix_bool_dry_run ? "[DRY-RUN] " : "")
	       << "Fixing bool gates for: " << GetFileName(annlay_path) << "\n";
	Cout() << "--------------------------------------------------\n";

	int added = 0;
	int normalized = 0;

	auto NormalizeGate = [&](AnnLaySlot& gate, const String& source_id) {
		bool changed = false;
		if(gate.method != ANNLAY_CLASSIFIER_BOOL) {
			changed = true;
			if(!args.fix_bool_dry_run)
				gate.method = ANNLAY_CLASSIFIER_BOOL;
		}
		if(gate.color_mode != "grayscale") {
			changed = true;
			if(!args.fix_bool_dry_run)
				gate.color_mode = "grayscale";
		}
		if(gate.classes.GetCount() != 2 || gate.classes[0] != "false" || gate.classes[1] != "true") {
			changed = true;
			if(!args.fix_bool_dry_run) {
				gate.classes.Clear();
				gate.classes.Add("false");
				gate.classes.Add("true");
			}
		}
		if(changed) {
			normalized++;
			Cout() << "  - " << (args.fix_bool_dry_run ? "Would normalize gate: " : "Normalized gate: ")
			       << gate.id << " (source " << source_id << ")\n";
		}
	};

	auto AddGate = [&](const String& source_id, const String& gate_id) {
		const AnnLaySlot* source = lay.FindSlot(source_id);
		if(!source) return;

		if(AnnLaySlot* existing = lay.FindSlot(gate_id)) {
			NormalizeGate(*existing, source_id);
			return;
		}

		Cout() << "  - " << (args.fix_bool_dry_run ? "Would add missing gate: " : "Adding missing gate: ")
		       << gate_id << " (cloned from " << source_id << ")\n";

		if(!args.fix_bool_dry_run) {
			AnnLaySlot& gate = lay.slots.Add();
			gate.id = gate_id;
			gate.method = ANNLAY_CLASSIFIER_BOOL;
			gate.color_mode = "grayscale";
			gate.crop_size = source->crop_size;
			gate.anchor = source->anchor;
			gate.anchor_candidates = clone(source->anchor_candidates);
			gate.template_w = source->template_w;
			gate.template_h = source->template_h;
			gate.template_crop_x = source->template_crop_x;
			gate.template_crop_y = source->template_crop_y;
			gate.template_crop_w = source->template_crop_w;
			gate.template_crop_h = source->template_crop_h;
			gate.classes.Add("false");
			gate.classes.Add("true");
		}
		added++;
	};

	for(int i = 0; i < lay.slots.GetCount(); i++) {
		String sid = lay.slots[i].id;
		if(sid.EndsWith("_bet") && (sid.StartsWith("seat") || sid.StartsWith("seat_"))) {
			String gate_id = sid.Left(sid.GetCount() - 4) + "_is_bet_chip";
			AddGate(sid, gate_id);
		}
	}

	AddGate("previous_streets_pot", "is_previous_streets_pot");
	AddGate("side_pot", "is_side_pot");

	AddGate("action_call", "is_action_call");
	AddGate("action_fold", "is_action_fold");
	AddGate("action_raise", "is_action_raise");

	for(int n = 1; n <= 5; n++)
		AddGate(Format("board_card_%d", n), Format("is_board_card_%d", n));
	for(int n = 1; n <= 2; n++)
		AddGate(Format("hero_card_%d", n), Format("is_hero_card_%d", n));

	Cout() << "--------------------------------------------------\n";
	if(added == 0 && normalized == 0) {
		Cout() << "No missing gates found. Everything looks good.\n";
	}
	else {
		if(added > 0)
			Cout() << (args.fix_bool_dry_run ? "Would add " : "Added ") << added << " missing gate slots.\n";
		if(normalized > 0)
			Cout() << (args.fix_bool_dry_run ? "Would normalize " : "Normalized ") << normalized << " existing gate slots.\n";

		if(!args.fix_bool_dry_run) {
			if(lay.Save(annlay_path)) {
				Cout() << "Successfully saved updated .annlay: " << annlay_path << "\n";
			}
			else {
				Cout() << "ERROR: Failed to save updated .annlay: " << annlay_path << "\n";
				return false;
			}
		}
	}

	return true;
}

bool RunMakeEvalSplit(const AppArguments& args) {
	String sln_path = NormalizePath(args.make_split_sln_path);
	AnnSln sln;
	if(!sln.Load(sln_path)) {
		Cout() << "make-eval-split: failed to load .annsln: " << sln_path << "\n";
		return false;
	}

	String sln_dir = GetFileDirectory(sln_path);
	String annprj_path = ResolveFromBase(sln_dir, sln.annprj);

	Value root = ParseJSON(LoadFile(annprj_path));
	if(IsNull(root) || !IsValueMap(root)) {
		Cout() << "make-eval-split: invalid annprj\n";
		return false;
	}
	Value datasets = root["datasets"];
	if(!IsValueArray(datasets)) return false;

	Vector<String> all_ids;
	for(int di = 0; di < datasets.GetCount(); di++) {
		Value images = datasets[di]["images"];
		if(!IsValueArray(images)) continue;
		for(int ii = 0; ii < images.GetCount(); ii++) {
			Value img_rec = images[ii];
			String id = img_rec["file_name"];
			if(id.IsEmpty()) id = img_rec["file_path"];
			if(id.IsEmpty()) id = img_rec["filename"];
			if(!id.IsEmpty())
				all_ids.Add(id);
		}
	}

	if(all_ids.IsEmpty()) {
		Cout() << "make-eval-split: no images found in annprj\n";
		return false;
	}

	SeedRandom(args.make_split_seed);
	for(int i = 0; i < all_ids.GetCount(); i++) {
		int j = Random(all_ids.GetCount());
		Swap(all_ids[i], all_ids[j]);
	}

	int holdout_count = (int)round(all_ids.GetCount() * args.make_split_ratio);
	if(holdout_count == 0 && args.make_split_ratio > 0) holdout_count = 1;
	if(holdout_count >= all_ids.GetCount()) holdout_count = all_ids.GetCount() - 1;

	SplitData sd;
	for(int i = 0; i < all_ids.GetCount(); i++) {
		if(i < holdout_count)
			sd.holdout.Add(all_ids[i]);
		else
			sd.train.Add(all_ids[i]);
	}
	sd.seed = args.make_split_seed;
	sd.ratio = args.make_split_ratio;

	if(StoreAsJsonFile(sd, args.make_split_out_path, true)) {
		Cout() << "Successfully saved split to: " << args.make_split_out_path << "\n";
		Cout() << "  Total images: " << all_ids.GetCount() << "\n";
		Cout() << "  Train:        " << sd.train.GetCount() << "\n";
		Cout() << "  Holdout:      " << sd.holdout.GetCount() << "\n";
		return true;
	}
	else {
		Cout() << "make-eval-split: failed to save output file: " << args.make_split_out_path << "\n";
		return false;
	}
}

bool RunRecomputeAnchors(const AppArguments& args) {
	String annlay_path = NormalizePath(args.recompute_annlay_path);
	String annprj_path = NormalizePath(args.recompute_annprj_path);

	if(!FileExists(annlay_path)) {
		Cout() << "recompute-anchors: .annlay file not found: " << annlay_path << "\n";
		return false;
	}
	if(!FileExists(annprj_path)) {
		Cout() << "recompute-anchors: .annprj file not found: " << annprj_path << "\n";
		return false;
	}

	AnnLay lay;
	if(!lay.Load(annlay_path)) {
		Cout() << "recompute-anchors: failed to load .annlay: " << annlay_path << "\n";
		return false;
	}

	Cout() << "Recomputing anchors for: " << GetFileName(annlay_path) << " from " << GetFileName(annprj_path) << "...\n";
	lay.ComputeAnchorsFromAnnprj(annprj_path);

	if(!lay.Save(annlay_path)) {
		Cout() << "recompute-anchors: failed to save updated .annlay: " << annlay_path << "\n";
		return false;
	}

	Cout() << "Successfully recomputed and saved anchors.\n";
	return true;
}

bool RunAuditBoolGates(const AppArguments& args) {
	String annlay_path = NormalizePath(args.audit_annlay_path);
	if(!FileExists(annlay_path)) {
		Cout() << "audit-bool-gates: .annlay file not found: " << annlay_path << "\n";
		return false;
	}

	AnnLay lay;
	if(!lay.Load(annlay_path)) {
		Cout() << "audit-bool-gates: failed to load .annlay: " << annlay_path << "\n";
		return false;
	}

	Cout() << "Bool Gate Coverage Audit for: " << GetFileName(annlay_path) << "\n";
	Cout() << "--------------------------------------------------\n";

	Vector<AuditFinding> findings;
	
	auto AddFinding = [&](String sev, String msg, String sid, String rsid = "") {
		AuditFinding& f = findings.Add();
		f.severity = sev;
		f.message = msg;
		f.slot_id = sid;
		f.related_slot_id = rsid;
		Cout() << Format("[%s] %s: %s%s\n", sev, sid, msg, rsid.IsEmpty() ? "" : " (required gate: " + rsid + ")");
	};

	Vector<String> gate_slots;
	for(const auto& slot : lay.slots) {
		if(slot.method == ANNLAY_CLASSIFIER_BOOL) {
			String gkey = AnchoredSlotClassifier::BoolSlotGroupKey(slot.id, &lay);
			if(gkey == "chip_box_gate") {
				gate_slots.Add(slot.id);
			}
		}
	}

	Cout() << "Detected chip_box_gate slots: " << gate_slots.GetCount() << "\n";
	for(const String& sid : gate_slots) {
		const AnnLaySlot* s = lay.FindSlot(sid);
		if(!s) continue;

		bool empty = (s->anchor.cx == 0 && s->anchor.cy == 0 && s->anchor.w == 0 && s->anchor.h == 0);
		if(empty && !s->anchor_candidates.IsEmpty()) empty = false;

		int n_ann = s->anchor.n_annotations;
		for(const auto& c : s->anchor_candidates) n_ann += c.n_annotations;

		if(empty) {
			AddFinding("WARNING", "Gate anchor is empty", sid);
		}
		else if(n_ann < 5) {
			AddFinding("WARNING", Format("Gate has low annotation support (n=%d)", n_ann), sid);
		}
	}

	auto CheckGate = [&](const String& target_id, const String& gate_id) {
		const AnnLaySlot* target = lay.FindSlot(target_id);
		if(target) {
			const AnnLaySlot* gate = lay.FindSlot(gate_id);
			if(!gate) {
				AddFinding("ERROR", "Target slot exists but required gate is missing", target_id, gate_id);
			}
			else if(gate->method != ANNLAY_CLASSIFIER_BOOL) {
				AddFinding("ERROR", "Gate exists but is not classifier_bool", target_id, gate_id);
			}
		}
	};

	for(const auto& slot : lay.slots) {
		if(slot.id.EndsWith("_bet") && (slot.id.StartsWith("seat") || slot.id.StartsWith("seat_"))) {
			String gate_id = slot.id.Left(slot.id.GetCount() - 4) + "_is_bet_chip";
			CheckGate(slot.id, gate_id);
		}
	}

	CheckGate("previous_streets_pot", "is_previous_streets_pot");
	CheckGate("side_pot", "is_side_pot");
	
	CheckGate("action_call", "is_action_call");
	CheckGate("action_fold", "is_action_fold");
	CheckGate("action_raise", "is_action_raise");

	Cout() << "--------------------------------------------------\n";
	
	int errors = 0, warnings = 0;
	for(const auto& f : findings) {
		if(f.severity == "ERROR") errors++;
		else warnings++;
	}
	
	Cout() << "Audit complete. Findings: " << findings.GetCount() << " (" << errors << " ERRORS, " << warnings << " WARNINGS)\n";

	if(!args.audit_out_path.IsEmpty()) {
		BoolAuditReport report;
		report.annlay_path = annlay_path;
		report.timestamp = MakeIsoTimestamp();
		report.error_count = errors;
		report.warning_count = warnings;
		report.findings = pick(findings);
		
		if(StoreAsJsonFile(report, args.audit_out_path, true))
			Cout() << "Audit report saved to: " << args.audit_out_path << "\n";
	}

	if(args.audit_strict && errors > 0) {
		Cout() << "STRICT MODE: Failing due to " << errors << " ERRORS.\n";
		return false;
	}

	return true;
}

bool RunSmokeTest(const AppArguments& args) {
	String sln_path = NormalizePath(args.smoke_sln_path);
	if(sln_path.IsEmpty()) {
		Cout() << "smoke-test: missing .annsln path\n";
		return false;
	}

	AnnSln sln;
	if(!sln.Load(sln_path)) {
		Cout() << "smoke-test: failed to load .annsln: " << sln_path << "\n";
		return false;
	}

	String sln_dir = GetFileDirectory(sln_path);
	String annlay_path = ResolveFromBase(sln_dir, sln.annlay);
	String annprj_path = ResolveFromBase(sln_dir, sln.annprj);
	String script_path = ResolveFromBase(sln_dir, sln.recognition_script);
	String images_dir = ResolveFromBase(sln_dir, sln.images_dir);
	if(images_dir.IsEmpty())
		images_dir = AppendFileName(GetFileDirectory(annprj_path), "images");

	if(annlay_path.IsEmpty() || !FileExists(annlay_path)) {
		Cout() << "smoke-test: annlay file does not exist: " << annlay_path << "\n";
		return false;
	}
	if(annprj_path.IsEmpty() || !FileExists(annprj_path)) {
		Cout() << "smoke-test: annprj file does not exist: " << annprj_path << "\n";
		return false;
	}
	if(script_path.IsEmpty() || !FileExists(script_path)) {
		Cout() << "smoke-test: recognition script does not exist: " << script_path << "\n";
		return false;
	}

	AnchoredSlotRecognizer layout_only;
	if(!layout_only.Load(annlay_path)) {
		Cout() << "smoke-test: failed to load annlay layout: " << annlay_path << "\n";
		return false;
	}
	Cout() << "smoke-test: layout loaded, slots=" << layout_only.GetLayout().slots.GetCount() << "\n";

	RecognitionScript script;
	if(!script.Load(script_path)) {
		Cout() << "smoke-test: failed to load recognition script: " << script_path << "\n";
		Cout() << "smoke-test: script error: " << script.GetLastError() << "\n";
		return false;
	}

	String annmdl_path;
	int pass1 = sln.model_sets.Find("pass1");
	if(pass1 >= 0 && !TrimBoth(sln.model_sets[pass1]).IsEmpty())
		annmdl_path = ResolveFromBase(sln_dir, sln.model_sets[pass1]);
	if(annmdl_path.IsEmpty())
		annmdl_path = AnnMdl::PathFromAnnlay(annlay_path);

	if(!FileExists(annmdl_path)) {
		Cout() << "No .annmdl found - skipping recognition smoke test: " << annmdl_path << "\n";
		Cout() << "Defined slots:\n";
		const AnnLay& lay = layout_only.GetLayout();
		for(int i = 0; i < lay.slots.GetCount(); i++)
			Cout() << "  - " << lay.slots[i].id << "\n";
		return true;
	}

	AnchoredSlotRecognizer rec;
	rec.SetOffsetMode(args.offset_mode_arg);
	rec.SetBoolGatePolicy(args.bool_gate_policy_arg);
	if(!rec.Load(annlay_path, annmdl_path)) {
		Cout() << "smoke-test: failed to load recognizer with model: " << annmdl_path << "\n";
		return false;
	}

	Value root = ParseJSON(LoadFile(annprj_path));
	if(IsNull(root) || !IsValueMap(root)) {
		Cout() << "smoke-test: invalid annprj JSON: " << annprj_path << "\n";
		return false;
	}
	Value datasets = root["datasets"];
	if(!IsValueArray(datasets)) {
		Cout() << "smoke-test: annprj has no datasets array\n";
		return false;
	}

	int total_images = 0;
	for(int di = 0; di < datasets.GetCount(); di++) {
		Value images = datasets[di]["images"];
		if(IsValueArray(images))
			total_images += images.GetCount();
	}
	int kLimit = args.smoke_all ? total_images : max(1, args.smoke_count);
	if(kLimit > total_images)
		kLimit = total_images;
	int processed = 0;
	bool ok = true;

	int total_cleared = 0;
	for(int di = 0; di < datasets.GetCount() && processed < kLimit; di++) {
		Value images = datasets[di]["images"];
		if(!IsValueArray(images))
			continue;
		for(int ii = 0; ii < images.GetCount() && processed < kLimit; ii++) {
			Value img_rec = images[ii];
			String display_name = img_rec["file_name"];
			if(display_name.IsEmpty())
				display_name = img_rec["file_path"];
			if(display_name.IsEmpty())
				display_name = img_rec["filename"];
			if(display_name.IsEmpty())
				display_name = Format("dataset[%d].images[%d]", di, ii);

			String img_path = ResolveImagePathForSmoke(sln_dir, annprj_path, images_dir, img_rec);
			if(img_path.IsEmpty() || !FileExists(img_path)) {
				Cout() << Format("[img %d/%d] %s -> image not found (%s)\n",
				                 processed + 1, kLimit, display_name, img_path);
				processed++;
				ok = false;
				continue;
			}

			Image img = LoadImageByExtension(img_path);
			if(img.IsEmpty()) {
				Cout() << Format("[img %d/%d] %s -> failed to decode image (%s)\n",
				                 processed + 1, kLimit, display_name, img_path);
				processed++;
				ok = false;
				continue;
			}

			Vector<SlotResult> results = rec.Recognize(img);
			
			for(const auto& r : results) {
				String gate_id;
				if(r.slot_id.EndsWith("_bet") && (r.slot_id.StartsWith("seat") || r.slot_id.StartsWith("seat_")))
					gate_id = r.slot_id.Left(r.slot_id.GetCount() - 4) + "_is_bet_chip";
				else if(r.slot_id == "previous_streets_pot")
					gate_id = "is_previous_streets_pot";
				else if(r.slot_id == "side_pot")
					gate_id = "is_side_pot";
				
				if(!gate_id.IsEmpty()) {
					for(const auto& g : results) {
						if(g.slot_id == gate_id && g.method == "classifier_bool") {
							bool is_true = (g.top_class == "true" || g.class_index == 1);
							if(!is_true && r.top_class.IsEmpty() && r.raw_text.IsEmpty()) {
								total_cleared++;
							}
							break;
						}
					}
				}
			}

			VectorMap<String, String> meta = script.Run(results);
			String script_err = script.GetLastError();
			String line_meta;
			if(!script_err.IsEmpty()) {
				line_meta = "script_error=" + script_err;
				ok = false;
			}
			else if(!meta.IsEmpty())
				line_meta = JoinPairs(meta);
			else
				line_meta = BuildFallbackMetadata(results);

			Cout() << Format("[img %d/%d] %s -> %s\n",
			                 processed + 1, kLimit, GetFileName(display_name), line_meta);
			processed++;
		}
	}

	if(processed == 0) {
		Cout() << "smoke-test: no images found in annprj datasets\n";
		return false;
	}

	if(total_cleared > 0)
		Cout() << "smoke-test: total gated fields cleared by bool gates: " << total_cleared << "\n";

	Cout() << "smoke-test: read-only run complete, annprj was not modified\n";
	return ok;
}



bool RunRunPhase6Preflight(const AppArguments& args) {
	String sln = args.annsln_path;
	String out_dir = args.mine_out_dir;
	String split = args.split_file;
	OffsetMode mode = (OffsetMode)args.offset_mode_arg;
	bool fail_on_warnings = args.fail_on_warnings_arg;

	AnnSln cfg;
	if(!cfg.Load(sln)) {
		Cout() << "run-phase6-preflight: failed to load solution: " << sln << "\n";
		return false;
	}
	String sln_dir = GetFileDirectory(sln);
	String annlay = ResolveFromBase(sln_dir, cfg.annlay);

	RealizeDirectory(out_dir);

	struct StepInfo : Moveable<StepInfo> {
		String name;
		String artifact;
		int exit_code = -1;
		bool passed = false;
		int warning_count = 0;
		int low_support_count = 0;
		String reason;
		double load_sec = 0;
		double eval_sec = 0;
		double total_sec = 0;
	};
	Vector<StepInfo> steps;

	uint64 start_all = GetTickCount();

	// Step 1: Audit
	{
		Cout() << "\n[Preflight Step 1/3] Audit Bool Gates...\n";
		StepInfo& s = steps.Add();
		s.name = "Audit";
		s.artifact = AppendFileName(out_dir, "audit_report.json");
		
		AppArguments a;
		a.mode = AppArguments::MODE_AUDIT_BOOL_GATES;
		a.audit_annlay_path = annlay;
		a.audit_strict = true;
		a.audit_out_path = s.artifact;
		
		// Representative command for summary
		s.name = "Audit";
		
		uint64 s_start = GetTickCount();
		s.passed = RunAuditBoolGates(a);
		s.total_sec = (GetTickCount() - s_start) / 1000.0;
		s.exit_code = s.passed ? 0 : 1;

		if(FileExists(s.artifact)) {
			Value audit = ParseJSON(LoadFile(s.artifact));
			if(!IsNull(audit)) {
				s.warning_count = audit["warning_count"];
				if(fail_on_warnings && s.warning_count > 0 && s.passed) {
					s.passed = false;
					s.exit_code = 2;
					Cout() << "FAIL: Audit has warnings and fail-on-warnings is enabled.\n";
				}
			}
		}
	}

	// Step 2: Check Bool Baseline
	if(steps.Top().passed) {
		Cout() << "\n[Preflight Step 2/3] Check Bool Baseline...\n";
		StepInfo& s = steps.Add();
		s.name = "Check Bool";
		s.artifact = AppendFileName(out_dir, "check_bool_report.json");
		
		AppArguments a;
		a.mode = AppArguments::MODE_CHECK_BOOL_BASELINE;
		a.check_baseline_sln_path = sln;
		a.check_baseline_in_path = args.check_baseline_in_path;
		a.split_file = split;
		a.check_bool_out_path = s.artifact;
		a.bool_gate_policy_arg = args.bool_gate_policy_arg;
		a.offset_mode_arg = mode;
		a.eval_all = true;
		a.allow_baseline_mismatch = args.allow_baseline_mismatch;
		a.min_bool_support = args.min_bool_support;
		a.fail_on_low_support_arg = args.fail_on_low_support_arg;
		a.min_holdout_f1 = args.min_holdout_f1;
		a.max_f1_drop = args.max_f1_drop;
		a.max_recall_drop = args.max_recall_drop;

		uint64 s_start = GetTickCount();

		s.passed = RunCheckBoolBaseline(a);
		s.total_sec = (GetTickCount() - s_start) / 1000.0;
		s.exit_code = s.passed ? 0 : 1;

		if(FileExists(s.artifact)) {
			Value check = ParseJSON(LoadFile(s.artifact));
			if(!IsNull(check)) {
				s.low_support_count = check["low_support_count"];
			}
		}
	}

	// Step 3: Check Offset Baseline
	if(steps.Top().passed) {
		Cout() << "\n[Preflight Step 3/3] Check Offset Baseline...\n";
		StepInfo& s = steps.Add();
		s.name = "Check Offset";
		s.artifact = AppendFileName(out_dir, "check_offset_report.json");
		
		AppArguments a;
		a.mode = AppArguments::MODE_CHECK_OFFSET_BASELINE;
		a.check_baseline_sln_path = sln;
		a.check_offset_baseline_in_path = args.check_offset_baseline_in_path;
		a.split_file = split;
		a.check_offset_out_path = s.artifact;
		a.eval_all = true;
		a.subset = args.subset; // holdout/all
		a.allow_baseline_mismatch = args.allow_baseline_mismatch;
		a.max_meanl1_rise = args.max_meanl1_rise;
		a.max_p90_rise = args.max_p90_rise;
		
		uint64 s_start = GetTickCount();
		s.passed = RunCheckOffsetBaseline(a);
		s.total_sec = (GetTickCount() - s_start) / 1000.0;
		s.exit_code = s.passed ? 0 : 1;
	}

	bool all_passed = true;
	int total_warnings = 0;
	String failed_step;
	
	for(const auto& s : steps) {
		if(!s.passed) {
			all_passed = false;
			if(failed_step.IsEmpty()) failed_step = s.name;
		}
		total_warnings += s.warning_count;
	}

	double total_elapsed = (GetTickCount() - start_all) / 1000.0;

	// Write summary JSON
	ValueMap summary;
	summary.Add("passed", all_passed);
	summary.Add("failed_step", failed_step.IsEmpty() ? Value() : Value(failed_step));
	summary.Add("timestamp", MakeIsoTimestamp());
	summary.Add("fail_on_warnings", fail_on_warnings);
	summary.Add("total_warnings", total_warnings);
	summary.Add("total_elapsed_sec", total_elapsed);
	
	ValueMap support_policy;
	support_policy.Add("min_bool_support", args.min_bool_support);
	support_policy.Add("fail_on_low_support", args.fail_on_low_support_arg);
	summary.Add("support_policy", support_policy);
	
	ValueArray steps_ja;
	for(const auto& s : steps) {
		ValueMap sm;
		sm.Add("name", s.name);
		
		// Construct a representative command string
		String cmd = "bin/AnnotationEditor ";
		if(s.name == "Audit") {
			cmd << "--audit-bool-gates " << annlay << " --strict --out " << s.artifact;
		}
		else if(s.name == "Check Bool") {
			cmd << "--check-bool-baseline " << sln << " --baseline " << args.check_baseline_in_path;
			if(!split.IsEmpty()) cmd << " --split-file " << split;
			cmd << " --check-out " << s.artifact << " --bool-gate-policy strict --offset-mode " << OffsetModeToString(mode);
			if(args.min_bool_support > 0) {
				cmd << " --min-bool-support " << args.min_bool_support;
				cmd << " --fail-on-low-support " << (args.fail_on_low_support_arg ? "1" : "0");
			}
		}
		else if(s.name == "Check Offset") {
			cmd << "--check-offset-baseline " << sln << " --baseline " << args.check_offset_baseline_in_path;
			if(!split.IsEmpty()) cmd << " --split-file " << split;
			cmd << " --check-out " << s.artifact;
		}
		
		sm.Add("command", cmd);
		sm.Add("exit_code", s.exit_code);
		sm.Add("passed", s.passed);
		sm.Add("artifact", s.artifact);
		sm.Add("warning_count", s.warning_count);
		sm.Add("low_support_count", s.low_support_count);
		sm.Add("reason", s.reason);
		sm.Add("total_sec", s.total_sec);
		steps_ja.Add(sm);
	}
	summary.Add("steps", steps_ja);
	
	String summary_path = AppendFileName(out_dir, "phase6_preflight_summary.json");
	SaveFile(summary_path, StoreAsJson(summary, true));

	if(all_passed) {
		Cout() << "\nPREFLIGHT PASSED (" << total_elapsed << "s, Warnings: " << total_warnings << ")\n";
		Cout() << "Summary saved to: " << summary_path << "\n";
	}
	else {
		Cout() << "\nPREFLIGHT FAILED at step '" << failed_step << "'\n";
		Cout() << "Summary saved to: " << summary_path << "\n";
	}

	return all_passed;
}


bool RunRefreshPhase6Baselines(const AppArguments& args) {
	String sln = args.annsln_path;
	String out_dir = args.mine_out_dir;
	String split = args.split_file;
	
	RealizeDirectory(out_dir);
	
	struct Step : Moveable<Step> {
		String name;
		int exit_code = -1;
		double duration = 0;
		String artifact;
		bool passed = false;
	};
	Vector<Step> steps;
	
	uint64 start_all = GetTickCount();
	bool all_ok = true;
	
	// 1. Split
	String effective_split = split;
	if(effective_split.IsEmpty() || !FileExists(effective_split)) {
		effective_split = AppendFileName(out_dir, "eval_split.json");
		Cout() << "\n[Step 1/3] Generating new split: " << effective_split << "\n";
		
		Step& s = steps.Add();
		s.name = "Make Split";
		s.artifact = effective_split;
		
		AppArguments a;
		a.mode = AppArguments::MODE_MAKE_EVAL_SPLIT;
		a.make_split_sln_path = sln;
		a.make_split_out_path = effective_split;
		a.make_split_ratio = args.make_split_ratio;
		a.make_split_seed = args.make_split_seed;
		
		uint64 s_start = GetTickCount();
		s.passed = RunMakeEvalSplit(a);
		s.duration = (GetTickCount() - s_start) / 1000.0;
		s.exit_code = s.passed ? 0 : 1;
		if(!s.passed) all_ok = false;
	} else {
		Cout() << "\n[Step 1/3] Using existing split: " << effective_split << "\n";
		Step& s = steps.Add();
		s.name = "Use Split";
		s.artifact = effective_split;
		s.exit_code = 0;
		s.passed = true;
	}
	
	// 2. Bool Baseline
	if(all_ok) {
		String bool_base = AppendFileName(out_dir, "bool_baseline.json");
		Cout() << "\n[Step 2/3] Saving Bool Baseline: " << bool_base << "\n";
		Step& s = steps.Add();
		s.name = "Save Bool Baseline";
		s.artifact = bool_base;
		
		AppArguments a;
		a.mode = AppArguments::MODE_SAVE_BOOL_BASELINE;
		a.save_baseline_sln_path = sln;
		a.save_baseline_out_path = bool_base;
		a.split_file = effective_split;
		a.tune_group = args.tune_group;
		a.eval_all = true;
		a.bool_gate_policy_arg = args.bool_gate_policy_arg;
		a.offset_mode_arg = args.offset_mode_arg;
		
		uint64 s_start = GetTickCount();
		s.passed = RunSaveBoolBaseline(a);
		s.duration = (GetTickCount() - s_start) / 1000.0;
		s.exit_code = s.passed ? 0 : 1;
		if(!s.passed) all_ok = false;
	}
	
	// 3. Offset Baseline
	if(all_ok) {
		String offset_base = AppendFileName(out_dir, "offset_baseline.json");
		Cout() << "\n[Step 3/3] Saving Offset Baseline: " << offset_base << "\n";
		Step& s = steps.Add();
		s.name = "Save Offset Baseline";
		s.artifact = offset_base;
		
		AppArguments a;
		a.mode = AppArguments::MODE_SAVE_OFFSET_BASELINE;
		a.save_baseline_sln_path = sln;
		a.save_offset_baseline_out_path = offset_base;
		a.split_file = effective_split;
		a.subset = args.subset; // holdout|all
		a.eval_all = true;
		
		uint64 s_start = GetTickCount();
		s.passed = RunSaveOffsetBaseline(a);
		s.duration = (GetTickCount() - s_start) / 1000.0;
		s.exit_code = s.passed ? 0 : 1;
		if(!s.passed) all_ok = false;
	}
	
	double total_sec = (GetTickCount() - start_all) / 1000.0;
	
	ValueMap summary;
	summary.Add("passed", all_ok);
	summary.Add("timestamp", MakeIsoTimestamp());
	summary.Add("out_dir", out_dir);
	summary.Add("split_file", effective_split);
	summary.Add("bool_baseline", AppendFileName(out_dir, "bool_baseline.json"));
	summary.Add("offset_baseline", AppendFileName(out_dir, "offset_baseline.json"));
	summary.Add("total_duration_sec", total_sec);
	
	ValueArray steps_ja;
	for(const auto& s : steps) {
		ValueMap sm;
		sm.Add("name", s.name);
		sm.Add("exit_code", s.exit_code);
		sm.Add("passed", s.passed);
		sm.Add("duration_sec", s.duration);
		sm.Add("artifact", s.artifact);
		steps_ja.Add(sm);
	}
	summary.Add("steps", steps_ja);
	
	String summary_path = AppendFileName(out_dir, "phase6_baseline_refresh_summary.json");
	SaveFile(summary_path, StoreAsJson(summary, true));
	
	if(all_ok) {
		Cout() << "\nBASELINE REFRESH COMPLETE (" << total_sec << "s)\n";
		Cout() << "Summary saved to: " << summary_path << "\n";
	} else {
		Cout() << "\nBASELINE REFRESH FAILED (" << total_sec << "s)\n";
		Cout() << "Summary saved to: " << summary_path << "\n";
	}
	
	return all_ok;
}


bool RunRunPhase6ExitGate(const AppArguments& args) {
	String sln = args.annsln_path;
	String out_dir = args.mine_out_dir;
	String split = args.split_file;
	OffsetMode mode = (OffsetMode)args.offset_mode_arg;
	
	RealizeDirectory(out_dir);
	
	struct StepInfo : Moveable<StepInfo> {
		String name;
		int exit_code = -1;
		double duration_sec = 0;
		String artifact;
		bool passed = false;
		String command;
		String reason;
	};
	Vector<StepInfo> steps;
	
	uint64 start_all = GetTickCount();
	bool all_ok = true;
	
	// 1. Refresh Baselines (Optional)
	String effective_split = split;
	String bool_base = AppendFileName(out_dir, "bool_baseline.json");
	String offset_base = AppendFileName(out_dir, "offset_baseline.json");

	if(args.exit_gate_refresh_baselines) {
		Cout() << "\n[Exit Gate Step 1/3] Refreshing Baselines...\n";
		StepInfo& s = steps.Add();
		s.name = "Refresh Baselines";
		s.artifact = AppendFileName(out_dir, "phase6_baseline_refresh_summary.json");
		
		String cmd = "bin/AnnotationEditor --refresh-phase6-baselines " + sln + " --out-dir " + out_dir;
		if(!split.IsEmpty()) cmd << " --split-file " << split;
		s.command = cmd;

		uint64 s_start = GetTickCount();
		s.passed = RunRefreshPhase6Baselines(args);
		s.duration_sec = (GetTickCount() - s_start) / 1000.0;
		s.exit_code = s.passed ? 0 : 1;
		
		if(s.passed) {
			// If refresh worked and split was empty, it generated eval_split.json
			if(effective_split.IsEmpty()) {
				String generated = AppendFileName(out_dir, "eval_split.json");
				if(FileExists(generated)) effective_split = generated;
			}
		} else {
			s.reason = "Baseline refresh failed";
			all_ok = false;
		}
	} else {
		Cout() << "\n[Exit Gate Step 1/3] Skipping Baseline Refresh.\n";
		if(!FileExists(bool_base) || !FileExists(offset_base)) {
			StepInfo& s = steps.Add();
			s.name = "Artifact Check";
			s.passed = false;
			s.exit_code = 2;
			s.reason = "Baselines missing in out-dir and --refresh-baselines 0 is set";
			Cout() << "FAIL: " << s.reason << "\n";
			Cout() << "  Expected bool baseline: " << bool_base << "\n";
			Cout() << "  Expected offset baseline: " << offset_base << "\n";
			all_ok = false;
		}
	}
	
	// 2. Preflight
	if(all_ok) {
		Cout() << "\n[Exit Gate Step 2/3] Running Preflight Checks...\n";
		StepInfo& s = steps.Add();
		s.name = "Preflight";
		s.artifact = AppendFileName(out_dir, "phase6_preflight_summary.json");
		
		AppArguments a = args;
		a.mode = AppArguments::MODE_RUN_PHASE6_PREFLIGHT;
		a.check_baseline_in_path = bool_base;
		a.check_offset_baseline_in_path = offset_base;
		a.split_file = effective_split;
		
		String cmd = "bin/AnnotationEditor --run-phase6-preflight " + sln;
		cmd << " --bool-baseline " << bool_base << " --offset-baseline " << offset_base;
		if(!effective_split.IsEmpty()) cmd << " --split-file " << effective_split;
		cmd << " --out-dir " << out_dir;
		s.command = cmd;

		uint64 s_start = GetTickCount();
		s.passed = RunRunPhase6Preflight(a);
		s.duration_sec = (GetTickCount() - s_start) / 1000.0;
		s.exit_code = s.passed ? 0 : 1;
		if(!s.passed) {
			s.reason = "Preflight checks failed";
			all_ok = false;
		}
	}
	
	// 3. Strict Smoke Test
	if(all_ok) {
		Cout() << "\n[Exit Gate Step 3/3] Running Strict Smoke Test...\n";
		StepInfo& s = steps.Add();
		s.name = "Smoke Test";
		
		AppArguments a;
		a.mode = AppArguments::MODE_SMOKE_TEST;
		a.smoke_sln_path = sln;
		a.smoke_all = true;
		a.offset_mode_arg = args.offset_mode_arg;
		a.bool_gate_policy_arg = BOOL_GATE_STRICT;
		
		String cmd = "bin/AnnotationEditor --smoke-test " + sln + " --all --bool-gate-policy strict";
		cmd << " --offset-mode " << OffsetModeToString(mode);
		s.command = cmd;

		uint64 s_start = GetTickCount();
		s.passed = RunSmokeTest(a);
		s.duration_sec = (GetTickCount() - s_start) / 1000.0;
		s.exit_code = s.passed ? 0 : 1;
		if(!s.passed) {
			s.reason = "Strict smoke test failed";
			all_ok = false;
		}
	}
	
	double total_sec = (GetTickCount() - start_all) / 1000.0;
	
	ValueMap summary;
	summary.Add("passed", all_ok);
	if(!all_ok) {
		summary.Add("failed_step", steps.GetCount() > 0 ? steps.Top().name : "Initialization");
		summary.Add("failure_reason", steps.GetCount() > 0 ? steps.Top().reason : "Unknown error");
	}
	summary.Add("timestamp", MakeIsoTimestamp());
	summary.Add("out_dir", out_dir);
	summary.Add("resolved_split", effective_split);
	summary.Add("resolved_bool_baseline", bool_base);
	summary.Add("resolved_offset_baseline", offset_base);
	summary.Add("total_duration_sec", total_sec);

	
	ValueMap policy;
	policy.Add("refresh_baselines", args.exit_gate_refresh_baselines);
	policy.Add("offset_mode", OffsetModeToString(mode));
	policy.Add("bool_gate_policy", BoolGatePolicyToString((BoolGatePolicy)args.bool_gate_policy_arg));
	policy.Add("min_bool_support", args.min_bool_support);
	policy.Add("fail_on_low_support", args.fail_on_low_support_arg);
	policy.Add("fail_on_warnings", args.fail_on_warnings_arg);
	policy.Add("allow_baseline_mismatch", args.allow_baseline_mismatch);
	summary.Add("policy", policy);
	
	ValueArray steps_ja;
	for(const auto& s : steps) {
		ValueMap sm;
		sm.Add("name", s.name);
		sm.Add("command", s.command);
		sm.Add("exit_code", s.exit_code);
		sm.Add("passed", s.passed);
		sm.Add("duration_sec", s.duration_sec);
		sm.Add("artifact", s.artifact);
		sm.Add("reason", s.reason);
		steps_ja.Add(sm);
	}
	summary.Add("steps", steps_ja);
	
	String summary_path = AppendFileName(out_dir, "phase6_exit_gate_summary.json");
	SaveFile(summary_path, StoreAsJson(summary, true));
	
	if(all_ok) {
		Cout() << "\nPHASE 6 EXIT GATE: PASSED (" << total_sec << "s)\n";
		Cout() << "Summary saved to: " << summary_path << "\n";
	} else {
		Cout() << "\nPHASE 6 EXIT GATE: FAILED at step '" << steps.Top().name << "'\n";
		Cout() << "Summary saved to: " << summary_path << "\n";
	}
	
	return all_ok;
}


bool RunTraceFrameLog(const AppArguments& args) {
	String sln_path = args.annsln_path;
	String img_path = args.trace_image_path;
	String out_path = args.trace_out_path;
	String model_set = args.subset; // used as model-set name

	AnnSln sln;
	if(!sln.Load(sln_path)) {
		Cout() << "Error: Failed to load solution: " << sln_path << "\n";
		return false;
	}

	String sln_dir = GetFileDirectory(sln_path);
	String annlay_path = NormalizePath(AppendFileName(sln_dir, sln.annlay));
	String annmdl_path;
	int ms = sln.model_sets.Find(model_set);
	if(ms >= 0)
		annmdl_path = NormalizePath(AppendFileName(sln_dir, sln.model_sets[ms]));

	AnchoredSlotRecognizer recognizer;
	recognizer.SetSln(sln);
	if(!recognizer.Load(annlay_path, annmdl_path)) {
		Cout() << "Error: Failed to load recognizer: " << annlay_path << " / " << annmdl_path << "\n";
		return false;
	}

	recognizer.SetOffsetMode(args.offset_mode_arg);
	recognizer.SetBoolGatePolicy(args.bool_gate_policy_arg);

	RecognitionScript script;
	if(!sln.recognition_script.IsEmpty()) {
		String script_path = NormalizePath(AppendFileName(sln_dir, sln.recognition_script));
		if(FileExists(script_path))
			script.Load(script_path);
	}

	uint64 t_start = GetTickCount();

	uint64 t_load_start = GetTickCount();
	Image img = StreamRaster::LoadFileAny(img_path);
	double t_load_ms = (double)(GetTickCount() - t_load_start);

	if(img.IsEmpty()) {
		Cout() << "Error: Failed to load image: " << img_path << "\n";
		return false;
	}

	ProcessingLogRecord log;
	log.frame_seq = 1;
	log.image_name = GetFileName(img_path);
	log.image_path = img_path;
	log.t_load_ms = t_load_ms;

	uint64 t_rec_start = GetTickCount();
	log.results = recognizer.Recognize(img, &log);
	log.t_recognize_ms = (double)(GetTickCount() - t_rec_start);

	if(script.IsLoaded()) {
		uint64 t_script_start = GetTickCount();
		log.meta = script.Run(log.results);
		log.t_script_ms = (double)(GetTickCount() - t_script_start);
		log.script_output = script.GetLastOutput();
		log.script_error = script.GetLastError();
		log.AddStep("SCRIPT", log.t_script_ms,
		            log.script_error.IsEmpty() ? "OK" : "ERROR",
		            log.script_error.IsEmpty()
		                ? Format("%d meta keys", log.meta.GetCount())
		                : log.script_error,
		            "GameState");
	}

	log.t_total_ms = (double)(GetTickCount() - t_start);

	// Quality/Status logic
	for(int i = 0; i < log.results.GetCount(); i++) {
		if(log.results[i].confidence > 0.5) log.detections_good++;
		else log.detections_missing++;
	}

	if(!log.script_error.IsEmpty()) {
		log.status = "ERROR";
		log.warnings = "Script error: " + log.script_error;
	} else if(log.results.GetCount() == 0) {
		log.status = "WARN";
		log.warnings = "No slots recognized";
	} else if(log.detections_missing > 0) {
		log.status = "WARN";
		log.warnings = Format("%d detections with low confidence", log.detections_missing);
	} else {
		log.status = "OK";
	}

	String verbose = log.FormatVerbose();
	Cout() << verbose;

	if(!out_path.IsEmpty()) {
		RealizeDirectory(GetFileDirectory(out_path));
		if(!SaveFile(out_path, verbose)) {
			Cout() << "Error: Failed to save log to: " << out_path << "\n";
			return false;
		}
		Cout() << "\nVerbose log saved to: " << out_path << "\n";
	}

	return true;
}

namespace {

struct ConsistencyStats : Moveable<ConsistencyStats> {
	String slot_id;
	String gate_id;
	int samples = 0;
	int agreement = 0;
	int gate_true_presence_false = 0;
	int gate_false_presence_true = 0;
	int both_true = 0;
	int both_false = 0;

	// GT-grounded metrics
	int gt_true = 0;
	int gt_false = 0;
	int gate_correct = 0;
	int presence_correct = 0;
	int both_pred_false_gt_true = 0;

	// Confusion metrics for final joint decision
	int tp = 0;
	int fp = 0;
	int tn = 0;
	int fn = 0;

	void Jsonize(JsonIO& jio) {
		jio("slot_id", slot_id)("gate_id", gate_id)
		   ("samples", samples)("agreement", agreement)
		   ("gate_true_presence_false", gate_true_presence_false)
		   ("gate_false_presence_true", gate_false_presence_true)
		   ("both_true", both_true)("both_false", both_false)
		   ("gt_true", gt_true)("gt_false", gt_false)
		   ("gate_correct", gate_correct)("presence_correct", presence_correct)
		   ("both_pred_false_gt_true", both_pred_false_gt_true)
		   ("tp", tp)("fp", fp)("tn", tn)("fn", fn);
	}
};

struct MismatchCase : Moveable<MismatchCase> {
	String image_path;
	String slot_id;
	String gate_slot_id;
	bool gate_pred = false;
	bool presence_pred = false;
	bool gt_presence = false;
	String mismatch_type;
	double gate_conf = 0;
	double presence_conf = 0;

	void Jsonize(JsonIO& jio) {
		jio("image_path", image_path)("slot_id", slot_id)("gate_slot_id", gate_slot_id)
		   ("gate_pred", gate_pred)("presence_pred", presence_pred)("gt_presence", gt_presence)
		   ("mismatch_type", mismatch_type)("gate_conf", gate_conf)
		   ("presence_conf", presence_conf);
	}
};

bool ComputeCardVisibilityConsistency(const AppArguments& args, ValueMap& out_report) {
	AnnSln sln;
	if(!sln.Load(args.consistency_sln_path)) return false;

	String sln_dir = GetFileDirectory(args.consistency_sln_path);
	String annlay_path = ResolveFromBase(sln_dir, sln.annlay);
	String annprj_path = ResolveFromBase(sln_dir, sln.annprj);
	String images_dir = ResolveFromBase(sln_dir, sln.images_dir);
	if(images_dir.IsEmpty())
		images_dir = AppendFileName(GetFileDirectory(annprj_path), "images");

	String annmdl_path;
	int pass1 = sln.model_sets.Find("pass1");
	if(pass1 >= 0 && !TrimBoth(sln.model_sets[pass1]).IsEmpty())
		annmdl_path = ResolveFromBase(sln_dir, sln.model_sets[pass1]);
	if(annmdl_path.IsEmpty())
		annmdl_path = AnnMdl::PathFromAnnlay(annlay_path);

	if(!FileExists(annmdl_path)) return false;

	AnnLay lay;
	if(!lay.Load(annlay_path)) return false;

	AnchoredSlotRecognizer rec;
	rec.SetOffsetMode(args.offset_mode_arg);
	rec.SetBoolGatePolicy(args.bool_gate_policy_arg);
	
	// Ensure shared models are loaded by adding group keys to head_filter
	Index<String> h_filter, s_filter;
	for(const auto& s : lay.slots) {
		if(s.id.StartsWith("board_card_") || s.id.StartsWith("hero_card_")) {
			s_filter.Add(s.id);
			if(s.composite_type == ANNLAY_COMPOSITE_ELEMENT) {
				h_filter.Add(s.id + "#presence");
				h_filter.Add(s.id + "#level");
				h_filter.Add(s.id + "#category");
			}
			else {
				h_filter.Add(s.id);
			}
			String gkey = AnchoredSlotClassifier::BoolSlotGroupKey(s.id);
			if(!gkey.IsEmpty()) h_filter.Add(gkey);
			
			// Also add the explicit bool gate slots
			String gate_id = "is_" + s.id;
			s_filter.Add(gate_id);
			h_filter.Add(gate_id);
			String gate_gkey = AnchoredSlotClassifier::BoolSlotGroupKey(gate_id);
			if(!gate_gkey.IsEmpty()) h_filter.Add(gate_gkey);
		}
	}
	rec.SetHeadFilter(h_filter);
	rec.SetSlotFilter(s_filter);

	if(!rec.Load(annlay_path, annmdl_path)) return false;

	Value root = ParseJSON(LoadFile(annprj_path));
	if(IsNull(root) || !IsValueMap(root)) return false;
	Value datasets = root["datasets"];
	if(!IsValueArray(datasets)) return false;

	Index<String> subset_ids;
	if(!LoadSubsetIds(args.split_file, args.subset, subset_ids)) return false;

	VectorMap<String, ConsistencyStats> stats;
	Vector<MismatchCase> mismatches;
	int gtpf_count = 0;
	int gfpt_count = 0;

	auto GetStats = [&](const String& sid, const String& gid) -> ConsistencyStats& {
		int q = stats.Find(sid);
		if(q >= 0) return stats[q];
		ConsistencyStats& s = stats.Add(sid);
		s.slot_id = sid;
		s.gate_id = gid;
		return s;
	};

	int total_imgs = 0;
	for(int d = 0; d < datasets.GetCount(); d++)
		if(IsValueArray(datasets[d]["images"])) total_imgs += datasets[d]["images"].GetCount();
	
	int kLimit = args.eval_all ? total_imgs : args.eval_count;
	int processed_imgs = 0;

	Cout() << "Reporting element visibility consistency from " << kLimit << " images...\n";

	for(int di = 0; di < datasets.GetCount() && processed_imgs < kLimit; di++) {
		Value images = datasets[di]["images"];
		if(!IsValueArray(images)) continue;
		for(int ii = 0; ii < images.GetCount() && processed_imgs < kLimit; ii++) {
			Value img_rec = images[ii];
			String id = img_rec["file_name"];
			if(id.IsEmpty()) id = img_rec["file_path"];
			if(id.IsEmpty()) id = img_rec["filename"];
			
			if(!subset_ids.IsEmpty() && subset_ids.Find(id) < 0)
				continue;

			String img_path = ResolveImagePathForSmoke(sln_dir, annprj_path, images_dir, img_rec);
			if(img_path.IsEmpty() || !FileExists(img_path)) continue;
			Image img = LoadImageByExtension(img_path);
			if(img.IsEmpty()) continue;

			Vector<SlotResult> results = rec.Recognize(img);
			
			Value annotations = img_rec["annotations"];

			for(int ri = 0; ri < results.GetCount(); ri++) {
				const SlotResult& r = results[ri];
				if(r.gate_slot_id.IsEmpty()) continue;
				if(!r.slot_id.StartsWith("board_card_") && !r.slot_id.StartsWith("hero_card_")) continue;

				// Extract GT presence
				bool gt_presence = false;
				if(IsValueArray(annotations)) {
					for(int ai = 0; ai < annotations.GetCount(); ai++) {
						Value ann = annotations[ai];
						if(ann["metadata"]["mlui_slot_id"] == r.slot_id) {
							Value polys = ann["polygons"];
							if(IsValueArray(polys) && polys.GetCount() > 0) {
								gt_presence = true;
								break;
							}
						}
					}
				}

				// Element presence from composite head
				// For composite cards, RecognizeCompositeElement sets top_class to level+category if visible, or empty if not.
				bool presence = !r.top_class.IsEmpty() && r.top_class != "no_card";
				double p_conf = r.confidence;

				// Gate output
				bool gate_val = false;
				double g_conf = 0;
				bool found_gate = false;
				for(int gj = 0; gj < results.GetCount(); gj++) {
					if(results[gj].slot_id == r.gate_slot_id) {
						gate_val = (results[gj].top_class == "true" || results[gj].class_index == 1);
						g_conf = results[gj].confidence;
						found_gate = true;
						break;
					}
				}
				if(!found_gate) continue;

				ConsistencyStats& s = GetStats(r.slot_id, r.gate_slot_id);
				s.samples++;

				// GT grounded
				if(gt_presence) s.gt_true++; else s.gt_false++;
				if(gate_val == gt_presence) s.gate_correct++;
				if(presence == gt_presence) s.presence_correct++;
				if(!gate_val && !presence && gt_presence) s.both_pred_false_gt_true++;

				// Final joint decision: STRICT (gate AND presence)
				bool final_pred = gate_val && presence;
				if(final_pred && gt_presence) s.tp++;
				else if(final_pred && !gt_presence) s.fp++;
				else if(!final_pred && !gt_presence) s.tn++;
				else if(!final_pred && gt_presence) s.fn++;

				if(presence == gate_val) {
					s.agreement++;
					if(presence) s.both_true++;
					else s.both_false++;
				}

				bool is_mismatch = (presence != gate_val);
				bool shared_miss = (!gate_val && !presence && gt_presence);

				if(is_mismatch || shared_miss) {
					String mtype;
					bool limit_reached = false;
					if(is_mismatch) {
						if(gate_val && !presence) {
							s.gate_true_presence_false++;
							mtype = "gate_true_presence_false";
							if(args.consistency_mismatch_limit > 0 && gtpf_count >= args.consistency_mismatch_limit)
								limit_reached = true;
							else
								gtpf_count++;
						}
						else {
							s.gate_false_presence_true++;
							mtype = "gate_false_presence_true";
							if(args.consistency_mismatch_limit > 0 && gfpt_count >= args.consistency_mismatch_limit)
								limit_reached = true;
							else
								gfpt_count++;
						}
					}
					else if(shared_miss) {
						mtype = "both_false_gt_true";
						// No separate limit for shared misses for now
					}

					if(!limit_reached && !args.consistency_mismatch_out_path.IsEmpty()) {
						MismatchCase& mc = mismatches.Add();
						mc.image_path = img_path;
						mc.slot_id = r.slot_id;
						mc.gate_slot_id = r.gate_slot_id;
						mc.gate_pred = gate_val;
						mc.presence_pred = presence;
						mc.gt_presence = gt_presence;
						mc.mismatch_type = mtype;
						mc.gate_conf = g_conf;
						mc.presence_conf = p_conf;
					}
				}
			}
			processed_imgs++;
			if(processed_imgs % 20 == 0) Cout() << "Processed " << processed_imgs << " images...\n";
		}
	}

	ValueArray per_slot;
	int total_samples = 0, total_agreement = 0;
	int total_gtpf = 0, total_gfpt = 0;
	int total_gt_true = 0, total_gt_false = 0;
	int total_gate_correct = 0, total_presence_correct = 0;
	int total_both_pred_false_gt_true = 0;
	int total_tp = 0, total_fp = 0, total_tn = 0, total_fn = 0;

	for(int i = 0; i < stats.GetCount(); i++) {
		per_slot.Add(StoreAsJsonValue(stats[i]));
		total_samples += stats[i].samples;
		total_agreement += stats[i].agreement;
		total_gtpf += stats[i].gate_true_presence_false;
		total_gfpt += stats[i].gate_false_presence_true;
		total_gt_true += stats[i].gt_true;
		total_gt_false += stats[i].gt_false;
		total_gate_correct += stats[i].gate_correct;
		total_presence_correct += stats[i].presence_correct;
		total_both_pred_false_gt_true += stats[i].both_pred_false_gt_true;
		total_tp += stats[i].tp;
		total_fp += stats[i].fp;
		total_tn += stats[i].tn;
		total_fn += stats[i].fn;
	}

	double precision = (total_tp + total_fp > 0) ? (double)total_tp / (total_tp + total_fp) : 1.0;
	double recall = (total_tp + total_fn > 0) ? (double)total_tp / (total_tp + total_fn) : 0.0;
	double f1 = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0.0;
	double accuracy = (total_samples > 0) ? (double)(total_tp + total_tn) / total_samples : 0.0;

	out_report.Add("sln_path", args.consistency_sln_path);
	out_report.Add("timestamp", MakeIsoTimestamp());
	out_report.Add("processed_images", processed_imgs);
	out_report.Add("total_samples", total_samples);
	out_report.Add("total_agreement", total_agreement);
	out_report.Add("agreement_rate", total_samples > 0 ? (double)total_agreement / total_samples : 0.0);
	out_report.Add("total_gate_true_presence_false", total_gtpf);
	out_report.Add("total_gate_false_presence_true", total_gfpt);
	out_report.Add("total_gt_true", total_gt_true);
	out_report.Add("total_gt_false", total_gt_false);
	out_report.Add("total_gate_correct", total_gate_correct);
	out_report.Add("total_presence_correct", total_presence_correct);
	out_report.Add("total_both_pred_false_gt_true", total_both_pred_false_gt_true);
	out_report.Add("tp", total_tp);
	out_report.Add("fp", total_fp);
	out_report.Add("tn", total_tn);
	out_report.Add("fn", total_fn);
	out_report.Add("precision", precision);
	out_report.Add("recall", recall);
	out_report.Add("f1", f1);
	out_report.Add("accuracy", accuracy);
	out_report.Add("per_slot", per_slot);

	if(!args.consistency_mismatch_out_path.IsEmpty()) {
		ValueArray mm_ja;
		for(const auto& m : mismatches)
			mm_ja.Add(StoreAsJsonValue(m));
		
		if(!StoreAsJsonFile(mm_ja, args.consistency_mismatch_out_path, true)) {
			Cout() << "Error: failed to save mismatch report to: " << args.consistency_mismatch_out_path << "\n";
		} else {
			Cout() << "Mismatch report saved to: " << args.consistency_mismatch_out_path << " (" << mismatches.GetCount() << " cases)\n";
		}
	}

	return true;
}

struct ScoreItem : Moveable<ScoreItem> {
	double score;
	bool gt_presence;
};

bool ComputeCardVisibilityScores(const AppArguments& args, VectorMap<String, Vector<ScoreItem>>& out_scores) {
	AnnSln sln;
	if(!sln.Load(args.consistency_sln_path)) return false;

	String sln_dir = GetFileDirectory(args.consistency_sln_path);
	String annlay_path = ResolveFromBase(sln_dir, sln.annlay);
	String annprj_path = ResolveFromBase(sln_dir, sln.annprj);
	String images_dir = ResolveFromBase(sln_dir, sln.images_dir);
	if(images_dir.IsEmpty())
		images_dir = AppendFileName(GetFileDirectory(annprj_path), "images");

	String annmdl_path;
	int pass1 = sln.model_sets.Find("pass1");
	if(pass1 >= 0 && !TrimBoth(sln.model_sets[pass1]).IsEmpty())
		annmdl_path = ResolveFromBase(sln_dir, sln.model_sets[pass1]);
	if(annmdl_path.IsEmpty())
		annmdl_path = AnnMdl::PathFromAnnlay(annlay_path);

	if(!FileExists(annmdl_path)) return false;

	AnnLay lay;
	if(!lay.Load(annlay_path)) return false;

	AnchoredSlotRecognizer rec;
	rec.SetOffsetMode(args.offset_mode_arg);
	rec.SetBoolGatePolicy(BOOL_GATE_PERMISSIVE); // Don't block our own checks
	
	Index<String> h_filter, s_filter;
	for(const auto& s : lay.slots) {
		if(s.id.StartsWith("board_card_") || s.id.StartsWith("hero_card_")) {
			s_filter.Add(s.id);
			if(s.composite_type == ANNLAY_COMPOSITE_ELEMENT) {
				h_filter.Add(s.id + "#presence");
				h_filter.Add(s.id + "#level");
				h_filter.Add(s.id + "#category");
			}
			else h_filter.Add(s.id);
			String gkey = AnchoredSlotClassifier::BoolSlotGroupKey(s.id);
			if(!gkey.IsEmpty()) h_filter.Add(gkey);
			
			String gate_id = "is_" + s.id;
			s_filter.Add(gate_id);
			h_filter.Add(gate_id);
			String gate_gkey = AnchoredSlotClassifier::BoolSlotGroupKey(gate_id);
			if(!gate_gkey.IsEmpty()) h_filter.Add(gate_gkey);
		}
	}
	rec.SetHeadFilter(h_filter);
	rec.SetSlotFilter(s_filter);

	if(!rec.Load(annlay_path, annmdl_path)) return false;

	Value root = ParseJSON(LoadFile(annprj_path));
	if(IsNull(root) || !IsValueMap(root)) return false;
	Value datasets = root["datasets"];
	if(!IsValueArray(datasets)) return false;

	Index<String> subset_ids;
	if(!LoadSubsetIds(args.split_file, args.subset, subset_ids)) return false;

	int total_imgs = 0;
	for(int d = 0; d < datasets.GetCount(); d++)
		if(IsValueArray(datasets[d]["images"])) total_imgs += datasets[d]["images"].GetCount();
	
	int kLimit = args.eval_all ? total_imgs : args.eval_count;
	int processed_imgs = 0;

	Cout() << "Collecting visibility scores from " << kLimit << " images...\n";

	for(int di = 0; di < datasets.GetCount() && processed_imgs < kLimit; di++) {
		Value images = datasets[di]["images"];
		if(!IsValueArray(images)) continue;
		for(int ii = 0; ii < images.GetCount() && processed_imgs < kLimit; ii++) {
			Value img_rec = images[ii];
			String id = img_rec["file_name"];
			if(id.IsEmpty()) id = img_rec["file_path"];
			if(id.IsEmpty()) id = img_rec["filename"];
			
			if(!subset_ids.IsEmpty() && subset_ids.Find(id) < 0)
				continue;

			String img_path = ResolveImagePathForSmoke(sln_dir, annprj_path, images_dir, img_rec);
			if(img_path.IsEmpty() || !FileExists(img_path)) continue;
			Image img = LoadImageByExtension(img_path);
			if(img.IsEmpty()) continue;

			Vector<SlotResult> results = rec.Recognize(img);
			
			for(int ri = 0; ri < results.GetCount(); ri++) {
				const SlotResult& r = results[ri];
				if(r.gate_slot_id.IsEmpty()) continue;
				if(!r.slot_id.StartsWith("board_card_") && !r.slot_id.StartsWith("hero_card_")) continue;

				bool presence = !r.top_class.IsEmpty() && r.top_class != "no_card";
				
				double gate_conf = 0;
				bool found_gate = false;
				for(int gj = 0; gj < results.GetCount(); gj++) {
					if(results[gj].slot_id == r.gate_slot_id) {
						if(results[gj].top_class == "true")
							gate_conf = results[gj].confidence;
						else
							gate_conf = 1.0 - results[gj].confidence;
						found_gate = true;
						break;
					}
				}
				if(!found_gate) continue;

				ScoreItem& si = out_scores.GetAdd(r.gate_slot_id).Add();
				si.score = gate_conf;
				si.gt_presence = presence;
			}
			processed_imgs++;
			if(processed_imgs % 20 == 0) Cout() << "Processed " << processed_imgs << " images...\n";
		}
	}
	return true;
}

} // namespace

bool RunReportCardVisibilityConsistency(const AppArguments& args) {
	ValueMap report;
	if(!ComputeCardVisibilityConsistency(args, report)) {
		Cout() << "Error: failed to compute consistency report\n";
		return false;
	}

	if(StoreAsJsonFile(report, args.consistency_out_path, true)) {
		Cout() << "Consistency report saved to: " << args.consistency_out_path << "\n";
		Cout() << "Overall Agreement: " << Format("%.2f%%", (double)report["agreement_rate"] * 100.0) << "\n";
		return true;
	}
	
	Cout() << "Error: failed to save consistency report to: " << args.consistency_out_path << "\n";
	return false;
}

bool RunTuneCardVisibilityByConsistency(const AppArguments& args) {
	VectorMap<String, Vector<ScoreItem>> score_data;
	if(!ComputeCardVisibilityScores(args, score_data)) {
		Cout() << "Error: failed to collect scores\n";
		return false;
	}

	Cout() << "\nThreshold Sweep (Objective: Max Agreement, Constraint: FN=0)\n";
	Cout() << "----------------------------------------------------------------------\n";
	Cout() << Format("%-25s | %-6s | %-7s | %-7s | %-7s\n", "Slot ID", "Thresh", "Agree%", "FP(gtpf)", "FN(gfpt)");
	Cout() << "----------------------------------------------------------------------\n";

	VectorMap<String, double> best_thresholds;

	for(int i = 0; i < score_data.GetCount(); i++) {
		String gate_id = score_data.GetKey(i);
		const auto& scores = score_data[i];
		
		double best_th = 0.5;
		double max_agree = -1;
		int best_fp = 0, best_fn = 0;

		for(int t = 0; t <= 1000; t++) {
			double th = t / 1000.0;
			int agree = 0, fp = 0, fn = 0;
			for(const auto& s : scores) {
				bool pred = s.score >= th;
				if(pred == s.gt_presence) {
					agree++;
				} else {
					if(pred && !s.gt_presence) fp++;
					else fn++;
				}
			}
			
			if(fn == 0) { // Hard constraint: no false negatives
				double rate = (double)agree / scores.GetCount();
				if(rate > max_agree) {
					max_agree = rate;
					best_th = th;
					best_fp = fp;
					best_fn = fn;
				}
			}
		}

		if(max_agree < 0) {
			int min_fn = scores.GetCount();
			for(int t = 0; t <= 1000; t++) {
				double th = t / 1000.0;
				int fn = 0;
				for(const auto& s : scores) if(!(s.score >= th) && s.gt_presence) fn++;
				if(fn < min_fn) min_fn = fn;
			}
			for(int t = 0; t <= 1000; t++) {
				double th = t / 1000.0;
				int agree = 0, fp = 0, fn = 0;
				for(const auto& s : scores) {
					bool pred = s.score >= th;
					if(pred == s.gt_presence) agree++;
					else { if(pred && !s.gt_presence) fp++; else fn++; }
				}
				if(fn == min_fn) {
					double rate = (double)agree / scores.GetCount();
					if(rate > max_agree) {
						max_agree = rate;
						best_th = th;
						best_fp = fp;
						best_fn = fn;
					}
				}
			}
		}

		Cout() << Format("%-25s |  %.3f  |  %6.2f%% |  %7d |  %7d\n", 
		                 gate_id, best_th, max_agree * 100.0, best_fp, best_fn);
		best_thresholds.Add(gate_id, best_th);
	}
	Cout() << "----------------------------------------------------------------------\n";

	if(args.tune_apply) {
		AnnLay lay;
		AnnSln sln;
		if(sln.Load(args.consistency_sln_path)) {
			String sln_dir = GetFileDirectory(args.consistency_sln_path);
			String annlay_path = ResolveFromBase(sln_dir, sln.annlay);
			
			if(lay.Load(annlay_path)) {
				int updated = 0;
				for(int i = 0; i < lay.slots.GetCount(); i++) {
					int q = best_thresholds.Find(lay.slots[i].id);
					if(q >= 0) {
						lay.slots[i].presence_threshold = best_thresholds[q];
						updated++;
					}
				}
				if(lay.Save(annlay_path)) {
					Cout() << "Successfully updated " << updated << " thresholds in " << annlay_path << "\n";
				}
			}
		}
	}

	return true;
}

struct JointScoreItem : Moveable<JointScoreItem> {
	double gate_score = 0;
	double presence_score = 0;
	bool gt_presence = false;
};

bool GatherJointVisibilityScores(const AppArguments& args, VectorMap<String, Vector<JointScoreItem>>& out_scores) {
	String sln_path = args.joint_tune_sln_path;
	String sln_dir = GetFileDirectory(sln_path);
	AnnSln sln;
	if(!sln.Load(sln_path)) return false;

	String annprj_path = AppendFileName(sln_dir, sln.annprj);
	String images_dir = AppendFileName(sln_dir, "images");
	String annlay_path = AppendFileName(sln_dir, sln.annlay);
	String annmdl_path;
	int pass1 = sln.model_sets.Find("pass1");
	if(pass1 >= 0) annmdl_path = AppendFileName(sln_dir, sln.model_sets[pass1]);

	AnnLay lay;
	if(!lay.Load(annlay_path)) return false;

	AnchoredSlotRecognizer rec;
	rec.SetOffsetMode(args.offset_mode_arg);
	rec.SetBoolGatePolicy(BOOL_GATE_PERMISSIVE); // See raw scores

	if(!rec.Load(annlay_path, annmdl_path)) return false;

	Value root = ParseJSON(LoadFile(annprj_path));
	if(IsNull(root) || !IsValueMap(root)) return false;
	Value datasets = root["datasets"];
	if(!IsValueArray(datasets)) return false;

	Index<String> subset_ids;
	if(!LoadSubsetIds(args.split_file, args.subset, subset_ids)) return false;

	int total_imgs = 0;
	for(int d = 0; d < datasets.GetCount(); d++)
		if(IsValueArray(datasets[d]["images"])) total_imgs += datasets[d]["images"].GetCount();
	
	int kLimit = args.eval_all ? total_imgs : args.eval_count;
	int processed_imgs = 0;

	Cout() << "Collecting joint visibility scores from " << kLimit << " images...\n";

	for(int di = 0; di < datasets.GetCount() && processed_imgs < kLimit; di++) {
		Value images = datasets[di]["images"];
		if(!IsValueArray(images)) continue;
		for(int ii = 0; ii < images.GetCount() && processed_imgs < kLimit; ii++) {
			Value img_rec = images[ii];
			String id = img_rec["file_name"];
			if(id.IsEmpty()) id = img_rec["file_path"];
			if(id.IsEmpty()) id = img_rec["filename"];
			
			if(!subset_ids.IsEmpty() && subset_ids.Find(id) < 0)
				continue;

			String img_path = ResolveImagePathForSmoke(sln_dir, annprj_path, images_dir, img_rec);
			if(img_path.IsEmpty() || !FileExists(img_path)) continue;
			Image img = LoadImageByExtension(img_path);
			if(img.IsEmpty()) continue;

			Vector<SlotResult> results = rec.Recognize(img);
			Value annotations = img_rec["annotations"];
			
			for(int ri = 0; ri < results.GetCount(); ri++) {
				const SlotResult& r = results[ri];
				if(r.gate_slot_id.IsEmpty()) continue;
				if(!r.slot_id.StartsWith("board_card_") && !r.slot_id.StartsWith("hero_card_")) continue;

				// Extract GT presence
				bool gt_presence = false;
				if(IsValueArray(annotations)) {
					for(int ai = 0; ai < annotations.GetCount(); ai++) {
						Value ann = annotations[ai];
						if(ann["metadata"]["mlui_slot_id"] == r.slot_id) {
							Value polys = ann["polygons"];
							if(IsValueArray(polys) && polys.GetCount() > 0) {
								gt_presence = true;
								break;
							}
						}
					}
				}

				double gate_conf = 0;
				bool found_gate = false;
				for(int gj = 0; gj < results.GetCount(); gj++) {
					if(results[gj].slot_id == r.gate_slot_id) {
						if(results[gj].top_class == "true" || results[gj].class_index == 1)
							gate_conf = results[gj].confidence;
						else
							gate_conf = 1.0 - results[gj].confidence;
						found_gate = true;
						break;
					}
				}
				if(!found_gate) continue;

				JointScoreItem& si = out_scores.GetAdd(r.slot_id).Add();
				si.gate_score = gate_conf;
				si.presence_score = r.confidence;
				si.gt_presence = gt_presence;
			}
			processed_imgs++;
			if(processed_imgs % 20 == 0) Cout() << "Processed " << processed_imgs << " images...\n";
		}
	}
	return true;
}

bool RunTuneCardJointPipeline(const AppArguments& args) {
	VectorMap<String, Vector<JointScoreItem>> scores;
	if(!GatherJointVisibilityScores(args, scores)) return false;

	ValueArray per_slot_results;
	ValueMap best_config;

	for(int i = 0; i < scores.GetCount(); i++) {
		String slot_id = scores.GetKey(i);
		const Vector<JointScoreItem>& items = scores[i];

		double best_gate_th = 0.5;
		double best_pres_th = 0.5;
		double min_shared_miss = 1e9;
		int best_fp = 0;
		int best_fn = 0;
		double max_f1 = -1;

		double best_precision = 0;

		// Sweep thresholds
		for(double gth = 0.05; gth <= 1.0; gth += 0.05) {
			if(gth > 0.96) gth = 0.999;
			for(double pth = 0.05; pth <= 1.0; pth += 0.05) {
				if(pth > 0.96) pth = 0.999;

				int tp = 0, fp = 0, fn = 0, tn = 0;
				int shared_miss = 0;

				for(const auto& s : items) {
					bool g_pred = (s.gate_score >= gth);
					bool p_pred = (s.presence_score >= pth);
					bool final_pred = g_pred && p_pred; // STRICT policy

					if(final_pred && s.gt_presence) tp++;
					else if(final_pred && !s.gt_presence) fp++;
					else if(!final_pred && s.gt_presence) {
						fn++;
						if(!g_pred && !p_pred) shared_miss++;
					}
					else tn++;
				}

				double precision = tp + fp > 0 ? (double)tp / (tp + fp) : 1.0;
				double recall = tp + fn > 0 ? (double)tp / (tp + fn) : 0.0;
				double f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;

				// Constrained objective: 
				// 1. Must satisfy precision floor (e.g. 0.90)
				// 2. Minimize shared_miss
				// 3. Maximize F1
				
				const double kMinPrecision = 0.90;
				bool current_ok = (precision >= kMinPrecision);
				bool best_ok = (best_precision >= kMinPrecision);
				
				bool better = false;
				if(current_ok && !best_ok) better = true;
				else if(current_ok == best_ok) {
					if(shared_miss < min_shared_miss) better = true;
					else if(shared_miss == min_shared_miss) {
						if(f1 > max_f1) better = true;
						else if(f1 == max_f1 && fp < best_fp) better = true;
					}
				}

				if(better) {
					min_shared_miss = shared_miss;
					max_f1 = f1;
					best_precision = precision;
					best_fp = fp;
					best_fn = fn;
					best_gate_th = gth;
					best_pres_th = pth;
				}
				if(pth >= 0.99) break;
			}
			if(gth >= 0.99) break;
		}

		ValueMap res;
		res.Add("slot_id", slot_id);
		res.Add("gate_threshold", best_gate_th);
		res.Add("presence_threshold", best_pres_th);
		res.Add("precision", best_precision);
		res.Add("f1", max_f1);
		res.Add("shared_miss", (int)min_shared_miss);
		res.Add("fp", best_fp);
		res.Add("fn", best_fn);
		per_slot_results.Add(res);
		
		ValueMap cfg;
		cfg.Add("gate_threshold", best_gate_th);
		cfg.Add("presence_threshold", best_pres_th);
		best_config.Add(slot_id, cfg);
	}

	ValueMap report;
	report.Add("per_slot", per_slot_results);
	
	if(!args.joint_tune_out_path.IsEmpty()) {
		StoreAsJsonFile(report, args.joint_tune_out_path, true);
		Cout() << "Joint calibration report saved to: " << args.joint_tune_out_path << "\n";
	}

	if(args.joint_tune_apply) {
		String sln_path = args.joint_tune_sln_path;
		String sln_dir = GetFileDirectory(sln_path);
		AnnSln sln;
		sln.Load(sln_path);
		String annlay_path = AppendFileName(sln_dir, sln.annlay);
		AnnLay lay;
		lay.Load(annlay_path);
		int updated = 0;
		for(int i = 0; i < lay.slots.GetCount(); i++) {
			AnnLaySlot& s = lay.slots[i];
			if(best_config.Find(s.id) >= 0) {
				ValueMap cfg = best_config[s.id];
				s.presence_threshold = cfg["presence_threshold"];
				updated++;
			}
			for(int j = 0; j < best_config.GetCount(); j++) {
				String target_id = best_config.GetKey(j);
				if(s.id == "is_" + target_id) {
					ValueMap cfg = best_config.GetValue(j);
					s.presence_threshold = cfg["gate_threshold"];
					updated++;
				}
			}
		}
		lay.Save(annlay_path);
		Cout() << "Applied " << updated << " thresholds to " << annlay_path << "\n";
	}
	return true;
}

bool RunSentinelTestRB(const AppArguments& args) {
	Cout() << "--- Sentinel R/B Channel Test ---\n";
	
	String sln_path = args.sentinel_sln_path;
	String sln_dir = GetFileDirectory(sln_path);
	AnnSln sln;
	if(!sln.Load(sln_path)) {
		Cerr() << "Failed to load solution: " << sln_path << "\n";
		return false;
	}
	
	String annlay_path = NormalizePath(AppendFileName(sln_dir, sln.annlay));
	String annmdl_path = NormalizePath(AppendFileName(sln_dir, sln.model_sets.Get("default", "")));
	
	AnchoredSlotRecognizer rec;
	if(!rec.Load(annlay_path, annmdl_path)) {
		Cerr() << "Failed to load recognizer: " << annlay_path << "\n";
		return false;
	}
	
	Cout() << "Recognizer loaded: " << GetFileName(annlay_path) << "\n";
	
	// Create synthetic red image
	Size sz(100, 100);
	ImageBuffer ib(sz);
	for(RGBA *p = ib.Begin(), *e = ib.End(); p < e; p++) {
		p->r = 255; p->g = 0; p->b = 0; p->a = 255;
	}
	Image source = ib;
	
	Cout() << "Synthetic source: RED (R=255, G=0, B=0)\n";
	
	// Run through processing path
	Vector<SlotResult> results;
	// results = rec.Recognize(source); // We don't necessarily need real detections for this test
	
	// Run overlay render (the suspicious stage)
	Image rendered = RenderOverlayImage(source, results, false);
	
	if(rendered.IsEmpty()) {
		Cerr() << "ERROR: RenderOverlayImage returned empty image.\n";
		return false;
	}
	
	RGBA c = rendered[50][50];
	Cout() << Format("Rendered center pixel: R=%d, G=%d, B=%d\n", (int)c.r, (int)c.g, (int)c.b);
	
	if(c.r > 200 && c.b < 50) {
		Cout() << "SENTINEL TEST PASSED: Red stayed red.\n";
		return true;
	}
	else if(c.b > 200 && c.r < 50) {
		Cerr() << "SENTINEL TEST FAILED: R/B SWAP DETECTED! (Red became Blue)\n";
		return false;
	}
	else {
		Cerr() << "SENTINEL TEST FAILED: Color distortion detected.\n";
		return false;
	}
}

bool RunPipelineCompletenessTest(const AppArguments& args) {
	Cout() << "--- Pipeline Completeness Guardrail Test ---\n";
	
	String sln_path = args.pipeline_test_sln_path;
	String sln_dir = GetFileDirectory(sln_path);
	AnnSln sln;
	if(!sln.Load(sln_path)) {
		Cerr() << "Failed to load solution: " << sln_path << "\n";
		return false;
	}
	
	String annprj_path = NormalizePath(AppendFileName(sln_dir, sln.annprj));
	String annlay_path = NormalizePath(AppendFileName(sln_dir, sln.annlay));
	String annmdl_path = NormalizePath(AppendFileName(sln_dir, sln.model_sets.Get("pass1", "")));
	String images_dir = NormalizePath(AppendFileName(sln_dir, sln.images_dir));
	
	Value annprj_root = ParseJSON(LoadFile(annprj_path));
	if(annprj_root.IsError()) {
		Cerr() << "Failed to parse annprj: " << annprj_path << "\n";
		return false;
	}
	
	// Find first image with annotations
	String test_img_name;
	Value datasets = annprj_root["datasets"];
	for(int i = 0; i < datasets.GetCount() && test_img_name.IsEmpty(); i++) {
		Value images = datasets[i]["images"];
		for(int j = 0; j < images.GetCount(); j++) {
			if(images[j]["annotations"].GetCount() > 0) {
				test_img_name = images[j]["file_name"];
				break;
			}
		}
	}
	
	if(test_img_name.IsEmpty()) {
		Cerr() << "No suitable test image found in annprj.\n";
		return false;
	}
	
	String img_path = AppendFileName(images_dir, test_img_name);
	Cout() << "Testing with image: " << test_img_name << "\n";
	
	AnchoredSlotRecognizer rec;
	if(!rec.Load(annlay_path, annmdl_path)) {
		Cerr() << "Failed to load recognizer: " << annlay_path << "\n";
		return false;
	}
	
	ProcessingLogRecord log;
	uint64 t0 = GetTickCount();
	
	// LOAD
	log.AddStep("LOAD", 0, "START", img_path);
	uint64 t_load_start = GetTickCount();
	Image source = StreamRaster::LoadFileAny(img_path);
	double t_load = (double)(GetTickCount() - t_load_start);
	if(source.IsEmpty()) {
		log.AddStep("LOAD", t_load, "ERROR", "Failed to load image", "Path: " + img_path);
	} else {
		log.AddStep("LOAD", t_load, "OK", Format("%dx%d", source.GetWidth(), source.GetHeight()), "Image loaded successfully");
	}
	
	// RECOGNIZE
	uint64 t_rec_start = GetTickCount();
	Vector<SlotResult> results = rec.Recognize(source, &log);
	double t_rec = (double)(GetTickCount() - t_rec_start);
	log.AddStep("RECOGNIZE", t_rec, results.IsEmpty() ? "WARN" : "OK", Format("%d detections", results.GetCount()));
	
	// SCRIPT
	RecognitionScript script;
	script.Load(AppendFileName(sln_dir, sln.recognition_script));
	uint64 t_script_start = GetTickCount();
	VectorMap<String, String> meta = script.Run(results);
	double t_script = (double)(GetTickCount() - t_script_start);
	log.AddStep("SCRIPT", t_script, "OK", Format("%d meta keys", meta.GetCount()));
	
	// OVERLAY
	uint64 t_overlay_start = GetTickCount();
	Image rendered = RenderOverlayImage(source, results, false);
	double t_overlay = (double)(GetTickCount() - t_overlay_start);
	log.AddStep("OVERLAY", t_overlay, rendered.IsEmpty() ? "ERROR" : "OK");
	
	// TOTAL
	double t_total = (double)(GetTickCount() - t0);
	log.AddStep("TOTAL", t_total, "FINISHED");
	
	Cout() << "\nPipeline Ledger:\n";
	Cout() << Format("%-15s %-10s %-10s %s\n", "Stage", "Status", "Time(ms)", "Note");
	Cout() << "------------------------------------------------------------\n";
	
	Index<String> required_stages;
	required_stages.Add("LOAD");
	required_stages.Add("RECOGNIZE");
	required_stages.Add("SCRIPT");
	required_stages.Add("OVERLAY");
	required_stages.Add("TOTAL");
	
	Index<String> found_stages;
	bool all_ok = true;
	
	for(const auto& step : log.steps) {
		Cout() << Format("%-15s %-10s %-10.1f %s\n", ~step.stage, ~step.status, step.duration_ms, ~step.note);
		found_stages.Add(step.stage);
		if(step.status == "ERROR" && required_stages.Find(step.stage) >= 0) all_ok = false;
	}
	
	Cout() << "\nValidation:\n";
	for(const String& s : required_stages) {
		bool found = found_stages.Find(s) >= 0;
		Cout() << "  Stage [" << s << "]: " << (found ? "FOUND" : "MISSING") << "\n";
		if(!found) all_ok = false;
	}
	
	if(results.IsEmpty()) {
		Cout() << "  Detections: EMPTY (FAIL)\n";
		all_ok = false;
	} else {
		Cout() << "  Detections: " << results.GetCount() << " (PASS)\n";
	}
	
	if(all_ok) {
		Cout() << "\nPIPELINE TEST PASSED\n";
		return true;
	} else {
		Cerr() << "\nPIPELINE TEST FAILED\n";
		return false;
	}
}

bool RunTestFRInvariants(const AppArguments& args) {
	Cout() << "--- FR-Fix2/3 Invariants Regression Test ---\n";
	
	if(!RunSentinelTestRB(args)) {
		Cerr() << "FR Invariants: Sentinel R/B test failed.\n";
		return false;
	}
	
	String sln_path = args.sentinel_sln_path;
	String sln_dir = GetFileDirectory(sln_path);
	AnnSln sln;
	if(!sln.Load(sln_path)) return false;
	
	String annprj_path = NormalizePath(AppendFileName(sln_dir, sln.annprj));
	String annlay_path = NormalizePath(AppendFileName(sln_dir, sln.annlay));
	String annmdl_path = NormalizePath(AppendFileName(sln_dir, sln.model_sets.Get("pass1", "")));
	String images_dir = NormalizePath(AppendFileName(sln_dir, sln.images_dir));
	
	Value annprj_root = ParseJSON(LoadFile(annprj_path));
	if(annprj_root.IsError()) return false;
	
	String test_img_name;
	Value datasets = annprj_root["datasets"];
	for(int i = 0; i < datasets.GetCount() && test_img_name.IsEmpty(); i++) {
		Value images = datasets[i]["images"];
		for(int j = 0; j < images.GetCount(); j++) {
			test_img_name = images[j]["file_name"];
			break;
		}
	}
	
	if(test_img_name.IsEmpty()) return false;
	
	String img_path = AppendFileName(images_dir, test_img_name);
	Image source = StreamRaster::LoadFileAny(img_path);
	if(source.IsEmpty()) return false;
	
	AnchoredSlotRecognizer rec;
	if(!rec.Load(annlay_path, annmdl_path)) return false;
	
	ProcessingLogRecord log;
	rec.Recognize(source, &log);
	
	bool active_timer_group_found = false;
	int active_timer_candidates = 0;
	bool ocr_nodes_ok = true;
	bool malformed_artifacts = false;
	
	for(const auto& s : log.steps) {
		if(s.note.Find("<N/A") >= 0 || s.detailed_note.Find("<N/A") >= 0) {
			Cerr() << "FR Invariants: Malformed artifact found in step: " << s.stage << " note: " << s.note << "\n";
			malformed_artifacts = true;
		}
		
		if(s.slot_id == "active_timer" && s.stage == "CANDIDATE") {
			active_timer_group_found = true;
			active_timer_candidates++;
		}
		
		if(s.stage == "OCR") {
			if(s.note.Find("Step details not found") >= 0 || s.detailed_note.Find("Step details not found") >= 0) {
				Cerr() << "FR Invariants: False details not found in OCR node.\n";
				ocr_nodes_ok = false;
			}
		}
	}
	
	bool success = true;
	if(malformed_artifacts) success = false;
	if(!ocr_nodes_ok) success = false;
	
	if(active_timer_group_found && active_timer_candidates != 8) {
		Cerr() << "FR Invariants: active_timer group candidates count is " << active_timer_candidates << " expected 8.\n";
		success = false;
	} else if(active_timer_group_found) {
		Cout() << "active_timer group verified with " << active_timer_candidates << " candidates.\n";
	}
	
	if(!success) {
		if(args.allow_baseline_mismatch) {
			Cout() << "\nWARNING: FR Invariants failed, but continuing due to --allow-baseline-mismatch (default).\n";
			Cout() << "Note: Pipeline baseline changed; update baseline after intentional changes.\n";
			return true;
		}
		Cerr() << "\nFATAL: FR Invariants failed. Use --allow-baseline-mismatch to bypass if intentional.\n";
		return false;
	}
	
	Cout() << "ALL FR-Fix2/3 INVARIANTS PASSED.\n";
	return true;
}

bool RunEvalOcrModes(const AppArguments& args) {
	Cout() << "--- OCR Preprocessing Mode Comparative Evaluation ---\n";
	
	String sln_path = args.annsln_path;
	String sln_dir = GetFileDirectory(sln_path);
	AnnSln sln;
	if(!sln.Load(sln_path)) {
		Cerr() << "ERROR: Failed to load .annsln: " << sln_path << "\n";
		return false;
	}
	
	String annprj_path = NormalizePath(AppendFileName(sln_dir, sln.annprj));
	String annlay_path = NormalizePath(AppendFileName(sln_dir, sln.annlay));
	String annmdl_path = NormalizePath(AppendFileName(sln_dir, sln.model_sets.Get("pass1", "")));
	String images_dir = NormalizePath(AppendFileName(sln_dir, sln.images_dir));
	
	Value annprj_root = ParseJSON(LoadFile(annprj_path));
	if(annprj_root.IsError()) {
		Cerr() << "ERROR: Failed to parse .annprj: " << annprj_path << "\n";
		return false;
	}
	
	Vector<String> test_images;
	Value datasets = annprj_root["datasets"];
	for(int i = 0; i < datasets.GetCount(); i++) {
		Value images = datasets[i]["images"];
		for(int j = 0; j < images.GetCount(); j++) {
			test_images.Add(images[j]["file_name"]);
			if(test_images.GetCount() >= args.smoke_count) break;
		}
		if(test_images.GetCount() >= args.smoke_count) break;
	}
	
	if(test_images.IsEmpty()) {
		Cerr() << "ERROR: No images found in .annprj\n";
		return false;
	}
	
	AnchoredSlotRecognizer rec;
	if(!rec.Load(annlay_path, annmdl_path)) {
		Cerr() << "ERROR: Failed to load recognizer\n";
		return false;
	}
	
	const AnnLay& lay = rec.GetLayout();
	Vector<String> ocr_slots;
	for(const auto& slot : lay.slots) {
		if(slot.method == ANNLAY_OCR_TEXT)
			ocr_slots.Add(slot.id);
	}
	
	if(ocr_slots.IsEmpty()) {
		Cerr() << "ERROR: No OCR slots found in layout\n";
		return false;
	}
	
	struct ModeStats : Moveable<ModeStats> {
		int recognized = 0;
		int total = 0;
		int exact_matches = 0;
		int gt_count = 0;
		double sum_conf = 0;
		
		double AvgConf() const { return total > 0 ? sum_conf / total : 0; }
		double RecogRate() const { return total > 0 ? (double)recognized / total : 0; }
		double ExactMatchRate() const { return gt_count > 0 ? (double)exact_matches / gt_count : 0; }
	};
	
	Vector<OCR::OCRPreprocessMode> modes;
	modes.Add(OCR::OCR_PREPROCESS_NONE);
	modes.Add(OCR::OCR_PREPROCESS_GRAYSCALE);
	modes.Add(OCR::OCR_PREPROCESS_GRAYSCALE_INVERSE);
	modes.Add(OCR::OCR_PREPROCESS_COLOR_INVERSE);
	modes.Add(OCR::OCR_PREPROCESS_ADAPTIVE_THRESHOLD);
	modes.Add(OCR::OCR_PREPROCESS_ADAPTIVE_THRESHOLD_INVERSE);
	modes.Add(OCR::OCR_PREPROCESS_OTSU);
	modes.Add(OCR::OCR_PREPROCESS_OTSU_INVERSE);

	Index<String> subset_ids = LoadSubsetIds(args.split_file, args.subset);

	// image_path -> slot_id -> GT string
	VectorMap<String, VectorMap<String, String>> ground_truth;
	for(int i = 0; i < datasets.GetCount(); i++) {
		Value images = datasets[i]["images"];
		for(int j = 0; j < images.GetCount(); j++) {
			Value img_rec = images[j];
			String fname = img_rec["file_name"];
			Value keys = img_rec["image_meta_keys"];
			Value vals = img_rec["image_meta_values"];
			if(IsValueArray(keys) && IsValueArray(vals)) {
				auto& gt_map = ground_truth.GetAdd(fname);
				for(int k = 0; k < min(keys.GetCount(), vals.GetCount()); k++) {
					gt_map.Add((String)keys[k], (String)vals[k]);
				}
			}
		}
	}
	
	// slot_id -> mode -> stats
	VectorMap<String, VectorMap<int, ModeStats>> slot_metrics;
	
	Progress progress("Evaluating OCR modes", test_images.GetCount());
	int filtered_count = 0;
	for(int i = 0; i < test_images.GetCount(); i++) {
		String img_fname = test_images[i];
		if(!subset_ids.IsEmpty() && subset_ids.Find(img_fname) < 0)
			continue;
		filtered_count++;
	}

	Cout() << "Evaluation starting on " << filtered_count << " images (subset: " << args.subset << ")\n";

	for(int i = 0; i < test_images.GetCount(); i++) {
		progress.Step();
		
		String img_fname = test_images[i];
		if(!subset_ids.IsEmpty() && subset_ids.Find(img_fname) < 0)
			continue;

		String img_path = AppendFileName(images_dir, img_fname);
		Image img = StreamRaster::LoadFileAny(img_path);
		if(img.IsEmpty()) {
			continue;
		}
		
		const VectorMap<String, String>* img_gt = ground_truth.FindPtr(img_fname);

		for(const String& sid : ocr_slots) {
			const AnnLaySlot* slot = lay.FindSlot(sid);
			if(!slot) continue;
			
			Rect bbox = rec.AnchorToRect(slot->anchor, img.GetSize());
			Image crop = rec.CropAndScale(img, bbox, slot->crop_size);
			if(crop.IsEmpty()) {
				continue;
			}
			
			auto& mode_map = slot_metrics.GetAdd(sid);
			
			String gt_text;
			if(img_gt) {
				int q = img_gt->Find(sid);
				if(q >= 0) gt_text = (*img_gt)[q];
			}

			for(OCR::OCRPreprocessMode mode : modes) {
				auto& stats = mode_map.GetAdd((int)mode);
				stats.total++;
				if(!gt_text.IsEmpty()) stats.gt_count++;
				
				String err;
				double conf = 0;
				Rect out_bbox;
				String text = rec.RunTesseract(crop, sid, err, conf, out_bbox, mode);
				
				if(!text.IsEmpty()) {
					stats.recognized++;
					stats.sum_conf += conf;
					
					if(!gt_text.IsEmpty() && text == gt_text) {
						stats.exact_matches++;
					}
				}
			}
		}
	}
	
	ValueMap report;
	report.Add("subset", args.subset);
	if(!args.split_file.IsEmpty()) report.Add("split_file", args.split_file);

	ValueArray slots_ja;
	
	for(int i = 0; i < slot_metrics.GetCount(); i++) {
		String sid = slot_metrics.GetKey(i);
		auto& mode_map = slot_metrics[i];
		
		const AnnLaySlot* orig_slot = lay.FindSlot(sid);
		OCR::OCRPreprocessMode current_mode = OCR::OCR_PREPROCESS_GRAYSCALE;
		if(orig_slot && !orig_slot->ocr_preprocess_mode.IsEmpty())
			current_mode = OCR::StringToOCRPreprocessMode(orig_slot->ocr_preprocess_mode);

		ValueMap slot_ja;
		slot_ja.Add("slot_id", sid);
		slot_ja.Add("current_mode", OCR::OCRPreprocessModeToString(current_mode));
		
		OCR::OCRPreprocessMode best_mode = current_mode;
		double best_score = -1;
		
		ModeStats current_stats;
		if(mode_map.Find((int)current_mode) >= 0)
			current_stats = mode_map.Get((int)current_mode);

		ValueArray modes_ja;
		for(int j = 0; j < modes.GetCount(); j++) {
			OCR::OCRPreprocessMode mode = modes[j];
			auto& stats = mode_map.GetAdd((int)mode);
			
			ValueMap m_ja;
			m_ja.Add("mode", OCR::OCRPreprocessModeToString(mode));
			m_ja.Add("recognized", stats.recognized);
			m_ja.Add("failed", stats.total - stats.recognized);
			m_ja.Add("total", stats.total);
			m_ja.Add("recog_rate", stats.RecogRate());
			m_ja.Add("avg_confidence", stats.AvgConf());
			if(stats.gt_count > 0) {
				m_ja.Add("exact_matches", stats.exact_matches);
				m_ja.Add("gt_count", stats.gt_count);
				m_ja.Add("exact_match_rate", stats.ExactMatchRate());
			}
			modes_ja.Add(m_ja);
			
			// Scoring: RecogRate * AvgConf + 2 * ExactMatchRate (if GT available)
			double score = stats.RecogRate() * stats.AvgConf();
			if(stats.gt_count > 0) {
				score += 2.0 * stats.ExactMatchRate();
			}

			if(score > best_score) {
				best_score = score;
				best_mode = mode;
			}
		}

		double best_recog = 0;
		double best_conf = 0;
		if(mode_map.Find((int)best_mode) >= 0) {
			const auto& bs = mode_map.Get((int)best_mode);
			best_recog = bs.RecogRate();
			best_conf = bs.AvgConf();
		}

		double recog_gain = best_recog - current_stats.RecogRate();
		double conf_gain = best_conf - current_stats.AvgConf();

		slot_ja.Add("modes", modes_ja);
		slot_ja.Add("best_mode", OCR::OCRPreprocessModeToString(best_mode));
		slot_ja.Add("recog_gain", recog_gain);
		slot_ja.Add("conf_gain", conf_gain);

		String apply_status = "eligible";
		if(best_mode == current_mode) apply_status = "no_change";
		else if(recog_gain < args.min_recog_gain) apply_status = "insufficient_recog_gain";
		else if(conf_gain < args.min_conf_gain) apply_status = "insufficient_conf_gain";
		
		slot_ja.Add("apply_status", apply_status);
		slots_ja.Add(slot_ja);
		
		Cout() << "Slot " << sid << ": recommended=" << OCR::OCRPreprocessModeToString(best_mode) 
		       << " (current=" << OCR::OCRPreprocessModeToString(current_mode) << ")"
		       << " recog_gain=" << Format("%.2f", recog_gain) << " conf_gain=" << Format("%.2f", conf_gain)
		       << " status=" << apply_status << "\n";
	}
	
	report.Add("slots", slots_ja);
	report.Add("timestamp", MakeIsoTimestamp());
	
	if(!args.eval_ocr_out_path.IsEmpty()) {
		if(SaveFile(args.eval_ocr_out_path, StoreAsJson(report, true))) {
			Cout() << "Report saved to: " << args.eval_ocr_out_path << "\n";
		}
	}
	
	if(args.eval_ocr_apply) {
		Cout() << "Applying best modes to .annlay (with safety policy)...\n";
		AnnLay updated_lay;
		if(!updated_lay.Load(annlay_path)) {
			Cerr() << "ERROR: Failed to reload .annlay for update\n";
			return false;
		}
		
		int applied_count = 0;
		for(int i = 0; i < slots_ja.GetCount(); i++) {
			ValueMap s_ja = slots_ja[i];
			if((String)s_ja["apply_status"] != "eligible")
				continue;

			String sid = s_ja["slot_id"];
			String best = s_ja["best_mode"];
			
			AnnLaySlot* s = updated_lay.FindSlot(sid);
			if(s) {
				s->ocr_preprocess_mode = best;
				applied_count++;
			}
		}
		if(updated_lay.Save(annlay_path)) {
			Cout() << "Updated .annlay saved successfully (modified " << applied_count << " slots).\n";
		} else {
			Cerr() << "ERROR: Failed to save updated .annlay\n";
		}
	}
	
	return true;
}

bool RunDumpNNSteps(const AppArguments& args) {
	Cout() << "--- Neural Network Step Dump (" << args.dump_nn_slot << " @ " << args.dump_nn_stage << ") ---\n";
	
	String sln_path = args.annsln_path;
	String sln_dir = GetFileDirectory(sln_path);
	AnnSln sln;
	if(!sln.Load(sln_path)) return false;
	
	String annprj_path = NormalizePath(AppendFileName(sln_dir, sln.annprj));
	String annlay_path = NormalizePath(AppendFileName(sln_dir, sln.annlay));
	String annmdl_path = NormalizePath(AppendFileName(sln_dir, sln.model_sets.Get("pass1", "")));
	String images_dir = NormalizePath(AppendFileName(sln_dir, sln.images_dir));
	
	Value annprj_root = ParseJSON(LoadFile(annprj_path));
	if(annprj_root.IsError()) return false;
	
	String test_img_name;
	Value datasets = annprj_root["datasets"];
	for(int i = 0; i < datasets.GetCount() && test_img_name.IsEmpty(); i++) {
		Value images = datasets[i]["images"];
		for(int j = 0; j < images.GetCount(); j++) {
			test_img_name = images[j]["file_name"];
			break;
		}
	}
	
	if(test_img_name.IsEmpty()) return false;
	
	String img_path = AppendFileName(images_dir, test_img_name);
	Image source = StreamRaster::LoadFileAny(img_path);
	if(source.IsEmpty()) return false;
	
	AnchoredSlotRecognizer rec;
	if(!rec.Load(annlay_path, annmdl_path)) return false;
	
	ProcessingLogRecord log;
	rec.Recognize(source, &log);
	
	const ProcessingStepRecord* target_step = nullptr;
	for(const auto& s : log.steps) {
		if(s.slot_id == args.dump_nn_slot && s.stage == args.dump_nn_stage) {
			target_step = &s;
			break;
		}
	}
	
	if(!target_step) {
		Cerr() << "ERROR: Target step not found in log.\n";
		return false;
	}
	
	if(!target_step->is_nn_step) {
		Cerr() << "ERROR: Selected step is not an NN step.\n";
		return false;
	}

	::ConvNet::Session* ses = rec.GetSession(target_step->head_id);
	if(!ses) {
		Cerr() << "ERROR: Session not found for head: " << target_step->head_id << "\n";
		return false;
	}
	
	// Ensure activations are current for the specific crop
	rec.PredictCrop(target_step->head_id, source,
	               target_step->candidate_bbox, target_step->crop_size,
	               target_step->is_grayscale, target_step->angle, target_step->is_equalized);
	
	ConvNet::SessionConvLayers inspector;
	inspector.SetSession(*ses);
	inspector.Dump();
	
	return true;
}

bool RunTestXOffsetConvLayers(const AppArguments& args) {
	String sln_path = NormalizePath(args.test_xoffset_sln_path);
	if(!FileExists(sln_path)) {
		Cout() << "--test-xoffset-convlayers: .annsln file not found: " << sln_path << "\n";
		return false;
	}
	
	AnnSln sln;
	if(!sln.Load(sln_path)) {
		Cout() << "--test-xoffset-convlayers: failed to load .annsln\n";
		return false;
	}
	
	FrameRecognizerWindow win;
	String sln_dir = GetFileDirectory(sln_path);
	String annprj = sln.annprj;
	if(!IsFullPath(annprj))
		annprj = NormalizePath(AppendFileName(sln_dir, annprj));
	
	win.OpenSlideshow(sln, sln_dir, annprj, "pass1");
	
	// Post the test logic to be executed once the event loop starts
	win.PostCallback([&win, &args] {
		win.TestDumpXOffsetConvLayers(args.test_xoffset_verbose);
		win.Break();
	});
	
	win.Run();
	return true;
}

bool RunFrameRecognizerDumpSteps(const AppArguments& args) {
	if(args.frame_recognizer_dump_memory_alloc)
		SetFrameRecognizerMemoryDumpEnabled(true);

	String sln_path = args.dump_steps_sln_path;
	if(!FileExists(sln_path)) {
		Cout() << "Error: .annsln file not found: " << sln_path << "\n";
		return false;
	}

	AnnSln sln;
	if(!sln.Load(sln_path)) {
		Cout() << "Error: failed to load .annsln: " << sln_path << "\n";
		return false;
	}

	String sln_dir = GetFileDirectory(sln_path);
	String annlay_path = sln.annlay;
	if(!IsFullPath(annlay_path))
		annlay_path = NormalizePath(AppendFileName(sln_dir, annlay_path));
	String requested_model_set = TrimBoth(args.dump_steps_model_set);
	if(requested_model_set.IsEmpty())
		requested_model_set = "pass1";
	String annmdl_path;
	String model_set_value;
	bool model_set_found = false;
	bool used_fallback_model = false;
	int ms = sln.model_sets.Find(requested_model_set);
	if(ms >= 0) {
		model_set_found = true;
		model_set_value = TrimBoth(sln.model_sets[ms]);
		if(!model_set_value.IsEmpty())
			annmdl_path = NormalizePath(AppendFileName(sln_dir, model_set_value));
	}
	if(!model_set_found) {
		Cout() << "WARNING: model-set '" << requested_model_set
		       << "' not found in .annsln model_sets; falling back to annlay sidecar model.\n";
	}
	else if(model_set_value.IsEmpty()) {
		Cout() << "WARNING: model-set '" << requested_model_set
		       << "' is configured but empty; falling back to annlay sidecar model.\n";
	}
	else if(!FileExists(annmdl_path)) {
		Cout() << "WARNING: model-set '" << requested_model_set
		       << "' points to missing model file: " << annmdl_path
		       << " ; falling back to annlay sidecar model.\n";
	}
	if(annmdl_path.IsEmpty() || !FileExists(annmdl_path)) {
		annmdl_path = AnnMdl::PathFromAnnlay(annlay_path);
		used_fallback_model = true;
	}

	Cout() << "dump-steps: requested model-set: " << requested_model_set << "\n";
	if(model_set_found)
		Cout() << "dump-steps: model-set entry: " << (model_set_value.IsEmpty() ? String("<empty>") : model_set_value) << "\n";
	Cout() << "dump-steps: resolved model path: " << annmdl_path
	       << (used_fallback_model ? " (fallback)" : "") << "\n";

	if(!FileExists(annlay_path)) {
		Cout() << "Error: .annlay file not found: " << annlay_path << "\n";
		return false;
	}

	AnchoredSlotRecognizer rec;
	rec.SetSln(sln);
	if(!rec.Load(annlay_path, annmdl_path)) {
		Cout() << "Error: failed to load recognizer from: " << annlay_path
		       << " model: " << annmdl_path << "\n";
		return false;
	}

	// Find the image in annprj datasets
	String annprj_path = sln.annprj;
	if(!IsFullPath(annprj_path))
		annprj_path = NormalizePath(AppendFileName(sln_dir, annprj_path));

	Value prj = ParseJSON(LoadFile(annprj_path));
	if(IsNull(prj)) {
		Cout() << "Error: failed to load annprj: " << annprj_path << "\n";
		return false;
	}

	String img_path;
	Value datasets = prj["datasets"];
	for(int i = 0; i < datasets.GetCount(); i++) {
		Value imgs = datasets[i]["images"];
		for(int j = 0; j < imgs.GetCount(); j++) {
			String name = imgs[j]["file_name"];
			if(GetFileName(name) == GetFileName(args.dump_steps_img_name)) {
				img_path = imgs[j]["file_path"];
				if(!IsFullPath(img_path)) {
					String base_dir = GetFileDirectory(annprj_path);
					img_path = NormalizePath(AppendFileName(base_dir, img_path));
				}
				break;
			}
		}
		if(!img_path.IsEmpty()) break;
	}

	if(img_path.IsEmpty()) {
		Cout() << "Error: image not found in project: " << args.dump_steps_img_name << "\n";
		return false;
	}

	uint64 t_load_start = GetTickCount();
	Image img = StreamRaster::LoadFileAny(img_path);
	double t_load_ms = (double)(GetTickCount() - t_load_start);
	if(img.IsEmpty()) {
		Cout() << "Error: failed to load image file: " << img_path << "\n";
		return false;
	}

	ProcessingLogRecord log;
	log.t_load_ms = t_load_ms;
	log.AddStep("LOAD", t_load_ms, "OK", Format("%d x %d", img.GetWidth(), img.GetHeight()),
	            String(), String(), false, "Path: " + img_path);
	log.results = rec.Recognize(img, &log);

	RecognitionScript script;
	if(!sln.recognition_script.IsEmpty()) {
		String script_path = NormalizePath(AppendFileName(sln_dir, sln.recognition_script));
		if(FileExists(script_path))
			script.Load(script_path);
	}
	if(script.IsLoaded()) {
		uint64 t_script_start = GetTickCount();
		log.meta = script.Run(log.results);
		log.t_script_ms = (double)(GetTickCount() - t_script_start);
		log.script_output = script.GetLastOutput();
		log.script_error = script.GetLastError();
		log.AddStep("SCRIPT", log.t_script_ms,
		            log.script_error.IsEmpty() ? "OK" : "ERROR",
		            log.script_error.IsEmpty()
		                ? Format("%d meta keys", log.meta.GetCount())
		                : log.script_error,
		            "GameState");
	}

	Cout() << "\nProcessing steps for image: " << args.dump_steps_img_name << "\n";
	Vector<ProcessingTreeNode> nodes;
	Vector<ProcessingInputRef> inputs;
	BuildProcessingStepsTree(log, sln, nodes, &inputs);

	String input_crops_dir = NormalizePath(AppendFileName(AppendFileName(GetCurrentDirectory(), "tmp"), "input_crops"));
	if(DirectoryExists(input_crops_dir))
		DeleteFolderDeep(input_crops_dir);
	RealizeDirectory(input_crops_dir);

	int saved_input_crops = 0;
	for(int i = 0; i < inputs.GetCount(); i++) {
		const ProcessingInputRef& in = inputs[i];
		Rect b = in.bbox;
		b.left = max(0, b.left);
		b.top = max(0, b.top);
		b.right = min(img.GetWidth(), b.right);
		b.bottom = min(img.GetHeight(), b.bottom);
		if(b.IsEmpty())
			continue;

		Image crop = Crop(img, b);
		if(crop.IsEmpty())
			continue;

		String out_path = AppendFileName(input_crops_dir, in.id + ".png");
		if(SavePngImage(crop, out_path))
			saved_input_crops++;
	}

	Cout() << "Input crops dir: " << input_crops_dir << "\n";
	Cout() << "Input crops saved: " << saved_input_crops << "\n";
	Cout() << "Input crops:\n";
	if(inputs.IsEmpty()) {
		Cout() << "  (none)\n";
	}
	else {
		for(int i = 0; i < inputs.GetCount(); i++) {
			const ProcessingInputRef& in = inputs[i];
			Cout() << Format("  %-4s x=%d y=%d w=%d h=%d\n",
			                 ~in.id, in.bbox.left, in.bbox.top,
			                 in.bbox.GetWidth(), in.bbox.GetHeight());
		}
	}

	VectorMap<String, String> model_id_by_head;
	VectorMap<String, String> cv_model_id_by_head;
	// requested_head_for_resolved[resolved_head] = set of logical requested heads that resolved to it
	VectorMap<String, Index<String>> requested_heads_by_resolved;
	Index<String> all_heads;
	int fallback_count = 0;

	for(int i = 0; i < log.steps.GetCount(); i++) {
		String head = TrimBoth(log.steps[i].head_id);
		if(head.IsEmpty())
			continue;
		
		all_heads.FindAdd(head);
		
		bool is_actual_nn = log.steps[i].is_nn_step;

		if(is_actual_nn && model_id_by_head.Find(head) < 0) {
			String model_id = Format("M%d", model_id_by_head.GetCount() + 1);
			model_id_by_head.Add(head, model_id);
		}
		else if(!is_actual_nn &&
		        (TrimBoth(log.steps[i].stage) == "LEVEL" ||
		         TrimBoth(log.steps[i].stage) == "LABEL_A" ||
		         TrimBoth(log.steps[i].stage) == "LABEL_B") &&
		        cv_model_id_by_head.Find(head) < 0) {
			String model_id = Format("T%d", cv_model_id_by_head.GetCount() + 1);
			cv_model_id_by_head.Add(head, model_id);
		}
		
		String req = TrimBoth(log.steps[i].requested_head_id);
		if(!req.IsEmpty() && req != head) {
			requested_heads_by_resolved.GetAdd(head).FindAdd(req);
			fallback_count++;
		}
	}

	Cout() << "convnet-neural-network:\n";
	if(model_id_by_head.IsEmpty()) {
		Cout() << "  (none)\n";
	}
	else {
		for(int i = 0; i < model_id_by_head.GetCount(); i++) {
			const String& head = model_id_by_head.GetKey(i);
			const String& model_id = model_id_by_head[i];
			String model_note = "";
			if(::ConvNet::Session* ses = rec.GetSession(head)) {
				int in_w = ses->GetInput() ? (int)round(ses->GetInput()->input_width) : 0;
				int in_h = ses->GetInput() ? (int)round(ses->GetInput()->input_height) : 0;
				int in_d = ses->GetInput() ? (int)round(ses->GetInput()->input_depth) : 0;
				model_note = " input=" + AsString(in_w) + "x" + AsString(in_h) + "x" + AsString(in_d)
				           + " classes=" + AsString((int)ses->Data().GetClassCount());
			}
			else {
				model_note = " (not loaded)";
			}
			Cout() << Format("  %-4s head=\"%s\"%s\n", ~model_id, ~head, ~model_note);
			int ri = requested_heads_by_resolved.Find(head);
			if(ri >= 0) {
				const Index<String>& reqs = requested_heads_by_resolved[ri];
				for(int j = 0; j < reqs.GetCount(); j++)
					Cout() << Format("       via \"%s\" (group-resolved)\n", ~reqs[j]);
			}
		}
	}

	Cout() << "\ncv-match-template:\n";
	bool any_cv = false;
	for(int i = 0; i < cv_model_id_by_head.GetCount(); i++) {
		const String& head = cv_model_id_by_head.GetKey(i);
		const String& model_id = cv_model_id_by_head[i];
		any_cv = true;
		Cout() << Format("  %-4s head=\"%s\"\n", ~model_id, ~head);
		int ri = requested_heads_by_resolved.Find(head);
		if(ri >= 0) {
			const Index<String>& reqs = requested_heads_by_resolved[ri];
			for(int j = 0; j < reqs.GetCount(); j++)
				Cout() << Format("       via \"%s\" (group-resolved)\n", ~reqs[j]);
		}
	}
	if(!any_cv) Cout() << "  (none)\n";

	Cout() << "\nLoaded-but-unused-sessions:\n";
	Vector<String> loaded_heads = rec.GetLoadedSessionHeads();
	bool any_unused_loaded = false;
	for(int i = 0; i < loaded_heads.GetCount(); i++) {
		const String& head = loaded_heads[i];
		if(all_heads.Find(head) >= 0)
			continue;
		any_unused_loaded = true;
		String model_note;
		if(::ConvNet::Session* ses = rec.GetSession(head)) {
			int in_w = ses->GetInput() ? (int)round(ses->GetInput()->input_width) : 0;
			int in_h = ses->GetInput() ? (int)round(ses->GetInput()->input_height) : 0;
			int in_d = ses->GetInput() ? (int)round(ses->GetInput()->input_depth) : 0;
			model_note = " input=" + AsString(in_w) + "x" + AsString(in_h) + "x" + AsString(in_d)
			           + " classes=" + AsString((int)ses->Data().GetClassCount());
		}
		Cout() << Format("       head=\"%s\"%s\n", ~head, ~model_note);
	}
	if(!any_unused_loaded)
		Cout() << "  (none)\n";

	if(fallback_count == 0 && !model_id_by_head.IsEmpty())
		Cout() << "Head resolution: all canonical (no fallbacks)\n";
	else if(fallback_count > 0)
		Cout() << Format("Head resolution: %d group-resolved step(s) (requested != session key)\n", fallback_count);

	Cout() << String('=', 80) << "\n";
	Cout() << Format("%-25s %-12s %-10s %-15s %s\n", "Step", "Stage", "Timing", "Status", "Note");
	Cout() << String('-', 80) << "\n";

	const String kTreeVert   = "│ ";
	const String kTreeBranch = "├─";
	const String kTreeLast   = "└─";
	const String kTreeSpace  = "  ";

	Function<void(int, const String&, bool, bool)> PrintNode =
	[&](int idx, const String& prefix, bool is_last, bool is_root) {
		String step_col = is_root ? nodes[idx].label : prefix + (is_last ? kTreeLast : kTreeBranch) + nodes[idx].label;
		if(nodes[idx].is_leaf) {
			String note = nodes[idx].note;
			if(nodes[idx].source_index >= 0 && nodes[idx].source_index < log.steps.GetCount()) {
				const ProcessingStepRecord& s = log.steps[nodes[idx].source_index];
				String head = TrimBoth(s.head_id);
				if(!head.IsEmpty()) {
					int m = model_id_by_head.Find(head);
					if(m >= 0) {
						if(!note.IsEmpty())
							note << " ";
						note << "model=" << model_id_by_head[m];
					}
					else {
						int t = cv_model_id_by_head.Find(head);
						if(t >= 0) {
							if(!note.IsEmpty())
								note << " ";
							note << "model=" << cv_model_id_by_head[t];
						}
					}
				}
			}
			if(nodes[idx].stage == "LOGICAL") {
				Cout() << Format("%-25s %-12s %-27s%s\n",
				                 step_col, nodes[idx].stage, "", note);
			} else {
				Cout() << Format("%-25s %-12s %8.2f ms %-15s %s\n",
				                 step_col, nodes[idx].stage, nodes[idx].duration_ms,
				                 nodes[idx].status, note);
			}
		}
		else {
			Cout() << step_col << "\n";
		}
		String child_prefix = is_root ? String() : (prefix + (is_last ? kTreeSpace : kTreeVert));
		for(int ci = 0; ci < nodes[idx].children.GetCount(); ci++) {
			bool child_last = (ci + 1 == nodes[idx].children.GetCount());
			PrintNode(nodes[idx].children[ci], child_prefix, child_last, false);
		}
	};

	PrintNode(0, String(), true, true);
	Cout() << String('=', 80) << "\n\n";

	return true;
}

bool RunFrameRecognizerDumpMemoryAlloc(const AppArguments& args) {
	SetFrameRecognizerMemoryDumpEnabled(true);

	String sln_path = args.annsln_path;
	if(!FileExists(sln_path)) {
		Cout() << "Error: .annsln file not found: " << sln_path << "\n";
		return false;
	}

	DumpFrameRecognizerMemoryEvent("Command.DumpMemoryAlloc.begin", "sln=" + sln_path);

	AnnSln sln;
	if(!sln.Load(sln_path)) {
		Cout() << "Error: failed to load .annsln: " << sln_path << "\n";
		return false;
	}

	String sln_dir = GetFileDirectory(sln_path);
	String annlay_path = sln.annlay;
	if(!IsFullPath(annlay_path))
		annlay_path = NormalizePath(AppendFileName(sln_dir, annlay_path));

	AnchoredSlotRecognizer rec;
	rec.Load(annlay_path);

	DumpFrameRecognizerMemoryEvent("Command.DumpMemoryAlloc.ready", "recognizer loaded");

	if(!args.dump_steps_img_name.IsEmpty()) {
		String annprj_path = sln.annprj;
		if(!IsFullPath(annprj_path))
			annprj_path = NormalizePath(AppendFileName(sln_dir, annprj_path));

		Value prj = ParseJSON(LoadFile(annprj_path));
		if(IsNull(prj)) {
			Cout() << "Error: failed to load annprj: " << annprj_path << "\n";
			return false;
		}

		String img_path;
		Value datasets = prj["datasets"];
		for(int i = 0; i < datasets.GetCount(); i++) {
			Value imgs = datasets[i]["images"];
			for(int j = 0; j < imgs.GetCount(); j++) {
				String name = imgs[j]["file_name"];
				if(GetFileName(name) == GetFileName(args.dump_steps_img_name)) {
					img_path = imgs[j]["file_path"];
					if(!IsFullPath(img_path)) {
						String base_dir = GetFileDirectory(annprj_path);
						img_path = NormalizePath(AppendFileName(base_dir, img_path));
					}
					break;
				}
			}
			if(!img_path.IsEmpty()) break;
		}

		if(img_path.IsEmpty()) {
			Cout() << "Error: image not found in project: " << args.dump_steps_img_name << "\n";
			return false;
		}

		Image img = StreamRaster::LoadFileAny(img_path);
		if(img.IsEmpty()) {
			Cout() << "Error: failed to load image file: " << img_path << "\n";
			return false;
		}

		DumpFrameRecognizerMemoryEvent("Command.DumpMemoryAlloc.recognize.begin",
			Format("frame='%s'", args.dump_steps_img_name));
		Vector<SlotResult> result = rec.Recognize(img);
		DumpFrameRecognizerMemoryEvent("Command.DumpMemoryAlloc.recognize.done",
			Format("frame='%s' slots=%d", args.dump_steps_img_name, result.GetCount()));

		Cout() << "\nRecognition result for: " << args.dump_steps_img_name << "\n";
		Cout() << String('=', 60) << "\n";
		for(int i = 0; i < result.GetCount(); i++) {
			const SlotResult& sr = result[i];
			Cout() << Format("  %-30s = %s (conf=%.3f)\n", sr.slot_id, sr.top_class, sr.confidence);
		}
		Cout() << String('=', 60) << "\n";
	}

	return true;
}


} // namespace Upp
