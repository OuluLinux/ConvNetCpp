#include "AnnotationDomain.h"
#include <AnnLayCore/AnchoredSlotRecognizer.h>
#include <AnnLayCore/RecognitionScript.h>

NAMESPACE_UPP

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

void AddPin(Node::NodeDoc& n, const String& id, Node::PinKind kind, const String& type_name,
            Color clr = Color(180, 180, 180)) {
	Node::PinDoc& p = n.pins.Add();
	p.id        = id;
	p.label     = id;
	p.kind      = kind;
	p.type_name = type_name;
	p.color     = clr;
}

void AddSlot(Node::NodeDoc& n, const String& id, const String& type, const Value& value) {
	Node::WidgetSlotDoc& s = n.slots.Add();
	s.id   = id;
	s.type = type;
	s.properties.GetAdd("value") = value;
}

String SlotVal(const Node::NodeDoc& n, const String& slot_id) {
	for(const Node::WidgetSlotDoc& s : n.slots) {
		if(s.id == slot_id) {
			int vi = s.properties.Find("value");
			if(vi >= 0) return s.properties.GetValue(vi).ToString();
		}
	}
	return String();
}

// Pin colors by semantic type
static Color kClrPath    = Color(160, 200, 240); // blue-grey — file paths
static Color kClrStatus  = Color(180, 240, 160); // green — status/bool output
static Color kClrData    = Color(240, 200, 120); // amber — data outputs

// ---------------------------------------------------------------------------
// AnnotationDomainImpl
// ---------------------------------------------------------------------------

class AnnotationDomainImpl : public INodeWorkbenchDomain {
	struct TemplateDef : Moveable<TemplateDef> {
		String type_id, category, label;
		Node::NodeDoc doc;
	};
	Vector<TemplateDef> templates;
	bool palette_ready = false;

	void EnsurePalette() {
		if(palette_ready) return;
		palette_ready = true;
		templates.Clear();

		auto Add = [&](const String& type_id, const String& category, const String& label,
		               Color fill, Color line,
		               Function<void(Node::NodeDoc&)> fn) {
			TemplateDef& t = templates.Add();
			t.type_id = type_id; t.category = category; t.label = label;
			t.doc.node_type_id = type_id; t.doc.category = category; t.doc.label = label;
			t.doc.fill_clr = fill; t.doc.line_clr = line;
			t.doc.sz = Sizef(280, 80);
			fn(t.doc);
		};

		Color bg_load    = Color(28, 38, 55);
		Color bg_compute = Color(32, 42, 32);
		Color bg_train   = Color(45, 35, 25);
		Color bg_recog   = Color(35, 28, 50);
		Color bg_qa      = Color(50, 40, 28);
		Color bg_launch  = Color(45, 45, 45);

		Color ln_load    = Color(80,  130, 200);
		Color ln_compute = Color(80,  160, 80);
		Color ln_train   = Color(200, 130, 60);
		Color ln_recog   = Color(140, 80,  200);
		Color ln_qa      = Color(200, 160, 60);
		Color ln_launch  = Color(120, 120, 120);

		// Stage 0: Load solution
		Add("ann.load_sln", "ann.io", "Load Solution (.annsln)", bg_load, ln_load, [](Node::NodeDoc& n) {
			n.sz = Sizef(280, 100);
			AddPin(n, "sln_path", Node::PinKind::Output, "ANNSLN_PATH", kClrPath);
			AddPin(n, "annlay_path", Node::PinKind::Output, "ANNLAY_PATH", kClrPath);
			AddPin(n, "annprj_path", Node::PinKind::Output, "ANNPRJ_PATH", kClrPath);
			AddSlot(n, "sln_file", "EditString", String());
		});

		// Stage 2: Recompute anchors
		Add("ann.recompute_anchors", "ann.pipeline", "Recompute Anchors", bg_compute, ln_compute, [](Node::NodeDoc& n) {
			n.sz = Sizef(280, 80);
			AddPin(n, "annlay_in",  Node::PinKind::Input,  "ANNLAY_PATH", kClrPath);
			AddPin(n, "annprj_in",  Node::PinKind::Input,  "ANNPRJ_PATH", kClrPath);
			AddPin(n, "annlay_out", Node::PinKind::Output, "ANNLAY_PATH", kClrPath);
			AddPin(n, "status",     Node::PinKind::Output, "STATUS",      kClrStatus);
		});

		// Stage 3: Export crops
		Add("ann.export_crops", "ann.pipeline", "Export Crops", bg_compute, ln_compute, [](Node::NodeDoc& n) {
			n.sz = Sizef(280, 120);
			AddPin(n, "annlay_in",    Node::PinKind::Input,  "ANNLAY_PATH", kClrPath);
			AddPin(n, "annprj_in",    Node::PinKind::Input,  "ANNPRJ_PATH", kClrPath);
			AddPin(n, "crops_dir_out",Node::PinKind::Output, "CROPS_DIR",   kClrPath);
			AddPin(n, "status",       Node::PinKind::Output, "STATUS",      kClrStatus);
			AddSlot(n, "crops_root",   "EditString", String());
			AddSlot(n, "images_dir",   "EditString", String());
			AddSlot(n, "templates_dir","EditString", String());
			AddSlot(n, "pass_index",   "EditIntSpin", 1);
		});

		// Stage 4b: Train all slots (CLI batch)
		Add("ann.train_slots_cli", "ann.pipeline", "Train Slots (CLI)", bg_train, ln_train, [](Node::NodeDoc& n) {
			n.sz = Sizef(280, 80);
			AddPin(n, "annlay_in",  Node::PinKind::Input,  "ANNLAY_PATH", kClrPath);
			AddPin(n, "crops_dir",  Node::PinKind::Input,  "CROPS_DIR",   kClrPath);
			AddPin(n, "status",     Node::PinKind::Output, "STATUS",      kClrStatus);
			AddSlot(n, "epochs", "EditIntSpin", 50);
		});

		// Stage 5: Recognition run + script + annprj writeback
		Add("ann.recognize", "ann.pipeline", "Recognize + Script", bg_recog, ln_recog, [](Node::NodeDoc& n) {
			n.sz = Sizef(280, 100);
			AddPin(n, "annlay_in",    Node::PinKind::Input,  "ANNLAY_PATH", kClrPath);
			AddPin(n, "annprj_in",    Node::PinKind::Input,  "ANNPRJ_PATH", kClrPath);
			AddPin(n, "annprj_out",   Node::PinKind::Output, "ANNPRJ_PATH", kClrPath);
			AddPin(n, "images_done",  Node::PinKind::Output, "DATA",        kClrData);
			AddPin(n, "status",       Node::PinKind::Output, "STATUS",      kClrStatus);
			AddSlot(n, "images_dir",         "EditString", String());
			AddSlot(n, "recognition_script", "EditString", String());
			AddSlot(n, "offset_mode",        "DropList",   String("auto"));
			AddSlot(n, "bool_gate_policy",   "DropList",   String("strict"));
		});

		// Stage 7: Audit bool gates
		Add("ann.audit_bool", "ann.qa", "Audit Bool Gates", bg_qa, ln_qa, [](Node::NodeDoc& n) {
			n.sz = Sizef(280, 80);
			AddPin(n, "annlay_in",    Node::PinKind::Input,  "ANNLAY_PATH", kClrPath);
			AddPin(n, "report_path",  Node::PinKind::Output, "REPORT_PATH", kClrPath);
			AddPin(n, "status",       Node::PinKind::Output, "STATUS",      kClrStatus);
			AddSlot(n, "out_dir", "EditString", String());
			AddSlot(n, "strict",  "Option",     true);
		});

		// Stage 8: Fix bool gates
		Add("ann.fix_bool", "ann.qa", "Fix Bool Gates", bg_qa, ln_qa, [](Node::NodeDoc& n) {
			n.sz = Sizef(280, 60);
			AddPin(n, "annlay_in",  Node::PinKind::Input,  "ANNLAY_PATH", kClrPath);
			AddPin(n, "annlay_out", Node::PinKind::Output, "ANNLAY_PATH", kClrPath);
			AddPin(n, "status",     Node::PinKind::Output, "STATUS",      kClrStatus);
		});

		// Stage 9: Eval bool gates
		Add("ann.eval_bool", "ann.qa", "Eval Bool Gates", bg_qa, ln_qa, [](Node::NodeDoc& n) {
			n.sz = Sizef(280, 100);
			AddPin(n, "annsln_in",  Node::PinKind::Input,  "ANNSLN_PATH", kClrPath);
			AddPin(n, "status",     Node::PinKind::Output, "STATUS",      kClrStatus);
			AddSlot(n, "split_json",       "EditString", String());
			AddSlot(n, "bool_gate_policy", "DropList",   String("strict"));
			AddSlot(n, "offset_mode",      "DropList",   String("auto"));
		});

		// Stage 10: Check baseline
		Add("ann.check_baseline", "ann.qa", "Check Baseline", bg_qa, ln_qa, [](Node::NodeDoc& n) {
			n.sz = Sizef(280, 120);
			AddPin(n, "annsln_in",    Node::PinKind::Input,  "ANNSLN_PATH", kClrPath);
			AddPin(n, "report_path",  Node::PinKind::Output, "REPORT_PATH", kClrPath);
			AddPin(n, "status",       Node::PinKind::Output, "STATUS",      kClrStatus);
			AddSlot(n, "baseline_json",        "EditString", String());
			AddSlot(n, "offset_baseline_json", "EditString", String());
			AddSlot(n, "split_json",           "EditString", String());
			AddSlot(n, "out_dir",              "EditString", String());
			AddSlot(n, "bool_gate_policy",     "DropList",   String("strict"));
			AddSlot(n, "offset_mode",          "DropList",   String("auto"));
		});

		// Stage 11: Preflight / exit gate
		Add("ann.preflight", "ann.qa", "Preflight / Exit Gate", bg_qa, ln_qa, [](Node::NodeDoc& n) {
			n.sz = Sizef(280, 120);
			AddPin(n, "annsln_in",    Node::PinKind::Input,  "ANNSLN_PATH", kClrPath);
			AddPin(n, "passed",       Node::PinKind::Output, "STATUS",      kClrStatus);
			AddPin(n, "report_dir",   Node::PinKind::Output, "REPORT_PATH", kClrPath);
			AddSlot(n, "bool_baseline",       "EditString", String());
			AddSlot(n, "offset_baseline",     "EditString", String());
			AddSlot(n, "split_json",          "EditString", String());
			AddSlot(n, "out_dir",             "EditString", String());
			AddSlot(n, "bool_gate_policy",    "DropList",   String("strict"));
			AddSlot(n, "offset_mode",         "DropList",   String("auto"));
			AddSlot(n, "fail_on_warnings",    "Option",     false);
		});

		// GUI-launch nodes (not automatable)
		Add("ann.open_annotation_editor", "ann.gui", "Open Annotation Editor", bg_launch, ln_launch, [](Node::NodeDoc& n) {
			n.sz = Sizef(260, 70);
			AddPin(n, "annprj_in",  Node::PinKind::Input,  "ANNPRJ_PATH", kClrPath);
			AddPin(n, "annlay_in",  Node::PinKind::Input,  "ANNLAY_PATH", kClrPath);
			AddSlot(n, "note", "EditString", String("(opens GUI — not automatable)"));
		});

		Add("ann.open_frame_recognizer", "ann.gui", "Open Frame Recognizer", bg_launch, ln_launch, [](Node::NodeDoc& n) {
			n.sz = Sizef(260, 80);
			AddPin(n, "annsln_in",  Node::PinKind::Input,  "ANNSLN_PATH", kClrPath);
			AddPin(n, "annprj_in",  Node::PinKind::Input,  "ANNPRJ_PATH", kClrPath);
			AddSlot(n, "model_set",        "DropList", String("pass1"));
			AddSlot(n, "offset_mode",      "DropList", String("auto"));
			AddSlot(n, "bool_gate_policy", "DropList", String("strict"));
		});
	}

	// -----------------------------------------------------------------------
	// RunGraph helpers — execute headless pipeline stages
	// -----------------------------------------------------------------------

	bool RunRecomputeAnchors(const Node::NodeDoc& nd, String& log_out) {
		String annlay_path = SlotVal(nd, "annlay_in");
		String annprj_path = SlotVal(nd, "annprj_in");
		if(annlay_path.IsEmpty()) { log_out << "[recompute_anchors] annlay_in slot empty.\n"; return false; }
		if(annprj_path.IsEmpty()) { log_out << "[recompute_anchors] annprj_in slot empty.\n"; return false; }

		AnnLay lay;
		if(!lay.Load(annlay_path)) {
			log_out << "[recompute_anchors] Failed to load: " << annlay_path << "\n";
			return false;
		}
		lay.ComputeAnchorsFromAnnprj(annprj_path);
		if(!lay.Save(annlay_path)) {
			log_out << "[recompute_anchors] Failed to save: " << annlay_path << "\n";
			return false;
		}
		log_out << "[recompute_anchors] OK — anchors recomputed in " << GetFileName(annlay_path) << "\n";
		return true;
	}

	bool RunRecognize(const Node::NodeDoc& nd, String& log_out) {
		String annlay_path  = SlotVal(nd, "annlay_in");
		String annprj_path  = SlotVal(nd, "annprj_in");
		String images_dir   = SlotVal(nd, "images_dir");
		String script_path  = SlotVal(nd, "recognition_script");
		String offset_mode  = SlotVal(nd, "offset_mode");
		String bool_policy  = SlotVal(nd, "bool_gate_policy");

		if(annlay_path.IsEmpty()) { log_out << "[recognize] annlay_in slot empty.\n"; return false; }
		if(annprj_path.IsEmpty()) { log_out << "[recognize] annprj_in slot empty.\n"; return false; }
		if(script_path.IsEmpty()) { log_out << "[recognize] recognition_script slot empty.\n"; return false; }
		if(!FileExists(script_path)) { log_out << "[recognize] script not found: " << script_path << "\n"; return false; }

		AnchoredSlotRecognizer rec;
		rec.SetOffsetMode(StringToOffsetMode(offset_mode));
		rec.SetBoolGatePolicy(StringToBoolGatePolicy(bool_policy));
		if(!rec.Load(annlay_path)) {
			log_out << "[recognize] Failed to load annlay: " << annlay_path << "\n";
			return false;
		}

		RecognitionScript script;
		if(!script.Load(script_path)) {
			log_out << "[recognize] Failed to load script: " << script_path << "\n";
			return false;
		}

		String prj_json = LoadFile(annprj_path);
		if(prj_json.IsEmpty()) { log_out << "[recognize] Failed to load annprj.\n"; return false; }
		Value root = ParseJSON(prj_json);
		if(IsNull(root) || !IsValueMap(root)) { log_out << "[recognize] Invalid annprj JSON.\n"; return false; }

		Value datasets = root["datasets"];
		if(!IsValueArray(datasets)) { log_out << "[recognize] No datasets in annprj.\n"; return false; }

		int total = 0;
		for(int di = 0; di < datasets.GetCount(); di++) {
			Value imgs = datasets[di]["images"];
			if(IsValueArray(imgs)) total += imgs.GetCount();
		}

		int done = 0, script_errors = 0;
		ValueArray new_datasets;

		for(int di = 0; di < datasets.GetCount(); di++) {
			Value ds = datasets[di];
			Value images = ds["images"];
			ValueArray new_images;
			int img_count = IsValueArray(images) ? images.GetCount() : 0;

			for(int ii = 0; ii < img_count; ii++) {
				Value img_rec = images[ii];
				String filename = img_rec["file_name"];
				if(filename.IsEmpty()) filename = img_rec["filename"];

				String img_path;
				if(IsFullPath(filename))
					img_path = filename;
				else if(!images_dir.IsEmpty())
					img_path = AppendFileName(images_dir, filename);
				else
					img_path = AppendFileName(GetFileDirectory(annprj_path), filename);

				ValueMap new_rec;
				if(IsValueMap(img_rec)) {
					ValueMap orig = img_rec;
					for(int fi = 0; fi < orig.GetCount(); fi++)
						new_rec.Add(orig.GetKey(fi), orig[fi]);
				}

				VectorMap<String, String> image_metadata;
				Value meta_keys = img_rec["image_meta_keys"];
				Value meta_vals = img_rec["image_meta_values"];
				if(IsValueArray(meta_keys) && IsValueArray(meta_vals)) {
					for(int mi = 0; mi < min(meta_keys.GetCount(), meta_vals.GetCount()); mi++) {
						String key = meta_keys[mi].ToString();
						if(!key.IsEmpty())
							image_metadata.GetAdd(key) = meta_vals[mi].ToString();
					}
				}

				if(FileExists(img_path)) {
					Image img = StreamRaster::LoadFileAny(img_path);
					if(!img.IsEmpty()) {
						Vector<SlotResult> results = rec.Recognize(img);
						ValueArray rec_arr;
						for(int ri = 0; ri < results.GetCount(); ri++) {
							ValueMap sr;
							sr.Add("slot_id",     results[ri].slot_id);
							sr.Add("method",      results[ri].method);
							sr.Add("raw_text",    results[ri].raw_text);
							sr.Add("top_class",   results[ri].top_class);
							sr.Add("class_index", results[ri].class_index);
							sr.Add("confidence",  results[ri].confidence);
							rec_arr.Add(sr);
						}
						new_rec.GetAdd("recognition") = rec_arr;

						VectorMap<String, String> script_meta = script.Run(results);
						if(!script.GetLastError().IsEmpty())
							script_errors++;
						else {
							for(int sm = 0; sm < script_meta.GetCount(); sm++)
								image_metadata.GetAdd(script_meta.GetKey(sm)) = script_meta[sm];
						}
					}
				}

				ValueArray out_keys, out_vals;
				for(int mi = 0; mi < image_metadata.GetCount(); mi++) {
					out_keys.Add(image_metadata.GetKey(mi));
					out_vals.Add(image_metadata[mi]);
				}
				new_rec.GetAdd("image_meta_keys")   = out_keys;
				new_rec.GetAdd("image_meta_values") = out_vals;
				new_images.Add(new_rec);
				done++;
			}

			ValueMap new_ds;
			if(IsValueMap(ds)) {
				ValueMap orig = ds;
				for(int fi = 0; fi < orig.GetCount(); fi++)
					new_ds.Add(orig.GetKey(fi), orig[fi]);
			}
			new_ds.GetAdd("images") = new_images;
			new_datasets.Add(new_ds);
		}

		ValueMap new_root;
		if(IsValueMap(root)) {
			ValueMap orig = root;
			for(int fi = 0; fi < orig.GetCount(); fi++)
				new_root.Add(orig.GetKey(fi), orig[fi]);
		}
		new_root.GetAdd("datasets") = new_datasets;

		if(!SaveFile(annprj_path, StoreAsJson(new_root, true))) {
			log_out << "[recognize] Failed to save annprj.\n";
			return false;
		}
		log_out << "[recognize] OK — " << done << "/" << total << " images";
		if(script_errors > 0) log_out << " (" << script_errors << " script errors)";
		log_out << "\n";
		return true;
	}

	bool RunCliStage(const String& stage_label, const Vector<String>& args, String& log_out) {
		String exe = GetExeFilePath();
		LocalProcess proc;
		if(!proc.Start(exe, args)) {
			log_out << "[" << stage_label << "] Failed to start subprocess.\n";
			return false;
		}
		String out;
		while(proc.IsRunning()) {
			out << proc.Get();
			Sleep(10);
		}
		out << proc.Get();
		int code = proc.GetExitCode();
		log_out << "[" << stage_label << "] exit=" << code << "\n";
		if(!out.IsEmpty()) log_out << out << "\n";
		return code == 0;
	}

public:
	virtual String GetDomainId()   const override { return "annotation"; }
	virtual String GetDomainName() const override { return "Annotation"; }
	virtual String GetDomainDesc() const override { return "AnnLay pipeline domain — recompute/export/train/recognize/QA"; }

	virtual String GetGraphFileFilter()    const override { return "Annotation Graph (*.anngrf)\t*.anngrf"; }
	virtual String GetProjectFileFilter()  const override { return "Annotation Project (*.annprj_graph)\t*.annprj_graph"; }
	virtual String GetSolutionFileFilter() const override { return "Annotation Solution (*.anngrf_sln)\t*.anngrf_sln"; }
	virtual String GetExtraExtensions()    const override { return ".anngrf|.annprj_graph|.anngrf_sln"; }

	virtual void OnDomainInit(NodeWorkbenchWindow& host) override {
		EnsurePalette();
		for(const TemplateDef& t : templates) {
			host.GetViewport().RegisterNodeType(
				t.type_id, t.label,
				[doc = t.doc]() mutable {
					Node::NodeDoc out;
					out <<= doc;
					return out;
				});
		}
	}

	virtual void BuildPalette(Vector<PaletteItem>& palette_out) override {
		EnsurePalette();
		palette_out.Clear();
		for(const TemplateDef& t : templates) {
			PaletteItem& p = palette_out.Add();
			p.category = t.category;
			p.label    = t.label;
			p.type_id  = t.type_id;
		}
	}

	virtual void ValidateGraph(NodeWorkbenchWindow& host,
	                           Vector<WorkbenchDiagnostic>& diag_out) override {
		diag_out.Clear();
		const Node::GraphDoc& doc = host.GetGraph().GetDoc();

		for(const Node::NodeDoc& n : doc.nodes) {
			// Check required file-path slots are non-empty
			auto CheckSlot = [&](const String& slot_id) {
				String v = SlotVal(n, slot_id);
				if(v.IsEmpty()) {
					WorkbenchDiagnostic& d = diag_out.Add();
					d.severity  = DiagSeverity::Warning;
					d.message   = "Node '" + n.id + "' (" + n.label + "): slot '" + slot_id + "' is empty.";
					d.source    = "annotation.validate";
					d.entity_id = n.id;
				}
				else if(!FileExists(v) && !DirectoryExists(v)) {
					WorkbenchDiagnostic& d = diag_out.Add();
					d.severity  = DiagSeverity::Warning;
					d.message   = "Node '" + n.id + "': path not found: " + v;
					d.source    = "annotation.validate";
					d.entity_id = n.id;
				}
			};

			const String& tid = n.node_type_id;
			if(tid == "ann.load_sln")
				CheckSlot("sln_file");
			else if(tid == "ann.recognize") {
				CheckSlot("recognition_script");
			}
			else if(tid == "ann.check_baseline") {
				CheckSlot("baseline_json");
				CheckSlot("offset_baseline_json");
			}
			else if(tid == "ann.preflight") {
				CheckSlot("bool_baseline");
				CheckSlot("offset_baseline");
			}
		}
	}

	virtual bool CompileGraph(NodeWorkbenchWindow& host, String& log_out) override {
		// Validate — treat as compile pass
		Vector<WorkbenchDiagnostic> diags;
		ValidateGraph(host, diags);
		bool ok = true;
		for(const WorkbenchDiagnostic& d : diags) {
			if(d.severity == DiagSeverity::Error) ok = false;
			log_out << (d.severity == DiagSeverity::Error ? "ERROR: " : "WARN:  ") << d.message << "\n";
		}
		if(ok && diags.IsEmpty()) log_out << "Compile OK — no issues.\n";
		return ok;
	}

	virtual bool RunGraph(NodeWorkbenchWindow& host, String& log_out) override {
		// Topological execution: walk nodes in document order,
		// drive headless-safe stages by type_id.
		// GUI-launch nodes are skipped (log a note).
		log_out.Clear();

		const Node::GraphDoc& doc = host.GetGraph().GetDoc();

		// First compile-check
		Vector<WorkbenchDiagnostic> diags;
		ValidateGraph(host, diags);
		for(const WorkbenchDiagnostic& d : diags) {
			if(d.severity == DiagSeverity::Error) {
				log_out << "Run aborted — validation errors:\n";
				for(const WorkbenchDiagnostic& d2 : diags)
					log_out << (d2.severity == DiagSeverity::Error ? "ERROR: " : "WARN:  ") << d2.message << "\n";
				return false;
			}
		}

		bool all_ok = true;
		for(const Node::NodeDoc& n : doc.nodes) {
			const String& tid = n.node_type_id;

			if(tid == "ann.load_sln") {
				// Not an executable stage — just a configuration node; skip
				log_out << "[load_sln] (configuration node — skipped in run)\n";
			}
			else if(tid == "ann.recompute_anchors") {
				if(!RunRecomputeAnchors(n, log_out)) all_ok = false;
			}
			else if(tid == "ann.export_crops") {
				// Delegate via CLI subprocess
				String annlay = SlotVal(n, "annlay_in");
				String crops  = SlotVal(n, "crops_root");
				String epochs_str = "0"; // export doesn't need epochs
				int pass = max(1, min(2, (int)StrInt(SlotVal(n, "pass_index"))));
				Vector<String> args;
				args.Add("--export-crops");
				args.Add(annlay);
				args.Add(AppendFileName(crops, String("pass") + IntStr(pass)));
				args.Add("--pass-index");
				args.Add(IntStr(pass));
				if(!RunCliStage("export_crops", args, log_out)) all_ok = false;
			}
			else if(tid == "ann.train_slots_cli") {
				String annlay = SlotVal(n, "annlay_in");
				String crops  = SlotVal(n, "crops_dir");
				String epochs = SlotVal(n, "epochs");
				if(epochs.IsEmpty()) epochs = "50";
				Vector<String> args;
				args.Add("--train-slots-from-crops");
				args.Add(annlay);
				args.Add(crops);
				args.Add("--epochs");
				args.Add(epochs);
				if(!RunCliStage("train_slots_cli", args, log_out)) all_ok = false;
			}
			else if(tid == "ann.recognize") {
				if(!RunRecognize(n, log_out)) all_ok = false;
			}
			else if(tid == "ann.audit_bool") {
				String annlay  = SlotVal(n, "annlay_in");
				String out_dir = SlotVal(n, "out_dir");
				RealizeDirectory(out_dir);
				Vector<String> args;
				args.Add("--audit-bool-gates");
				args.Add(annlay);
				args.Add("--strict");
				args.Add("--out");
				args.Add(AppendFileName(out_dir, "audit_report.json"));
				if(!RunCliStage("audit_bool", args, log_out)) all_ok = false;
			}
			else if(tid == "ann.fix_bool") {
				String annlay = SlotVal(n, "annlay_in");
				Vector<String> args;
				args.Add("--fix-bool-gates");
				args.Add(annlay);
				if(!RunCliStage("fix_bool", args, log_out)) all_ok = false;
			}
			else if(tid == "ann.eval_bool") {
				String sln     = SlotVal(n, "annsln_in");
				String split   = SlotVal(n, "split_json");
				String policy  = SlotVal(n, "bool_gate_policy");
				String mode    = SlotVal(n, "offset_mode");
				Vector<String> args;
				args.Add("--eval-bool-gates");
				args.Add(sln);
				args.Add("--all");
				if(!split.IsEmpty()) { args.Add("--split-file"); args.Add(split); }
				args.Add("--bool-gate-policy"); args.Add(policy.IsEmpty() ? "strict" : policy);
				args.Add("--offset-mode");      args.Add(mode.IsEmpty()   ? "auto"   : mode);
				if(!RunCliStage("eval_bool", args, log_out)) all_ok = false;
			}
			else if(tid == "ann.check_baseline") {
				String sln      = SlotVal(n, "annsln_in");
				String baseline = SlotVal(n, "baseline_json");
				String off_base = SlotVal(n, "offset_baseline_json");
				String split    = SlotVal(n, "split_json");
				String out_dir  = SlotVal(n, "out_dir");
				String policy   = SlotVal(n, "bool_gate_policy");
				String mode     = SlotVal(n, "offset_mode");
				RealizeDirectory(out_dir);
				Vector<String> args;
				args.Add("--check-bool-baseline");
				args.Add(sln);
				args.Add("--baseline");   args.Add(baseline);
				if(!split.IsEmpty()) { args.Add("--split-file"); args.Add(split); }
				args.Add("--check-out");  args.Add(AppendFileName(out_dir, "check_report.json"));
				args.Add("--bool-gate-policy"); args.Add(policy.IsEmpty() ? "strict" : policy);
				args.Add("--offset-mode");      args.Add(mode.IsEmpty()   ? "auto"   : mode);
				if(!RunCliStage("check_baseline", args, log_out)) all_ok = false;
			}
			else if(tid == "ann.preflight") {
				String sln       = SlotVal(n, "annsln_in");
				String bool_base = SlotVal(n, "bool_baseline");
				String off_base  = SlotVal(n, "offset_baseline");
				String split     = SlotVal(n, "split_json");
				String out_dir   = SlotVal(n, "out_dir");
				String policy    = SlotVal(n, "bool_gate_policy");
				String mode      = SlotVal(n, "offset_mode");
				RealizeDirectory(out_dir);
				// Preflight = eval + check_baseline chained
				{
					Vector<String> args;
					args.Add("--eval-bool-gates");
					args.Add(sln);
					args.Add("--all");
					if(!split.IsEmpty()) { args.Add("--split-file"); args.Add(split); }
					args.Add("--bool-gate-policy"); args.Add(policy.IsEmpty() ? "strict" : policy);
					args.Add("--offset-mode");      args.Add(mode.IsEmpty()   ? "auto"   : mode);
					RunCliStage("preflight/eval", args, log_out); // non-blocking — baseline check follows
				}
				{
					Vector<String> args;
					args.Add("--check-bool-baseline");
					args.Add(sln);
					args.Add("--baseline");   args.Add(bool_base);
					if(!split.IsEmpty()) { args.Add("--split-file"); args.Add(split); }
					args.Add("--check-out");  args.Add(AppendFileName(out_dir, "preflight_report.json"));
					args.Add("--bool-gate-policy"); args.Add(policy.IsEmpty() ? "strict" : policy);
					args.Add("--offset-mode");      args.Add(mode.IsEmpty()   ? "auto"   : mode);
					if(!RunCliStage("preflight/check", args, log_out)) all_ok = false;
				}
			}
			else if(tid == "ann.open_annotation_editor" || tid == "ann.open_frame_recognizer") {
				log_out << "[" << tid << "] GUI-launch node — skipped in headless run.\n";
			}
			// unknown nodes: silently skip (other domain nodes may be present)
		}

		log_out << (all_ok ? "RunGraph: ALL STAGES OK\n" : "RunGraph: SOME STAGES FAILED\n");
		return all_ok;
	}

	virtual void GetTemplates(Vector<TemplateDesc>& out) override {
		auto Desc = [&](const char* name, const char* cat, const char* desc) {
			TemplateDesc& d = out.Add();
			d.name = name; d.category = cat; d.description = desc;
		};
		Desc("Pass 1 Bootstrap Pipeline", "ann.templates",
		     "Full pass-1 pipeline: load → recompute → export → train → recognize");
		Desc("QA Baseline Check",         "ann.templates",
		     "Eval + check-baseline + preflight exit-gate chain");
	}

	virtual String GenerateTemplate(int index, const String& dest_dir, String& error_out) override {
		EnsurePalette();

		struct NodeEntry : Moveable<NodeEntry> {
			String id, type;
			Pointf pos;
		};
		struct EdgeEntry : Moveable<EdgeEntry> {
			String id, src_node, src_pin, tgt_node, tgt_pin;
		};
		struct TplDef {
			String name;
			Vector<NodeEntry> nodes;
			Vector<EdgeEntry> edges;
		};

		TplDef def;

		auto AN = [&](const char* id, const char* type, Pointf pos) -> NodeEntry& {
			NodeEntry& n = def.nodes.Add();
			n.id = id; n.type = type; n.pos = pos; return n;
		};
		auto AE = [&](const char* id,
		              const char* sn, const char* sp,
		              const char* tn, const char* tp) {
			EdgeEntry& e = def.edges.Add();
			e.id = id; e.src_node = sn; e.src_pin = sp;
			e.tgt_node = tn; e.tgt_pin = tp;
		};

		if(index == 0) {
			def.name = "Pass1 Bootstrap";
			AN("n_sln",     "ann.load_sln",           Pointf( 60, 200));
			AN("n_anchors", "ann.recompute_anchors",   Pointf(300, 100));
			AN("n_export",  "ann.export_crops",        Pointf(300, 220));
			AN("n_train",   "ann.train_slots_cli",     Pointf(560, 220));
			AN("n_recog",   "ann.recognize",           Pointf(800, 220));
			AE("e1","n_sln","annlay_path",    "n_anchors","annlay_in");
			AE("e2","n_sln","annprj_path",    "n_anchors","annprj_in");
			AE("e3","n_sln","annlay_path",    "n_export","annlay_in");
			AE("e4","n_sln","annprj_path",    "n_export","annprj_in");
			AE("e5","n_export","crops_dir_out","n_train","crops_dir");
			AE("e6","n_sln","annlay_path",    "n_train","annlay_in");
			AE("e7","n_sln","annlay_path",    "n_recog","annlay_in");
			AE("e8","n_sln","annprj_path",    "n_recog","annprj_in");
		}
		else if(index == 1) {
			def.name = "QA Baseline Check";
			AN("n_sln",     "ann.load_sln",       Pointf( 60, 200));
			AN("n_eval",    "ann.eval_bool",       Pointf(300, 120));
			AN("n_check",   "ann.check_baseline",  Pointf(300, 280));
			AN("n_pre",     "ann.preflight",       Pointf(560, 200));
			AE("e1","n_sln","annsln_path","n_eval","annsln_in");
			AE("e2","n_sln","annsln_path","n_check","annsln_in");
			AE("e3","n_sln","annsln_path","n_pre","annsln_in");
		}
		else {
			error_out = "Unknown template index: " + IntStr(index);
			return String();
		}

		// Build graph
		Node::Graph g;
		Node::GraphDoc& doc = g.GetDoc();

		for(const NodeEntry& ne : def.nodes) {
			Node::NodeDoc& nd = doc.nodes.Add();
			nd.id = ne.id; nd.pos = ne.pos;
			for(const TemplateDef& td : templates) {
				if(td.type_id == ne.type) {
					nd <<= td.doc;
					nd.id  = ne.id;
					nd.pos = ne.pos;
					break;
				}
			}
		}
		for(const EdgeEntry& ee : def.edges) {
			Node::EdgeDoc& ed = doc.edges.Add();
			ed.id = ee.id; ed.source_node = ee.src_node; ed.source_pin = ee.src_pin;
			ed.target_node = ee.tgt_node; ed.target_pin = ee.tgt_pin; ed.directed = true;
		}
		g.RebuildIndexPublic();

		String tpl_dir = AppendFileName(dest_dir, def.name);
		if(!DirectoryExists(tpl_dir) && !RealizeDirectory(tpl_dir)) {
			error_out = "Cannot create: " + tpl_dir; return String();
		}

		String graph_path = AppendFileName(tpl_dir, "main.anngrf");
		String prj_path   = AppendFileName(tpl_dir, def.name + ".annprj_graph");
		String sln_path   = AppendFileName(tpl_dir, def.name + ".anngrf_sln");

		if(!SaveFile(graph_path, Node::SaveEon(g))) {
			error_out = "Failed to save graph: " + graph_path; return String();
		}

		WorkbenchProject prj_new;
		prj_new.name = def.name; prj_new.graphs.Add(graph_path);
		prj_new.startup_graph = graph_path;
		if(!prj_new.Save(prj_path)) { error_out = "Failed to save project."; return String(); }

		WorkbenchSolution sln_new;
		sln_new.name = def.name; sln_new.projects.Add(prj_path);
		sln_new.active_project = prj_path;
		if(!sln_new.Save(sln_path)) { error_out = "Failed to save solution."; return String(); }

		// Update manifest.json
		String manifest_path = AppendFileName(dest_dir, "manifest.json");
		Value  mv; if(FileExists(manifest_path)) mv = ParseJSON(LoadFile(manifest_path));
		ValueMap mm; if(IsValueMap(mv)) mm = mv; else mm.GetAdd("version") = 1;
		Value gv = mm.GetAdd("generated");
		ValueArray ga; if(IsValueArray(gv)) ga = gv;
		ValueArray gn;
		for(int i = 0; i < ga.GetCount(); i++) {
			Value e = ga[i];
			if(IsValueMap(e) && e["name"].ToString() != def.name) gn.Add(e);
		}
		ValueMap entry;
		entry.GetAdd("name")     = def.name;
		entry.GetAdd("domain")   = GetDomainId();
		entry.GetAdd("solution") = sln_path;
		entry.GetAdd("project")  = prj_path;
		entry.GetAdd("graph")    = graph_path;
		gn.Add(entry);
		mm.GetAdd("generated") = gn;
		SaveFile(manifest_path, AsJSON(mm, true));

		return sln_path;
	}
};

static DomainRegistry::Entry s_annotation_domain_reg([] { return new AnnotationDomainImpl(); });

} // namespace

// ---------------------------------------------------------------------------
// AnnotationDomainWindow
// ---------------------------------------------------------------------------

AnnotationDomainWindow::AnnotationDomainWindow() {
	domain.Attach(DomainRegistry::CreateById("annotation"));
	if(!domain)
		domain.Attach(new AnnotationDomainImpl());
	RegisterDomain(*domain);
	Title("Annotation Pipeline Workbench");
}

AnnotationDomainWindow::~AnnotationDomainWindow() {
	domain.Clear();
}

END_UPP_NAMESPACE
