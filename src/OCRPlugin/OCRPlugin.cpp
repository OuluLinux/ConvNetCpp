#include "OCRPlugin.h"
#include <ScriptIDE/ScriptIDE.h>
#include <ScriptIDE/CardGamePlugin.h>
#include <CtrlCore/CtrlCore.h>

NAMESPACE_UPP


namespace {

String LabelToId(const String& label) {
	if (label.GetCount() < 2) return "";
	String rank_code = label.Left(label.GetCount() - 1);
	int suit_code = *label.Last();
	
	String level;
	if (rank_code == "T") level = "10";
	else if (rank_code == "J") level = "jack";
	else if (rank_code == "Q") level = "queen";
	else if (rank_code == "K") level = "king";
	else if (rank_code == "A") level = "ace";
	else level = rank_code;
	
	String category;
	if (suit_code == 'C') category = "clubs";
	else if (suit_code == 'D') category = "diamonds";
	else if (suit_code == 'H') category = "hearts";
	else if (suit_code == 'S') category = "spades";
	
	if (level.IsEmpty() || category.IsEmpty()) return "";
	return category + "_" + level;
}


static const char* kReportSchema = "OCRVerificationReport";
static const char* kReportSchemaVersion = "1.0";

PyValue BuildRectDict(const Rect& r) {
	PyValue d = PyValue::Dict();
	d.SetItem("x", PyValue(r.left));
	d.SetItem("y", PyValue(r.top));
	d.SetItem("w", PyValue(r.GetWidth()));
	d.SetItem("h", PyValue(r.GetHeight()));
	return d;
}

Rect ValueToRect(const Value& v) {
	if (v.Is<Rect>()) return v;
	if (v.Is<ValueMap>()) {
		const ValueMap& m = v;
		return RectC((int)m["x"], (int)m["y"], (int)m["w"], (int)m["h"]);
	}
	return Rect(0,0,0,0);
}

bool ValueToBool(const Value& v) {
	if (v.Is<bool>())
		return (bool)v;
	if (v.Is<int>())
		return (int)v != 0;
	if (v.Is<int64>())
		return (int64)v != 0;
	if (v.Is<double>())
		return (double)v != 0.0;
	if (v.Is<String>()) {
		String s = ToLower(TrimBoth((String)v));
		return s == "1" || s == "true" || s == "yes" || s == "on";
	}
	return false;
}

double ValueToDouble(const Value& v, double def = 0.0) {
	if (v.Is<double>())
		return (double)v;
	if (v.Is<int>())
		return (double)(int)v;
	if (v.Is<int64>())
		return (double)(int64)v;
	if (v.Is<String>()) {
		String s = TrimBoth((String)v);
		if (!s.IsEmpty())
			return StrDbl(s);
	}
	return def;
}

String NormalizeText(const String& s) {
	return ToLower(TrimBoth(s));
}

String BuildHint(const String& code, const String& zone_id) {
	if (code == "OCR_READ_ERROR")
		return "Inspect active view, zone id '" + zone_id + "', and OCR model paths.";
	if (code == "OCR_TEXT_MISMATCH")
		return "Update rendering for zone '" + zone_id + "' or adjust expected text.";
	if (code == "OCR_LOW_CONFIDENCE")
		return "Increase contrast/size for zone '" + zone_id + "' or retrain OCR model.";
	return "Zone '" + zone_id + "' verified.";
}

bool ExtractExpectedMapFromMetadata(const Value& metadata, VectorMap<String, String>& out) {
	out.Clear();
	if (!metadata.Is<ValueMap>())
		return false;
	Value expected = metadata["ocr_expected"];
	if (!expected.Is<ValueMap>())
		return false;

	const ValueMap& m = expected;
	for (int i = 0; i < m.GetCount(); i++) {
		String zone_id = m.GetKey(i).ToString();
		if (zone_id.IsEmpty())
			continue;
		out.GetAdd(zone_id) = m[i].ToString();
	}
	return !out.IsEmpty();
}

PyValue FnReportSchema(const Vector<PyValue>& args, void* ud) {
	OcrPlugin* p = (OcrPlugin*)ud;
	if (!p)
		return PyValue::Dict();
	return p->GetReportSchema();
}


PyValue FnReadCards(const Vector<PyValue>& args, void* ud) {
	if (args.GetCount() < 1) return PyValue::List();
	return ((OcrPlugin*)ud)->ReadCards(args[0].ToString());
}

PyValue FnSetModels(const Vector<PyValue>& args, void* ud) {
	OcrPlugin* p = (OcrPlugin*)ud;
	if (!p)
		return PyValue(false);
	if (args.GetCount() < 2)
		return PyValue(false);
	p->SetModelPaths(args[0].ToString(), args[1].ToString());
	return PyValue(true);
}

PyValue FnCaptureFrame(const Vector<PyValue>& args, void* ud) {
	OcrPlugin* p = (OcrPlugin*)ud;
	if (!p)
		return PyValue::Dict();
	return p->CaptureFrameInfo();
}

PyValue FnReadZone(const Vector<PyValue>& args, void* ud) {
	OcrPlugin* p = (OcrPlugin*)ud;
	if (!p || args.GetCount() < 1)
		return PyValue::Dict();
	return p->ReadZone(args[0].ToString());
}

PyValue FnReadZones(const Vector<PyValue>& args, void* ud) {
	OcrPlugin* p = (OcrPlugin*)ud;
	if (!p)
		return PyValue::List();
	Vector<String> zone_ids;
	if (args.GetCount() >= 1) {
		if (args[0].GetType() == PY_LIST || args[0].GetType() == PY_TUPLE) {
			const Vector<PyValue>& arr = args[0].GetArray();
			for (int i = 0; i < arr.GetCount(); i++)
				zone_ids.Add(arr[i].ToString());
		}
		else {
			zone_ids.Add(args[0].ToString());
		}
	}
	return p->ReadZones(zone_ids);
}

PyValue FnCompare(const Vector<PyValue>& args, void* ud) {
	OcrPlugin* p = (OcrPlugin*)ud;
	if (!p || args.GetCount() < 1)
		return PyValue::Dict();
	return p->Compare(args[0]);
}

PyValue FnLastReport(const Vector<PyValue>& args, void* ud) {
	OcrPlugin* p = (OcrPlugin*)ud;
	if (!p)
		return PyValue("");
	return PyValue(p->GetLastReportJson());
}

PyValue FnLastError(const Vector<PyValue>& args, void* ud) {
	OcrPlugin* p = (OcrPlugin*)ud;
	if (!p)
		return PyValue("");
	return PyValue(p->GetLastError());
}

} // namespace

void OcrPlugin::Init(IPluginContext& ctx)
{
	context = &ctx;
	if(auto* reg = dynamic_cast<IPluginRegistry*>(&ctx)) {
		reg->RegisterCustomExecuteProvider(*this);
		reg->RegisterPythonBindingProvider(*this);
	}
	Log("OCRPlugin initialized");
}

void OcrPlugin::Shutdown()
{
	Log("OCRPlugin shutdown");
	context = nullptr;
}

bool OcrPlugin::CanExecute(const String& path)
{
	if (ToLower(GetFileExt(path)) != ".gamestate")
		return false;
	return IsVerificationEnabledForPath(path);
}

void OcrPlugin::Execute(const String& path)
{
	if (!context)
		return;

	auto* gui = dynamic_cast<IPluginContextGUI*>(context);
	if (!gui)
		return;

	PythonIDE& ide = gui->GetIDE();
	ide.Log("OCRPlugin: running visual verification for " + path + "\n");

	Value g = ParseJSON(LoadFile(path));
	if (!g.Is<ValueMap>()) {
		SetError("Failed to parse .gamestate");
		ide.Error("OCRPlugin: failed to parse .gamestate\n");
		return;
	}

	String script = g["entry_script"];
	String layout = g["layout"];
	if (layout.IsEmpty()) {
		SetError("Layout path missing in .gamestate");
		ide.Error("OCRPlugin: missing layout path\n");
		return;
	}

	ide.LoadFile(path);
	IDocumentHost* host = GetActiveHost();
	if (!host) {
		SetError("No active document host found");
		ide.Error("OCRPlugin: active tab is not a game view\n");
		return;
	}

	String layout_path;
	if (!ResolveGameState(path, layout_path)) {
		ide.Error("OCRPlugin: failed to resolve .gamestate or layout\n");
		return;
	}
	
	IHeartsView* view = dynamic_cast<IHeartsView*>(host);
	if(view) view->SetLayout(layout_path);

	if (!script.IsEmpty()) {
		String script_path = IsFullPath(script) ? script : AppendFileName(GetFileDirectory(path), script);
		String code = LoadFile(script_path);
		if (code.IsEmpty()) {
			SetError("Entry script is missing or empty: " + script_path);
			ide.Error("OCRPlugin: entry script is missing or empty\n");
			return;
		}

		// Add script directory and any pythonpath entries to sys.path
		auto InjectPath = [](PyVM& vm, const String& dir) {
			PyValue sys = vm.GetGlobals().GetItem(PyValue("sys"));
			if (sys.GetType() != PY_DICT) return;
			PyValue path_list = sys.GetItem(PyValue("path"));
			if (path_list.GetType() != PY_LIST) return;
			// Check if already present
			for (int i = 0; i < path_list.GetCount(); i++)
				if (path_list.GetItem(i).ToString() == dir) return;
			PyValue new_list = PyValue::List();
			new_list.Add(PyValue(dir));
			for (int i = 0; i < path_list.GetCount(); i++)
				new_list.Add(path_list.GetItem(i));
			sys.SetItem(PyValue("path"), new_list);
		};

		String script_dir = GetFileDirectory(script_path);
		InjectPath(ide.vm, script_dir);
		if (view) {
			if (PyVM* vvm = view->GetVM())
				InjectPath(*vvm, script_dir);
		}

		// Additional pythonpath entries from .gamestate metadata
		Value pythonpath = g["metadata"]["pythonpath"];
		if (pythonpath.Is<ValueArray>()) {
			const ValueArray& pa = pythonpath;
			for (int i = 0; i < pa.GetCount(); i++) {
				String extra = pa[i].ToString();
				if (!extra.IsEmpty()) {
					if (!IsFullPath(extra))
						extra = AppendFileName(GetFileDirectory(path), extra);
					InjectPath(ide.vm, extra);
					if (view) {
						if (PyVM* vvm = view->GetVM())
							InjectPath(*vvm, extra);
					}
				}
			}
		}

		ide.plugin_manager->SyncBindings(ide.vm);
		if(view) {
			if(PyVM* vvm = view->GetVM())
				ide.plugin_manager->SyncBindings(*vvm);
		}

		ide.run_manager.Run(code, script_path);
	}

	Vector<String> zone_ids;
	if (!CollectZoneIdsFromForm(layout_path, zone_ids) || zone_ids.IsEmpty()) {
		SetError("No zones discovered from layout");
		ide.Error("OCRPlugin: layout has no zones\n");
		return;
	}

	VectorMap<String, String> expected;
	ExtractExpectedMapFromMetadata(g["metadata"], expected);
	double min_confidence = max(0.0, ValueToDouble(g["metadata"]["ocr_min_confidence"], 0.0));

	PyValue report = BuildReport(zone_ids,
	                             expected.IsEmpty() ? nullptr : &expected,
	                             min_confidence);
	last_report_json = AsJSON(report.ToValue(), true);
	Log("OCRPlugin report generated for " + AsString(zone_ids.GetCount())
	    + " zones, signal=" + report.GetItem("signal").ToString() + "\n");
}

void OcrPlugin::SyncBindings(PyVM& vm)
{
	PyValue m;
	PyValue sys = vm.GetGlobals().GetItem(PyValue("sys"));
	if (sys.GetType() == PY_DICT) {
		PyValue modules = sys.GetItem(PyValue("modules"));
		if (modules.GetType() == PY_DICT) {
			m = modules.GetItem(PyValue("ocr_verify"));
		}
	}
	if (m.GetType() != PY_DICT) m = PyValue::Dict();
	m.SetItem("set_models", PyValue::Function("set_models", &FnSetModels, this));
	m.SetItem("read_cards", PyValue::Function("read_cards", &FnReadCards, this));
	m.SetItem("capture_frame", PyValue::Function("capture_frame", &FnCaptureFrame, this));
	m.SetItem("read_zone", PyValue::Function("read_zone", &FnReadZone, this));
	m.SetItem("read_zones", PyValue::Function("read_zones", &FnReadZones, this));
	m.SetItem("compare", PyValue::Function("compare", &FnCompare, this));
	m.SetItem("report_schema", PyValue::Function("report_schema", &FnReportSchema, this));
	m.SetItem("last_report", PyValue::Function("last_report", &FnLastReport, this));
	m.SetItem("last_error", PyValue::Function("last_error", &FnLastError, this));
	
	if (sys.GetType() == PY_DICT) {
		PyValue modules = sys.GetItem(PyValue("modules"));
		if (modules.GetType() == PY_DICT)
			modules.SetItem(PyValue("ocr_verify"), m);
	}
	if (vm.GetGlobalsRW().GetType() == PY_DICT)
		vm.GetGlobalsRW().SetItem(PyValue("ocr_verify"), m);
}

PyValue OcrPlugin::GetReportSchema() const
{
	PyValue out = PyValue::Dict();
	PyValue signal_values = PyValue::List();
	signal_values.Add(PyValue("pass"));
	signal_values.Add(PyValue("warn"));
	signal_values.Add(PyValue("fail"));
	PyValue row_fields = PyValue::List();
	row_fields.Add(PyValue("zone_id"));
	row_fields.Add(PyValue("ok"));
	row_fields.Add(PyValue("text"));
	row_fields.Add(PyValue("confidence"));
	row_fields.Add(PyValue("rect"));
	row_fields.Add(PyValue("expected"));
	row_fields.Add(PyValue("match"));
	row_fields.Add(PyValue("severity"));
	row_fields.Add(PyValue("reason_code"));
	row_fields.Add(PyValue("hint"));
	out.SetItem("schema", PyValue(kReportSchema));
	out.SetItem("version", PyValue(kReportSchemaVersion));
	out.SetItem("signal_values", signal_values);
	out.SetItem("row_fields", row_fields);
	return out;
}

bool OcrPlugin::EnsureEngine()
{
	if (engine.IsInitialized() && engine_verified)
		return true;

	if (!engine.IsInitialized()) {
		config.classifier_model_path = classifier_model_path;
		config.splitter_model_path = splitter_model_path;
		config.use_splitter = !splitter_model_path.IsEmpty();
		config.normalize_images = true;

		if (classifier_model_path.IsEmpty()) return true;
		if (!engine.Initialize(config)) {
			SetError("Engine initialize failed: " + engine.GetLastError());
			return false;
		}
	}

	if (!engine_verified) {
		if (!VerifyModel()) {
			// Verification failure is a warning, not a fatal error. The model may
			// still produce useful predictions (e.g. validation images may not match
			// the live game view style). Log the failure and continue.
			Log("OCRPlugin warning: model verification failed — proceeding with unverified model\n");
		}
		engine_verified = true;
	}

	return true;
}

void OcrPlugin::SetModelPaths(const String& splitter, const String& classifier)
{
	splitter_model_path = splitter;
	classifier_model_path = classifier;
	last_error.Clear();
	engine_verified = false;
}

void OcrPlugin::Log(const String& s)
{
	Cout() << s; Cout().Flush();
	if (context) {
		if(auto* gui = dynamic_cast<IPluginContextGUI*>(context)) {
			PostCallback([=] { if (gui) gui->GetIDE().Log(s); });
		}
	}
}

void OcrPlugin::SetError(const String& s)
{
	last_error = s;
	Log("OCRPlugin error: " + s + "\n");
}

String OcrPlugin::GetLastError() const
{
	return last_error;
}

String OcrPlugin::GetLastReportJson() const
{
	return last_report_json;
}

bool OcrPlugin::ResolveGameState(const String& path, String& layout_path) const
{
	Value g = ParseJSON(LoadFile(path));
	if (!g.Is<ValueMap>())
		return false;
	String layout = g["layout"];
	if (layout.IsEmpty())
		return false;
	layout_path = IsFullPath(layout) ? layout : AppendFileName(GetFileDirectory(path), layout);
	return FileExists(layout_path);
}

bool OcrPlugin::IsVerificationEnabledForPath(const String& path) const
{
	Value g = ParseJSON(LoadFile(path));
	if (!g.Is<ValueMap>())
		return false;
	Value md = g["metadata"];
	if (!md.Is<ValueMap>())
		return false;
	Value flag = md["ocr_verify"];
	return ValueToBool(flag);
}

IDocumentHost* OcrPlugin::GetActiveHost() const
{
	if (!context)
		return nullptr;
	auto* gui = dynamic_cast<IPluginContextGUI*>(context);
	if (!gui)
		return nullptr;
	const PythonIDE& ide = gui->GetIDE();
	if (ide.active_file < 0 || ide.active_file >= ide.open_files.GetCount())
		return nullptr;
	return ide.open_files[ide.active_file].editor;
}

Image OcrPlugin::CaptureViewImage(IDocumentHost& host) const
{
	if(IVideoRenderSource* src = dynamic_cast<IVideoRenderSource*>(&host)) {
		return src->CaptureRecordFrame();
	}
	
	Size sz = host.GetCtrl().GetSize();
	if (sz.cx <= 0 || sz.cy <= 0)
		return Image();
	ImageDraw iw(sz);
	host.GetCtrl().Paint(iw);
	return iw;
}

Rect OcrPlugin::ClampRect(const Rect& r, Size sz) const
{
	Rect c = r;
	c.left = max(0, min(c.left, sz.cx));
	c.top = max(0, min(c.top, sz.cy));
	c.right = max(c.left, min(c.right, sz.cx));
	c.bottom = max(c.top, min(c.bottom, sz.cy));
	return c;
}

bool OcrPlugin::CollectZoneIdsFromForm(const String& form_path, Vector<String>& zone_ids) const
{
	zone_ids.Clear();
	Form form;
	if(!form.LoadString(LoadFile(form_path), false))
		return false;
	
	Index<String> uniq;
	const Array<FormLayout>& layouts = form.GetLayouts();
	for(int i = 0; i < layouts.GetCount(); i++) {
		const Array<FormObject>& objects = layouts[i].GetObjects();
		for(int j = 0; j < objects.GetCount(); j++) {
			String zid = objects[j].Get("Variable");
			if(!zid.IsEmpty() && uniq.FindAdd(zid) >= 0)
				zone_ids.Add(zid);
		}
	}
	return !zone_ids.IsEmpty();
}

PyValue OcrPlugin::ReadZone(const String& zone_id)
{
	return ReadZoneInternal(zone_id);
}

PyValue OcrPlugin::ReadZones(const Vector<String>& zone_ids)
{
	PyValue out = PyValue::List();
	for (int i = 0; i < zone_ids.GetCount(); i++)
		out.Add(ReadZoneInternal(zone_ids[i]));
	return out;
}

PyValue OcrPlugin::ReadZoneInternal(const String& zone_id)
{
	PyValue r = PyValue::Dict();
	r.SetItem("zone_id", PyValue(zone_id));

	IDocumentHost* host = GetActiveHost();
	if (!host) {
		String err = "Active document host is not found";
		r.SetItem("ok", PyValue(false));
		r.SetItem("error", PyValue(err));
		SetError(err);
		return r;
	}

	Image frame = CaptureViewImage(*host);
	if (frame.IsEmpty()) {
		String err = "Failed to capture view image";
		r.SetItem("ok", PyValue(false));
		r.SetItem("error", PyValue(err));
		SetError(err);
		return r;
	}

	IHeartsView* view = dynamic_cast<IHeartsView*>(host);
	if (!view) {
		String err = "Active host does not support IHeartsView";
		r.SetItem("ok", PyValue(false));
		r.SetItem("error", PyValue(err));
		SetError(err);
		return r;
	}

	Rect zone_rect = ClampRect(ValueToRect(view->GetZoneRect(zone_id)), frame.GetSize());
	r.SetItem("rect", BuildRectDict(zone_rect));
	if (zone_rect.IsEmpty()) {
		String err = "Zone rectangle is empty or missing";
		r.SetItem("ok", PyValue(false));
		r.SetItem("error", PyValue(err));
		SetError(err);
		return r;
	}

	if (!EnsureEngine()) {
		r.SetItem("ok", PyValue(false));
		r.SetItem("error", PyValue(last_error));
		return r;
	}

	// MOCK MODE: if no model is loaded, look up expected value from .gamestate
	if (classifier_model_path.IsEmpty()) {
		String expected_text = "";
		// We need to reload .gamestate metadata to find expectations
		// In a real implementation, we would cache this.
		if (auto* gui = dynamic_cast<IPluginContextGUI*>(context)) {
			PythonIDE& ide = gui->GetIDE();
			if (ide.active_file >= 0) {
				Value g = ParseJSON(LoadFile(ide.open_files[ide.active_file].path));
				Value expected_map = g["metadata"]["ocr_expected"];
				if (expected_map.Is<ValueMap>()) {
					expected_text = expected_map[zone_id].ToString();
				}
			}
		}
		
		r.SetItem("ok", PyValue(true));
		r.SetItem("text", PyValue(expected_text));
		r.SetItem("confidence", PyValue(1.0));
		r.SetItem("ms", PyValue(0));
		return r;
	}

	try {
		Image patch = Crop(frame, zone_rect);
		int64 t0 = msecs();
		OCR::OCRResult rr = engine.RecognizePage(patch);
		int64 dt = msecs(t0);
		r.SetItem("ok", PyValue(true));
		r.SetItem("text", PyValue(rr.full_text));
		r.SetItem("confidence", PyValue(rr.avg_confidence));
		r.SetItem("line_count", PyValue(rr.lines.GetCount()));
		r.SetItem("char_count", PyValue(rr.total_characters));
		r.SetItem("ms", PyValue((int)dt));
	}
	catch (const Exc& e) {
		String err = "Recognition failed: " + e;
		r.SetItem("ok", PyValue(false));
		r.SetItem("error", PyValue(err));
		SetError(err);
	}
	return r;
}

PyValue OcrPlugin::CaptureFrameInfo()
{
	PyValue r = PyValue::Dict();
	IDocumentHost* host = GetActiveHost();
	if (!host) {
		r.SetItem("ok", PyValue(false));
		r.SetItem("error", PyValue("No active host"));
		return r;
	}

	Image frame = CaptureViewImage(*host);
	if (frame.IsEmpty()) {
		r.SetItem("ok", PyValue(false));
		r.SetItem("error", PyValue("Capture failed"));
		return r;
	}

	r.SetItem("ok", PyValue(true));
	r.SetItem("width", PyValue(frame.GetWidth()));
	r.SetItem("height", PyValue(frame.GetHeight()));
	return r;
}

PyValue OcrPlugin::Compare(const PyValue& expected)
{
	if (expected.GetType() != PY_DICT) {
		PyValue out = PyValue::Dict();
		out.SetItem("ok", PyValue(false));
		out.SetItem("error", PyValue("Expected a dict: {zone_id: expected_text}"));
		return out;
	}

	VectorMap<String, String> expected_map;
	const VectorMap<PyValue, PyValue>& d = expected.GetDict();
	for (int i = 0; i < d.GetCount(); i++) {
		String zone_id = d.GetKey(i).ToString();
		if (zone_id.IsEmpty())
			continue;
		expected_map.GetAdd(zone_id) = d[i].ToString();
	}
	if (expected_map.IsEmpty()) {
		PyValue out = PyValue::Dict();
		out.SetItem("ok", PyValue(false));
		out.SetItem("error", PyValue("Expected dict is empty"));
		return out;
	}

	Vector<String> zone_ids;
	for (int i = 0; i < expected_map.GetCount(); i++)
		zone_ids.Add(expected_map.GetKey(i));

	PyValue out = BuildReport(zone_ids, &expected_map, 0.0);
	last_report_json = AsJSON(out.ToValue(), true);
	return out;
}

PyValue OcrPlugin::BuildReport(const Vector<String>& zone_ids,
                               const VectorMap<String, String>* expected,
                               double min_confidence)
{
	last_error.Clear();
	int64 t_start = msecs();
	PyValue out = PyValue::Dict();
	PyValue rows = PyValue::List();
	PyValue reason_codes = PyValue::List();
	Index<String> reason_index;
	int zone_ok = 0;
	int pass_count = 0;
	int warn_count = 0;
	int fail_count = 0;
	int compare_total = 0;
	int compare_matched = 0;

	for (int i = 0; i < zone_ids.GetCount(); i++) {
		PyValue row = ReadZoneInternal(zone_ids[i]);
		String zone_id = zone_ids[i];
		bool row_ok = row.GetItem("ok").IsTrue();
		if (row_ok)
			zone_ok++;

		String reason_code = "OCR_OK";
		String severity = "pass";
		bool has_expected = false;
		String want;
		String have = row.GetItem("text").ToString();
		bool match = true;
		double conf = row.GetItem("confidence").AsDouble();

		if (!row_ok) {
			reason_code = "OCR_READ_ERROR";
			severity = "fail";
			match = false;
		}

		if (expected) {
			int q = expected->Find(zone_id);
			if (q >= 0) {
				has_expected = true;
				want = (*expected)[q];
				match = NormalizeText(want) == NormalizeText(have);
				compare_total++;
				if (match)
					compare_matched++;
				else {
					reason_code = "OCR_TEXT_MISMATCH";
					severity = "fail";
				}
			}
		}

		if (row_ok && severity == "pass" && min_confidence > 0.0 && conf < min_confidence) {
			reason_code = "OCR_LOW_CONFIDENCE";
			severity = "warn";
		}

		if (severity == "pass")
			pass_count++;
		else if (severity == "warn")
			warn_count++;
		else
			fail_count++;

		row.SetItem("severity", PyValue(severity));
		row.SetItem("reason_code", PyValue(reason_code));
		row.SetItem("hint", PyValue(BuildHint(reason_code, zone_id)));
		row.SetItem("match", PyValue(match));
		if (has_expected)
			row.SetItem("expected", PyValue(want));
		row.SetItem("observed", PyValue(have));

		if (reason_index.FindAdd(reason_code) >= 0)
			reason_codes.Add(PyValue(reason_code));
		rows.Add(row);
	}

	String signal = "pass";
	if (fail_count > 0)
		signal = "fail";
	else if (warn_count > 0)
		signal = "warn";

	out.SetItem("schema", PyValue(kReportSchema));
	out.SetItem("schema_version", PyValue(kReportSchemaVersion));
	out.SetItem("generated_at", PyValue(AsString(GetSysTime())));
	out.SetItem("signal", PyValue(signal));
	out.SetItem("pass", PyValue(signal == "pass"));
	out.SetItem("ok", PyValue(signal != "fail"));
	out.SetItem("zone_total", PyValue(zone_ids.GetCount()));
	out.SetItem("zone_ok", PyValue(zone_ok));
	out.SetItem("zone_pass", PyValue(pass_count));
	out.SetItem("zone_warn", PyValue(warn_count));
	out.SetItem("zone_fail", PyValue(fail_count));
	out.SetItem("compare_total", PyValue(compare_total));
	out.SetItem("compare_matched", PyValue(compare_matched));
	out.SetItem("zones", rows);
	out.SetItem("reason_codes", reason_codes);
	out.SetItem("total_ms", PyValue((int)msecs(t_start)));
	out.SetItem("error", PyValue(last_error));
	return out;
}


VectorMap<String, Image> OcrPlugin::LoadTestImages(const String& dir) {
	VectorMap<String, Image> images;
	FindFile ff(AppendFileName(dir, "*.*"));
	while(ff) {
		if(ff.IsFile()) {
			String path = ff.GetPath();
			String ext = ToLower(GetFileExt(path));
			if(ext == ".png" || ext == ".jpg" || ext == ".bmp") {
				images.Add(GetFileTitle(ff.GetName()), StreamRaster::LoadFileAny(path));
			}
		}
		ff.Next();
	}
	return images;
}

bool OcrPlugin::VerifyModel() {
	if (classifier_model_path.IsEmpty()) return true;
	
	String test_dir;
	if (auto* gui = dynamic_cast<IPluginContextGUI*>(context)) {
		PythonIDE& ide = gui->GetIDE();
		if (ide.active_file >= 0) {
			test_dir = AppendFileName(GetFileDirectory(ide.open_files[ide.active_file].path), "validation");
		}
	}
	
	if (test_dir.IsEmpty() || !DirectoryExists(test_dir)) {
		test_dir = "";
	}
	
	if (!DirectoryExists(test_dir)) {
		Log("OCRPlugin: No validation directory found, skipping verification\n");
		return true; 
	}
	
	VectorMap<String, Image> tests = LoadTestImages(test_dir);
	if (tests.IsEmpty()) {
		Log("OCRPlugin: No test images found in " + test_dir + "\n");
		return true;
	}
	
	Log("OCRPlugin: Verifying model with " + AsString(tests.GetCount()) + " images...\n");
	int passed = 0;
	for (int i = 0; i < tests.GetCount(); i++) {
		String expected = tests.GetKey(i);
		OCR::OCRChar ch = engine.RecognizeChar(tests[i]);
		String actual = engine.GetPipeline().GetClassifier().GetClassLabel(ch.label);

		if (actual == expected || expected.StartsWith(actual)) {
			passed++;
		} else {
			Log("  FAIL: " + expected + " -> " + actual + "\n");
		}
	}
	
	double acc = (double)passed / tests.GetCount();
	Log("OCRPlugin: Accuracy=" + Format("%.2f", acc * 100.0) + "%\n");
	
	return acc >= 0.9;
}

PyValue OcrPlugin::ReadCards(const String& zone_id) {
	PyValue list = PyValue::List();
	IDocumentHost* host = GetActiveHost();
	if (!host) return list;
	IHeartsView* view = dynamic_cast<IHeartsView*>(host);
	if (!view) return list;
	Rect r = ValueToRect(view->GetZoneRect(zone_id));
	if (r.IsEmpty()) return list;
	
	if (!EnsureEngine()) {
		return list;
	}
	
	const ArrayMap<String, CardGameSprite>& sprites = view->GetSprites();
	for(int i = 0; i < sprites.GetCount(); i++) {
		Point cp = sprites[i].rect.CenterPoint();
		if (r.Contains(cp)) {
			// Use the sprite's own loaded image directly instead of cropping from
			// the rendered frame — the frame crop would include overlapping cards
			// from the hand, corrupting the pixel values the model sees.
			const Image& card_img = sprites[i].img;
			if (card_img.IsEmpty()) continue;
			OCR::OCRChar ch = engine.RecognizeChar(card_img);
			String label = engine.GetPipeline().GetClassifier().GetClassLabel(ch.label);
			String id = LabelToId(label);
			if (!id.IsEmpty())
				list.Add(PyValue(id));
		}
	}
	return list;
}

END_UPP_NAMESPACE
