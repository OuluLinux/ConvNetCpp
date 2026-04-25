#include <Core/Core.h>
#include <plugin/png/png.h>
#include <stdio.h>

#include <OCR/OCR.h>
#include <OCRForm/NetworkVideoSource.h>
#include <OCRForm/OcrFormDocument.h>
#include <OCRForm/OcrFormExporter.h>

using namespace Upp;
using namespace OCRForm;

namespace {

struct ZoneCheck : Moveable<ZoneCheck> {
	String path;
	Rect rect;
	String widget_type;
	String expected_text;
};

struct Options : Moveable<Options> {
	String ocrform_path;
	String remote;
	String classifier_model;
	String splitter_model;
	String save_dir;
	String emit_demo_ocrform;
	String emit_demo_mode = "flat";
	String export_form_path;
	String ocr_engine = "auto";
	int tesseract_psm = 7;
	int frames = 1;
	double min_confidence = 0.20;
	bool export_include_unknown = false;
	bool export_flatten_hierarchy = false;
	bool emit_json = false;
	bool fail_fast = false;
	bool require_expected_match = false;
	bool show_help = false;
};

String GetHintCaseInsensitive(const OcrAnnotation& a, const String& key)
{
	String want = ToLower(key);
	for(int i = 0; i < a.hints.GetCount(); i++) {
		if(ToLower(a.hints.GetKey(i)) == want)
			return a.hints[i];
	}
	return String();
}

String ResolveExpectedText(const OcrAnnotation& a)
{
	String out = GetHintCaseInsensitive(a, "text");
	if(out.IsEmpty()) out = GetHintCaseInsensitive(a, "label");
	if(out.IsEmpty()) out = GetHintCaseInsensitive(a, "caption");
	if(out.IsEmpty()) out = GetHintCaseInsensitive(a, "title");
	if(out.IsEmpty()) out = a.name;
	return TrimBoth(out);
}

bool IsReadableWidgetType(const String& widget_type)
{
	String w = ToLower(widget_type);
	return w.Find("label") >= 0 || w.Find("button") >= 0;
}

String BuildAnnotationPath(const String& prefix, const OcrAnnotation& annotation, int index)
{
	String label = annotation.name;
	if(label.IsEmpty()) label = annotation.widget_type;
	if(label.IsEmpty()) label = Format("annotation_%d", index);
	return prefix.IsEmpty() ? label : prefix + "/" + label;
}

void CollectReadableZones(const Array<OcrAnnotation>& annotations, const String& prefix, Vector<ZoneCheck>& out)
{
	for(int i = 0; i < annotations.GetCount(); i++) {
		const OcrAnnotation& a = annotations[i];
		String path = BuildAnnotationPath(prefix, a, i);

		if(IsReadableWidgetType(a.widget_type)) {
			ZoneCheck& z = out.Add();
			z.path = path;
			z.rect = a.rect;
			z.widget_type = a.widget_type;
			z.expected_text = ResolveExpectedText(a);
		}

		if(a.children.GetCount())
			CollectReadableZones(a.children, path, out);
	}
}

String NormalizeText(const String& in)
{
	String s = ToLower(in);
	s.Replace("\r", " ");
	s.Replace("\n", " ");
	Vector<String> parts = Split(s, ' ');
	String out;
	for(int i = 0; i < parts.GetCount(); i++) {
		String p = TrimBoth(parts[i]);
		if(p.IsEmpty())
			continue;
		if(!out.IsEmpty())
			out << ' ';
		out << p;
	}
	return out;
}

Rect NormalizeRect(const Rect& r)
{
	int left = min(r.left, r.right);
	int top = min(r.top, r.bottom);
	int right = max(r.left, r.right);
	int bottom = max(r.top, r.bottom);
	return Rect(left, top, right, bottom);
}

void PrintHelp()
{
	Cout() << "ocr-form-process\n\n"
	       << "Usage:\n"
	       << "  ocr-form-process --ocrform <file.ocrform> --remote <host:port>\n"
	       << "                   --classifier <model.ocrmodel> [options]\n"
	       << "  ocr-form-process --ocrform <file.ocrform> --export-form <file.form>\n"
	       << "                   [--export-include-unknown] [--export-flatten-hierarchy]\n"
	       << "  ocr-form-process --emit-demo-ocrform <file.ocrform> [--remote <host:port>]\n\n"
	       << "Options:\n"
	       << "  --ocr-engine <auto|convnet|tesseract>  OCR backend (default: auto)\n"
	       << "  --psm <int>                      Tesseract page segmentation mode (default: 7)\n"
	       << "  --splitter <model.ocrmodel>      Optional splitter model\n"
	       << "  --frames <count>                 Frames to evaluate (default: 1)\n"
	       << "  --min-confidence <0..1>          Readability threshold (default: 0.20)\n"
	       << "  --fail-fast                      Exit on first failed zone check\n"
	       << "  --emit-json                      Emit machine-readable JSON summary\n"
	       << "  --require-expected-match         Enforce expected text match when available\n"
	       << "  --save-dir <dir>                 Save per-zone crops for debugging\n"
	       << "  --emit-demo-ocrform <file>       Write canonical demo .ocrform and exit\n"
	       << "  --emit-demo-mode <flat|nested>   Demo layout style (default: flat)\n"
	       << "  --export-form <file.form>        Export .ocrform to .form and exit\n"
	       << "  --export-include-unknown         Export unknown widget types as Label placeholders\n"
	       << "  --export-flatten-hierarchy       Flatten hierarchy without warning\n"
	       << "  --help                           Show this help\n";
}

bool ParseOptions(const Vector<String>& args, Options& opt, String& error)
{
	for(int i = 0; i < args.GetCount(); i++) {
		const String& a = args[i];
		if(a == "--help" || a == "-h") {
			opt.show_help = true;
		}
		else if(a == "--ocrform" && i + 1 < args.GetCount()) {
			opt.ocrform_path = args[++i];
		}
		else if(a == "--remote" && i + 1 < args.GetCount()) {
			opt.remote = args[++i];
		}
		else if(a == "--classifier" && i + 1 < args.GetCount()) {
			opt.classifier_model = args[++i];
		}
		else if(a == "--splitter" && i + 1 < args.GetCount()) {
			opt.splitter_model = args[++i];
		}
		else if(a == "--ocr-engine" && i + 1 < args.GetCount()) {
			opt.ocr_engine = ToLower(args[++i]);
		}
		else if(a == "--psm" && i + 1 < args.GetCount()) {
			opt.tesseract_psm = max(0, StrInt(args[++i]));
		}
		else if(a == "--frames" && i + 1 < args.GetCount()) {
			opt.frames = max(1, StrInt(args[++i]));
		}
		else if(a == "--min-confidence" && i + 1 < args.GetCount()) {
			opt.min_confidence = min(1.0, max(0.0, StrDbl(args[++i])));
		}
		else if(a == "--fail-fast") {
			opt.fail_fast = true;
		}
		else if(a == "--emit-json") {
			opt.emit_json = true;
		}
		else if(a == "--require-expected-match") {
			opt.require_expected_match = true;
		}
		else if(a == "--save-dir" && i + 1 < args.GetCount()) {
			opt.save_dir = args[++i];
		}
		else if(a == "--emit-demo-ocrform" && i + 1 < args.GetCount()) {
			opt.emit_demo_ocrform = args[++i];
		}
		else if(a == "--emit-demo-mode" && i + 1 < args.GetCount()) {
			opt.emit_demo_mode = ToLower(args[++i]);
		}
		else if(a == "--export-form" && i + 1 < args.GetCount()) {
			opt.export_form_path = args[++i];
		}
		else if(a == "--export-include-unknown") {
			opt.export_include_unknown = true;
		}
		else if(a == "--export-flatten-hierarchy") {
			opt.export_flatten_hierarchy = true;
		}
		else {
			error = "Unknown argument: " + a;
			return false;
		}
	}

	if(opt.show_help)
		return true;
	if(!opt.emit_demo_ocrform.IsEmpty())
		return true;
	if(!opt.export_form_path.IsEmpty()) {
		if(opt.ocrform_path.IsEmpty()) {
			error = "--ocrform is required";
			return false;
		}
		return true;
	}
	if(opt.ocrform_path.IsEmpty()) {
		error = "--ocrform is required";
		return false;
	}
	if(opt.ocr_engine != "auto" && opt.ocr_engine != "convnet" && opt.ocr_engine != "tesseract") {
		error = "--ocr-engine must be one of: auto, convnet, tesseract";
		return false;
	}
	if(opt.emit_demo_mode != "flat" && opt.emit_demo_mode != "nested") {
		error = "--emit-demo-mode must be one of: flat, nested";
		return false;
	}
	if(opt.ocr_engine == "convnet" && opt.classifier_model.IsEmpty()) {
		error = "--classifier is required";
		return false;
	}
	return true;
}

Rect ClampRectToFrame(const Rect& zone, const Size& sz)
{
	Rect r = NormalizeRect(zone);
	r &= Rect(sz);
	return r;
}

String ToOneLine(const String& text)
{
	String out = text;
	out.Replace("\r", " ");
	out.Replace("\n", " ");
	return TrimBoth(out);
}

String RunCommandCapture(const String& cmd, int& exit_code)
{
	exit_code = -1;
#ifdef PLATFORM_WIN32
	FILE* pipe = _popen(~cmd, "r");
#else
	FILE* pipe = popen(~cmd, "r");
#endif
	if(!pipe)
		return String();

	String out;
	char buf[4096];
	while(fgets(buf, sizeof(buf), pipe))
		out.Cat(buf);

#ifdef PLATFORM_WIN32
	exit_code = _pclose(pipe);
#else
	exit_code = pclose(pipe);
#endif
	return out;
}

bool HasTesseract()
{
	int rc = -1;
#ifdef PLATFORM_WIN32
	RunCommandCapture("tesseract --version > NUL 2>&1", rc);
#else
	RunCommandCapture("tesseract --version >/dev/null 2>&1", rc);
#endif
	return rc == 0;
}

bool RecognizeWithTesseract(const Image& patch, int psm, String& text, double& confidence)
{
	text.Clear();
	confidence = 0.0;
	if(patch.IsEmpty())
		return false;

	Image prep = patch;
	if(prep.GetWidth() < 400 || prep.GetHeight() < 120)
		prep = Rescale(prep, max(1, prep.GetWidth() * 2), max(1, prep.GetHeight() * 2));

	int64 sum_luma = 0;
	int px_count = prep.GetWidth() * prep.GetHeight();
	for(int y = 0; y < prep.GetHeight(); y++)
		for(int x = 0; x < prep.GetWidth(); x++) {
			const RGBA& p = prep[y][x];
			sum_luma += (int)p.r * 77 + (int)p.g * 150 + (int)p.b * 29;
		}
	int avg_luma = px_count > 0 ? (int)(sum_luma / (px_count * 256)) : 128;
	bool invert = avg_luma < 120;
	int threshold = invert ? 130 : 170;

	ImageBuffer ib(prep.GetSize());
	for(int y = 0; y < prep.GetHeight(); y++) {
		RGBA* dst = ib[y];
		for(int x = 0; x < prep.GetWidth(); x++) {
			const RGBA& p = prep[y][x];
			int lum = ((int)p.r * 77 + (int)p.g * 150 + (int)p.b * 29) >> 8;
			bool ink = invert ? (lum > threshold) : (lum < threshold);
			byte v = ink ? 0 : 255;
			dst[x].r = v;
			dst[x].g = v;
			dst[x].b = v;
			dst[x].a = 255;
		}
	}
	ib.SetKind(IMAGE_OPAQUE);
	Image bw = ib;

	String tmp = GetTempFileName("ocrform_tess_") + ".png";
	if(!PNGEncoder().SaveFile(tmp, bw))
		return false;

	String cmd;
	String psm_arg = AsString(max(0, psm));
#ifdef PLATFORM_WIN32
	cmd = "tesseract \"" + tmp + "\" stdout --psm " + psm_arg + " -l eng 2> NUL";
#else
	cmd = "tesseract \"" + tmp + "\" stdout --psm " + psm_arg + " -l eng 2>/dev/null";
#endif
	int rc = -1;
	String out = RunCommandCapture(cmd, rc);
	DeleteFile(tmp);
	if(rc != 0)
		return false;

	out.Replace("\f", " ");
	text = TrimBoth(out);
	if(!text.IsEmpty())
		confidence = 1.0;
	return true;
}

}

CONSOLE_APP_MAIN
{
	StdLogSetup(LOG_COUT | LOG_FILE);

	Options opt;
	String parse_error;
	if(!ParseOptions(CommandLine(), opt, parse_error)) {
		Cerr() << "ERROR: " << parse_error << "\n\n";
		PrintHelp();
		Exit(2);
	}
	if(opt.show_help) {
		PrintHelp();
		return;
	}
	if(!opt.emit_demo_ocrform.IsEmpty()) {
		OcrFormDocument demo;
		demo.description = "OCRForm feedback loop demo";
		demo.video_source_type = "network";
		demo.video_source_param = opt.remote.IsEmpty() ? "127.0.0.1:8082" : opt.remote;

		OcrFormLayout layout;
		layout.name = "Default";
		layout.canvas_size = Size(1280, 720);

		OcrAnnotation label;
		label.rect = Rect(120, 190, 400, 242);
		label.widget_type = "label";
		label.name = "username_label";
		label.analyser_id = "manual";
		label.confidence = 1.0;
		label.hints.Add("text", "Username");

		OcrAnnotation button;
		button.rect = Rect(170, 368, 308, 424);
		button.widget_type = "button";
		button.name = "login_button";
		button.analyser_id = "manual";
		button.confidence = 1.0;
		button.hints.Add("text", "Login");

		if(opt.emit_demo_mode == "nested") {
			OcrAnnotation panel;
			panel.rect = Rect(80, 140, 980, 520);
			panel.widget_type = "panel";
			panel.name = "auth_panel";
			panel.analyser_id = "manual";
			panel.confidence = 1.0;
			panel.children.Add(label);
			panel.children.Add(button);
			layout.roots.Add(panel);
		}
		else {
			layout.roots.Add(label);
			layout.roots.Add(button);
		}
		demo.layouts.Add(layout);

		if(!demo.Save(opt.emit_demo_ocrform)) {
			Cerr() << "ERROR: failed to write demo .ocrform: " << opt.emit_demo_ocrform << "\n";
			Exit(2);
		}

		Cout() << "Wrote demo .ocrform: " << opt.emit_demo_ocrform
		       << " remote=" << demo.video_source_param << "\n";
		return;
	}

	OcrFormDocument doc;
	if(!doc.Load(opt.ocrform_path)) {
		Cerr() << "ERROR: failed to load .ocrform: " << opt.ocrform_path << "\n";
		Exit(2);
	}
	if(!opt.export_form_path.IsEmpty()) {
		String export_path = opt.export_form_path;
		if(GetFileExt(export_path).IsEmpty())
			export_path << ".form";

		OcrFormExporter exporter;
		OcrFormExporter::Options xopt;
		xopt.skip_unknown_types = !opt.export_include_unknown;
		xopt.flatten_hierarchy = opt.export_flatten_hierarchy;
		if(!exporter.ExportToFile(export_path, doc, xopt)) {
			Cerr() << "ERROR: failed to export .form: " << export_path << "\n";
			Exit(2);
		}

		Cout() << "Exported " << exporter.GetExportedCount() << " object(s) to "
		       << export_path << "\n";
		const Vector<String>& warns = exporter.GetWarnings();
		if(warns.GetCount()) {
			Cout() << "Warnings (" << warns.GetCount() << "):\n";
			for(int i = 0; i < warns.GetCount(); i++)
				Cout() << "  - " << warns[i] << "\n";
		}
		return;
	}

	if(opt.remote.IsEmpty() && ToLower(doc.video_source_type) == "network")
		opt.remote = doc.video_source_param;
	if(opt.remote.IsEmpty()) {
		Cerr() << "ERROR: --remote is required (or set video_source_type=network in .ocrform)\n";
		Exit(2);
	}

	Vector<ZoneCheck> zones;
	if(doc.layouts.GetCount())
		CollectReadableZones(doc.layouts[0].roots, String(), zones);
	if(zones.IsEmpty()) {
		Cerr() << "ERROR: no label/button annotations found in .ocrform\n";
		Exit(2);
	}

	bool use_tesseract = (opt.ocr_engine == "tesseract");
	if(opt.ocr_engine == "auto")
		use_tesseract = opt.classifier_model.IsEmpty();

	OCR::OCREngine engine;
	if(!use_tesseract) {
		OCR::OCRConfig cfg;
		cfg.classifier_model_path = opt.classifier_model;
		cfg.splitter_model_path = opt.splitter_model;
		cfg.use_splitter = !opt.splitter_model.IsEmpty();
		if(!engine.Initialize(cfg)) {
			Cerr() << "ERROR: OCR engine init failed: " << engine.GetLastError() << "\n";
			Exit(2);
		}
	}
	else {
		if(!HasTesseract()) {
			Cerr() << "ERROR: --ocr-engine tesseract requested but 'tesseract' was not found in PATH\n";
			Exit(2);
		}
	}

	NetworkVideoSource source;
	if(!source.Open(opt.remote)) {
		Cerr() << "ERROR: failed to connect VideoServer at " << opt.remote << "\n";
		Exit(2);
	}

	if(!opt.save_dir.IsEmpty())
		RealizeDirectory(opt.save_dir);

	Vector<bool> zone_passed;
	zone_passed.SetCount(zones.GetCount(), false);
	int total_checks = 0;
	int passed_checks = 0;
	ValueArray checks_json;
	bool failed_early = false;

	for(int frame_i = 0; frame_i < opt.frames; frame_i++) {
		Image frame;
		for(int retry = 0; retry < 50; retry++) {
			frame = source.GetFrame();
			if(!frame.IsEmpty())
				break;
			Sleep(20);
		}

		if(frame.IsEmpty()) {
			Cerr() << "ERROR: no frame received from VideoServer on frame " << (frame_i + 1) << "\n";
			source.Close();
			Exit(3);
		}

		Cout() << "Frame " << (frame_i + 1) << " size " << frame.GetWidth() << "x" << frame.GetHeight() << "\n";

		for(int zi = 0; zi < zones.GetCount(); zi++) {
			const ZoneCheck& zone = zones[zi];
			Rect rect = ClampRectToFrame(zone.rect, frame.GetSize());
			if(rect.IsEmpty()) {
				Cout() << "  [FAIL] " << zone.path << " empty rect after clamp\n";
				total_checks++;
				ValueMap rec;
				rec.Add("frame", frame_i + 1);
				rec.Add("zone_index", zi + 1);
				rec.Add("path", zone.path);
				rec.Add("status", "FAIL");
				rec.Add("reason", "empty-rect");
				checks_json.Add(rec);
				if(opt.fail_fast) {
					failed_early = true;
					break;
				}
				continue;
			}

			Image patch = Crop(frame, rect);
			String observed;
			double conf = 0.0;
			TimeStop zone_ts;
			if(use_tesseract) {
				String tess;
				double tess_conf = 0.0;
				if(RecognizeWithTesseract(patch, opt.tesseract_psm, tess, tess_conf)) {
					observed = ToOneLine(tess);
					conf = tess_conf;
				}
			}
			else {
				OCR::OCRResult rr = engine.RecognizePage(patch);
				observed = ToOneLine(rr.full_text);
				conf = rr.avg_confidence;
			}
			double zone_ms = zone_ts.Elapsed() / 1000.0;

			bool readable = !observed.IsEmpty() && conf >= opt.min_confidence;
			bool expected_ok = true;
			if(opt.require_expected_match && !zone.expected_text.IsEmpty()) {
				expected_ok = NormalizeText(observed) == NormalizeText(zone.expected_text);
			}
			bool ok = readable && expected_ok;
			if(ok)
				zone_passed[zi] = true;

			total_checks++;
			if(ok)
				passed_checks++;

			String status = ok ? "OK" : "FAIL";
			Cout() << "  [" << status << "] " << zone.path
			       << " type=" << zone.widget_type
			       << " conf=" << Format("%.3f", conf)
			       << " ms=" << Format("%.2f", zone_ms)
			       << " text=\"" << observed << "\"";
			if(!zone.expected_text.IsEmpty())
				Cout() << " expected=\"" << zone.expected_text << "\"";
			Cout() << "\n";

			ValueMap rec;
			rec.Add("frame", frame_i + 1);
			rec.Add("zone_index", zi + 1);
			rec.Add("path", zone.path);
			rec.Add("widget_type", zone.widget_type);
			rec.Add("status", status);
			rec.Add("confidence", conf);
			rec.Add("time_ms", zone_ms);
			rec.Add("text", observed);
			rec.Add("expected", zone.expected_text);
			rec.Add("rect", Format("%d,%d,%d,%d", rect.left, rect.top, rect.GetWidth(), rect.GetHeight()));
			checks_json.Add(rec);

			if(!opt.save_dir.IsEmpty()) {
				String base = Format("frame%03d_zone%03d", frame_i + 1, zi + 1);
				String img_path = AppendFileName(opt.save_dir, base + ".png");
				PNGEncoder().SaveFile(img_path, patch);
			}

			if(opt.fail_fast && !ok) {
				failed_early = true;
				break;
			}
		}
		if(failed_early)
			break;
	}

	source.Close();

	int zones_passed_once = 0;
	for(int i = 0; i < zone_passed.GetCount(); i++)
		if(zone_passed[i])
			zones_passed_once++;

	Cout() << "\nSummary:\n";
	Cout() << "  zones: " << zones.GetCount() << "\n";
	Cout() << "  checks: " << total_checks << "\n";
	Cout() << "  passed checks: " << passed_checks << "\n";
	Cout() << "  zones passed at least once: " << zones_passed_once << "/" << zones.GetCount() << "\n";
	if(failed_early)
		Cout() << "  fail-fast: triggered\n";

	bool pass = zones_passed_once == zones.GetCount() && !failed_early;
	if(pass) {
		Cout() << "RESULT: PASS\n";
	}
	else {
		Cout() << "RESULT: FAIL\n";
	}

	if(opt.emit_json) {
		ValueMap root;
		root.Add("result", pass ? "PASS" : "FAIL");
		root.Add("backend", use_tesseract ? "tesseract" : "convnet");
		root.Add("frames_requested", opt.frames);
		root.Add("checks", total_checks);
		root.Add("passed_checks", passed_checks);
		root.Add("zones_total", zones.GetCount());
		root.Add("zones_passed_once", zones_passed_once);
		root.Add("fail_fast_triggered", failed_early);
		root.Add("checks_detail", checks_json);
		Cout() << StoreAsJson(root) << "\n";
	}

	if(!pass)
		Exit(1);
}
