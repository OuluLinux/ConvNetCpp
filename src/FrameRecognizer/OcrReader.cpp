#include "OcrReader.h"

#include <Core/Core.h>

#include <limits>
#include <stdio.h>

NAMESPACE_UPP

namespace {

String OcrShellEscape(const String& s)
{
	String out = "'";
	for(int i = 0; i < s.GetCount(); i++) {
		if(s[i] == '\'')
			out << "'\\''";
		else
			out.Cat(s[i]);
	}
	out.Cat('\'');
	return out;
}

String OcrRunCommandCapture(const String& cmd, int& exit_code)
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

String WriteSlotsTemp(const Vector<DetectedSlot>& slots)
{
	ValueArray arr;
	for(const auto& s : slots) {
		ValueMap item;
		ValueMap bbox;
		bbox.Add("left", s.bbox.left);
		bbox.Add("top", s.bbox.top);
		bbox.Add("right", s.bbox.right);
		bbox.Add("bottom", s.bbox.bottom);
		item.Add("slot_id", s.slot_id);
		item.Add("confidence", s.confidence);
		item.Add("bbox", bbox);
		arr.Add(item);
	}
	ValueMap root;
	root.Add("detected_slots", arr);
	String path = AppendFileName(GetTempPath(), Format("fr_ocr_slots_%d.json", (int)GetTickCount()));
	SaveFile(path, StoreAsJson(root, true));
	return path;
}

double ParseNumericFromParsed(const Value& parsed)
{
	if(parsed.Is<int>() || parsed.Is<int64>() || parsed.Is<double>())
		return double(parsed);
	if(parsed.Is<ValueMap>()) {
		ValueMap m = parsed;
		Value v = m.Get("value", Value());
		if(v.Is<int>() || v.Is<int64>() || v.Is<double>())
			return double(v);
	}
	return std::numeric_limits<double>::quiet_NaN();
}

}

bool OcrReader::Init(const String& script_dir)
{
	script_dir_ = script_dir;
	return FileExists(AppendFileName(script_dir_, "ocr_reader.py"));
}

Vector<OcrResult> OcrReader::Read(const String& image_path, const Vector<DetectedSlot>& slots)
{
	Vector<OcrResult> out;
	if(script_dir_.IsEmpty() || !FileExists(image_path))
		return out;

	String slots_json = WriteSlotsTemp(slots);
	String out_json = AppendFileName(GetTempPath(), Format("fr_ocr_%d.json", (int)GetTickCount()));
	String script_path = AppendFileName(script_dir_, "ocr_reader.py");
	String cmd;
	cmd << "python3 " << OcrShellEscape(script_path)
	    << " --image " << OcrShellEscape(image_path)
	    << " --detections " << OcrShellEscape(slots_json)
	    << " --output " << OcrShellEscape(out_json)
	    << " 2>&1";

	int rc = -1;
	OcrRunCommandCapture(cmd, rc);
	DeleteFile(slots_json);
	if(rc != 0 || !FileExists(out_json)) {
		DeleteFile(out_json);
		return out;
	}

	Value parsed = ParseJSON(LoadFile(out_json));
	DeleteFile(out_json);
	if(!parsed.Is<ValueMap>())
		return out;

	ValueMap root = parsed;
	for(int i = 0; i < root.GetCount(); i++) {
		String slot_id = root.GetKey(i);
		Value entry_v = root[i];
		if(!entry_v.Is<ValueMap>())
			continue;
		ValueMap entry = entry_v;

		OcrResult& r = out.Add();
		r.slot_id = slot_id;
		r.text = entry.Get("raw", "");
		r.value = ParseNumericFromParsed(entry.Get("parsed", Value()));
	}
	return out;
}

END_UPP_NAMESPACE
