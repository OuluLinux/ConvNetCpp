#include "SlotDetector.h"

#include <Core/Core.h>

#ifdef PLATFORM_WIN32
#include <stdio.h>
#else
#include <stdio.h>
#endif

NAMESPACE_UPP

namespace {

String ShellEscape(const String& s)
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

ValueMap LoadJsonMapFile(const String& path)
{
	Value parsed = ParseJSON(LoadFile(path));
	if(!parsed.Is<ValueMap>())
		return ValueMap();
	return parsed;
}

Rect ParseBBox(const ValueMap& bbox)
{
	ValueMap m = bbox;
	int l = int(double(m.Get("left", 0.0)));
	int t = int(double(m.Get("top", 0.0)));
	int r = int(double(m.Get("right", 0.0)));
	int b = int(double(m.Get("bottom", 0.0)));
	return RectC(l, t, max(0, r - l), max(0, b - t));
}

}

bool SlotDetector::Init(const String& model_path, const String& script_dir)
{
	model_path_ = model_path;
	script_dir_ = script_dir;
	return FileExists(model_path_) && FileExists(AppendFileName(script_dir_, "slot_detector_infer.py"));
}

void SlotDetector::SetSlotConfOverrides(const VectorMap<String, double>& overrides)
{
	slot_conf_overrides_.Clear();
	for(int i = 0; i < overrides.GetCount(); i++)
		slot_conf_overrides_.GetAdd(overrides.GetKey(i)) = overrides[i];
}

Vector<DetectedSlot> SlotDetector::Detect(const String& image_path, double conf_threshold)
{
	Vector<DetectedSlot> out;
	if(model_path_.IsEmpty() || script_dir_.IsEmpty())
		return out;
	if(!FileExists(image_path))
		return out;

	String out_json = AppendFileName(GetTempPath(), Format("fr_slots_%d.json", (int)GetTickCount()));
	String script_path = AppendFileName(script_dir_, "slot_detector_infer.py");
	String slot_conf;
	for(int i = 0; i < slot_conf_overrides_.GetCount(); i++) {
		if(i)
			slot_conf << ",";
		slot_conf << slot_conf_overrides_.GetKey(i) << ":" << Format("%.2f", slot_conf_overrides_[i]);
	}
	String cmd;
	cmd << "python3 " << ShellEscape(script_path)
	    << " " << ShellEscape(image_path)
	    << " --weights " << ShellEscape(model_path_)
	    << " --conf " << AsString(conf_threshold)
	    << " --format detected-slots"
	    << " --out " << ShellEscape(out_json);
	if(!slot_conf.IsEmpty())
		cmd << " --slot-conf " << ShellEscape(slot_conf);
	cmd << " 2>&1";

	int rc = -1;
	RunCommandCapture(cmd, rc);
	if(rc != 0 || !FileExists(out_json)) {
		DeleteFile(out_json);
		return out;
	}

	ValueMap root = LoadJsonMapFile(out_json);
	DeleteFile(out_json);
	if(root.IsEmpty())
		return out;

	Value slots_v = root.Get("detected_slots", Value());
	if(!slots_v.Is<ValueArray>())
		return out;

	const ValueArray& slots = slots_v;
	for(int i = 0; i < slots.GetCount(); i++) {
		const Value& v = slots[i];
		if(!v.Is<ValueMap>())
			continue;
		ValueMap m = v;
		Value bbox_v = m.Get("bbox", Value());
		if(!bbox_v.Is<ValueMap>())
			continue;

		DetectedSlot& s = out.Add();
		s.slot_id = m.Get("slot_id", "");
		s.confidence = double(m.Get("confidence", 0.0));
		s.bbox = ParseBBox((ValueMap)bbox_v);
	}
	return out;
}

END_UPP_NAMESPACE
