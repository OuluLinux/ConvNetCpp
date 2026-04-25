#include "ElementRecognizer.h"

#include <Core/Core.h>

#include <stdio.h>

NAMESPACE_UPP

namespace {

String CardShellEscape(const String& s)
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

String CardRunCommandCapture(const String& cmd, int& exit_code)
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

bool ShouldKeepSlot(const String& slot_id, const Vector<DetectedSlot>& wanted)
{
	if(wanted.IsEmpty())
		return true;
	for(const auto& s : wanted)
		if(s.slot_id == slot_id)
			return true;
	return false;
}

}

bool CardRecognizer::Init(const String& script_dir, const String& templates_dir)
{
	script_dir_ = script_dir;
	templates_dir_ = templates_dir;
	return FileExists(AppendFileName(script_dir_, "card_recognizer.py"));
}

Vector<CardResult> CardRecognizer::Recognize(const String& image_path, const Vector<DetectedSlot>& card_slots)
{
	Vector<CardResult> out;
	if(script_dir_.IsEmpty() || !FileExists(image_path))
		return out;

	String out_json = AppendFileName(GetTempPath(), Format("fr_cards_%d.json", (int)GetTickCount()));
	String script_path = AppendFileName(script_dir_, "card_recognizer.py");
	String strip_path = AppendFileName(templates_dir_, "cards@2x.png");
	String mlui_path = AppendFileName(GetFileFolder(templates_dir_), "engine_table_8p.mlui");

	String cmd;
	cmd << "python3 " << CardShellEscape(script_path)
	    << " --image " << CardShellEscape(image_path)
	    << " --out " << CardShellEscape(out_json);
	if(FileExists(strip_path))
		cmd << " --strip " << CardShellEscape(strip_path);
	if(FileExists(mlui_path))
		cmd << " --mlui " << CardShellEscape(mlui_path);
	cmd << " 2>&1";

	int rc = -1;
	CardRunCommandCapture(cmd, rc);
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
		if(!ShouldKeepSlot(slot_id, card_slots))
			continue;

		Value entry_v = root[i];
		if(!entry_v.Is<ValueMap>())
			continue;
		ValueMap entry = entry_v;
		if(entry.Get("state", "") != "face_up")
			continue;
		Value card_v = entry.Get("element", Value());
		if(!card_v.Is<ValueMap>())
			continue;
		ValueMap element = card_v;
		String level = element.Get("level", "");
		String category = element.Get("category", "");
		if(level.IsEmpty() || category.IsEmpty())
			continue;

		CardResult& c = out.Add();
		c.slot_id = slot_id;
		c.level = level;
		c.category = category;
	}
	return out;
}

END_UPP_NAMESPACE
