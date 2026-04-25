#include "OcrFormExporter.h"

#include <FormEditor/FormEditor.h>

NAMESPACE_UPP

namespace {

struct ExportItem : Moveable<ExportItem> {
	const OcrAnnotation* ann = NULL;
	String               path;
};

Rect ExporterNormalizeRect(const Rect& rect)
{
	int left = min(rect.left, rect.right);
	int top = min(rect.top, rect.bottom);
	int right = max(rect.left, rect.right);
	int bottom = max(rect.top, rect.bottom);
	if(right <= left)
		right = left + 1;
	if(bottom <= top)
		bottom = top + 1;
	return Rect(left, top, right, bottom);
}

String GetHintCaseInsensitive(const OcrAnnotation& ann, const String& key)
{
	String want = ToLower(key);
	for(int i = 0; i < ann.hints.GetCount(); i++) {
		if(ToLower(ann.hints.GetKey(i)) == want)
			return ann.hints[i];
	}
	return String();
}

String ResolveLabelText(const OcrAnnotation& ann)
{
	String out = GetHintCaseInsensitive(ann, "text");
	if(out.IsEmpty()) out = GetHintCaseInsensitive(ann, "label");
	if(out.IsEmpty()) out = GetHintCaseInsensitive(ann, "caption");
	if(out.IsEmpty()) out = GetHintCaseInsensitive(ann, "title");
	if(out.IsEmpty()) out = ann.name;
	return TrimBoth(out);
}

String SanitizeIdentifier(const String& src)
{
	String out;
	for(int i = 0; i < src.GetCount(); i++) {
		int c = (byte)src[i];
		bool ok = (c >= 'A' && c <= 'Z')
		       || (c >= 'a' && c <= 'z')
		       || (c >= '0' && c <= '9')
		       || c == '_';
		out.Cat(ok ? c : '_');
	}

	if(out.IsEmpty())
		out = "annotation";

	while(out.GetCount() >= 2 && out.Find("__") >= 0)
		out.Replace("__", "_");

	if(out[0] >= '0' && out[0] <= '9')
		out = "n_" + out;

	return out;
}

String BuildPathToken(const OcrAnnotation& ann, int index)
{
	String token = ann.name;
	if(token.IsEmpty())
		token = ann.widget_type;
	if(token.IsEmpty())
		token = Format("annotation_%d", index);
	return SanitizeIdentifier(token);
}

String BuildPath(const String& prefix, const OcrAnnotation& ann, int index)
{
	String token = BuildPathToken(ann, index);
	return prefix.IsEmpty() ? token : prefix + "/" + token;
}

void CollectItems(const Array<OcrAnnotation>& annotations,
                  const String& prefix,
                  Vector<ExportItem>& out,
                  bool& hierarchy_notice)
{
	for(int i = 0; i < annotations.GetCount(); i++) {
		const OcrAnnotation& ann = annotations[i];
		String path = BuildPath(prefix, ann, i);

		ExportItem& item = out.Add();
		item.ann = &ann;
		item.path = path;

		if(ann.children.GetCount()) {
			hierarchy_notice = true;
			CollectItems(ann.children, path, out, hierarchy_notice);
		}
	}
}

String MapWidgetType(const OcrAnnotation& ann,
                     const OcrFormExporter::Options& opts,
                     const String& ann_path,
                     Vector<String>& warnings)
{
	String widget = ToLower(TrimBoth(ann.widget_type));
	if(widget == "button")
		return "Button";
	if(widget == "label")
		return "Label";
	if(widget == "editfield")
		return "EditField";
	if(widget == "menu_item" || widget == "menuitem")
		return "Button";
	if(widget == "menubar")
		return "Label";
	if(widget == "separator")
		return "Label";
	if(widget == "splitter")
		return "Label";
	if(widget == "icon" || widget == "image")
		return "Label";

	if(widget == "custom") {
		warnings.Add(Format("'%s': widget type '%s' exported as Label placeholder", ann_path, ann.widget_type));
		return "Label";
	}

	if(opts.skip_unknown_types) {
		warnings.Add(Format("'%s': unsupported widget type '%s' skipped", ann_path,
		                    ann.widget_type.IsEmpty() ? "<empty>" : ~ann.widget_type));
		return String();
	}

	warnings.Add(Format("'%s': unsupported widget type '%s' exported as Label placeholder", ann_path,
	                    ann.widget_type.IsEmpty() ? "<empty>" : ~ann.widget_type));
	return "Label";
}

String MakeUniqueVariable(const String& base, Index<String>& used)
{
	String id = SanitizeIdentifier(base);
	if(used.Find(id) < 0) {
		used.Add(id);
		return id;
	}

	for(int i = 2;; i++) {
		String candidate = id + "_" + AsString(i);
		if(used.Find(candidate) < 0) {
			used.Add(candidate);
			return candidate;
		}
	}
}

}

String OcrFormExporter::Export(const OcrFormDocument& doc, const Options& opts)
{
	warnings.Clear();
	exported_count = 0;

	const OcrFormLayout* src_layout = doc.layouts.GetCount() ? &doc.layouts[0] : NULL;

	Size canvas(1280, 720);
	String layout_name = opts.default_layout_name;
	if(src_layout) {
		if(src_layout->canvas_size.cx > 0 && src_layout->canvas_size.cy > 0)
			canvas = src_layout->canvas_size;
		if(!src_layout->name.IsEmpty())
			layout_name = src_layout->name;
	}

	FormView form;
	form.New();
	form.SetBool("Grid.Visible", false);
	form.SetBool("Grid.Binding", false);
	form.SetFormSize(canvas);
	if(FormLayout* lay = form.GetCurrentLayout()) {
		lay->SetNumber("Form.Width", canvas.cx);
		lay->SetNumber("Form.Height", canvas.cy);
		lay->Set("Form.Name", layout_name);
		lay->Set("Form.Title", doc.description.IsEmpty() ? layout_name : doc.description);
	}

	Vector<ExportItem> items;
	bool has_hierarchy = false;
	if(src_layout)
		CollectItems(src_layout->roots, String(), items, has_hierarchy);

	if(has_hierarchy && !opts.flatten_hierarchy) {
		warnings.Add("Hierarchy detected: .form format is flat, so child annotations were flattened");
	}

	Array<FormObject>* objects = form.GetObjects();
	if(!objects) {
		warnings.Add("Failed to access FormView object list");
		return String();
	}

	Index<String> used_vars;
	for(int i = 0; i < items.GetCount(); i++) {
		const OcrAnnotation& ann = *items[i].ann;
		String type = MapWidgetType(ann, opts, items[i].path, warnings);
		if(type.IsEmpty())
			continue;

		FormObject& obj = objects->Add(FormObject(ExporterNormalizeRect(ann.rect)));
		obj.Set("Type", type);
		obj.SetBool("OutlineDraw", false);

		String var_base = items[i].path;
		var_base.Replace("/", "_");
		String var = MakeUniqueVariable(var_base, used_vars);
		obj.Set("Variable", var);
		obj.Name = var;

		String label = ResolveLabelText(ann);
		String widget = ToLower(TrimBoth(ann.widget_type));
		if(label.IsEmpty()) {
			if(widget == "menubar")
				label = "Menu";
			else if(widget == "menu_item" || widget == "menuitem")
				label = "Item";
			else if(widget == "separator")
				label = "----------------";
			else if(widget == "splitter")
				label = "Splitter";
			else if(widget == "icon" || widget == "image")
				label = "[Icon]";
		}
		if(!label.IsEmpty())
			obj.Set("Label", label);

		obj.Set("OCR.SourcePath", items[i].path);
		obj.Set("OCR.WidgetType", ann.widget_type);
		if(!ann.analyser_id.IsEmpty())
			obj.Set("OCR.AnalyserID", ann.analyser_id);
		if(ann.confidence > 0.0)
			obj.Set("OCR.Confidence", FormatDouble(ann.confidence));

		exported_count++;
	}

	String xml;
	if(!form.SaveAllString(xml, false)) {
		warnings.Add("Failed to serialize exported .form XML");
		return String();
	}
	return xml;
}

bool OcrFormExporter::ExportToFile(const String& path, const OcrFormDocument& doc, const Options& opts)
{
	String out = path;
	if(out.IsEmpty())
		return false;
	if(GetFileExt(out).IsEmpty())
		out << ".form";

	String xml = Export(doc, opts);
	if(xml.IsEmpty())
		return false;

	return SaveFile(out, xml);
}

END_UPP_NAMESPACE
