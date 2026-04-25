#include "OcrAnnotation.h"

NAMESPACE_UPP

OcrAnnotation::OcrAnnotation()
	: confidence(0.0), selected(false)
{
}

OcrAnnotation::OcrAnnotation(const OcrAnnotation& src)
{
	*this = src;
}

OcrAnnotation& OcrAnnotation::operator=(const OcrAnnotation& src)
{
	XMLConfig::operator=(src);
	rect = src.rect;
	widget_type = src.widget_type;
	name = src.name;
	children = clone(src.children);
	analyser_id = src.analyser_id;
	confidence = src.confidence;
	hints = clone(src.hints);
	selected = src.selected;
	return *this;
}

void OcrAnnotation::Xmlize(XmlIO xml)
{
	int w = rect.GetWidth();
	int h = rect.GetHeight();

	xml.Attr("x", rect.left, 0);
	xml.Attr("y", rect.top, 0);
	xml.Attr("w", w, 0);
	xml.Attr("h", h, 0);

	if(xml.IsLoading()) {
		rect.right = rect.left + w;
		rect.bottom = rect.top + h;
	}

	xml.Attr("type", widget_type, String());
	xml.Attr("name", name, String());
	xml.Attr("analyser_id", analyser_id, String());
	xml.Attr("confidence", confidence, 0.0);

	if(xml.IsStoring()) {
		for(int i = 0; i < hints.GetCount(); i++) {
			XmlIO hint = xml.Add("hint");
			String key = hints.GetKey(i);
			String value = hints[i];
			hint.Attr("key", key);
			hint.Attr("value", value);
		}
	}
	else {
		hints.Clear();
		for(int i = 0; i < xml->GetCount(); i++) {
			const XmlNode& node = xml->Node(i);
			if(!node.IsTag("hint"))
				continue;
			String key = node.Attr("key");
			if(key.IsEmpty())
				continue;
			hints.GetAdd(key) = node.Attr("value");
		}
	}

	xml("annotation", children);
}

END_UPP_NAMESPACE
