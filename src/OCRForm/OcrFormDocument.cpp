#include "OcrFormDocument.h"

NAMESPACE_UPP

namespace {

String ResolveOcrFormPath(const String& path, const String& fallback)
{
	String out = path.IsEmpty() ? fallback : path;
	if(out.IsEmpty())
		return out;
	if(GetFileName(out).Find('.') < 0)
		out << ".ocrform";
	return out;
}

}

OcrFormDocument::OcrFormDocument()
{
}

OcrFormDocument::OcrFormDocument(const OcrFormDocument& src)
{
	*this = src;
}

OcrFormDocument& OcrFormDocument::operator=(const OcrFormDocument& src)
{
	XMLConfig::operator=(src);
	description = src.description;
	layouts = clone(src.layouts);
	video_source_type = src.video_source_type;
	video_source_param = src.video_source_param;
	analysis_trigger_mode = src.analysis_trigger_mode;
	analysis_interval_ms = src.analysis_interval_ms;
	changed_area_threshold = src.changed_area_threshold;
	changed_area_block_size = src.changed_area_block_size;
	proposal_accept_threshold = src.proposal_accept_threshold;
	auto_accept_proposals = src.auto_accept_proposals;
	return *this;
}

bool OcrFormDocument::Load(const String& path)
{
	String resolved = ResolveOcrFormPath(path, Path);
	if(resolved.IsEmpty() || !FileExists(resolved))
		return false;

	OcrFormDocument loaded;
	if(!LoadFromXML(loaded, LoadFile(resolved)))
		return false;

	loaded.Path = resolved;
	*this = loaded;
	return true;
}

bool OcrFormDocument::Save(const String& path) const
{
	String resolved = ResolveOcrFormPath(path, Path);
	if(resolved.IsEmpty())
		return false;

	String xml = StoreAsXML(*this, "ocrform");
	if(!SaveFile(resolved, xml))
		return false;
	const_cast<OcrFormDocument*>(this)->Path = resolved;
	return true;
}

bool OcrFormDocument::LoadString(const String& xml)
{
	OcrFormDocument loaded;
	if(!LoadFromXML(loaded, xml))
		return false;

	*this = loaded;
	return true;
}

String OcrFormDocument::ToString() const
{
	return StoreAsXML(*this, "ocrform");
}

void OcrFormDocument::Xmlize(XmlIO xml)
{
	xml.Attr("description", description, String());
	xml.Attr("analysis_mode", analysis_trigger_mode, String("on_change"));
	xml.Attr("analysis_interval_ms", analysis_interval_ms, 700);
	xml.Attr("changed_threshold", changed_area_threshold, 30);
	xml.Attr("changed_block_size", changed_area_block_size, 8);
	xml.Attr("accept_threshold", proposal_accept_threshold, 0.8);
	xml.Attr("auto_accept", auto_accept_proposals, false);
	if(analysis_trigger_mode.IsEmpty())
		analysis_trigger_mode = "on_change";
	analysis_interval_ms = max(0, analysis_interval_ms);
	changed_area_threshold = min(255, max(0, changed_area_threshold));
	changed_area_block_size = min(128, max(1, changed_area_block_size));
	proposal_accept_threshold = min(1.0, max(0.0, proposal_accept_threshold));

	if(xml.IsLoading()) {
		video_source_type.Clear();
		video_source_param.Clear();
		for(int i = 0; i < xml->GetCount(); i++) {
			const XmlNode& node = xml->Node(i);
			if(!node.IsTag("video_source"))
				continue;
			video_source_type = node.Attr("type");
			video_source_param = node.Attr("param");
			break;
		}
	}
	else {
		if(!video_source_type.IsEmpty() || !video_source_param.IsEmpty()) {
			XmlIO source = xml.Add("video_source");
			source.Attr("type", video_source_type, String());
			source.Attr("param", video_source_param, String());
		}
	}

	xml("layout", layouts);
}

END_UPP_NAMESPACE
