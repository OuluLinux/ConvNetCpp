#ifndef _OCRForm_OcrFormDocument_h_
#define _OCRForm_OcrFormDocument_h_

#include <Core/Core.h>
#include <Form/Common/XMLConfig.h>

#include "OcrFormLayout.h"

NAMESPACE_UPP

class OcrFormDocument : public XMLConfig, Moveable<OcrFormDocument> {
public:
	String description;
	Array<OcrFormLayout> layouts;
	String video_source_type;
	String video_source_param;
	String analysis_trigger_mode = "on_change";
	int analysis_interval_ms = 700;
	int changed_area_threshold = 30;
	int changed_area_block_size = 8;
	double proposal_accept_threshold = 0.8;
	bool auto_accept_proposals = false;

	OcrFormDocument();
	OcrFormDocument(const OcrFormDocument& src);
	OcrFormDocument& operator=(const OcrFormDocument& src);

	bool Load(const String& path);
	bool Save(const String& path) const;
	bool LoadString(const String& xml);
	String ToString() const;

	void Xmlize(XmlIO xml);
};

END_UPP_NAMESPACE

namespace OCRForm {
using Upp::OcrFormDocument;
}

#endif
