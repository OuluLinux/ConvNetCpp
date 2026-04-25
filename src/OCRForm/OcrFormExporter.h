#ifndef _OCRForm_OcrFormExporter_h_
#define _OCRForm_OcrFormExporter_h_

#include <Core/Core.h>

#include "OcrFormDocument.h"

NAMESPACE_UPP

class OcrFormExporter {
public:
	struct Options {
		bool   flatten_hierarchy;
		bool   skip_unknown_types;
		String default_layout_name;

		Options()
			: flatten_hierarchy(false)
			, skip_unknown_types(true)
			, default_layout_name("Default")
		{}
	};

	String Export(const OcrFormDocument& doc, const Options& opts = Options());
	bool   ExportToFile(const String& path, const OcrFormDocument& doc, const Options& opts = Options());

	const Vector<String>& GetWarnings() const { return warnings; }
	int                  GetExportedCount() const { return exported_count; }

private:
	Vector<String> warnings;
	int            exported_count = 0;
};

END_UPP_NAMESPACE

namespace OCRForm {
using Upp::OcrFormExporter;
}

#endif
