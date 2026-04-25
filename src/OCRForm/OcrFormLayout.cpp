#include "OcrFormLayout.h"

NAMESPACE_UPP

OcrFormLayout::OcrFormLayout()
{
}

OcrFormLayout::OcrFormLayout(const OcrFormLayout& src)
{
	*this = src;
}

OcrFormLayout& OcrFormLayout::operator=(const OcrFormLayout& src)
{
	XMLConfig::operator=(src);
	name = src.name;
	canvas_size = src.canvas_size;
	roots = clone(src.roots);
	return *this;
}

void OcrFormLayout::Xmlize(XmlIO xml)
{
	xml.Attr("name", name, String());
	xml.Attr("canvas_w", canvas_size.cx, 0);
	xml.Attr("canvas_h", canvas_size.cy, 0);
	xml("annotation", roots);
}

END_UPP_NAMESPACE
