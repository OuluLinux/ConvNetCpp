#ifndef _AnnotationEditor_CocoFormat_h_
#define _AnnotationEditor_CocoFormat_h_

#include "Dataset.h"

using namespace Upp;

struct CocoExporter {
	static bool Export(const Dataset& ds, const Vector<Category>& categories, const String& path);
};

struct CocoImporter {
	static bool Import(Dataset& ds, Vector<Category>& categories, const String& path);
};

// Geometry helpers
double ComputePolygonArea(const Vector<Pointf>& poly);
Rectf  ComputeAnnotationBBox(const AnnotationObject& obj);
double ComputeAnnotationArea(const AnnotationObject& obj);

#endif
