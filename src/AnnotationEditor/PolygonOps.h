#ifndef _AnnotationEditor_PolygonOps_h_
#define _AnnotationEditor_PolygonOps_h_

#include <Core/Core.h>
#include <Draw/Draw.h>

using namespace Upp;

Vector<Vector<Pointf>> UnionPolygons(const Vector<Vector<Pointf>>& existing, const Vector<Vector<Pointf>>& strokes);
Vector<Vector<Pointf>> SubtractPolygons(const Vector<Vector<Pointf>>& existing, const Vector<Vector<Pointf>>& strokes);

Vector<Pointf> CreateCirclePolygon(Pointf center, double radius, int segments = 16);

Vector<Vector<Pointf>> MagicWand(const Image& img, Point seed, int threshold, int blur);
Vector<Vector<Pointf>> TraceMask(const Vector<Vector<bool>>& mask, Rectf bbox, double scale);

#endif
