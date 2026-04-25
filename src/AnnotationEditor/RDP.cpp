#include "RDP.h"

using namespace Upp;

double PointLineDistanceSq(Pointf p, Pointf a, Pointf b) {
	double dx = b.x - a.x;
	double dy = b.y - a.y;
	if (dx == 0 && dy == 0) {
		dx = p.x - a.x;
		dy = p.y - a.y;
		return dx * dx + dy * dy;
	}
	double t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / (dx * dx + dy * dy);
	if (t < 0) {
		dx = p.x - a.x;
		dy = p.y - a.y;
	} else if (t > 1) {
		dx = p.x - b.x;
		dy = p.y - b.y;
	} else {
		Pointf nearest(a.x + t * dx, a.y + t * dy);
		dx = p.x - nearest.x;
		dy = p.y - nearest.y;
	}
	return dx * dx + dy * dy;
}

void RDPSimplify(const Vector<Pointf>& points, int first, int last, double epsSq, Vector<bool>& keep) {
	double maxDistSq = 0;
	int index = -1;

	for (int i = first + 1; i < last; i++) {
		double sqDist = PointLineDistanceSq(points[i], points[first], points[last]);
		if (sqDist > maxDistSq) {
			index = i;
			maxDistSq = sqDist;
		}
	}

	if (maxDistSq > epsSq && index != -1) {
		keep[index] = true;
		RDPSimplify(points, first, index, epsSq, keep);
		RDPSimplify(points, index, last, epsSq, keep);
	}
}

Vector<Pointf> SimplifyPolygonRDP(const Vector<Pointf>& points, double epsilon) {
	if (points.GetCount() < 3) {
		Vector<Pointf> res;
		res <<= points;
		return res;
	}

	Vector<bool> keep;
	keep.SetCount(points.GetCount(), false);
	keep[0] = true;
	keep[points.GetCount() - 1] = true;

	RDPSimplify(points, 0, points.GetCount() - 1, epsilon * epsilon, keep);

	Vector<Pointf> result;
	for (int i = 0; i < points.GetCount(); i++) {
		if (keep[i]) {
			result.Add(points[i]);
		}
	}
	return result;
}