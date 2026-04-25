#include "PolygonOps.h"
#include "RDP.h"
#include <Draw/Draw.h>

using namespace Upp;

Vector<Pointf> CreateCirclePolygon(Pointf center, double radius, int segments) {
	Vector<Pointf> poly;
	for(int i = 0; i < segments; i++) {
		double a = 2.0 * M_PI * i / segments;
		poly.Add(Pointf(center.x + radius * cos(a), center.y + radius * sin(a)));
	}
	return poly;
}

struct Span {
	int y;
	int x1, x2;
};

Vector<Vector<Pointf>> TraceMask(const Vector<Vector<bool>>& mask, Rectf bbox, double scale) {
	Vector<Vector<Pointf>> result;
	if(mask.IsEmpty()) return result;
	
	int rows = mask.GetCount();
	int cols = mask[0].GetCount();
	
	Vector<Span> spans;
	for(int y = 0; y < rows; y++) {
		int start = -1;
		for(int x = 0; x < cols; x++) {
			if(mask[y][x]) {
				if(start == -1) start = x;
			} else {
				if(start != -1) {
					spans.Add({y, start, x - 1});
					start = -1;
				}
			}
		}
		if(start != -1) spans.Add({y, start, cols - 1});
	}
	
	for(const auto& s : spans) {
		Vector<Pointf>& p = result.Add();
		double rx1 = bbox.left + s.x1 * scale;
		double rx2 = bbox.left + (s.x2 + 1) * scale;
		double ry1 = bbox.top + s.y * scale;
		double ry2 = bbox.top + (s.y + 1) * scale;
		
		p.Add(Pointf(rx1, ry1));
		p.Add(Pointf(rx2, ry1));
		p.Add(Pointf(rx2, ry2));
		p.Add(Pointf(rx1, ry2));
	}
	
	return result;
}

bool IsInPolygonsInternal(Pointf pt, const Vector<Vector<Pointf>>& polys) {
	for(const auto& poly : polys) {
		if(poly.GetCount() < 3) continue;
		bool inside = false;
		for(int i = 0, j = poly.GetCount() - 1; i < poly.GetCount(); j = i++) {
			if(((poly[i].y > pt.y) != (poly[j].y > pt.y)) &&
			   (pt.x < (poly[j].x - poly[i].x) * (pt.y - poly[i].y) / (poly[j].y - poly[i].y) + poly[i].x)) {
				inside = !inside;
			}
		}
		if(inside) return true;
	}
	return false;
}

Vector<Vector<Pointf>> ApplyOp(const Vector<Vector<Pointf>>& existing, const Vector<Vector<Pointf>>& strokes, bool subtract) {
	if(existing.IsEmpty() && !subtract) {
		Vector<Vector<Pointf>> res;
		res <<= strokes;
		return res;
	}
	
	double min_x = 1e18, min_y = 1e18, max_x = -1e18, max_y = -1e18;
	auto UpdateBBox = [&](const Vector<Vector<Pointf>>& polys) {
		for(const auto& poly : polys)
			for(const auto& p : poly) {
				min_x = min(min_x, p.x); max_x = max(max_x, p.x);
				min_y = min(min_y, p.y); max_y = max(max_y, p.y);
			}
	};
	UpdateBBox(existing);
	UpdateBBox(strokes);
	
	if(max_x <= min_x || max_y <= min_y) {
		Vector<Vector<Pointf>> res;
		res <<= existing;
		return res;
	}
	
	min_x -= 5; max_x += 5; min_y -= 5; max_y += 5;
	
	double scale = 2.0;
	int rows = int((max_y - min_y) / scale) + 1;
	int cols = int((max_x - min_x) / scale) + 1;
	
	if(rows > 1000) rows = 1000;
	if(cols > 1000) cols = 1000;
	
	Vector<Vector<bool>> mask;
	mask.SetCount(rows);
	for(int y = 0; y < rows; y++) {
		mask[y].SetCount(cols, false);
		for(int x = 0; x < cols; x++) {
			Pointf pt(min_x + x * scale + scale/2, min_y + y * scale + scale/2);
			bool in_existing = IsInPolygonsInternal(pt, existing);
			bool in_strokes = IsInPolygonsInternal(pt, strokes);
			if(subtract) mask[y][x] = in_existing && !in_strokes;
			else mask[y][x] = in_existing || in_strokes;
		}
	}
	
	return TraceMask(mask, Rectf(min_x, min_y, max_x, max_y), scale);
}

Vector<Vector<Pointf>> UnionPolygons(const Vector<Vector<Pointf>>& existing, const Vector<Vector<Pointf>>& strokes) {
	auto res = ApplyOp(existing, strokes, false);
	for(auto& p : res) p = SimplifyPolygonRDP(p, 1.0);
	return res;
}

Vector<Vector<Pointf>> SubtractPolygons(const Vector<Vector<Pointf>>& existing, const Vector<Vector<Pointf>>& strokes) {
	auto res = ApplyOp(existing, strokes, true);
	for(auto& p : res) p = SimplifyPolygonRDP(p, 1.0);
	return res;
}

Vector<Vector<Pointf>> MagicWand(const Image& img, Point seed, int threshold, int blur) {
	Size sz = img.GetSize();
	Vector<Vector<Pointf>> result;
	if(seed.x < 0 || seed.x >= sz.cx || seed.y < 0 || seed.y >= sz.cy) return result;
	
	RGBA target = img[seed.y][seed.x];
	
	Vector<Vector<bool>> visited;
	visited.SetCount(sz.cy);
	for(int i = 0; i < sz.cy; i++) visited[i].SetCount(sz.cx, false);
	
	Vector<Vector<bool>> mask;
	mask.SetCount(sz.cy);
	for(int i = 0; i < sz.cy; i++) mask[i].SetCount(sz.cx, false);
	
	Vector<Point> q;
	q.Add(seed);
	visited[seed.y][seed.x] = true;
	
	int head = 0;
	while(head < q.GetCount()) {
		Point p = q[head++];
		mask[p.y][p.x] = true;
		
		static const int dx[] = {0, 0, 1, -1};
		static const int dy[] = {1, -1, 0, 0};
		
		for(int i = 0; i < 4; i++) {
			int nx = p.x + dx[i];
			int ny = p.y + dy[i];
			
			if(nx >= 0 && nx < sz.cx && ny >= 0 && ny < sz.cy && !visited[ny][nx]) {
				RGBA cur = img[ny][nx];
				int diff = abs(cur.r - target.r) + abs(cur.g - target.g) + abs(cur.b - target.b);
				if(diff <= threshold * 3) {
					visited[ny][nx] = true;
					q.Add(Point(nx, ny));
				}
			}
		}
	}
	
	if(q.IsEmpty()) return result;
	
	// Trace full resolution mask
	auto res = TraceMask(mask, Rectf(0, 0, sz.cx, sz.cy), 1.0);
	for(auto& p : res) p = SimplifyPolygonRDP(p, 1.0);
	return res;
}