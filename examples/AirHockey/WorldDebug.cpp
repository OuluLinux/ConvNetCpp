#include "AirHockey.h"

namespace GameCtrl {
using namespace Upp;

WorldDebug::WorldDebug() {
	xoff = 0;
	yoff = 0;
}

void WorldDebug::Init(Draw& d, Size s)
{
	w = &d;
	sz = s;
	zoom = 15.0f;
	cx = float(sz.cx / 2.0f + xoff);
	cy = float(sz.cy / 2.0f + yoff);
	aspect = float(sz.cx / sz.cy);
	aspect *= zoom;
}

Point WorldDebug::conv(const b2Vec2& v)
{
	return Point(int(v.x * aspect + cx), int(cy - v.y * aspect));
}

b2Vec2 WorldDebug::conv(const Point& p)
{
	b2Vec2 v;

	v.x = (p.x - cx) / aspect;
	v.y = (cy - p.y) / aspect;
	
	return v;
}

Color WorldDebug::conv(const b2Color& c, double f)
{
	return Color(int(c.r * f), int(c.g * f), int(c.b * f));
}

void WorldDebug::DrawPolygon(const b2Vec2* v, int vertexCount, const b2Color& color)
{
	Vector<Point> p;
	p.SetCount(vertexCount + 1);
	for(int i = 0; i < vertexCount; ++i)
		p[i] = conv(v[i]);
	p[vertexCount] = p[0];
	
	w->DrawPolyline(p, 1, conv(color, 150.0));
}

void WorldDebug::DrawSolidPolygon(const b2Vec2* v, int vertexCount, const b2Color& color)
{
	Vector<Point> p;
	p.SetCount(vertexCount);
	for(int i = 0; i < vertexCount; ++i)
		p[i] = conv(v[i]);
	
	w->DrawPolygon(p, conv(color, 255.0), 1, conv(color, 150));		
}

void WorldDebug::DrawCircle(const b2Vec2& center, float32 radius, const b2Color& color)
{
	int r = int(aspect * radius * 2.0f);
	Point p = conv(b2Vec2(center.x - radius, center.y + radius));
	w->DrawEllipse(p.x, p.y, r, r, conv(color, 150.0));
}

void WorldDebug::DrawSolidCircle(const b2Vec2& center, float32 radius, const b2Vec2& axis, const b2Color& color)
{
	int r = int(aspect * radius * 2.0f);
	Point p = conv(b2Vec2(center.x - radius, center.y + radius));
	w->DrawEllipse(p.x, p.y, r, r, conv(color, 255.0), PEN_SOLID, conv(color, 150.0));
}

void WorldDebug::DrawSegment(const b2Vec2& p1, const b2Vec2& p2, const b2Color& color)
{
	w->DrawLine(conv(p1), conv(p2), 1, conv(color, 150.0));
}

void WorldDebug::DrawTransform(const b2Transform& xf) {
	
}

void WorldDebug::DrawPoint(const b2Vec2& p0, float32 size, const b2Color& color)
{
	Point p = conv(p0);
	int s = int(size * aspect);
	w->DrawRect(p.x, p.y, s, s, conv(color, 255.0));
}

void WorldDebug::DrawString(int x, int y, const char* string, ...)
{
	char buffer[256];

	va_list arg;
	va_start(arg, string);
	vsprintf(buffer, string, arg);
	va_end(arg);
	
	w->DrawText(x, y, buffer);
}

void WorldDebug::DrawAABB(b2AABB* aabb, const b2Color& color)
{
	Point lb = conv(aabb->lowerBound);
	Point ub = conv(aabb->upperBound);
	Color fg = conv(color, 150.0);
			
	w->DrawRect(lb.x, lb.y, ub.x, ub.y, fg);
}

}
