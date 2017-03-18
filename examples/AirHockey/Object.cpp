#include "AirHockey.h"

namespace GameCtrl {
using namespace Upp;

Object::Object() : world(0), fix(0), body(0) {
	SetDensity(1.0);
	
}

void Object::SetCategory(int b16_pos, bool value) {
	if (value)
		fd.filter.categoryBits |= 1 << b16_pos;
	else
		fd.filter.categoryBits &= ~(1 << b16_pos);
	if (fix) {
		b2Filter filter = fix->GetFilterData();
		if (value)
			filter.categoryBits |= 1 << b16_pos;
		else
			filter.categoryBits &= ~(1 << b16_pos);
		fix->SetFilterData(filter);
	}
}

void Object::SetCollisionFilter(int b16_pos, bool value) {
	if (!value)
		fd.filter.maskBits |= 1 << b16_pos;
	else
		fd.filter.maskBits &= ~(1 << b16_pos);
	if (fix) {
		b2Filter filter = fix->GetFilterData();
		if (!value)
			filter.maskBits |= 1 << b16_pos;
		else
			filter.maskBits &= ~(1 << b16_pos);
		fix->SetFilterData(filter);
	}
}

void Object::Create() {
	ASSERT(this->world);
	b2World& world = this->world->B2_GetWorld();
	body = world.CreateBody(&bd);
	fix = body->CreateFixture(&fd);
	
	body->SetUserData(this);
	fix->SetUserData(this);
}

bool Object::IsColliding(const Object& o) const {
	bool a = (o.fd.filter.maskBits & fd.filter.categoryBits) != 0;
	bool b = (o.fd.filter.categoryBits & fd.filter.maskBits) != 0;
	bool collide = a && b;
	return collide;
}




Circle::Circle() {
	fd.shape = &shape;
	
	SetRadius(0.5);
	
}

void Circle::PaintCircle(WorldDraw& wdraw, Draw& draw, Color fill_color, Color border_color) {
	double aspect = wdraw.GetAspect();
	double radius = GetRadius();
	Pointf center = GetPosition();
	int r = int(aspect * radius * 2.0);
	Point p = wdraw.ToScreen(center.x - radius, center.y + radius);
	draw.DrawEllipse(p.x, p.y, r, r, fill_color, PEN_SOLID, border_color);
}


Polygon::Polygon() {
	fd.shape = &shape;
	
}

void Polygon::Create() {
	shape.Set(vertices.Begin(), vertices.GetCount());
	
	Object::Create();
}

void Polygon::PaintPolyline(WorldDraw& wdraw, Draw& draw, Color color, int width) {
	Vector<Point> p;
	int vertexCount = vertices.GetCount();
	p.SetCount(vertexCount + 1);
	for(int i = 0; i < vertexCount; ++i)
		p[i] = wdraw.ToScreen(vertices[i]);
	p[vertexCount] = p[0];
	draw.DrawPolyline(p, width, color);
}

void Polygon::PaintPolygon(WorldDraw& wdraw, Draw& draw, Color fill_color, int width, Color border_color) {
	Vector<Point> p;
	int vertexCount = vertices.GetCount();
	p.SetCount(vertexCount + 1);
	for(int i = 0; i < vertexCount; ++i)
		p[i] = wdraw.ToScreen(vertices[i]);
	p[vertexCount] = p[0];
	draw.DrawPolygon(p, fill_color, width, fill_color);
}


}
