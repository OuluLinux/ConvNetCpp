#include "AirHockey.h"

namespace GameCtrl {
using namespace Upp;

WorldDraw::WorldDraw(void* world) : world(world) {
	xoff = 0;
	yoff = 0;
}

void WorldDraw::Init(Draw& d, Size s)
{
	w = &d;
	sz = s;
	zoom = 15.0;
	cx = sz.cx / 2.0 + xoff;
	cy = sz.cy / 2.0 + yoff;
	if (sz.cy < sz.cx)
		aspect = sz.cy / sz.cx;
	else
		aspect = sz.cx / sz.cy;
	aspect *= zoom;
}

Pointf WorldDraw::ToScreen(const b2Vec2& v) {
	return Pointf(v.x * aspect + cx, cy - v.y * aspect);
}

Pointf WorldDraw::ToScreen(const Pointf& p) {
	return Pointf(p.x * aspect + cx, cy - p.y * aspect);
}

Pointf WorldDraw::ToScreen(double x, double y) {
	return Pointf(x * aspect + cx, cy - y * aspect);
}

b2Vec2 WorldDraw::FromScreen(const Pointf& p)
{
	b2Vec2 v;

	v.x = (p.x - cx) / aspect;
	v.y = (cy - p.y) / aspect;
	
	return v;
}

void WorldDraw::DrawData() {
	World* world = (World*)this->world;
	for(int i = 0; i < world->obj_list.GetCount(); i++) {
		Object& obj = *world->obj_list[i];
		obj.Paint(*this, *w);
	}
}

}
