#include "AirHockey.h"

namespace GameCtrl {
using namespace Upp;

World::World() : world(b2Vec2(0.0, -10.0), false), world_draw(this) {
	draw_mode = DRAW_DRAW;
	dbg_showBoxes = false;
	mouseJoint = NULL;
	draw_debug = false;
	
	SetTimeCallback(-1, THISBACK(Render));
	SetGravityZero();
	
	world.SetDebugDraw(&debugDraw);
	groundBody = world.CreateBody(&bodyDef);
	
	ts.Reset();
}

void World::Paint(Draw& w) {
	float hz = 60;
	int velocityIterations = 8;
	int positionIterations = 10;

	float32 timeStep = ts.Elapsed() / 1000.0;
	ts.Reset();
	
	int flags = b2DebugDraw::e_shapeBit | b2DebugDraw::e_jointBit;
	if(dbg_showBoxes)
		flags |= b2DebugDraw::e_aabbBit;
	
	debugDraw.SetFlags(flags);
	
	world.SetWarmStarting(1);
	world.SetContinuousPhysics(1);
	world.Step(timeStep, velocityIterations, positionIterations);

	Point p1, p2;
	if(mouseJoint)
	{
		p1 = debugDraw.conv(mouseJoint->GetAnchorB());
		p2 = debugDraw.conv(mouseJoint->GetTarget());
	}

	Size sz = GetSize();
					
	if(draw_mode > 0)
	{
		ImageBuffer ib(sz);
		BufferPainter bp(ib, draw_mode == 1 ? MODE_NOAA : draw_mode == 2 ? MODE_ANTIALIASED : MODE_SUBPIXEL);
		RGBA bg;
		bg.r = bg.g = bg.b = bg.a = 255;
		bp.Clear(bg);
		if (draw_debug) {
			debugDraw.Init(bp, sz);
			world.DrawDebugData();
		} else {
			world_draw.Init(bp, sz);
			world_draw.DrawData();
		}
		
		if (mouseJoint) {
			bp.DrawLine(p1, p2, 2, LtGreen);
			bp.DrawEllipse(p2.x - 3, p2.y - 3, 6, 6, Green, PEN_SOLID, Black);
		}
		
		w.DrawImage(0, 0, ib);
	}
	else
	{
		w.DrawRect(sz, White);
		if (draw_debug) {
			debugDraw.Init(w, sz);
			world.DrawDebugData();
		} else {
			world_draw.Init(w, sz);
			world_draw.DrawData();
		}
		
		if (mouseJoint) {
			w.DrawLine(p1, p2, 2, LtGreen);
			w.DrawEllipse(p2.x - 3, p2.y - 3, 6, 6, Green, PEN_SOLID, Black);
		}
	}
}

void World::LeftDown(Point p0, dword keyflags) {
	b2Vec2 p = debugDraw.conv(p0);
	mouseWorld = p;
	
	if(mouseJoint != NULL)
		return;

	b2AABB aabb;
	b2Vec2 d;
	d.Set(0.001f, 0.001f);
	aabb.lowerBound = p - d;
	aabb.upperBound = p + d;

	QueryCallback callback(p);
	world.QueryAABB(&callback, aabb);

	if (callback.fixture)
	{
		b2Body* body = callback.fixture->GetBody();
		b2MouseJointDef md;
		md.bodyA = groundBody;
		md.bodyB = body;
		md.target = p;
		md.maxForce = 1000.0f * body->GetMass();
		mouseJoint = (b2MouseJoint*) world.CreateJoint(&md);
		body->SetAwake(true);
	}
}

void World::LeftUp(Point p0, dword keyflags) {
	if (mouseJoint) {
		world.DestroyJoint(mouseJoint);
		mouseJoint = NULL;
	}
}

void World::MouseMove(Point p, dword keyflags) {
	mouseWorld = debugDraw.conv(p);
	
	if (mouseJoint)
		mouseJoint->SetTarget(mouseWorld);
}

void World::MouseWheel(Point p, int zdelta, dword keyflags) {
	debugDraw.zoom += zdelta / 80.0f;
}

void World::Add(Object& obj) {
	obj.SetWorld(this);
	obj_list.Add((long)&obj, &obj);
}

void World::Remove(Object& obj) {
	if (obj.body) {
		world.DestroyBody(obj.body);
		obj.SetWorld(0);
	}
	int i = obj_list.Find((long)&obj);
	if (i != -1)
		obj_list.Remove(i);
}

void World::SetContactListener(ContactListener& cl) {
	world.SetContactListener(&cl);
}

}
