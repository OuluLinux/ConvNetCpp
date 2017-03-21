#include "AirHockey.h"

namespace GameCtrl {
using namespace Upp;

World::World() : world(b2Vec2(0.0, -10.0), false), world_draw(this) {
	draw_mode = DRAW_DRAW;
	dbg_showBoxes = false;
	mouseJoint = NULL;
	draw_debug = false;
	running = false;
	stopped = true;
	
	SetGravityZero();
	
	world.SetDebugDraw(&debugDraw);
	groundBody = world.CreateBody(&bodyDef);
	
	sim_speed = 0;
	tick_interval = 10; // ms to sleep between ticks in realtime
	
}

World::~World() {
	Stop();
}

void World::Start() {
	Stop();
	running = true;
	stopped = false;
	Thread::Start(THISBACK(Ticking));
}

void World::Stop() {
	running = false;
	while (!stopped) Sleep(100);
}

void World::Tick() {
	
	int velocityIterations = 8;
	int positionIterations = 10;

	// Real time
	if (sim_speed == false) {
		int elapsed = ts.Elapsed();
		ts.Reset();
		world.Step(elapsed / 1000.0, velocityIterations, positionIterations);
	}
	else {
		world.Step(tick_interval / 1000.0, velocityIterations, positionIterations);
	}
}

void World::Ticking() {
	
	while (running) {
		lock.Enter();
		Tick();
		lock.Leave();
	}
	
	stopped = true;
}

void World::Paint(Draw& w) {
	
	int flags = b2DebugDraw::e_shapeBit | b2DebugDraw::e_jointBit;
	if(dbg_showBoxes)
		flags |= b2DebugDraw::e_aabbBit;
	
	debugDraw.SetFlags(flags);
	
	world.SetWarmStarting(1);
	world.SetContinuousPhysics(1);
	
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
			
			lock.Enter();
			world.DrawDebugData();
			lock.Leave();
		} else {
			world_draw.Init(bp, sz);
			
			lock.Enter();
			world_draw.DrawData();
			lock.Leave();
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
			
			lock.Enter();
			world.DrawDebugData();
			lock.Leave();
		} else {
			world_draw.Init(w, sz);
			
			lock.Enter();
			world_draw.DrawData();
			lock.Leave();
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

void World::SetSpeed(bool simulate_speed, int interval) {
	this->sim_speed = simulate_speed;
	this->tick_interval = interval;
}


























// line intersection helper function: does line segment (l1a,l1b) intersect segment (l2a,l2b) ?
InterceptResult IsLineIntersect(Pointf l1a, Pointf l1b, Pointf l2a, Pointf l2b) {
	double denom = (l2b.y - l2a.y) * (l1b.x - l1a.x) - (l2b.x - l2a.x) * (l1b.y - l1a.y);
	if (denom == 0.0)
		return InterceptResult(false); // parallel lines
	double ua = ((l2b.x-l2a.x)*(l1a.y-l2a.y)-(l2b.y-l2a.y)*(l1a.x-l2a.x))/denom;
	double ub = ((l1b.x-l1a.x)*(l1a.y-l2a.y)-(l1b.y-l1a.y)*(l1a.x-l2a.x))/denom;
	if (ua > 0.0 && ua<1.0 && ub > 0.0 && ub < 1.0) {
		Pointf up(l1a.x+ua*(l1b.x-l1a.x), l1a.y+ua*(l1b.y-l1a.y));
		InterceptResult res;
		res.ua = ua;
		res.ub = ub;
		res.up = up;
		res.is_intercepting = true;
		return res;
	}
	return InterceptResult(false);
}

InterceptResult IsLinePointIntersect(Pointf a, Pointf b, Pointf p, int rad) {
	Pointf v(b.y-a.y,-(b.x-a.x)); // perpendicular vector
	double d = fabs((b.x-a.x)*(a.y-p.y)-(a.x-p.x)*(b.y-a.y));
	d = d / Length(v);
	if (d > rad)
		return false;
	
	Normalize(v);
	Scale(v, d);
	Pointf up = p + v;
	double ua;
	if (fabs(b.x-a.x) > fabs(b.y-a.y)) {
		ua = (up.x - a.x) / (b.x - a.x);
	}
	else {
		ua = (up.y - a.y) / (b.y - a.y);
	}
	if (ua > 0.0 && ua < 1.0) {
		InterceptResult ir;
		ir.up = up;
		ir.ua = ua;
		ir.up = up;
		ir.is_intercepting = true;
		return ir;
	}
	return false;
}

}
