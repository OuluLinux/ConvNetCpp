#include "WaterWorld.h"

World::World() {
	W = 700;
	H = 500;
	clock = 0;
	AddBox(walls, 0, 0, W, H);

	// set up food and poison
	for(int k=0; k < 50; k++) {
		double x = RandomRange(20, W - 20);
		double y = RandomRange(20, H - 20);
		int t = RandomRangeInt(1, 3); // food or poison (1 and 2)
		items.Add().Init(x, y, t);
	}
	
	// Add agent
	WaterWorldAgent& agent = agents.Add();
	
}

InterceptResult World::StuffCollide(Pointf p1, Pointf p2, bool check_walls, bool check_items) {
	InterceptResult minres(false);
	
	// collide with walls
	if (check_walls) {
		for(int i = 0, n = walls.GetCount(); i < n; i++) {
			Wall& wall = walls[i];
			InterceptResult res = IsLineIntersect(p1, p2, wall.p1, wall.p2);
			if (res) {
				res.type = 0; // 0 is wall
				if (!minres) {
					minres=res;
				}
				// check if its closer
				else if(res.ua < minres.ua) {
					// if yes replace it
					minres = res;
				}
			}
		}
	}
	
	// collide with items
	if(check_items) {
		for(int i = 0, n = items.GetCount(); i < n; i++) {
			Item& it = items[i];
			InterceptResult res = IsLinePointIntersect(p1, p2, it.p, it.rad);
			if(res) {
				res.type = it.type; // store type of item
				res.vx = it.v.x;
				res.vy = it.v.y;
				if (!minres) {
					minres=res;
				}
				else if(res.ua < minres.ua) {
					minres = res;
				}
			}
		}
	}
	
	return minres;
}

void World::Tick() {
	// tick the environment
	clock++;
	
	// fix input to all agents based on environment
	// process eyes
	
	for (int i = 0, n = agents.GetCount(); i < n; i++) {
		WaterWorldAgent& a = agents[i];
		for(int ei = 0, ne = a.eyes.GetCount(); ei < ne; ei++) {
			Eye& e = a.eyes[ei];
			
			// we have a line from p to p->eyep
			double angle = e.angle;
			Pointf eyep(
				a.p.x + e.max_range * sin(angle),
				a.p.y + e.max_range * cos(angle));
			InterceptResult res = StuffCollide(a.p, eyep, true, true);
			if(res) {
				// eye collided with wall
				e.sensed_proximity = Distance(res.up, a.p);
				e.sensed_type = res.type;
				e.vx = res.vx;
				e.vy = res.vy;
			} else {
				e.sensed_proximity = e.max_range;
				e.sensed_type = -1;
				e.vx = 0;
				e.vy = 0;
			}
		}
	}
	
	// let the agents behave in the world based on their input
	for (int i = 0, n = agents.GetCount(); i < n; i++) {
		agents[i].Forward();
	}
	
	// Reset digestion signal
	for (int j = 0; j < agents.GetCount(); j++)
		agents[j].digestion_signal = 0;
	
	int border = min(W, H) * 0.1;
	
	// apply outputs of agents on evironment
	for (int i = 0, n = agents.GetCount(); i < n; i++) {
		WaterWorldAgent& a = agents[i];
		a.op = a.p; // back up old position
		a.tail.Add(a.p);
		while (a.tail.GetCount() > a.max_tail)
			a.tail.Remove(0);
		
		// execute agent's desired action
		double speed = 1;
		if(a.action == ACT_LEFT) {
			a.v.x += -speed;
		}
		else if(a.action == ACT_RIGHT) {
			a.v.x += speed;
		}
		else if(a.action == ACT_UP) {
			a.v.y += -speed;
		}
		else if(a.action == ACT_DOWN) {
			a.v.y += speed;
		}
		
		// forward the agent by velocity
		a.v.x *= 0.95; a.v.y *= 0.95;
		a.p.x += a.v.x; a.p.y += a.v.y;
		
		// handle boundary conditions.. bounce agent
		if (a.p.x < 1)		{ a.p.x = 1;	a.v.x=0;	a.v.y=0;}
		if (a.p.x > W-1)	{ a.p.x = W-1;	a.v.x=0;	a.v.y=0;}
		if (a.p.y < 1)		{ a.p.y = 1;	a.v.x=0;	a.v.y=0;}
		if (a.p.y > H-1)	{ a.p.y = H-1;	a.v.x=0;	a.v.y=0;}
	}
	
	// tick all items
	bool update_items = false;
	
	for (int i = 0, n = items.GetCount(); i < n; i++) {
		Item& it = items[i];
		it.age += 1;
		
		// see if some agent gets lunch
		for (int j = 0, m = agents.GetCount(); j < m; j++) {
			WaterWorldAgent& a = agents[j];
			double d = Distance(a.p, it.p);
			if (d < it.rad + a.rad) {
				
				// ding! nom nom nom
				if (it.type == 1)
					a.digestion_signal += 1.0; // mmm delicious apple
				if (it.type == 2)
					a.digestion_signal += -1.0; // ewww poison
				it.cleanup_ = true;
				update_items = true;
				break; // break out of loop, item was consumed
			
			}
		}
		
		// move the items
		it.p.x += it.v.x;
		it.p.y += it.v.y;
		if (it.p.x < 1) { it.p.x = 1; it.v.x *= -1; }
		if (it.p.x > W-1) { it.p.x = W-1; it.v.x *= -1; }
		if (it.p.y < 1) { it.p.y = 1; it.v.y *= -1; }
		if (it.p.y > H-1) { it.p.y = H-1; it.v.y *= -1; }
		
		if (it.age > 5000 && (clock % 100) == 0 && Randomf() < 0.1) {
			it.cleanup_ = true; // replace this one, has been around too long
			update_items = true;
		}
	}
	if (update_items) {
		for(int i = 0; i < items.GetCount(); i++) {
			if (items[i].cleanup_) {
				items.Remove(i);
				i--;
			}
		}
	}
	if (items.GetCount() < 50 && (clock % 10) == 0 && Randomf() < 0.25) {
		double newitx = RandomRange(20, W - 20);
		double newity = RandomRange(20, H - 20);
		int newitt = RandomRangeInt(1, 3); // food or poison (1 and 2)
		items.Add().Init(newitx, newity, newitt);
	}
	
	// agents are given the opportunity to learn based on feedback of their action on environment
	for (int i = 0, n = agents.GetCount(); i < n; i++) {
		agents[i].Backward();
	}
}

void World::Paint(Draw& d) {
	
	Size sz = GetSize();
	
	ImageDraw id(sz);
	id.DrawRect(sz, White());
	
	
	// draw walls in environment
	Color wall_clr = GrayColor(64);
	for(int i = 0; i < walls.GetCount(); i++) {
		Wall& w = walls[i];
		id.DrawLine(w.p1, w.p2, 1, wall_clr);
	}
	
	// draw agents
	Color apple_clr(255,150,150);
	Color poison_clr(150,255,150);
	for(int i = 0; i < agents.GetCount(); i++) {
		WaterWorldAgent& a = agents[i];
		
		// color agent based on reward it is experiencing at the moment
		int r = min(255, max(0, (int)(a.reward * 200.0)));
		Color fill_clr(r, 150, 150);
		
		// draw tail (not in original);
		Pointf* prev = &a.op;
		for(int j = 0; j < a.tail.GetCount(); j++) {
			double radius = a.rad * ((double)(j + 1) / a.tail.GetCount() + 1.0)  / 2.0;
			double radius2 = radius * 2.0;
			Pointf& p = a.tail[j];
			id.DrawEllipse(p.x - radius, p.y - radius, radius2, radius2, fill_clr, 1, fill_clr);
			prev = &p;
		}
		
		// draw agents sight
		for(int j = 0; j < a.eyes.GetCount(); j++) {
			Eye& e = a.eyes[j];
			double sr = e.sensed_proximity;
			Color line_clr;
			if(e.sensed_type == -1 || e.sensed_type == 0)
				line_clr = Black(); // wall or nothing
			else if (e.sensed_type == 1)
				line_clr = apple_clr; // apples
			else if(e.sensed_type == 2)
				line_clr = poison_clr; // poison
			double angle = e.angle;
			Pointf b(
				a.op.x + sr * sin(angle),
				a.op.y + sr * cos(angle));
			id.DrawLine(a.op, b, 1, line_clr);
		}
		
		// draw agents body
		double radius = a.rad;
		double radius2 = radius * 2.0;
		id.DrawEllipse(a.op.x - radius, a.op.y - radius, radius2, radius2, fill_clr, 0);
		
	}
	
	// draw items
	for(int i = 0; i < items.GetCount(); i++) {
		Item& it = items[i];
		Color fill_clr;
		if      (it.type == 1) fill_clr = Color(255, 150, 150);
		else if (it.type == 2) fill_clr = Color(150, 255, 150);
		else                   fill_clr = Color(150, 150, 255);
		double radius = it.rad;
		double radius2 = radius * 2.0;
		id.DrawEllipse(it.p.x - radius, it.p.y - radius, radius2, radius2, fill_clr, 1, Black());
		
	}
	
	d.DrawImage(0, 0, id);
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

void AddBox(Vector<Wall>& lst, int x, int y, int w, int h) {
	lst.Add(Wall(Point(x,y),		Point(x+w,y)));
	lst.Add(Wall(Point(x+w,y),		Point(x+w,y+h+1)));
	lst.Add(Wall(Point(x+w,y+h),	Point(x,y+h)));
	lst.Add(Wall(Point(x,y+h),		Point(x,y)));
}



