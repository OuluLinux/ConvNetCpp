#include "ReinforcedLearning.h"

World::World() {
	W = 700;
	H = 500;
	pad = 10;
	clock = 0;
	AddBox(walls, pad, pad, W-pad*2, H-pad*2);
	AddBox(walls, 100, 100, 200, 300); // inner walls
	walls.Remove(walls.GetCount()-1); // remove last
	AddBox(walls, 400, 100, 200, 300);
	walls.Remove(walls.GetCount()-1); // remove last

	// set up food and poison
	for(int k=0; k<30; k++) {
		double x = RandomRange(20, W - 20);
		double y = RandomRange(20, H - 20);
		int t = RandomRangeInt(1, 3); // food or poison (1 and 2)
		items.Add().Init(x, y, t);
	}
	
	// Add agent
	RLAgent& agent = agents.Add();
	
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
		RLAgent& a = agents[i];
		for(int ei = 0, ne = a.eyes.GetCount(); ei < ne; ei++) {
			Eye& e = a.eyes[ei];
			
			// we have a line from p to p->eyep
			double angle = a.angle + e.angle;
			Pointf eyep(
				a.p.x + e.max_range * sin(angle),
				a.p.y + e.max_range * cos(angle));
			InterceptResult res = StuffCollide(a.p, eyep, true, true);
			if(res) {
				// eye collided with wall
				e.sensed_proximity = Distance(res.up, a.p);
				e.sensed_type = res.type;
			} else {
				e.sensed_proximity = e.max_range;
				e.sensed_type = -1;
			}
		}
	}
	
	// let the agents behave in the world based on their input
	for (int i = 0, n = agents.GetCount(); i < n; i++) {
		agents[i].Forward();
	}
	
	// apply outputs of agents on evironment
	for (int i = 0, n = agents.GetCount(); i < n; i++) {
		RLAgent& a = agents[i];
		a.op = a.p; // back up old position
		a.oangle = a.angle; // and angle
		
		// steer the agent according to outputs of wheel velocities
		Pointf v(0, a.rad / 2.0);
		v = Rotate(v, a.angle + M_PI/2);
		Pointf w1p = a.p + v; // positions of wheel 1 and 2
		Pointf w2p = a.p - v;
		Pointf vv = a.p - w2p;
		vv = Rotate(vv, -a.rot1);
		Pointf  vv2 = a.p - w1p;
		vv2 = Rotate(vv2, a.rot2);
		Pointf  np = w2p + vv;
		Scale(np, 0.5);
		Pointf  np2 = w1p + vv2;
		Scale(np2, 0.5);
		a.p = np + np2;
		
		a.angle -= a.rot1;
		if (a.angle < 0)
			a.angle+=2*M_PI;
		
		a.angle += a.rot2;
		if (a.angle > 2*M_PI)
			a.angle-=2*M_PI;
		
		// agent is trying to move from p to op. Check walls
		InterceptResult res = StuffCollide(a.op, a.p, true, false);
		if(res) {
			// wall collision! reset position
			a.p = a.op;
		}
		
		// handle boundary conditions
		if (a.p.x < 0)	a.p.x = 0;
		if (a.p.x > W)	a.p.x = W;
		if (a.p.y < 0)	a.p.y = 0;
		if (a.p.y > H)	a.p.y = H;
	}
	
	// tick all items
	bool update_items = false;
	for (int i = 0, n = items.GetCount(); i < n; i++) {
		Item& it = items[i];
		it.age += 1;
		
		// see if some agent gets lunch
		for (int j = 0, m = agents.GetCount(); j < m; j++) {
			RLAgent& a = agents[j];
			double d = Distance(a.p, it.p);
			if (d < it.rad + a.rad) {
				
				// wait lets just make sure that this isn't through a wall
				InterceptResult rescheck = StuffCollide(a.p, it.p, true, false);
				if (!rescheck) {
					// ding! nom nom nom
					if (it.type == 1)
						a.digestion_signal += 5.0; // mmm delicious apple
					if (it.type == 2)
						a.digestion_signal += -6.0; // ewww poison
					it.cleanup_ = true;
					update_items = true;
					break; // break out of loop, item was consumed
				}
			}
		}
		
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
	if (items.GetCount() < 30 && (clock % 10) == 0 && Randomf() < 0.25) {
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
		RLAgent& a = agents[i];
		
		// color agent based on reward it is experiencing at the moment
		int r = min(255, max(0, (int)(a.brain.GetLatestReward() * 200.0)));
		Color fill_clr(r, 150, 150);
		
		// draw agents body
		double radius = a.rad;
		double radius2 = radius * 2.0;
		id.DrawEllipse(a.op.x - radius, a.op.y - radius, radius2, radius2, fill_clr, 1, Black());
		
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
			double angle = a.oangle + e.angle;
			Pointf b(
				a.op.x + sr * sin(angle),
				a.op.y + sr * cos(angle));
			id.DrawLine(a.op, b, 1, line_clr);
		}
		
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
