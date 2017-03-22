#include "AirHockey.h"

namespace GameCtrl {
// Collision filter masks (bitwise)
//  - player A and area B collides, but not player A and area A
const int game_objects = 1;
const int player_a = 2;
const int player_b = 3;


Map::Map() {
	last_won = -1;
	last_score = -1;
}

void Map::PlayerWon(int i) {
	last_score = -1;
	last_won = i;
}

void Map::PlayerScore(int i) {
	last_won = -1;
	last_score = i;
}

void Map::Paint(WorldDraw& wdraw, Draw& draw) {
	Pointf tl(-w, h);
	Pointf tr(+w, h);
	Pointf br(+w, -h);
	Pointf bl(-w, -h);
	Pointf tli(-gw, h);
	Pointf tri(+gw, h);
	Pointf bli(-gw, -h);
	Pointf bri(+gw, -h);
	
	Color line(128,0,0);
	
	// Map outline
	{
		Vector<Point> p;
		double m = 1.01;
		p.Add(wdraw.ToScreen(tl));
		p.Add(wdraw.ToScreen(tli));
		p.Add(wdraw.ToScreen(tli * m));
		p.Add(wdraw.ToScreen(tri * m));
		p.Add(wdraw.ToScreen(tri));
		p.Add(wdraw.ToScreen(tr));
		p.Add(wdraw.ToScreen(br));
		p.Add(wdraw.ToScreen(bri));
		p.Add(wdraw.ToScreen(bri * m));
		p.Add(wdraw.ToScreen(bli * m));
		p.Add(wdraw.ToScreen(bli));
		p.Add(wdraw.ToScreen(bl));
		p.Add(wdraw.ToScreen(tl));
		draw.DrawPolyline(p, 2, GrayColor());
	}
	
	// Top goal helper line
	{
		Vector<Point> p;
		double m = 0.7;
		p.Add(wdraw.ToScreen(tli));
		p.Add(wdraw.ToScreen(tli * m));
		p.Add(wdraw.ToScreen(tri * m));
		p.Add(wdraw.ToScreen(tri));
		draw.DrawPolyline(p, 2, line);
	}
	
	// Bottom goal helper line
	{
		Vector<Point> p;
		double m = 0.7;
		p.Add(wdraw.ToScreen(bli));
		p.Add(wdraw.ToScreen(bli * m));
		p.Add(wdraw.ToScreen(bri * m));
		p.Add(wdraw.ToScreen(bri));
		draw.DrawPolyline(p, 2, line);
	}
	
	// Center helper line
	{
		Vector<Point> p;
		double m = 0.7;
		p.Add(wdraw.ToScreen(Pointf(-w, 0)));
		p.Add(wdraw.ToScreen(Pointf(+w, 0)));
		draw.DrawPolyline(p, 2, line);
	}
	
	// Center circle
	{
		double aspect = wdraw.GetAspect();
		double radius = 5;
		Pointf center(0, 0);
		int r = int(aspect * radius * 2.0);
		Point p = wdraw.ToScreen(center.x - radius, center.y + radius);
		draw.DrawEllipse(p.x, p.y, r, r, White(), 2, line);
	}
	
	
	// Draw texts
	{
		Table2* table = (Table2*)this->table;
		String s;
		if (last_won != -1) {
			if (last_won == 0)
				s << "Upper won!";
			else
				s << "Lower won!";
		}
		if (last_score != -1) {
			if (last_score == 0)
				s << "Upper scored!";
			else
				s << "Lower scored!";
		}
		Font fnt = SansSerifZ(8);
		Size text_sz = GetTextSize(s, fnt);
		Pointf center = wdraw.ToScreen(0, 0);
		draw.DrawText(
			center.x - text_sz.cx / 2,
			center.y - text_sz.cy / 2,
			s, fnt, Green());
	
		
		fnt = SansSerifZ(15);
		
		
		// Draw player A score
		s = IntStr(table->GetScoreA());
		text_sz = GetTextSize(s, fnt);
		center = wdraw.ToScreen(0, -22);
		draw.DrawText(
			center.x - text_sz.cx / 2,
			center.y - text_sz.cy / 2,
			s, fnt, Green());
		
		
		// Draw player B score
		s = IntStr(table->GetScoreB());
		text_sz = GetTextSize(s, fnt);
		center = wdraw.ToScreen(0, +22);
		draw.DrawText(
			center.x - text_sz.cx / 2,
			center.y - text_sz.cy / 2,
			s, fnt, Green());
		
	}
	
}

void Puck::Paint(WorldDraw& wdraw, Draw& draw) {
	Color fill_color = Red();
	Color border_color = Black();
	PaintCircle(wdraw, draw, fill_color, border_color);
}




















Table2::Table2() {
	agents.Add().SetId(0);
	agents.Add().SetId(1);
	score_limit = 3;
	score[0] = 0;
	score[1] = 0;
	area_a.table = this;
	debug_paint = false;
}

void Table2::ResetGame() {
	score[0] = 0;
	score[1] = 0;
	player_a_starts = true;
	ResetPuck();
}

void Table2::Reset() {
	ResetGame();
	
	int states = 152; // count of eyes
	int action_count = 4+1; // directions and idle
	for(int i = 0; i < agents.GetCount(); i++) {
		Player& agent = agents[i];
		agent.Init(1, states, 1, action_count);
		agent.Reset();
	}
}

void Table2::PlayerScore(int i) {
	lock.Enter();
	
	player_a_starts = i == 0;
	score[i]++;
	agents[0].game_score += i == 0 ? +1.0 : -1.0;
	agents[1].game_score += i == 0 ? -1.0 : +1.0;
	if (score[i] >= score_limit) {
		ResetGame();
		area_a.PlayerWon(i);
		WhenFinish(i);
	}
	else {
		ResetPuck();
		area_a.PlayerScore(i);
		WhenScore(i);
	}
	
	lock.Leave();
}

void Table2::Init() {
	Reset();
	
	Player& pl_a = agents[0];
	Player& pl_b = agents[1];
	
	// Key points of the map
	int w = 16, gw = 9, h = 16+8;
	area_a.w = w;
	area_a.h = h;
	area_a.gw = gw;
	Pointf tl(-w, h);
	Pointf tr(+w, h);
	Pointf br(+w, -h);
	Pointf bl(-w, -h);
	Pointf tli(-gw, h);
	Pointf tri(+gw, h);
	Pointf bli(-gw, -h);
	Pointf bri(+gw, -h);
	
	
	// Add map borders
	Add(map_l);
	map_l << bl << tl << tl*1.5 << bl*1.5;
	map_l.SetCategory(game_objects, true);
	map_l.Create();
	
	Add(map_tl);
	map_tl << tl << tli << tli*1.5 << tl*1.5;
	map_tl.SetCategory(game_objects, true);
	map_tl.Create();
	
	Add(map_tr);
	map_tr << tr << tr*1.5 << tri*1.5 << tri;
	map_tr.SetCategory(game_objects, true);
	map_tr.Create();
	
	Add(map_r);
	map_r << tr << br << br*1.5 << tr*1.5;
	map_r.SetCategory(game_objects, true);
	map_r.Create();
	
	Add(map_bl);
	map_bl << bli << bl << bl*1.5 << bli*1.5;
	map_bl.SetCategory(game_objects, true);
	map_bl.Create();
	
	Add(map_br);
	map_br << br << bri << bri*1.5 << br*1.5;
	map_br.SetCategory(game_objects, true);
	map_br.Create();
	
	
	// Add player areas
	Add(area_a);
	area_a << Pointf(-w, 0) << Pointf(w, 0) << tr << tl;
	area_a.SetCategory(game_objects, true);
	area_a.FilterAllCollision();
	area_a.SetCollisionFilter(player_b, false);
	area_a.Create();
	
	Add(area_b);
	area_b << Pointf(w, 0) << Pointf(-w, 0) << bl << br;
	area_b.SetCategory(game_objects, true);
	area_b.FilterAllCollision();
	area_b.SetCollisionFilter(player_a, false);
	area_b.Create();
	
	
	// Add players
	Add(pl_a);
	pl_a.SetRadius(3);
	pl_a.SetPosition(0, 16);
	pl_a.SetTypeDynamic();
	pl_a.SetRestitution(0.0);
	pl_a.SetFriction(0.5);
	pl_a.Create();
	pl_a.SetName("Dick");
	pl_a.SetCategory(0, false);
	pl_a.SetCategory(game_objects, true);
	pl_a.SetCategory(player_a, true);
	ASSERT(pl_a.IsDynamic());
	
	Add(pl_b);
	pl_b.SetRadius(3);
	pl_b.SetPosition(0, -16);
	pl_b.SetTypeDynamic();
	pl_b.SetRestitution(0.0);
	pl_b.SetFriction(0.5);
	pl_b.Create();
	pl_b.SetName("Mary");
	pl_b.SetCategory(0, false);
	pl_b.SetCategory(game_objects, true);
	pl_b.SetCategory(player_b, true);
	ASSERT(pl_b.IsDynamic());
	
	
	// Add the puck
	ResetPuck();
	
	
	// Add goals
	Add(goal_a);
	goal_a << tli*1.2 << tri*1.2 << tri*1.5 << tli*1.5;
	goal_a.Create();
	goal_a.id = 0;
	
	Add(goal_b);
	goal_b << bli*1.5 << bri*1.5 << bri*1.2 << bli*1.2;
	goal_b.Create();
	goal_b.id = 1;
	
	
	
	
	// Set contact listener
	SetContactListener(*this);
	
	
	// Assert some collisions
	ASSERT(map_l.IsColliding(pl_a));
	ASSERT(map_l.IsColliding(pl_b));
	ASSERT(map_l.IsColliding(puck[0]));
	ASSERT(pl_a.IsColliding(pl_b));
	ASSERT(pl_a.IsColliding(puck[0]));
	ASSERT(pl_b.IsColliding(puck[0]));
	ASSERT(pl_a.IsColliding(area_b));
	ASSERT(!pl_a.IsColliding(area_a));
	ASSERT(pl_b.IsColliding(area_a));
	ASSERT(!pl_b.IsColliding(area_b));
	ASSERT(!area_a.IsColliding(puck[0]));
	ASSERT(!area_b.IsColliding(puck[0]));
	
}

Polygon& Table2::GetPolygon(int i) {
	ASSERT(i >= 0 && i < 8);
	switch (i) {
		case 0: return map_l;
		case 1: return map_tl;
		case 2: return map_tr;
		case 3: return map_r;
		case 4: return map_bl;
		case 5: return map_br;
		case 6: return goal_a;
		case 7: return goal_b;
		default: area_b; // NEVER;
	}
}

void Table2::ContactBegin(Contact contact) {
	Goal* goal = contact.Get<Goal>();
	if (goal) {
		Puck* puck = contact.Get<Puck>();
		if (puck) {
			PostCallback(THISBACK1(PlayerScore, !goal->id));
		}
		return;
	}
	Player* player = contact.Get<Player>();
	if (player) {
		Puck* puck = contact.Get<Puck>();
		if (puck) {
			player->game_score += 0.2; // reward pushing puck
			if (player == &agents[0]) {
				agents[1].game_score -= 0.2; // punish for letting other hit
			} else {
				agents[0].game_score -= 0.2;
			}
		}
		return;
	}
}

void Table2::ContactEnd(Contact contact) {
	
}

void Table2::ResetPuck() {
	if (puck.GetCount()) {
		Remove(puck[0]);
		puck.Remove(0);
	}
	Puck& puck = this->puck.Add();
	agents[0].SetPuck(puck);
	agents[1].SetPuck(puck);
	Add(puck);
	puck.SetRadius(2);
	puck.SetPosition(0, player_a_starts ? 6 : -6);
	puck.SetTypeDynamic();
	puck.SetRestitution(0.9);
	puck.SetFriction(0.5);
	puck.SetCategory(game_objects, true);
	puck.Create();
}

InterceptResult Table2::StuffCollide(int skip_agent, Pointf p1, Pointf p2, bool check_walls, bool check_items) {
	InterceptResult minres(false);
	
	// collide with walls
	if (check_walls) {
		for(int i = 0; i < GetPolygonCount(); i++) {
			Polygon& poly = GetPolygon(i);
			for(int j = 0; j < poly.GetCount(); j++) {
				Pointf w1 = j == 0 ? poly[poly.GetCount()-1] : poly[j-1];
				Pointf w2 = poly[j];
				
				InterceptResult res = IsLineIntersect(p1, p2, w1, w2);
				res.vx = 0;
				res.vy = 0;
				if (res) {
					res.type = 0; // 0 is wall
					if (!minres) {
						minres = res;
					}
					// check if its closer
					else if (res.ua < minres.ua) {
						// if yes replace it
						minres = res;
					}
				}
			}
		}
	}
	
	// collide with other players and puck
	if (check_items) {
		
		// check players
		for(int i = 0; i < agents.GetCount(); i++) {
			if (i == skip_agent) continue;
			Player& p = agents[i];
			Pointf pos = p.GetPosition();
			double rad = p.GetRadius();
			InterceptResult res = IsLinePointIntersect(p1, p2, pos, rad);
			if (res) {
				Pointf velocity = p.GetSpeed();
				res.type = 1; // 1 is other player
				res.vx = velocity.x;
				res.vy = velocity.y;
				if (!minres) {
					minres = res;
				}
				else if(res.ua < minres.ua) {
					minres = res;
				}
			}
		}
		
		// check puck
		for(int i = 0; i < puck.GetCount(); i++) {
			Puck& p = puck[i];
			Pointf pos = p.GetPosition();
			double rad = p.GetRadius();
			InterceptResult res = IsLinePointIntersect(p1, p2, pos, rad);
			if (res) {
				Pointf velocity = p.GetSpeed();
				res.type = 2; // 2 is puck
				res.vx = velocity.x;
				res.vy = velocity.y;
				if (!minres) {
					minres = res;
				}
				else if(res.ua < minres.ua) {
					minres = res;
				}
			}
		}
		
		
	}
	
	return minres;
}

void Table2::Tick() {
	World::Tick();
	
	for (int i = 0, n = agents.GetCount(); i < n; i++) {
		Player& a = agents[i];
		Pointf ap = a.GetPosition();
		
		// Process DQN learning
		a.Backward();
		
		// Process AI player moves
		a.Process();
		
		for(int ei = 0, ne = a.eyes.GetCount(); ei < ne; ei++) {
			Eye& e = a.eyes[ei];
			
			// we have a line from p to p->eyep
			double angle = e.angle;
			Pointf eyep(
				ap.x + e.max_range * sin(angle),
				ap.y + e.max_range * cos(angle));
			InterceptResult res = StuffCollide(i, ap, eyep, true, true);
			if (res) {
				// eye collided with wall
				e.sensed_proximity = Distance(res.up, ap);
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
	
	
}

}

