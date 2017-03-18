#include "AirHockey.h"

namespace GameCtrl {
namespace AirHockey {

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
				s << "Player A won!";
			else
				s << "Player B won!";
		}
		if (last_score != -1) {
			if (last_score == 0)
				s << "Player A scored!";
			else
				s << "Player B scored!";
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


Player::Player(int id) : id(id) {
	puck = NULL;
}

void Player::Paint(WorldDraw& wdraw, Draw& draw) {
	Color fill_color = Blue();
	Color border_color = Black();
	PaintCircle(wdraw, draw, fill_color, border_color);
}

void Player::Process() {
	Puck& puck = *this->puck;
	
	Pointf puck_pos = puck.GetPosition();
	Pointf puck_speed = puck.GetSpeed();
	Pointf pos = GetPosition();
	
	double puck_y = puck_pos.y;
	double puck_cy = puck_speed.y;
	double y = id == 0 ? +16 : -16;
	int r;
	
	r = 2;
	Rectf goal_area(-r, y-r, r, y+r);
	
	r = 6;
	Rectf puck_area(-r + puck_pos.x, -r + puck_pos.y, r + puck_pos.x, r + puck_pos.y);
	
	bool puck_at_own_area = id == 0 ? puck_y > 0 : puck_y < 0;
	bool puck_moving = puck_speed.x != 0.0 || puck_speed.y != 0.0;
	bool puck_incoming = id == 0 ? puck_cy > 0 : puck_cy < 0;
	bool near_goal = goal_area.Contains(pos);
	bool near_puck = puck_area.Contains(pos);
	
	double force = 5000;
	
	// Starting
	if (!puck_at_own_area && !puck_moving) {
		if (pos.x == 0)
			MoveTo(Pointf(4, y), 50, force);
		else
			SeekPuck(50, force);
	}
	// Responding
	else {
		
		// If puck is coming, then estimate target point and move there, or attack.
		if (puck_incoming) {
			double y_diff = y - puck_y;
			double y_time = y_diff / puck_speed.y;
			double x_diff = y_time * puck_speed.x;
			double est_x = puck_pos.x + x_diff;
			est_x += -3 + Random(7);
			int w = 16;
			
			r = 1;
			if (est_x > +w-r) est_x = +w - r;
			if (est_x < -w+r) est_x = -w + r;
			
			// If puck is close enough, then attack
			r = 8;
			if ((id == 0 && puck_y >= y - r) || (id == 1 && puck_y <= y + r)) {
				SeekPuck(50, force);
			}
			
			// If puck is moving slowly and is in own area, then attack
			else if (y_time >= 2 && puck_at_own_area ) {
				SeekPuck(50, force);
			}
			
			// Go to estimated target position
			else {
				Pointf target(est_x, y);
				MoveTo(target, 50, force);
			}
		}
		
		// If puck is stuck, then add some correcting impulse to the puck
		else if (fabs(puck_pos.x) > 12 && fabs(puck_speed.x) <= 0.1 && fabs(puck_speed.y) <= 0.1 && (id == 0 && puck_y >= y) || (id == 1 && puck_y <= y)) {
			puck.SetSpeed(0, id == 0 ? -10 : +10);
		}
		
		// Just hit puck if it is at own area and moving slow
		else if (puck_at_own_area && fabs(puck_cy) < 50 && fabs(puck_pos.x) < 12) {
			SeekPuck(50, force);
		}
		
		// Go closer to goal
		else {
			MoveTo(Pointf(0, y), 50, force);
		}
	}
	
}

void Player::MoveTo(const Pointf& target, double speed, double force_scalar) {
	Pointf this_pos = GetPosition();
	
	double distance = Length(target - this_pos);
	ASSERT(!IsFin(distance) || distance >= 0.0);
	if (IsFin(distance) && prev_target == target && prev_distance < distance) {
		SetSpeed(0,0);
		return;
	}
	prev_distance = distance;
	prev_target = target;
	
	Pointf pos_diff = target - this_pos;
	Pointf target_unit = pos_diff / Length(pos_diff); // unit vector of target speed
	Pointf target_speed = target_unit * speed;
	Pointf this_speed = GetSpeed();
	Pointf speed_diff = target_speed - this_speed;
	double speed_diff_len = Length(speed_diff);
	if (speed_diff_len < 1) {
		SetSpeed(target_speed);
	}
	else {
		Pointf force = speed_diff / speed_diff_len * force_scalar;
		ApplyForceToCenter(force);
	}
}

void Player::SeekPuck(double speed, double force) {
	ResetPreviousMove();
	MoveTo(puck->GetPosition(), speed, force);
}






Table2::Table2() : pl_a(0), pl_b(1) {
	score_limit = 3;
	score[0] = 0;
	score[1] = 0;
	area_a.table = this;
}

void Table2::Reset() {
	score[0] = 0;
	score[1] = 0;
	player_a_starts = true;
	ResetPuck();
}

void Table2::PlayerScore(int i) {
	player_a_starts = i == 0;
	score[i]++;
	if (score[i] >= score_limit) {
		Reset();
		area_a.PlayerWon(i);
		WhenFinish(i);
	}
	else {
		ResetPuck();
		area_a.PlayerScore(i);
		WhenScore(i);
	}
}

void Table2::Init() {
	
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
	pl_a.SetRestitution(0.9);
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
	pl_b.SetRestitution(0.9);
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
	
	
	// Set periodcal callback for AI processing
	SetTimeCallback(-1, THISBACK(ProcessAI));
	
	
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

void Table2::ContactBegin(Contact contact) {
	Goal* goal = contact.Get<Goal>();
	if (goal) {
		Puck* puck = contact.Get<Puck>();
		if (puck) {
			PostCallback(THISBACK1(PlayerScore, goal->id));
		}
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
	pl_a.SetPuck(puck);
	pl_b.SetPuck(puck);
	Add(puck);
	puck.SetRadius(2);
	puck.SetPosition(0, player_a_starts ? 6 : -6);
	puck.SetTypeDynamic();
	puck.SetRestitution(0.9);
	puck.SetFriction(0.5);
	puck.SetCategory(game_objects, true);
	puck.Create();
}

void Table2::ProcessAI() {
	
	// Process AI player moves
	pl_a.Process();
	pl_b.Process();
	
}


}
}
