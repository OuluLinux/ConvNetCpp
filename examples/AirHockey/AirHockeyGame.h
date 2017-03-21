#ifndef _GameLib_AirHockey_h_
#define _GameLib_AirHockey_h_

#include "Player.h"

namespace GameCtrl {

struct Goal : public Polygon {
	int id;
};

struct Map : public Polygon {
	int w, h, gw;
	int last_won, last_score;
	void* table;
	
	Map();
	void PlayerWon(int i);
	void PlayerScore(int i);
	virtual void Paint(WorldDraw& wdraw, Draw& draw);
};

class Table2 : public World, public ContactListener {
	
public:
	Map area_a;
	Polygon area_b, map_l, map_tl, map_tr, map_r, map_bl, map_br;
	Array<Player> agents;
	Array<Puck> puck;
	Goal goal_a, goal_b;
	int score[2];
	int score_limit;
	bool player_a_starts;
	bool debug_paint;
	
public:
	typedef Table2 CLASSNAME;
	Table2();
	
	void Init();
	void ResetGame();
	void Reset();
	void PlayerScore(int i);
	void ResetPuck();
	void SetScoreLimit(int i) {score_limit = i;}
	int GetScoreA() {return score[0];}
	int GetScoreB() {return score[1];}
	int GetScoreLimit() {return score_limit;}
	
	InterceptResult StuffCollide(int skip_agent, Pointf p1, Pointf p2, bool check_walls, bool check_items);
	int GetPolygonCount() const {return 8;}
	Polygon& GetPolygon(int i);
	
	virtual void Tick();
	virtual void ContactBegin(Contact contact);
	virtual void ContactEnd(Contact contact);
	
	Callback1<int> WhenScore, WhenFinish;
};

}

#endif
