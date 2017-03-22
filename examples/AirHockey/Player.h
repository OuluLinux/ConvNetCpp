#ifndef _GameCtrl_Agent_h_
#define _GameCtrl_Agent_h_

#include <ConvNetCtrl/ConvNetCtrl.h>
#include "Object.h"

namespace GameCtrl {
using namespace Upp;
using namespace ConvNet;

struct InterceptResult {
	Pointf up;
	double ua, ub;
	int type;
	bool is_intercepting;
	int vx, vy;
	
	InterceptResult() {Reset();}
	InterceptResult(bool is_in) {Reset(); is_intercepting = is_in;}
	InterceptResult(const InterceptResult& ir) {*this = ir;}
	void Reset() {up.x = 0; up.y = 0; ua = 0; ub = 0; vx = 0; vy = 0; is_intercepting = false;}
	operator bool() const {return is_intercepting;}
	InterceptResult& operator=(const InterceptResult& ir) {
		up = ir.up;
		ua = ir.ua;
		ub = ir.ub;
		vx = ir.vx;
		vy = ir.vy;
		type = ir.type;
		is_intercepting = ir.is_intercepting;
		return *this;
	}
};

inline double RandomRange(double min, double max) {return min + Randomf() * (max - min);}
inline int RandomRangeInt(int min, int max) {return min + Random(max - min);}
InterceptResult IsLineIntersect(Pointf l1a, Pointf l1b, Pointf l2a, Pointf l2b);
InterceptResult IsLinePointIntersect(Pointf l1a, Pointf l1b, Pointf p, int rad);
template <class T> inline void Scale(T& p, double s) { p.x *= s; p.y *= s; }
template <class T> inline void Normalize(T& p) {double d = Length(p); Scale(p, 1.0 / d);}
template <class T> inline T Rotate(const T& p, double angle) {
	return T( // CLOCKWISE
		+p.x * cos(angle) + p.y * sin(angle),
		-p.x * sin(angle) + p.y * cos(angle));
}

// Eye sensor has a maximum range and senses walls
struct Eye : Moveable<Eye> {
	double angle; // angle relative to agent its on
	double max_range;
	double sensed_proximity; // what the eye is seeing. will be set in world.tick()
	double vx, vy;
	int sensed_type; // what does the eye see?
	
	void Init(double angle) {
		this->angle = angle;
		max_range = 60;
		sensed_proximity = 60;
		sensed_type = -1;
		vx = 0;
		vy = 0;
	}
};


class Puck : public Circle {
	
public:
	virtual void Paint(WorldDraw& wdraw, Draw& draw);
	
};


class AirHockeyDQN;

class Player : public Circle, public DQNAgent {
	
protected:
	String name;
	Puck* puck;
	int id;
	
public:
	Player();
	
	void Process();
	virtual void Paint(WorldDraw& wdraw, Draw& draw);
	
	void SetName(String s) {name = s;}
	void SetPuck(Puck& puck) {this->puck = &puck;}
	
	// Agent
	void Forward();
	void Backward();
	void Reset();
	void SetId(int i) {id = i;}
	
	AirHockeyDQN* world;
	Vector<Eye> eyes;
	Vector<double> smooth_reward_history;
	Vector<int> actions;
	double smooth_reward;
	double reward;
	double game_score;
	int nflot, iter;
	int action;
	bool do_training;
	bool paint_eyes;
	
};

}

#endif
