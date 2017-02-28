#ifndef _ReinforcedLearning_ReinforcedLearning_h_
#define _ReinforcedLearning_ReinforcedLearning_h_

#include <ConvNetCtrl/ConvNetCtrl.h>
#include <PlotCtrl/PlotCtrl.h>
#include <Docking/Docking.h>
using namespace Upp;
using namespace ConvNet;

// Wall is made up of two points
struct Wall : Moveable<Wall> {
	Wall() {}
	Wall(const Point& p1, const Point& p2) : p1(p1), p2(p2) {}
	Pointf p1, p2;
};


struct InterceptResult {
	Pointf up;
	double ua, ub;
	int type;
	bool is_intercepting;
	
	InterceptResult() {Reset();}
	InterceptResult(bool is_in) {Reset(); is_intercepting = is_in;}
	InterceptResult(const InterceptResult& ir) {*this = ir;}
	void Reset() {up.x = 0; up.y = 0; ua = 0; ub = 0; is_intercepting = false;}
	operator bool() const {return is_intercepting;}
	InterceptResult& operator=(const InterceptResult& ir) {
		up = ir.up;
		ua = ir.ua;
		ub = ir.ub;
		type = ir.type;
		is_intercepting = ir.is_intercepting;
		return *this;
	}
};

void AddBox(Vector<Wall>& lst, int x, int y, int w, int h);
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

// item is circle thing on the floor that agent can interact with (see or eat, etc)
struct Item : Moveable<Item> {
	void Init(int x, int y, int type) {
		p.x = x;
		p.y = y;
		this->type = type;
		rad = 20; // default radius
		age = 0;
		cleanup_ = false;
	}
	Pointf p;
	int type, rad, age;
	bool cleanup_;
};



// Eye sensor has a maximum range and senses walls
struct Eye : Moveable<Eye> {
	double angle; // angle relative to agent its on
	double max_range;
	double sensed_proximity; // what the eye is seeing. will be set in world.tick()
	int sensed_type; // what does the eye see?
	
	void Init(double angle) {
		this->angle = angle;
		max_range = 85;
		sensed_proximity = 85;
		sensed_type = -1;
	}
};


// A single agent
struct Agent {
	Brain brain;
	Pointf p;		// positional information
	Pointf op;		// old position
	double angle;	// direction facing
	double oangle;	// old angle
	double rad, reward_bonus, digestion_signal;
	double rot1, rot2;
	int prevactionix;
	int simspeed;
	int actionix;
	Vector<Pointf> actions;
	Vector<Eye> eyes;
	
	Agent();
	void Forward();
	void Backward();
	
};




struct World : public Ctrl {
	Array<Agent> agents;
	Vector<Item> items;
	int W, H;
	int clock;
	int pad;
	
	// set up walls in the world
	Vector<Wall> walls;
	
	World();
	InterceptResult StuffCollide(Pointf p1, Pointf p2, bool check_walls, bool check_items);
	void Tick();
	virtual void Paint(Draw& d);
	
};



class ReinforcedLearning : public DockWindow {
	World world; // global world object
	DocEdit net_edit;
	ParentCtrl net_ctrl;
	Button reload_btn;
	PlotCtrl reward_graph;
	BarView input_view;
	HeatmapView network_view;
	ParentCtrl controller;
	Button load_trained, load_file, store_file;
	String t;
	Label status;
	Option is_training;
	SliderCtrl speed;
	int current_interval_id;
	int simspeed;
	bool skipdraw;
	bool running, stopped;
	bool ticking_running, ticking_stopped;
	SpinLock ticking_lock;
	
public:
	typedef ReinforcedLearning CLASSNAME;
	ReinforcedLearning();
	~ReinforcedLearning();
	
	virtual void DockInit();
	
	void Tick();
	void Ticking();
	void GoVeryFast();
	void GoFast();
	void GoNormal();
	void GoSlow();
	void Reload();
	void Start();
	void Refresher();
	void AddReward();
	void RefreshStatus();
	void RefreshTrainingStatus();
	void RefreshSpeed();
	void LoadPreTrained();
	
};

#endif
