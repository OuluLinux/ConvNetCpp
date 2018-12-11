#ifndef _ReinforcedLearning_ReinforcedLearning_h_
#define _ReinforcedLearning_ReinforcedLearning_h_

#include <ConvNetCtrl/ConvNetCtrl.h>
#include <PlotCtrl/PlotCtrl.h>
#include <Docking/Docking.h>
using namespace Upp;
using namespace ConvNet;

#define IMAGECLASS ReinforcedLearningImg
#define IMAGEFILE <Martingale/ReinforcedLearning.iml>
#include <Draw/iml_header.h>


struct World;

// A single agent
struct RLAgent {
	
	static const int input_length = 10;
	static const int max_martingale = 8;
	static const int input_count = input_length + max_martingale * 2 + 1;
	
	enum {ACT_COLLECT, ACT_DOUBLE, ACT_COUNT};
	
	World* world = NULL;
	Brain brain;
	Vector<double> posv, negv;
	double reward_bonus, digestion_signal;
	int pos;
	int simspeed;
	int actionix;
	int failcount, doublelen;
	int multiplier;
	int prevdir;
	int collected_reward;
	
	RLAgent();
	void Forward();
	void Backward();
	
};




struct World : public Ctrl {
	Vector<double> buffer;
	Array<RLAgent> agents;
	Vector<Point> polyline;
	double zero_line = 0;
	int clock;
	
	World();
	void Tick();
	virtual void Paint(Draw& d);
	
};



class ReinforcedLearning : public DockWindow {
	World world; // global world object
	DocEdit net_edit;
	ParentCtrl net_ctrl;
	Button reload_btn;
	TrainingGraph reward_graph;
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
	int average_size;
	bool skipdraw;
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
	void RefreshStatus();
	void RefreshTrainingStatus();
	void RefreshSpeed();
	void LoadPreTrained();
	void OpenFile();
	void SaveFile();
	
};

#endif
