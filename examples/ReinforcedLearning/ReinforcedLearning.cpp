#include "ReinforcedLearning.h"
#include "pretrained.brc"
#include <plugin/bz2/bz2.h>

#define IMAGECLASS ReinforcedLearningImg
#define IMAGEFILE <ReinforcedLearning/ReinforcedLearning.iml>
#include <Draw/iml_source.h>


GUI_APP_MAIN {
	ReinforcedLearning().Run();
}


ReinforcedLearning::ReinforcedLearning() {
	Title("Deep Q Learning Reinforcement");
	Icon(ReinforcedLearningImg::icon());
	Sizeable().MaximizeBox().MinimizeBox().Zoomable();
	
	running = false;
	stopped = true;
	ticking_running = false;
	ticking_stopped = true;
	
	skipdraw = false;
	
	t = "[\n"
		"\t{\"type\":\"input\", \"input_width\":1, \"input_height\":1, \"input_depth\":59},\n"
		"\t{\"type\":\"fc\", \"neuron_count\": 50, \"activation\":\"relu\"},\n"
		"\t{\"type\":\"fc\", \"neuron_count\": 50, \"activation\":\"relu\"},\n"
		"\t{\"type\":\"regression\", \"neuron_count\":5},\n"
		"\t{\"type\":\"sgd\", \"learning_rate\":0.001, \"momentum\":0.0, \"batch_size\":64, \"l2_decay\":0.01}\n"
		"]\n";
		
	net_edit.SetData(t);
    
	Add(world.SizePos());
	
	net_ctrl.Add(net_edit.HSizePos().VSizePos(0,30));
	net_ctrl.Add(reload_btn.HSizePos().BottomPos(0,30));
	reload_btn.SetLabel("Reload Network");
	reload_btn <<= THISBACK(Reload);
	
	reward_graph.SetMode(PLOT_AA).SetLimits(-5,5,-5,5);
	reward_graph.data.Add();
	reward_graph.data.Add();
	reward_graph.data[0].SetTitle("Reward").SetThickness(1.5).SetColor(Red());
	reward_graph.data[1].SetDash("1.5").SetTitle("Average").SetThickness(1.0).SetColor(Blue());
	average_size = 10;
	
	network_view.SetSession(world.agents[0].brain);
    input_view.SetSession(world.agents[0].brain);
    
    is_training.SetLabel("Is training");
	load_trained.SetLabel("Load pre-trained");
	load_file.SetLabel("Load file");
	store_file.SetLabel("Save file");
	load_trained <<= THISBACK(LoadPreTrained);
	load_file <<= THISBACK(OpenFile);
	store_file <<= THISBACK(SaveFile);
	
	is_training.Set(true);
	is_training <<= THISBACK(RefreshTrainingStatus);
	
	// "Very fast" is meaningful only if the program doesn't have threads, and calculating
	// must be done in the GUI loop, like in the JS version.
	#ifdef flagMT
	speed.MinMax(0,2);
	speed.SetData(0);
	#else
	speed.MinMax(0,3);
	speed.SetData(1);
	#endif
	
	speed.Step(1);
	speed <<= THISBACK(RefreshSpeed);
	
	controller.Add(is_training.TopPos(0,30).LeftPos(0,150));
	controller.Add(load_trained.TopPos(30,30).LeftPos(0,150));
	controller.Add(load_file.TopPos(60,30).LeftPos(0,150));
	controller.Add(store_file.TopPos(90,30).LeftPos(0,150));
	controller.Add(speed.VSizePos().LeftPos(150,30));
	
	
    Start();
}

ReinforcedLearning::~ReinforcedLearning() {
	ticking_running = false;
	while (!ticking_stopped) Sleep(100);
}

void ReinforcedLearning::DockInit() {
	DockLeft(Dockable(input_view, "Input").SizeHint(Size(320, 240)));
	DockLeft(Dockable(reward_graph, "Reward").SizeHint(Size(320, 240)));
	DockLeft(Dockable(status, "Status").SizeHint(Size(120, 120)));
	DockBottom(Dockable(network_view, "Network View").SizeHint(Size(640, 120)));
	DockBottom(Dockable(controller, "Controls").SizeHint(Size(180, 30*4)));
	AutoHide(DOCK_LEFT, Dockable(net_ctrl, "Edit Network").SizeHint(Size(640, 320)));
}

void ReinforcedLearning::RefreshStatus() {
	Brain& b = world.agents[0].brain;
	String s;
	s << "   experience replay size: " << b.GetExperienceCount() << "\n";
	s << "   exploration epsilon: " << b.GetEpsilon() << "\n";
	s << "   age: " << b.GetAge() << "\n";
	s << "   average Q-learning loss: " << b.GetAverageLoss() << "\n";
	s << "   smooth-ish reward: " << b.GetAverageReward();
	status.SetLabel(s);
}

void ReinforcedLearning::RefreshTrainingStatus() {
	bool is_training = this->is_training;
	for(int i = 0; i < world.agents.GetCount(); i++) {
		world.agents[i].brain.SetLearning(is_training);
	}
}

void ReinforcedLearning::RefreshSpeed() {
	int speed = this->speed.GetMax() - (int)this->speed.GetData(); // Inverse direction
	switch (speed) {
		case 0:
			GoSlow();
			break;
		case 1:
			GoNormal();
			break;
		case 2:
			GoFast();
			break;
		case 3:
			GoVeryFast();
			break;
	}
}

void ReinforcedLearning::Tick() {
	// "Normal" is a little bit slower than fastest
	if      (simspeed == 1) Sleep(10);
	else if (simspeed == 0) Sleep(100);
	world.Tick();
	
	Brain& brain = world.agents[0].brain;
	int experiences = brain.GetExperienceCount();
	int average_window_size = brain.GetAverageLossWindowSize();
	if (experiences >= average_window_size && (experiences % 100) == 0)
		AddReward();
}

void ReinforcedLearning::Ticking() {
	while (ticking_running) {
		ticking_lock.Enter();
		Tick();
		ticking_lock.Leave();
	}
	ticking_stopped = true;
}

void ReinforcedLearning::GoVeryFast() {
	skipdraw = true;
	simspeed = 3;
}

void ReinforcedLearning::GoFast() {
	skipdraw = false;
	simspeed = 2;
}

void ReinforcedLearning::GoNormal() {
	skipdraw = false;
	simspeed = 1;
}

void ReinforcedLearning::GoSlow() {
	skipdraw = false;
	simspeed = 0;
}

void ReinforcedLearning::Reload() {
	String net_str = net_edit.GetData();
	Brain& brain = world.agents[0].brain;
	
	ticking_lock.Enter();
	brain.Reset();
	bool success = brain.MakeLayers(net_str);
	ticking_lock.Leave();
	
	if (success)
		brain.StartTraining();
	else
		brain.StopTraining();
	
}

void ReinforcedLearning::Start() {
	GoFast();
	
	running = true;
	stopped = false;
	PostCallback(THISBACK(Refresher));
	
	ticking_running = true;
	ticking_stopped = false;
	Thread::Start(THISBACK(Ticking));
	
}

void ReinforcedLearning::Refresher() {
	if (!skipdraw) {
		input_view.Refresh();
		network_view.Refresh();
		world.Refresh();
		reward_graph.Sync();
		reward_graph.Refresh();
		RefreshStatus();
	}
	if (running) PostCallback(THISBACK(Refresher));
	else stopped = true;
}

void ReinforcedLearning::AddReward() {
	int id = reward_graph.data[0].GetCount();
	double av = world.agents[0].brain.GetAverageReward();
	reward_graph.data[0].AddXY(id, av);
	int count = id + 1;
	if (count < 2) return;
	
	int pos = id;
	double sum = 0;
	int av_count = 0;
	for(int i = 0; i < average_size && pos >= 0; i++) {
		sum += reward_graph.data[0][pos].y;
		av_count++;
		pos--;
	}
	double avav = sum / av_count;
	reward_graph.data[1].AddXY(id, avav);
	
	
	double min = +DBL_MAX;
	double max = -DBL_MAX;
	for(int i = 0; i < count; i++) {
		double d = reward_graph.data[0][i].y;
		if (d > max) max = d;
		if (d < min) min = d;
	}
	double diff = max - min;
	if (diff <= 0) return;
	double center = min + diff / 2;
	reward_graph.SetLimits(0, id, min, max);
	reward_graph.SetModify();
	//reward_graph.Refresh();
	//reward_graph.Sync();
}

void ReinforcedLearning::LoadPreTrained() {
	
	// This is the pre-trained network from original ConvNetJS
	MemReadStream pretrained_mem(pretrained, pretrained_length);
	String json = BZ2Decompress(pretrained_mem);
	
	// Stop training
	ticking_lock.Enter();
	this->is_training = false;
	RefreshTrainingStatus();
	
	// Load json
	world.agents[0].brain.LoadOriginalJSON(json);
	ticking_lock.Leave();
	
	// Go slower
	GoNormal();
	
}

void ReinforcedLearning::OpenFile() {
	String file = SelectFileOpen("JSON files\t*.json\nAll files\t*.*");
	if (file.IsEmpty()) return;
	
	if (!FileExists(file)) {
		PromptOK("File does not exists");
		return;
	}
	
	// Stop training
	this->is_training = false;
	RefreshTrainingStatus();
	
	// Load json
	String json = LoadFile(file);
	if (json.IsEmpty()) {
		PromptOK("File is empty");
		return;
	}
	ticking_lock.Enter();
	bool res = world.agents[0].brain.LoadOriginalJSON(json);
	if (!res) {
		ticking_running = false;
	}
	ticking_lock.Leave();
	
	if (!res) {
		PromptOK("Loading failed.");
		return;
	}
	
	// Go slower
	GoNormal();
	
	if (!ticking_running) Start();
}

void ReinforcedLearning::SaveFile() {
	String file = SelectFileSaveAs("JSON files\t*.json\nAll files\t*.*");
	if (file.IsEmpty()) return;
	
	// Save json
	String json;
	if (!world.agents[0].brain.StoreOriginalJSON(json)) {
		PromptOK("Error: Getting JSON failed");
		return;
	}
	
	FileOut fout(file);
	if (!fout.IsOpen()) {
		PromptOK("Error: could not open file " + file);
		return;
	}
	fout << json;
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


