#include "ReinforcedLearning.h"
#include "pretrained.brc"
#include <plugin/bz2/bz2.h>

#define IMAGECLASS ReinforcedLearningImg
#define IMAGEFILE <Martingale/ReinforcedLearning.iml>
#include <Draw/iml_source.h>


GUI_APP_MAIN {
	ReinforcedLearning().Run();
}


ReinforcedLearning::ReinforcedLearning() {
	Title("Deep Q Learning Reinforcement");
	Icon(ReinforcedLearningImg::icon());
	Sizeable().MaximizeBox().MinimizeBox().Zoomable();
	
	ticking_running = false;
	ticking_stopped = true;
	
	skipdraw = false;
	
	t = "[\n"
		"\t{\"type\":\"input\", \"input_width\":1, \"input_height\":1, \"input_depth\":" + IntStr(RLAgent::input_count) + "},\n"
		"\t{\"type\":\"fc\", \"neuron_count\": 50, \"activation\":\"relu\"},\n"
		"\t{\"type\":\"fc\", \"neuron_count\": 50, \"activation\":\"relu\"},\n"
		"\t{\"type\":\"regression\", \"neuron_count\":" + IntStr(RLAgent::ACT_COUNT) + "},\n"
		"\t{\"type\":\"sgd\", \"learning_rate\":0.001, \"momentum\":0.0, \"batch_size\":64, \"l2_decay\":0.01}\n"
		"]\n";
		
	net_edit.SetData(t);
    
	Add(world.SizePos());
	
	net_ctrl.Add(net_edit.HSizePos().VSizePos(0,30));
	net_ctrl.Add(reload_btn.HSizePos().BottomPos(0,30));
	reload_btn.SetLabel("Reload Network");
	reload_btn <<= THISBACK(Reload);
	
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
	
	reward_graph.SetSession(world.agents[0].brain);
	
	PostCallback(THISBACK(Reload));
	PostCallback(THISBACK(Start));
	PostCallback(THISBACK(Refresher));
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
	int age = brain.GetAge();
	int average_window_size = brain.GetAverageLossWindowSize();
	if (age >= average_window_size && (age % 100) == 0)
		reward_graph.AddValue();
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
}

void ReinforcedLearning::Start() {
	GoFast();
	
	ticking_running = true;
	ticking_stopped = false;
	Thread::Start(THISBACK(Ticking));
}

void ReinforcedLearning::Refresher() {
	if (!skipdraw) {
		input_view.Refresh();
		network_view.Refresh();
		world.Refresh();
		reward_graph.RefreshData();
		RefreshStatus();
	}
	PostCallback(THISBACK(Refresher));
}

void ReinforcedLearning::LoadPreTrained() {
	
	// This is the pre-trained network from original ConvNetJS
	MemReadStream pretrained_mem(pretrained, pretrained_length);
	BZ2DecompressStream stream(pretrained_mem);
	
	// Stop training
	ticking_lock.Enter();
	this->is_training = false;
	RefreshTrainingStatus();
	
	// Load json
	world.agents[0].brain.SerializeWithoutExperience(stream);
	ticking_lock.Leave();
	
	// Go slower
	GoNormal();
	
}

void ReinforcedLearning::OpenFile() {
	String file = SelectFileOpen("BIN files\t*.bin\nAll files\t*.*");
	if (file.IsEmpty()) return;
	
	if (!FileExists(file)) {
		PromptOK("File does not exists");
		return;
	}
	
	// Stop training
	this->is_training = false;
	RefreshTrainingStatus();
	
	
	ticking_lock.Enter();
	FileIn fin(file);
	world.agents[0].brain.SerializeWithoutExperience(fin);
	ticking_lock.Leave();
	
	
	// Go slower
	GoNormal();
	
	if (!ticking_running) Start();
}

void ReinforcedLearning::SaveFile() {
	String file = SelectFileSaveAs("BIN files\t*.bin\nAll files\t*.*");
	if (file.IsEmpty()) return;
	
	FileOut fout(file);
	if (!fout.IsOpen()) {
		PromptOK("Error: could not open file " + file);
		return;
	}
	world.agents[0].brain.SerializeWithoutExperience(fout);
}








