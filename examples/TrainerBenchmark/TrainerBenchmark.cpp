#include "TrainerBenchmark.h"

#define IMAGECLASS TrainerBenchmarkImg
#define IMAGEFILE <TrainerBenchmark/TrainerBenchmark.iml>
#include <Draw/iml_source.h>

/*

// below fill out the trainer specs you wish to evaluate, and give them names for legend\n\
var LR = 0.01; // learning rate\n\
var BS = 8; // batch size\n\
var L2 = 0.001; // L2 weight decay\n\
nets = [];\n\
trainer_defs = [];\n\
trainer_defs.push({learning_rate:10*LR, method: 'sgd', momentum: 0.0, batch_size:BS, l2_decay:L2});\n\
trainer_defs.push({learning_rate:LR, method: 'sgd', momentum: 0.9, batch_size:BS, l2_decay:L2});\n\
trainer_defs.push({learning_rate:LR, method: 'adagrad', eps: 1e-6, batch_size:BS, l2_decay:L2});\n\
trainer_defs.push({learning_rate:LR, method: 'windowgrad', eps: 1e-6, ro: 0.95, batch_size:BS, l2_decay:L2});\n\
trainer_defs.push({learning_rate:1.0, method: 'adadelta', eps: 1e-6, ro:0.95, batch_size:BS, l2_decay:L2});\n\
trainer_defs.push({learning_rate:LR, method: 'nesterov', momentum: 0.9, batch_size:BS, l2_decay:L2});\n\

*/

TrainerBenchmark::TrainerBenchmark() {
	Title("Trainer benchmark demo on MNIST");
	Icon(TrainerBenchmarkImg::icon());
	Sizeable().MaximizeBox().MinimizeBox().Zoomable();
	
	trainer_running = false;
	trainer_stopped = true;
	
	t =		"[\n"
			"\t{\"type\":\"input\", \"input_width\":24, \"input_height\":24, \"input_depth\":1},\n"
			"\t{\"type\":\"fc\", \"neuron_count\":20, \"activation\": \"relu\"},\n"
			"\t{\"type\":\"fc\", \"neuron_count\":20, \"activation\": \"relu\"},\n"
			"\t{\"type\":\"softmax\", \"class_count\":10},\n"
			"]\n";
	
	
	net_edit.SetData(t);
	net_ctrl.Add(net_edit.HSizePos().VSizePos(0,30));
	net_ctrl.Add(reload_btn.HSizePos().BottomPos(0,30));
	reload_btn.SetLabel("Reload Network");
	reload_btn <<= THISBACK(Reload);
	
	split << loss_vs_num_graph << testacc_vs_num_graph << trainacc_vs_num_graph;
	split.Vert();
	Add(split.SizePos());
	
	loss_vs_num_graph		.SetMetaSession(mses);
	testacc_vs_num_graph	.SetMetaSession(mses);
	trainacc_vs_num_graph	.SetMetaSession(mses);
	
	loss_vs_num_graph.SetModeLoss();
	testacc_vs_num_graph.SetModeTestAccuracy();
	trainacc_vs_num_graph.SetModeTrainingAccuracy();
	
	testacc_vs_num_graph.HideLegend();
	trainacc_vs_num_graph.HideLegend();
	
	SetRect(0,0,640, 300*3);
	
}

TrainerBenchmark::~TrainerBenchmark() {
	StopTrainer();
}



void TrainerBenchmark::Reload() {
	StopTrainer();
	
	String net_json = net_edit.GetData();
	
	Array<Session>& sessions = mses.GetSessions();
	
	sessions.Clear();
	
	
	// below fill out the trainer specs you wish to evaluate, and give them names for legend
	double LR = 0.01; // learning rate
	int BS = 8; // batch size
	double L2 = 0.001; // L2 weight decay
	
	sessions.SetCount(6);
	
	for(int i = 0; i < sessions.GetCount(); i++) {
		Session& s = sessions[i];
		Net& net = s.GetNetwork();
		
		if (!s.MakeLayers(net_json)) {
			LOG("Invalid network json");
			return;
		}
		
		//{learning_rate:10*LR, method: 'sgd', momentum: 0.0, batch_size:BS, l2_decay:L2});\n\
		//{learning_rate:LR, method: 'sgd', momentum: 0.9, batch_size:BS, l2_decay:L2});\n\
		//{learning_rate:LR, method: 'adagrad', eps: 1e-6, batch_size:BS, l2_decay:L2});\n\
		//{learning_rate:LR, method: 'windowgrad', eps: 1e-6, ro: 0.95, batch_size:BS, l2_decay:L2});\n\
		//{learning_rate:1.0, method: 'adadelta', eps: 1e-6, ro:0.95, batch_size:BS, l2_decay:L2});\n\
		//{learning_rate:LR, method: 'nesterov', momentum: 0.9, batch_size:BS, l2_decay:L2});\n\
		
		if (i == 0) {
			SgdTrainer* t = new SgdTrainer(net);
			t->SetLearningRate(10 * LR);
			t->SetMomentum(0.0);
			t->SetBatchSize(BS);
			t->SetL2Decay(L2);
			s.AttachTrainer(t);
		}
		else if (i == 1) {
			SgdTrainer* t = new SgdTrainer(net);
			t->SetLearningRate(LR);
			t->SetMomentum(0.9);
			t->SetBatchSize(BS);
			t->SetL2Decay(L2);
			s.AttachTrainer(t);
		}
		else if (i == 2) {
			AdagradTrainer* t = new AdagradTrainer(net);
			t->SetLearningRate(LR);
			t->SetEps(1e-6);
			t->SetBatchSize(BS);
			t->SetL2Decay(L2);
			s.AttachTrainer(t);
		}
		else if (i == 3) {
			WindowgradTrainer* t = new WindowgradTrainer(net);
			t->SetLearningRate(LR);
			t->SetEps(1e-6);
			t->SetRo(0.95);
			t->SetBatchSize(BS);
			t->SetL2Decay(L2);
			s.AttachTrainer(t);
		}
		else if (i == 4) {
			AdadeltaTrainer* t = new AdadeltaTrainer(net);
			t->SetLearningRate(1.0);
			t->SetEps(1e-6);
			t->SetRo(0.95);
			t->SetBatchSize(BS);
			t->SetL2Decay(L2);
			s.AttachTrainer(t);
		}
		else if (i == 5) {
			NetsterovTrainer* t = new NetsterovTrainer(net);
			t->SetLearningRate(LR);
			t->SetMomentum(0.9);
			t->SetBatchSize(BS);
			t->SetL2Decay(L2);
			s.AttachTrainer(t);
		}
		else Panic("Invalid trainer id");
	}
		
	StartTrainer();
}

void TrainerBenchmark::DockInit() {
	
	AutoHide(DOCK_LEFT, Dockable(net_ctrl, "Edit Network").SizeHint(Size(640, 320)));
	
}

void TrainerBenchmark::StartTrainer() {
	StopTrainer();
	
	loss_vs_num_graph.Clear();
	testacc_vs_num_graph.Clear();
	trainacc_vs_num_graph.Clear();
	
	ContinueTrainer();
}

void TrainerBenchmark::ContinueTrainer() {
	trainer_running = true;
	trainer_stopped = false;
	Thread::Start(THISBACK(Runner));
}

void TrainerBenchmark::Runner() {
	while (trainer_running && !Thread::IsShutdownThreads()) {
		mses.Step();
	}
	trainer_stopped = true;
}
