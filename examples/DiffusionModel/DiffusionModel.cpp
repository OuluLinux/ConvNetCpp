#include "DiffusionModel.h"

DiffusionModel::DiffusionModel() {
	Icon(DiffusionModelImg::icon());
	Sizeable().MaximizeBox().MinimizeBox().Zoomable();

	CtrlLayout(panel);
	Add(panel.LeftPos(0,300).VSizePos());
	Add(vsplit.HSizePos(300).VSizePos());

	vsplit.Vert();
	vsplit << model_layer_view;

	model_layer_view.SetColor();

}

void DiffusionModel::Init() {

	l.Init(0);

	model_layer_view.SetSession(l.GetModel());

	model_layer_view.RefreshLayers();

	panel.graph.SetSession(l.GetModel()); // Using main model for graph
	panel.graph.SetModeLoss();

	Thread::Start(THISBACK(Training));
}

void DiffusionModel::Training() {
	running = true;
	stopped = false;

	TimeStop ts;

	int iter = 0;

	while (running) {

		lock.Enter();
		l.Train();
		iter++;
		lock.Leave();
		//Sleep(100);

		if (ts.Elapsed() >= 1000/60) {
			PostCallback(THISBACK(RefreshData));
			ts.Reset();
		}

		if (iter % 10 == 0) {
			panel.graph.PostAddValue(l.PickAverageLoss());
		}
	}

	stopped = true;
}

void DiffusionModel::RefreshData() {
	//lock.Enter();
	model_layer_view.Refresh();
	panel.graph.RefreshData();
	//lock.Leave();
}