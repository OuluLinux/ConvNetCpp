#include "CycleGAN.h"

CycleGAN::CycleGAN() {
	Icon(CycleGANImg::icon());
	Sizeable().MaximizeBox().MinimizeBox().Zoomable();

	CtrlLayout(panel);
	Add(panel.LeftPos(0,300).VSizePos());
	Add(vsplit.HSizePos(300).VSizePos());

	vsplit.Vert();
	vsplit << gen_XtoY_layer_view << disc_Y_layer_view;
	vsplit << gen_YtoX_layer_view << disc_X_layer_view;

	gen_XtoY_layer_view.SetColor();
	gen_YtoX_layer_view.SetColor();
	disc_X_layer_view.SetColor();
	disc_Y_layer_view.SetColor();

}

void CycleGAN::Init() {

	l.Init(0);


	disc_X_layer_view.SetSession(l.GetDiscriminatorX());
	disc_Y_layer_view.SetSession(l.GetDiscriminatorY());
	gen_XtoY_layer_view.SetSession(l.GetGeneratorXtoY());
	gen_YtoX_layer_view.SetSession(l.GetGeneratorYtoX());
	
	disc_X_layer_view.RefreshLayers();
	disc_Y_layer_view.RefreshLayers();
	gen_XtoY_layer_view.RefreshLayers();
	gen_YtoX_layer_view.RefreshLayers();

	panel.disc_graph.SetSession(l.GetDiscriminatorX()); // Using X discriminator for primary graph
	panel.gen_graph.SetSession(l.GetGeneratorXtoY());    // Using X->Y generator for primary graph
	panel.disc_graph.SetModeLoss();
	panel.gen_graph.SetModeLoss();

	Thread::Start(THISBACK(Training));
}

void CycleGAN::Training() {
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
			panel.disc_graph.PostAddValue(l.PickAverageDiscriminatorXCost() + l.PickAverageDiscriminatorYCost());
			panel.gen_graph.PostAddValue(l.PickAverageGeneratorXtoYCost() + l.PickAverageGeneratorYtoXCost());
		}
	}

	stopped = true;
}

void CycleGAN::RefreshData() {
	//lock.Enter();
	disc_X_layer_view.Refresh();
	disc_Y_layer_view.Refresh();
	gen_XtoY_layer_view.Refresh();
	gen_YtoX_layer_view.Refresh();
	panel.disc_graph.RefreshData();
	panel.gen_graph.RefreshData();
	//lock.Leave();
}