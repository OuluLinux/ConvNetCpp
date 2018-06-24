#include "GAN.h"

GAN::GAN() {
	CtrlLayout(panel);
	Add(panel.LeftPos(0,300).VSizePos());
	Add(vsplit.HSizePos(300).VSizePos());
	
	vsplit.Vert();
	vsplit << gen_layer_view << disc_layer_view;
	
	gen_layer_view.SetColor();
	disc_layer_view.SetColor();
	
}

void GAN::Init() {
	
	l.Init(0);
	
	
	disc_layer_view.SetSession(l.GetDiscriminator());
	gen_layer_view.SetSession(l.GetGenerator());
	disc_layer_view.RefreshLayers();
	gen_layer_view.RefreshLayers();
	
	panel.disc_graph.SetSession(l.GetDiscriminator());
	panel.gen_graph.SetSession(l.GetGenerator());
	panel.disc_graph.SetModeLoss();
	panel.gen_graph.SetModeLoss();
	
	Thread::Start(THISBACK(Training));
}

void GAN::Training() {
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
			panel.disc_graph.PostAddValue(l.PickAverageDiscriminatorCost());
			panel.gen_graph.PostAddValue(l.PickAverageGeneratorCost());
		}
	}
	
	stopped = true;
}

void GAN::RefreshData() {
	//lock.Enter();
	disc_layer_view.Refresh();
	gen_layer_view.Refresh();
	panel.disc_graph.RefreshData();
	panel.gen_graph.RefreshData();
	//lock.Leave();
}

