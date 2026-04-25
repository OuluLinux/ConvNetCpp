#include "ConditionalGAN.h"

ConditionalGAN::ConditionalGAN() {
    Sizeable().MaximizeBox().MinimizeBox().Zoomable();
    Title("ConditionalGAN example");
    running = false; stopped = true;

    Add(options_panel.LeftPos(0, 300).VSizePos());
    Add(vsplit.HSizePos(300, 0).VSizePos());
    
    options_panel.Add(ctrl_panel.SizePos());
    ctrl_panel.CtrlLayout(ctrl_panel);

    vsplit.Vert();
    vsplit << gen_layer_view << disc_layer_view;

    gen_layer_view.SetColor();
    disc_layer_view.SetColor();
}

void ConditionalGAN::Init() {
    l.Init(0);
    disc_layer_view.SetSession(l.GetDiscriminator());
    gen_layer_view.SetSession(l.GetGenerator());
    disc_layer_view.RefreshLayers();
    gen_layer_view.RefreshLayers();
    ctrl_panel.disc_graph.SetSession(l.GetDiscriminator());
    ctrl_panel.gen_graph.SetSession(l.GetGenerator());
    ctrl_panel.disc_graph.SetModeLoss();
    ctrl_panel.gen_graph.SetModeLoss();
    Thread::Start(THISBACK(Training));
}

void ConditionalGAN::Training() {
    running = true; stopped = false;
    TimeStop ts;
    int iter = 0;
    while (running) {
        lock.Enter();
        l.Train();
        iter++;
        lock.Leave();
        if (ts.Elapsed() >= 1000/60) {
            PostCallback(THISBACK(RefreshData));
            ts.Reset();
        }
        if (iter % 10 == 0) {
            ctrl_panel.disc_graph.PostAddValue(l.PickAverageDiscriminatorCost());
            ctrl_panel.gen_graph.PostAddValue(l.PickAverageGeneratorCost());
        }
    }
    stopped = true;
}

void ConditionalGAN::RefreshData() {
    disc_layer_view.Refresh();
    gen_layer_view.Refresh();
    ctrl_panel.disc_graph.RefreshData();
    ctrl_panel.gen_graph.RefreshData();
}
