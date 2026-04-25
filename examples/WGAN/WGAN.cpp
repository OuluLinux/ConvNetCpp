#include "WGAN.h"

WGAN::WGAN() {
    Sizeable().MaximizeBox().MinimizeBox().Zoomable();
    Title("Wasserstein GAN (WGAN)");

    Add(options_panel.LeftPos(0, 300).VSizePos());
    Add(main_vsplit.HSizePos(300, 0).VSizePos());
    
    options_panel.Add(ctrl_panel.SizePos());
    ctrl_panel.CtrlLayout(ctrl_panel);

    main_vsplit.Vert();
    main_vsplit << gen_layer_view << disc_layer_view;

    gen_layer_view.SetColor();
    disc_layer_view.SetColor();
}

void WGAN::Init() {
    WGANParams params;
    l.Init(0, params);

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

void WGAN::Training() {
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
            ctrl_panel.disc_graph.PostAddValue(l.PickAverageDiscriminatorCost());
            ctrl_panel.gen_graph.PostAddValue(l.PickAverageGeneratorCost());
        }
    }

    stopped = true;
}

void WGAN::RefreshData() {
    //lock.Enter();
    disc_layer_view.Refresh();
    gen_layer_view.Refresh();
    ctrl_panel.disc_graph.RefreshData();
    ctrl_panel.gen_graph.RefreshData();
    //lock.Leave();
}
