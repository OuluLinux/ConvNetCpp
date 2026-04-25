#include "StyleGAN.h"

StyleGAN::StyleGAN() {
    Title("StyleGAN");
    Sizeable().MaximizeBox().MinimizeBox().Zoomable();
    running = false; stopped = true;

    Add(options_panel.LeftPos(0, 300).VSizePos());
    Add(vsplit.HSizePos(300, 0).VSizePos());

    options_panel.Add(disc_graph.HSizePos().TopPos(0, 200));
    options_panel.Add(gen_graph.HSizePos().TopPos(200, 200));

    vsplit.Vert();
    vsplit << gen_layer_view << disc_layer_view;

    gen_layer_view.SetColor();
    disc_layer_view.SetColor();
}

void StyleGAN::Init() {
    StyleGANParams params;
    l.Init(0, params);
    disc_layer_view.SetSession(l.GetDiscriminator());
    gen_layer_view.SetSession(l.GetGenerator());
    disc_graph.SetSession(l.GetDiscriminator());
    gen_graph.SetSession(l.GetGenerator());
    disc_graph.SetModeLoss();
    gen_graph.SetModeLoss();
    Thread::Start(THISBACK(Training));
}

void StyleGAN::Training() {
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
            disc_graph.PostAddValue(l.PickAverageDiscriminatorCost());
            gen_graph.PostAddValue(l.PickAverageGeneratorCost());
        }
    }
    stopped = true;
}

void StyleGAN::RefreshData() {
    disc_layer_view.Refresh();
    gen_layer_view.Refresh();
    disc_graph.RefreshData();
    gen_graph.RefreshData();
}
