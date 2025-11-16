#include "BigGAN.h"

struct CtrlPanel : WithParentCtrl {
	TrainingGraph disc_graph;
	TrainingGraph gen_graph;

	CtrlPanel() {
		Add(disc_graph.VSizePos(0, 240).HSizePos());
		Add(gen_graph.VSizePos(240).HSizePos());
	}
};