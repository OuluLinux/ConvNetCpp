#include "DiffusionModel.h"

struct CtrlPanel : WithParentCtrl {
	TrainingGraph graph;

	CtrlPanel() {
		Add(graph.HSizePos().VSizePos());
	}
};