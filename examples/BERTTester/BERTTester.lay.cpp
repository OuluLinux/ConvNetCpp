#include "BERTTester.h"

WithCtrlPanel::WithCtrlPanel()
{
	Add(main_pctrl.LeftPos(0, 300).TopPos(0, 200));
	main_pctrl.Add(disc_pctrl.HSizePos(4, 4).TopPos(4, 24));
	disc_pctrl.Add(disc_graph.HSizePos().VSizePos(4, 4));
	main_pctrl.Add(gen_pctrl.HSizePos(4, 4).VPos(24, 24));
	gen_pctrl.Add(gen_graph.HSizePos().VSizePos(4, 4));
	Add(button.RightPos(4, 100).BottomPos(4, 24));
	button.SetLabel("Button");
}

WithCtrlPanel::~WithCtrlPanel()
{
}