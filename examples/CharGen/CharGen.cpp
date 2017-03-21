#include "CharGen.h"
#include "CharGen.brc"
#include <plugin/bz2/bz2.h>

CharGen::CharGen() {
	Title("Character Generation Demo");
	Icon(CharGenImg::icon());
	Sizeable().MaximizeBox().MinimizeBox();
	
	MemReadStream input_mem(input, input_length);
	String input_str = BZ2Decompress(input_mem);
	
	input_edit.SetData(input_str);
	input_ctrl.Add(input_edit.HSizePos().VSizePos(0,30));
	input_ctrl.Add(reload_btn1.HSizePos().BottomPos(0,30));
	reload_btn1.SetLabel("Reload Agent");
	reload_btn1 <<= THISBACK(Reload);
	
	model_edit.SetData(model);
	model_ctrl.Add(model_edit.HSizePos().VSizePos(0,30));
	model_ctrl.Add(reload_btn2.HSizePos().BottomPos(0,30));
	reload_btn2.SetLabel("Reload Agent");
	reload_btn2 <<= THISBACK(Reload);
	
	
}

CharGen::~CharGen() {
	
}

void CharGen::DockInit() {
	AutoHide(DOCK_LEFT, Dockable(input_ctrl, "Input").SizeHint(Size(800, 240)));
	DockLeft(Dockable(model_ctrl, "Model").SizeHint(Size(320, 240)));
}

void CharGen::Start() {
	
}

void CharGen::Reset(bool init_reward, bool start) {
	
}

void CharGen::Reload() {
	
}

void CharGen::Refresher() {
	
}

