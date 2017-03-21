#ifndef _CharGen_CharGen_h
#define _CharGen_CharGen_h

#include <CtrlLib/CtrlLib.h>
#include <ConvNetCtrl/ConvNetCtrl.h>
#include <Docking/Docking.h>
using namespace Upp;
using namespace ConvNet;

#define IMAGECLASS CharGenImg
#define IMAGEFILE <CharGen/CharGen.iml>
#include <Draw/iml_header.h>

class CharGen : public DockWindow {
	
	Button reload_btn1;
	ParentCtrl input_ctrl;
	DocEdit input_edit;
	
	Button reload_btn2;
	ParentCtrl model_ctrl;
	DocEdit model_edit;
	
	String model;
	
public:
	typedef CharGen CLASSNAME;
	CharGen();
	~CharGen();
	
	virtual void DockInit();
	
	void Start();
	void Reset(bool init_reward, bool start);
	void Reload();
	void Refresher();
	
};

#endif
