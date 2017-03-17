#ifndef _GridWorld_GridWorld_h
#define _GridWorld_GridWorld_h

#include <ConvNetCtrl/ConvNetCtrl.h>
using namespace ConvNet;
using namespace Upp;

#define IMAGECLASS GridWorldImg
#define IMAGEFILE <GridWorld/GridWorld.iml>
#include <Draw/iml_header.h>


class GridWorld : public TopWindow {
	GridWorldCtrl gworld;
	Label lbl_reward;
	SliderCtrl reward;
	Splitter btnsplit;
	Button poleval, polup, reset;
	ButtonOption toggle;
	
	
	DPAgent agent;
	bool running;
	
public:
	typedef GridWorld CLASSNAME;
	GridWorld();
	~GridWorld();
	
	void Start();
	void Stop();
	void Reset(bool init_reward);
	void Refresher();
	void UpdatePolicy();
	void EvaluatePolicy();
	void ToggleIteration();
	void GridFocus();
	void GridUnfocus();
	void RefreshReward();
	
};

#endif
