#ifndef _AirHockey_AirHockey_h
#define _AirHockey_AirHockey_h

#include <CtrlLib/CtrlLib.h>
#include <plugin/box2d/Box2D.h>
#include <Painter/Painter.h>

using namespace Upp;

#define IMAGECLASS AirHockeyImg
#define IMAGEFILE <AirHockey/AirHockey.iml>
#include <Draw/iml_header.h>


#include "World.h"
#include "Agent.h"
#include "Object.h"
#include "Contact.h"
#include "AirHockeyGame.h"

class AirHockey : public TopWindow {
	
	GameCtrl::AirHockey::Table2 table;
	
public:
	typedef AirHockey CLASSNAME;
	AirHockey();
	
};

#endif
