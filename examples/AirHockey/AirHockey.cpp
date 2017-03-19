#include "AirHockey.h"

#define IMAGECLASS AirHockeyImg
#define IMAGEFILE <AirHockey/AirHockey.iml>
#include <Draw/iml_source.h>

AirHockey::AirHockey()
{
	Icon(AirHockeyImg::icon());
	Sizeable().MaximizeBox().MinimizeBox().Zoomable();
	Title("UNFINISHED (!!!) AirHockey");
	
	int h = 512;
	int w = h * 3 / 5;
	
	table.Init();
	
	Add(table.SizePos());
	SetRect(0, 0, w, h);
}

