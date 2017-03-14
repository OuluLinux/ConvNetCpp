#include "GridWorld.h"

#define IMAGECLASS GridWorldImg
#define IMAGEFILE <GridWorld/GridWorld.iml>
#include <Draw/iml_source.h>

GridWorld::GridWorld() {
	Title("GridWorld");
	Icon(GridWorldImg::icon());
	Sizeable().MaximizeBox().MinimizeBox();
	
	
	//agent = NULL;
	//Rarr = NULL; // reward array
	//T = NULL; // cell types, 0 = normal, 1 = cliff
	Reset();
}
