#ifndef _ConvNetCtrl_ConvNetCtrl_h_
#define _ConvNetCtrl_ConvNetCtrl_h_

#include <CtrlLib/CtrlLib.h>

#include "LayerCtrl.h"
#include "PointCtrl.h"
#include "BarView.h"
#include "HeatmapView.h"
#include "ImagePrediction.h"
#include "ConvLayerCtrl.h"
#include "TrainingGraph.h"

namespace ConvNet {
using namespace Upp;

inline Color Rainbow(double f)
{
    double div = f * 6;
    byte ascending = (int) ((div - floor(div)) * 255);
    byte descending = 255 - ascending;

    switch ((int) div)
    {
        case 0:
            return Color(255, ascending, 0);
        case 1:
            return Color(descending, 255, 0);
        case 2:
            return Color(0, 255, ascending);
        case 3:
            return Color(0, descending, 255);
        case 4:
            return Color(ascending, 0, 255);
        default: // case 5:
            return Color(255, 0, descending);
    }
}

}

#endif
