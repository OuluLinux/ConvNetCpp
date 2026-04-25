#!/bin/bash
# Patch PlotLib bazaar package to fix Color constructor errors
PLOT_CPP="$HOME/topside-code/bazaar/PlotLib/Plot.cpp"

if [ -f "$PLOT_CPP" ]; then
    echo "Patching $PLOT_CPP..."
    sed -i 's/bgcol(0xffffff,0)/bgcol(255, 255, 255)/g' "$PLOT_CPP"
    sed -i 's/framecol(0xffffff,0)/framecol(255, 255, 255)/g' "$PLOT_CPP"
    sed -i 's/fontcol(0x0,0)/fontcol(0, 0, 0)/g' "$PLOT_CPP"
    sed -i 's/axiscol(0x0,0)/axiscol(0, 0, 0)/g' "$PLOT_CPP"
    
    # Also handle cases where my previous patch failed because it used single-arg constructor which is private
    sed -i 's/bgcol(0xffffff)/bgcol(255, 255, 255)/g' "$PLOT_CPP"
    sed -i 's/framecol(0xffffff)/framecol(255, 255, 255)/g' "$PLOT_CPP"
    sed -i 's/fontcol(0x0)/fontcol(0, 0, 0)/g' "$PLOT_CPP"
    sed -i 's/axiscol(0x0)/axiscol(0, 0, 0)/g' "$PLOT_CPP"
    
    echo "Done."
else
    echo "Error: $PLOT_CPP not found."
    exit 1
fi
