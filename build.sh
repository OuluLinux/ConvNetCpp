#!/usr/bin/sh
if [ "$1" = "--clean" ]; then
    umk examples,tutorial,src,$HOME/upp/bazaar,$HOME/upp/uppsrc "$1" ~/.config/u++/theide/CLANG.bm -bdsa +GUI,DEBUG_FULL "bin/${1}"
else
    umk examples,tutorial,src,$HOME/upp/bazaar,$HOME/upp/uppsrc "$1" ~/.config/u++/theide/CLANG.bm -bds +GUI,DEBUG_FULL "bin/${1}"
fi
