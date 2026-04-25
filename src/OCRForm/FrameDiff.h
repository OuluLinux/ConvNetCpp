#ifndef _OCRForm_FrameDiff_h_
#define _OCRForm_FrameDiff_h_

#include <Core/Core.h>
#include <Draw/Draw.h>

NAMESPACE_UPP

Image ComputeChangedAreaMask(const Image& prev, const Image& curr,
                             int threshold = 30, int block_size = 8);
Vector<Rect> ComputeChangedBlocks(const Image& prev, const Image& curr,
                                  int threshold = 30, int block_size = 8);

END_UPP_NAMESPACE

namespace OCRForm {
using Upp::ComputeChangedAreaMask;
using Upp::ComputeChangedBlocks;
}

#endif
