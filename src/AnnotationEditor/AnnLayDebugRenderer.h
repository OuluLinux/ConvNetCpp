#ifndef _AnnotationEditor_AnnLayDebugRenderer_h_
#define _AnnotationEditor_AnnLayDebugRenderer_h_

#include <AnnLayCore/AnchoredSlotRecognizer.h>
#include <plugin/png/png.h>

NAMESPACE_UPP

class AnnLayDebugRenderer {
public:
	Image Render(const Image& src, const Vector<SlotResult>& results,
	             const AnnLay& lay) const;

	bool RenderToFile(const Image& src, const Vector<SlotResult>& results,
	                  const AnnLay& lay, const String& out_path) const;
};

END_UPP_NAMESPACE

#endif
