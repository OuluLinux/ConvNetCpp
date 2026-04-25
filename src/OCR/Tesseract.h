#ifndef _OCR_Tesseract_h_
#define _OCR_Tesseract_h_

#include <Core/Core.h>
#include <Draw/Draw.h>

namespace Upp {
namespace OCR {

bool HasTesseract();
bool RecognizeWithTesseract(const Image& patch, int psm, String& text, double& confidence, const String& whitelist = String(), const String& blacklist = String());

} // namespace OCR
} // namespace Upp

#endif
