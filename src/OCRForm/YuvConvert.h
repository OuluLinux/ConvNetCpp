#ifndef _OCRForm_YuvConvert_h_
#define _OCRForm_YuvConvert_h_

#include <Draw/Draw.h>

NAMESPACE_UPP

inline void YUYVToImage(const unsigned char* src, int w, int h, ImageBuffer& ib)
{
	const unsigned char* s = src;
	for(int y = 0; y < h; y++) {
		RGBA* t = ib[y];
		for(int x = 0; x < w / 2; x++) {
			int y0 = s[0];
			int u0 = s[1];
			int y1 = s[2];
			int v0 = s[3];
			s += 4;

			auto YUV2RGB = [](int yy, int u, int v, RGBA& p) {
				int c = yy - 16;
				int d = u - 128;
				int e = v - 128;
				p.r = (byte)clamp((298 * c + 409 * e + 128) >> 8, 0, 255);
				p.g = (byte)clamp((298 * c - 100 * d - 208 * e + 128) >> 8, 0, 255);
				p.b = (byte)clamp((298 * c + 516 * d + 128) >> 8, 0, 255);
				p.a = 255;
			};

			YUV2RGB(y0, u0, v0, t[0]);
			YUV2RGB(y1, u0, v0, t[1]);
			t += 2;
		}
	}
}

END_UPP_NAMESPACE

namespace OCRForm {
using Upp::YUYVToImage;
}

#endif
