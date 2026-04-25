#ifdef _WIN32
#define _WINSOCKAPI_
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mferror.h>
#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfuuid.lib")
#endif

#include <Core/Core.h>
#include <Draw/Draw.h>
#include <plugin/jpg/jpg.h>

#ifdef PLATFORM_LINUX
#include <fcntl.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <unistd.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#endif

using namespace Upp;

static String g_latest_jpeg;
static uint32 g_latest_id = 0;
static Mutex  g_data_mutex;
static std::atomic<bool> g_running(false);
static std::atomic<int64> g_stat_frames(0);
static std::atomic<int64> g_stat_bytes(0);

struct LocalCommandLineArguments {
	int port = 8082;
	int fps = 10;
	int device_idx = 0;
	String source = "camera";
	String image_path;
	String image_dir;
	String screen_rect;
	bool list_devices = false;
	bool show_help = false;
	bool self_test = false;
	bool test_capture = false;
	bool red_test = false;
	bool no_touch = false;
	bool stats = false;
	String wire_format = "mjpeg";

	void Parse(const Vector<String>& args) {
		for(int i = 0; i < args.GetCount(); i++) {
			if(args[i] == "--port" && i + 1 < args.GetCount()) { port = StrInt(args[i+1]); i++; }
			else if(args[i] == "--fps" && i + 1 < args.GetCount()) { fps = StrInt(args[i+1]); i++; }
			else if(args[i] == "--device" && i + 1 < args.GetCount()) { device_idx = StrInt(args[i+1]); i++; }
			else if(args[i] == "--source" && i + 1 < args.GetCount()) { source = ToLower(args[i+1]); i++; }
			else if(args[i] == "--image" && i + 1 < args.GetCount()) { image_path = args[i+1]; i++; }
			else if(args[i] == "--image-dir" && i + 1 < args.GetCount()) { image_dir = args[i+1]; i++; }
			else if(args[i] == "--screen-rect" && i + 1 < args.GetCount()) { screen_rect = args[i+1]; i++; }
			else if(args[i] == "-l") list_devices = true;
			else if(args[i] == "--self-test") self_test = true;
			else if(args[i] == "--test-capture") test_capture = true;
			else if(args[i] == "--red-test") { red_test = true; source = "red"; }
			else if(args[i] == "--no-touch") no_touch = true;
			else if(args[i] == "--stats") stats = true;
			else if(args[i] == "--wire-format" && i + 1 < args.GetCount()) { wire_format = ToLower(args[i+1]); i++; }
			else if(args[i] == "--help" || args[i] == "-h") show_help = true;
		}
		if(source != "camera" && source != "red" && source != "image" && source != "image-dir" && source != "screen")
			source = "camera";
		if (wire_format != "mjpeg" && wire_format != "yuv")
			wire_format = "mjpeg";
	}

	void PrintHelp() {
		Cout() << "VideoServer - Remote Video Capture Server\n\n"
		       << "Usage: VideoServer [options]\n\n"
		       << "Options:\n"
		       << "  -l               List available video capture devices\n"
		       << "  --source <camera|red|image|image-dir|screen>\n"
		       << "                   Set frame source (default: camera)\n"
		       << "  --device <idx>   Set the capture device index (default: 0)\n"
		       << "  --image <path>   Source image path (required when --source image)\n"
		       << "  --image-dir <path>  Directory source for --source image-dir\n"
		       << "  --screen-rect <x,y,w,h>  Screen region (default: full screen)\n"
		       << "  --port <port>    Set the server port (default: 8082)\n"
		       << "  --fps <fps>      Set the target FPS (default: 10)\n"
		       << "  --self-test      Capture one frame from selected source and exit\n"
		       << "  --test-capture   Capture one frame, save to bin/test_capture.jpg, and exit\n"
		       << "  --red-test       Alias for --source red\n"
		       << "  --wire-format <mjpeg|yuv>  Payload format for streaming (default: mjpeg)\n"
		       << "  --no-touch       Skip rescale/extra transforms when possible\n"
		       << "  --stats          Print capture payload stats once per second\n"
		       << "  --help, -h       Show this help message\n";
	}
};

#ifdef _WIN32
String GetGuidName(const GUID& guid) {
	if (guid == MFVideoFormat_RGB32) return "RGB32";
	if (guid == MFVideoFormat_RGB24) return "RGB24";
	if (guid == MFVideoFormat_YUY2)  return "YUY2";
	if (guid == MFVideoFormat_MJPG)  return "MJPEG";
	if (guid == MFVideoFormat_NV12)  return "NV12";
	if (guid == MFMediaType_Video)   return "Video";
	
	const RPC_WSTR *pStr = nullptr;
	String s = "Unknown GUID";
	if (UuidToStringW(&guid, (RPC_WSTR*)&pStr) == RPC_S_OK) {
		s = WString((const wchar_t*)pStr).ToString();
		RpcStringFreeW((RPC_WSTR*)&pStr);
	}
	return s;
}

void DumpMediaType(IMFMediaType* pType) {
	GUID major, sub;
	pType->GetMajorType(&major);
	pType->GetGUID(MF_MT_SUBTYPE, &sub);
	UINT32 w, h;
	MFGetAttributeSize(pType, MF_MT_FRAME_SIZE, &w, &h);
	UINT32 num, den;
	MFGetAttributeRatio(pType, MF_MT_FRAME_RATE, &num, &den);
	
	Cout() << "  Type: " << GetGuidName(major) << "/" << GetGuidName(sub) << "\n";
	Cout() << "  Size: " << w << "x" << h << "\n";
	if (den > 0) Cout() << "  FPS:  " << (double)num / den << "\n";
}

class WinMFCapture {
	IMFSourceReader* reader = nullptr;
	Size size;
	int  stride = 0;
	int  bpp = 4;
	bool opened = false;

public:
	WinMFCapture() {}
	virtual ~WinMFCapture() { Close(); }

	Size GetSize() const { return size; }

	static Vector<String> GetDeviceList() {
		Vector<String> list;
		CoInitializeEx(nullptr, COINIT_MULTITHREADED);
		MFStartup(MF_VERSION);
		IMFAttributes* attrs = nullptr;
		if (SUCCEEDED(MFCreateAttributes(&attrs, 1))) {
			if (SUCCEEDED(attrs->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID))) {
				IMFActivate** devices = nullptr;
				UINT32 count = 0;
				if (SUCCEEDED(MFEnumDeviceSources(attrs, &devices, &count))) {
					for (UINT32 i = 0; i < count; i++) {
						WCHAR* name = nullptr;
						UINT32 nameLen = 0;
						if (SUCCEEDED(devices[i]->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, &name, &nameLen))) {
							list.Add(String(name));
							CoTaskMemFree(name);
						} else list.Add("Unknown Device " + AsString(i));
						devices[i]->Release();
					}
					CoTaskMemFree(devices);
				}
			}
			attrs->Release();
		}
		MFShutdown();
		CoUninitialize();
		return list;
	}

	bool Open(int deviceIndex = 0) {
		if (opened) return true;
		CoInitializeEx(nullptr, COINIT_MULTITHREADED);
		MFStartup(MF_VERSION);
		IMFAttributes* attrs = nullptr;
		MFCreateAttributes(&attrs, 1);
		attrs->SetGUID(MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE, MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID);
		IMFActivate** devices = nullptr;
		UINT32 count = 0;
		if (FAILED(MFEnumDeviceSources(attrs, &devices, &count)) || count == 0) { 
			Cerr() << "No video devices found.\n";
			attrs->Release(); return false; 
		}
		attrs->Release();
		if (deviceIndex >= (int)count) deviceIndex = 0;
		IMFMediaSource* source = nullptr;
		devices[deviceIndex]->ActivateObject(__uuidof(IMFMediaSource), reinterpret_cast<void**>(&source));
		for (UINT32 i = 0; i < count; ++i) devices[i]->Release();
		CoTaskMemFree(devices);
		IMFAttributes* readerAttrs = nullptr;
		MFCreateAttributes(&readerAttrs, 1);
		readerAttrs->SetUINT32(MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, TRUE);
		HRESULT hr = MFCreateSourceReaderFromMediaSource(source, readerAttrs, &reader);
		if (readerAttrs) readerAttrs->Release();
		source->Release();
		if (FAILED(hr)) { Cerr() << "Failed to create SourceReader.\n"; return false; }

		Cout() << "Negotiating media type...\n";
		IMFMediaType* pTargetType = nullptr;
		MFCreateMediaType(&pTargetType);
		pTargetType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
		pTargetType->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32);
		MFSetAttributeSize(pTargetType, MF_MT_FRAME_SIZE, 1920, 1080);
		
		hr = reader->SetCurrentMediaType((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, nullptr, pTargetType);
		pTargetType->Release();
		if (FAILED(hr)) Cout() << "Warning: Could not force RGB32, using device default.\n";

		IMFMediaType* pActualType = nullptr;
		if (SUCCEEDED(reader->GetCurrentMediaType((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, &pActualType))) {
			DumpMediaType(pActualType);
			UINT32 w, h;
			MFGetAttributeSize(pActualType, MF_MT_FRAME_SIZE, &w, &h);
			size = Size(w, h);
			GUID sub;
			pActualType->GetGUID(MF_MT_SUBTYPE, &sub);
			if (sub == MFVideoFormat_RGB24) bpp = 3;
			else bpp = 4;
			
			UINT32 s = 0;
			if (FAILED(pActualType->GetUINT32(MF_MT_DEFAULT_STRIDE, &s))) stride = size.cx * bpp;
			else stride = (int)s;
			pActualType->Release();
		}

		Cout() << "Final format: " << size.cx << "x" << size.cy << " Stride: " << stride << " BPP: " << bpp << "\n";
		opened = true;
		return true;
	}

	void Close() {
		if (reader) { reader->Release(); reader = nullptr; }
		if (opened) { MFShutdown(); CoUninitialize(); }
		opened = false;
	}

	Image GrabFrame() {
		if (!opened || !reader) return Null;
		DWORD streamIndex = 0, flags = 0;
		LONGLONG timestamp = 0;
		IMFSample* sample = nullptr;
		
		// Try a few times because first samples might be null/ticks
		for (int i = 0; i < 10; i++) {
			HRESULT hr = reader->ReadSample((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, 0, &streamIndex, &flags, &timestamp, &sample);
			if (FAILED(hr)) return Null;
			if (sample) break;
			Sleep(10);
		}
		
		if (!sample) return Null;
		
		IMFMediaBuffer* buffer = nullptr;
		sample->ConvertToContiguousBuffer(&buffer);
		if (!buffer) { sample->Release(); return Null; }
		BYTE* data = nullptr;
		DWORD curLen = 0;
		buffer->Lock(&data, nullptr, &curLen);

		ImageBuffer ib(size);
		int abs_stride = abs(stride);
		bool bottom_up = stride < 0;

		for (int y = 0; y < size.cy; y++) {
			const byte* src_row = data + (bottom_up ? (size.cy - 1 - y) : y) * abs_stride;
			RGBA* dst_row = ib[y];
			if (bpp == 4) {
				for (int x = 0; x < size.cx; x++) {
					dst_row[x].r = src_row[x * 4 + 2];
					dst_row[x].g = src_row[x * 4 + 1];
					dst_row[x].b = src_row[x * 4 + 0];
					dst_row[x].a = 255;
				}
			} else {
				for (int x = 0; x < size.cx; x++) {
					dst_row[x].r = src_row[x * 3 + 2];
					dst_row[x].g = src_row[x * 3 + 1];
					dst_row[x].b = src_row[x * 3 + 0];
					dst_row[x].a = 255;
				}
			}
		}
		buffer->Unlock();
		buffer->Release();
		sample->Release();
		return ib;
	}
};

static Rect GetWinVirtualScreenRect()
{
	int x = GetSystemMetrics(SM_XVIRTUALSCREEN);
	int y = GetSystemMetrics(SM_YVIRTUALSCREEN);
	int w = GetSystemMetrics(SM_CXVIRTUALSCREEN);
	int h = GetSystemMetrics(SM_CYVIRTUALSCREEN);
	return RectC(x, y, w, h);
}

static Image CaptureWinScreenFrame(const Rect& rect)
{
	if(rect.IsEmpty())
		return Null;

	const int w = rect.GetWidth();
	const int h = rect.GetHeight();

	HDC screen_dc = GetDC(NULL);
	if(!screen_dc)
		return Null;

	HDC mem_dc = CreateCompatibleDC(screen_dc);
	if(!mem_dc) {
		ReleaseDC(NULL, screen_dc);
		return Null;
	}

	BITMAPINFO bi;
	memset(&bi, 0, sizeof(bi));
	bi.bmiHeader.biSize = sizeof(BITMAPINFOHEADER);
	bi.bmiHeader.biWidth = w;
	bi.bmiHeader.biHeight = -h; // top-down
	bi.bmiHeader.biPlanes = 1;
	bi.bmiHeader.biBitCount = 32;
	bi.bmiHeader.biCompression = BI_RGB;

	void* pixels = nullptr;
	HBITMAP dib = CreateDIBSection(screen_dc, &bi, DIB_RGB_COLORS, &pixels, NULL, 0);
	if(!dib || !pixels) {
		if(dib) DeleteObject(dib);
		DeleteDC(mem_dc);
		ReleaseDC(NULL, screen_dc);
		return Null;
	}

	HGDIOBJ old = SelectObject(mem_dc, dib);
	BOOL ok = BitBlt(mem_dc, 0, 0, w, h, screen_dc, rect.left, rect.top, SRCCOPY | CAPTUREBLT);

	Image out = Null;
	if(ok) {
		ImageBuffer ib(w, h);
		const byte* src = (const byte*)pixels;
		for(int y = 0; y < h; y++) {
			const byte* s = src + y * w * 4;
			RGBA* d = ib[y];
			for(int x = 0; x < w; x++) {
				d[x].b = s[x * 4 + 0];
				d[x].g = s[x * 4 + 1];
				d[x].r = s[x * 4 + 2];
				d[x].a = 255;
			}
		}
		ib.SetKind(IMAGE_OPAQUE);
		out = ib;
	}

	SelectObject(mem_dc, old);
	DeleteObject(dib);
	DeleteDC(mem_dc);
	ReleaseDC(NULL, screen_dc);
	return out;
}
#endif

static void YUYVToImage(const unsigned char* src, int w, int h, ImageBuffer& ib) {
	const unsigned char* s = src;
	for(int y = 0; y < h; y++) {
		RGBA* t = ib[y];
		for(int x = 0; x < w / 2; x++) {
			int y0 = s[0]; int u0 = s[1]; int y1 = s[2]; int v0 = s[3];
			s += 4;
			auto YUV2RGB = [](int y, int u, int v, RGBA& p) {
				int c = y - 16; int d = u - 128; int e = v - 128;
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

static String BuildYuvPacket(int w, int h, const unsigned char* data, int bytes) {
	String pkt;
	pkt.Cat("YUV0", 4);
	uint32 ww = (uint32)w, hh = (uint32)h, sz = (uint32)bytes;
	pkt.Cat((const char*)&ww, 4);
	pkt.Cat((const char*)&hh, 4);
	pkt.Cat((const char*)&sz, 4);
	pkt.Cat((const char*)data, bytes);
	return pkt;
}

static bool ParseScreenRect(const String& text, Rect& out)
{
	Vector<String> p = Split(text, ',');
	if(p.GetCount() != 4)
		return false;
	int x = ScanInt(TrimBoth(p[0]));
	int y = ScanInt(TrimBoth(p[1]));
	int w = ScanInt(TrimBoth(p[2]));
	int h = ScanInt(TrimBoth(p[3]));
	if(w <= 0 || h <= 0)
		return false;
	out = RectC(x, y, w, h);
	return true;
}

static bool IsImageFileExt(const String& path)
{
	String e = ToLower(GetFileExt(path));
	return e == ".png" || e == ".jpg" || e == ".jpeg" || e == ".bmp" || e == ".gif" || e == ".tif" || e == ".tiff";
}

static bool CollectImageFiles(const String& dir, Vector<String>& files)
{
	files.Clear();
	FindFile ff(AppendFileName(dir, "*"));
	while(ff) {
		if(ff.IsFile()) {
			String p = AppendFileName(dir, ff.GetName());
			if(IsImageFileExt(p))
				files.Add(p);
		}
		ff.Next();
	}
	Sort(files);
	return !files.IsEmpty();
}

#ifdef PLATFORM_LINUX
static byte ChannelFromMask(unsigned long pixel, unsigned long mask)
{
	if(mask == 0)
		return 0;
	unsigned long v = pixel & mask;
	int shift = 0;
	while(((mask >> shift) & 1) == 0 && shift < 63)
		shift++;
	v >>= shift;
	unsigned long maxv = mask >> shift;
	if(maxv == 0)
		return 0;
	return (byte)((v * 255 + maxv / 2) / maxv);
}

static Image CaptureX11Frame(::Display* dpy, Window root, const Rect& rect)
{
	if(!dpy || rect.IsEmpty())
		return Null;

	XImage* ximg = XGetImage(dpy, root, rect.left, rect.top, rect.GetWidth(), rect.GetHeight(), AllPlanes, ZPixmap);
	if(!ximg)
		return Null;

	ImageBuffer ib(rect.GetWidth(), rect.GetHeight());
	for(int y = 0; y < rect.GetHeight(); y++) {
		RGBA* row = ib[y];
		for(int x = 0; x < rect.GetWidth(); x++) {
			unsigned long px = XGetPixel(ximg, x, y);
			row[x].r = ChannelFromMask(px, ximg->red_mask);
			row[x].g = ChannelFromMask(px, ximg->green_mask);
			row[x].b = ChannelFromMask(px, ximg->blue_mask);
			row[x].a = 255;
		}
	}
	ib.SetKind(IMAGE_OPAQUE);
	XDestroyImage(ximg);
	return Image(ib);
}
#endif

void ClientThread(TcpSocket* s_ptr) {
	One<TcpSocket> client(s_ptr);
	Cout() << "Client connected from " << client->GetPeerAddr() << "\n";
	while(g_running && client->IsOpen() && !client->IsError() && !client->IsEof()) {
		uint32 last_id = 0;
		if (client->Timeout(5000).GetAll(&last_id, 4)) {
			String data; uint32 id;
			{
				Mutex::Lock __(g_data_mutex);
				id = g_latest_id;
				if (id > last_id) data = g_latest_jpeg;
			}
			uint32 sz = (uint32)data.GetCount();
			if (client->Timeout(5000).Put(&id, 4) && client->Timeout(5000).Put(&sz, 4)) {
				if (sz > 0) client->Put(data);
			}
		} else if (client->IsError() || client->IsEof()) break;
	}
	Cout() << "Client disconnected.\n";
}

CONSOLE_APP_MAIN {
	LocalCommandLineArguments cla;
	cla.Parse(CommandLine());
	if (cla.show_help) { cla.PrintHelp(); return; }
	if (cla.list_devices) {
		#ifdef _WIN32
		Vector<String> devices = WinMFCapture::GetDeviceList();
		Cout() << "Available devices:\n";
		for (int i = 0; i < devices.GetCount(); i++) Cout() << i << ": " << devices[i] << "\n";
		#endif
		return;
	}

	if(cla.source == "image" && cla.image_path.IsEmpty()) {
		Cerr() << "--image is required when --source image\n";
		return;
	}
	if(cla.source == "image-dir" && cla.image_dir.IsEmpty()) {
		Cerr() << "--image-dir is required when --source image-dir\n";
		return;
	}
#if !defined(PLATFORM_LINUX) && !defined(_WIN32)
	if(cla.source == "screen") {
		Cerr() << "--source screen is supported on Linux and Windows only\n";
		return;
	}
#endif
	Rect screen_rect_req;
	bool has_screen_rect_req = false;
	if(cla.source == "screen" && !cla.screen_rect.IsEmpty()) {
		if(!ParseScreenRect(cla.screen_rect, screen_rect_req)) {
			Cerr() << "Invalid --screen-rect: " << cla.screen_rect << "\n";
			return;
		}
		has_screen_rect_req = true;
	}
	Vector<String> image_dir_files;
	if(cla.source == "image-dir") {
		if(!CollectImageFiles(cla.image_dir, image_dir_files)) {
			Cerr() << "No image files found in --image-dir: " << cla.image_dir << "\n";
			return;
		}
	}

	if (cla.self_test) {
		Image img;
		if(cla.source == "red") {
			ImageBuffer ib(1920, 1080);
			for(int y = 0; y < 1080; y++) {
				RGBA* row = ib[y];
				for(int x = 0; x < 1920; x++) {
					row[x].r = 255;
					row[x].g = 0;
					row[x].b = 0;
					row[x].a = 255;
				}
			}
			ib.SetKind(IMAGE_OPAQUE);
			img = ib;
		}
		else if(cla.source == "image") {
			img = StreamRaster::LoadFileAny(cla.image_path);
		}
		else if(cla.source == "image-dir") {
			img = StreamRaster::LoadFileAny(image_dir_files[0]);
		}
		else if(cla.source == "screen") {
#ifdef PLATFORM_LINUX
			::Display* dpy = XOpenDisplay(nullptr);
			if(!dpy) {
				Cerr() << "Self-test failed: cannot open X11 display\n";
				Exit(1);
			}
			int sn = DefaultScreen(dpy);
			Window root = RootWindow(dpy, sn);
			Rect rect = RectC(0, 0, DisplayWidth(dpy, sn), DisplayHeight(dpy, sn));
			if(has_screen_rect_req)
				rect &= screen_rect_req;
			img = CaptureX11Frame(dpy, root, rect);
			XCloseDisplay(dpy);
#elif defined(_WIN32)
			Rect rect = GetWinVirtualScreenRect();
			if(has_screen_rect_req)
				rect &= screen_rect_req;
			if(rect.IsEmpty()) {
				Cerr() << "Self-test failed: screen capture rect is empty\n";
				Exit(1);
			}
			img = CaptureWinScreenFrame(rect);
#else
			Cerr() << "Self-test failed: --source screen is not supported on this platform\n";
			Exit(1);
#endif
		}
		else if(cla.source == "camera") {
#ifdef _WIN32
			WinMFCapture win_cap;
			if(win_cap.Open(cla.device_idx)) {
				img = win_cap.GrabFrame();
				win_cap.Close();
			}
#elif defined(PLATFORM_LINUX)
			String device = "/dev/video" + AsString(cla.device_idx);
			int fd = open(device, O_RDWR);
			if(fd != -1) {
				v4l2_format fmt; memset(&fmt, 0, sizeof(fmt));
				fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
				fmt.fmt.pix.width = 1920; fmt.fmt.pix.height = 1080;
				fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
				ioctl(fd, VIDIOC_S_FMT, &fmt);
				int w = fmt.fmt.pix.width;
				int h = fmt.fmt.pix.height;
				int bsize = w * h * 2;
				Buffer<unsigned char> buf(bsize);
				if(read(fd, ~buf, bsize) == bsize) {
					ImageBuffer ib(w, h);
					YUYVToImage(~buf, w, h, ib);
					ib.SetKind(IMAGE_OPAQUE);
					img = ib;
				}
				close(fd);
			}
#endif
		}

		if(img.IsEmpty()) {
			Cerr() << "Self-test failed: could not capture frame from source '" << cla.source << "'\n";
			Exit(1);
		}

		String path = AppendFileName(GetFileDirectory(GetExeFilePath()), "self_test_capture.jpg");
		if(!JPGEncoder().Quality(95).SaveFile(path, img)) {
			Cerr() << "Self-test failed: captured frame but failed to save " << path << "\n";
			Exit(1);
		}
		Cout() << "Self-test PASS source=" << cla.source
		       << " size=" << img.GetWidth() << "x" << img.GetHeight()
		       << " saved=" << path << "\n";
		return;
	}

	if (cla.test_capture) {
		#ifdef _WIN32
		WinMFCapture win_cap;
		if (win_cap.Open(cla.device_idx)) {
			Image img = win_cap.GrabFrame();
			if (!img.IsEmpty()) {
				String path = AppendFileName(GetFileDirectory(GetExeFilePath()), "test_capture.jpg");
				if (JPGEncoder().Quality(100).SaveFile(path, img)) {
					Cout() << "Frame captured and saved to " << path << "\n";
				} else {
					Cerr() << "Failed to save JPEG to " << path << "\n";
				}
			} else {
				Cerr() << "Failed to grab frame.\n";
			}
			win_cap.Close();
		} else {
			Cerr() << "Failed to open device.\n";
		}
		#else
		Cout() << "Test capture not implemented for this platform.\n";
		#endif
		return;
	}

	g_running = true;
	int interval = 1000 / (cla.fps > 0 ? cla.fps : 10);
	
	TcpSocket listener;
	if (!listener.Listen(cla.port, 5)) { Cerr() << "Failed to listen on port " << cla.port << "\n"; return; }
	listener.Timeout(200);
	Cout() << "Video Server listening on port " << cla.port << "\n";
	Cout() << "Source: " << cla.source << ", wire format: " << cla.wire_format
	       << ", no-touch: " << (cla.no_touch ? "on" : "off") << "\n";

	Thread().Run([&] {
		#ifdef _WIN32
		WinMFCapture win_cap;
		Rect win_screen_rect;
		if (cla.source == "camera") {
			if (!win_cap.Open(cla.device_idx)) { Cerr() << "Failed to open device.\n"; g_running = false; return; }
		}
		else if(cla.source == "screen") {
			win_screen_rect = GetWinVirtualScreenRect();
			if(has_screen_rect_req)
				win_screen_rect &= screen_rect_req;
			if(win_screen_rect.IsEmpty()) {
				Cerr() << "Screen capture rect is empty\n";
				g_running = false;
				return;
			}
			Cout() << "Screen capture rect: " << win_screen_rect.left << "," << win_screen_rect.top
			       << " " << win_screen_rect.GetWidth() << "x" << win_screen_rect.GetHeight() << "\n";
		}
		#endif
		#ifdef PLATFORM_LINUX
		String device = "/dev/video" + AsString(cla.device_idx);
		int fd = -1;
		int actualW = 1920, actualH = 1080;
		::Display* xdisplay = nullptr;
		Window xroot = 0;
		Rect xrect;
		if (cla.source == "camera") {
			fd = open(device, O_RDWR);
			if(fd != -1) {
				v4l2_format fmt; memset(&fmt, 0, sizeof(fmt));
				fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
				fmt.fmt.pix.width = 1920; fmt.fmt.pix.height = 1080;
				fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
				ioctl(fd, VIDIOC_S_FMT, &fmt);
				actualW = fmt.fmt.pix.width; actualH = fmt.fmt.pix.height;
			} else { Cerr() << "Failed to open " << device << "\n"; g_running = false; return; }
		}
		else if(cla.source == "screen") {
			xdisplay = XOpenDisplay(nullptr);
			if(!xdisplay) {
				Cerr() << "Failed to open X11 display for --source screen\n";
				g_running = false;
				return;
			}
			int sn = DefaultScreen(xdisplay);
			xroot = RootWindow(xdisplay, sn);
			xrect = RectC(0, 0, DisplayWidth(xdisplay, sn), DisplayHeight(xdisplay, sn));
			if(has_screen_rect_req)
				xrect &= screen_rect_req;
			if(xrect.IsEmpty()) {
				Cerr() << "Screen capture rect is empty\n";
				g_running = false;
				XCloseDisplay(xdisplay);
				return;
			}
			Cout() << "Screen capture rect: " << xrect.left << "," << xrect.top
			       << " " << xrect.GetWidth() << "x" << xrect.GetHeight() << "\n";
		}
		#endif

		TimeStop ts;
		TimeStop stats_ts;
		int image_dir_index = 0;
		while (g_running) {
			Image img;
			String payload;
			if (cla.source == "red") {
				if (cla.wire_format == "yuv") {
					const int w = 1920, h = 1080;
					Buffer<byte> yuyv(w * h * 2);
					byte* d = ~yuyv;
					for (int i = 0; i < w * h; i += 2) {
						*d++ = 82;  // Y for red
						*d++ = 90;  // U
						*d++ = 82;  // Y
						*d++ = 240; // V
					}
					payload = BuildYuvPacket(w, h, ~yuyv, w * h * 2);
				} else {
					ImageBuffer ib(1920, 1080);
					for (int y = 0; y < 1080; y++) {
						RGBA* row = ib[y];
						for (int x = 0; x < 1920; x++) {
							row[x].r = 255;
							row[x].g = 0;
							row[x].b = 0;
							row[x].a = 255;
						}
					}
					img = ib;
				}
			} else if(cla.source == "image") {
				img = StreamRaster::LoadFileAny(cla.image_path);
				if(img.IsEmpty()) {
					Cerr() << "Failed to load image source: " << cla.image_path << "\n";
					g_running = false;
					break;
				}
			} else if(cla.source == "image-dir") {
				if(image_dir_files.IsEmpty()) {
					Cerr() << "No image files available in --image-dir\n";
					g_running = false;
					break;
				}
				const String& p = image_dir_files[image_dir_index % image_dir_files.GetCount()];
				image_dir_index++;
				img = StreamRaster::LoadFileAny(p);
				if(img.IsEmpty()) {
					Cerr() << "Failed to load image source from --image-dir: " << p << "\n";
					g_running = false;
					break;
				}
	} else if(cla.source == "screen") {
			#ifdef PLATFORM_LINUX
				img = CaptureX11Frame(xdisplay, xroot, xrect);
				if(img.IsEmpty()) {
					Cerr() << "Failed to capture X11 screen frame\n";
					g_running = false;
					break;
				}
			#elif defined(_WIN32)
				img = CaptureWinScreenFrame(win_screen_rect);
				if(img.IsEmpty()) {
					Cerr() << "Failed to capture Windows screen frame\n";
					g_running = false;
					break;
				}
			#else
				Cerr() << "--source screen is not supported on this platform\n";
				g_running = false;
				break;
			#endif
			} else {
			#ifdef _WIN32
			img = win_cap.GrabFrame();
			if (cla.wire_format == "yuv")
				cla.wire_format = "mjpeg";
			#endif
			#ifdef PLATFORM_LINUX
			int bsize = actualW * actualH * 2;
			Buffer<unsigned char> buf(bsize);
			if (read(fd, ~buf, bsize) == bsize) {
				if (cla.wire_format == "yuv")
					payload = BuildYuvPacket(actualW, actualH, ~buf, bsize);
				else {
					ImageBuffer ib(actualW, actualH);
					YUYVToImage(~buf, actualW, actualH, ib);
					img = ib;
				}
			}
			#endif
			}

			if (payload.IsEmpty() && !img.IsEmpty()) {
				if (!cla.no_touch && img.GetSize() != Size(1920, 1080))
					img = Rescale(img, 1920, 1080);
				payload = JPGEncoder().Quality(100).SaveString(img);
			}
			if (!payload.IsEmpty()) {
				{
					Mutex::Lock __(g_data_mutex);
					g_latest_jpeg = payload;
					g_latest_id++;
				}
				g_stat_frames.fetch_add(1, std::memory_order_relaxed);
				g_stat_bytes.fetch_add(payload.GetCount(), std::memory_order_relaxed);
			}
			if (cla.stats && stats_ts.Elapsed() >= 1000000) {
				static int64 last_frames = 0;
				static int64 last_bytes = 0;
				int64 frames = g_stat_frames.load(std::memory_order_relaxed);
				int64 bytes = g_stat_bytes.load(std::memory_order_relaxed);
				int64 df = frames - last_frames;
				int64 db = bytes - last_bytes;
				double avg = df > 0 ? (double)db / (double)df : 0.0;
				Cout() << "stats fps=" << df << " payload_avg=" << (int)avg << "B total_frames=" << frames << "\n";
				last_frames = frames;
				last_bytes = bytes;
				stats_ts.Reset();
			}
			int elapsed = (int)ts.Elapsed();
			if(elapsed < interval) Sleep(interval - elapsed);
			else Sleep(1);
			ts.Reset();
		}
		#ifdef PLATFORM_LINUX
		if (fd != -1) close(fd);
		if (xdisplay) XCloseDisplay(xdisplay);
		#endif
	});

	while (g_running) {
		TcpSocket *ns = new TcpSocket;
		ns->Timeout(200);
		if(ns->Accept(listener)) Thread().Run([=] { ClientThread(ns); });
		else {
			delete ns;
			if(!g_running)
				break;
			if (listener.IsError()) { Cerr() << "Listener error: " << listener.GetErrorDesc() << "\n"; break; }
		}
		Sleep(10);
	}
	g_running = false;
	Cout() << "Video Server exiting.\n";
}
