#include "OCRForm.h"

NAMESPACE_UPP

void OCRFormPlugin::Init(IPluginContext& ctx)
{
	context = &ctx;
	ctx.RegisterFileTypeHandler(file_type_handler);
	OcrAnalyserRegistry::Get().Register(manual_analyser);

	VideoSourceRegistry& video_sources = VideoSourceRegistry::Get();
	video_sources.Register("static_image", "Static Image", [] { return new StaticImageSource(); });
	video_sources.Register("screen", "Screen Capture", [] { return new ScreenCaptureSource(); });
	video_sources.Register("network", "VideoServer", [] { return new NetworkVideoSource(); });
}

void OCRFormPlugin::Shutdown()
{
	context = nullptr;
}

END_UPP_NAMESPACE
