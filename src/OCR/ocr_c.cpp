#include "ocr_c.h"
#include "OCR.h"

using namespace Upp;
using namespace Upp::OCR;

struct OCREngine_t {
    OCREngine engine;
    String last_error;
};

struct OCRResult_t {
    OCRResult result;
};

extern "C" {

OCREngineHandle ocr_engine_create(void) {
    return new OCREngine_t();
}

void ocr_engine_destroy(OCREngineHandle handle) {
    if (handle) {
        delete handle;
    }
}

int ocr_engine_init(OCREngineHandle handle, const char* classifier_path, const char* splitter_path) {
    if (!handle) return 0;

    OCRConfig config;
    if (classifier_path) config.classifier_model_path = classifier_path;
    if (splitter_path) config.splitter_model_path = splitter_path;

    bool success = handle->engine.Initialize(config);
    if (!success) {
        handle->last_error = handle->engine.GetLastError();
    }

    return success ? 1 : 0;
}

OCRResultHandle ocr_recognize_file(OCREngineHandle handle, const char* image_path) {
    if (!handle || !image_path) return nullptr;

    FileIn in(image_path);
    if (!in.IsOpen()) {
        handle->last_error = "Failed to open image file";
        return nullptr;
    }

    Image img = StreamRaster::LoadAny(in);
    if (img.IsEmpty()) {
        handle->last_error = "Failed to load image";
        return nullptr;
    }

    OCRResult_t* res = new OCRResult_t();
    res->result = handle->engine.RecognizePage(img);

    return res;
}

const char* ocr_result_get_text(OCRResultHandle result) {
    if (!result) return "";
    return ~result->result.full_text;
}

double ocr_result_get_confidence(OCRResultHandle result) {
    if (!result) return 0.0;
    return result->result.avg_confidence;
}

void ocr_result_destroy(OCRResultHandle result) {
    if (result) {
        delete result;
    }
}

const char* ocr_get_last_error(OCREngineHandle handle) {
    if (!handle) return "Invalid handle";
    return ~handle->last_error;
}

const char* ocr_get_version(void) {
    static String version = GetOCRVersion();
    return ~version;
}

} // extern "C"
