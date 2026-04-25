#ifndef OCR_C_H
#define OCR_C_H

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle types
typedef struct OCREngine_t* OCREngineHandle;
typedef struct OCRResult_t* OCRResultHandle;

// Initialization
OCREngineHandle ocr_engine_create(void);
void ocr_engine_destroy(OCREngineHandle engine);
int ocr_engine_init(OCREngineHandle engine, const char* classifier_path, const char* splitter_path);

// Recognition
OCRResultHandle ocr_recognize_file(OCREngineHandle engine, const char* image_path);
const char* ocr_result_get_text(OCRResultHandle result);
double ocr_result_get_confidence(OCRResultHandle result);
void ocr_result_destroy(OCRResultHandle result);

// Error handling
const char* ocr_get_last_error(OCREngineHandle engine);

// Version
const char* ocr_get_version(void);

#ifdef __cplusplus
}
#endif

#endif // OCR_C_H
