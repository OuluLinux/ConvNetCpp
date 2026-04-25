#ifndef _AnnotationEditor_RecognitionScript_h_
#define _AnnotationEditor_RecognitionScript_h_

#include <ByteVM/ByteVM.h>
#include "AnchoredSlotRecognizer.h"

NAMESPACE_UPP

class RecognitionScript {
public:
	bool Load(const String& script_path);
	bool IsLoaded() const { return loaded_; }
	String GetLastError() const { return last_error_; }
	String GetLastOutput() const { return last_output_; }

	VectorMap<String, String> Run(const Vector<SlotResult>& raw_results);

private:
	PyVM   vm_;
	bool   loaded_ = false;
	String last_error_;
	String last_output_;

	void RegisterModule(PyVM& vm,
	                    const Vector<SlotResult>& results,
	                    VectorMap<String, String>& out_meta);
};

END_UPP_NAMESPACE
#endif
