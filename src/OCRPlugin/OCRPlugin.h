#ifndef _OCRPlugin_OCRPlugin_h_
#define _OCRPlugin_OCRPlugin_h_

#include <ScriptCommon/ScriptCommon.h>
#include <EditorCommon/Recognition.h>
#include <OCR/OCR.h>

NAMESPACE_UPP

class OcrPlugin : public IPlugin, public ICustomExecuteProvider, public IPythonBindingProvider {
	IPluginContext* context = nullptr;
	OCR::OCREngine engine;
	OCR::OCRConfig config;
	String splitter_model_path;
	String classifier_model_path;
	String last_error;
	String last_report_json;
	bool   save_patches = false;
	String dataset_dir;
	bool   engine_verified = false;

public:
	virtual String GetID() const override { return "OCRPlugin"; }
	virtual String GetName() const override { return "OCR Plugin"; }
	virtual String GetDescription() const override { return "Provides OCR and visual verification support."; }
	virtual void Init(IPluginContext& ctx) override;
	virtual void Shutdown() override;

	// ICustomExecuteProvider
	virtual bool CanExecute(const String& path) override;
	virtual void Execute(const String& path) override;

	// IPythonBindingProvider
	virtual void SyncBindings(PyVM& vm) override;

	void SetModelPaths(const String& splitter, const String& classifier);
	void SetDatasetCollection(bool enable, const String& dir);
	
	PyValue GetReportSchema() const;
	PyValue CaptureFrameInfo();
	PyValue ReadZone(const String& zone_id);
	PyValue ReadZones(const Vector<String>& zone_ids);
	PyValue ReadCards(const String& zone_id);
	PyValue Compare(const PyValue& expected);

	String GetLastError() const;
	String GetLastReportJson() const;

private:
	PyValue ReadZoneInternal(const String& zone_id);
	PyValue BuildReport(const Vector<String>& zone_ids, 
	                    const VectorMap<String, String>* expected,
	                    double min_confidence);
	
	bool EnsureEngine();
	void Log(const String& s);
	void SetError(const String& s);
	bool ResolveGameState(const String& path, String& layout_path) const;
	bool IsVerificationEnabledForPath(const String& path) const;
	IDocumentHost* GetActiveHost() const;
	Image CaptureViewImage(IDocumentHost& host) const;
	Rect  ClampRect(const Rect& r, Size sz) const;
	bool  CollectZoneIdsFromForm(const String& form_path, Vector<String>& zone_ids) const;
	bool  VerifyModel();
	VectorMap<String, Image> LoadTestImages(const String& dir);
};

END_UPP_NAMESPACE

#endif
