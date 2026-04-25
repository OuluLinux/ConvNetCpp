#ifndef _ConvNetPlugin_ConvNetPlugin_h_
#define _ConvNetPlugin_ConvNetPlugin_h_

#include <ScriptCommon/ScriptCommon.h>
#include <ConvNet/ConvNet.h>

NAMESPACE_UPP

class ConvNetPlugin : public IPlugin, public IPythonBindingProvider {
public:
	virtual String GetID()          const override { return "ConvNetPlugin"; }
	virtual String GetName()        const override { return "ConvNet Plugin"; }
	virtual String GetDescription() const override { return "ConvNet training and inference."; }
	virtual void Init(IPluginContext& ctx) override;
	virtual void Shutdown() override;
	virtual void SyncBindings(PyVM& vm) override;

	ConvNet::Session& GetSession() { return session; }

private:
	IPluginContext* context = nullptr;
	ConvNet::Session session;
};

END_UPP_NAMESPACE

#endif
