#ifndef _ConvNet_SessionModules_h_
#define _ConvNet_SessionModules_h_

#include "Utilities.h"

namespace ConvNet {

class Session;
class TrainerBase;

typedef bool (*SessionLayerModuleFn)(Session& session, Value& row, String& error);
typedef bool (*SessionTrainerModuleFn)(TrainerBase& trainer, Value& row, String& error);

struct SessionModuleParamSpec : Moveable<SessionModuleParamSpec> {
	String key;
	String widget;
	Value default_value;

	void operator<<=(const SessionModuleParamSpec& src) {
		key = src.key;
		widget = src.widget;
		default_value = src.default_value;
	}
	SessionModuleParamSpec() {}
	SessionModuleParamSpec(const SessionModuleParamSpec& src) { *this <<= src; }
	SessionModuleParamSpec& operator=(const SessionModuleParamSpec& src) { *this <<= src; return *this; }
};

struct SessionLayerModuleSpec : Moveable<SessionLayerModuleSpec> {
	String type;
	String label;
	String category;
	bool   has_layer_stack_input = true;
	bool   has_layer_stack_output = true;
	Vector<SessionModuleParamSpec> params;

	void operator<<=(const SessionLayerModuleSpec& src) {
		type = src.type;
		label = src.label;
		category = src.category;
		has_layer_stack_input = src.has_layer_stack_input;
		has_layer_stack_output = src.has_layer_stack_output;
		params <<= src.params;
	}
	SessionLayerModuleSpec() {}
	SessionLayerModuleSpec(const SessionLayerModuleSpec& src) { *this <<= src; }
	SessionLayerModuleSpec& operator=(const SessionLayerModuleSpec& src) { *this <<= src; return *this; }
};

struct SessionTrainerModuleSpec : Moveable<SessionTrainerModuleSpec> {
	String type;
	String label;
	String category;
	Vector<SessionModuleParamSpec> params;

	void operator<<=(const SessionTrainerModuleSpec& src) {
		type = src.type;
		label = src.label;
		category = src.category;
		params <<= src.params;
	}
	SessionTrainerModuleSpec() {}
	SessionTrainerModuleSpec(const SessionTrainerModuleSpec& src) { *this <<= src; }
	SessionTrainerModuleSpec& operator=(const SessionTrainerModuleSpec& src) { *this <<= src; return *this; }
};

class SessionModuleRegistry {
public:
	static SessionModuleRegistry& Get();

	void RegisterLayer(const String& type, SessionLayerModuleFn fn,
	                   const SessionLayerModuleSpec& spec = SessionLayerModuleSpec());
	void RegisterTrainer(const String& type, SessionTrainerModuleFn fn,
	                     const SessionTrainerModuleSpec& spec = SessionTrainerModuleSpec());

	// Return values:
	//   1  = module found and applied successfully
	//   0  = no module for type
	//  -1  = module found but failed (error text filled)
	int ApplyLayer(const String& type, Session& session, Value& row, String& error) const;
	int ApplyTrainer(const String& type, TrainerBase& trainer, Value& row, String& error) const;
	Vector<String> GetLayerTypes() const;
	Vector<String> GetTrainerTypes() const;
	Vector<SessionLayerModuleSpec> GetLayerSpecs() const;
	Vector<SessionTrainerModuleSpec> GetTrainerSpecs() const;
	bool GetLayerSpec(const String& type, SessionLayerModuleSpec& out) const;
	bool GetTrainerSpec(const String& type, SessionTrainerModuleSpec& out) const;

private:
	VectorMap<String, SessionLayerModuleFn>   layer_modules;
	VectorMap<String, SessionTrainerModuleFn> trainer_modules;
	VectorMap<String, SessionLayerModuleSpec> layer_specs;
	VectorMap<String, SessionTrainerModuleSpec> trainer_specs;
};

// Registers built-in layer/trainer modules once (idempotent).
void RegisterBuiltinSessionModules();

}

#endif
