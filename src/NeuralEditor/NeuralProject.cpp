#include "NeuralProject.h"

NAMESPACE_UPP

void NeuralProject::Jsonize(JsonIO& jio) {
	jio("version", version)
	   ("name", name)
	   ("graphs", graphs)
	   ("startup_graph", startup_graph)
	   ("metadata", metadata);
}

bool NeuralProject::Load(const String& path) {
	String json = LoadFile(path);
	if(json.IsEmpty())
		return false;
	Value root = ParseJSON(json);
	if(IsNull(root) || !IsValueMap(root))
		return false;
	NeuralProject tmp;
	LoadFromJsonValue(tmp, root);
	if(tmp.version <= 0) tmp.version = 1;
	if(tmp.version > 1) return false;
	*this = pick(tmp);
	return true;
}

bool NeuralProject::Save(const String& path) const {
	ValueMap root;
	ValueArray graphs_arr;
	for(const String& g : graphs)
		graphs_arr.Add(g);
	root.Add("version", 1);
	root.Add("name", name);
	root.Add("graphs", graphs_arr);
	root.Add("startup_graph", startup_graph);
	root.Add("metadata", metadata);
	return SaveFile(path, StoreAsJson(root, true));
}

void NeuralSolution::Jsonize(JsonIO& jio) {
	jio("version", version)
	   ("name", name)
	   ("projects", projects)
	   ("active_project", active_project)
	   ("metadata", metadata);
}

bool NeuralSolution::Load(const String& path) {
	String json = LoadFile(path);
	if(json.IsEmpty())
		return false;
	Value root = ParseJSON(json);
	if(IsNull(root) || !IsValueMap(root))
		return false;
	NeuralSolution tmp;
	LoadFromJsonValue(tmp, root);
	if(tmp.version <= 0) tmp.version = 1;
	if(tmp.version > 1) return false;
	*this = pick(tmp);
	return true;
}

bool NeuralSolution::Save(const String& path) const {
	ValueMap root;
	ValueArray projects_arr;
	for(const String& p : projects)
		projects_arr.Add(p);
	root.Add("version", 1);
	root.Add("name", name);
	root.Add("projects", projects_arr);
	root.Add("active_project", active_project);
	root.Add("metadata", metadata);
	return SaveFile(path, StoreAsJson(root, true));
}

END_UPP_NAMESPACE
