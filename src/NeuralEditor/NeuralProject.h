#ifndef _NeuralEditor_NeuralProject_h_
#define _NeuralEditor_NeuralProject_h_

#include <Core/Core.h>

NAMESPACE_UPP

struct NeuralProject {
	int    version = 1;
	String name = "new_project";
	Vector<String> graphs;          // .nngrf paths (relative to .nnprj)
	String startup_graph;           // optional default graph
	ValueMap metadata;

	void Jsonize(JsonIO& jio);
	bool Load(const String& path);
	bool Save(const String& path) const;
};

struct NeuralSolution {
	int    version = 1;
	String name = "new_solution";
	Vector<String> projects;        // .nnprj paths (relative to .nnsln)
	String active_project;
	ValueMap metadata;

	void Jsonize(JsonIO& jio);
	bool Load(const String& path);
	bool Save(const String& path) const;
};

END_UPP_NAMESPACE

#endif
