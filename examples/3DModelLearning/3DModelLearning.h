#ifndef _3DModelLearning_3DModelLearning_h
#define _3DModelLearning_3DModelLearning_h

#include <CtrlLib/CtrlLib.h>
#include <Docking/Docking.h>
#include <ConvNetCtrl/ConvNetCtrl.h>

using namespace Upp;
using namespace ConvNet;

template <class T>
struct WithCtrlPanelLayout : public T {
	ParentCtrl m_ctrl;
	void CtrlLayout(T& parent) {
		parent.Add(m_ctrl.SizePos());
	}
};

// Custom control for 3D visualization
class Model3DControl : public ParentCtrl {
private:
    String modelType;
    Vector<Point> points;
    Vector<Pointf> vertices;
    Vector<int> indices;

public:
    Model3DControl();
    virtual void Paint(Draw& draw);

    void SetModelType(const String& type);
    void SetPointCloud(const Vector<Point>& pts);
    void SetMesh(const Vector<Pointf>& verts, const Vector<int>& idx);
    
    typedef Model3DControl CLASSNAME;
};

class Model3DApp : public WithCtrlPanelLayout<DockWindow> {
public:
	ParentCtrl settings;
	Label lrate, lmom, lbatch, ldecay;
	EditDouble rate, mom, decay;
	EditInt batch;
	Button apply, save_net, load_net, pause;
	TrainingGraph graph;
	Label status;
	SessionConvLayers layer_view;

	// Network
	ParentCtrl net_ctrl;
	DocEdit net_edit;
	Button reload_btn;

	Splitter v_split;

	Session ses;
	String t;
	SpinLock ticking_lock;
	int average_size;
	int max_diff_imgs;

	// 3D-specific controls
	Model3DControl model_display;
	Button generate_btn;
	Button train_vae_btn;
	Button train_gan_btn;
	Button load_model_btn;
	Button save_model_btn;
	Button render_3d_btn;
	DropList model_type;

public:
	typedef Model3DApp CLASSNAME;
	Model3DApp();
	~Model3DApp();

	void Init() {}

	virtual void DockInit();

	Session& GetSession() {return ses;}

	void Refresher();
	void ApplySettings();
	void OpenFile();
	void SaveFile();
	void Reload();
	void Pause();
	void RefreshStatus();

	void UpdateNetParamDisplay();
	void ResetAll();
	void PostReload() {PostCallback(THISBACK(Reload));}

	// 3D model-specific functionality
	void OnGenerate();
	void OnTrainVAE();
	void OnTrainGAN();
	void OnModelTypeChanged();
	void OnLoad3DModel();
	void OnSave3DModel();
	void OnRender3D();

	String BuildVoxelAEConfig();
	String BuildPointNetConfig();
	String BuildOccupancyNetConfig();
	String BuildDeepSDFConfig();
};

#endif
