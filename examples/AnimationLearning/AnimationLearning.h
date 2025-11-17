#ifndef _AnimationLearning_AnimationLearning_h
#define _AnimationLearning_AnimationLearning_h

#include <CtrlLib/CtrlLib.h>
#include <Docking/Docking.h>
#include <ConvNetCtrl/ConvNetCtrl.h>

using namespace Upp;
using namespace ConvNet;

#define LAYOUTFILE <AnimationLearning/AnimationLearning.lay>
#include <CtrlCore/lay.h>

#define IMAGECLASS AnimImg
#define IMAGEFILE <AnimationLearning/AnimationLearning.iml>
#include <Draw/iml_header.h>

// Custom control for animation visualization
class AnimationControl : public ParentCtrl {
private:
    // Skeleton/joint data representation
    Vector<Pointf> joints;  // Joint positions (x, y)
    Vector<Vector<int>> connections;  // Joint connections (bone structure)
    int currentFrame;
    int totalFrames;

public:
    AnimationControl();
    virtual void Paint(Draw& draw);

    void SetJoints(const Vector<Pointf>& jnts);
    void SetConnections(const Vector<Vector<int>>& conns);
    void SetFrame(int frame);
    void SetTotalFrames(int total);
    
    typedef AnimationControl CLASSNAME;
};

class AnimationApp : public DockWindow {
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

	// Animation-specific controls
	ParentCtrl anim_ctrl;
	AnimationControl anim_display;
	Button generate_anim_btn;
	Button train_rnn_btn;
	Button train_transformer_btn;
	Button load_motion_btn;
	Button save_motion_btn;
	Button play_btn;
	Button pause_btn;
	Button stop_btn;
	DropList anim_type;

public:
	typedef AnimationApp CLASSNAME;
	AnimationApp();
	~AnimationApp();

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

	// Animation-specific functionality
	void OnGenerateAnim();
	void OnTrainRNN();
	void OnTrainTransformer();
	void OnAnimTypeChanged();
	void OnLoadMotion();
	void OnSaveMotion();
	void OnPlay();
	void OnPause();
	void OnStop();

	String BuildLSTMSkeletonConfig();  // LSTM-based skeleton animation
	String BuildTransformerSkeletonConfig(); // Transformer-based animation
	String BuildVAESkeletonConfig(); // VAE-based animation
};

#endif