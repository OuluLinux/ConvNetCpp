#include "ClassifyImages.h"

class OpenDialog : public TopWindow {
	
	Array<Button> btns;
	Splitter split;
	
public:
	typedef OpenDialog CLASSNAME;
	
	OpenDialog() {
		Add(split.SizePos());
		split.Vert();
		ret = -1;
		AddButton("MNIST learner",			THISBACK1(Set, 0));
		AddButton("MNIST autoencoder",		THISBACK1(Set, 1));
		AddButton("MNIST convolution",		THISBACK1(Set, 2));
		AddButton("CIFAR-10 learner",		THISBACK1(Set, 3));
		AddButton("CIFAR-10 autoencoder",	THISBACK1(Set, 4));
		AddButton("CIFAR-10 convolution",	THISBACK1(Set, 5));
		AddButton("Exit",					THISBACK(Close));
		SetRect(0,0,240,320);
		Title("ConvNetC++ ClassifyImages open dialog");
	}
	
	void AddButton(String s, Callback cb) {
		Button& btn = btns.Add();
		btn.SetLabel(s);
		btn <<= cb;
		split << btn;
	}
	void Set(int i) {ret = i; Close();}
	void Close0() {Close();}
	
	int ret;
};

GUI_APP_MAIN {
	const Vector<String>& args = CommandLine();
	bool test_node = false;
	for(int i = 0; i < args.GetCount(); i++) {
		if(args[i] == "--test-node-render") {
			test_node = true;
			break;
		}
	}

	int loader, type;
	if(test_node) {
		loader = LOADER_MNIST;
		type = TYPE_LEARNER;
	}
	else {
		OpenDialog odlg;
		odlg.Run();
		if (odlg.ret == -1) return;
		int r = odlg.ret;
		loader	= r == 0 || r == 1  || r == 2 ? LOADER_MNIST : LOADER_CIFAR10;
		type	= r == 0 || r == 3 ? TYPE_LEARNER : (r == 1 || r == 4 ? TYPE_AUTOENCODER : TYPE_CONV);
	}
	
	{
		ClassifyImages ci(loader, type);
		
		if(test_node) {
			// Initialize dummy data dimensions to avoid crash in TrainBegin
			ci.GetSession().Data().BeginDataClass(10, 1, 24, 24, 1, 1);
			ci.GetSession().Data().EndData();

			// In test mode, we don't need real data, just the network
			ci.Reload(); 
			
			// Give it a moment to settle (e.g. if any async loading or layout happens)
			Sleep(1000);
			
			ci.SyncGraph();
			
			String path = GetExeDirFile("node_test.png");
			Cout() << "Saving viewport image to: " << path << "\n";
			ci.SaveViewportImage(path);
			
			if(ci.IsViewportWhite()) {
				Cout() << "TEST_FAILED: Viewport is white\n";
				Exit(1);
			} else {
				Cout() << "TEST_PASSED: Viewport is not white\n";
				Exit(0);
			}
		}

		{
			if (loader == LOADER_MNIST){
				LoaderMNIST l(ci.GetSession());
				l.Run();
				if (l.IsFail()) return;
			}
			else if (loader == LOADER_CIFAR10) {
				LoaderCIFAR10 l(ci.GetSession());
				l.Run();
				if (l.IsFail()) return;
			}
			else return;
		}
		
		{
			ci.PostReload();
			ci.Run();
		}
	}
	
	Thread::ShutdownThreads();
}
