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
		AddButton("CIFAR-10 learner",		THISBACK1(Set, 2));
		AddButton("CIFAR-10 autoencoder",	THISBACK1(Set, 3));
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
	int loader, type;
	{
		OpenDialog odlg;
		odlg.Run();
		if (odlg.ret == -1) return;
		int r = odlg.ret;
		loader	= r == 0 || r == 1 ? LOADER_MNIST : LOADER_CIFAR10;
		type	= r == 0 || r == 2 ? TYPE_LEARNER : TYPE_AUTOENCODER;
	}
	
	{
		ClassifyImages ci(loader, type);
		
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
