#include "LoaderMNIST.h"

namespace ConvNet {
using namespace Upp;

LoaderMNIST::LoaderMNIST() {
	ImageDraw id(64, 64);
	id.DrawRect(0,0,64,64,Black());
	id.DrawText(0,0, "M", Arial(64), White());
	Icon(id);
	
	Title("Loading MNIST images");
	Add(lbl.HSizePos(4, 4).TopPos(3, 24));
	Add(prog.HSizePos(4, 4).TopPos(33, 12));
	Add(sub.HSizePos(4, 4).TopPos(33+15, 12));
	Add(cancel.HCenterPos(100).TopPos(63, 24));
	SetRect(0, 0, 400, 90);
	cancel.SetLabel("Cancel");
	cancel <<= THISBACK(Cancel);
	prog.Set(0, 1);
	sub.Set(0, 1);
	ret_value = 0;
	
	
	ImageBank& bank = GetImageBank();
	bank.classes.Clear();
	bank.classes.Add("0");
	bank.classes.Add("1");
	bank.classes.Add("2");
	bank.classes.Add("3");
	bank.classes.Add("4");
	bank.classes.Add("5");
	bank.classes.Add("6");
	bank.classes.Add("7");
	bank.classes.Add("8");
	bank.classes.Add("9");
	
	fast_cache = false;
	
	Thread::Start(THISBACK(Load));
}

void LoaderMNIST::Progress(int actual, int total, String label) {
	LOG("Progress " << actual << "/" << total);
	prog.Set(actual, total);
	sub.Set(0, 1);
	lbl.SetLabel(label);
}

bool LoaderMNIST::SubProgress(int actual, int total) {
	GuiLock __;
	sub.Set(actual, total);
	return false;
}

void LoaderMNIST::Load() {
	
	ImageBank& bank = GetImageBank();
	
	// Try to load cached file, which is a lot faster than extracting included files.
	if (fast_cache) {
		TimeStop ts;
		FileIn fast_file(ConfigFile("mnist_databank.bin"));
		if (fast_file.IsOpen()) {
			PostCallback(THISBACK3(Progress, 0, 1, "Loading existing cache: mnist_databank.bin"));
			fast_file % bank;
			if (bank.images.GetCount() == 60000 && bank.labels.GetCount() == 60000 &&
				bank.test_images.GetCount() == 10000 && bank.test_labels.GetCount() == 10000) {
				LOG("Loaded data from mnist_databank.bin successfully in " << ts.ToString());
				PostCallback(THISBACK(Close0));
				return;
			}
		}
	}
	
	bank.images.Clear();
	bank.labels.Clear();
	bank.test_images.Clear();
	bank.test_labels.Clear();
	
	#ifdef LITTLEENDIAN
		#define SWAP32(x) x = SwapEndian32(x)
	#else
		#define SWAP32(x)
	#endif
	
	int mnist_data_count = 4;
	int actual = 0;
	int total = mnist_data_count * 2 + 1;
	for(int i = 0; i < mnist_data_count && !IsFail(); i++) {
		String file;
		switch (i) {
			case 0: file = "t10k-images.idx3-ubyte.bin"; break;
			case 1: file = "t10k-labels.idx1-ubyte.bin"; break;
			case 2: file = "train-images.idx3-ubyte.bin"; break;
			case 3: file = "train-labels.idx1-ubyte.bin"; break;
		}
		if (!FileExists(GetExeDirFile(file))) {
			PromptOK("Error: CIFAR-10 dataset file " + file + " is not included with this executable.");
			ret_value = 1;
			PostCallback(THISBACK(Close0));
			return;
		}
		
		FileIn in(GetExeDirFile(file));
		
		int length = in.GetSize();
		LOG("Load " <<  file << " size: " << length);
		actual++;
		if (IsFail()) break;
		
		PostCallback(THISBACK3(Progress, actual, total, "Reading: " + file));
		in.Seek(0);
		
		#undef ASSERT
		#define ASSERT(x) if (!(x)) {PromptOK("Reading failed in file '" + file + "'. Reason: '" #x "' is false"); ret_value = 1; PostCallback(THISBACK(Close0)); return;}
		
		if (file == "train-labels.idx1-ubyte.bin" || file == "t10k-labels.idx1-ubyte.bin") {
			int magic, items;
			bool main = file == "train-labels.idx1-ubyte.bin";
			
			in.Get(&magic, 4);
			SWAP32(magic);
			ASSERT(magic == 0x00000801);
			
			in.Get(&items, 4);
			SWAP32(items);
			ASSERT(items == (main ? 60000 : 10000));
			
			Vector<int>& labels = main ? bank.labels : bank.test_labels;
			labels.SetCount(items);
			for (int j = 0; j < items && !in.IsEof() && !IsFail(); j++) {
				byte label;
				in.Get(&label, 1);
				labels[j] = label;
			}
			LOG("Read OK: " << file);
		}
		else if (file == "train-images.idx3-ubyte.bin" || file == "t10k-images.idx3-ubyte.bin") {
			int magic, items, rows, cols;
			bool main = file == "train-images.idx3-ubyte.bin";
			
			in.Get(&magic, 4);
			SWAP32(magic);
			ASSERT(magic == 0x00000803);
			
			in.Get(&items, 4);
			SWAP32(items);
			ASSERT(items == (main ? 60000 : 10000));
			
			in.Get(&rows, 4);
			SWAP32(rows);
			ASSERT(rows == 28);
			
			in.Get(&cols, 4);
			SWAP32(cols);
			ASSERT(cols == 28);
			
			byte pixel;
			Vector<Image>& images = main ? bank.images : bank.test_images;
			images.SetCount(items);
			for(int j = 0; j < items && !in.IsEof() && !IsFail(); j++) {
				ImageBuffer ib(cols, rows);
				RGBA* it = ib.Begin();
				for (int y = 0; y < rows && !in.IsEof(); y++) {
					for (int x = 0; x < cols && !in.IsEof(); x++) {
						in.Get(&pixel, 1);
						it->a = 255;
						it->r = pixel;
						it->g = pixel;
						it->b = pixel;
						it++;
					}
				}
				if ((j % 100) == 0)
					SubProgress(j, items);
				images[j] = ib;
			}
			
			
			LOG("Read OK: " << file);
		}
		else Panic("What file is " + file);
		
		actual++;
	}
	
	if (fast_cache && !IsFail()) {
		PostCallback(THISBACK3(Progress, actual, total, "Saving fast cache: mnist_databank.bin"));
		FileOut fast_file(ConfigFile("mnist_databank.bin"));
		fast_file % bank;
	}
	
	PostCallback(THISBACK(Close0));
}

}
