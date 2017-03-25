#include "LoaderMNIST.h"

namespace ConvNet {
using namespace Upp;

LoaderMNIST::LoaderMNIST(SessionData& sd) {
	this->sd = &sd;
	
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
	
	Thread::Start(THISBACK(Load));
}

void LoaderMNIST::Progress(int actual, int total, String label) {
	LOG("Progress " << actual << "/" << total);
	prog.Set(actual, total);
	sub.Set(0, 1);
	lbl.SetLabel(label);
}

void LoaderMNIST::SubProgress(int actual, int total) {
	sub.Set(actual, total);
}

void LoaderMNIST::Load() {
	
	#ifdef CPU_LITTLE_ENDIAN
		#define SWAP32(x) x = SwapEndian32(x)
	#else
		#define SWAP32(x)
	#endif
	
	SessionData& d = *sd;
	
	d.BeginDataClass<VolumeDataDivider<byte, 255> >(10, 60000, 28, 28, 1, 10000);
	
	d.SetClass(0, "0");
	d.SetClass(1, "1");
	d.SetClass(2, "2");
	d.SetClass(3, "3");
	d.SetClass(4, "4");
	d.SetClass(5, "5");
	d.SetClass(6, "6");
	d.SetClass(7, "7");
	d.SetClass(8, "8");
	d.SetClass(9, "9");
	
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
			PromptOK("Error: MNIST dataset file " + file + " is not included with this executable.");
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
			
			for (int j = 0; j < items && !in.IsEof() && !IsFail(); j++) {
				byte label;
				in.Get(&label, 1);
				if (main)
					d.SetLabel(j, label);
				else
					d.SetTestLabel(j, label);
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
			
			int len = rows * cols;
			
			for(int j = 0; j < items && !in.IsEof() && !IsFail(); j++) {
				VolumeDataBase& out = main ? d.Get(j) : d.GetTest(j);
				for (int p = 0; p < len && !in.IsEof(); p++) {
					byte pixel;
					in.Get(&pixel, 1);
					out.Set(p, pixel / 255.0);
				}
				if ((j % 100) == 0)
					PostCallback(THISBACK2(SubProgress, j, items));
			}
			
			
			LOG("Read OK: " << file);
		}
		else Panic("What file is " + file);
		
		actual++;
	}
	
	d.EndData();
	
	PostCallback(THISBACK(Close0));
}

}
