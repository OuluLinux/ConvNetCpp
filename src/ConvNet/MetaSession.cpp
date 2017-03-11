#include "ConvNet.h"

namespace ConvNet {

MetaSession::MetaSession() {
	iter = 0; // iteration counter, goes from 0 -> num_epochs * num_training_data
	total_iter = 0;
	data.Add();
}

MetaSession::~MetaSession() {
	
}

void MetaSession::ClearSessions() {
	session.Clear();
	
	
}

void MetaSession::Step() {
	
	iter++;
	total_iter++;
	
	SessionData& sd = data[0];
	
	{
		int id = total_iter % sd.GetDataCount();
		tmp_in.Init(sd.GetDataWidth(), sd.GetDataHeight(), sd.GetDataDepth(), 0);
		VolumeDataBase& x = sd.Get(id);
		int label = sd.GetLabel(id);
		tmp_in.SetData(x);
		
		for (int i = 0; i < session.GetCount(); i++) {
			Session& ses = session[i];
			TrainerBase& trainer = *ses.GetTrainer();
			Net& net = ses.GetNetwork();
			
			// train on training example
			trainer.Train(tmp_in, label, 1.0);
			int yhat = net.GetPrediction();
			ses.accuracy_window.Add(yhat == label ? 1.0 : 0.0);
			ses.loss_window.Add(trainer.GetLoss());
		}
	}
	
	{
		int id = total_iter % sd.GetTestCount();
		tmp_in.Init(sd.GetDataWidth(), sd.GetDataHeight(), sd.GetDataDepth(), 0);
		VolumeDataBase& x = sd.GetTest(id);
		int label = sd.GetTestLabel(id);
		tmp_in.SetData(x);
		
		for (int i = 0; i < session.GetCount(); i++) {
			Session& ses = session[i];
			TrainerBase& trainer = *ses.GetTrainer();
			Net& net = ses.GetNetwork();
			
			// evaluate a test example
			net.Forward(tmp_in);
			int yhat_test = net.GetPrediction();
			ses.test_window.Add(yhat_test == label ? 1.0 : 0.0);
		}
	}
	
	if ((total_iter % 100) == 0) {
		WhenStepInterval(total_iter);
	}
}

void MetaSession::Store(ValueMap& map) const {
	Panic("TODO");
}

void MetaSession::Load(const ValueMap& map) {
	Panic("TODO");
}

}
