#include "EfficientNet.h"

#define IMAGECLASS EfficientNetImg
#define IMAGEFILE <EfficientNet/EfficientNet.iml>
#include <Draw/iml_source.h>

// EfficientNet implementation using compound scaling
EfficientNetApp::EfficientNetApp()
{
	Icon(EfficientNetImg::icon());
	Sizeable().MaximizeBox().MinimizeBox().Zoomable();
	Title("EfficientNet - Scalable Convolutional Networks");

	// Setup EfficientNet variant selector
	model_variant.Add(0, "EfficientNet-B0");
	model_variant.Add(1, "EfficientNet-B1");
	model_variant.Add(2, "EfficientNet-B2");
	model_variant.Add(3, "EfficientNet-B3");
	model_variant.Add(4, "EfficientNet-B4");
	model_variant.Add(5, "EfficientNet-B5");
	model_variant.Add(6, "EfficientNet-B6");
	model_variant.Add(7, "EfficientNet-B7");
	model_variant.Add(8, "EfficientNet-B8");
	model_variant.SetIndex(0);
	model_variant <<= THISBACK(OnModelVariantChanged);

	// Define EfficientNet-B0 architecture using compound scaling
	t = BuildEfficientNetB0Config();

	net_edit.SetData(t);

	average_size = 10;
	max_diff_imgs = 100000; // not limiting currently

	UpdateNetParamDisplay();

	Add(v_split.SizePos());
	v_split.Vert();

	v_split << layer_view;
	v_split.SetPos(6400);

	ses.SetTestPredict(true);

	layer_view.SetSession(ses);
	layer_view.SetColor();

	graph.SetSession(ses);
	graph.SetModeLoss();

	// Add EfficientNet-specific controls to the network control panel
	Add(efficientnet_ctrl.SizeHorz());
	CtrlLayout(efficientnet_ctrl, "EfficientNet Options");
	efficientnet_ctrl.Add(model_variant.HSizePos().TopPos(0, 24));

	PostCallback(THISBACK(Refresher));
}

EfficientNetApp::~EfficientNetApp() {
	ses.StopTraining();
}

void EfficientNetApp::DockInit() {
	DockLeft(Dockable(settings, "Settings").SizeHint(Size(320, 11*20)));
	DockLeft(Dockable(graph, "Loss").SizeHint(Size(320, 240)));
	DockLeft(Dockable(status, "Status").SizeHint(Size(120, 120)));
	AutoHide(DOCK_LEFT, Dockable(net_ctrl, "Edit Network").SizeHint(Size(640, 320)));
}

void EfficientNetApp::UpdateNetParamDisplay() {
	TrainerBase& trainer = ses.GetTrainer();
	rate.SetData(trainer.GetLearningRate());
	mom.SetData(trainer.GetMomentum());
	batch.SetData(trainer.GetBatchSize());
	decay.SetData(trainer.GetL2Decay());
}

void EfficientNetApp::ApplySettings() {
	TrainerBase& trainer = ses.GetTrainer();
	trainer.SetLearningRate(rate.GetData());
	trainer.SetMomentum(mom.GetData());
	trainer.SetBatchSize(batch.GetData());
	trainer.SetL2Decay(decay.GetData());
}

void EfficientNetApp::Pause() {
	if (ses.IsTraining())
		ses.StopTraining();
	else
		ses.StartTraining();
}

void EfficientNetApp::OpenFile() {
	String file = SelectFileOpen("BIN files\t*.bin\nAll files\t*.*");
	if (file.IsEmpty()) return;

	if (!FileExists(file)) {
		PromptOK("File does not exists");
		return;
	}

	ses.StopTraining();

	ticking_lock.Enter();
	FileIn fin(file);
	fin % ses;
	ticking_lock.Leave();

	ResetAll();
	ses.StartTraining();
}

void EfficientNetApp::SaveFile() {
	String file = SelectFileSaveAs("BIN files\t*.bin\nAll files\t*.*");
	if (file.IsEmpty()) return;

	FileOut fout(file);
	if (!fout.IsOpen()) {
		PromptOK("Error: could not open file " + file);
		return;
	}

	ses.StopTraining();

	fout % ses;

	ses.StartTraining();
}

void EfficientNetApp::Reload() {
	ses.StopTraining();

	String net_str = net_edit.GetData();

	ticking_lock.Enter();

	bool success = ses.MakeLayers(net_str);

	ticking_lock.Leave();

	ResetAll();
	layer_view.Layout();

	if (success) {
		ses.StartTraining();
	}
}

void EfficientNetApp::RefreshStatus() {
	String s;
	s << "   Forward time per example: " << ses.GetForwardTime() << "\n";
	s << "   Backprop time per example: " << ses.GetBackwardTime() << "\n";
	s << "   Classification loss: " << ses.GetLossAverage() << "\n";
	s << "   L2 Weight decay loss: " << ses.GetL2DecayLossAverage() << "\n";
	s << "   Training accuracy: " << ses.GetTrainingAccuracyAverage() << "\n";
	s << "   Validation accuracy: " << ses.GetValidationAccuracyAverage() << "\n";
	s << "   Examples seen: " << ses.GetStepCount();
	status.SetLabel(s);
}

void EfficientNetApp::Refresher() {
	layer_view.Refresh();
	graph.RefreshData();
	RefreshStatus();

	PostCallback(THISBACK(Refresher));
}

void EfficientNetApp::ResetAll() {
	UpdateNetParamDisplay();
	graph.Clear();
}

void EfficientNetApp::OnModelVariantChanged() {
	// Get the selected model variant and update the network config
	int variant = model_variant.GetIndex();
	t = BuildScaledConfig(variant);

	// Update network with new configuration
	net_edit.SetData(t);
	Reload();
}

// EfficientNet-specific functionality
String EfficientNetApp::BuildEfficientNetB0Config() {
	// EfficientNet uses compound scaling with baseline parameters:
	// width: 1.0, depth: 1.0, resolution: 1.0 for B0
	// This is a simplified representation - a full implementation would require
	// actual MBConv blocks with specific expansion ratios, kernel sizes, etc.
	return
		"[\n"
		// Input layer expecting 224x224x3 images (scaled appropriately for EfficientNet)
		"\t{\"type\":\"input\", \"input_width\":224, \"input_height\":224, \"input_depth\":3},\n"

		// Initial convolution layer (3x3, stride 2)
		"\t{\"type\":\"conv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":32, \"stride\":2, \"padding\":1, \"activation\":\"swish\"},\n"

		// MBConv blocks (Mobile Inverted Residual Bottleneck blocks)
		// MBConv1 with expansion 1
		"\t{\"type\":\"mbconv\", \"input_channels\":32, \"output_channels\":16, \"expansion_factor\":1, \"stride\":1, \"kernel_size\":3, \"se_ratio\":0.25, \"activation\":\"swish\", \"repetitions\":1},\n"

		// MBConv6 blocks
		"\t{\"type\":\"mbconv\", \"input_channels\":16, \"output_channels\":24, \"expansion_factor\":6, \"stride\":2, \"kernel_size\":3, \"se_ratio\":0.25, \"activation\":\"swish\", \"repetitions\":2},\n"
		"\t{\"type\":\"mbconv\", \"input_channels\":24, \"output_channels\":40, \"expansion_factor\":6, \"stride\":2, \"kernel_size\":5, \"se_ratio\":0.25, \"activation\":\"swish\", \"repetitions\":2},\n"

		// MBConv6 blocks with higher channels
		"\t{\"type\":\"mbconv\", \"input_channels\":40, \"output_channels\":80, \"expansion_factor\":6, \"stride\":2, \"kernel_size\":3, \"se_ratio\":0.25, \"activation\":\"swish\", \"repetitions\":3},\n"
		"\t{\"type\":\"mbconv\", \"input_channels\":80, \"output_channels\":112, \"expansion_factor\":6, \"stride\":1, \"kernel_size\":5, \"se_ratio\":0.25, \"activation\":\"swish\", \"repetitions\":3},\n"

		// MBConv6 blocks with even higher channels
		"\t{\"type\":\"mbconv\", \"input_channels\":112, \"output_channels\":192, \"expansion_factor\":6, \"stride\":2, \"kernel_size\":5, \"se_ratio\":0.25, \"activation\":\"swish\", \"repetitions\":4},\n"
		"\t{\"type\":\"mbconv\", \"input_channels\":192, \"output_channels\":320, \"expansion_factor\":6, \"stride\":1, \"kernel_size\":3, \"se_ratio\":0.25, \"activation\":\"swish\", \"repetitions\":1},\n"

		// Final convolution layer (1x1)
		"\t{\"type\":\"conv\", \"filter_width\":1, \"filter_height\":1, \"filter_count\":1280, \"activation\":\"swish\"},\n"

		// Global Average Pooling (conceptually)
		"\t{\"type\":\"global_avg_pool\"},\n"

		// Dropout
		"\t{\"type\":\"dropout\", \"dropout_rate\":0.2},\n"

		// Fully connected layer for classification (1000 classes for ImageNet)
		"\t{\"type\":\"fc\", \"neuron_count\":1000, \"activation\":\"softmax\"},\n"

		// Optimizer settings
		"\t{\"type\":\"rmsprop\", \"learning_rate\":0.002, \"rho\":0.9, \"batch_size\":64, \"l2_decay\":1e-5}\n"
		"]\n";
}

String EfficientNetApp::BuildScaledConfig(int model_variant) {
	// Compound scaling: new_width = width_coefficient * width^phi
	//                  new_depth = depth_coefficient * depth^phi
	//                  new_resolution = resolution_coefficient * resolution^phi
	// where phi is the scaling coefficient, and other coefficients are determined through grid search

	// Define scaling coefficients for each model variant
	double phi[] = {0.0, 0.5, 1.0, 1.4, 1.8, 2.2, 2.6, 3.0, 3.1}; // B0 to B8
	double width_coeff[] = {1.0, 1.0, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2};
	double depth_coeff[] = {1.0, 1.1, 1.2, 1.4, 1.8, 2.2, 2.6, 3.1, 3.6};
	double res_coeff[] = {1.0, 1.1, 1.2, 1.4, 1.8, 2.2, 2.6, 3.1, 3.2};

	// Clamp model_variant to valid range
	if (model_variant < 0) model_variant = 0;
	if (model_variant > 8) model_variant = 8;

	double w = width_coeff[model_variant];
	double d = depth_coeff[model_variant];
	double r = res_coeff[model_variant];

	// Calculate scaled parameters based on B0 baseline
	// B0 baseline: [32, 16, 24, 40, 80, 112, 192, 320] output channels
	// B0 baseline: [1, 2, 2, 3, 3, 4, 1] depths per block
	// B0 baseline: 224x224 input resolution

	// Calculate scaled channel sizes and depths
	int c1 = (int)(32 * w);   // Initial conv channels
	int c2 = (int)(16 * w);   // MBConv1 out channels
	int d2 = (int)(1 * d);    // MBConv1 depth
	int c3 = (int)(24 * w);   // MBConv6 out channels
	int d3 = (int)(2 * d);    // MBConv6 depth
	int c4 = (int)(40 * w);   // MBConv6 out channels
	int d4 = (int)(2 * d);    // MBConv6 depth
	int c5 = (int)(80 * w);   // MBConv6 out channels
	int d5 = (int)(3 * d);    // MBConv6 depth
	int c6 = (int)(112 * w);  // MBConv6 out channels
	int d6 = (int)(3 * d);    // MBConv6 depth
	int c7 = (int)(192 * w);  // MBConv6 out channels
	int d7 = (int)(4 * d);    // MBConv6 depth
	int c8 = (int)(320 * w);  // MBConv6 out channels
	int d8 = (int)(1 * d);    // MBConv6 depth
	int final_conv = (int)(1280 * w); // Final conv channels

	// Calculate scaled input resolution
	int input_res = (int)(224 * r);

	// Build configuration based on scaled parameters
	String config = "[\n";
	config += Format("\t{\"type\":\"input\", \"input_width\":%d, \"input_height\":%d, \"input_depth\":3},\n", input_res, input_res);
	config += Format("\t{\"type\":\"conv\", \"filter_width\":3, \"filter_height\":3, \"filter_count\":%d, \"stride\":2, \"padding\":1, \"activation\":\"swish\"},\n", c1);

	// Add MBConv blocks with scaled depths and channels
	for (int i = 0; i < d2; i++) {
		config += Format("\t{\"type\":\"mbconv\", \"input_channels\":%d, \"output_channels\":%d, \"expansion_factor\":1, \"stride\":%d, \"kernel_size\":3, \"se_ratio\":0.25, \"activation\":\"swish\", \"repetitions\":1},\n",
		                i == 0 ? c1 : c2, c2, i == 0 ? 1 : 1);
	}

	for (int i = 0; i < d3; i++) {
		config += Format("\t{\"type\":\"mbconv\", \"input_channels\":%d, \"output_channels\":%d, \"expansion_factor\":6, \"stride\":%d, \"kernel_size\":3, \"se_ratio\":0.25, \"activation\":\"swish\", \"repetitions\":1},\n",
		                i == 0 ? c2 : c3, c3, i == 0 ? 2 : 1);
	}

	for (int i = 0; i < d4; i++) {
		config += Format("\t{\"type\":\"mbconv\", \"input_channels\":%d, \"output_channels\":%d, \"expansion_factor\":6, \"stride\":%d, \"kernel_size\":5, \"se_ratio\":0.25, \"activation\":\"swish\", \"repetitions\":1},\n",
		                i == 0 ? c3 : c4, c4, i == 0 ? 2 : 1);
	}

	for (int i = 0; i < d5; i++) {
		config += Format("\t{\"type\":\"mbconv\", \"input_channels\":%d, \"output_channels\":%d, \"expansion_factor\":6, \"stride\":%d, \"kernel_size\":3, \"se_ratio\":0.25, \"activation\":\"swish\", \"repetitions\":1},\n",
		                i == 0 ? c4 : c5, c5, i == 0 ? 2 : 1);
	}

	for (int i = 0; i < d6; i++) {
		config += Format("\t{\"type\":\"mbconv\", \"input_channels\":%d, \"output_channels\":%d, \"expansion_factor\":6, \"stride\":%d, \"kernel_size\":5, \"se_ratio\":0.25, \"activation\":\"swish\", \"repetitions\":1},\n",
		                i == 0 ? c5 : c6, c6, i == 0 ? 1 : 1);
	}

	for (int i = 0; i < d7; i++) {
		config += Format("\t{\"type\":\"mbconv\", \"input_channels\":%d, \"output_channels\":%d, \"expansion_factor\":6, \"stride\":%d, \"kernel_size\":5, \"se_ratio\":0.25, \"activation\":\"swish\", \"repetitions\":1},\n",
		                i == 0 ? c6 : c7, c7, i == 0 ? 2 : 1);
	}

	for (int i = 0; i < d8; i++) {
		config += Format("\t{\"type\":\"mbconv\", \"input_channels\":%d, \"output_channels\":%d, \"expansion_factor\":6, \"stride\":%d, \"kernel_size\":3, \"se_ratio\":0.25, \"activation\":\"swish\", \"repetitions\":1},\n",
		                i == 0 ? c7 : c8, c8, i == 0 ? 1 : 1);
	}

	config += Format("\t{\"type\":\"conv\", \"filter_width\":1, \"filter_height\":1, \"filter_count\":%d, \"activation\":\"swish\"},\n", final_conv);
	config += "\t{\"type\":\"global_avg_pool\"},\n";
	config += "\t{\"type\":\"dropout\", \"dropout_rate\":0.2},\n";
	config += "\t{\"type\":\"fc\", \"neuron_count\":1000, \"activation\":\"softmax\"},\n";
	config += "\t{\"type\":\"rmsprop\", \"learning_rate\":0.002, \"rho\":0.9, \"batch_size\":64, \"l2_decay\":1e-5}\n";
	config += "]\n";

	return config;
}