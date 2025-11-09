#include "TransformerTester.h"

TransformerApp::TransformerApp() {
    Title("Transformer Visualization Tool");
    Sizeable().MaximizeBox().MinimizeBox();
    
    // Set up the layout
    CtrlLayout(*this);
    
    // Initialize components
    attentionVis.SetLabel("Attention Visualization");
    modelInfo.SetLabel("Model Information");
    seqInput.SetLabel("Sequence Input");
    seqOutput.SetLabel("Sequence Output");
    
    // Add controls to the layout
    Add(CtrlLayout(controls));
    
    // Set up event handlers
    trainBtn.SetLabel("Train Model");
    testBtn.SetLabel("Test Model");
    visualizeBtn.SetLabel("Visualize Attention");
    
    // Initialize the transformer model
    InitModel();
    
    // Set up the UI event handlers
    trainBtn <<= THISBACK(OnTrain);
    testBtn <<= THISBACK(OnTest);
    visualizeBtn <<= THISBACK(OnVisualize);
}

void TransformerApp::InitModel() {
    // Initialize a small transformer model for testing
    // For demonstration purposes, we'll use smaller dimensions
    transformer = CreateTransformer(
        1000,  // src_vocab_size 
        1000,  // tgt_vocab_size
        64,    // embed_dim
        4,     // num_heads
        2,     // num_encoder_layers
        2,     // num_decoder_layers
        256,   // ff_dim
        50,    // max_seq_len
        0.1    // dropout_rate
    );
    
    LOG("Transformer model initialized");
}

void TransformerApp::OnTrain() {
    // Placeholder for training functionality
    PromptOK("Training process would start here");
    
    // In a real implementation, this would:
    // 1. Prepare training data
    // 2. Set up optimizer
    // 3. Run training loop
    // 4. Update UI with training progress
}

void TransformerApp::OnTest() {
    // Placeholder for testing functionality
    String input_seq = inputEdit.GetText();
    if (input_seq.IsEmpty()) {
        PromptOK("Please enter an input sequence");
        return;
    }
    
    // For now, just echo the input
    outputEdit.SetText("Processed: " + input_seq);
    LOG("Test input: " + input_seq);
}

void TransformerApp::OnVisualize() {
    // Placeholder for attention visualization
    PromptOK("Attention visualization would appear here");
    
    // In a real implementation, this would:
    // 1. Process input through the transformer
    // 2. Extract attention weights from each layer/head
    // 3. Generate visualization of the attention patterns
}

// Custom control for attention visualization
TransformerAttentionControl::TransformerAttentionControl() {
    SetLabel("Attention Visualization");
    attentionData.Clear();
}

void TransformerAttentionControl::Paint(Draw& draw) {
    Size sz = GetSize();
    draw.DrawRect(sz, SColorFace());
    
    // Draw a simple grid to represent attention weights
    if (attentionData.GetCount() > 0) {
        int rows = attentionData.GetCount();
        int cols = attentionData[0].GetCount();
        
        if (rows > 0 && cols > 0) {
            int cellWidth = sz.cx / cols;
            int cellHeight = sz.cy / rows;
            
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    double weight = 0.0;
                    if (i < attentionData.GetCount() && j < attentionData[i].GetCount()) {
                        weight = attentionData[i][j];
                    }
                    
                    // Normalize weight to 0-1 for color
                    double normWeight = max(0.0, min(1.0, weight));
                    
                    // Create color based on attention weight
                    Color c = Color((int)(255 * normWeight), 0, (int)(255 * (1 - normWeight)));
                    
                    // Draw the cell
                    Rect cellRect(j * cellWidth, i * cellHeight, 
                                (j + 1) * cellWidth, (i + 1) * cellHeight);
                    draw.DrawRect(cellRect, c);
                    draw.DrawRect(cellRect, 1, Black);
                }
            }
        }
    } else {
        // Draw placeholder text
        draw.DrawText(10, 10, "No attention data to visualize", StdFont(), Black);
    }
}

void TransformerAttentionControl::SetAttentionData(const Vector<Vector<double>>& data) {
    attentionData = data;
    Refresh();
}

// Custom control for model performance metrics
TransformerMetricsControl::TransformerMetricsControl() {
    SetLabel("Performance Metrics");
    ClearMetrics();
}

void TransformerMetricsControl::Paint(Draw& draw) {
    Size sz = GetSize();
    draw.DrawRect(sz, SColorFace());
    
    // Draw metrics
    int y = 10;
    draw.DrawText(10, y, Format("Loss: %.4f", loss), StdFont(), Black);
    y += 20;
    draw.DrawText(10, y, Format("Accuracy: %.2f%%", accuracy * 100), StdFont(), Black);
    y += 20;
    draw.DrawText(10, y, Format("Training Steps: %d", trainingSteps), StdFont(), Black);
    y += 20;
    draw.DrawText(10, y, Format("Tokens/sec: %.2f", tokensPerSecond), StdFont(), Black);
}

void TransformerMetricsControl::UpdateMetrics(double newLoss, double newAccuracy, 
                                             int newTrainingSteps, double newTokensPerSec) {
    loss = newLoss;
    accuracy = newAccuracy;
    trainingSteps = newTrainingSteps;
    tokensPerSecond = newTokensPerSec;
    Refresh();
}

void TransformerMetricsControl::ClearMetrics() {
    loss = 0.0;
    accuracy = 0.0;
    trainingSteps = 0;
    tokensPerSecond = 0.0;
}