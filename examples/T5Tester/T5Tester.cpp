#include "T5Tester.h"

T5App::T5App() {
    Add(attentionVis.SizePos());
    Title("T5 (Text-to-Text Transfer Transformer) Demo");
    Sizeable().MaximizeBox().MinimizeBox();

    // Set up the layout
    // manual layout

    // Initialize components
    // // // attentionVis.SetLabel("Attention Visualization");
    // // modelInfo.SetLabel("Model Information");
    // // seqInput.SetLabel("Input Sequence");
    // // seqOutput.SetLabel("Output Sequence");
    // // taskLabel.SetLabel("Task:");

    // Add controls to the layout
    // manual layout

    // Set up event handlers
    // // trainBtn.SetLabel("Train Model");
    // // testBtn.SetLabel("Test Model");
    // // visualizeBtn.SetLabel("Visualize Attention");
    // // encodeBtn.SetLabel("Encode");
    // // decodeBtn.SetLabel("Decode");

    // Set up task selector for different T5 tasks
    taskSelect.Add(0, "Translation");
    taskSelect.Add(1, "Summarization");
    taskSelect.Add(2, "Q&A");
    taskSelect.Add(3, "Classification");
    taskSelect.SetIndex(0);  // Default to translation

    // Initialize the T5 model
    InitModel();

    // Set up the UI event handlers
    trainBtn <<= THISBACK(OnTrain);
    testBtn <<= THISBACK(OnTest);
    visualizeBtn <<= THISBACK(OnVisualize);
    encodeBtn <<= THISBACK(OnEncode);
    decodeBtn <<= THISBACK(OnDecode);
}

void T5App::InitModel() {
    // Initialize simplified T5 encoder and decoder models for demonstration
    // T5 uses an encoder-decoder architecture with shared embeddings

    // Encoder: processes input sequence
    encoder_model = CreateTransformer(
        32128,  // src_vocab_size (T5 uses 32128 for sentencepiece)
        32128,  // tgt_vocab_size (same as src for T5, for compatibility though encoder only produces hidden states)
        512,    // embed_dim (d_model)
        8,      // num_heads
        6,      // num_encoder_layers
        0,      // num_decoder_layers (encoder only has encoder layers)
        2048,   // ff_dim (d_ff in feed-forward)
        512,    // max_seq_len
        0.1     // dropout_rate
    );

    // Decoder: processes target sequence using encoder context
    decoder_model = CreateTransformer(
        32128,  // src_vocab_size (T5 uses 32128 for sentencepiece)
        32128,  // tgt_vocab_size (output vocabulary size)
        512,    // embed_dim (d_model)
        8,      // num_heads
        0,      // num_encoder_layers (decoder only has decoder layers)
        6,      // num_decoder_layers
        2048,   // ff_dim (d_ff in feed-forward)
        512,    // max_seq_len
        0.1     // dropout_rate
    );

    LOG("T5 model initialized with separate encoder and decoder components");
}

void T5App::OnTrain() {
    // Placeholder for training functionality
    PromptOK("T5 Training process would start here");

    // In a real implementation, this would:
    // 1. Load pretraining data (C4, etc.)
    // 2. Prepare inputs with task prefixes (e.g., "translate English to German: ...")
    // 3. Set up optimizer (typically Adam with learning rate scheduling)
    // 4. Run training loop with teacher forcing for decoder
    // 5. Update UI with training progress
}

void T5App::OnTest() {
    String input_seq = inputEdit.GetText().ToString().ToString();
    if (input_seq.IsEmpty()) {
        PromptOK("Please enter an input sequence");
        return;
    }

    // For T5, we need to prepend a task prefix based on selected task
    String task_prefix;
    int task_id = taskSelect.GetIndex();
    switch(task_id) {
        case 0: task_prefix = "translate English to German: "; break;
        case 1: task_prefix = "summarize: "; break;
        case 2: task_prefix = "question: "; break;
        case 3: task_prefix = "classification: "; break;
        default: task_prefix = "translate English to German: "; break;
    }

    String full_input = task_prefix + input_seq;

    // In a real implementation, this would run the encoder and then the decoder to generate text
    // For demo purposes, we'll simulate the process
    String encoded_repr = "[hidden states for '" + full_input + "']";
    String generated_output = "T5 Generated: " + full_input + " -> translated output";

    outputEdit.SetText(generated_output);
    LOG("T5 test input: " + full_input);
    LOG("T5 encoded to: " + encoded_repr);
    LOG("T5 decoded to: " + generated_output);

    // Update metrics to simulate processing
    modelMetrics.UpdateMetrics(0.82, 0.75, 1005, 49.2);
}

void T5App::OnEncode() {
    // In a real implementation, this would run the encoder part of the T5 model
    String input_seq = inputEdit.GetText().ToString().ToString();
    if (input_seq.IsEmpty()) {
        PromptOK("Please enter an input sequence");
        return;
    }

    // For T5, we need to prepend a task prefix based on selected task
    String task_prefix;
    int task_id = taskSelect.GetIndex();
    switch(task_id) {
        case 0: task_prefix = "translate English to German: "; break;
        case 1: task_prefix = "summarize: "; break;
        case 2: task_prefix = "question: "; break;
        case 3: task_prefix = "classification: "; break;
        default: task_prefix = "translate English to German: "; break;
    }

    String full_input = task_prefix + input_seq;

    // In a real implementation, this would run the encoder to produce hidden states
    // For demo, we'll just add a message about encoding
    outputEdit.SetText("Encoded representation: [hidden states for '" + full_input + "']");
    LOG("T5 encoded: " + full_input);

    // Update metrics to simulate processing
    modelMetrics.UpdateMetrics(0.85, 0.72, 1000, 50.0);
}

void T5App::OnDecode() {
    // In a real implementation, this would run the decoder part of the T5 model
    // using the encoder's hidden states
    String encoded_state = outputEdit.GetText().ToString().ToString();
    if (encoded_state.IsEmpty() || !encoded_state.StartsWith("Encoded representation:")) {
        PromptOK("Please encode a sequence first");
        return;
    }

    // For demo, we'll generate a simple output based on the encoded state
    String generated_output = "Generated: " + encoded_state + " -> decoded output";
    outputEdit.SetText(generated_output);

    LOG("T5 decoded to: " + generated_output);

    // Update metrics
    modelMetrics.UpdateMetrics(0.78, 0.69, 1010, 48.5);
}

void T5App::OnVisualize() {
    // Placeholder for attention visualization
    PromptOK("T5 Attention visualization would appear here");

    // In a real implementation, this would:
    // 1. Process input through the T5 model
    // 2. Extract attention weights from encoder, decoder, and encoder-decoder attention
    // 3. Generate visualization of the attention patterns across layers and heads
    // 4. Show different visualizations for self-attention in encoder, self-attention in decoder,
    //    and encoder-decoder attention

    // Simulate attention data for visualization
    Vector<Vector<double>> att_data;
    for (int i = 0; i < 10; i++) {  // 10x10 grid for demo
        Vector<double> row;
        for (int j = 0; j < 10; j++) {
            row.Add(0.0 + Randomf() * (1.0));
        }
        att_data.Add(clone(row));
    }

    attentionVis.SetAttentionData(att_data);
}

// Custom control for attention visualization
T5AttentionControl::T5AttentionControl() {
    // SetLabel("Attention Visualization");
    attentionData.Clear();
}

void T5AttentionControl::Paint(Draw& draw) {
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
                    Color c = Color((int)(255 * normWeight), (int)(128 * normWeight), (int)(255 * (1 - normWeight)));

                    // Draw the cell
                    Rect cellRect(j * cellWidth, i * cellHeight,
                                (j + 1) * cellWidth, (i + 1) * cellHeight);
                    draw.DrawRect(cellRect, c);
                    draw.DrawRect(cellRect, Black());
                }
            }
        }
    } else {
        // Draw placeholder text
        draw.DrawText(10, 10, "No attention data to visualize", StdFont(), Black());
    }
}

void T5AttentionControl::SetAttentionData(const Vector<Vector<double>>& data) {
    attentionData = clone(data);
    Refresh();
}

// Custom control for model performance metrics
T5MetricsControl::T5MetricsControl() {
    // SetLabel("Performance Metrics");
    ClearMetrics();
}

void T5MetricsControl::Paint(Draw& draw) {
    Size sz = GetSize();
    draw.DrawRect(sz, SColorFace());

    // Draw metrics
    int y = 10;
    draw.DrawText(10, y, Format("Loss: %.4f", loss), StdFont(), Black());
    y += 20;
    draw.DrawText(10, y, Format("Accuracy: %.2f%%", accuracy * 100), StdFont(), Black());
    y += 20;
    draw.DrawText(10, y, Format("Training Steps: %d", trainingSteps), StdFont(), Black());
    y += 20;
    draw.DrawText(10, y, Format("Tokens/sec: %.2f", tokensPerSecond), StdFont(), Black());
}

void T5MetricsControl::UpdateMetrics(double newLoss, double newAccuracy,
                                     int newTrainingSteps, double newTokensPerSec) {
    loss = newLoss;
    accuracy = newAccuracy;
    trainingSteps = newTrainingSteps;
    tokensPerSecond = newTokensPerSec;
    Refresh();
}

void T5MetricsControl::ClearMetrics() {
    loss = 0.0;
    accuracy = 0.0;
    trainingSteps = 0;
    tokensPerSecond = 0.0;
}