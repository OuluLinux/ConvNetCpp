#include "GptTester.h"

GptAttentionControl::GptAttentionControl() {
    title = "Attention Visualization";
    attentionData.Clear();
}

void GptAttentionControl::Paint(Draw& draw) {
    Size sz = GetSize();
    draw.DrawRect(sz, SColorFace());
    
    // Draw title
    if (!title.IsEmpty()) {
        draw.DrawText(5, 5, title, StdFont(), Black);
    }
    
    // Draw a simple grid to represent attention weights
    if (attentionData.GetCount() > 0) {
        int rows = attentionData.GetCount();
        int cols = attentionData[0].GetCount();
        
        if (rows > 0 && cols > 0) {
            int offsetX = 30;  // Leave space for title and labels
            int offsetY = 30;
            int availableWidth = sz.cx - 2 * offsetX;
            int availableHeight = sz.cy - 2 * offsetY;
            
            int cellWidth = availableWidth / cols;
            int cellHeight = availableHeight / rows;
            
            // Limit cell size to prevent extremely small cells
            cellWidth = max(2, min(cellWidth, 20));
            cellHeight = max(2, min(cellHeight, 20));
            
            for (int i = 0; i < rows && i < (availableHeight / cellHeight); i++) {
                for (int j = 0; j < cols && j < (availableWidth / cellWidth); j++) {
                    double weight = 0.0;
                    if (i < attentionData.GetCount() && j < attentionData[i].GetCount()) {
                        weight = attentionData[i][j];
                    }
                    
                    // Normalize weight to 0-1 for color
                    double normWeight = max(0.0, min(1.0, weight + 1.0)); // Shift to positive range
                    
                    // Create color based on attention weight (blue to red scale)
                    int red = (int)(255 * normWeight);
                    int blue = (int)(255 * (1.0 - normWeight));
                    Color c = Color(red, 0, blue);
                    
                    // Draw the cell
                    Rect cellRect(offsetX + j * cellWidth, offsetY + i * cellHeight, 
                                offsetX + (j + 1) * cellWidth, offsetY + (i + 1) * cellHeight);
                    draw.DrawRect(cellRect, c);
                    draw.DrawRect(cellRect, 1, Gray);
                }
            }
        }
    } else {
        // Draw placeholder text
        draw.DrawText(10, 30, "No attention data to visualize", StdFont(), Black);
    }
}

void GptAttentionControl::SetAttentionData(const Vector<Vector<double>>& data) {
    attentionData = data;
    Refresh();
}

GptTextDisplay::GptTextDisplay() {
    generatedText = "GPT Generation will appear here...";
    cursorPosition = generatedText.GetLength();
}

void GptTextDisplay::Paint(Draw& draw) {
    Size sz = GetSize();
    draw.DrawRect(sz, White);
    
    // Draw the generated text
    // We'll use simple text wrapping
    int x = 10;
    int y = 10;
    int lineHeight = 16;
    int maxWidth = sz.cx - 20;
    
    // Split text into lines to fit the control
    Vector<String> lines;
    String currentLine;
    
    for (int i = 0; i < generatedText.GetCount(); i++) {
        char c = generatedText[i];
        String testLine = currentLine + String(c);
        
        if (GetTextWidth(testLine, StdFont()) > maxWidth || c == '\n') {
            lines.Add(currentLine);
            currentLine = (c == '\n') ? String() : String(c);
        } else {
            currentLine = testLine;
        }
    }
    if (!currentLine.IsEmpty()) {
        lines.Add(currentLine);
    }
    
    // Draw each line
    for (int i = 0; i < lines.GetCount(); i++) {
        if (y + lineHeight > sz.cy - 10) break; // Stop if we run out of space
        
        draw.DrawText(x, y, lines[i], StdFont(), Black);
        y += lineHeight;
    }
    
    // Draw cursor if needed (simplified)
    if (HasFocus()) {
        // Draw cursor at the end of text
        int cursorX = x + GetTextWidth(lines.GetCount() > 0 ? lines.Top() : String(), StdFont());
        int cursorY = y - lineHeight + 2;
        draw.DrawRect(cursorX, cursorY, 2, lineHeight - 4, Black);
    }
}

void GptTextDisplay::SetGeneratedText(const String& text) {
    generatedText = text;
    Refresh();
}

void GptTextDisplay::AddText(const String& text) {
    generatedText += text;
    Refresh();
}

GptApp::GptApp() {
    Title("GPT Model Tester");
    Sizeable().MaximizeBox().MinimizeBox();
    
    // Initialize with default values
    temperatureCtrl.SetData(1.0);
    topKCtrl.SetData(50);
    topKEnable.SetData(false);
    nucleusPCtrl.SetData(0.9);
    nucleusEnable.SetData(false);
    maxTokensCtrl.SetData(50);
    
    // Set up the UI layout
    Size(800, 600);
    
    // Create controls
    promptEdit.SetLabel("Prompt:");
    generateBtn.SetLabel("Generate");
    clearBtn.SetLabel("Clear");
    temperatureCtrl.SetLabel("Temperature:");
    topKCtrl.SetLabel("Top-K:");
    nucleusPCtrl.SetLabel("Nucleus (p):");
    maxTokensCtrl.SetLabel("Max Tokens:");
    
    // Add controls to the window
    WithStdLayout(*this);
    Add(promptEdit.HSizePos(10, 200).TopPos(10, 30));
    Add(generateBtn.RightPos(100, 80).TopPos(10, 30));
    Add(clearBtn.RightPos(10, 80).TopPos(10, 30));
    
    // Add hyperparameter controls
    Add(temperatureCtrl.HSizePos(10, 200).TopPos(50, 25));
    Add(topKEnable.RightPos(300, 100).TopPos(50, 25));
    Add(topKCtrl.RightPos(200, 80).TopPos(50, 25));
    
    Add(nucleusEnable.RightPos(300, 100).TopPos(85, 25));
    Add(nucleusPCtrl.RightPos(200, 80).TopPos(85, 25));
    Add(maxTokensCtrl.RightPos(100, 80).TopPos(85, 25));
    
    // Add text display area
    Add(textDisplay.HSizePos(10).VSizePos(130, 10));
    
    // Set up event handlers
    generateBtn <<= THISBACK(OnGenerate);
    clearBtn <<= THISBACK(OnClear);
    
    // Connect hyperparameter changes
    temperatureCtrl.WhenAction = THISBACK(OnHyperparametersChanged);
    topKCtrl.WhenAction = THISBACK(OnHyperparametersChanged);
    topKEnable.WhenAction = THISBACK(OnHyperparametersChanged);
    nucleusPCtrl.WhenAction = THISBACK(OnHyperparametersChanged);
    nucleusEnable.WhenAction = THISBACK(OnHyperparametersChanged);
    maxTokensCtrl.WhenAction = THISBACK(OnHyperparametersChanged);
    
    // Initialize the GPT model
    InitModel();
}

void GptApp::InitModel() {
    // Initialize a small GPT model for testing
    gpt_session = CreateGPTSession(
        vocab_size,   // vocab_size
        embed_dim,    // embed_dim
        num_heads,    // num_heads
        num_layers,   // num_layers
        ff_dim,       // ff_dim
        max_seq_len,  // max_seq_len
        0.1,          // dropout_rate
        3e-4          // learning_rate
    );
    
    LOG("GPT model initialized");
    textDisplay.SetGeneratedText("GPT model initialized. Enter a prompt and click 'Generate'.");
}

void GptApp::OnGenerate() {
    String prompt = promptEdit.GetText();
    if (prompt.IsEmpty()) {
        prompt = "The future of artificial intelligence";
    }
    
    // Update display with the prompt first
    textDisplay.SetGeneratedText("Prompt: " + prompt + "\n\nGenerating...\n");
    
    // Get hyperparameters
    double temperature = max(0.1, temperatureCtrl.GetData());
    bool use_top_k = topKEnable.GetData();
    int k = (int)topKCtrl.GetData();
    k = max(1, min(k, 1000)); // Limit k to reasonable range
    
    bool use_nucleus = nucleusEnable.GetData();
    double p = nucleusPCtrl.GetData();
    p = max(0.01, min(p, 0.99)); // Limit p to reasonable range
    
    int max_tokens = (int)maxTokensCtrl.GetData();
    max_tokens = max(1, min(max_tokens, 200)); // Limit generation length
    
    // In a real implementation, we would:
    // 1. Tokenize the prompt
    // 2. Generate text using the GPT model
    // 3. Detokenize and display the result
    
    // For demonstration, we'll simulate the generation
    String generated = prompt + " is an exciting field that continues to evolve. "
                      "New developments emerge regularly, pushing the boundaries of what's possible. "
                      "Researchers are constantly exploring innovative approaches and methodologies. "
                      "The impact on society is significant and far-reaching. "
                      "As we move forward, the potential applications multiply exponentially.";
    
    // Truncate to max_tokens (simulated)
    Vector<String> words = Split(prompt + " " + generated, ' ');
    String result = Join(words, ' ', 0, min(words.GetCount(), max_tokens));
    
    textDisplay.SetGeneratedText("Prompt: " + prompt + "\n\n" + result + "\n\n[Generation complete]");
    
    LOG("Text generation completed with prompt: " + prompt);
}

void GptApp::OnClear() {
    promptEdit.Clear();
    textDisplay.SetGeneratedText("GPT Generation will appear here...");
}

void GptApp::OnHyperparametersChanged() {
    // Update the UI or model based on hyperparameter changes
    double temp = temperatureCtrl.GetData();
    LOG(Format("Hyperparameters updated - Temperature: %.2f", temp));
}