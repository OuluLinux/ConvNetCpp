#include "MidiLearner.h"
#include <plugin/bz2/bz2.h>

// MIDI Analysis Implementation
MidiProperties MidiAnalyzer::AnalyzeMidiFile(const String& filename) {
    FileIn in(filename);
    if (!in) {
        LOG("Could not open MIDI file: " << filename);
        return MidiProperties();
    }
    
    // Read the entire file into a byte vector
    Vector<byte> data;
    int c;
    while ((c = in.Get()) != -1) {
        data.Add(c);
    }
    
    return AnalyzeMidiData(data);
}

MidiProperties MidiAnalyzer::AnalyzeMidiData(const Vector<byte>& midi_data) {
    MidiProperties props;
    
    props.tempo = ExtractTempo(midi_data);
    props.key = ExtractKey(midi_data);
    props.signature = ExtractTimeSignature(midi_data);
    props.is_swing = DetectSwing(midi_data);
    props.ticks_per_beat = ExtractTicksPerBeat(midi_data);
    
    return props;
}

double MidiAnalyzer::ExtractTempo(const Vector<byte>& midi_data) {
    // Look for tempo meta events in MIDI data (0xFF 0x51)
    // This is a simplified implementation
    for (int i = 0; i < midi_data.GetCount() - 6; i++) {
        if (midi_data[i] == 0xFF && midi_data[i+1] == 0x51) {
            // Found tempo change
            // Tempo is stored as 3 bytes representing microseconds per quarter note
            int microsec_per_quarter = (midi_data[i+3] << 16) | (midi_data[i+4] << 8) | midi_data[i+5];
            double tempo_bpm = 60000000.0 / microsec_per_quarter;
            return tempo_bpm;
        }
    }
    
    // Default tempo if not found
    return 120.0;
}

String MidiAnalyzer::ExtractKey(const Vector<byte>& midi_data) {
    // Look for key signature meta events in MIDI data (0xFF 0x59)
    for (int i = 0; i < midi_data.GetCount() - 4; i++) {
        if (midi_data[i] == 0xFF && midi_data[i+1] == 0x59) {
            // Found key signature
            int key_index = midi_data[i+3];
            int mode = midi_data[i+4]; // 0 = major, 1 = minor
            
            // Map MIDI key index to musical key
            String keys[] = {"Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#"};
            String key_name = keys[key_index + 7]; // Adjust for signed values
            
            if (mode == 0) {
                return key_name + " major";
            } else {
                return key_name + " minor";
            }
        }
    }
    
    return "C major"; // Default
}

String MidiAnalyzer::ExtractTimeSignature(const Vector<byte>& midi_data) {
    // Look for time signature meta events in MIDI data (0xFF 0x58)
    for (int i = 0; i < midi_data.GetCount() - 6; i++) {
        if (midi_data[i] == 0xFF && midi_data[i+1] == 0x58) {
            // Found time signature
            int numerator = midi_data[i+3];
            int denominator = 1 << midi_data[i+4]; // Denominator is stored as power of 2
            return String() << numerator << "/" << denominator;
        }
    }
    
    return "4/4"; // Default
}

bool MidiAnalyzer::DetectSwing(const Vector<byte>& midi_data) {
    // Detect swing by looking for patterns in note timing that deviate from straight quantization
    // This is a simplified implementation
    // More advanced algorithms would analyze the timing patterns of consecutive notes
    
    // For now, return false as a placeholder
    // In a real implementation, we would analyze the delta times between MIDI events
    return false;
}

int MidiAnalyzer::ExtractTicksPerBeat(const Vector<byte>& midi_data) {
    // MIDI header chunk contains ticks per beat
    if (midi_data.GetCount() >= 14) {
        // MIDI header is at beginning: MThd chunk
        if (midi_data[0] == 'M' && midi_data[1] == 'T' && midi_data[2] == 'h' && midi_data[3] == 'd') {
            // Ticks per beat is in bytes 12-13 of header (after chunk size)
            int ticks = (midi_data[12] << 8) | midi_data[13];
            return ticks;
        }
    }
    
    return 480; // Default ticks per beat
}

// Piano Roll Visualization Implementation
void PianoRollCtrl::Paint(Draw& w) {
    Size sz = GetSize();
    w.DrawRect(sz, SColorFace());
    
    // Draw grid
    int note_height = sz.cy / 128; // 128 MIDI notes
    
    // Draw horizontal lines for each MIDI note
    for (int i = 0; i < 129; i++) {
        int y = sz.cy - (i * note_height);
        Color line_color = (i % 12 == 0) ? Gray() : LtGray(); // Every 12 notes (octave)
        w.DrawLine(0, y, sz.cx, y, line_color);
    }
    
    // Draw vertical time grid
    for (int i = 0; i <= 10; i++) {
        int x = i * sz.cx / 10;
        w.DrawLine(x, 0, x, sz.cy, LtGray());
    }
    
    // Draw notes
    for (const auto& note : notes) {
        int start_time = note.start_time;
        int duration = note.duration;
        int pitch = note.pitch;
        
        // Map time to X coordinate
        int x_start = (start_time * sz.cx) / (max_time + 1000); // 1000ms buffer
        int x_width = (duration * sz.cx) / (max_time + 1000);
        
        // Map pitch to Y coordinate (inverted: lower notes at bottom)
        int y = sz.cy - ((pitch + 1) * note_height);
        int height = note_height;
        
        // Color based on note velocity and whether it's input or generated
        Color note_color = is_input_roll ? Color(0, 0, 200) : Color(200, 0, 0); // Blue for input, Red for generated
        note_color = Blend(note_color, White(), note.velocity / 127.0); // Brightness based on velocity
        
        w.DrawRect(x_start, y, x_width, height, note_color);
        w.DrawRect(x_start, y, x_width, height, Black()); // Outline
    }
    
    // Draw title
    w.DrawText(10, 5, is_input_roll ? "Input MIDI" : "Generated MIDI", StdFont().Bold(), Black());
}

void PianoRollCtrl::MouseWheel(Point p, int zdelta, dword keyflags) {
    // Zoom in/out based on mouse wheel
    // Implementation would adjust the time range displayed
    Refresh();
}

// Attention Visualization Implementation
void AttentionVisualizerCtrl::SetAttentionWeights(const Array<Volume>& weights) {
    attention_weights.Clear();
    for(int i = 0; i < weights.GetCount(); i++) {
        attention_weights.Add(weights[i]);
    }
    Refresh();
}

void AttentionVisualizerCtrl::Paint(Draw& w) {
    Size sz = GetSize();
    w.DrawRect(sz, SColorFace());
    
    // Draw title
    w.DrawText(5, 5, "Attention Heatmap", StdFont().Bold(), Black());
    
    if (attention_weights.IsEmpty()) return;
    
    if (layer_idx < attention_weights.GetCount()) {
        Volume layer_attention = attention_weights[layer_idx];
        if (head_idx < layer_attention.GetLength()) {
            // This is a simplified visualization
            // Real implementation would need to extract the specific attention head
            w.DrawText(10, 25, Format("Layer %d, Head %d", layer_idx, head_idx), StdFont(), Black());
            
            // Draw attention as a heatmap if the dimensions are available
            // For now, just show placeholder
            w.DrawText(10, 45, "Attention weights visualization", StdFont(), Black());
        }
    }
}

// Main MidiLearner Implementation
MidiLearner::MidiLearner() {
    Title("MIDI Learner and Generator");
    // Icon(MidiLearnerImg::icon()); // Comment out until we create a proper icon
    Sizeable().MaximizeBox().MinimizeBox().Zoomable();
    
    // Initialize training state
    running = false;
    stopped = true;
    paused = false;
    
    // Setup UI components
    load_midi_btn.SetLabel("Load MIDI Files");
    play_btn.SetLabel("Play");
    stop_btn.SetLabel("Stop");
    start_training_btn.SetLabel("Start Training");
    stop_training_btn.SetLabel("Stop Training");
    
    tempo_slider.MinMax(20, 240);
    tempo_slider.SetData(120);
    
    // Setup callbacks
    load_midi_btn <<= THISBACK(LoadMIDIFiles);
    play_btn <<= THISBACK(PlayCurrentSong);
    stop_btn <<= THISBACK(StopPlayback);
    start_training_btn <<= THISBACK(StartTraining);
    stop_training_btn <<= THISBACK(StopTraining);
    tempo_slider <<= THISBACK(SetTempo);
    
    // Initialize network config
    network_config_editor.SetData(
        "[\n"
        "\t{\"type\":\"input\", \"input_width\":1, \"input_height\":1, \"input_depth\":388},\n"  // MIDI vocab size
        "\t{\"type\":\"transformer\", \"n_layer\":8, \"n_head\":8, \"n_embd\":512},\n"
        "\t{\"type\":\"fc\", \"neuron_count\":388, \"activation\": \"linear\"},\n"  // Output same size as input
        "\t{\"type\":\"adam\", \"learning_rate\":0.0003, \"beta1\":0.9, \"beta2\":0.98, \"eps\":1e-9}\n"
        "]"
    );
    
    // Initialize status labels
    current_song_label.SetLabel("No song loaded");
    status_label.SetLabel("Ready");
    analysis_info_label.SetLabel("MIDI Analysis: ");
    
    // Setup visualization controls
    input_piano_roll.SetNotes(Vector<struct MIDINote>());
    generated_piano_roll.SetNotes(Vector<struct MIDINote>());
    
    // Setup layout
    Add(load_midi_btn.LeftPos(5, 100).TopPos(5, 24));
    Add(play_btn.LeftPos(110, 80).TopPos(5, 24));
    Add(stop_btn.LeftPos(195, 80).TopPos(5, 24));
    Add(start_training_btn.LeftPos(280, 100).TopPos(5, 24));
    Add(stop_training_btn.LeftPos(385, 100).TopPos(5, 24));
    Add(tempo_slider.LeftPos(490, 150).TopPos(5, 24));
    Add(current_song_label.LeftPos(5, 200).TopPos(50, 20));
    Add(status_label.LeftPos(210, 200).TopPos(50, 20));
    Add(analysis_info_label.LeftPos(415, 300).TopPos(50, 20));
    
    // Set initial position and size
    Size sz = Zsz(1000, 700);
    SetRect(0, 0, sz.cx, sz.cy);
    
    // Start the refresh loop
    PostCallback(THISBACK(RefreshUI));
}

void MidiLearner::DockInit() {
    // Setup dockable panels
    ParentCtrl network_editor_panel;
    network_editor_panel.Add(network_config_editor.HSizePos().VSizePos(0, 30));
    network_editor_panel.Add(start_training_btn.HSizePos(0, 105).BottomPos(0, 30));
    network_editor_panel.Add(stop_training_btn.RightPos(0, 105).BottomPos(0, 30));
    
    ParentCtrl training_panel;
    training_panel.Add(training_graph_ctrl.HSizePos().VSizePos());
    
    AutoHide(DOCK_LEFT, Dockable(network_editor_panel, "Network Config").SizeHint(Size(400, 300)));
    AutoHide(DOCK_BOTTOM, Dockable(training_panel, "Training Status").SizeHint(Size(600, 200)));
    DockRight(Dockable(attention_visualizer, "Attention").SizeHint(Size(300, 400)));
}

void MidiLearner::LoadMIDIFiles() {
    String files = SelectFileOpen("MIDI files\t*.mid;*.midi\nAll files\t*.*");
    if (files.IsEmpty()) return;
    
    // For now, just add the single selected file
    // In a real implementation, you'd use a multi-file selector
    
    midi_filenames.Clear();
    midi_analysis_strings.Clear();
    
    midi_filenames.Add(files);
    MidiProperties props = MidiAnalyzer::AnalyzeMidiFile(files);
    // Create a simple string representation of the properties
    String props_str = Format("Tempo:%.2f,Key:%s,Signature:%s,Swing:%s", 
                              props.tempo, props.key, props.signature, 
                              props.is_swing ? "Yes" : "No");
    midi_analysis_strings.Add(props_str);
    
    AnalyzeMIDIFiles();
    UpdateAnalysisDisplay();
    
    // TODO: Load the MIDI data into the application
    current_song_label.SetLabel(Format("%d MIDI files loaded", midi_filenames.GetCount()));
}

void MidiLearner::AnalyzeMIDIFiles() {
    // Already analyzed in LoadMIDIFiles, but could perform additional analysis here
}

void MidiLearner::UpdateAnalysisDisplay() {
    if (midi_analysis_strings.GetCount() > 0) {
        // For now, just show that analysis is available
        // In a real implementation, this would deserialize the JSON and show details
        analysis_info_label.SetLabel("MIDI analysis completed");
        // Future: Parse the JSON string to extract and display properties
    }
}

void MidiLearner::StartTraining() {
    if (running) StopTraining();
    
    running = true;
    stopped = false;
    ReloadNetwork();
    
    Thread::Start(THISBACK(TrainingLoop));
    status_label.SetLabel("Training started");
}

void MidiLearner::StopTraining() {
    running = false;
    while (!stopped) Sleep(100);  // Wait for thread to finish
    status_label.SetLabel("Training stopped");
}

void MidiLearner::ReloadNetwork() {
    session.StopTraining();
    String config = network_config_editor.GetData();
    bool success = session.MakeLayers(config);
    if (success) {
        session.StartTraining();
    }
}

void MidiLearner::PlayCurrentSong() {
    // TODO: Implement actual MIDI playback
    status_label.SetLabel("Playing MIDI...");
}

void MidiLearner::StopPlayback() {
    status_label.SetLabel("Playback stopped");
}

void MidiLearner::SetTempo() {
    // TODO: Implement tempo adjustment
    status_label.SetLabel(Format("Tempo set to %d BPM", (int)tempo_slider.GetData()));
}

void MidiLearner::TrainingLoop() {
    while (running) {
        if (!paused && !midi_filenames.IsEmpty()) {
            // Train on MIDI files
            for (int i = 0; i < midi_filenames.GetCount(); i++) {
                TrainOnMIDISong(midi_filenames[i]);
            }
            
            // Generate new song periodically
            if (session.GetIteration() % 100 == 0) {
                GenerateNewSong();
            }
        }
        Sleep(10);  // Control training speed
    }
    stopped = true;
}

MIDIDataManager::MIDISong MIDIDataManager::LoadMIDIFile(const String& filename) {
    FileIn in(filename);
    if (!in) {
        LOG("Could not open MIDI file: " << filename);
        return MIDISong();
    }
    
    // Read the entire file into a byte vector
    Vector<byte> data;
    int c;
    while ((c = in.Get()) != -1) {
        data.Add(c);
    }
    
    // Analyze MIDI properties
    MidiProperties props = MidiAnalyzer::AnalyzeMidiData(data);
    
    // Create a simple song structure
    // In a real implementation, we would properly parse the MIDI format
    MIDISong song;
    song.tempo = props.tempo;
    song.ticks_per_beat = props.ticks_per_beat;
    // For this example, we'll create some dummy notes
    // A real implementation would extract actual notes from the MIDI file
    
    // Add some notes based on the MIDI analysis
    for (int i = 0; i < 50; i++) {
        MIDINote note;
        note.start_time = i * 480; // Every beat (assuming 480 ticks per beat)
        note.duration = 240; // Half a beat
        note.pitch = 60 + (i % 24); // Cycle through notes
        note.velocity = 80 + (i % 40); // Vary velocity
        note.channel = 0;
        song.notes.Add(note);
    }
    
    return song;
}

void MIDIDataManager::PlayMIDISong(const MIDISong& song, int tempo_bpm) {
    // TODO: Implement actual MIDI playback functionality
    // This would require actual MIDI output to the system
}

void MIDIDataManager::StopPlayback() {
    // TODO: Implement MIDI playback stop
}

Vector<int> MIDIDataManager::ConvertToTokenSequence(const MIDISong& song) {
    Vector<int> tokens;
    
    // Use a vocabulary that includes:
    // 0-127: Note on events
    // 128-255: Note off events 
    // 256-383: Time shift events
    // 384-387: Special tokens
    
    // Add start token
    tokens.Add(384);  // Start token
    
    // Convert notes to tokens
    for (const auto& note : song.notes) {
        // Add time shift if needed
        if (note.start_time > 0) {
            tokens.Add(min(127, note.start_time / 10) + 256);  // Time shift token
        }
        
        // Add note on
        tokens.Add(min(127, note.pitch));  // Note on token
        
        // Add velocity or duration info (simplified)
        tokens.Add(min(127, note.velocity + 128));  // Note off token with velocity encoding
    }
    
    // Add end token
    tokens.Add(385);  // End token
    
    return tokens;
}

MIDIDataManager::MIDISong MIDIDataManager::GenerateSong(Session& session, int length) {
    MIDISong generated_song;
    
    // Start with initial tokens
    Vector<int> input_tokens;
    input_tokens.Add(384);  // Start token
    
    // Generate sequence
    for (int i = 0; i < length && i < 200; i++) {  // Max 200 notes to prevent infinite generation
        // Convert tokens to volume for the neural network
        Volume input_vol(1, 1, 1);
        if (!input_tokens.IsEmpty()) {
            input_vol.Set(0, 0, 0, input_tokens.Top());
        }
        
        // Get prediction from the network - convert Volume to Vector<double>
        Vector<double> input_vec;
        if (input_vol.GetCount() > 0) {
            input_vec.SetCount(input_vol.GetCount());
            for(int i = 0; i < input_vol.GetCount(); i++) {
                input_vec[i] = input_vol.Get(i, 0, 0);  // Use Get(i, j, k) for Volume access
            }
        }
        Vector<double> output = session.Predict(input_vec);
        
        // Sample next token from output distribution
        int next_token = SampleFromDistribution(output, 0.8);
        
        // Process the token
        if (next_token >= 0 && next_token <= 127) {
            // Note ON - create a new note
            MIDINote note;
            note.start_time = generated_song.notes.GetCount() * 480; // Simple timing
            note.pitch = next_token;
            note.velocity = 100;
            note.duration = 240; // Half beat
            note.channel = 0;
            generated_song.notes.Add(note);
        } else if (next_token >= 128 && next_token <= 255) {
            // Note OFF or other event
            if (!generated_song.notes.IsEmpty()) {
                generated_song.notes.Top().duration = 240; // Update last note duration
            }
        } else if (next_token >= 256 && next_token <= 383) {
            // Time shift
            // In a real implementation, adjust timing based on this
        } else if (next_token == 385) {  // End token
            break;
        }
        
        // Add to input sequence for next prediction
        input_tokens.Add(next_token);
        if (input_tokens.GetCount() > 1024) {
            input_tokens.Remove(0); // Limit context
        }
    }
    
    return generated_song;
}

Vector<double> MIDIDataManager::ApplyTemperature(const Vector<double>& distribution, double temperature) {
    // Apply temperature scaling to the distribution
    Vector<double> scaled_dist;
    scaled_dist.SetCount(distribution.GetCount());
    
    for (int i = 0; i < distribution.GetCount(); i++) {
        scaled_dist[i] = distribution[i];  // Copy element
        if (scaled_dist[i] > 0) {  // Only apply to positive probabilities
            scaled_dist[i] = pow(scaled_dist[i], 1.0 / temperature);
        }
    }
    
    // Normalize
    double sum = 0;
    for (int i = 0; i < scaled_dist.GetCount(); i++) {
        sum += scaled_dist[i];
    }
    
    if (sum > 0) {
        for (int i = 0; i < scaled_dist.GetCount(); i++) {
            scaled_dist[i] /= sum;
        }
    }
    
    return scaled_dist;
}

int MIDIDataManager::SampleFromDistribution(const Vector<double>& distribution, double temperature) {
    Vector<double> temp_dist = ApplyTemperature(distribution, temperature);
    
    // Sample from the distribution
    double r = Randomf();
    double sum = 0.0;
    
    for (int i = 0; i < temp_dist.GetCount(); i++) {
        sum += temp_dist[i];
        if (r <= sum) {
            return i;
        }
    }
    
    return 0; // Fallback
}

void MidiLearner::TrainOnMIDISong(const String& song_path) {
    // Load the MIDI song
    MIDIDataManager::MIDISong song = midi_manager.LoadMIDIFile(song_path);
    
    // Convert to token sequence
    Vector<int> token_sequence = midi_manager.ConvertToTokenSequence(song);
    
    if (token_sequence.IsEmpty()) return;
    
    // Process in chunks if sequence is too long
    int chunk_size = 512; // Max context length
    for (int i = 0; i < token_sequence.GetCount() - 1; i += chunk_size) {
        int end_idx = min(i + chunk_size, token_sequence.GetCount());
        
        // Create input and target sequences (teacher forcing)
        Vector<int> input_tokens;
        Vector<int> target_tokens;
        
        for (int j = i; j < end_idx - 1; j++) {
            input_tokens.Add(token_sequence[j]);
        }
        for (int j = i + 1; j < end_idx; j++) {
            target_tokens.Add(token_sequence[j]);
        }
        
        // Convert tokens to volumes for neural network training
        if (!input_tokens.IsEmpty()) {
            Volume input_vol(input_tokens.GetCount(), 1, 1);
            Volume target_vol(target_tokens.GetCount(), 1, 1);
            
            for (int j = 0; j < input_tokens.GetCount(); j++) {
                input_vol.Set(j, 0, 0, input_tokens[j]);
            }
            for (int j = 0; j < target_tokens.GetCount(); j++) {
                target_vol.Set(j, 0, 0, target_tokens[j]);
            }
            
            // Convert target_vol to Vector<double> for training
            Vector<double> target_vec;
            target_vec.SetCount(target_vol.GetCount());
            for (int j = 0; j < target_vol.GetCount(); j++) {
                target_vec[j] = target_vol.Get(j, 0, 0);  // Use Get(i, j, k) for Volume access
            }
            
            // Train the network
            session.TrainOnce(input_vol, target_vec);
        }
    }
}

void MidiLearner::GenerateNewSong() {
    // Use the trained network to generate a new song
    MIDIDataManager::MIDISong generated_song = midi_manager.GenerateSong(session, 100);
    
    // Update the visualization
    generated_piano_roll.SetNotes(generated_song.notes);
    
    status_label.SetLabel(Format("New song generated at iteration: %d", session.GetIteration()));
}

void MidiLearner::RefreshUI() {
    // Update visualization controls
    layer_ctrl.Refresh();
    training_graph_ctrl.Refresh();
    attention_visualizer.Refresh();
    
    // Update status
    status_label.SetLabel(Format("Training Iteration: %d, Loss: %.4f", 
                                session.GetIteration(), session.GetLossAverage()));
    
    // Schedule next refresh
    PostCallback(THISBACK(RefreshUI));
}