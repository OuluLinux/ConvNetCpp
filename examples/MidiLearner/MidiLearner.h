#ifndef _MidiLearner_MidiLearner_h
#define _MidiLearner_MidiLearner_h

#include <CtrlLib/CtrlLib.h>
#include <ConvNet/ConvNet.h>
#include <Docking/Docking.h>
#include <Core/Core.h>
#include <Draw/Draw.h>
using namespace Upp;
using namespace ConvNet;

#define IMAGECLASS MidiLearnerImg
#define IMAGEFILE <MidiLearner/MidiLearner.iml>
#include <Draw/iml_header.h>

// MIDI Note structure - U++ compatible
struct MIDINote {
    int start_time;    // Start time in ticks
    int duration;      // Duration in ticks
    int pitch;         // MIDI note number (0-127)
    int velocity;      // Note velocity (0-127)
    int channel;       // MIDI channel (0-15)
    
    MIDINote() : start_time(0), duration(0), pitch(0), velocity(0), channel(0) {}
    MIDINote(int st, int d, int p, int v, int c) : start_time(st), duration(d), pitch(p), velocity(v), channel(c) {}
    
    bool operator==(const MIDINote& other) const {
        return start_time == other.start_time && duration == other.duration && 
               pitch == other.pitch && velocity == other.velocity && channel == other.channel;
    }
    
    void Jsonize(JsonIO& json) {
        json
            ("start_time", start_time)
            ("duration", duration)
            ("pitch", pitch)
            ("velocity", velocity)
            ("channel", channel);
    }
};

// MIDI Analysis structures - U++ compatible
struct MidiProperties {
    double tempo;           // Tempo in BPM
    String key;             // Key signature (e.g., "C major", "A minor")
    String signature;       // Time signature (e.g., "4/4", "3/4")
    bool is_swing;          // True if swing/shuffle, false if straight
    int ticks_per_beat;     // Ticks per beat for timing precision
    
    MidiProperties() : tempo(120.0), is_swing(false), ticks_per_beat(480) {}
    
    bool operator==(const MidiProperties& other) const {
        return tempo == other.tempo && key == other.key && 
               signature == other.signature && is_swing == other.is_swing && 
               ticks_per_beat == other.ticks_per_beat;
    }
    
    void Jsonize(JsonIO& json) {
        json
            ("tempo", tempo)
            ("key", key)
            ("signature", signature)
            ("is_swing", is_swing)
            ("ticks_per_beat", ticks_per_beat);
    }
};





class MidiAnalyzer {
public:
    static MidiProperties AnalyzeMidiFile(const String& filename);
    static MidiProperties AnalyzeMidiData(const Vector<byte>& midi_data);
    
private:
    static double ExtractTempo(const Vector<byte>& midi_data);
    static String ExtractKey(const Vector<byte>& midi_data);
    static String ExtractTimeSignature(const Vector<byte>& midi_data);
    static bool DetectSwing(const Vector<byte>& midi_data);
    static int ExtractTicksPerBeat(const Vector<byte>& midi_data);
};

// Piano Roll Visualization Control
class PianoRollCtrl : public Ctrl {
    Vector<struct MIDINote> notes;
    int min_time, max_time;
    int min_pitch, max_pitch;
    bool is_input_roll;
    
public:
    PianoRollCtrl(bool input_roll = true) : is_input_roll(input_roll), min_pitch(0), max_pitch(127) {}
    
    void SetNotes(const Vector<struct MIDINote>& n) { 
        notes.Clear();
        for(int i = 0; i < n.GetCount(); i++) {
            notes.Add(n[i]);
        }
        Refresh(); 
    }
    
    void Paint(Draw& w) override;
    void MouseWheel(Point p, int zdelta, dword keyflags) override;
    
    typedef PianoRollCtrl CLASSNAME;
};

// Attention Visualization Control
class AttentionVisualizerCtrl : public Ctrl {
    Array<Volume> attention_weights;
    int layer_idx, head_idx;
    
public:
    AttentionVisualizerCtrl() : layer_idx(0), head_idx(0) {}
    
    void SetAttentionWeights(const Array<Volume>& weights);
    void SetLayer(int layer) { layer_idx = layer; Refresh(); }
    void SetHead(int head) { head_idx = head; Refresh(); }
    
    void Paint(Draw& w) override;
    
    typedef AttentionVisualizerCtrl CLASSNAME;
};

// MIDI Data Manager
class MIDIDataManager {
public:
    struct MIDISong {
        Vector<MIDINote> notes;
        double tempo;
        String key;
        String time_signature;
        bool is_swing;
        int ticks_per_beat;
        
        MIDISong() : tempo(120.0), ticks_per_beat(480), is_swing(false) {}
    };
    
    MIDISong LoadMIDIFile(const String& filename);
    void PlayMIDISong(const MIDISong& song, int tempo_bpm);
    void StopPlayback();
    Vector<int> ConvertToTokenSequence(const MIDISong& song);
    MIDISong GenerateSong(Session& session, int length);
    
private:
    Vector<double> ApplyTemperature(const Vector<double>& distribution, double temperature = 1.0);
    int SampleFromDistribution(const Vector<double>& distribution, double temperature = 1.0);
};

class MidiLearner : public DockWindow {
    // MIDI Analysis and Management
    Vector<String> midi_filenames;
    // Store simple properties as strings since complex structs cause U++ compatibility issues
    Vector<String> midi_analysis_strings;  // Store as JSON strings
    MIDIDataManager midi_manager;
    
    // UI Components
    Button load_midi_btn;
    Button play_btn, stop_btn;
    Button start_training_btn, stop_training_btn;
    SliderCtrl tempo_slider;
    DocEdit network_config_editor;
    
    // Status Labels
    Label current_song_label;
    Label status_label;
    Label analysis_info_label;
    
    // Visualizations
    PianoRollCtrl input_piano_roll;
    PianoRollCtrl generated_piano_roll;
    AttentionVisualizerCtrl attention_visualizer;
    
    // Network Components
    Session session;
    // Using simple custom controls instead of ConvNetCtrl components that require PlotCtrl
    Ctrl training_graph_ctrl;
    Ctrl layer_ctrl;
    
    // Training State
    bool running, stopped, paused;
    
public:
    typedef MidiLearner CLASSNAME;
    MidiLearner();
    
    virtual void DockInit() override;
    
    // MIDI Management
    void LoadMIDIFiles();
    void AnalyzeMIDIFiles();
    void UpdateAnalysisDisplay();
    
    // Training Controls
    void StartTraining();
    void StopTraining();
    void PauseTraining();
    void ResumeTraining();
    void ReloadNetwork();
    
    // Playback Controls
    void PlayCurrentSong();
    void StopPlayback();
    void SetTempo();
    
    // Training Loop
    void TrainingLoop();
    void TrainOnMIDISong(const String& song_path);
    void GenerateNewSong();
    
    // UI Updates
    void RefreshUI();
};

#endif