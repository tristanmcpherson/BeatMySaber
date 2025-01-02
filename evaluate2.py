import torch
import torch.nn.functional as F
import torchaudio
import json

from AudioFeatureExtraction import AudioFeatureExtractor

from UNUSED_BeatSaberLSTM import BeatSaberLSTMModel
from BeatSaberTransformer import BeatSaberTransformerModel
from config import Config
from utils import analyze_label_distribution  # Updated model import

def load_and_preprocess_audio(audio_path, audio_feature_extractor):
    audio, sr = torchaudio.load(audio_path)
    audio = audio.float()
    # Ensure audio is mono
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    # Extract features
    features = audio_feature_extractor(audio)
    return features

config = Config()


# Initialize the audio feature extractor
audio_feature_extractor = AudioFeatureExtractor(sample_rate=44100)

# Load the trained Transformer model
checkpoint_path = 'beat_saber_model.ckpt'

model = BeatSaberTransformerModel.load_from_checkpoint(
    checkpoint_path
)
model.eval()
model.to('cuda')  # Use 'cuda' if GPU is available


# Process the input audio file
audio_path = 'paranoid.ogg'
features = load_and_preprocess_audio(audio_path, audio_feature_extractor)

# Define slicing parameters
slice_duration_ms = config.slice_duration_ms
sample_rate = config.sample_rate
hop_length = audio_feature_extractor.hop_length

# Slice the features

# Make predictions
predicted_notes = []
with torch.no_grad():
    for i, slice_feat in enumerate(slices):
        # Add batch and seq_len dimensions
        slice_feat = slice_feat.unsqueeze(0).unsqueeze(1).to('cuda').float()  # [1, 1, n_features, time_steps]
        outputs = model(slice_feat)
        beat_time_offset_pred, lineIndex_logits, lineLayer_logits, note_type_logits, cut_direction_logits = outputs

        beat_time_offset = beat_time_offset_pred.item()

        lineIndex = torch.argmax(lineIndex_logits, dim=1).item()
        lineLayer = torch.argmax(lineLayer_logits, dim=1).item()
        note_type = torch.argmax(note_type_logits, dim=1).item()
        cut_direction = torch.argmax(cut_direction_logits, dim=1).item()
        
        # Skip "no note" cases
        if note_type == 2:  # Assuming 2 represents "no note"
            continue  # Skip processing this slice

        predicted_notes.append({
            'slice_index': i,
            'beat_time_offset': beat_time_offset,
            'lineIndex': lineIndex,
            'lineLayer': lineLayer,
            'note_type': note_type,
            'cut_direction': cut_direction
        })

# Post-process predictions
def time_to_beat(time_sec, bpm):
    return time_sec * bpm / 60.0

bpm = 85.0  # Set BPM appropriately

note_events = []
for note in predicted_notes:
    note_type = int(note['note_type'])
    cut_direction = int(note['cut_direction'])

    # Skip "no note" cases
    if note_type == 2:
        continue
    if cut_direction == 9:
        continue

    # Process regression outputs only if they are valid
    if note['beat_time_offset'] == -1.0:
        continue

    # Denormalize regression outputs
    lineIndex = note['lineIndex']
    lineLayer = note['lineLayer']

    slice_start_time_sec = (note['slice_index'] * slice_duration_ms) / 1000.0
    note_time_sec = slice_start_time_sec + (note['beat_time_offset'] * (slice_duration_ms / 1000.0))
    beat_time = time_to_beat(note_time_sec, bpm)

    # Build the note event
    note_event = {
        "_time": beat_time,
        "_lineIndex": int(round(lineIndex)),
        "_lineLayer": int(round(lineLayer)),
        "_type": note_type,
        "_cutDirection": cut_direction
    }
    note_events.append(note_event)

# Generate beatmap file
beatmap_data = {
    "_version": "2.0.0",
    "_notes": note_events,
    "_obstacles": [],
    "_events": [],
}

output_beatmap_path = 'ExpertPlus.dat'
with open(output_beatmap_path, 'w') as f:
    json.dump(beatmap_data, f, indent=4)

print(f"Beatmap generated and saved to {output_beatmap_path}")
