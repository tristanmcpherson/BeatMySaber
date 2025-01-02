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
print("Features shape:", features.shape)  # Debug print

# Define slicing parameters
slice_duration_ms = config.slice_duration_ms
sample_rate = config.sample_rate
hop_length = audio_feature_extractor.hop_length

# Modify slicing to handle the actual feature shape
frames_per_slice = int((slice_duration_ms / 1000) * sample_rate / hop_length)
slices = []
for i in range(0, features.shape[-1] - frames_per_slice, frames_per_slice):
    slice_feat = features[..., i:i + frames_per_slice]
    slices.append(slice_feat)

# Make predictions
predicted_notes = []
with torch.no_grad():
    for i, slice_feat in enumerate(slices):
        slice_feat = slice_feat.unsqueeze(0).to('cuda').float()
        outputs = model(slice_feat)
        beat_time_offset_pred, lineIndex_logits, lineLayer_logits, note_type_logits, cut_direction_logits = outputs

        # Handle multiple predictions per slice
        for note_idx in range(model.max_notes):
            # Get probabilities for all classifications
            note_type_probs = F.softmax(note_type_logits[0, note_idx], dim=0)
            cut_direction_probs = F.softmax(cut_direction_logits[0, note_idx], dim=0)
            
            # Only proceed if we have valid note types (0=red, 1=blue) with high confidence
            note_type = torch.argmax(note_type_logits[0, note_idx]).item()
            if note_type not in [0, 1] or note_type_probs.max().item() < 0.8:
                continue

            # Only proceed if we have valid cut directions (0-8) with high confidence
            cut_direction = torch.argmax(cut_direction_logits[0, note_idx]).item()
            if cut_direction not in range(9) or cut_direction_probs.max().item() < 0.8:
                continue

            # Only proceed if we have valid beat time offset
            beat_time_offset = beat_time_offset_pred[0, note_idx].item()
            if beat_time_offset < 0:
                continue

            # Only proceed if we have valid line positions
            lineIndex = torch.argmax(lineIndex_logits[0, note_idx]).item()
            lineLayer = torch.argmax(lineLayer_logits[0, note_idx]).item()
            if not (0 <= lineIndex <= 3 and 0 <= lineLayer <= 2):
                continue

            predicted_notes.append({
                'slice_index': i,
                'beat_time_offset': beat_time_offset,
                'lineIndex': lineIndex,
                'lineLayer': lineLayer,
                'note_type': note_type,
                'cut_direction': cut_direction
            })

def filter_notes(notes, min_time_spacing=0.1):
    # Sort by time
    notes.sort(key=lambda x: x['slice_index'] * slice_duration_ms + x['beat_time_offset'])
    
    # Filter notes that are too close together
    filtered_notes = []
    last_time = -float('inf')
    
    for note in notes:
        current_time = note['slice_index'] * slice_duration_ms + note['beat_time_offset']
        if current_time - last_time >= min_time_spacing * 1000:  # Convert to ms
            filtered_notes.append(note)
            last_time = current_time
    
    return filtered_notes

# Move filtering before note_events processing
predicted_notes = filter_notes(predicted_notes)

# Post-process predictions
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

# After generating the beatmap
print(f"Generated {len(note_events)} notes")
print(f"Average notes per second: {len(note_events) / (audio.shape[1] / sample_rate):.2f}")
