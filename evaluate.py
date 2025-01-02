import torch
import torchaudio
import torch.nn.functional as F
import json

from AudioFeatureExtraction import AudioFeatureExtractor
from UNUSED_models import BeatSaberLightningModel

def load_and_preprocess_audio(audio_path):
    audio, sr = torchaudio.load(audio_path)
    audio = audio.float()
    # Ensure audio is mono
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    # Extract features
    features = audio_feature_extractor(audio)
    return features

# Load the trained model
checkpoint_path = 'beat_saber_model.ckpt'
n_features = 25
time_steps = 10

model = BeatSaberLightningModel.load_from_checkpoint(
    checkpoint_path,
    n_features=n_features,
    time_steps=time_steps
)
model.eval()
model.to('cuda')  # Or 'cuda' if using GPU

# Initialize the audio feature extractor
audio_feature_extractor = AudioFeatureExtractor(sample_rate=44100)

# Process the input audio file
audio_path = 'autumn.ogg'
features = load_and_preprocess_audio(audio_path)

# Slice the features
slice_duration_ms = 100
sample_rate = 44100
hop_length = audio_feature_extractor.hop_length

slice_frames = int(slice_duration_ms / 1000 * sample_rate / hop_length)
total_frames = features.shape[-1]
num_slices = total_frames // slice_frames

slices = []
for i in range(num_slices):
    start_frame = i * slice_frames
    end_frame = start_frame + slice_frames
    slice_features = features[:, :, start_frame:end_frame]
    if slice_features.shape[-1] < slice_frames:
        padding = slice_frames - slice_features.shape[-1]
        slice_features = F.pad(slice_features, (0, padding))
    if slice_features.dim() == 2:
        slice_features = slice_features.unsqueeze(0)
    slices.append(slice_features)

# Make predictions
predicted_notes = []
with torch.no_grad():
    for i, slice_features in enumerate(slices):
        slice_features = slice_features.unsqueeze(0)
        slice_features = slice_features.to(model.device).float()
        outputs = model(slice_features)
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
