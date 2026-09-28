import torch
import librosa
import numpy as np
from pathlib import Path
from src.processing import is_audible_rms 
from src.config_loader import load_config
from train_model import initialize_model
import json

CONFIG = load_config()


sr = CONFIG['audio']['sample_rate']
chunk_size = CONFIG['audio']['chunk_size_samples']
rms_thresh = CONFIG['audio']['rms_threshold']
n_mels = CONFIG['audio']['n_mels']
fmax = CONFIG['audio']['fmax']

# Loads .mp3 file, converts to array, splits into 5-second chunks
file_path = input("Enter file path for bird call .mp3 file: ")
bird_call_arr = librosa.load(file_path, sr=sr)[0]
indices = list(range(chunk_size, len(bird_call_arr), chunk_size))
bird_calls = np.array_split(bird_call_arr, indices)
tensor_list = []

# Temporary counter to see how many chunks were silenced
num_chunks_before_silencing = 0

for bird_call in bird_calls:
    num_chunks_before_silencing += 1
    if is_audible_rms(bird_call, rms_thresh) == True: # Filters silent chunks out of the file
                if len(bird_call) < 110250:
                    padding_needed = 110250 - len(bird_call) # Pads last chunk to fit size
                    bird_call = np.pad(bird_call, (0, padding_needed))
                # Converts to Mel-Spectrogram (Decibels)
                mel_spectrogram = librosa.feature.melspectrogram(y=bird_call, 
                    sr=sr, 
                    n_mels=n_mels, 
                    fmax=fmax
                )
                mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max) 
                # Converts into tensor, standardizes data, expands to 3 channels
                tensor_data = torch.from_numpy(mel_spectrogram_db).float().unsqueeze(0)
                std_val, mean_val = torch.std_mean(tensor_data)
                tensor_data = (tensor_data - mean_val) / (std_val + 1e-7)
                tensor_data = tensor_data.expand(3, -1, -1) # Tensor size [[3],[128],[216]]
                tensor_list.append(tensor_data)
# temporary to see which chunk was discarded aswell
    else:
        print(f"Chunk discard = chunk {num_chunks_before_silencing}")
print(f"Number of chunks before silencing: {num_chunks_before_silencing}")
print(f"Number of chunks after silencing: {len(tensor_list)}")

tensor_batch = torch.stack(tensor_list)
torch.manual_seed(42)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

model = initialize_model(device=device)
model_path = Path.cwd() / 'models' / 'avian_net_v3_loss_scheduler.pth'
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

with torch.inference_mode():
        # Forward Pass, returns 2D tensor of shape [[batch_size],[num_classes]]
        pred_logits = model(tensor_batch.to(device)) 
        # Convert logits to probabilities using softmax
        softmax = torch.nn.Softmax(dim=1)
        pred_probs = softmax(pred_logits)
        # Now we have multiple data samples with their own probabilities. We want to combine that into 1 probability/prediction for the whole batch
        pred_probs = torch.mean(pred_probs, dim=0) # leads to 1D tensor of shape [num_classes]
        pred_idx = torch.argmax(pred_probs, dim=0).cpu().item()
        pred_prob = torch.max(pred_probs).cpu().item()

# Loads the species_to_idx_map.json then flips the dictionary around
with open(Path.cwd() / 'models' / 'species_to_idx_map.json', 'r', encoding='utf-8') as file:
        species_to_idx_map = json.load(file)

idx_to_species_map = {}
for key, value in species_to_idx_map.items():
    idx_to_species_map[value] = key

predicted_species = idx_to_species_map[pred_idx]
print(f"Model predicts bird as {predicted_species} with confidence level of {pred_prob}")