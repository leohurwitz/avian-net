import librosa
import numpy as np
from pathlib import Path
from src.config_loader import load_config

CONFIG = load_config()

def is_audible_rms(audio_chunk, threshold):
    rms_values = librosa.feature.rms(y=audio_chunk)
    mean_rms = np.mean(rms_values)

    if mean_rms > threshold:
        return True
    else:
        return False

def process_species_data(file_path ,base_directory=None):

    # Creating Processed File Path
    species_name = Path(file_path).parent.name

    if base_directory is None:
        base_directory = Path.cwd() / "data"
    else:
        base_directory = Path(base_directory)

    raw_directory = base_directory / "raw" / species_name
    processed_directory = base_directory / 'processed' / species_name

    processed_directory.mkdir(parents=True, exist_ok=True)

    # Pull variables from CONFIG dict
    sr = CONFIG['audio']['sample_rate']
    chunk_size = CONFIG['audio']['chunk_size_samples']
    rms_thresh = CONFIG['audio']['rms_threshold']
    n_mels = CONFIG['audio']['n_mels']
    fmax = CONFIG['audio']['fmax']

    # Loading files through librosa, chunks them
    bird_call_arr = librosa.load(file_path, sr=22050)[0]
    chunk_size = 110250
    indices = list(range(chunk_size, len(bird_call_arr), chunk_size))
    bird_calls = np.array_split(bird_call_arr, indices)


    # Loop through bird_calls list and convert every array into a Mel-Spectrogram
    num_discarded_chunks = 0
    num_kept_chunks = 0
    for i, bird_call in enumerate(bird_calls):
        if is_audible_rms(bird_call, rms_thresh) == True: # Filters silent chunks out of the dataset
            if len(bird_call) < 110250:
                padding_needed = 110250 - len(bird_call) # Pads last chunk to fit size
                bird_call = np.pad(bird_call, (0, padding_needed))

            mel_spectrogram = librosa.feature.melspectrogram(y=bird_call, 
                sr=sr, 
                n_mels=n_mels, 
                fmax=fmax
            )
            mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
            np.save(processed_directory / f"{Path(file_path).stem}_chunk_{i}", mel_spectrogram_db)
            num_kept_chunks += 1
        else:
            num_discarded_chunks += 1
    # Added functionality to determine # of kept/discarded chunks per species after using RMS Filtering
    return {
        "species" : species_name,
        "kept" : num_kept_chunks,
        "discarded" : num_discarded_chunks
    }
        
