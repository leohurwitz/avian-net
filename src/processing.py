import librosa
import numpy as np
from pathlib import Path


def process_species_data(file_path ,base_directory=None):
    # File path should be just the species folder + .mp3 name

    # Creating Processed File Path
    species_name = Path(file_path).parent.name

    if base_directory is None:
        base_directory = Path.cwd() / "data"
    else:
        base_directory = Path(base_directory)

    raw_directory = base_directory / "raw" / species_name
    processed_directory = base_directory / 'processed' / species_name

    processed_directory.mkdir(parents=True, exist_ok=True)

    # Loading files through librosa & setting up chunking/padding
    bird_call_arr = librosa.load(file_path, sr=22050)[0]
    chunk_size = 110250
    indices = list(range(chunk_size, len(bird_call_arr), chunk_size))
    bird_calls = np.array_split(bird_call_arr, indices)

    padding_needed = 110250 - len(bird_calls[-1])
    bird_calls[-1] = np.pad(bird_calls[-1], (0, padding_needed))

    # Loop through bird_calls list and convert every array into a Mel-Spectrogram

    for i, bird_call in enumerate(bird_calls):
        mel_spectrogram = librosa.feature.melspectrogram(y=bird_call, 
            sr=22050, 
            n_mels=128, 
            fmax=8000
        )
        mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
        np.save(processed_directory / f"{Path(file_path).stem}_chunk_{i}", mel_spectrogram_db)
