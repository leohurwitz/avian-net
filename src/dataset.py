import torch
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import FrequencyMasking, TimeMasking

import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
import json
from src.config_loader import load_config

CONFIG = load_config()

# Defining SpecAugment Transformations Class
class SpecAugment:
    def __init__(self, freq_param=15, time_param=30, mask_value=0):
        self.freq_mask = FrequencyMasking(freq_mask_param=freq_param)
        self.time_mask = TimeMasking(time_mask_param=time_param)
        self.mask_value = mask_value

    def __call__(self, tensor):
        x = self.freq_mask(tensor, mask_value=self.mask_value)
        x = self.freq_mask(x, mask_value=self.mask_value)
        x = self.time_mask(x, mask_value=self.mask_value)
        x = self.time_mask(x, mask_value=self.mask_value)
        return x


# PyTorch Dataset Class
class BirdCallDataset(Dataset):
    def __init__(self, filepaths, labels, species_to_idx_dict, transform=None): # self is the name we decide for the class instance, filepaths is NumPy array, labels is a list

        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform
        self.species_to_idx_dict= species_to_idx_dict

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx): 
        # Use idx to index the arrays for the filepath and label (then convert label to integer)
        file_path = self.filepaths[idx]
        int_label = self.species_to_idx_dict[self.labels[idx]]
        tensor_label = torch.tensor(int_label, dtype=torch.long)

        # Then use np.load(filepath) 
        spectrogram_arr = np.load(file_path)

        # Convert to PyTorch Tensor, Add Channel Dimension, Normalize Tensor Values [[3],[128],[216]]
        tensor_data = torch.from_numpy(spectrogram_arr).float().unsqueeze(0)
        std_val, mean_val = torch.std_mean(tensor_data)
        tensor_data = (tensor_data - mean_val) / (std_val + 1e-7)
        
        # Do some sort of transformation
        if self.transform:
            tensor_data = self.transform(tensor_data)

        # Copy Values Across Channel Dimension
        tensor_data = tensor_data.expand(3, -1, -1)

        # Return the tensor and its label (tensor_data, tensor_label)
        return (tensor_data, tensor_label)

    
# Grouping Chunks for Train/Test Split

# Create list of all file paths
processed_dir = Path.cwd() / 'data' / 'processed'
all_file_paths = np.fromiter(processed_dir.rglob("*.npy"), dtype=object) # Note this turns the file_paths from Path objects to strings

labels = [] # Bird Species
groups = [] # Xeno-Canto IDs

for file_path in all_file_paths: # Creates 2 lists of Species/ID so GroupShuffleSplit can properly split to avoid data leakage
    species_label = Path(file_path).parent.name ## Path() I believe is obselete as it already should be a path ##
    labels.append(species_label)

    xc_id = file_path.name.split('_')[0]
    groups.append(xc_id)

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, train_size=0.8, random_state=42)
labels = np.array(labels) # Convert from list to array for indexing

# Split and filter file paths/labels
train_idx, test_idx = next(gss.split(all_file_paths, labels, groups))
train_paths = all_file_paths[train_idx]
test_paths = all_file_paths[test_idx]
train_labels = labels[train_idx]
test_labels = labels[test_idx]

# Create species --> integer dictionary
unique_species = list(np.unique(labels)) # Identifies/Sorts Species Names

idx_to_species_dict = {index: species for index, species in enumerate(unique_species)}
species_to_idx_dict = {species: index for index, species in enumerate(unique_species)} 

# Saving Dictionary as JSON File
with open(Path.cwd() / 'models' / 'species_to_idx_map.json', 'w') as file:
    json.dump(species_to_idx_dict, file, indent=4)

def get_dataloaders():
    # Compiling Pytorch Dataset
    training_data = BirdCallDataset(filepaths=train_paths, 
        labels=train_labels, 
        species_to_idx_dict=species_to_idx_dict, 
        transform=SpecAugment() # Default SpecAug parameters
    ) 
    test_data = BirdCallDataset(filepaths=test_paths, 
        labels=test_labels,     
        species_to_idx_dict=species_to_idx_dict
    )
    batch_size = CONFIG['training']['batch_size']

    # Data Loading
    train_dataloader = DataLoader(training_data,  batch_size=batch_size, num_workers=4, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, num_workers=4)

    return train_dataloader, test_dataloader
