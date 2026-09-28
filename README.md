# AvianNet: Bioacoustics Machine Learning Pipeline

**Author:** Leo Hurwitz

## Overview
AvianNet is an end-to-end deep learning pipeline designed to classify five species of North American birds from raw environmental audio. 

This project utilizes both a custom-built Convolutional Neural Network (CNN) and a fine-tuned ResNet18 architecture. The pipeline converts `.mp3` audio files into Mel Spectrograms while handling background noise, class imbalance, and data leakage.

### Target Species
* *Buteo jamaicensis* (Red-tailed Hawk)
* *Cardinalis cardinalis* (Northern Cardinal)
* *Corvus brachyrhynchos* (American Crow)
* *Cyanocitta cristata* (Blue Jay)
* *Strix varia* (Barred Owl)

---

## Data Engineering Pipeline

The model pipeline is designed to prevent learning wind patterns and static. The general architecture is below.

1. Audio is sourced directly from the public Xeno-Canto API.
2. 1D waveforms are evaluated for Root Mean Square (RMS) energy. Chunks that fail to cross the mean amplitude threshold (ex: silent pauses) are dropped, filtering out ~40% of the dead space. The data was also similarly evaluated using a max RMS threshold, filtering out ~14% of the data, far lower than the ~40% of mean RMS. While this reduced the model's performance, it provided a larger and more diverse dataset that trains a more robust model.
3. The surviving 5-second chunks are zero-padded for uniform size and converted into 3-channel, 128-band Mel Spectrograms.

### Mitigating Bias and Leakage
* Train and test splits are separated by the original recording ID. This ensures that sliced chunks from the exact same audio file do not bleed across the train/test boundary. This prevents data leakage.
* Because hawks produce significantly less vocalization than the other target species, the RMS filter caused some class imbalance. This was rectified through implementing inverse frequency class weights into the loss function.

---

## Model Architectures & Results

The project evaluates three separate neural network strategies to establish a baseline and further performance via Transfer Learning. Additionally, both RMS mean and max filtering methods are tested. Finally, a learning rate scheduler is tested on AvianNet V3 (Transfer Learning + RMS Max), though with little benefit.
### RMS Mean Filtering
| Architecture | Strategy | Test Accuracy | Test Loss |
| :--- | :--- | :--- | :--- |
| **AvianNet V1** | Custom CNN (3 Conv Layers) | 81.4% | ~0.59 | 
| **AvianNet V2** | ResNet18 (Frozen Backbone) | 88.0% | ~0.37 | 
| **AvianNet V3** | ResNet18 (Fine-Tuned) | **89.0%** | **~0.31** |
### RMS Max Filtering
| Architecture | Strategy | Test Accuracy | Test Loss |
| :--- | :--- | :--- | :--- |
| **AvianNet V1** | Custom CNN (3 Conv Layers) | 68% | ~0.93 | 
| **AvianNet V2** | ResNet18 (Frozen Backbone) | 82% | ~0.60 | 
| **AvianNet V3** | ResNet18 (Fine-Tuned) | **83%** | **~0.56** |
### RMS Max Filtering with Learning Rate Scheduler
| Architecture | Strategy | Test Accuracy | Test Loss |
| :--- | :--- | :--- | :--- |
| **AvianNet V3** | ResNet18 (Fine-Tuned) | **83%** | **~0.56** |

### Biological Context & Model Confusion
While AvianNet V3 (Mean RMS) achieved 89% accuracy, the most complex challenge involved the **Blue Jay** (*Cyanocitta cristata*), which achieved the lowest class F1-score (81%). 

Analysis of the Confusion Matrix revealed the model frequently misclassified Blue Jays as Red-tailed Hawks. This is most likely due to the fact that Blue Jays actively imitate the call of the Red-tailed Hawk to scare other birds away from feeders. The model successfully identified the acoustic features of a hawk, even when produced by a jay.

---

## Repository Structure
```text
bioacoustics/
├── .gitignore
├── requirements.txt
├── config.yaml              # Centralized hyperparameters (RMS, chunk size, LR)
├── model_evaluation.ipynb   # Visual dashboard: Confusion matrices and F1 reports
├── run_pipeline.py          # Executes data ingestion and processing
├── train_model.py           # Executes the training loop and saves best weights
├── predict.py               # Executes inference and outputs prediction and probability
└── src/                     
    ├── config_loader.py     # Parses the YAML configuration
    ├── ingestion.py         # Xeno-Canto API wrapper
    ├── processing.py        # RMS silence filter and spectrogram conversion
    ├── dataset.py           # PyTorch Dataset and DataLoader classes
    └── model.py             # AvianNet architectures
```
## How to Run

**1. Set Configs**
Open `config.yaml` in the root directory. Here you can adjust data thresholds (like the RMS silence filter) and switch the `model_version` between `v1` (Custom CNN), `v2` (ResNet18, frozen weights), and `v3` (ResNet18, unfrozen weights). If you are pulling data from the Xeno-Canto API, you must insert your key there.

**2. Generate the Dataset**
Run the ingestion and processing script by executing `run_pipeline.py`. This will download the raw `.mp3` files, dynamically filter out the silent chunks, and save the final `.npy` Mel Spectrograms to a `data/processed/` folder. 

**3. Train the Model**
Execute the training loop by running `train_model.py`. The script will read your dataset, apply inverse frequency class weights, and train the architecture specified in your config file. The best weights will be saved.

**4. Evaluate Results**
Launch the visual analysis with `model_evaluation.ipynb`. The notebook runs each model on the test data to generate a Confusion Matrix and a Classification Report.
