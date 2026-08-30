# AvianNet: Bioacoustics Machine Learning Pipeline

**Author:** Leo Hurwitz

## Overview
AvianNet is an end-to-end deep learning pipeline designed to classify five species of North American birds from raw environmental audio. 

This project utilizes both a custom-built Convolutional Neural Network (CNN) and a fine-tuned ResNet18 architecture. The pipeline converts `.mp3` audio files into Mel Spectrograms handling background noise, class imbalance, and data leakage.

### Target Species
* *Buteo jamaicensis* (Red-tailed Hawk)
* *Cardinalis cardinalis* (Northern Cardinal)
* *Corvus brachyrhynchos* (American Crow)
* *Cyanocitta cristata* (Blue Jay)
* *Strix varia* (Barred Owl)

---

## Data Engineering Pipeline

To prevent the model from learning wind patterns and static, the data ingestion pipeline is strictly regulated:

1. Audio is sourced directly from the public Xeno-Canto API.
2. Raw 1D waveforms are evaluated for Root Mean Square (RMS) energy. Chunks failing to cross the amplitude threshold (e.g., silent pauses) are dropped, successfully filtering out ~40% of the dead space.
3.  The surviving, non-silent 5-second chunks are padded for uniformity and converted into 3-channel, 128-band Mel Spectrograms to natively support computer vision architectures.

### Mitigating Bias and Leakage
* Train and test splits are strictly separated by the original recording ID. This ensures that time-sliced chunks from the exact same audio file do not bleed across the train/test boundary, forcing the model to evaluate truly unseen acoustic environments.
* Because hawks produce significantly less continuous vocalization than backyard birds, the RMS filter caused some class imbalance. This was rectified through implementing inverse frequency class weights into the loss function.

---

## Model Architectures & Results

The project evaluates three separate neural network strategies to establish a baseline and further perfomance via Transfer Learning.

| Architecture | Strategy | Test Accuracy | Test Loss |
| :--- | :--- | :--- | :--- |
| **AvianNet V1** | Custom CNN (3 Conv Layers) | 81.4% | ~0.59 | 
| **AvianNet V2** | ResNet18 (Frozen Backbone) | 88.0% | ~0.37 | 
| **AvianNet V3** | ResNet18 (Fine-Tuned) | **89.0%** | **~0.31** |

### Biological Context & Model Confusion
While the model achieved 89% accuracy, the most complex challenge involved the **Blue Jay** (*Cyanocitta cristata*), which achieved the lowest class F1-score (81%). 

Analysis of the Confusion Matrix revealed the model frequently misclassified Blue Jays as Red-tailed Hawks. This is most likely due to the fact that Blue Jays actively imitate the call of the Red-tailed Hawk to scare other birds away from feeders. The model successfully identified the acoustic features of a hawk, even when produced by a jay.

---

## Repository Structure
```text
bioacoustics/
├── .gitignore
├── requirements.txt
├── config.yaml              # Centralized hyperparameters (RMS, chunk size, LR)
├── model_evaluation.ipynb        # Visual dashboard: Confusion matrices and F1 reports
├── run_pipeline.py          # Executes data ingestion and processing
├── train_model.py           # Executes the training loop and saves best weights
└── src/                     
    ├── config_loader.py     # Parses the YAML configuration
    ├── ingestion.py         # Xeno-Canto API wrapper
    ├── processing.py        # RMS silence filter and spectrogram conversion
    ├── dataset.py           # PyTorch Dataset and DataLoader classes
    └── model.py             # AvianNet architectures
```
## How to Run

**1. Set Configs**
Open `config.yaml` in the root directory. Here you can adjust data thresholds (like the RMS silence filter) and switch the `model_version` between `v1` (Custom CNN) and `v2` (ResNet18). If you are pulling data from the Xeno-Canto API, you must insert your key there.

**2. Generate the Dataset**
Run the ingestion and processing script by executing `run_pipeline.py`. This will download the raw `.mp3` files, dynamically filter out the silent chunks, and save the final `.npy` Mel Spectrograms to a `data/processed/` folder.

**3. Train the Model**
Execute the training loop by running `train_model.py`. The script will read your dataset, apply inverse frequency class weights, and train the architecture specified in your config file. The best weights will be saved.

**4. Evaluate Results**
Launch the visual dashboard by running `model_evaluation.ipynb`. The notebook the test data and the saved weights to generate the Confusion Matrix and F1-scores.
