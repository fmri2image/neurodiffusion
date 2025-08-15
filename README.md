# NeuroDiffusion: fMRI-to-Image Reconstruction

## Overview
**NeuroDiffusion** is a research framework for reconstructing visual stimuli from human brain activity using latent diffusion models.  
It builds on the **Natural Scenes Dataset (NSD)** and integrates modern generative models (such as Stable Diffusion) with advanced fMRI preprocessing, feature extraction, and decoding pipelines.

The repository is designed for modularity, allowing researchers to:
- Preprocess fMRI beta weights and apply ROI-based masking.
- Extract and prepare feature representations for training.
- Decode brain signals into latent image features.
- Reconstruct high-fidelity images using generative models.

---

## Preprocessing Pipeline

### `src/data_preprocessing/nsd_fmri_preprocessor.py`
This script handles **ROI-based preprocessing of NSD fMRI beta weights**.  
It extracts stimulus-specific data from the NSD sessions, applies atlas masks to isolate ROIs, and saves training/testing splits for further model training.

#### **Features:**
- Loads behavioral data to match stimuli with trials.
- Reads beta weights for all sessions (`betas_fithrf_GLMdenoise_RR`, `func1pt8mm` format).
- Applies atlas-based ROI masking (default: `streams` atlas).
- Computes averaged beta weights per unique stimulus.
- Splits data into training and test sets based on NSD’s `sharedix` split.
- Saves `.npy` files for:
  - Raw beta weights (train/test)
  - Averaged beta weights (train/test)
  - Stimulus ID mappings

#### **Inputs:**
- **`--subject`**: Subject ID (`subj01`, `subj02`, etc.)
- **`--output_dir`**: Directory to save extracted features.
- **`--atlas`**: Atlas name (default: `streams`).
- **`--nsd_root`**: Path to NSD dataset root.

#### **Outputs:**
The script produces `.npy` files under:
